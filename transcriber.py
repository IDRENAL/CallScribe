"""Транскрипция через faster-whisper.

На CPU аудио режется на чанки и обрабатывается в нескольких процессах
(ProcessPoolExecutor), т.к. CTranslate2 внутри одного процесса упирается
в один поток декодирования. На GPU параллелизм не нужен.
"""
from __future__ import annotations

import json
import os
import tempfile
import wave
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
OVERLAP_SEC = 2.0
MIN_CHUNK_SEC = 60  # на меньшем куске Whisper плохо ловит контекст
DEDUP_THRESHOLD_SEC = 0.5


# --------------------------------------------------------------------------- #
#  Выбор железа и модели
# --------------------------------------------------------------------------- #
def detect_compute() -> tuple[str, str]:
    """Вернуть (device, compute_type)."""
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            return "cuda", ("float16" if vram_gb >= 3 else "int8")
    except ImportError:
        pass
    return "cpu", "int8"


def choose_model(device: str) -> str:  # noqa: ARG001
    """Всегда максимальное качество — после звонка спешить некуда."""
    return "large-v3"


def get_physical_cores() -> int:
    import psutil
    cores = psutil.cpu_count(logical=False)
    return max(1, cores or 2)


# --------------------------------------------------------------------------- #
#  Форматирование
# --------------------------------------------------------------------------- #
def format_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# --------------------------------------------------------------------------- #
#  Воркер (отдельный процесс) — должен быть на уровне модуля, чтобы пиклиться
# --------------------------------------------------------------------------- #
def _transcribe_chunk(args: tuple) -> dict:
    (chunk_idx, chunk_path, offset_sec,
     model_size, language, compute_type,
     models_dir, threads_per_worker) = args

    # Ограничить внутренние потоки ДО импорта/загрузки модели
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)

    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(
            model_size, device="cpu", compute_type=compute_type,
            cpu_threads=threads_per_worker, num_workers=1, download_root=models_dir)

        segments_gen, info = model.transcribe(
            chunk_path, language=language, beam_size=5, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500}, word_timestamps=False)

        segments = [
            {"start": seg.start + offset_sec,
             "end": seg.end + offset_sec,
             "text": seg.text.strip()}
            for seg in segments_gen
        ]
        return {"chunk_idx": chunk_idx, "offset_sec": offset_sec,
                "segments": segments, "info_language": info.language,
                "info_language_probability": info.language_probability,
                "error": None}
    except Exception as e:  # noqa: BLE001
        return {"chunk_idx": chunk_idx, "offset_sec": offset_sec,
                "segments": [], "error": str(e)}


# --------------------------------------------------------------------------- #
#  Transcriber
# --------------------------------------------------------------------------- #
class Transcriber:
    def __init__(self, transcripts_dir: str | Path, models_dir: str | Path,
                 language: str = "ru", model_name: str | None = None):
        self.transcripts_dir = Path(transcripts_dir)
        self.models_dir = str(models_dir)
        self.language = language
        self.device, self.compute_type = detect_compute()
        self.model_name = model_name or choose_model(self.device)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)

    # -- публичный API ----------------------------------------------------- #
    def transcribe(self, wav_path: str | Path) -> dict:
        wav_path = Path(wav_path)
        print(f"✓ Модель: {self.model_name} | устройство: {self.device}/{self.compute_type}")

        with wave.open(str(wav_path), "rb") as wf:
            n_frames = wf.getnframes()
            sample_rate = wf.getframerate()
        duration_sec = n_frames / sample_rate

        if self.device == "cuda":
            segments, info = self._transcribe_single(wav_path)
        else:
            segments, info = self._transcribe_parallel(wav_path, n_frames, sample_rate)

        result = self._build_result(wav_path, segments, info, duration_sec)
        self._write_outputs(wav_path, result)
        return result

    # -- GPU / одиночный режим -------------------------------------------- #
    def _transcribe_single(self, wav_path: Path) -> tuple[list[dict], dict]:
        from faster_whisper import WhisperModel
        model = WhisperModel(self.model_name, device=self.device,
                             compute_type=self.compute_type, download_root=self.models_dir)
        segments_gen, info = model.transcribe(
            str(wav_path), language=self.language, beam_size=5, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500}, word_timestamps=False)
        segments = []
        for seg in segments_gen:
            segments.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
            print(f"[{format_timestamp(seg.start)}] {seg.text.strip()}")
        return segments, {"language": info.language,
                          "language_probability": info.language_probability}

    # -- CPU / мультипроцессинг ------------------------------------------- #
    def _transcribe_parallel(self, wav_path: Path, n_frames: int,
                             sample_rate: int) -> tuple[list[dict], dict]:
        phys_cores = get_physical_cores()
        n_workers = max(1, min(phys_cores // 2, 8))
        threads_per_worker = max(1, phys_cores // n_workers)

        duration_sec = n_frames / sample_rate
        n_chunks = min(n_workers, max(1, int(duration_sec // MIN_CHUNK_SEC)))
        print(f"✓ Ядер: {phys_cores} | воркеров: {n_workers} | "
              f"потоков/воркер: {threads_per_worker} | чанков: {n_chunks}")

        with tempfile.TemporaryDirectory() as tmp:
            task_args = self._slice_wav(wav_path, n_frames, sample_rate, n_chunks,
                                        threads_per_worker, Path(tmp))
            results_map: dict[int, dict] = {}
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_transcribe_chunk, a): a[0] for a in task_args}
                for future in as_completed(futures):
                    idx = futures[future]
                    res = future.result()
                    results_map[idx] = res
                    if res["error"]:
                        print(f"✗ Чанк {idx + 1}/{n_chunks}: {res['error']}")
                    else:
                        print(f"✓ Чанк {idx + 1}/{n_chunks} готов "
                              f"({len(res['segments'])} сегм.)")

        return self._merge_results(results_map)

    def _slice_wav(self, wav_path: Path, n_frames: int, sample_rate: int,
                   n_chunks: int, threads_per_worker: int, tmp: Path) -> list[tuple]:
        overlap_frames = int(OVERLAP_SEC * sample_rate)
        chunk_frames = n_frames // n_chunks

        with wave.open(str(wav_path), "rb") as wf:
            sampwidth = wf.getsampwidth()
            channels = wf.getnchannels()
            raw = wf.readframes(n_frames)
        audio = np.frombuffer(raw, dtype=np.int16)

        task_args: list[tuple] = []
        for i in range(n_chunks):
            start_frame = max(0, i * chunk_frames - overlap_frames)
            end_frame = min(n_frames, (i + 1) * chunk_frames + overlap_frames)
            offset_sec = start_frame / sample_rate

            chunk_path = tmp / f"chunk_{i}.wav"
            with wave.open(str(chunk_path), "wb") as cw:
                cw.setnchannels(channels)
                cw.setsampwidth(sampwidth)
                cw.setframerate(sample_rate)
                cw.writeframes(audio[start_frame:end_frame].tobytes())

            task_args.append((i, str(chunk_path), offset_sec, self.model_name,
                              self.language, self.compute_type, self.models_dir,
                              threads_per_worker))
        return task_args

    @staticmethod
    def _merge_results(results_map: dict[int, dict]) -> tuple[list[dict], dict]:
        all_segments: list[dict] = []
        lang, lang_prob = None, None
        for idx in sorted(results_map.keys()):
            res = results_map[idx]
            all_segments.extend(res["segments"])
            if lang is None and not res["error"]:
                lang = res.get("info_language")
                lang_prob = res.get("info_language_probability")

        all_segments.sort(key=lambda s: s["start"])
        deduped: list[dict] = []
        for seg in all_segments:
            if deduped and seg["start"] - deduped[-1]["start"] < DEDUP_THRESHOLD_SEC \
                    and seg["text"] == deduped[-1]["text"]:
                continue
            deduped.append(seg)
        return deduped, {"language": lang, "language_probability": lang_prob}

    # -- сборка и запись --------------------------------------------------- #
    def _build_result(self, wav_path: Path, segments: list[dict], info: dict,
                      duration_sec: float) -> dict:
        full_text = "\n".join(
            f"[{format_timestamp(s['start'])}] {s['text']}" for s in segments)
        return {
            "source_file": str(wav_path),
            "transcribed_at": datetime.now().isoformat(timespec="seconds"),
            "language": info.get("language") or self.language,
            "language_probability": info.get("language_probability"),
            "duration_seconds": round(duration_sec, 1),
            "model": self.model_name,
            "segments": segments,
            "full_text": full_text,
        }

    def _write_outputs(self, wav_path: Path, result: dict) -> tuple[Path, Path]:
        stem = wav_path.stem
        md_path = self.transcripts_dir / f"{stem}_transcript.md"
        json_path = self.transcripts_dir / f"{stem}_transcript.json"

        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        md_path.write_text(self._render_markdown(wav_path, result), encoding="utf-8")
        print(f"✓ Стенограмма: {md_path.name}")
        return md_path, json_path

    @staticmethod
    def _render_markdown(wav_path: Path, result: dict) -> str:
        prob = result.get("language_probability")
        prob_str = f" (уверенность: {prob * 100:.0f}%)" if prob else ""
        dt = result["transcribed_at"].replace("T", " ")[:16]
        lines = [
            "# Стенограмма звонка",
            "",
            f"**Файл:** {wav_path.name}  ",
            f"**Дата:** {dt}  ",
            f"**Длительность:** {result['duration_seconds'] / 60:.1f} мин  ",
            f"**Язык:** {result['language']}{prob_str}  ",
            f"**Модель:** whisper-{result['model']}  ",
            "",
            "---",
            "",
            "## Текст",
            "",
        ]
        for s in result["segments"]:
            lines.append(f"**[{format_timestamp(s['start'])}]** {s['text']}")
            lines.append("")
        return "\n".join(lines)
