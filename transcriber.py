"""Транскрипция через faster-whisper.

Два режима:
- "accurate" (по умолчанию): каналы (микрофон / системный звук) расшифровываются
  РАЗДЕЛЬНО за один проход без нарезки на чанки. Не теряются реплики при наложении
  речи, сохраняется контекст, мягкий VAD не режет тихую речь.
- "fast": запись микшируется в моно и режется на чанки в нескольких процессах —
  быстрее, но возможны потери на стыках чанков и при перебивках.
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

from recorder import read_wav, save_wav

SAMPLE_RATE = 16000
OVERLAP_SEC = 2.0
MIN_CHUNK_SEC = 60  # на меньшем куске Whisper плохо ловит контекст
DEDUP_THRESHOLD_SEC = 0.5

# Параметры режима максимальной точности: широкий луч, temperature-fallback,
# контекст между сегментами, отсев галлюцинаций, мягкий VAD (не режет тихую речь).
ACCURATE_OPTS = dict(
    beam_size=10,
    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    condition_on_previous_text=True,
    compression_ratio_threshold=2.4,
    log_prob_threshold=-1.0,
    no_speech_threshold=0.6,
    word_timestamps=False,
)
ACCURATE_VAD_PARAMS = {"min_silence_duration_ms": 1500, "speech_pad_ms": 400}


# --------------------------------------------------------------------------- #
#  Выбор железа и модели
# --------------------------------------------------------------------------- #
def ensure_cuda_libs() -> None:
    """Подгрузить CUDA-библиотеки (cuBLAS/cuDNN) из pip-пакетов nvidia-*.

    CTranslate2 ищет libcublas.so.12 / libcudnn*.so.9 через dlopen. Предзагрузка
    их по полному пути с RTLD_GLOBAL делает символы доступными — тогда GPU работает
    без ручного LD_LIBRARY_PATH (так же поступает PyTorch со своими либами).
    """
    import ctypes
    import glob
    import site
    bases = list(site.getsitepackages())
    if hasattr(site, "getusersitepackages"):
        bases.append(site.getusersitepackages())
    sos: list[str] = []
    for base in bases:
        sos += glob.glob(os.path.join(base, "nvidia", "*", "lib", "*.so*"))
    pending = sos
    for _ in range(4):  # несколько проходов из-за взаимозависимостей библиотек
        still = []
        for so in pending:
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                still.append(so)
        if not still or len(still) == len(pending):
            break
        pending = still


def detect_compute() -> tuple[str, str]:
    """Вернуть (device, compute_type).

    Наличие CUDA определяем через CTranslate2 (torch не нужен), а compute_type
    выбираем из реально поддерживаемых карте типов. Важно для Pascal (GTX 1080):
    float16 там не эффективен — берётся int8_float32 (точность ~float32, но быстро).
    """
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            ensure_cuda_libs()
            supported = ctranslate2.get_supported_compute_types("cuda")
            for ct in ("float16", "int8_float16", "int8_float32", "int8", "float32"):
                if ct in supported:
                    return "cuda", ct
    except Exception:  # noqa: BLE001
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
#  Нарезка по паузам (для параллелизма accurate без разрезания слов)
# --------------------------------------------------------------------------- #
def find_silence_cuts(audio: np.ndarray, sr: int, target_sec: float = 300.0,
                      min_silence_sec: float = 0.7,
                      frame_ms: int = 30) -> list[tuple[int, int]]:
    """Границы чанков, проходящие ТОЛЬКО по тишине — слова не режутся.

    Чанк растёт минимум до target_sec, затем закрывается на ближайшей паузе
    длиннее min_silence_sec. Если подходящих пауз нет — остаётся один кусок
    (корректность важнее параллелизма). Возвращает список (start, end) в сэмплах.
    """
    n = len(audio)
    frame = max(1, int(sr * frame_ms / 1000))
    nf = n // frame
    if nf < 2:
        return [(0, n)]

    fr = audio[:nf * frame].reshape(nf, frame).astype(np.float32)
    rms = np.sqrt((fr ** 2).mean(axis=1) + 1.0)
    speech_level = float(np.percentile(rms, 90))
    thresh = max(150.0, speech_level * 0.06)  # консервативный порог тишины
    silent = rms < thresh

    min_sil_frames = max(1, int(min_silence_sec * 1000 / frame_ms))
    cut_points: list[int] = []
    i = 0
    while i < nf:
        if silent[i]:
            j = i
            while j < nf and silent[j]:
                j += 1
            if j - i >= min_sil_frames:
                cut_points.append((i + j) // 2 * frame)  # середина паузы
            i = j
        else:
            i += 1

    target = target_sec * sr
    ranges: list[tuple[int, int]] = []
    start = 0
    for c in cut_points:
        if c - start >= target:
            ranges.append((start, c))
            start = c
    ranges.append((start, n))
    return ranges


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


def _transcribe_channel(args: tuple) -> dict:
    """Воркер режима точности: один проход по куску канала (нарезка по паузам).

    Запускается в отдельном процессе. offset_sec прибавляется к таймстампам,
    сегментам проставляется speaker. Куски режутся по тишине, слова не теряются.
    """
    (channel, speaker, wav_path, offset_sec, chunk_dur, model_size, language, device,
     compute_type, models_dir, cpu_threads, vad) = args

    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)

    if device == "cuda":
        ensure_cuda_libs()

    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_size, device=device, compute_type=compute_type,
                             cpu_threads=cpu_threads, num_workers=1,
                             download_root=models_dir)
        segments_gen, info = model.transcribe(
            wav_path, language=language,
            vad_filter=vad, vad_parameters=ACCURATE_VAD_PARAMS, **ACCURATE_OPTS)

        # Стриминг прогресса для длинных кусков (один проход на GPU был «слепым»)
        segments = []
        last_pct = -10
        for seg in segments_gen:
            segments.append(
                {"start": seg.start + offset_sec, "end": seg.end + offset_sec,
                 "text": seg.text.strip(), "speaker": speaker, "channel": channel})
            if chunk_dur > 120:
                pct = min(99, int(seg.end / chunk_dur * 100))
                if pct >= last_pct + 5:
                    last_pct = pct
                    print(f"  {channel}: {pct}% ({len(segments)} сегм.)", flush=True)

        return {"channel": channel, "segments": segments,
                "info_language": info.language,
                "info_language_probability": info.language_probability,
                "error": None}
    except Exception as e:  # noqa: BLE001
        return {"channel": channel, "segments": [], "error": str(e)}


# --------------------------------------------------------------------------- #
#  Transcriber
# --------------------------------------------------------------------------- #
class Transcriber:
    def __init__(self, transcripts_dir: str | Path, models_dir: str | Path,
                 language: str = "ru", model_name: str | None = None,
                 mode: str = "accurate", compute_type: str | None = None,
                 vad: bool = True, speaker_labels: dict | None = None):
        self.transcripts_dir = Path(transcripts_dir)
        self.models_dir = str(models_dir)
        self.language = language
        self.mode = mode
        self.vad = vad
        self.speaker_labels = speaker_labels or {"mic": "Я", "loopback": "Собеседник"}
        self.device, auto_compute = detect_compute()
        self.compute_type = compute_type or auto_compute  # явный из конфига имеет приоритет
        self.model_name = model_name or choose_model(self.device)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)

    # -- публичный API ----------------------------------------------------- #
    def transcribe(self, src_path: str | Path) -> dict:
        src_path = Path(src_path)
        print(f"✓ Режим: {self.mode} | модель: {self.model_name} | "
              f"устройство: {self.device}/{self.compute_type}")

        channels_list, sr = self._load_channels(src_path)
        duration_sec = max((len(a) for _, _, a in channels_list), default=0) / sr

        if self.mode == "accurate":
            segments, info = self._transcribe_accurate(channels_list, sr)
        else:
            segments, info = self._transcribe_fast(channels_list, sr)

        result = self._build_result(src_path, segments, info, duration_sec)
        self._write_outputs(src_path, result)
        return result

    def _load_channels(self, src_path: Path) -> tuple[list[tuple], int]:
        """Загрузить источник в список каналов (channel_id, speaker, int16 audio).

        WAV: стерео → два канала (mic=L, loopback=R), моно → один канал.
        Прочие форматы (mp4/mkv/mp3/...) декодируются PyAV в один моно-канал
        (микшированная дорожка — разделение спикеров недоступно).
        """
        if src_path.suffix.lower() == ".wav":
            audio, sr, channels = read_wav(src_path)
            if channels >= 2:
                return [("mic", self.speaker_labels.get("mic", "Я"), audio[:, 0].copy()),
                        ("loopback", self.speaker_labels.get("loopback", "Собеседник"),
                         audio[:, 1].copy())], sr
            return [("mic", None, audio)], sr

        # Видео/прочее аудио → PyAV декодирует в float32 моно 16 kHz
        from faster_whisper.audio import decode_audio
        print(f"✓ Декодирую {src_path.suffix} через PyAV…")
        samples = decode_audio(str(src_path), sampling_rate=SAMPLE_RATE)
        audio = np.clip(np.asarray(samples) * 32767.0, -32768, 32767).astype(np.int16)
        return [("audio", None, audio)], SAMPLE_RATE

    # -- режим максимальной точности: раздельные каналы, нарезка по паузам -- #
    def _transcribe_accurate(self, channels_list: list[tuple],
                             sr: int) -> tuple[list[dict], dict]:
        # Каждый канал режем по паузам (на CPU) → больше параллельных задач
        # без разрезания слов. На GPU нарезка не нужна.
        specs: list[tuple] = []  # (channel, speaker, audio_slice, offset_sec)
        for channel, speaker, audio in channels_list:
            ranges = ([(0, len(audio))] if self.device == "cuda"
                      else find_silence_cuts(audio, sr))
            for s, e in ranges:
                specs.append((channel, speaker, audio[s:e], s / sr))

        phys = get_physical_cores()
        n_workers = max(1, min(len(specs), max(1, phys // 2)))
        cpu_threads = max(1, phys // n_workers)
        print(f"✓ Каналов: {len(channels_list)} | чанков по паузам: {len(specs)} | "
              f"воркеров: {n_workers} | потоков/воркер: {cpu_threads} | "
              f"VAD: {'вкл' if self.vad else 'выкл'}")

        with tempfile.TemporaryDirectory() as tmp:
            args_list = []
            for k, (channel, speaker, audio, offset) in enumerate(specs):
                cpath = Path(tmp) / f"{channel}_{k}.wav"
                save_wav(cpath, audio, sr)
                chunk_dur = len(audio) / sr
                args_list.append((channel, speaker, str(cpath), offset, chunk_dur,
                                  self.model_name, self.language, self.device,
                                  self.compute_type, self.models_dir, cpu_threads,
                                  self.vad))

            results = []
            done = 0
            if self.device == "cuda":
                for a in args_list:
                    results.append(_transcribe_channel(a))
                    done += 1
                    self._log_chunk(results[-1], done, len(specs))
            else:
                with ProcessPoolExecutor(max_workers=n_workers) as ex:
                    futures = [ex.submit(_transcribe_channel, a) for a in args_list]
                    for fut in as_completed(futures):
                        results.append(fut.result())
                        done += 1
                        self._log_chunk(results[-1], done, len(specs))

        return self._merge_channels(results)

    @staticmethod
    def _log_chunk(res: dict, done: int, total: int) -> None:
        if res["error"]:
            print(f"✗ Чанк {done}/{total} ({res['channel']}): {res['error']}")
        else:
            print(f"✓ Чанк {done}/{total} ({res['channel']}) — "
                  f"{len(res['segments'])} сегм.")

    # -- быстрый режим: микс в моно + нарезка на чанки в процессах --------- #
    def _transcribe_fast(self, channels_list: list[tuple],
                         sr: int) -> tuple[list[dict], dict]:
        from recorder import mix_audio
        if len(channels_list) >= 2:
            mono = mix_audio(channels_list[0][2], channels_list[1][2])
        else:
            mono = channels_list[0][2]

        with tempfile.TemporaryDirectory() as tmp:
            mono_path = Path(tmp) / "mono.wav"
            save_wav(mono_path, mono, sr)
            n_frames = len(mono)
            if self.device == "cuda":
                return self._transcribe_single(mono_path)
            return self._transcribe_parallel(mono_path, n_frames, sr)

    @staticmethod
    def _merge_channels(results: list[dict]) -> tuple[list[dict], dict]:
        segments: list[dict] = []
        lang, lang_prob = None, None
        for res in results:
            segments.extend(res["segments"])
            if lang is None and not res["error"]:
                lang = res.get("info_language")
                lang_prob = res.get("info_language_probability")
        segments.sort(key=lambda s: s["start"])  # склейка каналов по времени
        return segments, {"language": lang, "language_probability": lang_prob}

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
        def line(s: dict) -> str:
            spk = s.get("speaker")
            prefix = f"{spk}: " if spk else ""
            return f"[{format_timestamp(s['start'])}] {prefix}{s['text']}"

        full_text = "\n".join(line(s) for s in segments)
        return {
            "source_file": str(wav_path),
            "transcribed_at": datetime.now().isoformat(timespec="seconds"),
            "language": info.get("language") or self.language,
            "language_probability": info.get("language_probability"),
            "duration_seconds": round(duration_sec, 1),
            "model": self.model_name,
            "mode": self.mode,
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
            spk = s.get("speaker")
            label = f" {spk}:" if spk else ""
            lines.append(f"**[{format_timestamp(s['start'])}]{label}** {s['text']}")
            lines.append("")
        return "\n".join(lines)
