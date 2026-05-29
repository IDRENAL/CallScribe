"""Захват аудио: микрофон + системный звук (loopback) одновременно.

Библиотека: sounddevice (обёртка над PortAudio, кросс-платформенная).
Два независимых InputStream пишут в свои буферы через callback,
после остановки потоки микшируются в один моно-WAV 16 kHz.
"""
from __future__ import annotations

import sys
import threading
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000  # Whisper ожидает 16 kHz
CHANNELS = 1         # моно
DTYPE = "int16"


def _sd():
    """Ленивый импорт sounddevice — нужен только для записи, не для STT.

    Так транскрипция и веб-интерфейс работают и без системного PortAudio.
    """
    import sounddevice as sd
    return sd


def query_devices() -> list[dict]:
    """Список всех аудио-устройств системы."""
    return list(_sd().query_devices())


def get_input_devices() -> list[tuple[int, str]]:
    """Устройства, способные принимать звук (max_input_channels > 0)."""
    result = []
    for i, dev in enumerate(query_devices()):
        if dev["max_input_channels"] > 0:
            result.append((i, dev["name"]))
    return result


def get_loopback_candidates() -> list[tuple[int, str]]:
    """Кандидаты на роль loopback (системный звук) — эвристика по имени."""
    result = []
    for i, dev in enumerate(query_devices()):
        if dev["max_input_channels"] < 1:
            continue
        name = dev["name"].lower()
        if sys.platform == "win32":
            if any(x in name for x in
                   ["loopback", "stereo mix", "что слышит", "wave out", "cable output"]):
                result.append((i, dev["name"]))
        elif sys.platform.startswith("linux"):
            if "monitor" in name:
                result.append((i, dev["name"]))
        elif sys.platform == "darwin":
            if any(x in name for x in ["blackhole", "loopback", "soundflower"]):
                result.append((i, dev["name"]))
    return result


def mix_audio(mic: np.ndarray, sys_audio: np.ndarray,
              mic_vol: float = 1.0, sys_vol: float = 0.8) -> np.ndarray:
    """Смешать два int16 моно-потока в один с клиппингом."""
    if mic.size == 0 and sys_audio.size == 0:
        return np.zeros(0, dtype=np.int16)
    if mic.size == 0:
        return np.clip(sys_audio.astype(np.float32) * sys_vol, -32768, 32767).astype(np.int16)
    if sys_audio.size == 0:
        return np.clip(mic.astype(np.float32) * mic_vol, -32768, 32767).astype(np.int16)

    n = min(len(mic), len(sys_audio))
    mic, sys_audio = mic[:n], sys_audio[:n]
    mixed = mic.astype(np.float32) * mic_vol + sys_audio.astype(np.float32) * sys_vol
    return np.clip(mixed, -32768, 32767).astype(np.int16)


def save_wav(path: str | Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Сохранить моно int16 в WAV через stdlib wave."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())


class CallRecorder:
    """Захват микрофона и системного звука в параллельные буферы."""

    def __init__(self, mic_device: int | None, loopback_device: int | None,
                 output_dir: str | Path):
        self.mic_device = mic_device
        self.loopback_device = loopback_device
        self.output_dir = Path(output_dir)
        self.mic_frames: list[np.ndarray] = []
        self.sys_frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stop_event: threading.Event = threading.Event()

    # --- callbacks (вызываются из C-потока PortAudio) ---
    def _mic_callback(self, indata, frames, time, status):  # noqa: ANN001
        if status:
            print(f"⚠ Микрофон: {status}")
        with self._lock:
            self.mic_frames.append(indata.copy())  # copy обязателен

    def _sys_callback(self, indata, frames, time, status):  # noqa: ANN001
        if status:
            print(f"⚠ Системный звук: {status}")
        with self._lock:
            self.sys_frames.append(indata.copy())

    def record(self, stop_event: threading.Event | None = None) -> Path:
        """Запустить запись. Блокирует до остановки, возвращает путь к WAV.

        CLI-режим: stop_event=None → ждём Enter.
        UI-режим: передаётся внешний Event, выставляемый кнопкой "Стоп".
        """
        self._stop_event = stop_event or threading.Event()
        self.mic_frames.clear()
        self.sys_frames.clear()
        sd = _sd()

        streams = []
        try:
            if self.mic_device is not None:
                streams.append(sd.InputStream(
                    device=self.mic_device, channels=CHANNELS, samplerate=SAMPLE_RATE,
                    dtype=DTYPE, callback=self._mic_callback))
            if self.loopback_device is not None:
                streams.append(sd.InputStream(
                    device=self.loopback_device, channels=CHANNELS, samplerate=SAMPLE_RATE,
                    dtype=DTYPE, callback=self._sys_callback))

            if not streams:
                raise RuntimeError("Не задано ни одного устройства записи (mic/loopback)")

            for s in streams:
                s.start()
            print("✓ Запись началась")

            if stop_event is None:
                input()  # CLI: блокируемся до Enter
            else:
                while not self._stop_event.is_set():
                    self._stop_event.wait(0.2)
        finally:
            for s in streams:
                s.stop()
                s.close()

        return self._save()

    def _save(self) -> Path:
        with self._lock:
            mic = (np.concatenate(self.mic_frames).flatten()
                   if self.mic_frames else np.zeros(0, dtype=np.int16))
            sys_audio = (np.concatenate(self.sys_frames).flatten()
                         if self.sys_frames else np.zeros(0, dtype=np.int16))

        mixed = mix_audio(mic, sys_audio)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = "call_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = self.output_dir / f"{stem}.wav"
        save_wav(path, mixed)
        dur = len(mixed) / SAMPLE_RATE
        print(f"✓ Сохранено: {path.name} ({dur:.1f} сек)")
        return path

    def stop(self) -> None:
        """Остановить запись (из UI)."""
        self._stop_event.set()
