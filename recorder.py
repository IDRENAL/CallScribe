"""Захват аудио: микрофон + системный звук (loopback) одновременно.

Библиотека: sounddevice (обёртка над PortAudio, кросс-платформенная).
Два независимых InputStream пишут в свои буферы через callback,
после остановки потоки микшируются в один моно-WAV 16 kHz.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000  # Whisper ожидает 16 kHz
CHANNELS = 1         # моно
DTYPE = "int16"
PULSE_DEVICE = "pulse"  # имя PortAudio-устройства для PulseAudio/PipeWire


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


def _pactl(*args: str) -> str:
    """Вызвать pactl, вернуть stdout (или '' если pactl недоступен)."""
    try:
        out = subprocess.run(["pactl", *args], capture_output=True,
                             text=True, timeout=5)
        return out.stdout if out.returncode == 0 else ""
    except (FileNotFoundError, subprocess.SubprocessError):
        return ""


def get_pulse_monitor_sources() -> list[str]:
    """Имена PulseAudio/PipeWire monitor-источников (системный звук).

    Монитор текущего default sink идёт первым — это «то, что в колонках».
    """
    short = _pactl("list", "sources", "short")
    if not short:
        return []
    monitors = []
    for line in short.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and ".monitor" in parts[1]:
            monitors.append(parts[1])
    default_sink = _pactl("get-default-sink").strip()
    preferred = f"{default_sink}.monitor"
    monitors.sort(key=lambda m: m != preferred)  # preferred первым
    return monitors


def get_loopback_candidates() -> list[tuple[int | str, str]]:
    """Кандидаты на роль loopback (системный звук).

    Идентификатор — либо int (индекс sounddevice, Windows VB-Cable/Stereo Mix),
    либо str (имя PulseAudio-источника, Linux PipeWire/PulseAudio).
    """
    result: list[tuple[int | str, str]] = []
    if sys.platform.startswith("linux"):
        # На PipeWire/PulseAudio monitor видны через pactl, а не через PortAudio
        for name in get_pulse_monitor_sources():
            result.append((name, name))
        if result:
            return result

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
            # PortAudio считывает PULSE_SOURCE в момент start(), поэтому каждый
            # поток стартуем сразу после открытия — пока окружение корректно.
            # Микрофон: при чистой PULSE_SOURCE берётся default source.
            if self.mic_device is not None:
                os.environ.pop("PULSE_SOURCE", None)
                s = self._open_stream(sd, self.mic_device, self._mic_callback,
                                      is_loopback=False)
                s.start()
                streams.append(s)

            # Loopback: строка → имя PulseAudio-источника (monitor) через 'pulse'
            # + PULSE_SOURCE; int → обычный индекс sounddevice (Windows VB-Cable).
            if self.loopback_device is not None:
                s = self._open_stream(sd, self.loopback_device, self._sys_callback,
                                      is_loopback=True)
                s.start()
                streams.append(s)

            if not streams:
                raise RuntimeError("Не задано ни одного устройства записи (mic/loopback)")

            print("✓ Запись началась")

            if stop_event is None:
                input()  # CLI: блокируемся до Enter
            else:
                while not self._stop_event.is_set():
                    self._stop_event.wait(0.2)
        finally:
            os.environ.pop("PULSE_SOURCE", None)
            for s in streams:
                s.stop()
                s.close()

        return self._save()

    def _open_stream(self, sd, device, callback, is_loopback: bool):  # noqa: ANN001
        """Создать InputStream.

        Для loopback строковый device на Linux — это имя PulseAudio-источника
        (monitor): открываем через 'pulse' + PULSE_SOURCE. Для микрофона device
        (int-индекс или имя устройства вроде 'pulse'/'default') передаётся как есть.
        """
        if is_loopback and isinstance(device, str) and sys.platform.startswith("linux"):
            os.environ["PULSE_SOURCE"] = device
            device = PULSE_DEVICE
        return sd.InputStream(device=device, channels=CHANNELS, samplerate=SAMPLE_RATE,
                              dtype=DTYPE, callback=callback)

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
