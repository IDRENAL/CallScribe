"""Тесты аудио-утилит recorder.py (без реального PortAudio/устройств)."""
from __future__ import annotations

import numpy as np
import pytest

import recorder
from recorder import (
    get_pulse_monitor_sources,
    mix_audio,
    read_wav,
    save_stereo_wav,
    save_wav,
    split_stereo,
)


# --------------------------------------------------------------------------- #
#  mix_audio
# --------------------------------------------------------------------------- #
def test_mix_audio_basic_sum():
    mic = np.array([100, 200, 300], dtype=np.int16)
    sysv = np.array([10, 20, 30], dtype=np.int16)
    out = mix_audio(mic, sysv, mic_vol=1.0, sys_vol=1.0)
    assert out.tolist() == [110, 220, 330]
    assert out.dtype == np.int16


def test_mix_audio_truncates_to_shortest():
    mic = np.array([100, 100, 100, 100], dtype=np.int16)
    sysv = np.array([1, 1], dtype=np.int16)
    out = mix_audio(mic, sysv)
    assert len(out) == 2


def test_mix_audio_clips_to_int16_range():
    mic = np.array([30000, -30000], dtype=np.int16)
    sysv = np.array([30000, -30000], dtype=np.int16)
    out = mix_audio(mic, sysv, mic_vol=1.0, sys_vol=1.0)
    assert out[0] == 32767   # переполнение вверх обрезано
    assert out[1] == -32768  # переполнение вниз обрезано


def test_mix_audio_empty_inputs():
    empty = np.zeros(0, dtype=np.int16)
    assert mix_audio(empty, empty).size == 0
    only_mic = mix_audio(np.array([5], dtype=np.int16), empty)
    assert only_mic.tolist() == [5]


# --------------------------------------------------------------------------- #
#  WAV round-trip (моно и стерео)
# --------------------------------------------------------------------------- #
def test_save_read_mono_roundtrip(tmp_path):
    audio = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16)
    p = tmp_path / "mono.wav"
    save_wav(p, audio, sample_rate=16000)
    back, sr, ch = read_wav(p)
    assert sr == 16000
    assert ch == 1
    assert back.tolist() == audio.tolist()


def test_save_read_stereo_keeps_channels_separate(tmp_path):
    mic = np.array([1, 2, 3], dtype=np.int16)
    sysv = np.array([10, 20, 30], dtype=np.int16)
    p = tmp_path / "stereo.wav"
    save_stereo_wav(p, mic, sysv)
    back, sr, ch = read_wav(p)
    assert ch == 2
    assert back[:, 0].tolist() == mic.tolist()      # L = микрофон
    assert back[:, 1].tolist() == sysv.tolist()     # R = системный звук


def test_split_stereo_returns_left_right(tmp_path):
    mic = np.array([7, 8, 9], dtype=np.int16)
    sysv = np.array([70, 80, 90], dtype=np.int16)
    p = tmp_path / "s.wav"
    save_stereo_wav(p, mic, sysv)
    left, right, sr = split_stereo(p)
    assert left.tolist() == mic.tolist()
    assert right.tolist() == sysv.tolist()
    assert sr == 16000


def test_split_stereo_mono_yields_silence_for_second_channel(tmp_path):
    audio = np.array([5, 6, 7], dtype=np.int16)
    p = tmp_path / "m.wav"
    save_wav(p, audio)
    left, right, _ = split_stereo(p)
    assert left.tolist() == audio.tolist()
    assert right.tolist() == [0, 0, 0]   # второго канала нет → тишина


def test_stereo_roundtrip_truncates_to_shortest(tmp_path):
    mic = np.array([1, 2, 3, 4], dtype=np.int16)
    sysv = np.array([1, 2], dtype=np.int16)
    p = tmp_path / "trunc.wav"
    save_stereo_wav(p, mic, sysv)
    back, _, ch = read_wav(p)
    assert ch == 2
    assert back.shape[0] == 2   # длина по короткому каналу


# --------------------------------------------------------------------------- #
#  Разбор monitor-источников (pactl мокается)
# --------------------------------------------------------------------------- #
def test_get_pulse_monitor_sources_puts_default_sink_first(monkeypatch):
    short = (
        "0\talsa_output.usb.analog-stereo.monitor\tmodule\ts16le\tRUNNING\n"
        "1\talsa_input.pci.analog-stereo\tmodule\ts16le\tIDLE\n"
        "2\talsa_output.pci.hdmi-stereo.monitor\tmodule\ts16le\tIDLE\n"
    )

    def fake_pactl(*args):
        if args[:2] == ("list", "sources"):
            return short
        if args[0] == "get-default-sink":
            return "alsa_output.pci.hdmi-stereo\n"
        return ""

    monkeypatch.setattr(recorder, "_pactl", fake_pactl)
    monitors = get_pulse_monitor_sources()
    # только .monitor источники, и монитор default sink — первым
    assert monitors == [
        "alsa_output.pci.hdmi-stereo.monitor",
        "alsa_output.usb.analog-stereo.monitor",
    ]


def test_get_pulse_monitor_sources_empty_when_no_pactl(monkeypatch):
    monkeypatch.setattr(recorder, "_pactl", lambda *a: "")
    assert get_pulse_monitor_sources() == []
