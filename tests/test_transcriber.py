"""Тесты transcriber.py — чистые функции (без загрузки модели Whisper)."""
from __future__ import annotations

import numpy as np
import pytest

from recorder import save_stereo_wav, save_wav
from transcriber import (
    PER_WORKER_GB,
    Transcriber,
    TranscriptionCancelled,
    choose_model,
    detect_compute,
    find_silence_cuts,
    format_timestamp,
    get_physical_cores,
    plan_cpu_workers,
)

SR = 16000


def _speech(seconds: float, amp: int = 6000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-amp, amp, size=int(seconds * SR), dtype=np.int16)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.int16)


# --------------------------------------------------------------------------- #
#  format_timestamp
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sec,expected", [
    (0, "00:00"),
    (5, "00:05"),
    (65, "01:05"),
    (3600, "01:00:00"),
    (3661, "01:01:01"),
])
def test_format_timestamp(sec, expected):
    assert format_timestamp(sec) == expected


# --------------------------------------------------------------------------- #
#  find_silence_cuts — главный инвариант: режем ТОЛЬКО по тишине
# --------------------------------------------------------------------------- #
def test_silence_cuts_cover_full_signal_contiguously():
    audio = np.concatenate([_speech(4, seed=1), _silence(1.0), _speech(4, seed=2)])
    ranges = find_silence_cuts(audio, SR, target_sec=3.0)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == len(audio)
    # стыки непрерывны, без пропусков и нахлёстов
    for (a_s, a_e), (b_s, b_e) in zip(ranges, ranges[1:]):
        assert a_e == b_s


def test_silence_cuts_split_happens_inside_silence():
    audio = np.concatenate([_speech(4, seed=1), _silence(1.0), _speech(4, seed=2)])
    ranges = find_silence_cuts(audio, SR, target_sec=3.0)
    assert len(ranges) >= 2, "должна быть хотя бы одна точка реза по паузе"
    # каждая внутренняя граница должна попадать в тихий участок (слово не режется)
    speech_level = float(np.percentile(np.abs(audio), 90))
    for _, end in ranges[:-1]:
        window = audio[max(0, end - 200):end + 200]
        assert np.abs(window).max() < speech_level * 0.5


def test_silence_cuts_single_chunk_without_pauses():
    audio = _speech(10, seed=3)  # сплошная речь, пауз нет
    ranges = find_silence_cuts(audio, SR, target_sec=3.0)
    assert ranges == [(0, len(audio))]


def test_silence_cuts_tiny_input():
    assert find_silence_cuts(_speech(0.001), SR) == [(0, len(_speech(0.001)))]


# --------------------------------------------------------------------------- #
#  plan_cpu_workers — защита от OOM
# --------------------------------------------------------------------------- #
def test_plan_cpu_workers_invariants():
    n, threads = plan_cpu_workers(12)
    phys = get_physical_cores()
    assert 1 <= n <= max(1, phys // 2)
    assert n <= 12
    assert threads >= 1


def test_plan_cpu_workers_respects_task_count():
    n, _ = plan_cpu_workers(1)
    assert n == 1


def test_plan_cpu_workers_override_caps_workers():
    n, threads = plan_cpu_workers(12, override=2)
    assert n == 2
    assert threads == max(1, get_physical_cores() // 2)


def test_plan_cpu_workers_override_zero_falls_back_to_auto():
    # override=0 (falsy) не должен обнулять число воркеров
    n, _ = plan_cpu_workers(8, override=0)
    assert n >= 1


def test_per_worker_estimate_is_reasonable():
    assert 2.0 <= PER_WORKER_GB <= 6.0  # оценка ОЗУ на large-v3 int8


# --------------------------------------------------------------------------- #
#  detect_compute / choose_model
# --------------------------------------------------------------------------- #
def test_detect_compute_cpu_is_deterministic():
    assert detect_compute("cpu") == ("cpu", "int8")


def test_choose_model_always_large():
    assert choose_model("cpu") == "large-v3"
    assert choose_model("cuda") == "large-v3"


# --------------------------------------------------------------------------- #
#  Склейка каналов и дедупликация
# --------------------------------------------------------------------------- #
def test_merge_channels_sorts_by_time_and_keeps_language():
    results = [
        {"channel": "mic", "error": None, "info_language": "ru",
         "info_language_probability": 0.97,
         "segments": [{"start": 5.0, "text": "второй"}]},
        {"channel": "loopback", "error": None, "info_language": "ru",
         "info_language_probability": 0.9,
         "segments": [{"start": 1.0, "text": "первый"}]},
    ]
    segs, info = Transcriber._merge_channels(results)
    assert [s["text"] for s in segs] == ["первый", "второй"]  # по времени
    assert info["language"] == "ru"


def test_merge_results_dedupes_overlap_duplicates():
    results_map = {
        0: {"error": None, "info_language": "ru", "info_language_probability": 0.95,
            "segments": [{"start": 0.0, "text": "привет"},
                         {"start": 1.0, "text": "как дела"}]},
        1: {"error": None, "info_language": "ru", "info_language_probability": 0.95,
            "segments": [{"start": 1.2, "text": "как дела"},   # дубль из перекрытия
                         {"start": 3.0, "text": "пока"}]},
    }
    segs, _ = Transcriber._merge_results(results_map)
    texts = [s["text"] for s in segs]
    assert texts == ["привет", "как дела", "пока"]


def test_merge_results_keeps_distinct_text_at_same_time():
    results_map = {
        0: {"error": None, "info_language": "ru", "info_language_probability": 0.9,
            "segments": [{"start": 1.0, "text": "раз"}]},
        1: {"error": None, "info_language": "ru", "info_language_probability": 0.9,
            "segments": [{"start": 1.1, "text": "два"}]},  # близко по времени, но другой текст
    }
    segs, _ = Transcriber._merge_results(results_map)
    assert [s["text"] for s in segs] == ["раз", "два"]


# --------------------------------------------------------------------------- #
#  Сборка результата и рендер Markdown (с метками спикеров)
# --------------------------------------------------------------------------- #
def _cpu_transcriber(tmp_path):
    return Transcriber(tmp_path / "tr", tmp_path / "models", device="cpu")


def test_build_result_full_text_has_speaker_prefix(tmp_path):
    tr = _cpu_transcriber(tmp_path)
    segments = [
        {"start": 0.0, "end": 2.0, "text": "Привет", "speaker": "Я"},
        {"start": 2.0, "end": 4.0, "text": "Здравствуйте", "speaker": "Собеседник"},
    ]
    res = tr._build_result(tmp_path / "call.wav", segments, {"language": "ru"}, 4.0)
    assert "Я: Привет" in res["full_text"]
    assert "Собеседник: Здравствуйте" in res["full_text"]
    assert res["mode"] == "accurate"
    assert res["model"] == "large-v3"


def test_render_markdown_contains_header_and_speakers(tmp_path):
    tr = _cpu_transcriber(tmp_path)
    result = {
        "transcribed_at": "2026-05-30T09:20:00",
        "language": "ru", "language_probability": 0.98,
        "duration_seconds": 120.0, "model": "large-v3", "mode": "accurate",
        "segments": [{"start": 0.0, "end": 2.0, "text": "Привет", "speaker": "Я"}],
    }
    md = tr._render_markdown(tmp_path / "call.wav", result)
    assert "# Стенограмма звонка" in md
    assert "whisper-large-v3" in md
    assert "Я:" in md and "Привет" in md


# --------------------------------------------------------------------------- #
#  Загрузка каналов из WAV (стерео → 2, моно → 1)
# --------------------------------------------------------------------------- #
def test_load_channels_stereo_splits_into_two_speakers(tmp_path):
    tr = _cpu_transcriber(tmp_path)
    p = tmp_path / "stereo.wav"
    save_stereo_wav(p, _speech(0.5, seed=1), _speech(0.5, seed=2))
    channels, sr = tr._load_channels(p)
    assert sr == SR
    assert len(channels) == 2
    ids = {c[0] for c in channels}
    assert ids == {"mic", "loopback"}
    labels = {c[1] for c in channels}
    assert "Я" in labels and "Собеседник" in labels


def test_load_channels_mono_single_channel(tmp_path):
    tr = _cpu_transcriber(tmp_path)
    p = tmp_path / "mono.wav"
    save_wav(p, _speech(0.5, seed=4))
    channels, sr = tr._load_channels(p)
    assert len(channels) == 1
    assert channels[0][1] is None   # нет разделения спикеров для моно


# --------------------------------------------------------------------------- #
#  Отмена транскрипции
# --------------------------------------------------------------------------- #
def test_check_cancel_raises_when_requested(tmp_path):
    tr = _cpu_transcriber(tmp_path)
    tr._cancel = lambda: True
    with pytest.raises(TranscriptionCancelled):
        tr._check_cancel()


def test_check_cancel_silent_when_not_requested(tmp_path):
    tr = _cpu_transcriber(tmp_path)
    tr._cancel = lambda: False
    tr._check_cancel()   # не должно бросать


def test_transcribe_channel_propagates_cancel_from_cb(monkeypatch):
    """Внутри прохода cb бросает отмену → воркер её ПРОБРАСЫВАЕТ (не глотает)."""
    import sys
    import types

    from transcriber import _transcribe_channel

    fake = types.ModuleType("faster_whisper")

    class _Seg:
        def __init__(self, i):
            self.start, self.end, self.text = float(i), float(i + 1), f"t{i}"

    class _Info:
        language, language_probability = "ru", 0.9

    class _Model:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, *a, **k):
            return iter([_Seg(i) for i in range(10)]), _Info()

    fake.WhisperModel = _Model
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)

    args = ("mic", "Я", "x.wav", 0.0, 60.0, "large-v3", "ru", "cpu",
            "int8", "models", 1, False)
    n = {"v": 0}

    def cb(frac):
        n["v"] += 1
        if n["v"] >= 3:           # отмена на третьем сегменте
            raise TranscriptionCancelled()

    with pytest.raises(TranscriptionCancelled):
        _transcribe_channel(args, progress_cb=cb)


def test_transcribe_stores_cancel_callable(tmp_path, monkeypatch):
    tr = _cpu_transcriber(tmp_path)
    flag = {"v": True}
    # подменяем тяжёлый разбор источника, чтобы дойти до проверки отмены без модели
    monkeypatch.setattr(tr, "_load_channels", lambda p: ([("audio", None, [])], SR))

    def fake_accurate(channels, sr):
        tr._check_cancel()   # имитируем точку проверки внутри пайплайна
        return [], {}
    monkeypatch.setattr(tr, "_transcribe_accurate", fake_accurate)

    with pytest.raises(TranscriptionCancelled):
        tr.transcribe(tmp_path / "x.wav", cancel=lambda: flag["v"])


# --------------------------------------------------------------------------- #
#  Диаризация (интеграция в transcriber)
# --------------------------------------------------------------------------- #
def _seg(s, e, t="x"):
    return {"start": float(s), "end": float(e), "text": t}


def test_maybe_diarize_noop_when_disabled(tmp_path):
    tr = _cpu_transcriber(tmp_path)
    tr.diarize = {"enabled": False}
    segs = [_seg(0, 1)]
    chans = [("audio", None, np.zeros(10, dtype=np.int16))]
    assert tr._maybe_diarize(chans, SR, segs) is segs


def test_maybe_diarize_noop_for_stereo(tmp_path):
    tr = _cpu_transcriber(tmp_path)
    tr.diarize = {"enabled": True}   # включено, но каналы уже разделены
    segs = [_seg(0, 1)]
    chans = [("mic", "Я", np.zeros(10, dtype=np.int16)),
             ("loopback", "Собеседник", np.zeros(10, dtype=np.int16))]
    assert tr._maybe_diarize(chans, SR, segs) is segs


def test_maybe_diarize_graceful_on_error(tmp_path, monkeypatch):
    tr = _cpu_transcriber(tmp_path)
    tr.diarize = {"enabled": True}
    import diarizer
    monkeypatch.setattr(diarizer.Diarizer, "diarize",
                        lambda self, p: (_ for _ in ()).throw(RuntimeError("no model")))
    segs = [_seg(0, 1)]
    chans = [("audio", None, np.zeros(SR, dtype=np.int16))]
    out = tr._maybe_diarize(chans, SR, segs)
    assert out == segs   # транскрипция не падает, просто без меток спикеров


def test_maybe_diarize_assigns_speakers_for_single_track(tmp_path, monkeypatch):
    tr = _cpu_transcriber(tmp_path)
    tr.diarize = {"enabled": True}
    import diarizer
    monkeypatch.setattr(diarizer.Diarizer, "diarize",
                        lambda self, p: [(0.0, 2.0, "SPEAKER_00"),
                                         (2.0, 4.0, "SPEAKER_01")])
    segs = [_seg(0.0, 1.5, "a"), _seg(2.5, 3.5, "b")]
    chans = [("audio", None, np.zeros(SR * 4, dtype=np.int16))]
    out = tr._maybe_diarize(chans, SR, segs)
    assert out[0]["speaker"] == "Спикер 1"
    assert out[1]["speaker"] == "Спикер 2"
