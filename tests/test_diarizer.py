"""Тесты диаризации — чистая логика назначения спикеров (без pyannote)."""
from __future__ import annotations

import pytest

from diarizer import DiarizerError, _overlap, assign_speakers


# --------------------------------------------------------------------------- #
#  _overlap
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("a,b,exp", [
    ((0, 10), (5, 15), 5),
    ((0, 5), (5, 10), 0),     # касание — пересечения нет
    ((0, 10), (3, 7), 4),     # вложенный
    ((0, 2), (5, 8), 0),      # не пересекаются
])
def test_overlap(a, b, exp):
    assert _overlap(a[0], a[1], b[0], b[1]) == exp


# --------------------------------------------------------------------------- #
#  assign_speakers
# --------------------------------------------------------------------------- #
def test_assign_by_max_overlap():
    segments = [
        {"start": 0.0, "end": 2.0, "text": "привет"},
        {"start": 3.0, "end": 5.0, "text": "здравствуйте"},
    ]
    turns = [(0.0, 2.5, "SPEAKER_00"), (2.5, 6.0, "SPEAKER_01")]
    out = assign_speakers(segments, turns)
    assert out[0]["speaker"] == "Спикер 1"
    assert out[1]["speaker"] == "Спикер 2"


def test_labels_numbered_by_first_appearance():
    segments = [{"start": 0.0, "end": 1.0, "text": "x"}]
    # turns намеренно не по порядку — имена по времени старта, не по строке метки
    turns = [(10.0, 12.0, "SPEAKER_09"), (0.0, 2.0, "SPEAKER_03")]
    out = assign_speakers(segments, turns)
    # SPEAKER_03 появляется раньше → «Спикер 1»; сегмент 0-1 пересекается с ним
    assert out[0]["speaker"] == "Спикер 1"


def test_segment_without_overlap_keeps_no_speaker():
    segments = [{"start": 100.0, "end": 101.0, "text": "вне диапазона"}]
    turns = [(0.0, 5.0, "SPEAKER_00")]
    out = assign_speakers(segments, turns)
    assert "speaker" not in out[0] or out[0].get("speaker") is None


def test_assign_does_not_mutate_input():
    segments = [{"start": 0.0, "end": 2.0, "text": "x"}]
    turns = [(0.0, 2.0, "SPEAKER_00")]
    assign_speakers(segments, turns)
    assert "speaker" not in segments[0]   # исходный список не изменён


def test_custom_label_format():
    segments = [{"start": 0.0, "end": 1.0, "text": "x"}]
    turns = [(0.0, 2.0, "SPEAKER_00")]
    out = assign_speakers(segments, turns, label_fmt="Участник {n}")
    assert out[0]["speaker"] == "Участник 1"


# --------------------------------------------------------------------------- #
#  Загрузка pipeline без установленного pyannote → понятная ошибка
# --------------------------------------------------------------------------- #
def test_load_pipeline_without_pyannote_raises(monkeypatch):
    import builtins

    from diarizer import Diarizer

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "pyannote.audio" or name.startswith("pyannote"):
            raise ImportError("no pyannote")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(DiarizerError):
        Diarizer()._load_pipeline()
