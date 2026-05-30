"""Тесты вспомогательных функций ui.py (без поднятия сервера)."""
from __future__ import annotations

import ui


def test_safe_name_strips_path_separators():
    assert ui._safe_name("../../etc/passwd") == "passwd"
    assert ui._safe_name("/abs/path/call.mp4") == "call.mp4"


def test_safe_name_drops_weird_chars_keeps_allowed():
    out = ui._safe_name("за пись@#$.wav")
    assert "@" not in out and "#" not in out
    assert out.endswith(".wav")


def test_safe_name_empty_falls_back():
    assert ui._safe_name("") == "upload"


def test_media_exts_cover_common_formats():
    for ext in (".wav", ".mp4", ".mkv", ".mp3", ".m4a"):
        assert ext in ui.MEDIA_EXTS
