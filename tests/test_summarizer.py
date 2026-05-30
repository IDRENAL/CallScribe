"""Тесты summarizer.py — оркестрация без реального LLM (провайдер мокается)."""
from __future__ import annotations

import pytest

import summarizer
from summarizer import CHUNK_CHARS, SINGLE_PASS_CHARS, Summarizer, SummarizerError


def _fixed(sm: Summarizer, text="## Краткое содержание\nок", calls=None):
    """Подменить обращение к LLM на детерминированную заглушку."""
    def fake(system, user):
        if calls is not None:
            calls.append(user)
        return text
    sm._complete = fake  # type: ignore[assignment]
    return sm


# --------------------------------------------------------------------------- #
#  Нарезка
# --------------------------------------------------------------------------- #
def test_split_short_text_single_chunk():
    assert Summarizer._split("короткий текст") == ["короткий текст"]


def test_split_long_text_multiple_chunks_within_budget():
    line = "реплика собеседника номер такой-то с деталями"
    text = "\n".join([line] * 2000)  # заведомо длиннее SINGLE_PASS_CHARS
    assert len(text) > SINGLE_PASS_CHARS
    chunks = Summarizer._split(text)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= CHUNK_CHARS


def test_split_does_not_break_lines():
    line = "x" * 100
    text = "\n".join([line] * 1000)
    chunks = Summarizer._split(text)
    # строки целы: пересборка даёт тот же набор непустых строк
    rejoined = "\n".join(chunks).splitlines()
    assert rejoined == text.splitlines()


# --------------------------------------------------------------------------- #
#  Оркестрация single-pass / map-reduce
# --------------------------------------------------------------------------- #
def test_single_pass_calls_llm_once():
    sm = Summarizer()
    calls = []
    _fixed(sm, calls=calls)
    md = sm.summarize("короткий разговор")
    assert len(calls) == 1
    assert "# Выжимка звонка" in md


def test_map_reduce_calls_llm_per_chunk_plus_reduce():
    sm = Summarizer()
    calls = []
    _fixed(sm, calls=calls)
    text = "\n".join(["строка разговора с содержанием"] * 3000)
    n_chunks = len(Summarizer._split(text))
    assert n_chunks > 1
    sm.summarize(text)
    assert len(calls) == n_chunks + 1   # map по чанкам + один reduce


def test_progress_reaches_100():
    sm = Summarizer()
    seen = []
    sm._progress = seen.append
    _fixed(sm)
    sm.summarize("короткий разговор")
    assert seen and seen[-1] == 100


# --------------------------------------------------------------------------- #
#  Ошибки и провайдеры
# --------------------------------------------------------------------------- #
def test_empty_transcript_raises():
    with pytest.raises(SummarizerError):
        Summarizer().summarize("   ")


def test_unknown_provider_raises():
    sm = Summarizer(provider="nonexistent")
    with pytest.raises(SummarizerError):
        sm.summarize("разговор")


def test_openai_without_key_raises():
    sm = Summarizer(provider="openai")
    with pytest.raises(SummarizerError):
        sm.summarize("разговор")


# --------------------------------------------------------------------------- #
#  Формирование запроса к Ollama и рендер
# --------------------------------------------------------------------------- #
def test_ollama_payload_has_model_messages_and_num_ctx():
    sm = Summarizer(model="qwen2.5:7b", num_ctx=4096)
    captured = {}

    def fake_post(url, payload, extra_headers=None):
        captured["url"] = url
        captured["payload"] = payload
        return {"message": {"content": "## Краткое содержание\nготово"}}

    sm._post_json = fake_post  # type: ignore[assignment]
    md = sm.summarize("короткий разговор")

    assert captured["url"].endswith("/api/chat")
    p = captured["payload"]
    assert p["model"] == "qwen2.5:7b"
    assert p["options"]["num_ctx"] == 4096
    assert p["stream"] is False
    roles = [m["role"] for m in p["messages"]]
    assert roles == ["system", "user"]
    assert "готово" in md


def test_render_md_contains_header_meta_and_model():
    sm = Summarizer(model="llama3.1:8b", provider="ollama")
    _fixed(sm, text="## Краткое содержание\nитог")
    md = sm.summarize("разговор", meta={"source_file": "call.mp4",
                                         "date": "2026-05-30 09:20",
                                         "duration_min": 12.3})
    assert "# Выжимка звонка" in md
    assert "call.mp4" in md
    assert "llama3.1:8b (ollama)" in md
    assert "итог" in md
