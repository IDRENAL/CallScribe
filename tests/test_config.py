"""Тесты config.py — загрузка/слияние/пути (с временным CONFIG_PATH)."""
from __future__ import annotations

import json

import pytest

import config


@pytest.fixture
def tmp_config(tmp_path, monkeypatch):
    """Изолировать config.json и BASE во временную папку."""
    cfg_path = tmp_path / "config.json"
    monkeypatch.setattr(config, "CONFIG_PATH", cfg_path)
    monkeypatch.setattr(config, "BASE", tmp_path)
    return cfg_path


def test_load_config_returns_defaults_when_missing(tmp_config):
    cfg = config.load_config()
    assert cfg["language"] == "ru"
    assert cfg["mode"] == "accurate"
    assert cfg["device"] == "auto"
    assert cfg["cpu_workers"] is None
    assert config.config_exists() is False


def test_load_config_does_not_mutate_default():
    a = config.load_config()
    a["language"] = "en"
    b = config.load_config()
    assert b["language"] == "ru"   # дефолт не испортился


def test_save_then_load_roundtrip(tmp_config):
    cfg = config.load_config()
    cfg["mic_device"] = 3
    cfg["cpu_workers"] = 2
    config.save_config(cfg)
    assert config.config_exists() is True
    loaded = config.load_config()
    assert loaded["mic_device"] == 3
    assert loaded["cpu_workers"] == 2


def test_load_config_fills_missing_keys_from_defaults(tmp_config):
    # старый/частичный конфиг без новых ключей
    tmp_config.write_text(json.dumps({"mic_device": 1}), encoding="utf-8")
    cfg = config.load_config()
    assert cfg["mic_device"] == 1            # сохранён
    assert cfg["cpu_workers"] is None        # добавлен из дефолта
    assert cfg["device"] == "auto"
    assert cfg["output"]["recordings"] == "recordings"


def test_get_paths_are_absolute_and_created(tmp_config):
    cfg = config.load_config()
    paths = config.get_paths(cfg)
    for key in ("recordings", "transcripts", "models"):
        assert paths[key].is_absolute()
        assert paths[key].exists()
