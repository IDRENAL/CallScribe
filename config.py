"""Работа с config.json — единственным источником истины для настроек.

Все пути относительные, разрешаются от корня проекта (BASE).
"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).parent
CONFIG_PATH = BASE / "config.json"

DEFAULT_CONFIG: dict = {
    "mic_device": None,
    "loopback_device": None,
    "language": "ru",
    "whisper_model": None,  # None → авто-выбор (всегда large-v3 по плану)
    "mode": "accurate",     # "accurate" (раздельные каналы, один проход) | "fast"
    "device": "auto",       # "auto" (GPU если есть) | "cpu" | "cuda"
    "compute_type": None,   # None → авто; "float32"/"float16"/"int8"
    "vad": True,            # мягкий VAD (не режет тихую речь)
    "speaker_labels": {"mic": "Я", "loopback": "Собеседник"},
    "output": {
        "recordings": "recordings",
        "transcripts": "transcripts",
    },
}


def load_config() -> dict:
    """Прочитать config.json. Если его нет — вернуть копию дефолта."""
    if not CONFIG_PATH.exists():
        return json.loads(json.dumps(DEFAULT_CONFIG))
    with CONFIG_PATH.open(encoding="utf-8") as f:
        cfg = json.load(f)
    # Дополнить недостающие ключи дефолтами
    merged = json.loads(json.dumps(DEFAULT_CONFIG))
    merged.update(cfg)
    merged["output"] = {**DEFAULT_CONFIG["output"], **cfg.get("output", {})}
    return merged


def save_config(config: dict) -> None:
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def config_exists() -> bool:
    return CONFIG_PATH.exists()


def get_paths(config: dict) -> dict[str, Path]:
    """Абсолютные пути к директориям вывода, гарантированно созданные."""
    out = config.get("output", {})
    paths = {
        "recordings": BASE / out.get("recordings", "recordings"),
        "transcripts": BASE / out.get("transcripts", "transcripts"),
        "models": BASE / "models",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths
