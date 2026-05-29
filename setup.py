"""Мастер первичной настройки: выбор аудио-устройств → config.json.

Запуск: python main.py setup  (или python setup.py)
"""
from __future__ import annotations

import sys

from config import DEFAULT_CONFIG, load_config, save_config


def _print_devices() -> None:
    import sounddevice as sd
    print("\nДоступные устройства (index — name — in/out каналы):\n")
    for i, dev in enumerate(sd.query_devices()):
        ins, outs = dev["max_input_channels"], dev["max_output_channels"]
        tag = []
        if ins > 0:
            tag.append("вход")
        if outs > 0:
            tag.append("выход")
        print(f"  [{i:>2}] {dev['name']}  ({', '.join(tag)}; in={ins} out={outs})")


def _ask_int(prompt: str, allow_none: bool = True) -> int | None:
    while True:
        raw = input(prompt).strip()
        if not raw and allow_none:
            return None
        try:
            return int(raw)
        except ValueError:
            print("  Введи число (или пусто чтобы пропустить).")


def run_setup() -> None:
    try:
        import sounddevice  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"✗ sounddevice недоступен: {e}")
        print("  Установи зависимости: pip install -r requirements.txt")
        sys.exit(1)

    print("=== CallScribe — настройка ===")
    _print_devices()

    from recorder import get_loopback_candidates
    cands = get_loopback_candidates()
    if cands:
        print("\nВозможные loopback-устройства (системный звук):")
        for idx, name in cands:
            print(f"  [{idx}] {name}")
    else:
        print("\n⚠ Loopback-устройства не найдены автоматически.")
        if sys.platform == "win32":
            print("  Windows: установи VB-Cable или включи 'Стерео микшер'.")
        elif sys.platform.startswith("linux"):
            print("  Linux: проверь monitor-источник: pactl list sources short")

    cfg = load_config()
    print()
    mic = _ask_int("ID микрофона (вход): ")
    loop = _ask_int("ID системного звука/loopback (Enter — пропустить): ")
    lang = input(f"Язык [{cfg.get('language', 'ru')}]: ").strip() or cfg.get("language", "ru")

    cfg["mic_device"] = mic
    cfg["loopback_device"] = loop
    cfg["language"] = lang
    cfg.setdefault("whisper_model", None)
    cfg.setdefault("output", DEFAULT_CONFIG["output"])

    save_config(cfg)
    print("\n✓ Сохранено в config.json")
    print(f"  mic_device={mic}  loopback_device={loop}  language={lang}")
    print("\nЗапуск: python main.py ui")


if __name__ == "__main__":
    run_setup()
