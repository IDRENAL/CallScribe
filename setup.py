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


def _ask_mic() -> int | str | None:
    """Микрофон: число — индекс sounddevice; на Linux можно имя ('pulse'/'default').

    На PipeWire/PulseAudio индекс 'pulse' часто не открывает моно, поэтому по
    умолчанию (Enter) предлагаем имя 'pulse' — оно пишет с default source.
    """
    linux = sys.platform.startswith("linux")
    default = "pulse" if linux else None
    hint = " [Enter — 'pulse']" if linux else " (Enter — пропустить)"
    while True:
        raw = input(f"Микрофон: ID или имя устройства{hint}: ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            return raw  # имя устройства, напр. 'pulse'/'default'


def _ask_loopback(cands: list) -> int | str | None:
    """Выбрать loopback: номер из списка кандидатов, либо ручной индекс, либо пропуск."""
    if cands:
        prompt = "Номер loopback-источника из списка (Enter — пропустить): "
    else:
        prompt = "ID loopback-устройства (Enter — пропустить): "
    while True:
        raw = input(prompt).strip()
        if not raw:
            return None
        try:
            num = int(raw)
        except ValueError:
            print("  Введи число (или пусто чтобы пропустить).")
            continue
        if cands:
            if 0 <= num < len(cands):
                return cands[num][0]  # идентификатор (str-имя или int-индекс)
            print(f"  Номер должен быть 0..{len(cands) - 1}.")
            continue
        return num  # кандидатов нет — трактуем как индекс sounddevice


def run_setup() -> None:
    try:
        import sounddevice  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"✗ sounddevice недоступен: {e}")
        print("  Установи зависимости: uv sync")
        sys.exit(1)

    print("=== CallScribe — настройка ===")
    _print_devices()

    from recorder import get_loopback_candidates
    cands = get_loopback_candidates()
    if cands:
        print("\nВозможные loopback-источники (системный звук), первый — рекомендуемый:")
        for n, (_ident, name) in enumerate(cands):
            print(f"  ({n}) {name}")
    else:
        print("\n⚠ Loopback-источники не найдены автоматически.")
        if sys.platform == "win32":
            print("  Windows: установи VB-Cable или включи 'Стерео микшер'.")
        elif sys.platform.startswith("linux"):
            print("  Linux: проверь monitor-источник: pactl list sources short")

    cfg = load_config()
    print()
    mic = _ask_mic()
    loop = _ask_loopback(cands)
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
