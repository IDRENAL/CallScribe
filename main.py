"""CallScribe — точка входа и CLI-роутер.

    python main.py ui                 веб-интерфейс
    python main.py run                запись + транскрипция
    python main.py record             только запись
    python main.py transcribe <wav>   только транскрипция
    python main.py last               обработать последний WAV
    python main.py setup              мастер настройки устройств
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path

from config import config_exists, get_paths, load_config


def _require_config() -> dict:
    if not config_exists():
        print("✗ Нет config.json. Запусти: python main.py setup")
        sys.exit(1)
    return load_config()


def cmd_ui(args) -> None:
    from ui import run_ui
    run_ui(host=args.host, port=args.port)


def cmd_record(args) -> Path:
    cfg = _require_config()
    paths = get_paths(cfg)
    from recorder import CallRecorder
    rec = CallRecorder(cfg.get("mic_device"), cfg.get("loopback_device"),
                       paths["recordings"])
    print("● Запись. Нажми Enter чтобы остановить…")
    return rec.record()


def _transcribe_path(cfg: dict, wav_path: Path, device: str | None = None) -> None:
    paths = get_paths(cfg)
    from transcriber import Transcriber
    tr = Transcriber(paths["transcripts"], paths["models"],
                     language=cfg.get("language", "ru"),
                     model_name=cfg.get("whisper_model"),
                     mode=cfg.get("mode", "accurate"),
                     compute_type=cfg.get("compute_type"),
                     vad=cfg.get("vad", True),
                     speaker_labels=cfg.get("speaker_labels"),
                     device=device or cfg.get("device", "auto"))
    tr.transcribe(wav_path)


def cmd_transcribe(args) -> None:
    cfg = _require_config()
    wav = Path(args.wav)
    if not wav.exists():
        print(f"✗ Нет файла: {wav}")
        sys.exit(1)
    _transcribe_path(cfg, wav, getattr(args, "device", None))


def cmd_run(args) -> None:
    cfg = _require_config()
    wav = cmd_record(args)
    _transcribe_path(cfg, wav, getattr(args, "device", None))


def cmd_last(args) -> None:
    cfg = _require_config()
    paths = get_paths(cfg)
    wavs = sorted(paths["recordings"].glob("call_*.wav"))
    if not wavs:
        print("✗ Нет записей в recordings/")
        sys.exit(1)
    print(f"✓ Последняя запись: {wavs[-1].name}")
    _transcribe_path(cfg, wavs[-1], getattr(args, "device", None))


def cmd_setup(args) -> None:
    import setup
    setup.run_setup()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="callscribe", description="CallScribe")
    sub = p.add_subparsers(dest="command", required=True)

    p_ui = sub.add_parser("ui", help="веб-интерфейс")
    p_ui.add_argument("--host", default="127.0.0.1")
    p_ui.add_argument("--port", type=int, default=5000)
    p_ui.set_defaults(func=cmd_ui)

    p_run = sub.add_parser("run", help="запись + транскрипция")
    p_run.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    p_run.set_defaults(func=cmd_run)
    sub.add_parser("record", help="только запись").set_defaults(func=cmd_record)

    p_tr = sub.add_parser("transcribe", help="только транскрипция")
    p_tr.add_argument("wav")
    p_tr.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None,
                      help="на чём считать (по умолчанию из config.json)")
    p_tr.set_defaults(func=cmd_transcribe)

    p_last = sub.add_parser("last", help="обработать последний WAV")
    p_last.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    p_last.set_defaults(func=cmd_last)
    sub.add_parser("setup", help="мастер настройки").set_defaults(func=cmd_setup)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    multiprocessing.freeze_support()  # обязательно для Windows + multiprocessing
    main()
