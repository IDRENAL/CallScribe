"""Веб-интерфейс: FastAPI + WebSocket.

WebSocket — только push сервер→браузер (лог, статус, результаты).
Команды идут обычными HTTP POST.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from config import get_paths, load_config
from exporter import build_export
from recorder import CallRecorder
from summarizer import Summarizer, SummarizerError
from transcriber import Transcriber, TranscriptionCancelled, cuda_available

BASE = Path(__file__).parent
TEMPLATES = BASE / "templates"


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    manager.loop = asyncio.get_running_loop()
    sys.stdout = WsLogStream(sys.__stdout__)
    yield
    sys.stdout = sys.__stdout__


app = FastAPI(title="CallScribe", lifespan=lifespan)


# --------------------------------------------------------------------------- #
#  ConnectionManager
# --------------------------------------------------------------------------- #
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []
        self.loop: asyncio.AbstractEventLoop | None = None

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active = [c for c in self.active if c is not ws]

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(json.dumps(data, ensure_ascii=False))
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    def broadcast_sync(self, data: dict):
        """Вызов из обычного (не-async) потока."""
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast(data), self.loop)


manager = ConnectionManager()


# --------------------------------------------------------------------------- #
#  Перехват print() → WebSocket
# --------------------------------------------------------------------------- #
class WsLogStream(io.TextIOBase):
    def __init__(self, original_stream):
        self.original = original_stream

    def write(self, text: str) -> int:
        if text.strip():
            level = ("ok" if text.startswith("✓") else
                     "err" if text.startswith("✗") else
                     "warn" if text.startswith("⚠") else "info")
            manager.broadcast_sync({"type": "log", "text": text.rstrip(), "level": level})
        self.original.write(text)
        return len(text)

    def flush(self):
        self.original.flush()


# --------------------------------------------------------------------------- #
#  Глобальное состояние
# --------------------------------------------------------------------------- #
class AppState:
    recording: bool = False
    busy: bool = False
    recorder: CallRecorder | None = None
    _rec_stop_event: threading.Event | None = None
    _job_thread: threading.Thread | None = None
    _cancel_event: threading.Event | None = None


state = AppState()


def _config_and_paths():
    cfg = load_config()
    return cfg, get_paths(cfg)


# --------------------------------------------------------------------------- #
#  Routes
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
async def index():
    return (TEMPLATES / "index.html").read_text(encoding="utf-8")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        await ws.send_text(json.dumps(
            {"type": "files", "files": _list_files()}, ensure_ascii=False))
        while True:
            await ws.receive_text()  # держим соединение; входящие игнорируем
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:  # noqa: BLE001
        manager.disconnect(ws)


MEDIA_EXTS = {".wav", ".mp4", ".mkv", ".mov", ".webm", ".m4a", ".mp3", ".ogg", ".flac"}


def _find_source(stem: str):
    """Найти исходный медиафайл по stem в recordings (предпочитая .wav)."""
    cfg, paths = _config_and_paths()
    for ext in [".wav", *sorted(MEDIA_EXTS - {".wav"})]:
        p = paths["recordings"] / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _list_files() -> list[dict]:
    cfg, paths = _config_and_paths()
    recordings = paths["recordings"]
    transcripts = paths["transcripts"]
    media = [p for p in recordings.iterdir()
             if p.is_file() and p.suffix.lower() in MEDIA_EXTS]
    files = []
    for src in sorted(media, key=lambda p: p.stat().st_mtime, reverse=True):
        stem = src.stem
        stat = src.stat()
        try:
            dt = datetime.strptime(stem, "call_%Y-%m-%d_%H-%M-%S").strftime("%Y-%m-%d %H:%M")
        except ValueError:
            dt = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        files.append({
            "stem": stem,
            "name": src.name,
            "date": dt,
            "size_mb": round(stat.st_size / 1024 ** 2, 1),
            "has_transcript": (transcripts / f"{stem}_transcript.md").exists(),
            "has_summary": (transcripts / f"{stem}_summary.md").exists(),
        })
    return files


@app.get("/api/files")
async def api_files():
    return JSONResponse(_list_files())


@app.get("/api/content")
async def api_content(type: str, stem: str):  # noqa: A002
    cfg, paths = _config_and_paths()
    suffix = "_transcript.md" if type == "transcript" else "_summary.md"
    path = paths["transcripts"] / f"{stem}{suffix}"
    if not path.exists():
        return JSONResponse({"ok": False, "reason": "not_found"})
    return JSONResponse({"ok": True, "content": path.read_text(encoding="utf-8"),
                         "content_type": "markdown", "file_path": str(path)})


@app.post("/api/record/start")
async def api_record_start():
    if state.busy or state.recording:
        return JSONResponse({"ok": False, "reason": "busy"})
    cfg, paths = _config_and_paths()
    state.recording = True
    stop_event = threading.Event()
    state._rec_stop_event = stop_event
    state.recorder = CallRecorder(cfg.get("mic_device"), cfg.get("loopback_device"),
                                  paths["recordings"])

    def _worker():
        try:
            manager.broadcast_sync({"type": "recording_started"})
            state.recorder.record(stop_event=stop_event)
        except Exception as e:  # noqa: BLE001
            manager.broadcast_sync({"type": "job_error", "error": str(e)})
        finally:
            state.recording = False
            manager.broadcast_sync({"type": "recording_stopped"})
            manager.broadcast_sync({"type": "files", "files": _list_files()})

    t = threading.Thread(target=_worker, daemon=True)
    state._job_thread = t
    t.start()
    return JSONResponse({"ok": True})


@app.post("/api/job/stop")
async def api_job_stop():
    """Прервать текущую обработку (транскрипцию) с потерей результата."""
    if not state.busy or state._cancel_event is None:
        return JSONResponse({"ok": False, "reason": "no_job"})
    state._cancel_event.set()
    return JSONResponse({"ok": True})


@app.post("/api/record/stop")
async def api_record_stop():
    if not state.recording or not state.recorder:
        return JSONResponse({"ok": False, "reason": "not_recording"})
    state.recorder.stop()
    return JSONResponse({"ok": True})


def _run_job(label: str, fn):
    """Запустить долгую задачу в фоновом потоке с broadcast-уведомлениями."""
    if state.busy or state.recording:
        return JSONResponse({"ok": False, "reason": "busy"})
    state.busy = True
    state._cancel_event = threading.Event()

    def _worker():
        try:
            manager.broadcast_sync({"type": "job_started", "label": label})
            fn()
            manager.broadcast_sync({"type": "files", "files": _list_files()})
        except TranscriptionCancelled:
            print("■ Обработка остановлена пользователем")
            manager.broadcast_sync({"type": "job_cancelled"})
            manager.broadcast_sync({"type": "files", "files": _list_files()})
        except Exception as e:  # noqa: BLE001
            manager.broadcast_sync({"type": "job_error", "error": str(e)})
        finally:
            state.busy = False
            state._cancel_event = None

    t = threading.Thread(target=_worker, daemon=True)
    state._job_thread = t
    t.start()
    return JSONResponse({"ok": True})


def _do_transcribe_path(src: Path, device: str | None = None):
    cfg, paths = _config_and_paths()
    tr = Transcriber(paths["transcripts"], paths["models"],
                     language=cfg.get("language", "ru"),
                     model_name=cfg.get("whisper_model"),
                     mode=cfg.get("mode", "accurate"),
                     compute_type=cfg.get("compute_type"),
                     vad=cfg.get("vad", True),
                     speaker_labels=cfg.get("speaker_labels"),
                     device=device or cfg.get("device", "auto"),
                     cpu_workers=cfg.get("cpu_workers"))
    tr.transcribe(src, progress=lambda pct: manager.broadcast_sync(
        {"type": "job_progress", "pct": pct}),
        cancel=lambda: state._cancel_event is not None and state._cancel_event.is_set())
    md = paths["transcripts"] / f"{src.stem}_transcript.md"
    manager.broadcast_sync({
        "type": "job_done", "label": "Транскрипция готова",
        "content": md.read_text(encoding="utf-8"),
        "content_type": "transcript", "file_path": str(md)})


def _do_transcribe(stem: str, device: str | None = None):
    src = _find_source(stem)
    if src is None:
        raise FileNotFoundError(f"Нет исходного файла для {stem}")
    _do_transcribe_path(src, device)


def _do_summarize(stem: str):
    """Сделать выжимку из готовой стенограммы (без повторной транскрипции)."""
    cfg, paths = _config_and_paths()
    scfg = cfg.get("summary", {})
    if not scfg.get("enabled", True) or scfg.get("provider") == "none":
        print("⚠ Выжимка отключена в config.summary")
        return
    json_path = paths["transcripts"] / f"{stem}_transcript.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Нет стенограммы для выжимки: {stem}")
    data = json.loads(json_path.read_text(encoding="utf-8"))

    sm = Summarizer(provider=scfg.get("provider", "ollama"),
                    model=scfg.get("model", "qwen2.5:7b"),
                    host=scfg.get("host", "http://localhost:11434"),
                    language=scfg.get("language", "ru"),
                    num_ctx=scfg.get("num_ctx", 8192),
                    api_key=scfg.get("api_key"),
                    progress=lambda pct: manager.broadcast_sync(
                        {"type": "job_progress", "pct": pct}))
    print(f"✓ Выжимка: {sm.provider}/{sm.model}…")
    meta = {"source_file": Path(data.get("source_file", stem)).name,
            "date": data.get("transcribed_at", "").replace("T", " ")[:16],
            "duration_min": (data.get("duration_seconds") or 0) / 60}
    summary_md = sm.summarize(data.get("full_text", ""), meta)

    md_path = paths["transcripts"] / f"{stem}_summary.md"
    md_path.write_text(summary_md, encoding="utf-8")
    print(f"✓ Выжимка готова: {md_path.name}")
    manager.broadcast_sync({
        "type": "job_done", "label": "Выжимка готова",
        "content": summary_md, "content_type": "summary", "file_path": str(md_path)})


def _do_process(stem: str, device: str | None = None):
    """Полная обработка: транскрипция + выжимка."""
    _do_transcribe(stem, device)
    _do_summarize(stem)


@app.get("/api/info")
async def api_info():
    cfg, _ = _config_and_paths()
    return JSONResponse({"has_gpu": cuda_available(),
                         "default_device": cfg.get("device", "auto")})


@app.post("/api/transcribe")
async def api_transcribe(body: dict):
    stem = body.get("stem")
    if not stem:
        return JSONResponse({"ok": False, "reason": "no_stem"})
    device = body.get("device")
    return _run_job("Транскрипция…", lambda: _do_transcribe(stem, device))


@app.post("/api/process")
async def api_process(body: dict):
    # Полная обработка: транскрипция + LLM-выжимка
    stem = body.get("stem")
    if not stem:
        return JSONResponse({"ok": False, "reason": "no_stem"})
    device = body.get("device")
    return _run_job("Обработка…", lambda: _do_process(stem, device))


@app.post("/api/summarize")
async def api_summarize(body: dict):
    # Выжимка из уже готовой стенограммы (без повторной транскрипции)
    stem = body.get("stem")
    if not stem:
        return JSONResponse({"ok": False, "reason": "no_stem"})
    return _run_job("Выжимка…", lambda: _do_summarize(stem))


def _safe_name(filename: str) -> str:
    """Безопасное имя файла: только базовое имя, без разделителей путей."""
    base = Path(filename or "upload").name
    return "".join(c for c in base if c.isalnum() or c in " ._-()").strip() or "upload"


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...), device: str = Form("auto")):  # noqa: ARG001
    if state.busy or state.recording:
        return JSONResponse({"ok": False, "reason": "busy"})
    cfg, paths = _config_and_paths()
    name = _safe_name(file.filename)
    if Path(name).suffix.lower() not in MEDIA_EXTS:
        return JSONResponse({"ok": False, "reason": "unsupported_format"})

    dest = paths["recordings"] / name
    # Потоковая запись на диск (файлы могут быть большими — видео звонка)
    with dest.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
    print(f"✓ Загружен файл: {name} ({dest.stat().st_size / 1024 ** 2:.1f} MB) "
          f"— выбери его и запусти обработку")

    # Транскрипция НЕ стартует автоматически — ждём запуска пользователем.
    manager.broadcast_sync({"type": "files", "files": _list_files()})
    return JSONResponse({"ok": True, "stem": Path(name).stem})


@app.get("/api/export")
async def api_export(stem: str):
    """Собрать и отдать <stem>.docx (стенограмма + выжимка) на скачивание."""
    cfg, paths = _config_and_paths()
    try:
        out = build_export(stem, paths["transcripts"])
    except FileNotFoundError:
        return JSONResponse({"ok": False, "reason": "no_transcript"}, status_code=404)
    print(f"✓ Экспорт DOCX: {out.name}")
    return FileResponse(
        out, filename=out.name,
        media_type="application/vnd.openxmlformats-officedocument."
                   "wordprocessingml.document")


@app.post("/api/open")
async def api_open(body: dict):
    path = body.get("path")
    if not path or not Path(path).exists():
        return JSONResponse({"ok": False, "reason": "not_found"})
    _open_in_system(path)
    return JSONResponse({"ok": True})


def _open_in_system(path: str):
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


# --------------------------------------------------------------------------- #
#  Запуск
# --------------------------------------------------------------------------- #
def run_ui(host: str = "127.0.0.1", port: int = 5000):
    import webbrowser
    url = f"http://{host}:{port}"

    def _open():
        import time
        time.sleep(1.2)
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()
    print(f"✓ CallScribe UI: {url}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run_ui()
