"""Диаризация спикеров для ОДНОЙ сведённой дорожки (v2.0).

Нужна только для источников без разделения каналов — например, загруженного
mp4 звонка (одна дорожка). Собственные стерео-записи CallScribe (L=mic, R=система)
уже разделены по каналам, там диаризация не требуется.

Движок — pyannote.audio (опциональный extra: `uv sync --extra diarize`, тянет
torch). Импортируется лениво, поэтому базовая установка без него не ломается.
Чистая логика назначения спикеров сегментам (assign_speakers) от pyannote не
зависит и тестируется отдельно.
"""
from __future__ import annotations

from pathlib import Path

DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"


class DiarizerError(RuntimeError):
    """Понятная ошибка (нет extra, нет HF-токена, модель не скачалась)."""


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Длина пересечения интервалов [a0,a1] и [b0,b1] (>= 0)."""
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_speakers(segments: list[dict], turns: list[tuple],
                    label_fmt: str = "Спикер {n}") -> list[dict]:
    """Проставить segments[*]['speaker'] по максимальному перекрытию с turns.

    turns — список (start, end, raw_label) от диаризатора. Сырые метки
    (SPEAKER_00…) переименовываются в человекочитаемые по порядку появления.
    Сегменты, не пересёкшиеся ни с одним turn, остаются без спикера.
    """
    # человекочитаемые имена по первому появлению (turns по времени старта)
    order: list[str] = []
    for _s, _e, lab in sorted(turns, key=lambda t: t[0]):
        if lab not in order:
            order.append(lab)
    names = {lab: label_fmt.format(n=i + 1) for i, lab in enumerate(order)}

    out: list[dict] = []
    for seg in segments:
        best_lab, best_ov = None, 0.0
        for s, e, lab in turns:
            ov = _overlap(seg["start"], seg["end"], s, e)
            if ov > best_ov:
                best_ov, best_lab = ov, lab
        seg = dict(seg)
        if best_lab is not None:
            seg["speaker"] = names[best_lab]
        out.append(seg)
    return out


class Diarizer:
    def __init__(self, model: str = DEFAULT_MODEL, hf_token: str | None = None,
                 device: str = "auto", min_speakers: int | None = None,
                 max_speakers: int | None = None):
        self.model = model
        self.hf_token = hf_token
        self.device = device
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

    def diarize(self, wav_path: str | Path) -> list[tuple]:
        """Вернуть список (start, end, raw_label) для дорожки wav_path."""
        pipeline = self._load_pipeline()
        kwargs = {}
        if self.min_speakers is not None:
            kwargs["min_speakers"] = self.min_speakers
        if self.max_speakers is not None:
            kwargs["max_speakers"] = self.max_speakers
        annotation = pipeline(str(wav_path), **kwargs)
        return [(turn.start, turn.end, label)
                for turn, _, label in annotation.itertracks(yield_label=True)]

    def _load_pipeline(self):  # noqa: ANN202
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as e:
            raise DiarizerError(
                "pyannote.audio не установлен. Поставь: uv sync --extra diarize"
            ) from e
        try:
            pipeline = Pipeline.from_pretrained(self.model, use_auth_token=self.hf_token)
        except Exception as e:  # noqa: BLE001
            raise DiarizerError(
                f"Не удалось загрузить модель {self.model}: {e}. Нужен HF-токен "
                "(config.diarize.hf_token) и принятые условия модели на huggingface.co."
            ) from e
        if pipeline is None:
            raise DiarizerError(
                "Pipeline=None — обычно не принят токен/условия модели на huggingface.co")
        use_cuda = self.device == "cuda" or (self.device == "auto" and torch.cuda.is_available())
        if use_cuda:
            pipeline.to(torch.device("cuda"))
        return pipeline
