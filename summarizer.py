"""Выжимка стенограммы через LLM (v2.0).

Local-first: по умолчанию локальный Ollama (HTTP, без облака и без новых
тяжёлых зависимостей — вызов через stdlib urllib). Облачный OpenAI — опционально.

Длинные стенограммы обрабатываются map-reduce: текст режется на части, каждая
сворачивается отдельно (map), затем частичные выжимки объединяются в финальную
(reduce) — так не упираемся в окно контекста модели.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

OLLAMA_DEFAULT_HOST = "http://localhost:11434"

# Порог одиночного прохода и размер чанка (в символах). Кириллица ≈ 3 симв./токен,
# поэтому 14000 симв. ≈ 4–5k токенов — комфортно для num_ctx=8192 с запасом на ответ.
SINGLE_PASS_CHARS = 14000
CHUNK_CHARS = 11000

SYSTEM_PROMPT = (
    "Ты — ассистент, который делает точную деловую выжимку телефонного разговора "
    "на русском языке. Опирайся только на текст стенограммы, ничего не выдумывай. "
    "Если чего-то нет в разговоре — не добавляй. Сохрани все важные детали: "
    "договорённости, цифры, сроки, имена."
)

FINAL_INSTRUCTION = (
    "Сделай выжимку разговора строго в формате Markdown со следующими разделами "
    "(пустые разделы опускай):\n"
    "## Краткое содержание\n"
    "## Ключевые решения\n"
    "## Задачи\n"
    "(списком, с ответственным и сроком, если они названы)\n"
    "## Открытые вопросы\n\n"
    "Текст стенограммы:\n"
)

MAP_INSTRUCTION = (
    "Это ФРАГМЕНТ длинного разговора. Кратко и без потери деталей выпиши из него: "
    "тезисы, решения, задачи, открытые вопросы. Markdown, без вступлений.\n\n"
    "Фрагмент:\n"
)

REDUCE_INSTRUCTION = (
    "Ниже — выжимки последовательных фрагментов одного разговора. Объедини их в "
    "одну цельную выжимку, убери повторы, сохрани все детали. Формат строго:\n"
    "## Краткое содержание\n## Ключевые решения\n## Задачи\n## Открытые вопросы\n\n"
    "Выжимки фрагментов:\n"
)


class SummarizerError(RuntimeError):
    """Понятная ошибка для UI (недоступен Ollama, пустой ответ и т.п.)."""


class Summarizer:
    def __init__(self, provider: str = "ollama", model: str = "qwen2.5:7b",
                 host: str = OLLAMA_DEFAULT_HOST, language: str = "ru",
                 num_ctx: int = 8192, api_key: str | None = None,
                 timeout: int = 600, progress=None):  # noqa: ANN001
        self.provider = (provider or "ollama").lower()
        self.model = model
        self.host = host.rstrip("/")
        self.language = language
        self.num_ctx = num_ctx
        self.api_key = api_key
        self.timeout = timeout
        self._progress = progress or (lambda pct: None)

    # -- публичный API ----------------------------------------------------- #
    def summarize(self, transcript_text: str, meta: dict | None = None) -> str:
        """Вернуть Markdown-выжимку. meta — шапка (файл/дата/длительность)."""
        text = (transcript_text or "").strip()
        if not text:
            raise SummarizerError("Пустая стенограмма — нечего сворачивать")

        chunks = self._split(text)
        if len(chunks) == 1:
            body = self._complete(SYSTEM_PROMPT, FINAL_INSTRUCTION + chunks[0])
            self._progress(100)
        else:
            partials = []
            for i, chunk in enumerate(chunks):
                partials.append(self._complete(SYSTEM_PROMPT, MAP_INSTRUCTION + chunk))
                self._progress(int((i + 1) / (len(chunks) + 1) * 100))
            body = self._complete(SYSTEM_PROMPT, REDUCE_INSTRUCTION + "\n\n".join(partials))
            self._progress(100)

        return self._render_md(body.strip(), meta or {})

    # -- нарезка ----------------------------------------------------------- #
    @staticmethod
    def _split(text: str) -> list[str]:
        """Разбить по строкам (не разрывая реплики) на чанки <= CHUNK_CHARS."""
        if len(text) <= SINGLE_PASS_CHARS:
            return [text]
        chunks: list[str] = []
        buf: list[str] = []
        size = 0
        for line in text.splitlines():
            if size + len(line) + 1 > CHUNK_CHARS and buf:
                chunks.append("\n".join(buf))
                buf, size = [], 0
            buf.append(line)
            size += len(line) + 1
        if buf:
            chunks.append("\n".join(buf))
        return chunks

    # -- провайдеры -------------------------------------------------------- #
    def _complete(self, system: str, user: str) -> str:
        if self.provider == "ollama":
            return self._ollama_chat(system, user)
        if self.provider == "openai":
            return self._openai_chat(system, user)
        raise SummarizerError(f"Неизвестный provider выжимки: {self.provider}")

    def _ollama_chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "options": {"num_ctx": self.num_ctx, "temperature": 0.2},
        }
        data = self._post_json(f"{self.host}/api/chat", payload)
        content = (data.get("message") or {}).get("content", "").strip()
        if not content:
            raise SummarizerError("Ollama вернул пустой ответ")
        return content

    def _openai_chat(self, system: str, user: str) -> str:
        if not self.api_key:
            raise SummarizerError("Для provider=openai нужен api_key в config.summary")
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0.2,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = self._post_json("https://api.openai.com/v1/chat/completions",
                               payload, headers)
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            raise SummarizerError(f"Неожиданный ответ OpenAI: {e}") from e

    def _post_json(self, url: str, payload: dict, extra_headers: dict | None = None) -> dict:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", **(extra_headers or {})}
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise SummarizerError(
                f"Не удалось обратиться к LLM ({url}): {e}. "
                "Проверь, что Ollama запущен (`ollama serve`) и модель скачана."
            ) from e

    # -- рендер ------------------------------------------------------------ #
    def _render_md(self, body: str, meta: dict) -> str:
        lines = ["# Выжимка звонка", ""]
        if meta.get("source_file"):
            lines.append(f"**Файл:** {meta['source_file']}  ")
        if meta.get("date"):
            lines.append(f"**Дата:** {meta['date']}  ")
        if meta.get("duration_min") is not None:
            lines.append(f"**Длительность:** {meta['duration_min']:.1f} мин  ")
        lines.append(f"**Модель выжимки:** {self.model} ({self.provider})  ")
        lines += ["", "---", "", body, ""]
        return "\n".join(lines)
