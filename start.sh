#!/usr/bin/env bash
# CallScribe — запуск всего одной командой.
# Поднимает локальный Ollama (для выжимки, если установлен) и веб-интерфейс.
# Аргументы пробрасываются в UI, например:  ./start.sh --host 0.0.0.0 --port 8080
set -euo pipefail
cd "$(dirname "$0")"

export PATH="$HOME/.local/bin:$PATH"
OLLAMA_URL="http://localhost:11434/api/version"

# 1) Ollama (нужен только для выжимки; транскрипция работает и без него).
if ! curl -fsS "$OLLAMA_URL" >/dev/null 2>&1; then
  if systemctl --user list-unit-files ollama.service >/dev/null 2>&1; then
    echo "▶ Запускаю Ollama (systemd --user)…"
    systemctl --user start ollama 2>/dev/null || true
  elif command -v ollama >/dev/null 2>&1; then
    echo "▶ Запускаю Ollama…"
    nohup ollama serve >/tmp/ollama.log 2>&1 &
  fi
  # ждём до 10 c, пока сервер ответит (необязательно для старта UI)
  for _ in $(seq 1 10); do
    curl -fsS "$OLLAMA_URL" >/dev/null 2>&1 && break
    sleep 1
  done
fi

if curl -fsS "$OLLAMA_URL" >/dev/null 2>&1; then
  echo "✓ Ollama готов (выжимка доступна)"
else
  echo "⚠ Ollama не запущен — выжимка будет недоступна (транскрипция работает)"
fi

# 2) Синхронизация окружения. ВАЖНО: при наличии NVIDIA ставим extra gpu —
#    иначе CUDA-библиотеки (libcublas/cudnn) не подтянутся и GPU-транскрипция
#    молча выдаст пустой результат. Дальше запускаем с --no-sync, чтобы
#    повторный авто-sync от `uv run` не срезал эти либы.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "▶ Синхронизирую окружение (с GPU)…"
  uv sync --extra gpu
else
  echo "▶ Синхронизирую окружение…"
  uv sync
fi

# 3) Веб-интерфейс (сам откроет браузер).
echo "▶ Запускаю веб-интерфейс…"
exec uv run --no-sync python main.py ui "$@"
