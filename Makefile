# CallScribe — удобные команды запуска.
# Использование: make <цель>. Например: make ui  |  make transcribe FILE=call.mp4

PY   ?= uv run python
PORT ?= 5000
HOST ?= 127.0.0.1

.DEFAULT_GOAL := help

.PHONY: help install install-gpu setup start ui ui-lan run record last transcribe test dist clean

help:  ## Показать список команд
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Установить зависимости (CPU)
	uv sync

install-gpu:  ## Установить зависимости + CUDA-библиотеки (NVIDIA GPU)
	uv sync --extra gpu

setup:  ## Мастер настройки аудио-устройств → config.json
	$(PY) main.py setup

start:  ## ⭐ Запуск всего одной командой: Ollama + веб-интерфейс
	@./start.sh --host $(HOST) --port $(PORT)

ui:  ## Запустить только веб-интерфейс (HOST/PORT, по умолчанию 127.0.0.1:5000)
	$(PY) main.py ui --host $(HOST) --port $(PORT)

ui-lan:  ## UI, доступный по сети (для записи с ноутбука → GPU-десктоп)
	@ip=$$(hostname -I 2>/dev/null | awk '{print $$1}'); \
	echo "✓ Открой с ноутбука: http://$$ip:$(PORT)"; \
	echo "⚠ Без пароля — только доверенная домашняя сеть"; \
	$(PY) main.py ui --host 0.0.0.0 --port $(PORT)

run:  ## Запись + транскрипция (CLI)
	$(PY) main.py run

record:  ## Только запись (Enter — стоп)
	$(PY) main.py record

last:  ## Обработать последнюю запись
	$(PY) main.py last

transcribe:  ## Транскрибировать файл: make transcribe FILE=call.mp4
	@test -n "$(FILE)" || { echo "Укажи файл: make transcribe FILE=путь"; exit 1; }
	$(PY) main.py transcribe "$(FILE)"

test:  ## Прогнать тесты (pytest)
	uv run pytest

dist:  ## Собрать zip для передачи (исходники без моделей/записей/config.json)
	@git archive --format=zip --prefix=callscribe/ -o callscribe.zip HEAD
	@echo "✓ callscribe.zip готов ($$(du -h callscribe.zip | cut -f1)) — без models/, recordings/, config.json"

clean:  ## Удалить кэш Python
	find . -type d -name __pycache__ -prune -exec rm -rf {} + ; \
	find . -type f -name '*.pyc' -delete
