# CallScribe — удобные команды запуска.
# Использование: make <цель>. Например: make ui  |  make transcribe FILE=call.mp4

PY   ?= uv run python
PORT ?= 5000

.DEFAULT_GOAL := help

.PHONY: help install install-gpu setup ui run record last transcribe clean

help:  ## Показать список команд
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Установить зависимости (CPU)
	uv sync

install-gpu:  ## Установить зависимости + CUDA-библиотеки (NVIDIA GPU)
	uv sync --extra gpu

setup:  ## Мастер настройки аудио-устройств → config.json
	$(PY) main.py setup

ui:  ## Запустить веб-интерфейс (PORT=5000 по умолчанию)
	$(PY) main.py ui --port $(PORT)

run:  ## Запись + транскрипция (CLI)
	$(PY) main.py run

record:  ## Только запись (Enter — стоп)
	$(PY) main.py record

last:  ## Обработать последнюю запись
	$(PY) main.py last

transcribe:  ## Транскрибировать файл: make transcribe FILE=call.mp4
	@test -n "$(FILE)" || { echo "Укажи файл: make transcribe FILE=путь"; exit 1; }
	$(PY) main.py transcribe "$(FILE)"

clean:  ## Удалить кэш Python
	find . -type d -name __pycache__ -prune -exec rm -rf {} + ; \
	find . -type f -name '*.pyc' -delete
