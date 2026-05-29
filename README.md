# CallScribe

Запись звонков (микрофон + системный звук) и их транскрипция в текст через
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). Кросс-платформенно:
Windows 10/11 и Linux (Ubuntu 22.04+/Debian 12+).

## Возможности

- Одновременный захват микрофона и системного звука (loopback), микширование в один WAV
- Транскрипция моделью `large-v3` с мультипроцессингом на CPU (или GPU, если есть CUDA)
- Стенограмма в Markdown + JSON
- Веб-интерфейс (FastAPI + WebSocket) с живым логом
- Только локальные файлы, без баз данных и облака

## Установка

Требуется **Python 3.12+**.

```bash
# с uv (рекомендуется)
uv sync

# или через pip
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Системная зависимость: PortAudio (нужна только для записи)

`sounddevice` требует библиотеку PortAudio. Транскрипция и веб-интерфейс
работают и без неё — она нужна лишь чтобы **записывать** звук.

```bash
# Ubuntu/Debian
sudo apt install libportaudio2

# Windows — PortAudio идёт в составе колеса sounddevice, доп. установка не нужна
```

### Системный звук (loopback)

- **Windows:** установить [VB-Cable](https://vb-audio.com/Cable/) и выбрать «CABLE Output»,
  либо включить «Стерео микшер» в записывающих устройствах.
- **Linux (PulseAudio/PipeWire):** системный звук — это `*.monitor` текущего sink'а.
  Поставь `pulseaudio-utils` (`sudo apt install pulseaudio-utils`) — тогда `setup`
  сам найдёт monitor-источники через `pactl` и предложит нужный (монитор default
  sink — первым). Запись с monitor идёт через устройство `pulse` + `PULSE_SOURCE`,
  поэтому индекс PortAudio для этого не нужен.

## Настройка

```bash
python main.py setup
```

Мастер покажет аудио-устройства и сохранит их ID в `config.json`
(этот файл не коммитится — на каждой машине свои устройства).

## Использование

```bash
python main.py ui                 # веб-интерфейс на http://127.0.0.1:5000
python main.py run                # запись + транскрипция (CLI)
python main.py record             # только запись (Enter — стоп)
python main.py transcribe a.wav   # транскрибировать готовый WAV
python main.py last               # обработать последнюю запись
```

> **CPU:** ориентируйся на ~x3 от длительности звонка (час созвона ≈ 3 часа обработки).
> **GPU:** на уровне GTX 1080 — ~20 минут на часовой звонок.

Модель `large-v3` (~3 GB) скачивается автоматически при первом запуске в `models/`.

## Структура

```
main.py          точка входа, CLI-роутер
recorder.py      захват аудио (sounddevice)
transcriber.py   STT + мультипроцессинг (faster-whisper)
ui.py            FastAPI + WebSocket
setup.py         мастер настройки устройств
config.py        работа с config.json и путями
templates/
  index.html     весь веб-интерфейс одним файлом
```
