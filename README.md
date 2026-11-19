# ЛР1: федеративне навчання на ESP32-S3 (Seeed XIAO ESP32S3)

Інфраструктура федеративного навчання: плата ESP32-S3 навчає невелику нейромережу 64 -> 16 -> 10 на локальних даних, а Flower керує федерацією з чотирьох клієнтів, де плата є клієнтом 0, решта віртуальні.

# Вимоги

- Ubuntu 24.04, Docker (користувач у групах docker і dialout), Python 3.12 з пакетом python3-venv, make, git.
- Плата Seeed XIAO ESP32S3 і кабель USB-C для передачі даних, порт з'являється як /dev/ttyACM0.
- Версії закріплено: ESP-IDF v5.5.1 (docker/Dockerfile), Flower 1.43.0, numpy 2.2.6, pyserial 3.5, esptool 4.8.1 (host/requirements.txt, host/pyproject.toml).

# Швидкий старт

Усі команди виконуються з кореня репозиторію. Параметр PORT за замовчуванням /dev/ttyACM0.

```
make venv-core                      # віртуальне середовище для роботи з платою
make selftest                       # перевірка наданого коду: CRC f8692cac і c40a1c43
make sim-test                       # перевірка протоколу без плати
make image build                    # образ Docker і збірка прошивки
make flash PORT=/dev/ttyACM0        # прошивання програми
make flash-data PORT=/dev/ttyACM0 DATA=data/data_0.bin   # дані у розділ fldata
make monitor PORT=/dev/ttyACM0      # консоль плати, вихід Ctrl+]
make measure PORT=/dev/ttyACM0      # вимірювання для таблиці звіту
make one-round PORT=/dev/ttyACM0    # один раунд з комп'ютера
make venv                           # додатково Flower
make fed-virtual RUN=flwr-virtual   # федерація без плати
make fed-board PORT=/dev/ttyACM0 RUN=flwr-board   # федерація з платою
make check RUN=flwr-board           # шість перевірок федеративності
make plot RUN=flwr-board            # графік точності
make data-stats                     # статистика розбиття даних
```

# Цілі Makefile

| Ціль | Що робить | Потрібна плата |
|-|-|-|
| selftest, sim-test | перевірка наданого коду й протоколу на хості | ні |
| image, build | образ Docker з ESP-IDF і компіляція прошивки | ні |
| flash, flash-data | запис програми та даних у плату з хоста | так |
| monitor | консоль плати (діагностика: help, info, heap, restart) | так |
| measure | час епохи й оцінки, дані для таблиці вимірювань | так |
| one-round | INFO, SETW, FIT, GETW, EVAL з комп'ютера | так |
| fed-virtual | Flower, усі клієнти віртуальні | ні |
| fed-board | Flower, клієнт 0 це плата; перед прогоном RESET, після прогону читання лічильників | так |
| check | шість перевірок федеративності за журналами | ні |
| plot, data-stats | графік точності, розбиття даних | ні |
| ref-virtual, ref-sim | еталонний FedAvg без Flower, для звірки | ні |

# Структура репозиторію

docker/Dockerfile        закріплений образ ESP-IDF
firmware/               проєкт ESP-IDF (partitions.csv, sdkconfig.defaults)
  main/                 fl_rng, fl_model, fl_train, fl_crc, fl_b64,
                        fl_store, fl_proto, esp_storage, main
  host_test/            хостовий симулятор плати (те саме ядро протоколу)
host/                   pyproject.toml, requirements, config/
  fl_lab/               клієнт протоколу, Flower-застосунки, перевірка, тести
scripts/                вимірювання, графік, порівняння прогонів, статистика даних
data/                   data_0..3.bin, test.bin
results/<прогін>/       rounds.jsonl, clients.jsonl, transcript.log, board_stats.json, config.json
docs/                   interfaces.md, lab1-report.docx
Makefile, README.md

# Відомі обмеження

- Агрегація DeterministicFedAvg використовує внутрішній метод Flower і перевірена лише з Flower 1.43.0.
- Відкриття послідовного порту виконується з DTR=RTS=True, інакше плата скидається при кожному відкритті.
- Baseline-и (FL-09) і кліпінг (FL-11) у цій роботі не виконувались.
