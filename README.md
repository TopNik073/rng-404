
---

# Генератор Случайных Чисел (ГСЧ) на основе Аудиопотоков

## О проекте

Сервис представляет собой API для генерации случайных чисел на основе аудиопотоков из источников звука по всему миру.

## Ключевые возможности

* **Генерация случайных чисел** на основе аудиопотоков от Locus Sonus
* **Загрузка пользовательских аудиофайлов** в качестве источника энтропии
* **Гибкая настройка параметров** генерации:

  * Диапазон значений
  * Количество цифр
  * Основание системы счисления (2–36)
  * Формат вывода (JSON/TXT)
  * Количество генерируемых чисел
* **Генерация бинарных последовательностей** для тестирования (указанное количество 0 и 1)
* **Полная батарея статистических тестов NIST** (15 тестов) для верификации качества ГСЧ
* **Подробная аналитическая отчетность** по результатам тестирования

## Технический стек

* **Backend**: FastAPI (Python 3.13)
* **Кэширование**: Redis
* **Аудиоисточники**: Locus Sonus API & ffmpeg
* **Контейнеризация**: Docker & Docker Compose

## Быстрый запуск

### Локальный запуск с uv

```bash
# Создание виртуальной среды
uv venv .venv

# Активация (в зависимости от вашей системы)
source .venv/bin/activate # MacOs & Linux
.venv/Scripts/activate # Windows

# Установка зависимостей
uv sync

# Запуск сервиса
uv run src/main.py
```

### Запуск через Docker Compose

```bash
# Сборка и запуск всех сервисов
docker compose -f docker-compose.local.yml up --build
```

При установленном just доступно:
```bash
just build-local
```

Сервис будет доступен по адресу: `http://localhost:8000`

## Документация API

После запуска доступна интерактивная документация:

* **Swagger UI**: `http://localhost:8000/docs`
* **ReDoc**: `http://localhost:8000/redoc`

## Полный набор тестов NIST

1. **Frequency (Monobit) Test** — Тест частоты
2. **Block Frequency Test** — Тест блочной частоты
3. **Runs Test** — Тест серий
4. **Longest Run of Ones Test** — Тест самой длинной серии единиц
5. **Binary Matrix Rank Test** — Тест ранга бинарной матрицы
6. **Discrete Fourier Transform Test** — Тест дискретного преобразования Фурье
7. **Non-overlapping Template Matching Test** — Тест неперекрывающихся шаблонов
8. **Overlapping Template Matching Test** — Тест перекрывающихся шаблонов
9. **Maurer’s Universal Statistical Test** — Универсальный статистический тест Маурера
10. **Linear Complexity Test** — Тест линейной сложности
11. **Serial Test** — Серийный тест
12. **Approximate Entropy Test** — Тест приближенной энтропии
13. **Cumulative Sums (Cusum) Test** — Тест кумулятивных сумм
14. **Random Excursions Test** — Тест случайных блужданий
15. **Random Excursions Variant Test** — Вариант теста случайных блужданий

## Конфигурация

Основные настройки могут быть изменены через переменные окружения:

* `DEBUG` - режим работы логгера
* `APP_NAME` - название сервиса
* `APP_HOST` - хост сервиса
* `APP_PORT` - порт сервиса
* `MAX_AUDIO_DURATION` - максимальная длинна ауди дорожки, которую может загрузить пользователь
* `REDIS_HOST` - хост редиса
* `REDIS_PORT` - порт редиса
* `REDIS_PASS` - пароль редиса
* `REDIS_DB` - используемая база данных редиса


## Структура проекта

```
.
├── README.md
├── justfile
├── pyproject.toml
├── uv.lock
├── logs/
│
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.local.yml
│   ├── docker-compose.dev.yml
│   └── docker-compose.prod.yml
│
└── src/
    ├── main.py
    ├── core/
    │   ├── config.py
    │   ├── logger.py
    │   └── __init__.py
    │
    ├── integrations/
    │   ├── locusonus/
    │   │   ├── api.py
    │   │   ├── cache.py
    │   │   ├── client.py
    │   │   ├── models.py
    │   │   └── __init__.py
    │   │
    │   ├── redis/
    │   │   ├── client.py
    │   │   └── __init__.py
    │   │
    │   └── __init__.py
    │
    ├── presentation/
    │   ├── api/
    │   │   ├── v1/
    │   │   │   ├── nist/
    │   │   │   │   ├── dep.py
    │   │   │   │   ├── models.py
    │   │   │   │   ├── router.py
    │   │   │   │   └── __init__.py
    │   │   │   │
    │   │   │   ├── rng/
    │   │   │   │   ├── dep.py
    │   │   │   │   ├── models.py
    │   │   │   │   ├── router.py
    │   │   │   │   └── __init__.py
    │   │   │   │
    │   │   │   └── __init__.py
    │   │   │
    │   │   └── __init__.py
    │   │
    │   ├── middlewares/
    │   │   ├── logging.py
    │   │   └── __init__.py
    │   │
    │   └── __init__.py
    │
    ├── services/
    │   ├── rng/
    │   │   ├── rng.py
    │   │   ├── service.py
    │   │   └── __init__.py
    │   │
    │   ├── nist/
    │   │   ├── service.py
    │   │   ├── tests/
    │   │   │   ├── approximate_entropy_test.py
    │   │   │   ├── binary_matrix_rank_test.py
    │   │   │   ├── block_frequency_test.py
    │   │   │   ├── cumulative_sums_test.py
    │   │   │   ├── discrete_fourier_transform_test.py
    │   │   │   ├── frequency_test.py
    │   │   │   ├── linear_complexity_test.py
    │   │   │   ├── longest_run_ones_test.py
    │   │   │   ├── non_overlapping_template_test.py
    │   │   │   ├── overlapping_template_test.py
    │   │   │   ├── random_excursions_test.py
    │   │   │   ├── random_excursions_variant_test.py
    │   │   │   ├── runs_test.py
    │   │   │   ├── serial_test.py
    │   │   │   ├── universal_test.py
    │   │   │   └── templates/
    │   │   │       ├── dataInfo
    │   │   │       ├── template2 … template21.Z
    │   │   │       └── другие шаблоны NIST
    │   │   │
    │   │   └── __init__.py
    │   │
    │   └── __init__.py
    │
    └── __init__.py

```
---
