# Fine-tuning MusicGen on MusicCaps with structured metadata

Проект для fine-tuning `facebook/musicgen-small` на датасете MusicCaps с предварительным обогащением исходных текстовых описаний через LLM.

## Что реализовано

В проекте реализован рабочий пайплайн fine-tuning MusicGen на MusicCaps с LLM-обогащением метаданных:

1. скачивание только нужных аудиофрагментов из MusicCaps;
2. преобразование исходных `caption` и `aspect_list` в структурированный JSON через LLM;
3. сборка train/valid manifest-файлов;
4. модификация AudioCraft для чтения новых полей метаданных;
5. fine-tuning MusicGen через `dora`;
6. инференс по структурированным JSON-промптам.

## Важная идея реализации

В этом проекте новые поля JSON **не** заведены как отдельные `conditioners` и **не** подаются в `fuser` по отдельности.

Вместо этого используется более простой и практичный вариант:

- новые поля сначала парсятся в расширенный `MusicInfo`;
- затем на этапе загрузки датасета они сериализуются в текст;
- этот текст склеивается с `description`;
- дальше модель получает **одну** текстовую строку через стандартный текстовый conditioner `description`.

То есть структурированные поля реально участвуют в обучении, но не как отдельные независимые каналы кондиционирования, а как часть общего текстового описания.

Это сделано намеренно, чтобы минимально вмешиваться в архитектуру AudioCraft и сохранить рабочий fine-tuning pipeline.

---

## Структура репозитория

```text
.
├── RES/                            # папка с аудиофайлами, полученными при генерации
├── audiocraft/                     # submodule с модифицированным AudioCraft
├── configs/                        # конфиги для обучения и генерации
│   ├── musiccaps_train.yaml
│   └── text2music.yaml
├── data_prep/                      # скрипты для получения данных и создания манифестов
│   ├── 01_download_audio.py
│   ├── 02_llm_enrichment.py
│   └── 03_create_manifests.py
├── utils/                          # вспомогательные утилиты
│   ├── data_utils.py
│   ├── dir_utils.py
│   └── requests_utils.py
├── finetune.py                     # код до-обучения
├── inference.py                    # код инференса
└── manifests/                      # папка с полученными манифестами
```

---

## Требования

Минимально нужны:

* Python 3.10+
* `ffmpeg`
* `yt-dlp`
* PyTorch + CUDA
* `soundfile`
* `datasets`
* `openai`
* `python-dotenv`
* `tqdm`

Также должен быть инициализирован submodule `audiocraft`.

## Подготовка окружения

### 1. Клонирование проекта

```bash
git clone https://github.com/Avvonna/DL_musicgen_ft_project.git
cd DL_musicgen_ft_project
git submodule update --init --recursive
```

### 2. Установка зависимостей

Зависимости ставятся как для основного проекта, так и частично для `audiocraft`.  
Для запуска использовался только тот поднабор зависимостей audiocraft, который требуется для fine-tuning и инференса в данной конфигурации. Полная установка зависимостей submodule под Windows оказалась нестабильной из-за конфликта версий библиотек.

### 3. Переменные окружения

Для этапа LLM enrichment нужен `.env` файл с ключом OpenRouter:

```env
OPENROUTER_API_KEY=...
```

---

## Пайплайн данных

### Шаг 1. Скачивание аудио из MusicCaps

Скрипт [`data_prep/01_download_audio.py`](./data_prep/01_download_audio.py):

* загружает `google/MusicCaps`;
* для каждого элемента берет `ytid`, `start_s`, `end_s`;
* через `yt-dlp` получает прямую ссылку на аудиопоток;
* через `ffmpeg` сохраняет только нужный фрагмент;
* результат сохраняется как `.wav` с параметрами:
  * sample rate: `32000`
  * channels: `1`
  * codec: `pcm_s16le`

Дополнительно рядом сохраняются исходные метаданные MusicCaps:

* `ytid`
* `caption`
* `aspect_list`

Запуск:

```bash
python data_prep/01_download_audio.py
```

Результат:

* `dataset/raw/*.wav`
* `dataset/meta/*.json`

---

### Шаг 2. Обогащение метаданных через LLM

Скрипт [`data_prep/02_llm_enrichment.py`](./data_prep/02_llm_enrichment.py) берет:

* исходный `caption`
* `aspect_list`

и преобразует их в JSON-схему:

```json
{
  "description": "string",
  "general_mood": "string",
  "genre_tags": ["string"],
  "lead_instrument": "string",
  "accompaniment": "string",
  "tempo_and_rhythm": "string",
  "vocal_presence": "string",
  "production_quality": "string"
}
```

Используемая модель:

* OpenRouter: `qwen/qwen3.5-9b`
* либо локальная модель через Ollama  
    *Это оказалось медленнее, чем использовать API, так что использовался первый способ*

Результат сохраняется в отдельные `.json` файлы.

Запуск:

```bash
python data_prep/02_llm_enrichment.py
```

Результат по умолчанию:

* `dataset/qwen3.5-9b/*.json`

---

### Шаг 3. Создание manifest-файлов

Скрипт [`data_prep/03_create_manifests.py`](./data_prep/03_create_manifests.py):

* проходит по enriched JSON;
* проверяет наличие `.wav`;
* читает длительность через `soundfile`;
* отбрасывает слишком короткие записи;
* случайно делит данные на train/valid;
* пишет `jsonl.gz` manifest-файлы.

Запуск:

```bash
python data_prep/03_create_manifests.py
```

Результат:

* `manifests/train.jsonl.gz`
* `manifests/valid.jsonl.gz`

---

## Как новые поля проходят через модель

### 1. Расширение `MusicInfo`

В [`audiocraft/data/music_dataset.py`](./audiocraft/audiocraft/data/music_dataset.py) датакласс `MusicInfo` расширен новыми полями:

* `general_mood`
* `genre_tags`
* `lead_instrument`
* `accompaniment`
* `tempo_and_rhythm`
* `vocal_presence`
* `production_quality`

### 2. Чтение JSON-метаданных

При загрузке примера эти поля читаются из enriched JSON и попадают в `MusicInfo`.

### 3. Преобразование в единый текст

Далее используется функция `augment_music_info_description(...)`, которая при заданной вероятности:

* берет исходное `description`;
* добавляет к нему сериализованные новые поля в формате:

```text
general_mood: ...
genre_tags: ...
lead_instrument: ...
...
```

Пример итоговой строки:

```text
A relaxing lo-fi hip-hop instrumental with a muffled electric piano. general_mood: relaxing, nostalgic, chill. genre_tags: lo-fi hip hop, chillhop, instrumental. lead_instrument: muffled electric piano. accompaniment: dusty vinyl crackle, deep sub-bass, soft boom-bap drum loop.
```

### 4. Почему нет отдельных conditioners

В этом проекте использован только один текстовый conditioner:

* `description`

То есть новые поля не обрабатываются отдельными T5-ветками и не имеют собственного `fuser`-маршрута. Они участвуют в обучении как часть общего текстового описания.

Плюсы такого решения:

* минимальные изменения в AudioCraft
* простой и стабильный запуск fine-tuning
* не нужно перестраивать архитектуру conditioner/fuser

Минусы:

* слабее контроль по отдельным полям
* модель не видит поля как независимые структурные источники

---

## Конфиги

### [`configs/musiccaps_train.yaml`](./configs/musiccaps_train.yaml)

Задает источники train/valid/evaluate/generate manifest-файлов.

**Важно:** в текущем виде в конфиге используются абсолютные пути под Windows.  
Перед запуском на другой машине их нужно отредактировать.

### [`configs/text2music.yaml`](./configs/text2music.yaml)

Используется один текстовый conditioner:

```yaml
conditioners:
  description:
    model: t5
```

Во `fuser` в cross-attention подается только:

```yaml
cross: [description]
```

Параметры аугментации текста:

* `merge_text_p`
* `drop_desc_p`
* `drop_other_p`

Текущая логика:
* на train часть примеров получает объединенное описание
* на valid/evaluate/generate объединенное описание используется всегда

Это позволяет:
* не потерять полностью исходный режим обучения по обычному описанию
* при этом частично научить модель воспринимать структурированные поля

---

## Обучение

Используется скрипт [`finetune.py`](./finetune.py)

Основной запуск:

```bash
python finetune.py
```

Что делает скрипт:

1. копирует локальные конфиги в `audiocraft/config/...`
2. запускает `dora run` внутри submodule `audiocraft`
3. стартует fine-tuning `musicgen-small`

---

## Инференс

Скрипт [`inference.py`](./inference.py) ожидает папку с JSON-промптами.

Каждый JSON должен иметь ту же структуру, что и enriched metadata:

```json
{
  "description": "...",
  "general_mood": "...",
  "genre_tags": ["..."],
  "lead_instrument": "...",
  "accompaniment": "...",
  "tempo_and_rhythm": "...",
  "vocal_presence": "...",
  "production_quality": "..."
}
```

Во время инференса JSON снова преобразуется в одну текстовую строку по той же логике:

* сначала `description`,
* затем остальные поля как `key: value`.

### Инференс базовой модели

```bash
python inference.py --json_dir test_prompts --duration 10 --trained_duration 10
```

### Инференс fine-tuned модели

```bash
python inference.py \
  --json_dir test_prompts \
  --finetuned \
  --checkpoint <path_to_checkpoint.pt> \
  --duration 10 \
  --trained_duration 10
```

Результат:

* `.wav` файлы сохраняются рядом с JSON-промптами.

---

## Пример полного запуска

### 1. Скачать аудио

```bash
python data_prep/01_download_audio.py
```

### 2. Обогатить метаданные через LLM

```bash
python data_prep/02_llm_enrichment.py
```

### 3. Создать manifests

```bash
python data_prep/03_create_manifests.py
```

### 4A. Запустить обучение

```bash
python finetune.py
```

### 4B. Либо скачать веса до-обученной модели

```bash
python scripts/download_weights.py --file_id "1-_wpl5My9pjTzh_ENPv7lWOasD7BvVW4"
```

### 5. Сгенерировать тестовые примеры

```bash
python inference.py \
  --json_dir test_prompts \
  --finetuned \
  --checkpoint <path_to_checkpoint.pt> \
  --duration 10 \
  --trained_duration 10
```

---
## Результаты

В проведенных экспериментах базовая модель `musicgen-small` уже демонстрировала приемлемое следование промптам без дополнительного fine-tuning.

Было протестировано несколько конфигураций обучения с варьированием числа эпох и learning rate. Несмотря на улучшение training-метрик, субъективное качество итоговой генерации после fine-tuning в ряде запусков не улучшалось, а в отдельных случаях снижалось.

Возможное объяснение состоит в том, что в текущей конфигурации новые структурированные поля не подаются как независимые каналы кондиционирования, а сводятся к одному текстовому полю `description`. Поэтому fine-tuning в этой постановке не всегда дает заметное преимущество относительно базовой модели.

Результаты генерации находятся в папке [`RES`](./RES/):
- `BASE_prompt_N.wav` - генерация базовой модели
- `prompt_N.wav` - генерация fine-tuned модели

Веса доступны по [**ссылке**](https://drive.google.com/file/d/1-_wpl5My9pjTzh_ENPv7lWOasD7BvVW4/view?usp=sharing). Для скачивания можно использовать скрипт [`download_weights.py`](./scripts/download_weights.py)

Отчет можно посмотреть тут: [**WANDB REPORT**](https://wandb.ai/vaalkaev-hse/musicgen-hw4/reports/Untitled-Report--VmlldzoxNjIyNDY3MQ)

---

## Ограничения текущей реализации

1. Новые JSON-поля не заведены как отдельные conditioners.
2. Структурированное описание сводится к одному текстовому каналу `description`.
3. Конфиги используют абсолютные пути под локальную Windows-машину.
4. Часть параметров в аугментации текста наследует логику оригинального AudioCraft, где `drop_other_p` фактически используется как вероятность оставить поле (что контринтуитивно).
