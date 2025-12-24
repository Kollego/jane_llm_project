# Джейн - ИИ ассистент преподавателя урбанистики

Бот «Джейн» — это AI-ассистент для преподавателей и студентов, сочетающий урбанистику, социальные науки и активные методы обучения. У него две ключевые функции:

- **Проверка домашних заданий** — автоматически даёт обратную связь и рекомендации по улучшению.
- **Консультация по НИР** — ведёт диалог со студентами, помогая генерировать и развивать исследовательские идеи.

«Джейн» мотивирует студентов и упрощает работу преподавателей, выступая в роли умного тьютора с выдержанным стилем общения.

---

## Быстрый старт с Docker

### 1. Создайте `.env` файл

```bash
BOT_TOKEN=ваш_telegram_bot_token
YC_API_KEY=ваш_yandex_cloud_api_key
YC_FOLDER_ID=ваш_yandex_folder_id
```

### 2. Запустите через Docker Compose

```bash
# Сборка и запуск
docker-compose up -d --build

# Просмотр логов
docker-compose logs -f

# Логи только бота
docker-compose logs -f bot

# Остановка
docker-compose down
```

### 3. Убедитесь, что папка `data/` содержит:
- `qdrant_local/` — векторная база данных
- `chunks/` — parent chunks для retriever

---

## Локальная установка

```bash
# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или: venv\Scripts\activate  # Windows

# Установка зависимостей
pip install -r requirements.txt
```

Создайте файл `.env` в корне проекта:
```bash
LLAMA_CLOUD_API_KEY=your_llamaparse_key
MISTRAL_API_KEY=your_mistral_key
YC_API_KEY=your_yandex_api_key
YC_FOLDER_ID=your_folder_id
BOT_TOKEN=your_telegram_bot_token
```

### Запуск локально

```bash
# Терминал 1: Backend
export DATA_DIR=./data QDRANT_PATH=./data/qdrant_local PARENT_CHUNKS_DIR=./data/chunks
python -m src.backend.backend

# Терминал 2: Bot
export DATA_DIR=./data BOT_TOKEN=your_token BACKEND_URL=http://localhost:5001
python -m src.backend.bot
```

---

## Пайплайн обработки PDF

Полный пайплайн подготовки книг для RAG системы:

```
PDF книги → Парсинг TOC → Парсинг страниц → Чанкинг → Эмбеддинги → Qdrant
```

### Структура данных

```
data/
├── books/                      # Исходные PDF книги
├── toc/                        # JSON файлы с оглавлениями
├── cache/                      # Кэш распарсенных страниц (LlamaParse)
├── chunks/                     # Parent chunks (целые главы)
├── child_chunks_with_embeddings/  # Child chunks с эмбеддингами
└── qdrant_local/               # Векторная база данных
```

### Шаг 1: Парсинг оглавления

Скрипт `scripts/parse_toc.py` преобразует текстовое оглавление в структурированный JSON с помощью Mistral LLM.

```bash
# Скопируйте оглавление из PDF в текстовый файл, затем:
python scripts/parse_toc.py toc.txt > data/toc/book_name_toc.json
```

Формат TOC (поддерживает любой уровень вложенности):
```json
{
  "TOC": {
    "Часть I": ["Глава 1", "Глава 2"],
    "Часть II": {
      "Глава 3": ["Раздел 3.1", "Раздел 3.2"]
    }
  }
}
```

### Шаг 2: Парсинг и чанкинг книг

```python
from src.data import process_book, ChunkConfig

config = ChunkConfig(
    child_chunk_size=400,    # Размер чанков для поиска
    child_chunk_overlap=50,  # Перекрытие между чанками
    language="ru"
)

child_docs, parent_docs = process_book(
    "data/books/book.pdf",
    config=config,
    save_parsed_path="data/cache/book_parsed.json",  # Кэш парсинга
    toc_path="data/toc/book_toc.json"                # Оглавление
)
```

### Шаг 3: Сохранение чанков

```bash
# Сохраняет parent и child chunks в JSON файлы
python scripts/save_chunks.py
```

Результат:
- `data/chunks/` — parent chunks (целые главы)
- Готовые child chunks для следующего шага

### Шаг 4: Вычисление эмбеддингов

```bash
# Вычисляет эмбеддинги для child chunks через Yandex Cloud
python scripts/compute_embeddings.py
```

Результат: `data/child_chunks_with_embeddings/` — JSON файлы с векторами

### Шаг 5: Загрузка в Qdrant

```bash
# Загружает child chunks с эмбеддингами в векторную БД
python scripts/load_to_qdrant.py
```

Результат: `data/qdrant_local/` — готовая векторная база

### Batch обработка всех книг

```python
from src.data import process_books_directory

child_docs, parent_docs = process_books_directory(
    "data/books",
    config=config,
    parsed_cache_dir="data/cache",
    toc_dir="data/toc"
)
```

### Полный пайплайн одной командой

```bash
# 1. Парсинг и чанкинг
python scripts/save_chunks.py

# 2. Эмбеддинги
python scripts/compute_embeddings.py

# 3. Загрузка в Qdrant
python scripts/load_to_qdrant.py
```

---

## Parent Document Retriever

Пайплайн реализует стратегию Parent Document Retriever:

- **Parent chunks** = целые главы (для полного контекста LLM)
- **Child chunks** = мелкие части (для точного векторного поиска)

Каждый child chunk содержит `parent_id` для получения полного контекста главы.

### Метаданные чанков

```python
{
    "source": "data/books/book.pdf",
    "book_title": "Название книги",
    "page_number": 45,
    "page_range": [45, 46, 47, 48],
    "chapter": "Часть II > Глава 3 > Раздел 3.1",
    "chunk_type": "child",  # или "parent"
    "parent_id": "uuid...",
    "chunk_id": "uuid..."
}
```

---

## Скрипты

| Скрипт | Описание |
|--------|----------|
| `scripts/parse_toc.py` | Парсинг оглавления через Mistral |
| `scripts/save_chunks.py` | Сохранение parent/child chunks |
| `scripts/compute_embeddings.py` | Вычисление эмбеддингов (YC) |
| `scripts/load_to_qdrant.py` | Загрузка в векторную БД |
| `scripts/search.py` | Тестовый поиск по базе |
| `scripts/check_essay.py` | Проверка эссе (CLI) |
| `scripts/check_nir.py` | Проверка НИР (CLI) |
| `scripts/evaluate_rag.py` | Оценка качества RAG |

---

## Переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `BOT_TOKEN` | Telegram Bot Token | — |
| `YC_API_KEY` | Yandex Cloud API Key | — |
| `YC_FOLDER_ID` | Yandex Cloud Folder ID | — |
| `BACKEND_URL` | URL backend сервера | `http://localhost:5001` |
| `DATA_DIR` | Директория данных | `./data` |
| `QDRANT_PATH` | Путь к Qdrant | `./data/qdrant_local` |
| `PARENT_CHUNKS_DIR` | Путь к parent chunks | `./data/chunks` |
