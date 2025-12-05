# Джейн - ИИ ассистент преподавателя урбанистики

Бот «Джейн» — это AI-ассистент для преподавателей и студентов, сочетающий урбанистику, социальные науки и активные методы обучения. У него две ключевые функции:

- **Проверка домашних заданий** — автоматически даёт обратную связь и рекомендации по улучшению.
- **Консультация по НИР** — ведёт диалог со студентами, помогая генерировать и развивать исследовательские идеи.

«Джейн» мотивирует студентов и упрощает работу преподавателей, выступая в роли умного тьютора с выдержанным стилем общения.

## Установка

```bash
pip install -r requirements.txt
```

Создайте файл `.env` в корне проекта:
```
LLAMA_CLOUD_API_KEY=your_llamaparse_key
MISTRAL_API_KEY=your_mistral_key
```

## RAG Pipeline для книг по урбанистике

### Структура данных

```
data/
├── books/           # PDF книги
├── toc/             # JSON файлы с оглавлениями
└── cache/           # Кэш распарсенных страниц
```

### 1. Парсинг оглавления

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

### 2. Парсинг и чанкинг книг

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

### 3. Parent Document Retriever

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
