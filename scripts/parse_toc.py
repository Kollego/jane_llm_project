#!/usr/bin/env python3
"""
Script to parse raw table of contents text into structured JSON using Mistral LLM.

Usage:
    python scripts/parse_toc.py toc.txt

Environment:
    MISTRAL_API_KEY: Your Mistral API key
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from mistralai import Mistral

# Load .env from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


SYSTEM_PROMPT = """You are a helpful assistant that parses book table of contents into structured JSON format.

Your task is to analyze the raw TOC text and convert it into a hierarchical JSON structure with book metadata.

Rules:
1. Extract book metadata: author, title, and year (if available)
2. Identify parts, chapters, sections, and subsections from the text
3. Roman numerals (I, II, III) or words like "Часть", "Part" usually indicate major parts
4. Numbers or "Глава", "Chapter" usually indicate chapters
5. Nested items should be grouped under their parent using recursive "children" arrays
6. Extract page numbers from the text and store them in "page" field (as integer)
7. Page numbers should be REMOVED from titles (stored separately in "page" field)
8. Keep the original language of titles (Russian, English, etc.)
9. Preserve the hierarchical structure based on indentation and numbering
10. Each TOC item must have "title" (string), "page" (integer or null), and "children" (array) fields
11. The structure supports unlimited nesting depth
12. If page number is not available for an item, set "page" to null

Output format:
{
  "meta": {
    "author": "Author Name",
    "title": "Book Title"
  },
  "toc": [
    {
      "title": "Section Title",
      "page": 10,
      "children": [
        {
          "title": "Subsection Title",
          "page": 15,
          "children": []
        }
      ]
    }
  ]
}

Example:

Raw TOC from 

# РИЧАРД ФЛОРИДА

## КТО ТВОЙ ГОРОД?

### Креативная экономика и выбор места жительства

# ОГЛАВЛЕНИЕ

9  1. ВОПРОС «ГДЕ?»

## ЧАСТЬ I. МЕСТО ИМЕЕТ ЗНАЧЕНИЕ
24  2. МИР ПИКОВ
46  3. ПОДЪЕМ МЕГАРЕГИОНА
66  4. СИЛА КЛАСТЕРИЗАЦИИ

## ЧАСТЬ II. БОГАТСТВО МЕСТА
82  5. МОБИЛЬНЫЕ И УКОРЕНЕННЫЕ
93  6. ГДЕ НАХОДЯТСЯ МОЗГИ
102  7. ПЕРЕМЕЩЕНИЕ ПРОФЕССИЙ
127  8. ГОРОДА СУПЕРЗВЕЗДЫ

## ЧАСТЬ III. ГЕОГРАФИЯ СЧАСТЬЯ
147  9. ТАМ, ГДЕ СЧАСТЬЕ
161  10. ЗА ПРЕДЕЛАМИ ГОРОДА МАСЛОУ
185  11. ГОРОД ТОЖЕ ЛИЧНОСТЬ

## ЧАСТЬ IV. ГДЕ МЫ ЖИВЕМ СЕЙЧАС
213  12. ТРИ БОЛЬШИХ ПЕРЕЕЗДА
218  13. МОЛОДЫЕ И БЕСПОКОЙНЫЕ
246  14. ЖЕНАТЫ… С ДЕТЬМИ
270  15. КОГДА ДЕТИ УЕХАЛИ
284  16. НАЙДИТЕ СЕБЕ МЕСТО

303     БЛАГОДАРНОСТИ
307     ПРИЛОЖЕНИЯ
344     ПРИМЕЧАНИЯ
364     УКАЗАТЕЛЬ ИМЕН


Result:
{
  "meta": {
    "author": "Ричард Флорида",
    "title": "Кто твой город? Креативная экономика и выбор места жительства"
  },
  "toc": [
    {
      "title": "1. ВОПРОС «ГДЕ?»",
      "page": 9,
      "children": []
    },
    {
      "title": "ЧАСТЬ I. МЕСТО ИМЕЕТ ЗНАЧЕНИЕ",
      "page": null,
      "children": [
        {"title": "2. МИР ПИКОВ", "page": 24, "children": []},
        {"title": "3. ПОДЪЕМ МЕГАРЕГИОНА", "page": 46, "children": []},
        {"title": "4. СИЛА КЛАСТЕРИЗАЦИИ", "page": 66, "children": []}
      ]
    },
    {
      "title": "ЧАСТЬ II. БОГАТСТВО МЕСТА",
      "page": null,
      "children": [
        {"title": "5. МОБИЛЬНЫЕ И УКОРЕНЕННЫЕ", "page": 82, "children": []},
        {"title": "6. ГДЕ НАХОДЯТСЯ МОЗГИ", "page": 93, "children": []},
        {"title": "7. ПЕРЕМЕЩЕНИЕ ПРОФЕССИЙ", "page": 102, "children": []},
        {"title": "8. ГОРОДА СУПЕРЗВЕЗДЫ", "page": 127, "children": []}
      ]
    },
    {
      "title": "ЧАСТЬ III. ГЕОГРАФИЯ СЧАСТЬЯ",
      "page": null,
      "children": [
        {"title": "9. ТАМ, ГДЕ СЧАСТЬЕ", "page": 147, "children": []},
        {"title": "10. ЗА ПРЕДЕЛАМИ ГОРОДА МАСЛОУ", "page": 161, "children": []},
        {"title": "11. ГОРОД ТОЖЕ ЛИЧНОСТЬ", "page": 185, "children": []}
      ]
    },
    {
      "title": "ЧАСТЬ IV. ГДЕ МЫ ЖИВЕМ СЕЙЧАС",
      "page": null,
      "children": [
        {"title": "12. ТРИ БОЛЬШИХ ПЕРЕЕЗДА", "page": 213, "children": []},
        {"title": "13. МОЛОДЫЕ И БЕСПОКОЙНЫЕ", "page": 218, "children": []},
        {"title": "14. ЖЕНАТЫ... С ДЕТЬМИ", "page": 246, "children": []},
        {"title": "15. КОГДА ДЕТИ УЕХАЛИ", "page": 270, "children": []},
        {"title": "16. НАЙДИТЕ СЕБЕ МЕСТО", "page": 284, "children": []}
      ]
    },
    {"title": "БЛАГОДАРНОСТИ", "page": 303, "children": []},
    {"title": "ПРИЛОЖЕНИЯ", "page": 307, "children": []},
    {"title": "ПРИМЕЧАНИЯ", "page": 344, "children": []},
    {"title": "УКАЗАТЕЛЬ ИМЕН", "page": 364, "children": []},
  ]
}
"""


def parse_toc_with_mistral(toc_text: str) -> dict:
    """Parse raw TOC text into structured format using Mistral.
    
    Args:
        toc_text: Raw table of contents text
    
    Returns:
        Structured TOC dict with meta and toc fields
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable is not set")
    
    client = Mistral(api_key=api_key, timeout_ms=120000)  # 2 min timeout
    
    user_message = f"Parse this TOC into JSON with metadata (author, title) and hierarchical structure.\n\nTOC:\n{toc_text}"
    
    print("Sending request to Mistral...")
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    
    return json.loads(response.choices[0].message.content)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <toc_file.txt>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    toc_text = input_path.read_text(encoding="utf-8")
    result = parse_toc_with_mistral(toc_text)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
