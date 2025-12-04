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

Your task is to analyze the raw TOC text and convert it into a hierarchical JSON structure.

Rules:
1. Identify parts, chapters, sections, and subsections from the text
2. Roman numerals (I, II, III) or words like "Часть", "Part" usually indicate major parts
3. Numbers or "Глава", "Chapter" usually indicate chapters
4. Nested items should be grouped under their parent
5. Page numbers should be REMOVED from titles
6. Keep the original language of titles (Russian, English, etc.)
7. Preserve the hierarchical structure based on indentation and numbering

Output format should match this structure:
{
  "TOC": {
    "Part/Section Name": ["Chapter 1", "Chapter 2"],
    "Another Part": {
        "Chapter with subsections": ["Subsection 1", "Subsection 2"]
    }
  }
}
"""


def parse_toc_with_mistral(toc_text: str) -> dict:
    """Parse raw TOC text into structured format using Mistral."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable is not set")
    
    client = Mistral(api_key=api_key)
    print("Sending request to Mistral...")
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Parse this TOC into JSON:\n\n{toc_text}"},
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
