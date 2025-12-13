#!/usr/bin/env python3
"""
Step 1: Process books and save chunks to data/chunks.

Saves parent and child chunks as JSON files.

Usage:
    # Single book:
    python scripts/save_chunks.py \
        --book data/books/book.pdf \
        --parsed data/cache/book.json \
        --toc data/toc/book_toc.json

    # All books from directory:
    python scripts/save_chunks.py \
        --books-dir data/books \
        --parsed-dir data/cache \
        --toc-dir data/toc
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from src.data import process_book, ChunkConfig


# Default paths
DEFAULT_CHUNKS_DIR = "./data/chunks"


def save_chunks(
    child_docs: List[Document],
    parent_docs: List[Document],
    output_dir: str,
    book_name: str,
) -> None:
    """Save chunks to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save child chunks
    child_data = []
    for doc in child_docs:
        child_data.append({
            "text": doc.page_content,
            "metadata": doc.metadata,
        })
    
    child_file = output_path / f"{book_name}_child.json"
    with open(child_file, "w", encoding="utf-8") as f:
        json.dump(child_data, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(child_data)} child chunks to {child_file}")
    
    # Save parent chunks
    parent_data = []
    for doc in parent_docs:
        parent_data.append({
            "text": doc.page_content,
            "metadata": doc.metadata,
        })
    
    parent_file = output_path / f"{book_name}_parent.json"
    with open(parent_file, "w", encoding="utf-8") as f:
        json.dump(parent_data, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(parent_data)} parent chunks to {parent_file}")


def process_single_book(
    book_path: str,
    parsed_path: str,
    toc_path: str | None,
    output_dir: str,
) -> int:
    """Process a single book and save chunks.
    
    Returns:
        Number of child chunks saved.
    """
    book_name = Path(book_path).stem
    
    print(f"\n{'='*60}")
    print(f"Processing: {book_name}")
    print(f"{'='*60}")
    
    config = ChunkConfig()
    child_docs, parent_docs = process_book(
        file_path=book_path,
        parsed_path=parsed_path,
        config=config,
        toc_path=toc_path,
    )
    
    if not child_docs:
        print("No chunks generated, skipping...")
        return 0
    
    save_chunks(child_docs, parent_docs, output_dir, book_name)
    
    return len(child_docs)


def process_directory(
    books_dir: str,
    parsed_dir: str,
    toc_dir: str | None,
    output_dir: str,
) -> int:
    """Process all books from a directory.
    
    Returns:
        Total number of child chunks saved.
    """
    books_path = Path(books_dir)
    parsed_path = Path(parsed_dir)
    toc_path = Path(toc_dir) if toc_dir else None
    
    if not parsed_path.exists():
        print(f"Error: Parsed directory not found: {parsed_dir}")
        return 0
    
    # Find all parsed JSON files
    parsed_files = list(parsed_path.glob("*.json"))
    if not parsed_files:
        print(f"No parsed JSON files found in {parsed_dir}")
        return 0
    
    print(f"Found {len(parsed_files)} parsed books")
    
    total_chunks = 0
    
    for parsed_file in parsed_files:
        stem = parsed_file.stem.replace("_parsed", "")
        
        # Find book file
        book_file = None
        for ext in [".pdf", ".PDF"]:
            candidate = books_path / f"{stem}{ext}"
            if candidate.exists():
                book_file = candidate
                break
        
        if not book_file:
            book_file = parsed_file
        
        # Find TOC file
        toc_file = None
        if toc_path and toc_path.exists():
            for pattern in [f"{stem}_toc.json", f"{stem}.json"]:
                candidate = toc_path / pattern
                if candidate.exists():
                    toc_file = candidate
                    break
        
        try:
            count = process_single_book(
                book_path=str(book_file),
                parsed_path=str(parsed_file),
                toc_path=str(toc_file) if toc_file else None,
                output_dir=output_dir,
            )
            total_chunks += count
        except Exception as e:
            print(f"✗ Error processing {stem}: {e}")
            continue
    
    return total_chunks


def main():
    parser = argparse.ArgumentParser(
        description="Process books and save chunks to JSON files"
    )
    
    # Single book mode
    parser.add_argument("--book", help="Path to single book PDF")
    parser.add_argument("--parsed", help="Path to parsed JSON for single book")
    parser.add_argument("--toc", help="Path to TOC JSON for single book")
    
    # Directory mode
    parser.add_argument("--books-dir", help="Directory with book PDFs")
    parser.add_argument("--parsed-dir", help="Directory with parsed JSONs")
    parser.add_argument("--toc-dir", help="Directory with TOC JSONs")
    
    # Output
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_CHUNKS_DIR,
        help=f"Output directory for chunks (default: {DEFAULT_CHUNKS_DIR})"
    )
    
    args = parser.parse_args()
    
    if args.book and args.parsed:
        total = process_single_book(
            book_path=args.book,
            parsed_path=args.parsed,
            toc_path=args.toc,
            output_dir=args.output_dir,
        )
    elif args.parsed_dir:
        total = process_directory(
            books_dir=args.books_dir or args.parsed_dir,
            parsed_dir=args.parsed_dir,
            toc_dir=args.toc_dir,
            output_dir=args.output_dir,
        )
    else:
        parser.print_help()
        print("\nError: Specify --book/--parsed or --parsed-dir")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Done! Total child chunks saved: {total}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

