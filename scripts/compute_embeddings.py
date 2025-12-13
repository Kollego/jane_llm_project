#!/usr/bin/env python3
"""
Step 2: Compute embeddings for child chunks using Yandex Cloud.

Reads child chunks from data/chunks and saves with embeddings to 
data/child_chunks_with_embeddings.

Each book is processed sequentially. Results are saved immediately 
after each book completes.

Environment variables required:
    YC_API_KEY or YC_IAM_TOKEN: Yandex Cloud authentication
    YC_FOLDER_ID: Yandex Cloud folder ID

Usage:
    python scripts/compute_embeddings.py
    python scripts/compute_embeddings.py --input-dir data/chunks --output-dir data/embeddings
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedder import YCEmbedder


# Default paths
DEFAULT_INPUT_DIR = "./data/chunks"
DEFAULT_OUTPUT_DIR = "./data/child_chunks_with_embeddings"


def process_single_book(
    book_file: Path,
    output_dir: str,
    embedder: YCEmbedder,
) -> Tuple[str, int, bool]:
    """
    Process a single book file: load chunks, compute embeddings, save.
    
    Args:
        book_file: Path to the book's child chunks JSON file
        output_dir: Directory to save results
        embedder: YCEmbedder instance
        
    Returns:
        Tuple of (book_name, num_chunks, success)
    """
    book_name = book_file.stem.replace("_child", "")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f"{book_name}_with_embeddings.json"
    
    # Skip if already processed
    if output_file.exists():
        print(f"[{book_name}] Already processed, skipping...")
        return (book_name, 0, True)
    
    try:
        print(f"[{book_name}] Loading chunks...")
        with open(book_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        if not chunks:
            print(f"[{book_name}] No chunks found, skipping...")
            return (book_name, 0, True)
        
        print(f"[{book_name}] Computing embeddings for {len(chunks)} chunks...")
        
        # Extract texts
        texts = [chunk["text"] for chunk in chunks]
        
        # Compute embeddings
        embeddings = embedder.embed_texts(texts, verbose=True)
        
        # Build output data
        chunks_with_embeddings = []
        for i, chunk in enumerate(chunks):
            chunks_with_embeddings.append({
                "text": chunk["text"],
                "embedding": embeddings[i].tolist(),
                "metadata": chunk["metadata"],
            })
        
        # Save immediately
        print(f"[{book_name}] Saving {len(chunks_with_embeddings)} chunks...")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks_with_embeddings, f, ensure_ascii=False, indent=2)
        
        print(f"[{book_name}] ✓ Done! Saved to {output_file.name}")
        return (book_name, len(chunks), True)
        
    except Exception as e:
        print(f"[{book_name}] ✗ Error: {e}")
        return (book_name, 0, False)


def find_book_files(input_dir: str) -> List[Path]:
    """Find all child chunk files in the input directory."""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        return []
    
    return sorted(input_path.glob("*_child.json"))


def main():
    parser = argparse.ArgumentParser(
        description="Compute embeddings for child chunks using Yandex Cloud"
    )
    
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Directory with chunk JSON files (default: {DEFAULT_INPUT_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess books even if output already exists"
    )
    
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print("Step 2: Computing embeddings (Yandex Cloud)")
    print(f"{'='*60}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}")
    
    # Find all book files
    book_files = find_book_files(args.input_dir)
    
    if not book_files:
        print(f"\nNo child chunk files found in {args.input_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(book_files)} books to process:")
    for f in book_files:
        print(f"  - {f.name}")
    
    # Check which are already processed
    output_path = Path(args.output_dir)
    if not args.force:
        to_process = []
        for bf in book_files:
            book_name = bf.stem.replace("_child", "")
            output_file = output_path / f"{book_name}_with_embeddings.json"
            if output_file.exists():
                print(f"  [skip] {book_name} - already processed")
            else:
                to_process.append(bf)
        book_files = to_process
    
    if not book_files:
        print("\nAll books already processed. Use --force to reprocess.")
        sys.exit(0)
    
    print(f"\nProcessing {len(book_files)} books sequentially...")
    print(f"{'='*60}\n")
    
    # Initialize embedder once
    print("Initializing Yandex Cloud Embedder...")
    embedder = YCEmbedder(variant="text-search-doc")
    
    # Process books one by one
    results = []
    
    for i, book_file in enumerate(book_files, 1):
        print(f"\n[{i}/{len(book_files)}] Processing {book_file.name}")
        print("-" * 40)
        
        result = process_single_book(
            book_file=book_file,
            output_dir=args.output_dir,
            embedder=embedder,
        )
        results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    
    successful = [r for r in results if r[2]]
    failed = [r for r in results if not r[2]]
    total_chunks = sum(r[1] for r in successful)
    
    print(f"Successful: {len(successful)}/{len(results)} books")
    print(f"Total chunks processed: {total_chunks}")
    
    if failed:
        print(f"\nFailed books:")
        for name, _, _ in failed:
            print(f"  - {name}")
    
    print(f"\nOutput directory: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
