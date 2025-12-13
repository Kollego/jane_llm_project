#!/usr/bin/env python3
"""
Step 4: Vector search in Qdrant.

Search for chunks using a query string.
Uses Yandex Cloud embeddings for query encoding.

Environment variables required:
    YC_API_KEY or YC_IAM_TOKEN: Yandex Cloud authentication
    YC_FOLDER_ID: Yandex Cloud folder ID

Usage:
    python scripts/search.py "ваш поисковый запрос"
    python scripts/search.py "что такое креативный город" -k 10
    python scripts/search.py "урбанизм" --filter-book "Образ города"
    python scripts/search.py --stats  # Show collection statistics
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedder import YCEmbedder
from src.qdrant_manager import QdrantManager, QdrantConfig


# Default paths
DEFAULT_QDRANT_PATH = "./data/qdrant_local"
DEFAULT_COLLECTION_NAME = "chunks"


def search(
    query: str,
    qdrant_path: str,
    collection_name: str,
    k: int = 5,
    filter_book: str | None = None,
    filter_author: str | None = None,
    show_content: bool = True,
):
    """Perform vector search and display results."""
    
    # Initialize manager
    config = QdrantConfig(path=qdrant_path, collection_name=collection_name)
    manager = QdrantManager(config)
    
    if not manager.exists():
        print(f"Error: Qdrant storage not found at {qdrant_path}")
        print("Run the pipeline first:")
        print("  1. python scripts/save_chunks.py ...")
        print("  2. python scripts/compute_embeddings.py")
        print("  3. python scripts/load_to_qdrant.py")
        sys.exit(1)
    
    # Initialize Yandex Cloud embedder
    print("Initializing Yandex Cloud Embedder...")
    embedder = YCEmbedder()
    
    # Compute query embedding (uses text-search-query variant)
    print("Computing query embedding...")
    query_vector = embedder.embed_query(query)
    
    # Search
    print(f"\nSearching for: \"{query}\"")
    if filter_book:
        print(f"Filter: book = \"{filter_book}\"")
    if filter_author:
        print(f"Filter: author = \"{filter_author}\"")
    print(f"\nTop {k} results:\n")
    
    results = manager.search(
        query_vector=query_vector,
        k=k,
        filter_book=filter_book,
        filter_author=filter_author,
    )
    
    if not results:
        print("No results found.")
        return
    
    for i, result in enumerate(results, 1):
        print(f"{'='*60}")
        print(f"Result {i} (score: {result.score:.4f})")
        print(f"{'='*60}")
        
        # Book info
        print(f"📚 Book: {result.book_title}")
        if result.book_author:
            print(f"✍️  Author: {result.book_author}")
        if result.chapter:
            print(f"📖 Chapter: {result.chapter}")
        if result.book_pages_str:
            print(f"📄 Pages: {result.book_pages_str}")
        
        # Content
        if show_content:
            text = result.text
            if len(text) > 500:
                text = text[:500] + "..."
            print(f"\n{text}")
        
        print()


def show_stats(qdrant_path: str, collection_name: str):
    """Show collection statistics."""
    
    config = QdrantConfig(path=qdrant_path, collection_name=collection_name)
    manager = QdrantManager(config)
    
    if not manager.exists():
        print(f"Error: Qdrant storage not found at {qdrant_path}")
        return
    
    try:
        stats = manager.get_stats()
        
        print(f"\n{'='*60}")
        print(f"Collection: {stats['collection_name']}")
        print(f"{'='*60}")
        print(f"Total chunks: {stats['points_count']}")
        print(f"Vector size: {stats['vector_size']}")
        print(f"Distance metric: {stats['distance']}")
        
        books = stats['books']
        print(f"\nBooks indexed: {len(books)}")
        print(f"{'-'*60}")
        
        for book, info in sorted(books.items()):
            author_str = f" ({info['author']})" if info['author'] else ""
            print(f"  {book}{author_str}: {info['count']} chunks")
            
    except Exception as e:
        print(f"Error: {e}")


def list_books(qdrant_path: str, collection_name: str):
    """List all books in the collection."""
    
    config = QdrantConfig(path=qdrant_path, collection_name=collection_name)
    manager = QdrantManager(config)
    
    if not manager.exists():
        print(f"Error: Qdrant storage not found at {qdrant_path}")
        return
    
    try:
        books = manager.list_books()
        
        print("\nAvailable books:")
        for book in books:
            print(f"  - {book}")
            
    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Vector search in Qdrant"
    )
    
    parser.add_argument("query", nargs="?", help="Search query")
    
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=5,
        help="Number of results (default: 5)"
    )
    parser.add_argument(
        "--filter-book",
        help="Filter by book title"
    )
    parser.add_argument(
        "--filter-author",
        help="Filter by author"
    )
    parser.add_argument(
        "--no-content",
        action="store_true",
        help="Don't show chunk content"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show collection statistics"
    )
    parser.add_argument(
        "--list-books",
        action="store_true",
        help="List all books in the collection"
    )
    
    # Paths
    parser.add_argument(
        "--qdrant-path",
        default=DEFAULT_QDRANT_PATH,
        help=f"Path to Qdrant storage (default: {DEFAULT_QDRANT_PATH})"
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help=f"Collection name (default: {DEFAULT_COLLECTION_NAME})"
    )
    
    args = parser.parse_args()
    
    if args.stats:
        show_stats(args.qdrant_path, args.collection)
    elif args.list_books:
        list_books(args.qdrant_path, args.collection)
    elif args.query:
        search(
            query=args.query,
            qdrant_path=args.qdrant_path,
            collection_name=args.collection,
            k=args.top_k,
            filter_book=args.filter_book,
            filter_author=args.filter_author,
            show_content=not args.no_content,
        )
    else:
        parser.print_help()
        print("\nExamples:")
        print('  python scripts/search.py "что такое креативный город"')
        print('  python scripts/search.py "урбанизм" -k 10')
        print('  python scripts/search.py "публичные пространства" --filter-book "Образ города"')
        print('  python scripts/search.py --stats')
        print('  python scripts/search.py --list-books')
        sys.exit(1)


if __name__ == "__main__":
    main()
