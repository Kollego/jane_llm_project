#!/usr/bin/env python3
"""
Step 3: Load child chunks with embeddings into Qdrant.

Reads chunks with pre-computed embeddings and loads them into 
local Qdrant vector store.

Usage:
    python scripts/load_to_qdrant.py
    python scripts/load_to_qdrant.py --input-dir data/child_chunks_with_embeddings
    python scripts/load_to_qdrant.py --recreate  # Clear existing collection
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# Default paths
DEFAULT_INPUT_DIR = "./data/child_chunks_with_embeddings"
DEFAULT_QDRANT_PATH = "./data/qdrant_local"
DEFAULT_COLLECTION_NAME = "chunks"


def load_chunks_with_embeddings(input_dir: str) -> List[Dict[str, Any]]:
    """Load all chunks with embeddings from JSON files."""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return []
    
    embedding_files = list(input_path.glob("*_with_embeddings.json"))
    
    if not embedding_files:
        print(f"No embedding files found in {input_dir}")
        return []
    
    all_chunks = []
    
    for file_path in embedding_files:
        with open(file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        all_chunks.extend(chunks)
        print(f"  Loaded {len(chunks)} chunks from {file_path.name}")
    
    return all_chunks


def get_vector_size(chunks: List[Dict[str, Any]]) -> int:
    """Determine vector size from first chunk."""
    if not chunks or "embedding" not in chunks[0]:
        raise ValueError("No embeddings found in chunks")
    return len(chunks[0]["embedding"])


def create_points(chunks: List[Dict[str, Any]]) -> List[PointStruct]:
    """Create Qdrant points from chunks."""
    points = []
    
    for chunk in chunks:
        # Use chunk_id from metadata, or generate one
        chunk_id = chunk["metadata"].get("chunk_id")
        if not chunk_id:
            import uuid
            chunk_id = str(uuid.uuid4())
        
        # Build payload (all metadata + text)
        payload = {
            "text": chunk["text"],
            **chunk["metadata"],
        }
        
        point = PointStruct(
            id=chunk_id,
            vector=chunk["embedding"],
            payload=payload,
        )
        points.append(point)
    
    return points


def load_to_qdrant(
    chunks: List[Dict[str, Any]],
    qdrant_path: str,
    collection_name: str,
    recreate: bool = False,
    batch_size: int = 100,
) -> int:
    """Load chunks into Qdrant.
    
    Returns:
        Number of points loaded.
    """
    vector_size = get_vector_size(chunks)
    print(f"Vector size: {vector_size}")
    
    # Initialize Qdrant client (local mode)
    client = QdrantClient(path=qdrant_path)
    
    # Check if collection exists
    collections = [c.name for c in client.get_collections().collections]
    collection_exists = collection_name in collections
    
    if collection_exists:
        if recreate:
            print(f"Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name)
            collection_exists = False
        else:
            print(f"Collection '{collection_name}' already exists, adding to it...")
    
    if not collection_exists:
        print(f"Creating collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
    
    # Create points
    print("Creating points...")
    points = create_points(chunks)
    
    # Upload in batches
    print(f"Uploading {len(points)} points...")
    
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch,
            wait=True,
        )
        
        progress = min(i + batch_size, len(points))
        print(f"  Progress: {progress}/{len(points)} ({100*progress/len(points):.1f}%)")
    
    # Verify
    count = client.count(collection_name=collection_name, exact=True).count
    print(f"\nCollection now has {count} points")
    
    return len(points)


def show_collection_info(qdrant_path: str, collection_name: str):
    """Show information about the collection."""
    qdrant_path_obj = Path(qdrant_path)
    
    if not qdrant_path_obj.exists():
        print(f"Qdrant storage not found at {qdrant_path}")
        return
    
    client = QdrantClient(path=qdrant_path)
    
    try:
        info = client.get_collection(collection_name)
        print(f"\nCollection: {collection_name}")
        print(f"{'='*40}")
        print(f"Total points: {info.points_count}")
        print(f"Vector size: {info.config.params.vectors.size}")
        print(f"Distance: {info.config.params.vectors.distance}")
        
        # Get sample of books
        result = client.scroll(
            collection_name=collection_name,
            limit=1000,
            with_payload=["book_title"],
        )
        
        books = set()
        for point in result[0]:
            if point.payload.get("book_title"):
                books.add(point.payload["book_title"])
        
        print(f"\nBooks indexed: {len(books)}")
        for book in sorted(books):
            print(f"  - {book}")
            
    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Load child chunks with embeddings into Qdrant"
    )
    
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Directory with embedding files (default: {DEFAULT_INPUT_DIR})"
    )
    parser.add_argument(
        "--qdrant-path",
        default=DEFAULT_QDRANT_PATH,
        help=f"Path to Qdrant local storage (default: {DEFAULT_QDRANT_PATH})"
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help=f"Collection name (default: {DEFAULT_COLLECTION_NAME})"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete existing collection and create new one"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for uploading (default: 100)"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show collection info and exit"
    )
    
    args = parser.parse_args()
    
    if args.info:
        show_collection_info(args.qdrant_path, args.collection)
        return
    
    print(f"{'='*60}")
    print("Step 3: Loading chunks into Qdrant")
    print(f"{'='*60}")
    print(f"Input: {args.input_dir}")
    print(f"Qdrant path: {args.qdrant_path}")
    print(f"Collection: {args.collection}")
    print(f"Recreate: {args.recreate}")
    print(f"{'='*60}")
    
    # Load chunks
    print("\nLoading chunks with embeddings...")
    chunks = load_chunks_with_embeddings(args.input_dir)
    
    if not chunks:
        print("No chunks to load")
        sys.exit(1)
    
    print(f"\nTotal chunks: {len(chunks)}")
    
    # Load to Qdrant
    loaded = load_to_qdrant(
        chunks=chunks,
        qdrant_path=args.qdrant_path,
        collection_name=args.collection,
        recreate=args.recreate,
        batch_size=args.batch_size,
    )
    
    print(f"\n{'='*60}")
    print(f"Done! Loaded {loaded} chunks into Qdrant")
    print(f"Qdrant path: {args.qdrant_path}")
    print(f"Collection: {args.collection}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

