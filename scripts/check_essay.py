#!/usr/bin/env python3
"""
Essay checker with Parent Document Retriever and RAGEssayChecker.

Searches by child chunks, returns parent chunks as context,
and generates feedback using LLM.

Environment variables required:
    YC_API_KEY or YC_IAM_TOKEN: Yandex Cloud authentication
    YC_FOLDER_ID: Yandex Cloud folder ID
    OPENAI_API_KEY: OpenAI/compatible API key
    OPENAI_API_BASE: API base URL (optional)

Usage:
    python scripts/check_essay.py --essay-file essay.txt --assignment-file task.txt
    python scripts/check_essay.py --essay "текст эссе" --top-k 5
    python scripts/check_essay.py --essay-file essay.txt --no-llm  # Only retrieve sources
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedder import YCEmbedder
from src.qdrant_manager import QdrantManager, QdrantConfig
from src.essay_checker import RAGEssayChecker


# Default paths
DEFAULT_QDRANT_PATH = "./data/qdrant_local"
DEFAULT_COLLECTION_NAME = "chunks"
DEFAULT_PARENT_CHUNKS_DIR = "./data/chunks"
DEFAULT_MODEL = "gemma-3-27b-it/latest"


class ParentDocumentRetriever:
    """
    Retriever that searches child chunks and returns parent chunks.
    
    Implements the retrieve(query, top_k) interface expected by BaseRAGChecker.
    """
    
    def __init__(
        self,
        embedder: YCEmbedder,
        manager: QdrantManager,
        parent_index: Dict[str, Dict[str, Any]],
    ):
        self.embedder = embedder
        self.manager = manager
        self.parent_index = parent_index
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search child chunks and return corresponding parent chunks.
        
        Returns:
            List of dicts with 'text' and 'metadata' keys
        """
        # Get query embedding
        query_vector = self.embedder.embed_query(query)
        
        # Search child chunks (get more to deduplicate)
        results = self.manager.search(query_vector=query_vector, k=top_k * 3)
        
        # Collect unique parent chunks
        seen_parent_ids = set()
        parent_chunks = []
        
        for result in results:
            parent_id = result.metadata.get("parent_id")
            
            if not parent_id or parent_id in seen_parent_ids:
                continue
            
            seen_parent_ids.add(parent_id)
            parent = self.parent_index.get(parent_id)
            
            if parent:
                parent_chunks.append({
                    "text": parent["text"],
                    "metadata": parent["metadata"],
                    "score": result.score,
                })
            
            if len(parent_chunks) >= top_k:
                break
        
        return parent_chunks


def load_parent_chunks(parent_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load all parent chunks and index by parent_id."""
    parent_path = Path(parent_dir)
    
    if not parent_path.exists():
        return {}
    
    parent_files = list(parent_path.glob("*_parent.json"))
    parent_index = {}
    
    for file_path in parent_files:
        with open(file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        for chunk in chunks:
            parent_id = chunk["metadata"].get("chunk_id")
            if parent_id:
                parent_index[parent_id] = chunk
    
    return parent_index


def print_sources(chunks: List[Dict[str, Any]]):
    """Print sources in a nice format."""
    print("\n" + "="*60)
    print("📚 НАЙДЕННЫЕ ИСТОЧНИКИ:")
    print("="*60)
    
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("meta", chunk.get("metadata", {}))
        print(f"\n{i}. {meta.get('book_title', 'Unknown')}")
        if meta.get('book_author'):
            print(f"   Автор: {meta['book_author']}")
        if meta.get('chapter'):
            print(f"   Глава: {meta['chapter']}")
        if meta.get('book_pages_str'):
            print(f"   Страницы: {meta['book_pages_str']}")
        if 'score' in chunk:
            print(f"   Релевантность: {chunk['score']:.4f}")


def print_feedback(feedback: str):
    """Print feedback in a nice format."""
    print("\n" + "="*60)
    print("📝 ОБРАТНАЯ СВЯЗЬ:")
    print("="*60)
    print(feedback)


def main():
    parser = argparse.ArgumentParser(
        description="Essay checker with RAGEssayChecker and Parent Document Retriever"
    )
    
    # Input
    parser.add_argument("--essay", help="Essay text directly")
    parser.add_argument("--essay-file", help="Path to essay text file")
    parser.add_argument("--assignment", help="Assignment text directly")
    parser.add_argument("--assignment-file", help="Path to assignment text file")
    
    # Search params
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of sources to retrieve (default: 5)"
    )
    
    # LLM params
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Only retrieve sources, don't generate feedback"
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
    parser.add_argument(
        "--parent-dir",
        default=DEFAULT_PARENT_CHUNKS_DIR,
        help=f"Directory with parent chunks (default: {DEFAULT_PARENT_CHUNKS_DIR})"
    )
    
    # Output
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Show full context text"
    )
    parser.add_argument(
        "--output-json",
        help="Save result to JSON file"
    )
    
    args = parser.parse_args()
    
    # Get essay text
    if args.essay:
        essay_text = args.essay
    elif args.essay_file:
        with open(args.essay_file, "r", encoding="utf-8") as f:
            essay_text = f.read()
    else:
        parser.print_help()
        print("\nError: Specify --essay or --essay-file")
        sys.exit(1)
    
    # Get assignment text
    if args.assignment:
        assignment_text = args.assignment
    elif args.assignment_file:
        with open(args.assignment_file, "r", encoding="utf-8") as f:
            assignment_text = f.read()
    else:
        assignment_text = ""
    
    print(f"{'='*60}")
    print("🎓 Essay Checker with RAGEssayChecker")
    print(f"{'='*60}")
    print(f"Essay: {len(essay_text)} chars")
    if assignment_text:
        print(f"Assignment: {len(assignment_text)} chars")
    print(f"Sources: top-{args.top_k}")
    print(f"Model: {args.model}")
    print(f"{'='*60}\n")
    
    try:
        # Initialize Qdrant
        print("Initializing Qdrant...")
        config = QdrantConfig(path=args.qdrant_path, collection_name=args.collection)
        manager = QdrantManager(config)
        
        if not manager.exists():
            raise RuntimeError(f"Qdrant storage not found at {args.qdrant_path}")
        
        # Load parent chunks
        print("Loading parent chunks...")
        parent_index = load_parent_chunks(args.parent_dir)
        
        if not parent_index:
            raise RuntimeError(f"No parent chunks found in {args.parent_dir}")
        print(f"  Loaded {len(parent_index)} parent chunks")
        
        # Initialize embedder
        print("Initializing Yandex Cloud Embedder...")
        embedder = YCEmbedder()
        
        # Create Parent Document Retriever
        retriever = ParentDocumentRetriever(
            embedder=embedder,
            manager=manager,
            parent_index=parent_index,
        )
        
        # Create RAGEssayChecker
        print(f"Initializing RAGEssayChecker with {args.model}...")
        checker = RAGEssayChecker(
            retriever=retriever,
            model_name=args.model,
        )
        
        if args.no_llm:
            # Only retrieve sources
            print("Searching for relevant sources...")
            chunks, _ = checker.retrieve_top_k(essay_text, args.top_k)
            print_sources(chunks)
            
            if args.show_context:
                context = checker.format_context(chunks)
                print("\n" + "="*60)
                print("📖 КОНТЕКСТ:")
                print("="*60)
                print(context)
        else:
            # Generate verdict with LLM
            print("Generating feedback...")
            feedback, chunks = checker.generate_verdict(
                assignment_text=assignment_text,
                essay_text=essay_text,
                top_k=args.top_k,
                return_chunks=True,
            )
            
            # Print sources
            print_sources(chunks)
            
            # Print context if requested
            if args.show_context:
                context = checker.format_context(chunks)
                print("\n" + "="*60)
                print("📖 КОНТЕКСТ:")
                print("="*60)
                print(context)
            
            # Print feedback
            print_feedback(feedback)
            
            # Save to JSON if requested
            if args.output_json:
                sources = []
                for chunk in chunks:
                    meta = chunk.get("meta", {})
                    sources.append({
                        "book_title": meta.get("book_title"),
                        "book_author": meta.get("book_author"),
                        "chapter": meta.get("chapter"),
                        "book_pages_str": meta.get("book_pages_str"),
                    })
                
                output = {
                    "sources": sources,
                    "feedback": feedback,
                    "essay_preview": essay_text[:500] + "..." if len(essay_text) > 500 else essay_text,
                    "assignment_preview": assignment_text[:500] + "..." if len(assignment_text) > 500 else assignment_text,
                }
                with open(args.output_json, "w", encoding="utf-8") as f:
                    json.dump(output, f, ensure_ascii=False, indent=2)
                print(f"\n💾 Result saved to: {args.output_json}")
        
        print(f"\n{'='*60}")
        print("✅ Done!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
