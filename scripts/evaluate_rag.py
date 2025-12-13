#!/usr/bin/env python3
"""
RAG Pipeline Evaluation Script.

Evaluates the quality of RAG responses by:
1. Running essay check through the RAG pipeline
2. Using Mistral to evaluate chunk relevance and response quality
3. Comparing with existing verdict

Usage:
    python scripts/evaluate_rag.py
    python scripts/evaluate_rag.py --limit 5
    python scripts/evaluate_rag.py --output results.json

Environment variables:
    YC_API_KEY, YC_FOLDER_ID: Yandex Cloud credentials
    MISTRAL_API_KEY: Mistral API key for evaluation
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from mistralai import Mistral

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedder import YCEmbedder
from src.qdrant_manager import QdrantManager, QdrantConfig
from src.essay_checker import RAGEssayChecker
from src.parent_retriever import ParentDocumentRetriever, load_parent_chunks

# Load environment
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Default paths
DEFAULT_TEST_ESSAYS_DIR = "./data/test_essays"
DEFAULT_QDRANT_PATH = "./data/qdrant_local"
DEFAULT_COLLECTION_NAME = "chunks"
DEFAULT_PARENT_CHUNKS_DIR = "./data/chunks"
DEFAULT_MODEL = "yandexgpt/rc"


EVALUATION_PROMPT = """Ты — эксперт по оценке качества RAG-систем для проверки студенческих эссе.

Твоя задача — оценить качество ответа системы по нескольким критериям.

## Контекст

**Задание студенту:**
{assignment}

**Эссе студента:**
{essay}

**Найденные чанки из книг (контекст для LLM):**
{chunks_info}

**Ответ RAG-системы:**
{rag_response}

**Предыдущий ответ (baseline для сравнения):**
{baseline_verdict}

## Задание

Оцени качество работы RAG-системы по следующим критериям:

### 1. Релевантность чанков (0-100%)
Оцени, насколько найденные чанки релевантны теме эссе.
- Какой % чанков действительно связан с темой эссе?
- Есть ли среди них "шум" (нерелевантные фрагменты)?

### 2. Использование чанков в ответе
- Сколько из найденных чанков реально использовано в ответе?
- Приведены ли конкретные цитаты/ссылки на источники?

### 3. Качество ответа (0-3 балла)
- **1 балл** — есть стиль Джейн Джейкобс (живой, образный, с легкой иронией, от первого лица)
- **1 балл** — есть хорошие, конкретные примеры из чанков с указанием источников
- **1 балл** — ответ RAG-системы лучше, чем baseline verdict

## Формат ответа (JSON)

{{
  "chunk_relevance_percent": <число 0-100>,
  "chunk_relevance_explanation": "<краткое пояснение>",
  "chunks_used_count": <число>,
  "chunks_total_count": <число>,
  "chunks_usage_explanation": "<какие источники использованы>",
  "score_jane_style": <0 или 1>,
  "score_jane_style_explanation": "<пояснение>",
  "score_good_examples": <0 или 1>,
  "score_good_examples_explanation": "<пояснение>",
  "score_better_than_baseline": <0 или 1>,
  "score_better_than_baseline_explanation": "<пояснение>",
  "total_score": <0-3>,
  "overall_comment": "<общий комментарий к качеству ответа>"
}}
"""


def format_chunks_info(chunks: List[Dict[str, Any]], max_chars: int = 10000) -> str:
    """Format chunks for evaluation prompt."""
    parts = []
    total_chars = 0
    
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("meta", chunk.get("metadata", {}))
        
        # Build source info
        source_parts = []
        if meta.get("book_author"):
            source_parts.append(meta["book_author"])
        if meta.get("book_title"):
            source_parts.append(f"«{meta['book_title']}»")
        if meta.get("chapter"):
            source_parts.append(f"глава: {meta['chapter'][:50]}...")
        if meta.get("book_pages_str"):
            source_parts.append(f"стр. {meta['book_pages_str']}")
        
        source = ", ".join(source_parts) if source_parts else f"Источник {i}"
        
        # Get text (full or truncated if too long)
        text = chunk.get("text", "")[:1500]
        if len(chunk.get("text", "")) > 1500:
            text += "..."
        
        entry = f"[{i}] {source}\n{text}\n"
        
        if total_chars + len(entry) > max_chars:
            parts.append(f"... и ещё {len(chunks) - i + 1} чанков")
            break
        
        parts.append(entry)
        total_chars += len(entry)
    
    return "\n".join(parts)


def evaluate_with_mistral(
    assignment: str,
    essay: str,
    chunks: List[Dict[str, Any]],
    rag_response: str,
    baseline_verdict: str,
) -> Dict[str, Any]:
    """Evaluate RAG response using Mistral."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable is not set")
    
    client = Mistral(api_key=api_key, timeout_ms=120000)
    
    # Format prompt
    prompt = EVALUATION_PROMPT.format(
        assignment=assignment if assignment else "Не указано",
        essay=essay,
        chunks_info=format_chunks_info(chunks),
        rag_response=rag_response,
        baseline_verdict=baseline_verdict,
    )
    
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    
    return json.loads(response.choices[0].message.content)


def load_test_case(file_path: Path) -> Dict[str, Any]:
    """Load a single test case."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_rag_pipeline(
    essay_text: str,
    assignment_text: str,
    retriever: ParentDocumentRetriever,
    checker: RAGEssayChecker,
    top_k: int = 5,
) -> tuple:
    """Run RAG pipeline and return response and chunks."""
    # Generate verdict
    feedback, chunks = checker.generate_verdict(
        assignment_text=assignment_text,
        essay_text=essay_text,
        top_k=top_k,
        return_chunks=True,
    )
    
    return feedback, chunks


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline quality")
    
    parser.add_argument(
        "--test-dir",
        default=DEFAULT_TEST_ESSAYS_DIR,
        help=f"Directory with test essays (default: {DEFAULT_TEST_ESSAYS_DIR})"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of test cases to evaluate"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of sources to retrieve (default: 5)"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--skip-rag",
        action="store_true",
        help="Skip RAG generation, only evaluate existing verdicts"
    )
    parser.add_argument(
        "--qdrant-path",
        default=DEFAULT_QDRANT_PATH,
        help=f"Path to Qdrant storage (default: {DEFAULT_QDRANT_PATH})"
    )
    parser.add_argument(
        "--parent-dir",
        default=DEFAULT_PARENT_CHUNKS_DIR,
        help=f"Directory with parent chunks (default: {DEFAULT_PARENT_CHUNKS_DIR})"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📊 RAG Pipeline Evaluation")
    print("=" * 60)
    
    # Find test files
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        sys.exit(1)
    
    test_files = sorted(test_dir.glob("*.json"))
    if args.limit:
        test_files = test_files[:args.limit]
    
    print(f"Found {len(test_files)} test cases")
    
    # Initialize RAG components (if not skipping)
    retriever = None
    checker = None
    
    if not args.skip_rag:
        print("\nInitializing RAG components...")
        
        # Qdrant
        config = QdrantConfig(path=args.qdrant_path, collection_name=DEFAULT_COLLECTION_NAME)
        manager = QdrantManager(config)
        
        if not manager.exists():
            print(f"❌ Qdrant storage not found at {args.qdrant_path}")
            sys.exit(1)
        
        # Parent chunks
        parent_index = load_parent_chunks(args.parent_dir)
        if not parent_index:
            print(f"❌ No parent chunks found in {args.parent_dir}")
            sys.exit(1)
        print(f"  Loaded {len(parent_index)} parent chunks")
        
        # Embedder
        embedder = YCEmbedder()
        
        # Retriever
        retriever = ParentDocumentRetriever(
            embedder=embedder,
            manager=manager,
            parent_index=parent_index,
        )
        
        # Checker
        checker = RAGEssayChecker(
            retriever=retriever,
            model_name=args.model,
        )
        print(f"  Using model: {args.model}")
    
    # Run evaluations
    results = []
    total_scores = {"jane_style": 0, "good_examples": 0, "better_than_baseline": 0, "total": 0}
    total_chunk_relevance = 0
    total_chunks_used = 0
    total_chunks_found = 0
    
    print("\n" + "=" * 60)
    print("Running evaluations...")
    print("=" * 60)
    
    for i, test_file in enumerate(test_files, 1):
        print(f"\n[{i}/{len(test_files)}] {test_file.name}")
        
        try:
            # Load test case
            test_case = load_test_case(test_file)
            essay_text = test_case.get("essay_text", "")
            assignment_text = test_case.get("assignment_text", "")
            baseline_verdict = test_case.get("verdict_text", "")
            
            if not essay_text:
                print("  ⚠️ Empty essay, skipping")
                continue
            
            # Run RAG or use baseline
            if args.skip_rag:
                rag_response = baseline_verdict
                chunks = []
                print("  📝 Using existing verdict (skip-rag mode)")
            else:
                print("  🔍 Running RAG pipeline...")
                start_time = time.time()
                rag_response, chunks = run_rag_pipeline(
                    essay_text=essay_text,
                    assignment_text=assignment_text,
                    retriever=retriever,
                    checker=checker,
                    top_k=args.top_k,
                )
                elapsed = time.time() - start_time
                print(f"  ✅ RAG completed in {elapsed:.1f}s, {len(chunks)} chunks")
            
            # Evaluate with Mistral
            print("  📊 Evaluating with Mistral...")
            evaluation = evaluate_with_mistral(
                assignment=assignment_text,
                essay=essay_text,
                chunks=chunks,
                rag_response=rag_response,
                baseline_verdict=baseline_verdict,
            )
            
            # Accumulate stats
            total_scores["jane_style"] += evaluation.get("score_jane_style", 0)
            total_scores["good_examples"] += evaluation.get("score_good_examples", 0)
            total_scores["better_than_baseline"] += evaluation.get("score_better_than_baseline", 0)
            total_scores["total"] += evaluation.get("total_score", 0)
            total_chunk_relevance += evaluation.get("chunk_relevance_percent", 0)
            total_chunks_used += evaluation.get("chunks_used_count", 0)
            total_chunks_found += evaluation.get("chunks_total_count", len(chunks))
            
            # Print summary
            print(f"  📈 Score: {evaluation.get('total_score', 0)}/3")
            print(f"     Chunk relevance: {evaluation.get('chunk_relevance_percent', 0)}%")
            print(f"     Chunks used: {evaluation.get('chunks_used_count', 0)}/{evaluation.get('chunks_total_count', len(chunks))}")
            
            # Store result
            result = {
                "file": test_file.name,
                "essay_preview": essay_text[:200] + "...",
                "rag_response_preview": rag_response[:300] + "..." if rag_response else "",
                "chunks_count": len(chunks),
                "evaluation": evaluation,
            }
            results.append(result)
            
            # Small delay to avoid rate limits
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate averages
    n = len(results) if results else 1
    
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nTest cases evaluated: {len(results)}/{len(test_files)}")
    
    print("\n📈 AVERAGE SCORES:")
    print(f"  Jane Jacobs style:    {total_scores['jane_style']}/{len(results)} ({100*total_scores['jane_style']/n:.0f}%)")
    print(f"  Good examples:        {total_scores['good_examples']}/{len(results)} ({100*total_scores['good_examples']/n:.0f}%)")
    print(f"  Better than baseline: {total_scores['better_than_baseline']}/{len(results)} ({100*total_scores['better_than_baseline']/n:.0f}%)")
    print(f"  Total score:          {total_scores['total']/n:.2f}/3")
    
    print("\n📚 CHUNK METRICS:")
    print(f"  Avg chunk relevance: {total_chunk_relevance/n:.1f}%")
    print(f"  Avg chunks used:     {total_chunks_used/n:.1f}")
    print(f"  Avg chunks found:    {total_chunks_found/n:.1f}")
    
    # Save results
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "test_dir": str(args.test_dir),
                "model": args.model,
                "top_k": args.top_k,
                "skip_rag": args.skip_rag,
            },
            "summary": {
                "test_cases_total": len(test_files),
                "test_cases_evaluated": len(results),
                "avg_total_score": total_scores['total'] / n,
                "avg_jane_style": total_scores['jane_style'] / n,
                "avg_good_examples": total_scores['good_examples'] / n,
                "avg_better_than_baseline": total_scores['better_than_baseline'] / n,
                "avg_chunk_relevance_percent": total_chunk_relevance / n,
                "avg_chunks_used": total_chunks_used / n,
                "avg_chunks_found": total_chunks_found / n,
            },
            "results": results,
        }
        
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

