#!/usr/bin/env python3
"""
RAG Pipeline Evaluation Script using RAGAS.

Evaluates the quality of RAG responses using standard RAGAS metrics:
- Faithfulness: How factually consistent is the response with the context
- Answer Relevancy: How relevant is the answer to the question
- Context Precision: How precise is the retrieved context
- Context Recall: How well does the context cover the ground truth

Usage:
    python scripts/evaluate_rag_ragas.py
    python scripts/evaluate_rag_ragas.py --limit 5
    python scripts/evaluate_rag_ragas.py --output results_ragas.json
    python scripts/evaluate_rag_ragas.py --evaluator-model mistral-small-latest

Environment variables:
    YC_API_KEY, YC_FOLDER_ID: Yandex Cloud credentials
    MISTRAL_API_KEY: Mistral API key for RAGAS evaluation
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
DEFAULT_MODEL = "gemma-3-27b-it/latest"
DEFAULT_EVALUATOR_MODEL = "mistral-small-latest"


def prepare_ragas_dataset(
    test_cases: List[Dict[str, Any]],
    retriever: ParentDocumentRetriever,
    checker: RAGEssayChecker,
    top_k: int = 5,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Prepare dataset in RAGAS format by running RAG pipeline.
    
    RAGAS expects:
    - user_input: The question/query
    - response: Generated answer
    - retrieved_contexts: List of context strings
    - reference: Ground truth answer (optional, for context_recall)
    """
    samples = []
    
    for i, test_case in enumerate(test_cases, 1):
        essay_text = test_case.get("essay_text", "")
        assignment_text = test_case.get("assignment_text", "")
        baseline_verdict = test_case.get("verdict_text", "")
        file_name = test_case.get("file_name", f"test_{i}")
        
        if not essay_text:
            if verbose:
                print(f"  [{i}] ⚠️ Empty essay, skipping")
            continue
        
        if verbose:
            print(f"  [{i}/{len(test_cases)}] Processing {file_name}...")
        
        try:
            # Build user input (question for RAG)
            user_input = f"Задание: {assignment_text}\n\nЭссе студента:\n{essay_text}"
            
            # Run RAG pipeline
            start_time = time.time()
            response, chunks = checker.generate_verdict(
                assignment_text=assignment_text,
                essay_text=essay_text,
                top_k=top_k,
                return_chunks=True,
            )
            elapsed = time.time() - start_time
            
            # Extract context texts
            contexts = []
            for chunk in chunks:
                text = chunk.get("text", "")
                meta = chunk.get("meta", chunk.get("metadata", {}))
                
                # Add source info to context
                source_parts = []
                if meta.get("book_author"):
                    source_parts.append(meta["book_author"])
                if meta.get("book_title"):
                    source_parts.append(f"«{meta['book_title']}»")
                
                source = ", ".join(source_parts) if source_parts else ""
                if source:
                    text = f"[{source}]\n{text}"
                
                contexts.append(text)
            
            sample = {
                "user_input": user_input,
                "response": response,
                "retrieved_contexts": contexts,
                "reference": baseline_verdict,  # Ground truth for context_recall
                "metadata": {
                    "file_name": file_name,
                    "elapsed_time": elapsed,
                    "chunks_count": len(chunks),
                }
            }
            samples.append(sample)
            
            if verbose:
                print(f"      ✅ Done in {elapsed:.1f}s, {len(chunks)} chunks")
            
            # Small delay to avoid rate limits
            time.sleep(0.3)
            
        except Exception as e:
            if verbose:
                print(f"      ❌ Error: {e}")
            continue
    
    return samples


def evaluate_with_ragas(
    samples: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
    evaluator_model: str = DEFAULT_EVALUATOR_MODEL,
) -> Dict[str, Any]:
    """
    Evaluate samples using RAGAS metrics with Mistral.
    
    Available metrics:
    - faithfulness: Is the response factually consistent with context?
    - context_precision: Are the relevant contexts ranked higher?
    - context_recall: Does context cover the ground truth?
    
    Note: answer_relevancy requires embeddings and is excluded by default.
    
    Args:
        samples: List of samples with user_input, response, retrieved_contexts, reference
        metrics: List of metric names to compute (default: all except answer_relevancy)
        evaluator_model: Mistral model name for evaluation
    """
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        LLMContextRecall,
    )
    from langchain_mistralai import ChatMistralAI
    from ragas.llms import LangchainLLMWrapper
    
    # Setup Mistral evaluator LLM
    llm = ChatMistralAI(model=evaluator_model, temperature=0)
    wrapped_llm = LangchainLLMWrapper(llm)
    
    # Create metric instances with LLM (no embeddings needed)
    faithfulness = Faithfulness(llm=wrapped_llm)
    context_precision = LLMContextPrecisionWithoutReference(llm=wrapped_llm)
    context_recall = LLMContextRecall(llm=wrapped_llm)
    
    # Select metrics (excluding answer_relevancy which requires embeddings)
    default_metrics = [
        faithfulness,
        context_precision,
        context_recall,
    ]
    
    metric_names = ["faithfulness", "context_precision", "context_recall"]
    selected_metrics = default_metrics
    if metrics:
        metric_map = {
            "faithfulness": faithfulness,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }
        selected_metrics = [metric_map[m] for m in metrics if m in metric_map]
        metric_names = [m for m in metrics if m in metric_map]
    
    # Convert to RAGAS 0.4.x format using SingleTurnSample
    ragas_samples = []
    for s in samples:
        sample = SingleTurnSample(
            user_input=s["user_input"],
            response=s["response"],
            retrieved_contexts=s["retrieved_contexts"],
            reference=s.get("reference", ""),
        )
        ragas_samples.append(sample)
    
    dataset = EvaluationDataset(samples=ragas_samples)
    
    # Run evaluation
    print(f"\n🔬 Running RAGAS evaluation with {len(selected_metrics)} metrics...")
    print(f"   Evaluator: Mistral/{evaluator_model}")
    print(f"   Samples: {len(samples)}")
    print(f"   Metrics: {', '.join(metric_names)}")
    
    results = evaluate(
        dataset=dataset,
        metrics=selected_metrics,
    )
    
    return results


def load_test_cases(test_dir: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load test cases from directory."""
    test_files = sorted(test_dir.glob("*.json"))
    if limit:
        test_files = test_files[:limit]
    
    test_cases = []
    for file_path in test_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            data["file_name"] = file_path.name
            test_cases.append(data)
    
    return test_cases


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline with RAGAS")
    
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
        help=f"LLM model for RAG generation (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--evaluator-model",
        default=DEFAULT_EVALUATOR_MODEL,
        help=f"Mistral model for RAGAS evaluation (default: {DEFAULT_EVALUATOR_MODEL})"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["faithfulness", "context_precision", "context_recall"],
        default=None,
        help="Specific metrics to compute (default: all). Note: answer_relevancy excluded (requires OpenAI embeddings)"
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
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip RAG generation, load samples from file"
    )
    parser.add_argument(
        "--samples-file",
        default=None,
        help="Path to pre-generated samples JSON file"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📊 RAG Pipeline Evaluation with RAGAS")
    print("=" * 60)
    
    # Check for Mistral API key
    if not os.getenv("MISTRAL_API_KEY"):
        print("❌ MISTRAL_API_KEY not set.")
        print("   Export MISTRAL_API_KEY to use RAGAS evaluation.")
        sys.exit(1)
    
    samples = []
    
    if args.skip_generation and args.samples_file:
        # Load pre-generated samples
        print(f"\n📂 Loading samples from {args.samples_file}...")
        with open(args.samples_file, "r", encoding="utf-8") as f:
            samples = json.load(f)
        print(f"   Loaded {len(samples)} samples")
    else:
        # Find test files
        test_dir = Path(args.test_dir)
        if not test_dir.exists():
            print(f"❌ Test directory not found: {test_dir}")
            sys.exit(1)
        
        test_cases = load_test_cases(test_dir, args.limit)
        print(f"\n📚 Found {len(test_cases)} test cases")
        
        # Initialize RAG components
        print("\n🔧 Initializing RAG components...")
        
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
        print(f"   ✅ Loaded {len(parent_index)} parent chunks")
        
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
        print(f"   ✅ Using model: {args.model}")
        
        # Prepare dataset
        print("\n" + "=" * 60)
        print("🔄 Running RAG pipeline on test cases...")
        print("=" * 60)
        
        samples = prepare_ragas_dataset(
            test_cases=test_cases,
            retriever=retriever,
            checker=checker,
            top_k=args.top_k,
            verbose=True,
        )
        
        print(f"\n✅ Prepared {len(samples)} samples")
        
        # Save samples for reuse
        samples_output = args.output.replace(".json", "_samples.json") if args.output else "ragas_samples.json"
        with open(samples_output, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"💾 Samples saved to: {samples_output}")
    
    if not samples:
        print("❌ No samples to evaluate")
        sys.exit(1)
    
    # Run RAGAS evaluation
    print("\n" + "=" * 60)
    print("🔬 RAGAS Evaluation")
    print("=" * 60)
    
    try:
        results = evaluate_with_ragas(
            samples=samples,
            metrics=args.metrics,
            evaluator_model=args.evaluator_model,
        )
        
        # Print results
        print("\n" + "=" * 60)
        print("📊 EVALUATION RESULTS")
        print("=" * 60)
        
        # Get per-sample scores as DataFrame
        df = results.to_pandas()
        
        # Metric columns (exclude text columns)
        text_cols = ["user_input", "response", "retrieved_contexts", "reference"]
        metric_cols = [col for col in df.columns if col not in text_cols]
        
        print("\n📈 RAGAS SCORES (averages):")
        summary = {}
        for col in metric_cols:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                avg = df[col].mean()
                summary[col] = float(avg)
                print(f"   {col}: {avg:.4f}")
        
        print(f"\n📋 Per-metric statistics:")
        for col in metric_cols:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                print(f"   {col}:")
                print(f"      mean: {df[col].mean():.4f}")
                print(f"      std:  {df[col].std():.4f}")
                print(f"      min:  {df[col].min():.4f}")
                print(f"      max:  {df[col].max():.4f}")
        
        # Save results
        if args.output:
            # Clean DataFrame for JSON serialization
            df_clean = df.drop(columns=text_cols, errors='ignore')
            
            output_data = {
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "test_dir": str(args.test_dir),
                    "model": args.model,
                    "evaluator_model": args.evaluator_model,
                    "top_k": args.top_k,
                    "metrics": args.metrics or ["faithfulness", "context_precision", "context_recall"],
                },
                "summary": summary,
                "per_sample": df_clean.to_dict(orient="records"),
            }
            
            output_path = Path(args.output)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"\n💾 Results saved to: {output_path}")
        
    except Exception as e:
        print(f"\n❌ RAGAS evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
