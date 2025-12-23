import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

from scripts.check_essay import ParentDocumentRetriever
from src.chat import RAGChatSession
from src.essay_checker import RAGEssayChecker
from src.nir_checker import RAGNirChecker
from src.embedder import YCEmbedder
from src.qdrant_manager import QdrantManager, QdrantConfig


DEFAULT_QDRANT_PATH = "./data/qdrant_local"
DEFAULT_COLLECTION_NAME = "chunks"
DEFAULT_PARENT_CHUNKS_DIR = "./data/chunks"
DEFAULT_MODEL = "gemma-3-27b-it/latest"


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

def main():
    mode = input("Режим проверки (essay / nir): ").strip().lower()

    assignment_text = input("\nВведите задание:\n")
    work_text = input("\nВставьте текст работы студента:\n")
    try:
        config = QdrantConfig(path=DEFAULT_QDRANT_PATH, collection_name=DEFAULT_COLLECTION_NAME)
        manager = QdrantManager(config)

        if not manager.exists():
            raise RuntimeError(f"Qdrant storage not found at {DEFAULT_QDRANT_PATH}")

        # Load parent chunks
        print("Loading parent chunks...")
        parent_index = load_parent_chunks(DEFAULT_PARENT_CHUNKS_DIR)

        if not parent_index:
            raise RuntimeError(f"No parent chunks found in {DEFAULT_PARENT_CHUNKS_DIR}")
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
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if mode == "essay":
        checker = RAGEssayChecker(retriever=retriever)
    elif mode == "nir":
        checker = RAGNirChecker(retriever=retriever)
    else:
        raise ValueError("Неизвестный режим")

    chat = RAGChatSession(
        checker=checker,
        assignment_text=assignment_text,
        work_text=work_text,
    )

    print("\n🔍 Чат начат. Напишите 'exit' для выхода.\n")

    while True:
        question = input("Вы: ")
        if question.lower() in ("exit", "quit"):
            break

        answer = chat.ask(question)
        print(f"\n🤖 Ответ:\n{answer}\n")


if __name__ == "__main__":
    main()
