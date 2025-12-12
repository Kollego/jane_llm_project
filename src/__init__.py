# RAG Pipeline for urbanistics books

"""
RAG-модуль для проверки текстов (эссе, НИР и др.).
Содержит:
- Парсинг и чанкинг PDF книг
- Базовый класс BaseRAGChecker и реализации на его основе
"""

# Data processing
from .data import (
    ChunkConfig,
    ParsedPage,
    TOCItem,
    TOCData,
    load_toc,
    load_parsed_pages,
    save_parsed_pages,
    parse_pdf,
    parse_and_save,
    chunk_documents,
    process_book,
    process_books_directory,
    assign_chapters_to_pages,
)

# RAG components
from .base_rag import BaseRAGChecker
from .essay_checker import RAGEssayChecker
from .retriever import load_retriever

__all__ = [
    # Data processing
    "ChunkConfig",
    "ParsedPage",
    "TOCItem",
    "TOCData",
    "load_toc",
    "load_parsed_pages",
    "save_parsed_pages",
    "parse_pdf",
    "parse_and_save",
    "chunk_documents",
    "process_book",
    "process_books_directory",
    "assign_chapters_to_pages",
    # RAG
    "BaseRAGChecker",
    "RAGEssayChecker",
    "load_retriever",
]
