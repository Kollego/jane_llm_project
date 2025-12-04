"""
PDF parsing and chunking pipeline for RAG.

This module provides functions for:
1. Parsing PDF books using LlamaParse
2. Chunking with Parent Document Retriever strategy
3. Outputting LangChain Document objects with rich metadata
"""

import json
import os
import re
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_parse import LlamaParse


# Type alias for nested TOC structure
# Can be: {"Part 1": ["Chapter 1", "Chapter 2"]} 
# Or: {"Part 1": {"Chapter 1": ["Section 1", "Section 2"]}}
TOCStructure = Dict[str, Union[List[str], "TOCStructure"]]

# Load environment variables from .env file
load_dotenv()


@dataclass
class ChunkConfig:
    """Configuration for document chunking.
    
    Attributes:
        child_chunk_size: Size of child chunks in characters (for vector search).
        child_chunk_overlap: Overlap between child chunks.
        parent_chunk_size: Size of parent chunks in characters (for context).
        parent_chunk_overlap: Overlap between parent chunks.
        language: Language of the documents (for better parsing).
    """
    child_chunk_size: int = 400
    child_chunk_overlap: int = 50
    parent_chunk_size: int = 2000
    parent_chunk_overlap: int = 200
    language: str = "ru"


@dataclass
class ParsedPage:
    """Represents a parsed page from a PDF document.
    
    Attributes:
        page_number: The page number (1-indexed, position in PDF).
        content: The text content of the page.
        chapter: Chapter/section this page belongs to (hierarchical path).
        metadata: Additional metadata from parsing.
    """
    page_number: int
    content: str
    chapter: Optional[List[str]] = None  # e.g., ["Part 1", "Chapter 2", "Section 1"]
    metadata: dict = field(default_factory=dict)


# =============================================================================
# Table of Contents (TOC) Functions
# =============================================================================

def load_toc(toc_path: str) -> TOCStructure:
    """Load table of contents from a JSON file.
    
    The TOC file can have a nested structure like:
    {
        "TOC": {
            "Part 1": ["Chapter 1", "Chapter 2"],
            "Part 2": {
                "Chapter 3": ["Section 3.1", "Section 3.2"]
            }
        }
    }
    
    Or without the "TOC" wrapper:
    {
        "Part 1": ["Chapter 1", "Chapter 2"],
        ...
    }
    
    Args:
        toc_path: Path to the JSON file with TOC.
        
    Returns:
        Nested dictionary representing the TOC structure.
    """
    toc_path = Path(toc_path)
    
    if not toc_path.exists():
        raise FileNotFoundError(f"TOC file not found: {toc_path}")
    
    with open(toc_path, 'r', encoding='utf-8') as f:
        toc = json.load(f)
    
    # Handle "TOC" wrapper if present
    if "TOC" in toc:
        toc = toc["TOC"]
    
    return toc


def flatten_toc(toc: TOCStructure, parent_path: List[str] = None) -> List[List[str]]:
    """Flatten nested TOC into a list of paths.
    
    Each path is a list representing the hierarchy, e.g.:
    ["Part 1", "Chapter 2", "Section 1"]
    
    Args:
        toc: Nested TOC structure.
        parent_path: Current path in the hierarchy (used for recursion).
        
    Returns:
        List of paths, where each path is a list of titles from root to leaf.
    """
    if parent_path is None:
        parent_path = []
    
    paths = []
    
    for key, value in toc.items():
        current_path = parent_path + [key]
        
        if isinstance(value, list):
            # Leaf level - list of chapter/section names
            for item in value:
                paths.append(current_path + [item])
        elif isinstance(value, dict):
            # Nested level - recurse
            paths.extend(flatten_toc(value, current_path))
        else:
            # Single string value
            paths.append(current_path + [value])
    
    return paths


def get_all_titles_from_toc(toc: TOCStructure) -> List[str]:
    """Extract all unique titles from TOC (at all levels).
    
    Args:
        toc: Nested TOC structure.
        
    Returns:
        List of all unique titles.
    """
    titles = set()
    
    def extract_titles(obj: Any):
        if isinstance(obj, dict):
            for key, value in obj.items():
                titles.add(key)
                extract_titles(value)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, str):
                    titles.add(item)
                else:
                    extract_titles(item)
        elif isinstance(obj, str):
            titles.add(obj)
    
    extract_titles(toc)
    return list(titles)


def normalize_text_for_search(text: str) -> str:
    """Normalize text for fuzzy matching.
    
    Args:
        text: Text to normalize.
        
    Returns:
        Normalized text (lowercase, extra spaces removed).
    """
    # Lowercase
    text = text.lower()
    # Replace multiple spaces/newlines with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:\-—–]', '', text)
    return text.strip()


def find_title_in_text(title: str, text: str) -> bool:
    """Check if a title appears in text (fuzzy matching).
    
    Args:
        title: Title to search for.
        text: Text to search in.
        
    Returns:
        True if title found in text.
    """
    norm_title = normalize_text_for_search(title)
    norm_text = normalize_text_for_search(text)
    
    # Direct substring match
    if norm_title in norm_text:
        return True
    
    # Try matching with some flexibility (e.g., "Глава 1" matches "ГЛАВА 1.")
    # Remove common prefixes/suffixes for matching
    title_words = norm_title.split()
    if len(title_words) >= 2:
        # Try matching first few significant words
        partial_title = ' '.join(title_words[:3])
        if partial_title in norm_text:
            return True
    
    return False


def assign_chapters_to_pages(
    parsed_pages: List[ParsedPage],
    toc: TOCStructure,
) -> List[ParsedPage]:
    """Assign chapter information to parsed pages based on TOC.
    
    Searches for chapter/section titles in page text and assigns
    the hierarchical chapter path to each page.
    
    Args:
        parsed_pages: List of parsed pages.
        toc: Table of contents structure.
        
    Returns:
        List of parsed pages with chapter field filled in.
    """
    # Flatten TOC to get all paths
    all_paths = flatten_toc(toc)
    
    # Create a mapping: title -> full path
    title_to_path = {}
    for path in all_paths:
        # Map each title in the path to the path up to and including it
        for i, title in enumerate(path):
            if title not in title_to_path:
                title_to_path[title] = path[:i + 1]
    
    # Sort paths by depth (deepest first) for more specific matching
    sorted_paths = sorted(all_paths, key=lambda x: -len(x))
    
    # Find chapter boundaries
    # chapter_starts[page_number] = chapter_path
    chapter_starts: Dict[int, List[str]] = {}
    
    for page in parsed_pages:
        page_text = page.content
        
        # Check each path (from most specific to least)
        for path in sorted_paths:
            # Check the last (most specific) title in the path
            title = path[-1]
            if find_title_in_text(title, page_text):
                chapter_starts[page.page_number] = path
                break  # Found the most specific match
    
    # Propagate chapters to all pages
    current_chapter = None
    updated_pages = []
    
    for page in parsed_pages:
        # Check if new chapter starts on this page
        if page.page_number in chapter_starts:
            current_chapter = chapter_starts[page.page_number]
        
        # Create updated page with chapter info
        updated_page = ParsedPage(
            page_number=page.page_number,
            content=page.content,
            chapter=current_chapter.copy() if current_chapter else None,
            metadata=page.metadata.copy(),
        )
        updated_pages.append(updated_page)
    
    # Log statistics
    pages_with_chapters = sum(1 for p in updated_pages if p.chapter)
    chapters_found = len(chapter_starts)
    print(f"Found {chapters_found} chapter boundaries, assigned chapters to {pages_with_chapters}/{len(updated_pages)} pages")
    
    return updated_pages


def save_parsed_pages(
    parsed_pages: List[ParsedPage],
    output_path: str,
) -> None:
    """Save parsed pages to a JSON file.
    
    Useful for caching parsing results to avoid re-parsing expensive PDFs.
    
    Args:
        parsed_pages: List of ParsedPage objects to save.
        output_path: Path to the output JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = [asdict(page) for page in parsed_pages]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(parsed_pages)} parsed pages to {output_path}")


def load_parsed_pages(input_path: str) -> List[ParsedPage]:
    """Load parsed pages from a JSON file.
    
    Args:
        input_path: Path to the JSON file with parsed pages.
        
    Returns:
        List of ParsedPage objects.
        
    Raises:
        FileNotFoundError: If the JSON file doesn't exist.
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Parsed pages file not found: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    parsed_pages = [
        ParsedPage(
            page_number=item['page_number'],
            content=item['content'],
            chapter=item.get('chapter'),
            metadata=item.get('metadata', {}),
        )
        for item in data
    ]
    
    print(f"Loaded {len(parsed_pages)} parsed pages from {input_path}")
    return parsed_pages


def parse_pdf(
    file_path: str,
    language: str = "ru",
) -> List[ParsedPage]:
    """Parse a PDF file using LlamaParse.
    
    This function uses LlamaParse (cloud-based) to extract text from PDF
    while preserving structure, handling tables, and maintaining page boundaries.
    
    Args:
        file_path: Path to the PDF file.
        language: Language of the document for better OCR results.
        
    Returns:
        List of ParsedPage objects, one per page.
        
    Raises:
        ValueError: If LLAMA_CLOUD_API_KEY is not set.
        FileNotFoundError: If the PDF file doesn't exist.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise ValueError(
            "LLAMA_CLOUD_API_KEY environment variable is not set. "
            "Get your API key at https://cloud.llamaindex.ai/"
        )
    
    # Initialize LlamaParse with settings optimized for books
    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",  # Markdown preserves structure well
        language=language,
        system_prompt=(
            "This is a book about urbanism/urban studies. "
            "Preserve paragraph structure, headings, and any citations. "
            "Keep chapter titles and section headers clearly marked."
        ),
    )
    
    # Parse the document - LlamaParse returns documents per page
    documents = parser.load_data(str(file_path))
    
    parsed_pages = []
    for i, doc in enumerate(documents):
        # LlamaParse includes metadata about the source
        page_metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        
        parsed_pages.append(ParsedPage(
            page_number=i + 1,  # 1-indexed page numbers
            content=doc.text,
            metadata=page_metadata,
        ))
    
    return parsed_pages


def chunk_documents(
    parsed_pages: List[ParsedPage],
    config: ChunkConfig,
    source_path: str,
    book_title: str,
) -> Tuple[List[Document], List[Document]]:
    """Chunk parsed pages using Parent Document Retriever strategy.
    
    This function implements a two-level chunking strategy:
    - Parent chunks: One per chapter (full chapter text for context)
    - Child chunks: Small chunks (e.g., 400 chars) optimized for vector search
    
    Each child chunk references its parent chunk via parent_id in metadata.
    
    Args:
        parsed_pages: List of ParsedPage objects from parse_pdf().
        config: ChunkConfig with size/overlap settings.
        source_path: Original file path (for metadata).
        book_title: Title of the book (for metadata).
        
    Returns:
        Tuple of (child_documents, parent_documents).
    """
    # Group pages by chapter
    # chapter_key -> list of pages
    from collections import OrderedDict
    chapters: OrderedDict[str, List[ParsedPage]] = OrderedDict()
    
    for page in parsed_pages:
        # Create chapter key from chapter path
        if page.chapter:
            chapter_key = " > ".join(page.chapter)
        else:
            chapter_key = "_no_chapter_"
        
        if chapter_key not in chapters:
            chapters[chapter_key] = []
        chapters[chapter_key].append(page)
    
    # Create child splitter
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.child_chunk_size,
        chunk_overlap=config.child_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    
    parent_documents = []
    child_documents = []
    
    for chapter_key, chapter_pages in chapters.items():
        # Combine all pages in this chapter into one parent chunk
        chapter_text = "\n\n".join(page.content for page in chapter_pages)
        
        # Get page range for this chapter
        page_numbers = [page.page_number for page in chapter_pages]
        page_range = sorted(page_numbers)
        
        # Generate unique ID for parent chunk
        parent_id = str(uuid.uuid4())
        
        # Build parent metadata
        parent_metadata = {
            "source": source_path,
            "book_title": book_title,
            "page_number": page_range[0],  # First page of chapter
            "page_range": page_range,
            "chunk_type": "parent",
            "chunk_id": parent_id,
        }
        
        # Add chapter info if available
        if chapter_key != "_no_chapter_":
            parent_metadata["chapter"] = chapter_key
        
        # Create parent document (full chapter)
        parent_doc = Document(
            page_content=chapter_text,
            metadata=parent_metadata,
        )
        parent_documents.append(parent_doc)
        
        # Split chapter into child chunks
        child_texts = child_splitter.split_text(chapter_text)
        
        for child_text in child_texts:
            child_metadata = {
                "source": source_path,
                "book_title": book_title,
                "page_number": page_range[0],
                "page_range": page_range,
                "chunk_type": "child",
                "parent_id": parent_id,
                "chunk_id": str(uuid.uuid4()),
            }
            
            if chapter_key != "_no_chapter_":
                child_metadata["chapter"] = chapter_key
            
            child_doc = Document(
                page_content=child_text,
                metadata=child_metadata,
            )
            child_documents.append(child_doc)
    
    return child_documents, parent_documents


def extract_book_title(file_path: str) -> str:
    """Extract book title from filename.
    
    Cleans up the filename to create a human-readable title.
    
    Args:
        file_path: Path to the PDF file.
        
    Returns:
        Cleaned book title string.
    """
    path = Path(file_path)
    # Remove extension and clean up
    title = path.stem
    # Replace underscores with spaces
    title = title.replace("_", " ")
    # Remove common file suffixes
    for suffix in [".pdf", ".PDF"]:
        title = title.replace(suffix, "")
    return title.strip()


def process_book(
    file_path: str,
    config: ChunkConfig | None = None,
    save_parsed_path: Optional[str] = None,
    load_parsed_path: Optional[str] = None,
    toc_path: Optional[str] = None,
) -> Tuple[List[Document], List[Document]]:
    """Process a book PDF file through the full RAG pipeline.
    
    This is the main entry point that orchestrates:
    1. PDF parsing with LlamaParse (or loading from cache)
    2. Chapter assignment based on TOC (if provided)
    3. Chunking with Parent Document Retriever strategy
    4. Metadata enrichment
    
    Args:
        file_path: Path to the PDF book file.
        config: Optional ChunkConfig. Uses defaults if not provided.
        save_parsed_path: Optional path to save parsed pages as JSON (for caching).
        load_parsed_path: Optional path to load parsed pages from JSON (skip parsing).
        toc_path: Optional path to JSON file with table of contents.
            Format: {"Part 1": ["Chapter 1", "Chapter 2"], "Part 2": {...}}
        
    Returns:
        Tuple of (child_documents, parent_documents) as LangChain Documents.
        
    Example:
        >>> config = ChunkConfig(child_chunk_size=400, parent_chunk_size=2000)
        >>> # With TOC for chapter assignment
        >>> child_docs, parent_docs = process_book(
        ...     "data/books/book.pdf", 
        ...     config,
        ...     save_parsed_path="data/parsed/book.json",
        ...     toc_path="data/toc/book_toc.json"
        ... )
    """
    if config is None:
        config = ChunkConfig()
    
    # Extract book title from filename
    book_title = extract_book_title(file_path)
    
    # Step 1: Parse PDF or load from cache
    if load_parsed_path and Path(load_parsed_path).exists():
        print(f"Loading parsed pages from cache: {load_parsed_path}")
        parsed_pages = load_parsed_pages(load_parsed_path)
    else:
        print(f"Parsing PDF: {file_path}")
        parsed_pages = parse_pdf(file_path, language=config.language)
        print(f"Parsed {len(parsed_pages)} pages")
        
        # Save parsed pages if path provided
        if save_parsed_path:
            save_parsed_pages(parsed_pages, save_parsed_path)
    
    # Step 2: Assign chapters based on TOC (if provided)
    if toc_path and Path(toc_path).exists():
        print(f"Loading TOC from: {toc_path}")
        toc = load_toc(toc_path)
        parsed_pages = assign_chapters_to_pages(parsed_pages, toc)
        
        # Re-save with chapter info if save path provided
        if save_parsed_path:
            save_parsed_pages(parsed_pages, save_parsed_path)
    
    # Step 3: Chunk documents
    print("Chunking documents...")
    child_docs, parent_docs = chunk_documents(
        parsed_pages=parsed_pages,
        config=config,
        source_path=file_path,
        book_title=book_title,
    )
    print(f"Created {len(child_docs)} child chunks, {len(parent_docs)} parent chunks")
    
    return child_docs, parent_docs


def process_books_directory(
    directory_path: str,
    config: ChunkConfig | None = None,
    parsed_cache_dir: Optional[str] = None,
    toc_dir: Optional[str] = None,
) -> Tuple[List[Document], List[Document]]:
    """Process all PDF books in a directory.
    
    Args:
        directory_path: Path to directory containing PDF files.
        config: Optional ChunkConfig. Uses defaults if not provided.
        parsed_cache_dir: Optional directory to cache/load parsed pages.
            If provided, saves parsed pages as JSON files in this directory.
            On subsequent runs, loads from cache instead of re-parsing.
        toc_dir: Optional directory containing TOC JSON files.
            Files should be named <book_stem>_toc.json (e.g., "mybook_toc.json" for "mybook.pdf").
        
    Returns:
        Tuple of (all_child_documents, all_parent_documents) combined from all books.
    """
    if config is None:
        config = ChunkConfig()
    
    directory = Path(directory_path)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    pdf_files = list(directory.glob("*.pdf")) + list(directory.glob("*.PDF"))
    
    if not pdf_files:
        print(f"No PDF files found in {directory_path}")
        return [], []
    
    all_child_docs = []
    all_parent_docs = []
    
    for pdf_file in pdf_files:
        print(f"\n{'='*50}")
        print(f"Processing: {pdf_file.name}")
        print('='*50)
        
        # Determine cache paths if cache directory is provided
        cache_path = None
        if parsed_cache_dir:
            cache_path = str(Path(parsed_cache_dir) / f"{pdf_file.stem}.json")
        
        # Determine TOC path if TOC directory is provided
        toc_path = None
        if toc_dir:
            toc_file = Path(toc_dir) / f"{pdf_file.stem}_toc.json"
            if toc_file.exists():
                toc_path = str(toc_file)
                print(f"Found TOC file: {toc_file.name}")
        
        try:
            child_docs, parent_docs = process_book(
                str(pdf_file), 
                config,
                save_parsed_path=cache_path,
                load_parsed_path=cache_path,
                toc_path=toc_path,
            )
            all_child_docs.extend(child_docs)
            all_parent_docs.extend(parent_docs)
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")
            continue
    
    print(f"\n{'='*50}")
    print(f"Total: {len(all_child_docs)} child chunks, {len(all_parent_docs)} parent chunks")
    
    return all_child_docs, all_parent_docs
