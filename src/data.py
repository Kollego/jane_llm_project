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
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_parse import LlamaParse


# Type aliases for new TOC structure
@dataclass
class TOCItem:
    """Single item in the table of contents."""
    title: str
    page: Optional[int]
    children: List["TOCItem"] = field(default_factory=list)


@dataclass
class TOCData:
    """Full table of contents with metadata."""
    author: str
    title: str
    offset: Optional[int] = None  # Page offset. None = auto-detect by title search
    div: int = 1  # Divisor. Formula: page_in_file = (page_in_toc // div) + offset
    items: List[TOCItem] = field(default_factory=list)

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

def _parse_toc_item(data: Dict[str, Any]) -> TOCItem:
    """Parse a single TOC item from JSON."""
    children = [_parse_toc_item(child) for child in data.get("children", [])]
    return TOCItem(
        title=data["title"],
        page=data.get("page"),
        children=children,
    )


def load_toc(toc_path: str) -> TOCData:
    """Load table of contents from a JSON file.
    
    Expected format:
    {
        "meta": {
            "author": "Author Name",
            "title": "Book Title"
        },
        "toc": [
            {
                "title": "Chapter 1",
                "page": 10,
                "children": [...]
            }
        ]
    }
    
    Args:
        toc_path: Path to the JSON file with TOC.
        
    Returns:
        TOCData object with metadata and items.
    """
    toc_path = Path(toc_path)
    
    if not toc_path.exists():
        raise FileNotFoundError(f"TOC file not found: {toc_path}")
    
    with open(toc_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    meta = data.get("meta", {})
    items = [_parse_toc_item(item) for item in data.get("toc", [])]
    
    return TOCData(
        author=meta.get("author", ""),
        title=meta.get("title", ""),
        offset=meta.get("offset"),  # None = auto-detect
        div=meta.get("div", 1),
        items=items,
    )


def flatten_toc_items(
    items: List[TOCItem],
    parent_path: List[str] = None,
) -> List[Tuple[List[str], Optional[int]]]:
    """Flatten TOC items into a list of (path, page) tuples.
    
    Each path is a list representing the hierarchy, e.g.:
    (["Part 1", "Chapter 2", "Section 1"], 42)
    
    Args:
        items: List of TOCItem objects.
        parent_path: Current path in the hierarchy (used for recursion).
        
    Returns:
        List of (path, page) tuples in order of appearance.
    """
    if parent_path is None:
        parent_path = []
    
    result = []
    
    for item in items:
        current_path = parent_path + [item.title]
        result.append((current_path, item.page))
        
        if item.children:
            result.extend(flatten_toc_items(item.children, current_path))
    
    return result


def get_all_titles_from_toc(toc: TOCData) -> List[str]:
    """Extract all unique titles from TOC (at all levels).
    
    Args:
        toc: TOCData object.
        
    Returns:
        List of all unique titles.
    """
    titles = []
    
    def extract_titles(items: List[TOCItem]):
        for item in items:
            titles.append(item.title)
            if item.children:
                extract_titles(item.children)
    
    extract_titles(toc.items)
    return titles


def normalize_text_for_search(text: str) -> str:
    """Normalize text for fuzzy matching."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:\-—–]', '', text)
    return text.strip()


def find_title_in_text(title: str, text: str) -> bool:
    """Check if a title appears in text (fuzzy matching)."""
    norm_title = normalize_text_for_search(title)
    norm_text = normalize_text_for_search(text)
    
    if norm_title in norm_text:
        return True
    
    # Try matching first few significant words
    title_words = norm_title.split()
    if len(title_words) >= 2:
        partial_title = ' '.join(title_words[:3])
        if partial_title in norm_text:
            return True
    
    return False


def _find_chapters_by_title_search(
    parsed_pages: List[ParsedPage],
    toc_items: List[Tuple[List[str], Optional[int]]],
    search_range: int = 15,
    div: int = 1,
) -> List[Tuple[int, List[str]]]:
    """Find chapter boundaries by searching for titles in text.
    
    Used when offset is not specified in TOC.
    Searches for each title within ±search_range pages of the estimated TOC page number.
    
    Formula for estimated page: page_in_file ≈ page_in_toc // div
    
    Args:
        parsed_pages: List of parsed pages.
        toc_items: Flattened TOC items with (path, page) tuples.
        search_range: Number of pages to search around estimated page (default 15).
        div: Divisor for page calculation (default 1).
        
    Returns:
        List of (pdf_page, path) tuples for found chapters.
    """
    # Build mapping: title -> (path, toc_page)
    title_to_info: Dict[str, Tuple[List[str], Optional[int]]] = {}
    for path, toc_page in toc_items:
        title = path[-1]
        if title not in title_to_info:
            title_to_info[title] = (path, toc_page)
    
    all_titles = list(title_to_info.keys())
    
    # Create page lookup by number for faster access
    page_by_number: Dict[int, ParsedPage] = {p.page_number: p for p in parsed_pages}
    max_page = max(page_by_number.keys()) if page_by_number else 0
    
    # Detect TOC pages (pages with many titles) to skip them
    toc_pages: set = set()
    for page in parsed_pages:
        titles_on_page = sum(1 for t in all_titles if find_title_in_text(t, page.content))
        if titles_on_page > 3:
            toc_pages.add(page.page_number)
    
    if toc_pages:
        print(f"Detected TOC pages (skipping): {sorted(toc_pages)}")
    
    # Find chapter boundaries by searching for titles
    chapter_starts: List[Tuple[int, List[str]]] = []
    not_found = []
    
    for title in all_titles:
        path, toc_page = title_to_info[title]
        found = False
        
        if toc_page is not None:
            # Estimate PDF page: page_in_file = page_in_toc // div
            estimated_page = toc_page // div
            # Search within ±search_range pages of estimated page
            start_page = max(1, estimated_page - search_range)
            end_page = min(max_page, estimated_page + search_range)
            
            for page_num in range(start_page, end_page + 1):
                if page_num in toc_pages:
                    continue
                if page_num not in page_by_number:
                    continue
                
                page = page_by_number[page_num]
                if find_title_in_text(title, page.content):
                    chapter_starts.append((page.page_number, path))
                    found = True
                    break
        else:
            # No page number in TOC - search all pages
            for page in parsed_pages:
                if page.page_number in toc_pages:
                    continue
                if find_title_in_text(title, page.content):
                    chapter_starts.append((page.page_number, path))
                    found = True
                    break
        
        if not found:
            not_found.append(title)
    
    if not_found:
        print(f"WARNING: {len(not_found)} chapter(s) not found in text")
    
    return chapter_starts


def assign_chapters_to_pages(
    parsed_pages: List[ParsedPage],
    toc: TOCData,
) -> List[ParsedPage]:
    """Assign chapter information to parsed pages based on TOC.
    
    Uses page numbers from TOC to determine chapter boundaries.
    If offset is set in toc.offset, uses it to calculate actual PDF pages.
    If offset is None, searches for chapter titles in text to find boundaries.
    
    Args:
        parsed_pages: List of parsed pages.
        toc: TOCData object with metadata, offset, and items.
        
    Returns:
        List of parsed pages with chapter field filled in.
    """
    # Flatten TOC to get all (path, page) pairs
    flattened = flatten_toc_items(toc.items)
    
    if not flattened:
        print("WARNING: Empty TOC, no chapters assigned")
        return parsed_pages
    
    print(f"TOC has {len(flattened)} entries")
    
    # Build chapter boundaries
    chapter_starts: List[Tuple[int, List[str]]] = []
    
    if toc.offset is None:
        # No offset specified - search for titles in text
        print("No offset specified, searching for chapter titles in text...")
        chapter_starts = _find_chapters_by_title_search(parsed_pages, flattened, div=toc.div)
    else:
        # Use offset and div to calculate PDF pages: page_in_file = (page_in_toc // div) + offset
        print(f"Using page offset: {toc.offset}, div: {toc.div}")
        for path, toc_page in flattened:
            if toc_page is not None:
                actual_page = (toc_page // toc.div) + toc.offset
                chapter_starts.append((actual_page, path))
    
    # Sort by page number
    chapter_starts.sort(key=lambda x: x[0])
    
    # Assign chapters to pages based on boundaries
    updated_pages = []
    chapter_idx = 0
    current_chapter = None
    
    for page in parsed_pages:
        # Check if we've reached the next chapter boundary
        while (chapter_idx < len(chapter_starts) and 
               chapter_starts[chapter_idx][0] <= page.page_number):
            current_chapter = chapter_starts[chapter_idx][1]
            chapter_idx += 1
        
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
    print(f"Assigned {chapters_found} chapters to {pages_with_chapters}/{len(updated_pages)} pages")
    
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


def parse_and_save(
    file_path: str,
    output_path: str,
    language: str = "ru",
) -> List[ParsedPage]:
    """Parse a PDF file and save the result to JSON.
    
    This is the main entry point for PDF parsing. It combines parse_pdf()
    and save_parsed_pages() into a single convenient function.
    
    Args:
        file_path: Path to the PDF file.
        output_path: Path to save the parsed JSON file.
        language: Language of the document for better OCR results.
        
    Returns:
        List of ParsedPage objects.
        
    Example:
        >>> parsed_pages = parse_and_save(
        ...     "data/books/mybook.pdf",
        ...     "data/cache/mybook_parsed.json"
        ... )
    """
    print(f"Parsing PDF: {file_path}")
    parsed_pages = parse_pdf(file_path, language=language)
    print(f"Parsed {len(parsed_pages)} pages")
    
    save_parsed_pages(parsed_pages, output_path)
    
    return parsed_pages


def chunk_documents(
    parsed_pages: List[ParsedPage],
    config: ChunkConfig,
    source_path: str,
    book_title: str,
    book_author: Optional[str] = None,
    page_offset: int = 0,
    page_div: int = 1,
) -> Tuple[List[Document], List[Document]]:
    """Chunk parsed pages using Parent Document Retriever strategy.
    
    This function implements a two-level chunking strategy:
    - Parent chunks: One per chapter (full chapter text for context)
    - Child chunks: Small chunks (e.g., 400 chars) optimized for vector search
    
    Each child chunk references its parent chunk via parent_id in metadata.
    
    Book page calculation: toc_page = (pdf_page - offset) * div
    
    Args:
        parsed_pages: List of ParsedPage objects from parse_pdf().
        config: ChunkConfig with size/overlap settings.
        source_path: Original file path (for metadata).
        book_title: Title of the book (for metadata).
        book_author: Author of the book (for metadata).
        page_offset: Offset for page calculation.
        page_div: Divisor for page calculation.
        
    Returns:
        Tuple of (child_documents, parent_documents).
    """
    # Group pages by chapter
    # chapter_key -> list of pages
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
        
        # Get PDF page range for this chapter
        pdf_page_numbers = [page.page_number for page in chapter_pages]
        pdf_page_range = sorted(pdf_page_numbers)
        
        # Calculate book page numbers (as they appear in TOC)
        # Formula: toc_page = (pdf_page - offset) * div
        book_page_numbers = [(p - page_offset) * page_div for p in pdf_page_range]
        book_pages_str = f"{book_page_numbers[0]}-{book_page_numbers[-1]}" if len(book_page_numbers) > 1 else str(book_page_numbers[0])
        
        # Generate unique ID for parent chunk
        parent_id = str(uuid.uuid4())
        
        # Build parent metadata
        parent_metadata = {
            "source": source_path,
            "book_title": book_title,
            "page_number": pdf_page_range[0],  # First PDF page of chapter
            "page_range": pdf_page_range,
            "book_pages": book_page_numbers,  # Book page numbers
            "book_pages_str": book_pages_str,  # "10-15" format
            "chunk_type": "parent",
            "chunk_id": parent_id,
        }
        
        # Add author if available
        if book_author:
            parent_metadata["book_author"] = book_author
        
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
                "page_number": pdf_page_range[0],
                "page_range": pdf_page_range,
                "book_pages": book_page_numbers,
                "book_pages_str": book_pages_str,
                "chunk_type": "child",
                "parent_id": parent_id,
                "chunk_id": str(uuid.uuid4()),
            }
            
            # Add author if available
            if book_author:
                child_metadata["book_author"] = book_author
            
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
    parsed_path: str,
    config: ChunkConfig | None = None,
    toc_path: Optional[str] = None,
) -> Tuple[List[Document], List[Document]]:
    """Process a pre-parsed book through the RAG pipeline.
    
    This function works with already parsed PDFs (use parse_and_save() first).
    It orchestrates:
    1. Loading parsed pages from cache
    2. Chapter assignment based on TOC (if provided)
    3. Chunking with Parent Document Retriever strategy
    4. Metadata enrichment (author, title from TOC)
    
    Args:
        file_path: Path to the original PDF book file (for metadata).
        parsed_path: Path to JSON file with parsed pages (from parse_and_save).
        config: Optional ChunkConfig. Uses defaults if not provided.
        toc_path: Optional path to JSON file with table of contents.
        
    Returns:
        Tuple of (child_documents, parent_documents) as LangChain Documents.
        
    Example:
        >>> # First parse the PDF (separately)
        >>> parse_and_save("data/books/book.pdf", "data/cache/book_parsed.json")
        >>> 
        >>> # Then process the parsed data
        >>> config = ChunkConfig(child_chunk_size=400, parent_chunk_size=2000)
        >>> child_docs, parent_docs = process_book(
        ...     "data/books/book.pdf",
        ...     "data/cache/book_parsed.json",
        ...     config,
        ...     toc_path="data/toc/book_toc.json"
        ... )
    """
    if config is None:
        config = ChunkConfig()
    
    # Default: extract book title from filename
    book_title = extract_book_title(file_path)
    book_author = None
    
    # Step 1: Load parsed pages from cache
    print(f"Loading parsed pages from: {parsed_path}")
    parsed_pages = load_parsed_pages(parsed_path)
    
    # Step 2: Assign chapters based on TOC (if provided)
    toc = None
    page_offset = 0
    page_div = 1
    if toc_path and Path(toc_path).exists():
        print(f"Loading TOC from: {toc_path}")
        toc = load_toc(toc_path)
        
        # Override title and author from TOC metadata
        if toc.title:
            book_title = toc.title
            print(f"Book title from TOC: {book_title}")
        if toc.author:
            book_author = toc.author
            print(f"Book author from TOC: {book_author}")
        
        page_offset = toc.offset if toc.offset is not None else 0
        page_div = toc.div
        if toc.offset is not None:
            print(f"Page offset from TOC: {page_offset}, div: {page_div}")
        else:
            print("No offset in TOC, book page numbers will equal PDF page numbers")
        
        parsed_pages = assign_chapters_to_pages(parsed_pages, toc)
    
    # Step 3: Chunk documents
    print("Chunking documents...")
    child_docs, parent_docs = chunk_documents(
        parsed_pages=parsed_pages,
        config=config,
        source_path=file_path,
        book_title=book_title,
        book_author=book_author,
        page_offset=page_offset,
        page_div=page_div,
    )
    print(f"Created {len(child_docs)} child chunks, {len(parent_docs)} parent chunks")
    
    return child_docs, parent_docs


def process_books_directory(
    directory_path: str,
    parsed_cache_dir: str,
    config: ChunkConfig | None = None,
    toc_dir: Optional[str] = None,
) -> Tuple[List[Document], List[Document]]:
    """Process all pre-parsed books in a directory.
    
    Note: PDFs must be parsed first using parse_and_save().
    This function only processes books that have cached parsed JSON files.
    
    Args:
        directory_path: Path to directory containing PDF files.
        parsed_cache_dir: Directory with parsed JSON files (required).
            Files should be named <book_stem>_parsed.json.
        config: Optional ChunkConfig. Uses defaults if not provided.
        toc_dir: Optional directory containing TOC JSON files.
            Files should be named <book_stem>_toc.json.
        
    Returns:
        Tuple of (all_child_documents, all_parent_documents) combined from all books.
    """
    if config is None:
        config = ChunkConfig()
    
    directory = Path(directory_path)
    cache_dir = Path(parsed_cache_dir)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {parsed_cache_dir}")
    
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
        
        # Determine cache path
        cache_path = cache_dir / f"{pdf_file.stem}_parsed.json"
        if not cache_path.exists():
            print(f"Skipping {pdf_file.name}: parsed cache not found at {cache_path}")
            print(f"Run parse_and_save() first to parse this PDF.")
            continue
        
        # Determine TOC path if TOC directory is provided
        toc_path = None
        if toc_dir:
            toc_file = Path(toc_dir) / f"{pdf_file.stem}_toc.json"
            if toc_file.exists():
                toc_path = str(toc_file)
                print(f"Found TOC file: {toc_file.name}")
        
        try:
            child_docs, parent_docs = process_book(
                file_path=str(pdf_file),
                parsed_path=str(cache_path),
                config=config,
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
