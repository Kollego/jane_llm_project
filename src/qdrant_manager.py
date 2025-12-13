"""
Qdrant vector store manager.

Provides unified interface for local and server-based Qdrant operations.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)


@dataclass
class QdrantConfig:
    """Configuration for Qdrant connection.
    
    Supports both local (path-based) and server (host/port) modes.
    
    Attributes:
        path: Path to local Qdrant storage (for local mode)
        host: Server host (for server mode)
        port: Server port (for server mode)
        collection_name: Name of the collection
        vector_size: Dimension of vectors
    """
    path: Optional[str] = None
    host: Optional[str] = None
    port: int = 6333
    collection_name: str = "chunks"
    vector_size: int = 256  # YC embeddings size
    
    def __post_init__(self):
        # Load from env if not specified
        if not self.path and not self.host:
            self.path = os.getenv("QDRANT_PATH", "./data/qdrant_local")
        if self.host is None and self.path is None:
            self.host = os.getenv("QDRANT_HOST")
        if self.host:
            self.port = int(os.getenv("QDRANT_PORT", self.port))


@dataclass
class SearchResult:
    """Single search result."""
    text: str
    score: float
    book_title: str
    book_author: Optional[str] = None
    chapter: Optional[str] = None
    book_pages_str: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class QdrantManager:
    """Manager for Qdrant vector store operations.
    
    Supports both local (path-based) and server (host/port) modes.
    
    Example:
        # Local mode
        >>> manager = QdrantManager(QdrantConfig(path="./data/qdrant_local"))
        
        # Server mode
        >>> manager = QdrantManager(QdrantConfig(host="localhost", port=6333))
        
        # Search
        >>> results = manager.search(query_vector, k=5)
    """
    
    def __init__(self, config: Optional[QdrantConfig] = None):
        """Initialize the manager.
        
        Args:
            config: Qdrant configuration. Uses defaults if not provided.
        """
        self.config = config or QdrantConfig()
        self._client: Optional[QdrantClient] = None
    
    @property
    def client(self) -> QdrantClient:
        """Get or create Qdrant client."""
        if self._client is None:
            if self.config.path:
                self._client = QdrantClient(path=self.config.path)
            else:
                self._client = QdrantClient(
                    host=self.config.host,
                    port=self.config.port,
                )
        return self._client
    
    def exists(self) -> bool:
        """Check if the Qdrant storage exists."""
        if self.config.path:
            return Path(self.config.path).exists()
        # For server mode, try to connect
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
    
    def collection_exists(self) -> bool:
        """Check if the collection exists."""
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            return self.config.collection_name in collections
        except Exception:
            return False
    
    def create_collection(self, recreate: bool = False) -> None:
        """Create collection if it doesn't exist.
        
        Args:
            recreate: Delete existing collection before creating.
        """
        if recreate and self.collection_exists():
            self.client.delete_collection(self.config.collection_name)
        
        if not self.collection_exists():
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.vector_size,
                    distance=Distance.COSINE,
                ),
            )
    
    def add_points(
        self,
        points: List[PointStruct],
        batch_size: int = 100,
    ) -> int:
        """Add points to the collection.
        
        Args:
            points: List of points to add.
            batch_size: Batch size for uploading.
            
        Returns:
            Number of points added.
        """
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=batch,
                wait=True,
            )
        return len(points)
    
    def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filter_book: Optional[str] = None,
        filter_author: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector.
            k: Number of results to return.
            filter_book: Filter by book title.
            filter_author: Filter by author.
            
        Returns:
            List of SearchResult objects.
        """
        # Build filter
        filter_conditions = []
        if filter_book:
            filter_conditions.append(
                FieldCondition(key="book_title", match=MatchValue(value=filter_book))
            )
        if filter_author:
            filter_conditions.append(
                FieldCondition(key="book_author", match=MatchValue(value=filter_author))
            )
        
        query_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        # Search
        results = self.client.query_points(
            collection_name=self.config.collection_name,
            query=query_vector,
            limit=k,
            query_filter=query_filter,
        )
        
        # Convert to SearchResult
        search_results = []
        for point in results.points:
            payload = point.payload
            search_results.append(SearchResult(
                text=payload.get("text", ""),
                score=point.score,
                book_title=payload.get("book_title", "Unknown"),
                book_author=payload.get("book_author"),
                chapter=payload.get("chapter"),
                book_pages_str=payload.get("book_pages_str"),
                metadata=payload,
            ))
        
        return search_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics.
        
        Returns:
            Dictionary with collection stats.
        """
        info = self.client.get_collection(self.config.collection_name)
        
        # Get books info
        result = self.client.scroll(
            collection_name=self.config.collection_name,
            limit=10000,
            with_payload=["book_title", "book_author"],
        )
        
        books = {}
        for point in result[0]:
            book = point.payload.get("book_title", "Unknown")
            author = point.payload.get("book_author", "")
            
            if book not in books:
                books[book] = {"count": 0, "author": author}
            books[book]["count"] += 1
        
        return {
            "collection_name": self.config.collection_name,
            "points_count": info.points_count,
            "vector_size": info.config.params.vectors.size,
            "distance": str(info.config.params.vectors.distance),
            "books": books,
        }
    
    def list_books(self) -> List[str]:
        """Get list of all books in the collection.
        
        Returns:
            Sorted list of book titles.
        """
        result = self.client.scroll(
            collection_name=self.config.collection_name,
            limit=10000,
            with_payload=["book_title"],
        )
        
        books = set()
        for point in result[0]:
            if point.payload.get("book_title"):
                books.add(point.payload["book_title"])
        
        return sorted(books)
    
    def count(self) -> int:
        """Get total number of points in the collection."""
        result = self.client.count(
            collection_name=self.config.collection_name,
            exact=True,
        )
        return result.count
