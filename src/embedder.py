"""
Yandex Cloud Embeddings client.

Provides embeddings using Yandex Cloud Foundation Models API.
"""

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Примерный лимит символов (2048 токенов ≈ 6000 символов для русского текста)
MAX_CHARS_PER_REQUEST = 5000


class YCEmbedder:
    """
    Client for getting embeddings via Yandex Cloud Foundation Models.
    
    Attributes:
        api_key: API key for Yandex Cloud
        iam_token: IAM token (alternative to API key)
        folder_id: Yandex Cloud folder ID
        variant: Embedding model variant ("text-search-doc" or "text-search-query")
        endpoint: API endpoint URL
        timeout: Request timeout in seconds
    
    Environment variables:
        YC_API_KEY: API key
        YC_IAM_TOKEN: IAM token
        YC_FOLDER_ID: Folder ID
    
    Example:
        >>> embedder = YCEmbedder()
        >>> vectors = embedder.embed_texts(["Hello world", "Привет мир"])
        >>> print(vectors.shape)  # (2, 256)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        iam_token: Optional[str] = None,
        folder_id: Optional[str] = None,
        variant: str = "text-search-doc",
        endpoint: str = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding",
        timeout: int = 30,
        requests_per_second: float = 8.0,  # YC limit is 10 RPS, use 8 for safety
        max_retries: int = 5,
        max_chars: int = MAX_CHARS_PER_REQUEST,
    ) -> None:
        """Initialize the embedder.
        
        Args:
            api_key: API key (or set YC_API_KEY env var)
            iam_token: IAM token (or set YC_IAM_TOKEN env var)
            folder_id: Folder ID (or set YC_FOLDER_ID env var)
            variant: Model variant ("text-search-doc" for documents, "text-search-query" for queries)
            endpoint: API endpoint URL
            timeout: Request timeout in seconds
            requests_per_second: Rate limit (default 8, YC allows 10)
            max_retries: Max retries on rate limit errors
            max_chars: Maximum characters per embedding request
        """
        self.api_key = api_key or os.getenv("YC_API_KEY")
        self.iam_token = iam_token or os.getenv("YC_IAM_TOKEN")
        self.folder_id = folder_id or os.getenv("YC_FOLDER_ID")
        self.variant = variant
        self.endpoint = endpoint
        self.timeout = timeout
        self.requests_per_second = requests_per_second
        self.max_retries = max_retries
        self.max_chars = max_chars
        self._min_interval = 1.0 / requests_per_second
        self._last_request_time = 0.0
        
        if not self.api_key and not self.iam_token:
            raise ValueError(
                "Provide either api_key or iam_token for Yandex Cloud. "
                "Set YC_API_KEY or YC_IAM_TOKEN environment variable."
            )
        if not self.folder_id:
            raise ValueError(
                "folder_id is required for Yandex Cloud embeddings. "
                "Set YC_FOLDER_ID environment variable."
            )

    def _headers(self) -> Dict[str, str]:
        """Create request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Api-Key {self.api_key}"
        elif self.iam_token:
            headers["Authorization"] = f"Bearer {self.iam_token}"
        return headers

    def _wait_for_rate_limit(self) -> None:
        """Wait if needed to respect rate limit."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def _chunk_text(self, text: str, chunk_size: int = None, overlap: int = 500) -> List[str]:
        """
        Разбивает текст на чанки с перекрытием.
        
        Args:
            text: Текст для разбиения
            chunk_size: Размер чанка в символах (по умолчанию self.max_chars)
            overlap: Размер перекрытия между чанками
            
        Returns:
            Список чанков
        """
        if chunk_size is None:
            chunk_size = self.max_chars
            
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Пытаемся разорвать по границе предложения или абзаца
            if end < len(text):
                # Ищем конец предложения
                for sep in ['\n\n', '\n', '. ', '! ', '? ', '; ']:
                    pos = text.rfind(sep, start + chunk_size // 2, end)
                    if pos != -1:
                        end = pos + len(sep)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Следующий чанк начинается с перекрытием
            start = end - overlap
            if start < 0:
                start = 0
            if start >= len(text):
                break
                
        return chunks

    def _post_text(self, model_uri: str, text: str) -> List[float]:
        """
        Send text to get embeddings with rate limiting and retry.
        
        Args:
            model_uri: Model URI
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            RuntimeError: On API error after all retries
        """
        payload = {
            "modelUri": model_uri,
            "text": text,
        }
        
        for attempt in range(self.max_retries):
            # Rate limiting
            self._wait_for_rate_limit()
            
            resp = requests.post(
                self.endpoint,
                headers=self._headers(),
                data=json.dumps(payload),
                timeout=self.timeout,
            )
            
            # Handle rate limit (429)
            if resp.status_code == 429:
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8, 16 sec
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"YC embeddings error: {resp.status_code} {resp.text}")
            
            if resp.status_code != 200:
                raise RuntimeError(f"YC embeddings error: {resp.status_code} {resp.text}")
            
            data = resp.json()
            vec = None
            
            if isinstance(data, dict):
                emb = data.get("embedding")
                if isinstance(emb, dict) and "vector" in emb:
                    vec = emb["vector"]
                elif isinstance(emb, list):
                    vec = emb
                elif "vector" in data:
                    vec = data["vector"]
            
            if vec is None or not isinstance(vec, list):
                raise RuntimeError("Unexpected YC embeddings response format.")
            
            return [float(x) for x in vec]
        
        raise RuntimeError("Max retries exceeded")

    def _embed_long_text(self, text: str, model_uri: str) -> List[float]:
        """
        Эмбеддинг длинного текста через чанкинг и усреднение.
        
        Args:
            text: Длинный текст
            model_uri: URI модели
            
        Returns:
            Усредненный вектор эмбеддинга
        """
        chunks = self._chunk_text(text)
        
        if len(chunks) == 1:
            return self._post_text(model_uri, chunks[0])
        
        # Получаем эмбеддинги для всех чанков
        embeddings = []
        for chunk in chunks:
            vec = self._post_text(model_uri, chunk)
            embeddings.append(vec)
        
        # Усредняем эмбеддинги
        arr = np.array(embeddings, dtype=np.float32)
        mean_vec = np.mean(arr, axis=0)
        
        # Нормализуем результат
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        
        return mean_vec.tolist()

    def embed_texts(
        self,
        texts: List[str],
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            verbose: Whether to show progress bar
            
        Returns:
            NumPy array of embeddings with shape (len(texts), embedding_dim)
            
        Raises:
            RuntimeError: On mismatch between texts and embeddings count
        """
        vectors: List[List[float]] = []
        model_uri = f"emb://{self.folder_id}/{self.variant}/latest"
        
        # The API accepts only single 'text' per request
        iterator = texts
        if verbose:
            try:
                from tqdm import tqdm
                iterator = tqdm(texts, desc="Embedding")
            except ImportError:
                pass
        
        for text in iterator:
            if len(text) > self.max_chars:
                vec = self._embed_long_text(text, model_uri)
            else:
                vec = self._post_text(model_uri, text)
            vectors.append(vec)
        
        arr = np.array(vectors, dtype=np.float32)
        
        if arr.shape[0] != len(texts):
            raise RuntimeError("Mismatch between number of texts and embeddings returned.")
        
        return arr

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query.
        
        Uses "text-search-query" variant for better search results.
        Automatically handles long queries via chunking.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector as list of floats
        """
        model_uri = f"emb://{self.folder_id}/text-search-query/latest"
        
        if len(query) > self.max_chars:
            return self._embed_long_text(query, model_uri)
        
        return self._post_text(model_uri, query)

    def embed_document(self, text: str) -> List[float]:
        """
        Embed a single document.
        
        Uses "text-search-doc" variant.
        Automatically handles long documents via chunking.
        
        Args:
            text: Document text
            
        Returns:
            Embedding vector as list of floats
        """
        model_uri = f"emb://{self.folder_id}/text-search-doc/latest"
        
        if len(text) > self.max_chars:
            return self._embed_long_text(text, model_uri)
        
        return self._post_text(model_uri, text)
