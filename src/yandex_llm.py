"""
Yandex Cloud LLM client.

Provides LangChain-compatible interface for Yandex Cloud Foundation Models.
"""

import json
import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class AIMessage:
    """AI response message (LangChain-compatible)."""
    content: str
    
    def __str__(self) -> str:
        return self.content


class YandexCloudModel:
    """
    Yandex Cloud LLM client with LangChain-compatible interface.
    
    Uses direct HTTP requests to Yandex Cloud API.
    
    Environment variables:
        YC_API_KEY: API key for Yandex Cloud
        YC_FOLDER_ID: Yandex Cloud folder ID
    
    Example:
        >>> llm = YandexCloudModel(model="gemma-3-27b-it/latest")
        >>> response = llm.invoke([
        ...     {"role": "system", "content": "You are a helpful assistant"},
        ...     {"role": "user", "content": "Hello!"}
        ... ])
        >>> print(response.content)
    """
    
    def __init__(
        self,
        model: str = "gemma-3-27b-it/latest",
        api_key: Optional[str] = None,
        folder_id: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        endpoint: str = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 30.0,
        timeout: int = 120,
    ):
        """
        Initialize Yandex Cloud Model client.
        
        Args:
            model: Model name (e.g., "gpt-oss-120b/latest", "gemma-3-27b-it/latest")
            api_key: Yandex Cloud API key (or set YC_API_KEY env var)
            folder_id: Yandex Cloud folder ID (or set YC_FOLDER_ID env var)
            temperature: Generation temperature
            max_tokens: Maximum tokens in response
            endpoint: API endpoint URL
            max_retries: Max retry attempts
            retry_base_delay: Base delay for retries (seconds)
            retry_max_delay: Max delay for retries (seconds)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("YC_API_KEY")
        self.folder_id = folder_id or os.getenv("YC_FOLDER_ID")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.endpoint = endpoint
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.timeout = timeout
        
        if not self.api_key:
            raise ValueError(
                "API key is required. Set YC_API_KEY environment variable."
            )
        if not self.folder_id:
            raise ValueError(
                "Folder ID is required. Set YC_FOLDER_ID environment variable."
            )
    
    @property
    def model_uri(self) -> str:
        """Get full model URI."""
        return f"gpt://{self.folder_id}/{self.model}"
    
    def _headers(self) -> Dict[str, str]:
        """Create request headers."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}",
            "x-folder-id": self.folder_id,
        }
    
    def _is_transient_error(self, status_code: int, message: str) -> bool:
        """Check if error is transient and can be retried."""
        if status_code in (429, 500, 502, 503, 504):
            return True
        
        transient_keywords = [
            "temporarily unavailable",
            "server_overloaded", 
            "timeout",
            "rate limit",
        ]
        return any(keyword in message.lower() for keyword in transient_keywords)
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format messages for Yandex API."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            # Yandex uses "assistant" not "ai"
            if role == "ai":
                role = "assistant"
            formatted.append({
                "role": role,
                "text": msg.get("content", msg.get("text", "")),
            })
        return formatted
    
    def invoke(
        self,
        messages: Union[List[Dict[str, str]], str],
        **kwargs,
    ) -> AIMessage:
        """
        Invoke the LLM with messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content',
                     or a single string (treated as user message)
            **kwargs: Override temperature, max_tokens, etc.
        
        Returns:
            AIMessage with response content
        """
        # Handle string input
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        # Convert LangChain message objects to dicts if needed
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, "content"):
                # LangChain message object
                role = "user"
                if hasattr(msg, "role"):
                    role = msg.role
                elif msg.__class__.__name__ == "SystemMessage":
                    role = "system"
                elif msg.__class__.__name__ == "AIMessage":
                    role = "assistant"
                elif msg.__class__.__name__ == "HumanMessage":
                    role = "user"
                formatted_messages.append({"role": role, "content": msg.content})
            else:
                formatted_messages.append(msg)
        
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        # Build request payload
        payload = {
            "modelUri": self.model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": str(max_tokens),
            },
            "messages": self._format_messages(formatted_messages),
        }
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.endpoint,
                    headers=self._headers(),
                    data=json.dumps(payload),
                    timeout=self.timeout,
                )
                
                if response.status_code != 200:
                    error_msg = response.text
                    if self._is_transient_error(response.status_code, error_msg):
                        if attempt < self.max_retries - 1:
                            delay = min(
                                self.retry_max_delay,
                                self.retry_base_delay * (2 ** attempt)
                            )
                            delay *= 1.0 + 0.1 * random.random()
                            time.sleep(delay)
                            continue
                    raise RuntimeError(f"Yandex API error {response.status_code}: {error_msg}")
                
                data = response.json()
                
                # Extract content from response
                content = ""
                if "result" in data:
                    result = data["result"]
                    if "alternatives" in result and result["alternatives"]:
                        message = result["alternatives"][0].get("message", {})
                        content = message.get("text", "")
                    elif "message" in result:
                        content = result["message"].get("text", "")
                
                return AIMessage(content=content)
                
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = min(
                        self.retry_max_delay,
                        self.retry_base_delay * (2 ** attempt)
                    )
                    delay *= 1.0 + 0.1 * random.random()
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"Yandex API request error: {e}") from e
        
        raise RuntimeError(f"Max retries exceeded. Last error: {last_error}")
    
    def __call__(self, messages, **kwargs) -> AIMessage:
        """Allow calling instance directly."""
        return self.invoke(messages, **kwargs)
