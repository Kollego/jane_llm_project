from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional

from langchain_core.prompts import ChatPromptTemplate

from .yandex_llm import YandexCloudModel


class BaseRAGChecker(ABC):
    """
    Базовый класс для RAG-проверок (эссе, НИР).
    
    Поддерживает любой retriever, который реализует метод retrieve(query, top_k).
    Использует Yandex Cloud LLM для генерации ответов.
    """

    def __init__(
        self,
        retriever: Any,
        model_name: str = "gemma-3-27b-it/latest",
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ):
        """
        Args:
            retriever: Объект с методом retrieve(query, top_k) -> List[Dict]
                       Каждый Dict должен содержать 'text' и 'metadata'
            model_name: Название модели Yandex GPT
            system_prompt: Системный промт
            temperature: Температура LLM
            max_tokens: Максимальное количество токенов в ответе
        """
        self.retriever = retriever
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

        self.llm = YandexCloudModel(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    # ---------- ABSTRACT METHODS ----------
    @abstractmethod
    def build_prompt(
        self,
        assignment_text: str,
        essay_text: str,
        context: str
    ) -> ChatPromptTemplate:
        """Построение промта. Переопределяется в наследниках."""
        pass

    # ---------- COMMON LOGIC ----------
    def retrieve_top_k(
        self,
        essay_text: str,
        top_k: int = 3
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Возвращает наиболее релевантные чанки и их идентификаторы.
        """
        chunks = self.retriever.retrieve(essay_text, top_k)
        
        result_chunks = []
        chunk_ids = []
        
        for chunk in chunks:
            result_chunks.append({
                "text": chunk.get("text", ""),
                "meta": chunk.get("metadata", {}),
            })
            chunk_ids.append(chunk.get("metadata", {}).get("chunk_id", ""))
        
        return result_chunks, chunk_ids

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Форматирует чанки в контекст для LLM с метаданными.
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("meta", {})
            
            # Build source reference
            ref_parts = []
            if meta.get("book_author"):
                ref_parts.append(meta["book_author"])
            if meta.get("book_title"):
                ref_parts.append(f"«{meta['book_title']}»")
            if meta.get("chapter"):
                ref_parts.append(f"глава: {meta['chapter']}")
            if meta.get("book_pages_str"):
                ref_parts.append(f"стр. {meta['book_pages_str']}")
            
            reference = ", ".join(ref_parts) if ref_parts else f"Источник {i}"
            
            text = chunk.get("text", "")
            if len(text) > 2500:
                text = text[:2500] + "\n[...текст сокращён...]"
            
            entry = f"[{reference}]\n{text}"
            context_parts.append(entry)
        
        return "\n\n---\n\n".join(context_parts)

    def llm_call(self, messages: List[Dict[str, str]]) -> Any:
        """
        Унифицированный вызов LLM.
        """
        return self.llm.invoke(messages)

    def generate_verdict(
        self,
        assignment_text: str,
        essay_text: str,
        top_k: int = 3,
        return_chunks: bool = False,
    ) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        """
        Генерирует вердикт на основе эссе и найденных источников.
        """
        chunks, _ = self.retrieve_top_k(essay_text, top_k)
        context = self.format_context(chunks)

        prompt = self.build_prompt(
            assignment_text=assignment_text,
            essay_text=essay_text,
            context=context
        )

        # Format prompt to messages
        formatted_messages = prompt.format_messages(
            assignment=assignment_text or "Не указано",
            essay=essay_text,
            context=context,
        )
        
        # Convert LangChain messages to dicts
        messages = []
        for msg in formatted_messages:
            role = "user"
            if msg.__class__.__name__ == "SystemMessage":
                role = "system"
            elif msg.__class__.__name__ == "AIMessage":
                role = "assistant"
            elif msg.__class__.__name__ == "HumanMessage":
                role = "user"
            messages.append({"role": role, "content": msg.content})

        response = self.llm_call(messages)

        if return_chunks:
            return response.content, chunks

        return response.content, None
