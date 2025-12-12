from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.vectorstores import FAISS



class BaseRAGChecker(ABC):
    """
    Базовый класс для RAG-проверок (эссе, НИР).
    """

    def __init__(
        self,
        faiss_retriever: FAISS,
        model_name: str = "gemma-3-27b-it/latest",
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
    ):
        self.retriever = faiss_retriever
        self.model_name = model_name
        self.temperature = temperature

        # Общая системная инструкция для всех потомков
        self.system_prompt = system_prompt

        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
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
    ) -> Tuple[List[Dict[str, Any]], List[int]]:
        """
        Возвращает наиболее релевантные чанки и их индексы.
        """

        results = self.retriever.get_relevant_documents(essay_text)
        results = results[:top_k]

        chunks = [{"text": d.page_content, "meta": d.metadata} for d in results]
        indices = [d.metadata.get("chunk_id") for d in results]

        return chunks, indices

    def llm_call(self, prompt):
        """
        Унифицированный вызов LLM.
        """
        return self.llm.invoke(prompt)

    def generate_verdict(
        self,
        assignment_text: str,
        essay_text: str,
        top_k: int = 3,
        return_chunks: bool = False,
    ) -> Tuple[str, Optional[List[Dict[str, Any]]]]:

        chunks, _ = self.retrieve_top_k(essay_text, top_k)
        context = "\n\n".join([c["text"] for c in chunks])

        prompt = self.build_prompt(
            assignment_text=assignment_text,
            essay_text=essay_text,
            context=context
        )

        response = self.llm_call(prompt.format())

        if return_chunks:
            return response.content, chunks

        return response.content, None
