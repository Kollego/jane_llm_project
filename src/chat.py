"""
RAG Chat Session for interactive dialog with essay/NIR checking.

Provides a stateful chat session that maintains conversation history
and integrates with RAG retrieval for context-aware responses.
"""

from typing import List, Dict, Optional, Any

from .base_rag import BaseRAGChecker


# Максимальная длина текста работы для включения в диалог
MAX_WORK_TEXT_LENGTH = 3000
# Максимальная длина контекста источников
MAX_CONTEXT_LENGTH = 4000


class RAGChatSession:
    """
    Итерируемый чат для проверки эссе / НИР.
    
    Поддерживает:
    - Хранение истории диалога
    - RAG-контекст для каждого вопроса
    - Генерацию ответов через LLM
    """

    def __init__(
        self,
        checker: BaseRAGChecker,
        assignment_text: str,
        work_text: str,
        top_k: int = 5,
    ):
        """
        Инициализация сессии диалога.
        """
        self.checker = checker
        self.assignment_text = assignment_text
        self.work_text = work_text
        self.top_k = top_k
        self.history: List[Dict[str, str]] = []
        
        # Кешируем источники при инициализации
        self._cached_chunks = None
        self._cached_context = None

    def _get_truncated_work_text(self) -> str:
        """Возвращает сокращенный текст работы для диалога."""
        if len(self.work_text) <= MAX_WORK_TEXT_LENGTH:
            return self.work_text
        
        # Берем начало и конец работы
        half = MAX_WORK_TEXT_LENGTH // 2
        return (
            self.work_text[:half] + 
            "\n\n[...текст сокращён...]\n\n" + 
            self.work_text[-half:]
        )

    def _get_context(self) -> str:
        """Получает и кеширует контекст источников."""
        if self._cached_context is None:
            chunks, _ = self.checker.retrieve_top_k(self.work_text, self.top_k)
            self._cached_chunks = chunks
            self._cached_context = self.checker.format_context(chunks)
            
            # Ограничиваем длину контекста
            if len(self._cached_context) > MAX_CONTEXT_LENGTH:
                self._cached_context = self._cached_context[:MAX_CONTEXT_LENGTH] + "\n[...источники сокращены...]"
        
        return self._cached_context

    def ask(self, question: str) -> str:
        """
        Отправляет вопрос и получает ответ.
        """
        # Добавляем вопрос пользователя в историю
        self.history.append({
            "role": "user",
            "content": question
        })

        # Формируем сообщения для LLM
        llm_messages = []
        
        # Системный промт
        if self.checker.system_prompt:
            llm_messages.append({
                "role": "system",
                "content": self.checker.system_prompt
            })
        
        # Для первого вопроса - полный контекст
        # Для последующих - только ссылка на контекст
        if len(self.history) <= 1:
            # Первый вопрос - добавляем полный контекст
            work_text = self._get_truncated_work_text()
            context = self._get_context()
            
            context_message = (
                f"{self.assignment_text}\n\n" if self.assignment_text else ""
            ) + (
                f"РАБОТА СТУДЕНТА:\n---\n{work_text}\n---\n\n"
                f"ИСТОЧНИКИ:\n---\n{context}\n---\n\n"
                f"ВОПРОС: {question}"
            )
            
            llm_messages.append({
                "role": "user",
                "content": context_message
            })
        else:
            # Последующие вопросы - краткий контекст + история
            # Добавляем краткое напоминание о контексте
            llm_messages.append({
                "role": "user",
                "content": (
                    "Контекст: Ты анализируешь НИР студента. "
                    "Предыдущий диалог ниже. Отвечай на новый вопрос."
                )
            })
            
            # Добавляем предыдущие сообщения (ограничиваем историю)
            recent_history = self.history[-6:]  # Последние 3 пары вопрос-ответ
            for msg in recent_history[:-1]:  # Без текущего вопроса
                llm_messages.append(msg)
            
            # Добавляем текущий вопрос
            llm_messages.append({
                "role": "user",
                "content": question
            })

        # Вызываем LLM
        response = self.checker.llm_call(llm_messages)
        answer = response.content

        # Сохраняем ответ в историю
        self.history.append({
            "role": "assistant",
            "content": answer
        })

        return answer

    def get_history(self, include_system: bool = False) -> List[Dict[str, str]]:
        """Возвращает историю диалога."""
        return self.history.copy()

    def clear_history(self) -> None:
        """Очищает историю диалога."""
        self.history = []

    def get_sources(self) -> List[Dict[str, Any]]:
        """Возвращает источники, релевантные работе."""
        if self._cached_chunks is None:
            self._get_context()
        return self._cached_chunks or []
