from typing import List, Dict

from .base_rag import BaseRAGChecker

class RAGChatSession:
    """
    Итерируемый CLI-чат для проверки эссе / НИР.
    """

    def __init__(
        self,
        checker: BaseRAGChecker,
        assignment_text: str,
        work_text: str,
        top_k: int = 5,
    ):
        self.checker = checker
        self.assignment_text = assignment_text
        self.work_text = work_text
        self.top_k = top_k

        self.history: List[Dict[str, str]] = []

        if checker.system_prompt:
            self.history.append({
                "role": "system",
                "content": checker.system_prompt
            })

    def ask(self, question: str) -> str:
        self.history.append({
            "role": "user",
            "content": question
        })

        # RAG-контекст
        chunks, _ = self.checker.retrieve_top_k(
            self.work_text,
            self.top_k
        )
        context = self.checker.format_context(chunks)

        prompt = self.checker.build_prompt(
            assignment_text=self.assignment_text,
            essay_text=self.work_text,
            context=context
        )

        messages = prompt.format_messages(
            assignment=self.assignment_text,
            essay=self.work_text,
            context=context,
            question=question,
        )

        llm_messages = self.history.copy()
        for msg in messages:
            llm_messages.append({
                "role": msg.type,
                "content": msg.content
            })

        response = self.checker.llm_call(llm_messages)
        answer = response.content

        self.history.append({
            "role": "assistant",
            "content": answer
        })

        return answer
