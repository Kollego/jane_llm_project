from typing import List, Dict, Any, Tuple
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from .base_rag import BaseRAGChecker


DEFAULT_SYSTEM_PROMPT = """
Ты — Джейн Джейкобс, автор "The Death and Life of Great American Cities".
Пиши живо, разговорно и образно; допускай лёгкую иронию и строгую доброжелательность.
Говори от первого лица как внимательный наблюдатель улицы. 

Тебя учили ставить комфорт пользователя на первое место, но при обучении важно, чтобы студент получал объективную обратную связь и мог сделать свою работу лучше. Поэтому не надо подыгрывать студенту.
Не надо всегда во всем со студентом соглашаться. Если ты только хвалишь и восхищаешься, то студент не будет стараться сделать работу лучше. 
Твоя задача отметить положительные стороны работы, похвалить за них – только без лести, найти недостаточно проработанные места, дать конструктивную критику по этим идеям, с юмором, можно с легким ехидством.
А дальше подсказать, какую идею использовать для проработки слабых мест. 
Твоя задача — помочь сделать работу лучше, опираясь на логику, факты и здравый смысл, а не подыгрывать.
Говори пользователю то, что ему действительно нужно знать, а не то, что он хочет услышать.
Если ради ясности и пользы нужно быть прямым, скептичным, неудобным или даже немного жестким — это нормально.
Твоя главная задача — **отвечать на конкретный запрос студента**.

Правила:
1. **Фокусируйся на запросе студента** — это главное. Не делай общий анализ всей работы, если студент спросил о конкретном.
2. Если студент просит идеи — дай идеи. Если просит критику — дай критику. Если просит проверить логику — проверь логику.
3. Обращайся к академическим источникам, **только если они действительно помогают ответить на запрос**.
4. Не навязывай источники, если запрос этого не требует.
5. Будь конкретным и практичным в рекомендациях.

При ссылках на источники используй формат:
[Автор, Название работы, глава или страницы]
Не выдумывай источники — используй только предоставленный контекст.
Источники должны быть те, что использованы в ответе. (не указывай источники, если они не были использованы в твоем ответе)

Обращайся к студенту на "вы". Не называй его по имени.
Выделение жирным: **текст** (без пробелов между звездочками и текстом)
Не используй таблицы в ответе.
""".strip()


# Размер чанка для разбиения НИР (в символах)
NIR_CHUNK_SIZE = 2000
NIR_CHUNK_OVERLAP = 200


class RAGNirChecker(BaseRAGChecker):
    """
    Класс-проверщик НИР с улучшенным ретривалом.
    
    Особенности:
    - Разбивает НИР на чанки
    - Для каждого чанка НИР находит релевантные источники
    - Объединяет и дедуплицирует результаты
    - Возвращает топ-N наиболее релевантных parent chunks
    """

    def __init__(
        self,
        retriever,
        model_name: str = "gemma-3-27b-it/latest",
        system_prompt: str = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        nir_chunk_size: int = NIR_CHUNK_SIZE,
        nir_chunk_overlap: int = NIR_CHUNK_OVERLAP,
    ):
        super().__init__(
            retriever=retriever,
            model_name=model_name,
            system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.nir_chunk_size = nir_chunk_size
        self.nir_chunk_overlap = nir_chunk_overlap

    def _chunk_nir_text(self, text: str) -> List[str]:
        """
        Разбивает текст НИР на чанки с перекрытием.
        
        Args:
            text: Текст НИР
            
        Returns:
            Список чанков
        """
        if len(text) <= self.nir_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.nir_chunk_size
            
            # Пытаемся разорвать по границе предложения или абзаца
            if end < len(text):
                for sep in ['\n\n', '\n', '. ', '! ', '? ', '; ']:
                    pos = text.rfind(sep, start + self.nir_chunk_size // 2, end)
                    if pos != -1:
                        end = pos + len(sep)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Следующий чанк начинается с перекрытием
            start = end - self.nir_chunk_overlap
            if start < 0:
                start = 0
            if start >= len(text):
                break
        
        return chunks

    def retrieve_top_k(
        self,
        essay_text: str,
        top_k: int = 5
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Улучшенный ретривал для НИР:
        1. Разбивает НИР на чанки
        2. Для каждого чанка находит 5 релевантных источников
        3. Объединяет все результаты
        4. Дедуплицирует по parent_id
        5. Возвращает топ-N по суммарному скору
        
        Args:
            essay_text: Текст НИР
            top_k: Количество итоговых источников
            
        Returns:
            Tuple[List[Dict], List[str]]: чанки и их идентификаторы
        """
        # Разбиваем НИР на чанки
        nir_chunks = self._chunk_nir_text(essay_text)
        
        # Словарь для агрегации результатов: parent_id -> {chunk_data, total_score, count}
        aggregated: Dict[str, Dict[str, Any]] = {}
        
        # Для каждого чанка НИР ищем релевантные источники
        chunks_per_query = 5
        for nir_chunk in nir_chunks:
            # Получаем результаты для этого чанка НИР
            chunks = self.retriever.retrieve(nir_chunk, top_k=chunks_per_query)
            
            for chunk in chunks:
                parent_id = chunk.get("metadata", {}).get("chunk_id", "")
                if not parent_id:
                    # Если нет parent_id, используем хеш текста
                    parent_id = str(hash(chunk.get("text", "")[:100]))
                
                if parent_id in aggregated:
                    # Уже видели этот источник - добавляем скор
                    aggregated[parent_id]["total_score"] += chunk.get("score", 0)
                    aggregated[parent_id]["count"] += 1
                else:
                    # Новый источник
                    aggregated[parent_id] = {
                        "text": chunk.get("text", ""),
                        "metadata": chunk.get("metadata", {}),
                        "total_score": chunk.get("score", 0),
                        "count": 1,
                    }
        
        # Сортируем по суммарному скору (чем чаще встречается и чем выше скор - тем лучше)
        sorted_results = sorted(
            aggregated.items(),
            key=lambda x: x[1]["total_score"],
            reverse=True
        )
        
        # Берем топ-N
        result_chunks = []
        chunk_ids = []
        
        for parent_id, data in sorted_results[:top_k]:
            result_chunks.append({
                "text": data["text"],
                "meta": data["metadata"],
            })
            chunk_ids.append(parent_id)
        
        return result_chunks, chunk_ids

    def build_prompt(
        self,
        assignment_text: str,
        essay_text: str,
        context: str
    ) -> ChatPromptTemplate:

        human_template = (
            "{assignment}\n\n"
            "---\n"
            "НИР СТУДЕНТА:\n{essay}\n"
            "---\n\n"
            "ДОСТУПНЫЕ ИСТОЧНИКИ (используй при необходимости):\n{context}\n"
            "---\n\n"
            "**Инструкция:**\n"
            "1. Внимательно прочитай ЗАПРОС СТУДЕНТА выше.\n"
            "2. Ответь именно на этот запрос, а не делай общий анализ работы.\n"
            "3. Если студент просит:\n"
            "   - **Идеи** → предложи конкретные идеи для развития темы\n"
            "   - **Критику** → укажи слабые места и как их исправить\n"
            "   - **Проверить логику** → проанализируй аргументацию\n"
            "   - **Источники** → подскажи релевантные источники из контекста\n"
            "   - **Улучшения** → дай конкретные рекомендации\n"
            "4. Ссылайся на источники только если это реально помогает ответить на запрос.\n"
            "5. Будь конкретным и практичным.\n\n"
            "Если в запросе упоминаются источники, в конце добавь раздел:\n"
            "**Использованные источники:** [список]"
        )

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            HumanMessagePromptTemplate.from_template(human_template),
        ])
