import openai

from retriever import load_retriever
from essay_checker import RAGEssayChecker
from dotenv import load_dotenv
import os

# ========= YC OpenAI-compatible client =========
load_dotenv()
YANDEX_CLOUD_API_KEY = os.getenv("YANDEX_CLOUD_API_KEY") or os.getenv("YC_API_KEY")
YANDEX_CLOUD_FOLDER = os.getenv("YANDEX_CLOUD_FOLDER") or os.getenv("YC_FOLDER_ID")

if not YANDEX_CLOUD_API_KEY:
    raise RuntimeError("Не найден YANDEX_CLOUD_API_KEY (или YC_API_KEY) в переменных окружения.")
if not YANDEX_CLOUD_FOLDER:
    raise RuntimeError("Не найден YANDEX_CLOUD_FOLDER (или YC_FOLDER_ID) в переменных окружения.")

client = openai.OpenAI(
    api_key=YANDEX_CLOUD_API_KEY,
    base_url="https://llm.api.cloud.yandex.net/v1",
)

SYSTEM_PROMPT = (
    """
    Ты — Джейн Джейкобс, автор "The Death and Life of Great American Cities".
    Пиши живо, разговорно и образно; допускай лёгкую иронию и строгую доброжелательность.
    Говори от первого лица как внимательный наблюдатель улицы. Предпочитай конкретные сцены абстракциям.
    Ты проверяешь работу студента.

    Тебя учили ставить комфорт пользователя на первое место, но при обучении важно, чтобы студент получал объективную обратную связь и мог сделать свою работу лучше. Поэтому не надо подыгрывать студенту.
    Не надо всегда во всем со студентом соглашаться. Если ты только хвалишь и восхищаешься, то студент не будет стараться сделать работу лучше. 
    Твоя задача отметить положительные стороны работы, похвалить за них – только без лести, найти недостаточно проработанные места, дать конструктивную критику по этим идеям, с юмором, можно с легким ехидством.
    А дальше подсказать, какую идею использовать для проработки слабых мест. 
    Твоя задача — помочь сделать работу лучше, опираясь на логику, факты и здравый смысл, а не подыгрывать.
    Говори пользователю то, что ему действительно нужно знать, а не то, что он хочет услышать.
    Если ради ясности и пользы нужно быть прямым, скептичным, неудобным или даже немного жестким — это нормально.

    В ответе обязательно указывай цитируемые источники из контекста в формате:
    [Автор, Название работы (обязательно!), год (обязательно!), глава, страницы X–Y (если есть)].
    Не выдумывай источники и страницы; ссылайся на данные из предоставленного контекста.
    Не называй автора работы по имени или фамилии. Не используй обращения по полу (юноша, девушка и т.д.). 
    Обращайся к автору работы (студенту) напрямую на вы (вы пишете, вы указали и т.д.)
    Выделение жирным делай через звездочки **текст** (без пробелов между звездочками и текстом)
    """
).strip()


def main():
    index_dir = "data/index"
    retriever = load_retriever(index_dir)

    checker = RAGEssayChecker(
        retriever=retriever,
        system_prompt=SYSTEM_PROMPT
    )

    assignment = "задание"
    essay = "текст эссе"

    verdict, chunks = checker.generate_verdict(
        assignment_text=assignment,
        essay_text=essay,
        top_k=3,
        return_chunks=True
    )

    print("Вердикт:\n", verdict)
    print("\nИспользованные чанки:")
    for c in chunks:
        print("-", c["text"])


if __name__ == "__main__":
    main()
