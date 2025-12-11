from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings


def load_retriever(index_dir: str):
    """
    Загрузка FAISS retriever.
    """
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local(index_dir, embedding, allow_dangerous_deserialization=True)
    return db.as_retriever()
