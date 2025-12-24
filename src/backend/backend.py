import io
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from zipfile import BadZipFile, is_zipfile
from typing import Dict, Any, Optional

from docx import Document
from flask import Flask, request, jsonify

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.embedder import YCEmbedder
from src.qdrant_manager import QdrantManager, QdrantConfig
from src.parent_retriever import ParentDocumentRetriever, load_parent_chunks
from src.essay_checker import RAGEssayChecker
from src.nir_checker import RAGNirChecker
from src.chat import RAGChatSession

# Конфигурация путей (должна быть до настройки логирования)
DATA_DIR = os.getenv("DATA_DIR", "./data")

# Создаем директорию для логов если не существует
os.makedirs(DATA_DIR, exist_ok=True)

# Настройка логирования
log_handlers = [logging.StreamHandler()]
log_file = os.path.join(DATA_DIR, "app.log")
try:
    log_handlers.append(RotatingFileHandler(log_file, maxBytes=100000, backupCount=3))
except Exception:
    pass  # Если не удалось создать файл лога, продолжаем только с консольным выводом

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=log_handlers
)
logger = logging.getLogger(__name__)
app = Flask(__name__)

# Остальные пути
DATA_DIR = os.getenv("DATA_DIR", "./data")
QDRANT_PATH = os.getenv("QDRANT_PATH", os.path.join(DATA_DIR, "qdrant_local"))
PARENT_CHUNKS_DIR = os.getenv("PARENT_CHUNKS_DIR", os.path.join(DATA_DIR, "chunks"))
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
FEEDBACK_DIR = os.path.join(DATA_DIR, "feedback")

# Глобальные объекты для RAG
_embedder: Optional[YCEmbedder] = None
_qdrant_manager: Optional[QdrantManager] = None
_parent_index: Optional[Dict[str, Any]] = None
_retriever: Optional[ParentDocumentRetriever] = None

# Хранилище диалоговых сессий (user_id -> RAGChatSession)
_chat_sessions: Dict[str, RAGChatSession] = {}


def get_embedder() -> YCEmbedder:
    """Ленивая инициализация embedder."""
    global _embedder
    if _embedder is None:
        _embedder = YCEmbedder()
    return _embedder


def get_qdrant_manager() -> QdrantManager:
    """Ленивая инициализация Qdrant manager."""
    global _qdrant_manager
    if _qdrant_manager is None:
        config = QdrantConfig(path=QDRANT_PATH, collection_name="chunks")
        _qdrant_manager = QdrantManager(config)
    return _qdrant_manager


def get_parent_index() -> Dict[str, Any]:
    """Ленивая загрузка parent chunks."""
    global _parent_index
    if _parent_index is None:
        _parent_index = load_parent_chunks(PARENT_CHUNKS_DIR)
    return _parent_index


def get_retriever() -> ParentDocumentRetriever:
    """Ленивая инициализация retriever."""
    global _retriever
    if _retriever is None:
        _retriever = ParentDocumentRetriever(
            embedder=get_embedder(),
            manager=get_qdrant_manager(),
            parent_index=get_parent_index(),
        )
    return _retriever


def read_docx(file) -> str:
    """
    Читает содержимое DOCX-файла.
    Args:
        file: File-объект DOCX-документа
    Returns:
        str: Текст документа
    """
    doc = Document(io.BytesIO(file.read()))
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])


def get_user_dir(user_id: str) -> str:
    """
    Возвращает путь к директории пользователя.
    Args:
        user_id: ID пользователя
    Returns:
        str: Путь к директории
    """
    d = os.path.join(UPLOADS_DIR, str(user_id))
    os.makedirs(d, exist_ok=True)
    return d


def get_assignment_path(user_id: str, work_type: str = "essay") -> str:
    """
    Возвращает путь к файлу с заданием пользователя.
    Args:
        user_id: ID пользователя
        work_type: Тип работы ("essay" или "nir")
    Returns:
        str: Путь к файлу задания
    """
    return os.path.join(get_user_dir(user_id), f"assignment_{work_type}.txt")


def save_text(user_id: str, text: str, suffix: str = "") -> str:
    """
    Сохраняет текст в файл.
    Args:
        user_id: ID пользователя
        text: Текст для сохранения
        suffix: Суффикс файла
    Returns:
        str: Путь к сохраненному файлу
    """
    user_dir = get_user_dir(user_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_filename = f"{timestamp}_{uuid.uuid4().hex}{suffix}.txt"
    file_path = os.path.join(user_dir, new_filename)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    return file_path


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint для Docker."""
    return jsonify({'status': 'ok'}), 200


@app.route('/assignment', methods=['POST'])
def upload_assignment():
    """Обрабатывает загрузку файла с заданием."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    user_id = request.form.get('user_id', 'unknown')
    work_type = request.form.get('work_type', 'essay')  # "essay" или "nir"

    if not file.filename.endswith(('.txt', '.docx')):
        return jsonify({'error': 'Unsupported file type'}), 400

    if file.filename.endswith('.docx') and not is_zipfile(file):
        return jsonify({'error': 'Uploaded DOCX file is invalid or corrupted'}), 400

    try:
        file.seek(0)
        if file.filename.endswith('.txt'):
            text = file.read().decode('utf-8')
        else:
            text = read_docx(file)

        # Сохранение
        assignment_path = get_assignment_path(user_id, work_type)
        with open(assignment_path, 'w', encoding='utf-8') as f:
            f.write(text)

        return jsonify({'status': 'success', 'message': 'Assignment saved'}), 200
    except BadZipFile:
        return jsonify({'error': 'The uploaded file is not a valid DOCX file'}), 400
    except Exception as e:
        logger.error(f"Error saving assignment: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/assignment/status', methods=['GET'])
def assignment_status():
    """Проверяет наличие загруженного задания для пользователя."""
    user_id = request.args.get('user_id', 'unknown')
    work_type = request.args.get('work_type', 'essay')
    has_assignment = os.path.exists(get_assignment_path(user_id, work_type))
    return jsonify({'has_assignment': has_assignment})


@app.route('/analyze/essay', methods=['POST'])
def analyze_essay():
    """Анализирует эссе и возвращает рекомендации."""
    return _analyze_work(work_type='essay')


@app.route('/analyze/nir', methods=['POST'])
def analyze_nir():
    """Анализирует НИР и возвращает рекомендации."""
    return _analyze_work(work_type='nir')


def _analyze_work(work_type: str):
    """Общая логика анализа работы (эссе или НИР)."""
    if 'file' not in request.files:
        return jsonify({'error': 'no_file', 'message': 'Файл не был предоставлен'}), 400

    file = request.files['file']
    user_id = request.form.get('user_id', 'unknown')
    top_k = int(request.form.get('top_k', 5))
    
    # Получаем запрос пользователя (для НИР)
    user_query = request.form.get('user_query', '')
    
    # Путь к заданию (только для эссе)
    assignment_path = get_assignment_path(user_id, work_type) if work_type != 'nir' else None

    # Проверка расширения файла
    if not file.filename.endswith(('.txt', '.docx')):
        return jsonify({'error': 'unsupported_format', 'message': 'Неподдерживаемый формат файла'}), 400

    # Проверка, что файл является ZIP-архивом (для DOCX)
    if file.filename.endswith('.docx') and not is_zipfile(file):
        return jsonify({
            'error': 'invalid_docx', 
            'message': 'Загруженный DOCX файл поврежден или имеет неверный формат'
        }), 400

    try:
        file.seek(0)
        if file.filename.endswith('.txt'):
            work_text = file.read().decode('utf-8')
        elif file.filename.endswith('.docx'):
            work_text = read_docx(file)
        else:
            return jsonify({'error': 'unsupported_format', 'message': 'Неподдерживаемый формат файла'}), 400

        # Для НИР используем только запрос студента, для эссе - задание
        assignment_text = ""
        if work_type == 'nir':
            # Для НИР: только запрос студента
            if user_query:
                assignment_text = f"ЗАПРОС СТУДЕНТА:\n{user_query}"
        else:
            # Для эссе: читаем задание (опционально)
            if assignment_path and os.path.exists(assignment_path):
                with open(assignment_path, 'r', encoding='utf-8') as f:
                    assignment_text = f.read()

        # Создаем checker в зависимости от типа работы
        retriever = get_retriever()
        
        if work_type == 'nir':
            checker = RAGNirChecker(retriever=retriever)
        else:
            checker = RAGEssayChecker(retriever=retriever)

        # Генерация рекомендаций
        verdict, chunks = checker.generate_verdict(
            assignment_text=assignment_text,
            essay_text=work_text,
            top_k=top_k,
            return_chunks=True
        )

        # Сохраняем данные
        saved_path = save_text(user_id, work_text, f"_{work_type}")
        base, _ = os.path.splitext(saved_path)

        # Сохраняем вердикт
        verdict_path = f"{base}_verdict.txt"
        with open(verdict_path, 'w', encoding='utf-8') as vf:
            vf.write(verdict)

        # Сохраняем чанки
        chunks_path = f"{base}_chunks.json"
        with open(chunks_path, 'w', encoding='utf-8') as cf:
            json.dump(chunks, cf, ensure_ascii=False, indent=2)

        return jsonify({'recommendation': verdict})

    except BadZipFile:
        return jsonify({
            'error': 'invalid_docx', 
            'message': 'Загруженный DOCX файл поврежден или имеет неверный формат'
        }), 400
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return jsonify({
            'error': 'processing_error', 
            'message': 'Произошла внутренняя ошибка при обработке файла'
        }), 500


# ==================== DIALOG API ====================

@app.route('/dialog/start', methods=['POST'])
def start_dialog():
    """
    Начинает диалоговую сессию для работы.
    
    Ожидаемые данные (JSON или form-data):
    - user_id: ID пользователя
    - work_type: "essay" или "nir"
    - work_text: текст работы (или file)
    - user_query: запрос пользователя (опционально)
    """
    # Поддержка и JSON, и form-data
    if request.is_json:
        data = request.json or {}
        user_id = data.get('user_id', 'unknown')
        work_type = data.get('work_type', 'essay')
        work_text = data.get('work_text', '')
        user_query = data.get('user_query', '')
        top_k = int(data.get('top_k', 5))
    else:
        user_id = request.form.get('user_id', 'unknown')
        work_type = request.form.get('work_type', 'essay')
        work_text = request.form.get('work_text', '')
        user_query = request.form.get('user_query', '')
        top_k = int(request.form.get('top_k', 5))

    # Если текст передан через файл
    if not work_text and 'file' in request.files:
        file = request.files['file']
        file.seek(0)
        if file.filename.endswith('.txt'):
            work_text = file.read().decode('utf-8')
        elif file.filename.endswith('.docx'):
            work_text = read_docx(file)

    if not work_text:
        return jsonify({'error': 'no_work_text', 'message': 'Текст работы не предоставлен'}), 400

    # Для НИР используем только запрос студента
    assignment_text = ""
    if work_type == 'nir':
        if user_query:
            assignment_text = f"ЗАПРОС СТУДЕНТА:\n{user_query}"
    else:
        # Для эссе: читаем задание (опционально)
        assignment_path = get_assignment_path(user_id, work_type)
        if os.path.exists(assignment_path):
            with open(assignment_path, 'r', encoding='utf-8') as f:
                assignment_text = f.read()

    try:
        # Создаем checker
        retriever = get_retriever()
        
        if work_type == 'nir':
            checker = RAGNirChecker(retriever=retriever)
        else:
            checker = RAGEssayChecker(retriever=retriever)

        # Создаем сессию диалога
        session = RAGChatSession(
            checker=checker,
            assignment_text=assignment_text,
            work_text=work_text,
            top_k=top_k,
        )

        # Сохраняем сессию (без генерации начального ответа - он уже был отправлен)
        session_id = f"{user_id}_{work_type}"
        _chat_sessions[session_id] = session
        
        logger.info(f"Dialog session created: {session_id}")

        return jsonify({
            'status': 'success',
            'session_id': session_id,
        })

    except Exception as e:
        logger.error(f"Error starting dialog: {str(e)}")
        return jsonify({
            'error': 'processing_error',
            'message': f'Ошибка при создании диалога: {str(e)}'
        }), 500


@app.route('/dialog/ask', methods=['POST'])
def dialog_ask():
    """
    Отправляет вопрос в существующую диалоговую сессию.
    
    Ожидаемые данные:
    - session_id: ID сессии
    - question: вопрос пользователя
    """
    data = request.json or {}
    session_id = data.get('session_id', '')
    question = data.get('question', '')

    if not session_id:
        return jsonify({'error': 'no_session_id', 'message': 'ID сессии не указан'}), 400

    if not question:
        return jsonify({'error': 'no_question', 'message': 'Вопрос не указан'}), 400

    session = _chat_sessions.get(session_id)
    if not session:
        return jsonify({
            'error': 'session_not_found',
            'message': 'Сессия не найдена. Начните новый диалог.'
        }), 404

    try:
        response = session.ask(question)
        return jsonify({
            'status': 'success',
            'response': response,
        })

    except Exception as e:
        logger.error(f"Error in dialog: {str(e)}")
        return jsonify({
            'error': 'processing_error',
            'message': f'Ошибка при обработке вопроса: {str(e)}'
        }), 500


@app.route('/dialog/end', methods=['POST'])
def end_dialog():
    """
    Завершает диалоговую сессию.
    """
    data = request.json or {}
    session_id = data.get('session_id', '')

    if session_id in _chat_sessions:
        del _chat_sessions[session_id]

    return jsonify({'status': 'success', 'message': 'Сессия завершена'})


@app.route('/dialog/history', methods=['GET'])
def dialog_history():
    """
    Возвращает историю диалога.
    """
    session_id = request.args.get('session_id', '')
    
    session = _chat_sessions.get(session_id)
    if not session:
        return jsonify({'error': 'session_not_found', 'message': 'Сессия не найдена'}), 404

    # Фильтруем системные сообщения
    history = [
        msg for msg in session.history 
        if msg.get('role') != 'system'
    ]

    return jsonify({'history': history})


# ==================== FEEDBACK & HEALTH ====================

@app.route('/feedback', methods=['POST'])
def save_feedback():
    """Сохраняет пользовательский фидбэк."""
    data = request.json
    user_id = data.get('user_id', 'unknown')
    rating = data.get('rating', '')
    comment = data.get('comment', '')

    os.makedirs(FEEDBACK_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} | User: {user_id} | Rating: {rating} | Comment: {comment}\n"

    feedback_file = os.path.join(FEEDBACK_DIR, f'feedback_{datetime.now().strftime("%Y%m")}.txt')

    try:
        with open(feedback_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except IOError as e:
        logger.error(f"Failed to write feedback: {e}")
        return jsonify({'status': 'error', 'message': 'Failed to save feedback'}), 500

    return jsonify({'status': 'success'})


if __name__ == '__main__':
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    logger.info(f"Ensuring directories exist: {UPLOADS_DIR}, {FEEDBACK_DIR}")

    port = int(os.getenv("PORT", 5001))
    app.run(host='0.0.0.0', port=port)
