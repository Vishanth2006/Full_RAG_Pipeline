"""
Application configuration and settings management.
All defaults can be overridden via environment variables or .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(Path(__file__).parent.parent / ".env")

# ─── Base paths ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_data"
UPLOADS_DIR = BASE_DIR / "uploads"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# ─── SQLite ───────────────────────────────────────────────────────────────────
SQLITE_DB_PATH = str(DATA_DIR / "rag_app.db")

# ─── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = str(CHROMA_DIR)
CHROMA_COLLECTION_NAME = "pdf_rag_chunks"

# ─── Embedding ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIMENSIONS = 384

# ─── Chunking defaults ────────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# ─── Retrieval defaults ───────────────────────────────────────────────────────
DEFAULT_TOP_K = int(os.getenv("TOP_K", 4))
MAX_TOP_K = 20

# ─── Upload limits ────────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", 50))
ALLOWED_FILE_TYPE = "pdf"

# ─── Gemini ───────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
SUPPORTED_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
]

DEFAULT_SYSTEM_PROMPT = (
    "You are a document-grounded assistant. Answer the user's question using "
    "the supplied document context. Do not invent facts or claim that unsupported "
    "information appears in the documents. If the context is insufficient, clearly "
    "state that the available documents do not provide enough information. Cite "
    "supporting sources using the supplied document names and page numbers."
)

# ─── Gemini generation config defaults ────────────────────────────────────────
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_OUTPUT_TOKENS = 2048

# ─── Pagination ───────────────────────────────────────────────────────────────
CHUNKS_PER_PAGE = 10
VECTORS_PER_PAGE = 10
HISTORY_PER_PAGE = 10

# ─── Processing states ────────────────────────────────────────────────────────
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_INDEXED = "indexed"
STATUS_FAILED = "failed"
STATUS_DUPLICATE = "duplicate"
STATUS_EMPTY = "empty_text"
STATUS_DELETED = "deleted"
