# 📚 PDF RAG Assistant

A complete, modular, locally-runnable **Retrieval-Augmented Generation (RAG)** application built with Python, Streamlit, and Google Gemini.

Every intermediate step in the RAG pipeline is **fully inspectable** from the UI:
extraction → chunking → embedding → vector storage → retrieval → prompt construction → answer generation.

---

## ✨ Features

| Feature | Details |
|---|---|
| PDF Upload | Multi-file upload with validation, deduplication, size limits |
| Text Extraction | Page-by-page with PyMuPDF, warnings for image-only pages |
| Chunking | `RecursiveCharacterTextSplitter`, configurable size/overlap |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, local) |
| Vector Store | ChromaDB (persistent, cosine distance, HNSW index) |
| Retrieval | Top-k similarity search with per-document filtering |
| Prompt Inspector | View/edit every component sent to Gemini |
| LLM | Google Gemini via official `google-genai` SDK |
| History | Full query history with context snapshots (SQLite) |
| Testing | pytest suite covering all pipeline stages |

---

## 🗂 Project Structure

```
Basic_RAG/
├── app.py                   # Streamlit entry point
├── requirements.txt
├── .env.example             # API key template
├── .env                     # Your real config (not committed)
│
├── config/
│   └── settings.py          # All configuration & defaults
│
├── services/                # Core pipeline logic
│   ├── pdf_service.py
│   ├── chunking_service.py
│   ├── embedding_service.py
│   ├── vector_store_service.py
│   ├── retrieval_service.py
│   ├── prompt_service.py
│   ├── gemini_service.py
│   ├── history_service.py
│   └── document_service.py
│
├── ui/                      # Streamlit page modules
│   ├── dashboard.py
│   ├── upload.py
│   ├── extraction.py
│   ├── chunk_explorer.py
│   ├── embedding_explorer.py
│   ├── vector_database.py
│   ├── query_retrieval.py
│   ├── prompt_inspector.py
│   ├── answer_history.py
│   └── settings.py
│
├── models/
│   └── schemas.py           # Typed dataclasses
│
├── storage/
│   ├── document_registry.py # SQLite document records
│   └── history_repository.py# SQLite query history
│
├── tests/                   # pytest test suite
│   ├── conftest.py
│   ├── test_pdf_extraction.py
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_retrieval.py
│   ├── test_prompt_construction.py
│   ├── test_document_lifecycle.py
│   ├── test_history.py
│   ├── test_gemini_service.py
│   └── test_end_to_end.py
│
├── data/                    # SQLite DB (runtime, gitignored)
└── chroma_data/             # ChromaDB (runtime, gitignored)
```

---

## 🚀 Installation

### 1. Clone or download the project

```bash
cd "e:\visha\Documents\Sem-6\AI\Basic_RAG"
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will download the `all-MiniLM-L6-v2` model (~90 MB). This is cached locally after the first download.

---

## 🔑 Gemini API Configuration

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Open `.env` and fill in your key:

```env
GEMINI_API_KEY=your_actual_api_key_here
GEMINI_MODEL=gemini-2.0-flash
```

Supported models:
- `gemini-2.0-flash` *(recommended — fast, capable)*
- `gemini-1.5-flash`
- `gemini-1.5-pro`

**Your API key is never displayed in the UI or logs.**

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_chunking.py -v

# Run with coverage
pip install pytest-cov
pytest tests/ --cov=. --cov-report=term-missing

# Skip slow embedding tests
pytest tests/ -v -k "not embedding"
```

> Tests do **not** require a Gemini API key. All LLM calls are mocked.

---

## 📋 Usage Walkthrough

### Step 1: Upload Documents
- Navigate to **Document Upload**
- Upload one or more PDFs
- Click **Process Documents**
- Watch the progress bar: extraction → chunking → embedding → indexing

### Step 2: Explore Intermediate Results
- **Text Extraction Explorer** — see per-page extracted text
- **Chunk Explorer** — browse all generated chunks with metadata
- **Embedding Explorer** — inspect 384-dimensional vectors with heatmap
- **Vector Database Explorer** — browse ChromaDB records directly

### Step 3: Ask Questions
- Navigate to **Query & Retrieval**
- Type your question, set top-k, optionally filter by document
- View query embedding, retrieved chunks, and similarity scores

### Step 4: Inspect & Generate
- Navigate to **Prompt Inspector & Answer**
- Edit the system prompt if desired
- View the exact context and user message sent to Gemini
- Click **Generate Answer with Gemini**
- View the grounded answer with source citations

### Step 5: Review History
- Navigate to **Answer & Query History**
- Browse past queries with full context reconstruction
- Compare two queries side-by-side
- Inspect prompts used for historical queries

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model name |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `120` | Overlap between chunks |
| `TOP_K` | `4` | Default retrieved chunks |
| `MAX_UPLOAD_SIZE_MB` | `50` | Per-file upload limit |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | ChromaDB storage path |
| `SQLITE_DB_PATH` | `./data/rag_app.db` | SQLite database path |

---

## ⚠️ Known Limitations

1. **Scanned/image-only PDFs** — PyMuPDF cannot extract text from scanned pages. OCR is not included. The app shows a clear warning when this occurs.
2. **Embedding model change** — Changing the embedding model requires recreating the ChromaDB collection and re-indexing all documents.
3. **Single-user** — ChromaDB's persistent client is not safe for concurrent multi-user access.
4. **Token limits** — Very long documents may produce many chunks. Be mindful of context length when setting a high top-k.
5. **API quotas** — Free-tier Gemini keys have rate limits. The app shows user-friendly errors for quota exceeded cases.
6. **Windows path lengths** — If you encounter errors, ensure your path does not exceed Windows' MAX_PATH limit (consider enabling long paths).

---

## 📝 Sample PDF Testing

To test without your own PDFs, create a simple text file and convert it, or use any academic paper or report PDF.

The `tests/conftest.py` includes a minimal PDF generator for automated testing that produces valid PDFs with extractable text.

---

*Built for educational and demonstrable RAG pipeline exploration.*
