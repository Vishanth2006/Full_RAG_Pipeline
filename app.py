"""
PDF RAG Assistant — Main Streamlit Application Entry Point.

Run with:
    streamlit run app.py
"""

import logging
import sys
from pathlib import Path

import streamlit as st

# ── Ensure project root is on PYTHONPATH ──────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "**PDF RAG Assistant** — Interactive Retrieval-Augmented Generation with Gemini",
    },
)

# ── Bootstrap databases ────────────────────────────────────────────────────────
from storage.document_registry import init_db
from storage.history_repository import init_history_db

init_db()
init_history_db()

# ── Configure logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ── Navigation ─────────────────────────────────────────────────────────────────
PAGES = {
    "📊 Overview Dashboard":          "dashboard",
    "📤 Document Upload":             "upload",
    "📖 Text Extraction Explorer":    "extraction",
    "🧩 Chunk Explorer":              "chunk_explorer",
    "🗄️ Data Explorer":              "data_explorer",
    "🔍 Query & Retrieval":           "query_retrieval",
    "🔬 Prompt Inspector & Answer":   "prompt_inspector",
    "📜 Answer & Query History":      "answer_history",
    "⚙️ Settings":                    "settings",
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 12px 0 8px 0;">
            <span style="font-size:2.2rem;">📚</span><br>
            <span style="font-size:1.25rem; font-weight:700; color:#7c3aed;">PDF RAG Assistant</span><br>
            <span style="font-size:0.78rem; color:#6b7280;">Transparent · Inspectable · Gemini-powered</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # Page selector
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = list(PAGES.keys())[0]

    for page_label in PAGES:
        is_current = st.session_state["current_page"] == page_label
        btn_type = "primary" if is_current else "secondary"
        if st.button(
            page_label,
            key=f"nav_{page_label}",
            use_container_width=True,
            type=btn_type,
        ):
            st.session_state["current_page"] = page_label
            st.rerun()

    st.divider()

    # Quick stats in sidebar
    try:
        from storage.document_registry import count_documents, count_indexed
        from storage.history_repository import count_queries
        from services.vector_store_service import get_collection_stats

        n_docs = count_documents()
        n_indexed = count_indexed()
        n_queries = count_queries()
        n_vectors = get_collection_stats().get("total_records", 0)

        st.markdown(
            f"""
            <div style="font-size:0.82rem; color:#6b7280; line-height:2;">
            📄 <b>{n_docs}</b> docs &nbsp;|&nbsp; ✅ <b>{n_indexed}</b> indexed<br>
            🔢 <b>{n_vectors}</b> vectors<br>
            ❓ <b>{n_queries}</b> queries
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        pass

# ── Page rendering ─────────────────────────────────────────────────────────────
current = st.session_state.get("current_page", list(PAGES.keys())[0])
page_key = PAGES[current]

if page_key == "dashboard":
    from ui.dashboard import render_dashboard
    render_dashboard()

elif page_key == "upload":
    from ui.upload import render_upload
    render_upload()

elif page_key == "extraction":
    from ui.extraction import render_extraction
    render_extraction()

elif page_key == "chunk_explorer":
    from ui.chunk_explorer import render_chunk_explorer
    render_chunk_explorer()

elif page_key == "data_explorer":
    from ui.data_explorer import render_data_explorer
    render_data_explorer()

elif page_key == "query_retrieval":
    from ui.query_retrieval import render_query_retrieval
    render_query_retrieval()

elif page_key == "prompt_inspector":
    from ui.prompt_inspector import render_prompt_inspector
    render_prompt_inspector()

elif page_key == "answer_history":
    from ui.answer_history import render_answer_history
    render_answer_history()

elif page_key == "settings":
    from ui.settings import render_settings
    render_settings()
