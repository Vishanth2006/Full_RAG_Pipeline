"""
Overview Dashboard UI.
Shows workspace-level metrics, recent activity, and system configuration status.
"""

import streamlit as st
from datetime import datetime

from config.settings import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSIONS,
    GEMINI_MODEL,
    SQLITE_DB_PATH,
)
from services.gemini_service import validate_api_config
from services.vector_store_service import get_collection_stats
from storage.document_registry import (
    list_documents,
    count_documents,
    count_indexed,
)
from storage.history_repository import count_queries, list_queries


def render_dashboard():
    st.header("📊 Overview Dashboard")
    st.markdown("*Real-time workspace statistics and system configuration status.*")

    # ── Fetch data ──────────────────────────────────────────────────────────
    docs = list_documents()
    total_docs = len(docs)
    indexed_docs = sum(1 for d in docs if d.status == "indexed")
    total_chunks = sum(d.chunk_count for d in docs)

    try:
        stats = get_collection_stats()
        indexed_vectors = stats.get("total_records", 0)
    except Exception:
        indexed_vectors = 0

    total_queries = count_queries()
    api_valid, api_msg = validate_api_config()

    # ── Top metrics ─────────────────────────────────────────────────────────
    st.subheader("📈 Workspace Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📄 PDFs Uploaded", total_docs)
    c2.metric("✅ Processed", indexed_docs)
    c3.metric("🧩 Total Chunks", total_chunks)
    c4.metric("🔢 Indexed Vectors", indexed_vectors)
    c5.metric("❓ Questions Asked", total_queries)

    st.divider()

    col_left, col_right = st.columns(2)

    # ── Recent documents ────────────────────────────────────────────────────
    with col_left:
        st.subheader("📁 Recent Documents")
        if docs:
            for doc in docs[:5]:
                status_icon = {
                    "indexed": "✅",
                    "failed": "❌",
                    "processing": "⏳",
                    "pending": "🕐",
                    "duplicate": "🔄",
                    "empty_text": "⚠️",
                }.get(doc.status, "❓")
                with st.expander(
                    f"{status_icon} {doc.filename}  —  {doc.chunk_count} chunks",
                    expanded=False,
                ):
                    st.write(f"**Status:** `{doc.status}`")
                    st.write(f"**Document ID:** `{doc.document_id[:16]}…`")
                    st.write(f"**Pages:** {doc.page_count}")
                    st.write(
                        f"**Size:** {doc.file_size_bytes / 1024:.1f} KB"
                    )
                    st.write(f"**Uploaded:** {_fmt_time(doc.upload_time)}")
                    if doc.error_message:
                        st.error(doc.error_message)
        else:
            st.info("No documents uploaded yet. Go to **Document Upload** to get started.")

    # ── Recent queries ───────────────────────────────────────────────────────
    with col_right:
        st.subheader("🔍 Recent Queries")
        recent = list_queries(limit=5)
        if recent:
            for q in recent:
                with st.expander(
                    f"🗨️ {q.user_question[:60]}{'…' if len(q.user_question) > 60 else ''}",
                    expanded=False,
                ):
                    st.write(f"**Time:** {_fmt_time(q.timestamp)}")
                    st.write(f"**Model:** `{q.gemini_model}`")
                    st.write(f"**Chunks retrieved:** {len(q.retrieved_chunk_ids)}")
                    st.write(f"**Answer preview:** {q.generated_answer[:200]}…")
        else:
            st.info("No queries yet. Go to **Query & Retrieval** to ask questions.")

    st.divider()

    # ── System info ─────────────────────────────────────────────────────────
    st.subheader("⚙️ System Configuration")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Embedding**")
        st.code(f"Model : {EMBEDDING_MODEL_NAME}\nDims  : {EMBEDDING_DIMENSIONS}")
        st.markdown("**Vector Database**")
        st.code(f"Engine : ChromaDB (persistent)\nPath   : {CHROMA_PERSIST_DIR}")
        st.markdown("**Metadata Store**")
        st.code(f"Engine : SQLite\nPath   : {SQLITE_DB_PATH}")

    with c2:
        st.markdown("**Gemini Configuration**")
        if api_valid:
            st.success(f"✅ API key configured  |  Model: `{GEMINI_MODEL}`")
        else:
            st.error(f"❌ {api_msg}")

        st.markdown("**Documents by Status**")
        if docs:
            from collections import Counter
            status_counts = Counter(d.status for d in docs)
            for status, count in status_counts.items():
                st.write(f"- `{status}`: **{count}**")
        else:
            st.write("No documents.")


def _fmt_time(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return iso_str
