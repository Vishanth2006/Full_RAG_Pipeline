"""
Vector Database Explorer UI.
Browse ChromaDB records with filtering, pagination, and metadata inspection.
"""

import json
import streamlit as st
import numpy as np

from config.settings import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR, VECTORS_PER_PAGE
from services.vector_store_service import (
    clear_collection,
    delete_document_chunks,
    get_all_records,
    get_collection_stats,
)
from storage.document_registry import list_documents, mark_deleted


def render_vector_database():
    st.header("🗄️ Vector Database Explorer")
    st.markdown(
        f"Persistent **ChromaDB** collection: `{CHROMA_COLLECTION_NAME}`  |  "
        f"Path: `{CHROMA_PERSIST_DIR}`  |  Similarity: **cosine**"
    )

    # ── Collection stats ──────────────────────────────────────────────────
    try:
        stats = get_collection_stats()
        total_records = stats.get("total_records", 0)
    except Exception as exc:
        st.error(f"Cannot connect to ChromaDB: {exc}")
        return

    docs = [d for d in list_documents() if d.status == "indexed"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Vectors", total_records)
    c2.metric("Documents", len(docs))
    c3.metric("Collection", CHROMA_COLLECTION_NAME)

    st.divider()

    # ── Filter ────────────────────────────────────────────────────────────
    col_filter, col_per_page = st.columns([3, 1])
    with col_filter:
        doc_options = {"All documents": None}
        doc_options.update({d.filename: d.document_id for d in docs})
        selected_label = st.selectbox("Filter by document", list(doc_options.keys()))
        filter_doc_id = doc_options[selected_label]

    with col_per_page:
        per_page = st.number_input("Per page", min_value=5, max_value=50, value=VECTORS_PER_PAGE)

    # ── Pagination state ──────────────────────────────────────────────────
    page_key = "vdb_page"
    if page_key not in st.session_state:
        st.session_state[page_key] = 0
    offset = st.session_state[page_key] * per_page

    # ── Fetch records ─────────────────────────────────────────────────────
    try:
        records = get_all_records(
            document_id=filter_doc_id,
            limit=per_page,
            offset=offset,
        )
    except Exception as exc:
        st.error(f"Failed to fetch records: {exc}")
        return

    ids = records.get("ids", [])
    docs_text = records.get("documents", [])
    metadatas = records.get("metadatas", [])
    embeddings = records.get("embeddings", [])
    total = records.get("total", 0)

    # ── Pager controls ────────────────────────────────────────────────────
    total_pages = max(1, (total + per_page - 1) // per_page)
    current_page = st.session_state[page_key] + 1

    col_prev, col_info, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("◀ Prev", disabled=current_page <= 1):
            st.session_state[page_key] -= 1
            st.rerun()
    with col_info:
        st.caption(f"Page **{current_page}** of **{total_pages}**  ({total} records total)")
    with col_next:
        if st.button("Next ▶", disabled=current_page >= total_pages):
            st.session_state[page_key] += 1
            st.rerun()

    st.divider()

    if not ids:
        st.info("No records found.")
        return

    # ── Record cards ──────────────────────────────────────────────────────
    for i, (chunk_id, text, meta) in enumerate(zip(ids, docs_text, metadatas)):
        emb = embeddings[i] if embeddings and i < len(embeddings) else None
        doc_name = meta.get("document_name", "")
        pages = meta.get("source_page_numbers", "[]")
        try:
            pages_list = json.loads(pages) if isinstance(pages, str) else pages
            pages_str = ", ".join(str(p) for p in pages_list)
        except Exception:
            pages_str = str(pages)

        label = f"`{chunk_id[:16]}…`  |  {doc_name}  |  Page(s): {pages_str}"
        with st.expander(label, expanded=False):
            tab_meta, tab_text, tab_emb = st.tabs(["Metadata", "Chunk Text", "Embedding"])

            with tab_meta:
                st.write(f"**Full Chunk ID:** `{chunk_id}`")
                for k, v in meta.items():
                    st.write(f"- **{k}:** `{v}`")

            with tab_text:
                st.text_area(
                    "",
                    value=text,
                    height=160,
                    disabled=True,
                    key=f"vdb_text_{chunk_id}",
                )

            with tab_emb:
                if emb is not None:
                    arr = np.array(emb, dtype=np.float32)
                    st.write(f"**Shape:** `{arr.shape}`  |  **L2 norm:** `{np.linalg.norm(arr):.4f}`")
                    st.write("**First 10 values:**")
                    st.code(", ".join(f"{v:.6f}" for v in arr[:10]))
                else:
                    st.info("Embedding not available in this view.")

    st.divider()

    # ── Danger zone ───────────────────────────────────────────────────────
    st.subheader("⚠️ Collection Management")
    if st.button("🧨 Clear Entire Collection", type="secondary"):
        st.session_state["confirm_clear_vdb"] = True

    if st.session_state.get("confirm_clear_vdb"):
        st.error(
            "This will **permanently delete all vectors** in the collection. "
            "Document registry entries will be marked as deleted."
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Confirm Clear"):
                clear_collection()
                for doc in docs:
                    mark_deleted(doc.document_id)
                st.session_state.pop("confirm_clear_vdb", None)
                st.session_state[page_key] = 0
                st.success("Collection cleared.")
                st.rerun()
        with c2:
            if st.button("❌ Cancel"):
                st.session_state.pop("confirm_clear_vdb", None)
                st.rerun()
