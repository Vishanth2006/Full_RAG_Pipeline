"""
Data Explorer — combined Vector Database + Embedding inspector.
Merges the previous Embedding Explorer and Vector Database Explorer into one page.
Also provides a "Clear Everything" option to wipe all databases for a fresh start.
"""

import json
import numpy as np
import streamlit as st

import config.settings as _settings
from services.vector_store_service import (
    clear_collection,
    delete_document_chunks,
    get_all_records,
    get_collection_stats,
)
from storage.document_registry import delete_document, list_documents
from storage.history_repository import clear_history


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_list(val):
    """Convert any array-like to a plain Python list (avoids numpy truthiness bug)."""
    if val is None:
        return []
    try:
        if hasattr(val, "tolist"):
            return val.tolist()
        return list(val)
    except Exception:
        return []


def _emb_safe(raw) -> np.ndarray | None:
    """Return a float32 numpy array or None."""
    if raw is None:
        return None
    try:
        arr = np.array(raw, dtype=np.float32)
        return arr if arr.ndim == 1 and len(arr) > 0 else None
    except Exception:
        return None


# ── main renderer ─────────────────────────────────────────────────────────────

def render_data_explorer():
    st.header("🗄️ Data Explorer")
    st.markdown(
        f"Collection: `{_settings.CHROMA_COLLECTION_NAME}` &nbsp;|&nbsp; "
        f"Path: `{_settings.CHROMA_PERSIST_DIR}` &nbsp;|&nbsp; "
        f"Similarity: **cosine** &nbsp;|&nbsp; "
        f"Embedding model: `{_settings.EMBEDDING_MODEL_NAME}` ({_settings.EMBEDDING_DIMENSIONS}‑dim)"
    )

    # ── Stats bar ─────────────────────────────────────────────────────────────
    try:
        stats = get_collection_stats()
        total_vectors = stats.get("total_records", 0)
    except Exception as exc:
        st.error(f"Cannot connect to ChromaDB: {exc}")
        return

    all_docs = list_documents()
    indexed_docs = [d for d in all_docs if d.status == "indexed"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🔢 Total Vectors", total_vectors)
    m2.metric("📄 Indexed Docs", len(indexed_docs))
    m3.metric("📦 Collection", _settings.CHROMA_COLLECTION_NAME)
    m4.metric("📐 Embedding Dim", _settings.EMBEDDING_DIMENSIONS)

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_records, tab_embed, tab_danger = st.tabs([
        "🗄️ Records Browser",
        "🔢 Embedding Inspector",
        "🧨 Clear Everything",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Records Browser
    # ══════════════════════════════════════════════════════════════════════════
    with tab_records:
        _render_records_browser(indexed_docs, total_vectors)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Embedding Inspector
    # ══════════════════════════════════════════════════════════════════════════
    with tab_embed:
        _render_embedding_inspector(indexed_docs)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — Clear Everything
    # ══════════════════════════════════════════════════════════════════════════
    with tab_danger:
        _render_clear_zone(all_docs)


# ── Tab 1: Records Browser ────────────────────────────────────────────────────

def _render_records_browser(indexed_docs, total_vectors):
    st.subheader("📋 Browse Vector Records")

    # Filter + per-page controls
    col_filter, col_per_page = st.columns([3, 1])
    with col_filter:
        doc_options = {"All documents": None}
        doc_options.update({d.filename: d.document_id for d in indexed_docs})
        selected_label = st.selectbox(
            "Filter by document", list(doc_options.keys()), key="rb_filter"
        )
        filter_doc_id = doc_options[selected_label]

    with col_per_page:
        per_page = st.number_input(
            "Per page", min_value=5, max_value=50,
            value=_settings.VECTORS_PER_PAGE, key="rb_per_page"
        )

    # Pagination state
    page_key = "explorer_vdb_page"
    if page_key not in st.session_state:
        st.session_state[page_key] = 0
    offset = st.session_state[page_key] * per_page

    # Fetch
    try:
        records = get_all_records(document_id=filter_doc_id, limit=per_page, offset=offset)
    except Exception as exc:
        st.error(f"Failed to fetch records from ChromaDB: {exc}")
        return

    ids        = _safe_list(records.get("ids", []))
    docs_text  = _safe_list(records.get("documents", []))
    metadatas  = _safe_list(records.get("metadatas", []))
    raw_embs   = records.get("embeddings", [])
    total      = records.get("total", 0)

    # Make embeddings a safe list
    embeddings = _safe_list(raw_embs)

    # Pager
    total_pages  = max(1, (total + per_page - 1) // per_page)
    current_page = st.session_state[page_key] + 1

    col_prev, col_info, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("◀ Prev", disabled=current_page <= 1, key="rb_prev"):
            st.session_state[page_key] -= 1
            st.rerun()
    with col_info:
        st.caption(f"Page **{current_page}** of **{total_pages}**  ({total} records total)")
    with col_next:
        if st.button("Next ▶", disabled=current_page >= total_pages, key="rb_next"):
            st.session_state[page_key] += 1
            st.rerun()

    st.divider()

    if not ids:
        st.info("No records found. Upload and process PDFs first.")
        return

    # Record cards
    for i, (chunk_id, text, meta) in enumerate(zip(ids, docs_text, metadatas)):
        raw_emb  = embeddings[i] if i < len(embeddings) else None
        emb_arr  = _emb_safe(raw_emb)
        doc_name = meta.get("document_name", "")
        pages    = meta.get("source_page_numbers", "[]")
        try:
            pages_list = json.loads(pages) if isinstance(pages, str) else list(pages)
            pages_str  = ", ".join(str(p) for p in pages_list)
        except Exception:
            pages_str = str(pages)

        label = f"`{chunk_id[:18]}…`  |  {doc_name}  |  Page(s): {pages_str}"
        with st.expander(label, expanded=False):
            t_meta, t_text, t_emb = st.tabs(["Metadata", "Chunk Text", "Embedding"])

            with t_meta:
                st.write(f"**Full Chunk ID:** `{chunk_id}`")
                for k, v in meta.items():
                    st.write(f"- **{k}:** `{v}`")

            with t_text:
                st.text_area(
                    "", value=text, height=160, disabled=True,
                    key=f"rb_text_{chunk_id}_{i}"
                )

            with t_emb:
                if emb_arr is not None:
                    st.write(
                        f"**Shape:** `{emb_arr.shape}`  |  "
                        f"**L2 norm:** `{np.linalg.norm(emb_arr):.4f}`"
                    )
                    st.write("**First 10 values:**")
                    st.code(", ".join(f"{v:.6f}" for v in emb_arr[:10]))
                else:
                    st.info("Embedding not available.")


# ── Tab 2: Embedding Inspector ────────────────────────────────────────────────

def _render_embedding_inspector(indexed_docs):
    st.subheader("🔢 Inspect a Chunk's Embedding Vector")
    st.markdown(
        f"Model: `{_settings.EMBEDDING_MODEL_NAME}` &nbsp;|&nbsp; "
        f"Dimensions: **{_settings.EMBEDDING_DIMENSIONS}**"
    )

    if not indexed_docs:
        st.info("No indexed documents. Upload and process PDFs first.")
        return

    # Document selector
    doc_options  = {d.filename: d.document_id for d in indexed_docs}
    selected_name = st.selectbox("Select document", list(doc_options.keys()), key="ei_doc")
    selected_doc_id = doc_options[selected_name]

    # Fetch all records for this document
    try:
        records = get_all_records(document_id=selected_doc_id, limit=500)
    except Exception as exc:
        st.error(f"Failed to fetch records from ChromaDB: {exc}")
        return

    ids       = _safe_list(records.get("ids", []))
    docs_text = _safe_list(records.get("documents", []))
    metadatas = _safe_list(records.get("metadatas", []))
    raw_embs  = records.get("embeddings", [])
    embeddings = _safe_list(raw_embs)
    total     = records.get("total", 0)

    if not ids:
        st.info("No records found for this document in ChromaDB.")
        return

    st.info(f"Found **{total}** indexed chunks for **{selected_name}**.")

    # Chunk selector
    chunk_labels = [
        f"Chunk {i}: {txt[:65]}…" if len(txt) > 65 else f"Chunk {i}: {txt}"
        for i, txt in enumerate(docs_text)
    ]
    selected_idx = st.selectbox(
        "Select chunk to inspect",
        range(len(chunk_labels)),
        format_func=lambda i: chunk_labels[i],
        key="ei_chunk",
    )

    chunk_id   = ids[selected_idx]
    chunk_text = docs_text[selected_idx]
    meta       = metadatas[selected_idx] if selected_idx < len(metadatas) else {}
    raw_emb    = embeddings[selected_idx] if selected_idx < len(embeddings) else None
    emb_arr    = _emb_safe(raw_emb)

    st.divider()

    # Chunk info
    st.subheader("📄 Chunk Information")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Chunk ID:** `{chunk_id}`")
        st.write(f"**Document:** {meta.get('document_name', 'N/A')}")
        st.write(f"**Pages:** {meta.get('source_page_numbers', 'N/A')}")
    with c2:
        st.write(f"**Characters:** {meta.get('char_count', len(chunk_text))}")
        st.write(f"**Embedding model:** `{meta.get('embedding_model', _settings.EMBEDDING_MODEL_NAME)}`")
        st.write(f"**Vector shape:** `({_settings.EMBEDDING_DIMENSIONS},)`")

    st.text_area(
        "Chunk Text", value=chunk_text, height=120, disabled=True,
        key=f"ei_text_{chunk_id}",
    )

    st.divider()

    # Embedding display
    st.subheader("🔢 Embedding Vector")

    if emb_arr is None:
        st.warning(
            "Embedding not returned by ChromaDB for this chunk. "
            "Ensure the collection was created with `include=['embeddings']`."
        )
        return

    st.write(f"**Shape:** `{emb_arr.shape}`  |  **dtype:** `{emb_arr.dtype}`")

    col_preview, col_norm = st.columns(2)
    with col_preview:
        st.markdown("**First 10 values:**")
        st.dataframe(
            {"Index": list(range(10)), "Value": [f"{v:.6f}" for v in emb_arr[:10]]},
            use_container_width=True, hide_index=True,
        )
    with col_norm:
        norm = float(np.linalg.norm(emb_arr))
        st.metric("L2 Norm", f"{norm:.6f}", help="Should be ≈ 1.0 for normalised embeddings")
        st.metric("Min value", f"{emb_arr.min():.6f}")
        st.metric("Max value", f"{emb_arr.max():.6f}")

    # Full vector toggle
    if st.checkbox("Show all values", key=f"ei_show_all_{chunk_id}"):
        import pandas as pd
        df = pd.DataFrame({"Index": range(len(emb_arr)), "Value": emb_arr.tolist()})
        st.dataframe(df, use_container_width=True, height=400, hide_index=True)

    # Heatmap
    st.subheader("🌡️ Vector Heatmap")
    st.caption(
        "Each cell = one embedding dimension. Brighter = higher value. "
        "Cosine similarity uses relative magnitudes."
    )
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        reshaped = emb_arr.reshape(16, 24)  # 16 × 24 = 384
        fig, ax = plt.subplots(figsize=(10, 3.5))
        im = ax.imshow(reshaped, aspect="auto", cmap="RdYlGn", interpolation="nearest")
        ax.set_title(f"Embedding Heatmap — Chunk {selected_idx}", fontsize=11)
        ax.set_xlabel("Dimension (column within row)")
        ax.set_ylabel("Row group (×24 dims each)")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except ImportError:
        st.info("Install matplotlib for the heatmap visualisation.")


# ── Tab 3: Clear Everything ───────────────────────────────────────────────────

def _render_clear_zone(all_docs):
    st.subheader("🧨 Clear Everything")
    st.markdown(
        "Use this to **wipe all data** and start fresh. "
        "This is useful for testing a clean pipeline run."
    )

    # What will be cleared
    indexed = [d for d in all_docs if d.status == "indexed"]
    try:
        stats = get_collection_stats()
        n_vectors = stats.get("total_records", 0)
    except Exception:
        n_vectors = "?"

    st.info(
        f"**What will be deleted:**\n"
        f"- 🔢 **{n_vectors}** vectors from ChromaDB collection `{_settings.CHROMA_COLLECTION_NAME}`\n"
        f"- 📄 **{len(all_docs)}** document records from the registry (including deleted)\n"
        f"- 📜 All query history entries\n\n"
        f"⚠️ This action **cannot be undone**."
    )

    st.divider()

    # Option A — Clear everything
    st.markdown("#### Option A — Wipe Everything")
    if st.button("🧨 Clear ALL Databases", type="primary", key="clear_all_btn"):
        st.session_state["confirm_clear_all"] = True

    if st.session_state.get("confirm_clear_all"):
        st.error("Are you absolutely sure? This will delete ALL vectors, ALL documents, and ALL history.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Yes, wipe everything", key="confirm_clear_all_yes"):
                with st.spinner("Clearing all databases..."):
                    _do_clear_all(all_docs)
                st.session_state.pop("confirm_clear_all", None)
                st.success("✅ All databases cleared. You can now upload fresh PDFs.")
                st.rerun()
        with c2:
            if st.button("❌ Cancel", key="confirm_clear_all_no"):
                st.session_state.pop("confirm_clear_all", None)
                st.rerun()

    st.divider()

    # Option B — Clear only vectors (keep doc registry)
    st.markdown("#### Option B — Clear Vectors Only")
    st.caption("Wipes ChromaDB vectors but keeps document and history records.")
    if st.button("🗑️ Clear Vector Store Only", type="secondary", key="clear_vectors_btn"):
        st.session_state["confirm_clear_vectors"] = True

    if st.session_state.get("confirm_clear_vectors"):
        st.warning("This will delete all vectors but keep the document registry and history.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Confirm", key="confirm_clear_vectors_yes"):
                with st.spinner("Clearing vector store..."):
                    clear_collection()
                    for doc in indexed:
                        from storage.document_registry import mark_deleted
                        mark_deleted(doc.document_id)
                st.session_state.pop("confirm_clear_vectors", None)
                st.success("✅ Vector store cleared.")
                st.rerun()
        with c2:
            if st.button("❌ Cancel", key="confirm_clear_vectors_no"):
                st.session_state.pop("confirm_clear_vectors", None)
                st.rerun()

    st.divider()

    # Option C — Clear history only
    st.markdown("#### Option C — Clear Query History Only")
    st.caption("Deletes all saved query/answer history. Documents and vectors are untouched.")
    if st.button("📜 Clear History Only", type="secondary", key="clear_history_btn"):
        with st.spinner("Clearing history..."):
            clear_history()
        st.success("✅ Query history cleared.")
        st.rerun()


def _do_clear_all(all_docs):
    """Wipe ChromaDB, document registry, and query history."""
    # 1. Clear ChromaDB
    try:
        clear_collection()
    except Exception:
        pass

    # 2. Delete all document records
    for doc in all_docs:
        try:
            delete_document(doc.document_id)
        except Exception:
            pass

    # 3. Clear history
    try:
        clear_history()
    except Exception:
        pass
