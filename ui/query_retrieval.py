"""
Query & Retrieval UI.
Embeds user query and displays retrieved chunks with scores.
"""

import streamlit as st
import numpy as np

from config.settings import DEFAULT_TOP_K, EMBEDDING_MODEL_NAME, MAX_TOP_K
from services.retrieval_service import retrieve_chunks
from storage.document_registry import list_documents


def render_query_retrieval():
    st.header("🔍 Query & Retrieval")
    st.markdown(
        "Ask a question in natural language. Your query is embedded using the "
        "same model as the documents, then ChromaDB returns the most relevant chunks."
    )

    docs = [d for d in list_documents() if d.status == "indexed"]
    if not docs:
        st.warning("No indexed documents. Please upload and process PDFs first.")
        return

    # ── Query controls ────────────────────────────────────────────────────
    col_q, col_k = st.columns([5, 1])
    with col_q:
        query_text = st.text_area(
            "Your question",
            value=st.session_state.get("current_query", ""),
            height=80,
            placeholder="e.g. What are the main conclusions of the study?",
            key="query_input",
        )
    with col_k:
        top_k = st.slider(
            "Top-k", min_value=1, max_value=MAX_TOP_K,
            value=st.session_state.get("top_k", DEFAULT_TOP_K),
            key="top_k_slider",
        )

    # ── Document filter ────────────────────────────────────────────────────
    doc_names = {d.document_id: d.filename for d in docs}
    all_doc_ids = [d.document_id for d in docs]

    filter_mode = st.radio(
        "Search scope",
        ["🌐 Workspace-wide (all documents)", "📄 Specific document(s)"],
        horizontal=True,
        key="search_scope",
    )

    selected_doc_ids = None
    if "Specific" in filter_mode:
        selected_names = st.multiselect(
            "Select document(s)",
            options=[d.filename for d in docs],
        )
        selected_doc_ids = [
            d.document_id for d in docs if d.filename in selected_names
        ]
        if not selected_doc_ids:
            st.warning("Select at least one document to search.")

    col_search, col_reset = st.columns([2, 1])
    with col_search:
        search_clicked = st.button(
            "🔎 Search Documents",
            type="primary",
            use_container_width=True,
        )
    with col_reset:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.pop("last_retrieval", None)
            st.session_state.pop("current_query", None)
            st.rerun()

    if search_clicked:
        if not query_text.strip():
            st.error("Please enter a question before searching.")
            return
        if "Specific" in filter_mode and not selected_doc_ids:
            st.error("Select at least one document to search.")
            return

        with st.spinner("Embedding query and searching…"):
            try:
                results, query_vec, duration = retrieve_chunks(
                    query_text=query_text,
                    top_k=top_k,
                    document_ids=selected_doc_ids,
                    embedding_model=EMBEDDING_MODEL_NAME,
                )
                st.session_state["last_retrieval"] = {
                    "query": query_text,
                    "results": results,
                    "query_vec": query_vec,
                    "duration": duration,
                    "top_k": top_k,
                    "doc_filter": selected_doc_ids,
                }
                st.session_state["current_query"] = query_text
            except Exception as exc:
                st.error(f"Retrieval error: {exc}")
                return

    # ── Display results ───────────────────────────────────────────────────
    retrieval = st.session_state.get("last_retrieval")
    if retrieval:
        _render_results(retrieval)


def _render_results(retrieval: dict):
    results = retrieval["results"]
    query_vec = retrieval["query_vec"]
    duration = retrieval["duration"]
    query = retrieval["query"]

    st.divider()
    st.subheader("📊 Retrieval Results")

    m1, m2, m3 = st.columns(3)
    m1.metric("Results Returned", len(results))
    m2.metric("Search Duration", f"{duration*1000:.0f} ms")
    m3.metric("Query Vector Dims", f"{len(query_vec):,}")

    # Query vector preview
    with st.expander("🔢 Query Embedding Vector", expanded=False):
        arr = np.array(query_vec, dtype=np.float32)
        st.write(
            f"**Shape:** `{arr.shape}`  |  "
            f"**L2 norm:** `{np.linalg.norm(arr):.4f}`  |  "
            f"**Model:** `{EMBEDDING_MODEL_NAME}`"
        )
        st.write("**First 10 values:**")
        st.code(", ".join(f"{v:.6f}" for v in arr[:10]))

    if not results:
        st.info(
            "No results returned. "
            "The database may be empty or no chunks matched the filter."
        )
        return

    st.markdown(
        "> **Note:** Cosine similarity = 1 − cosine distance. "
        "This is a derived similarity score, **not** a probability or confidence percentage."
    )

    for chunk in results:
        pages_str = ", ".join(str(p) for p in chunk.source_page_numbers)
        label = (
            f"Rank #{chunk.rank}  |  {chunk.document_name}  "
            f"|  Page(s): {pages_str}  "
            f"|  Similarity: {chunk.cosine_similarity:.4f}"
        )

        # Colour bar for similarity
        sim_pct = int(chunk.cosine_similarity * 100)
        color = "#22c55e" if sim_pct > 70 else "#f59e0b" if sim_pct > 40 else "#ef4444"

        with st.expander(label, expanded=chunk.rank == 1):
            c_left, c_right = st.columns([3, 2])
            with c_left:
                st.markdown(f"**Chunk ID:** `{chunk.chunk_id}`")
                st.markdown(f"**Document:** `{chunk.document_name}`")
                st.markdown(f"**Source Pages:** {pages_str}")
                st.text_area(
                    "Retrieved Text",
                    value=chunk.chunk_text,
                    height=150,
                    disabled=True,
                    key=f"retr_text_{chunk.chunk_id}_{chunk.rank}",
                )
            with c_right:
                st.markdown(f"**Cosine Distance:** `{chunk.cosine_distance:.6f}`")
                st.markdown(f"**Cosine Similarity:** `{chunk.cosine_similarity:.6f}`")
                st.markdown(
                    f"<div style='width:100%;background:#e5e7eb;border-radius:6px;height:16px;'>"
                    f"<div style='width:{sim_pct}%;background:{color};height:16px;border-radius:6px;'></div>"
                    f"</div><p style='font-size:12px;margin-top:4px;'>{sim_pct}%</p>",
                    unsafe_allow_html=True,
                )
