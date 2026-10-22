"""
Chunk Explorer UI.
Paginated browsable list of all generated chunks with filtering.
"""

import streamlit as st

from config.settings import CHUNKS_PER_PAGE
from services.chunking_service import get_chunk_stats
from storage.document_registry import list_documents


def render_chunk_explorer():
    st.header("🧩 Chunk Explorer")
    st.markdown(
        "Browse every chunk generated from your PDFs. "
        "Chunks preserve page-level source information and are used as the "
        "retrieval unit for the vector similarity search."
    )

    docs = [d for d in list_documents() if d.status == "indexed"]
    if not docs:
        st.info("No indexed documents yet.")
        return

    # ── Controls ──────────────────────────────────────────────────────────
    col_filter, col_page_filter = st.columns(2)
    with col_filter:
        doc_options = {"All documents": None}
        doc_options.update({d.filename: d.document_id for d in docs})
        selected_label = st.selectbox("Filter by document", list(doc_options.keys()))
        selected_doc_id = doc_options[selected_label]

    # ── Gather chunks ──────────────────────────────────────────────────────
    all_chunks = []
    for doc in docs:
        cache_key = f"chunks_{doc.document_id}"
        if cache_key in st.session_state:
            chunks = st.session_state[cache_key]
            if selected_doc_id is None or doc.document_id == selected_doc_id:
                all_chunks.extend(chunks)

    if not all_chunks:
        st.info(
            "Chunk data not in memory. "
            "Re-upload and process the document(s) to populate the explorer."
        )
        return

    # ── Page filter ────────────────────────────────────────────────────────
    with col_page_filter:
        all_pages = sorted({p for c in all_chunks for p in c.source_page_numbers})
        page_filter = st.selectbox(
            "Filter by source page",
            ["All pages"] + [str(p) for p in all_pages],
        )

    if page_filter != "All pages":
        page_num = int(page_filter)
        all_chunks = [c for c in all_chunks if page_num in c.source_page_numbers]

    # ── Stats ──────────────────────────────────────────────────────────────
    stats = get_chunk_stats(all_chunks)
    st.subheader("📊 Chunk Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Chunks", stats["total"])
    c2.metric("Avg Chars", stats["avg_chars"])
    c3.metric("Min Chars", stats["min_chars"])
    c4.metric("Max Chars", stats["max_chars"])

    # ── Overlap explanation ───────────────────────────────────────────────
    with st.expander("ℹ️ How does chunk overlap work?", expanded=False):
        st.markdown(
            """
**Chunk overlap** means that adjacent chunks share a small portion of text.

For example, with **chunk_size=800** and **chunk_overlap=120**:
- Chunk 0 contains characters 0–800
- Chunk 1 contains characters 680–1480 (starts 120 chars before where chunk 0 ended)
- Chunk 2 contains characters 1360–2160

This ensures that sentences or concepts split across chunk boundaries are still
retrievable in at least one chunk, improving recall during similarity search.
            """
        )

    st.divider()

    # ── Pagination ────────────────────────────────────────────────────────
    total = len(all_chunks)
    total_pages = max(1, (total + CHUNKS_PER_PAGE - 1) // CHUNKS_PER_PAGE)

    page_key = "chunk_explorer_page"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1

    col_prev, col_pager, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("◀ Prev") and st.session_state[page_key] > 1:
            st.session_state[page_key] -= 1
    with col_pager:
        st.caption(
            f"Page **{st.session_state[page_key]}** of **{total_pages}**  "
            f"({total} chunks total)"
        )
    with col_next:
        if st.button("Next ▶") and st.session_state[page_key] < total_pages:
            st.session_state[page_key] += 1

    page_num = st.session_state[page_key]
    start = (page_num - 1) * CHUNKS_PER_PAGE
    end = start + CHUNKS_PER_PAGE
    page_chunks = all_chunks[start:end]

    # ── Chunk cards ───────────────────────────────────────────────────────
    for chunk in page_chunks:
        pages_str = ", ".join(str(p) for p in chunk.source_page_numbers)
        label = (
            f"Chunk #{chunk.chunk_index}  |  {chunk.document_name}  "
            f"|  Page(s): {pages_str}  |  {chunk.char_count} chars"
        )
        with st.expander(label, expanded=False):
            col_meta, col_text = st.columns([2, 3])
            with col_meta:
                st.markdown("**Metadata**")
                st.write(f"- **Chunk ID:** `{chunk.chunk_id}`")
                st.write(f"- **Document:** {chunk.document_name}")
                st.write(f"- **Source pages:** {pages_str}")
                st.write(f"- **Characters:** {chunk.char_count}")
                st.write(f"- **Chunk size setting:** {chunk.chunk_size}")
                st.write(f"- **Overlap setting:** {chunk.chunk_overlap}")
                st.write(f"- **Embedding model:** `{chunk.embedding_model}`")
            with col_text:
                st.markdown("**Chunk Text**")
                st.text_area(
                    "",
                    value=chunk.chunk_text,
                    height=180,
                    disabled=True,
                    key=f"chunk_text_{chunk.chunk_id}",
                )
