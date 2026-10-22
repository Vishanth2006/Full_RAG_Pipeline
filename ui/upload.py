"""
Document Upload and Processing UI.
Handles file upload, validation, duplicate detection, and ingestion pipeline.
"""

import streamlit as st
from datetime import datetime

from config.settings import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    MAX_UPLOAD_SIZE_MB,
    STATUS_DUPLICATE,
    STATUS_INDEXED,
)
from services.document_service import (
    ingest_document,
    prepare_document_record,
    validate_upload,
)
from services.pdf_service import compute_sha256
from services.vector_store_service import delete_document_chunks
from storage.document_registry import (
    delete_document,
    list_documents,
    mark_deleted,
)


def render_upload():
    st.header("📤 Document Upload & Processing")
    st.markdown(
        "Upload one or more PDF files, then click **Process Documents** to run "
        "the full ingestion pipeline (extraction → chunking → embedding → indexing)."
    )

    # ── Chunking settings (from session or defaults) ──────────────────────
    chunk_size = st.session_state.get("chunk_size", DEFAULT_CHUNK_SIZE)
    chunk_overlap = st.session_state.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)

    # ── File uploader ─────────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Maximum file size: {MAX_UPLOAD_SIZE_MB} MB per file.",
    )

    if uploaded_files:
        st.subheader("📋 Upload Queue")
        valid_files = []

        for uf in uploaded_files:
            file_bytes = uf.read()
            ok, err = validate_upload(uf.name, file_bytes)

            if not ok:
                st.error(f"❌ **{uf.name}**: {err}")
                continue

            doc_id = compute_sha256(file_bytes)
            existing = _find_existing(doc_id)

            col_name, col_size, col_pages, col_status = st.columns([3, 2, 1, 2])
            with col_name:
                st.write(f"📄 **{uf.name}**")
                st.caption(f"ID: `{doc_id[:16]}…`")
            with col_size:
                st.write(f"{len(file_bytes)/1024:.1f} KB")
            with col_pages:
                from services.pdf_service import get_page_count
                pg = get_page_count(file_bytes)
                st.write(f"{pg} pp.")
            with col_status:
                if existing:
                    if existing.status == STATUS_INDEXED:
                        st.warning("⚠️ Already indexed")
                    else:
                        st.info(f"ℹ️ Status: `{existing.status}`")
                else:
                    st.success("🆕 New file")

            valid_files.append((uf.name, file_bytes, doc_id, existing))

        if valid_files:
            st.divider()
            col_btn, col_note = st.columns([2, 5])
            with col_btn:
                process_clicked = st.button(
                    "⚡ Process Documents",
                    type="primary",
                    use_container_width=True,
                )
            with col_note:
                st.caption(
                    f"Chunk size: **{chunk_size}** chars | "
                    f"Overlap: **{chunk_overlap}** chars | "
                    f"Model: **{EMBEDDING_MODEL_NAME}**"
                )

            if process_clicked:
                _run_ingestion(valid_files, chunk_size, chunk_overlap)

    st.divider()

    # ── Existing documents table ──────────────────────────────────────────
    _render_document_manager()


def _run_ingestion(
    valid_files: list,
    chunk_size: int,
    chunk_overlap: int,
):
    """Process each file through the ingestion pipeline."""
    results = {"indexed": 0, "duplicate": 0, "failed": 0, "empty": 0}

    for filename, file_bytes, doc_id, existing in valid_files:
        if existing and existing.status == STATUS_INDEXED:
            st.warning(f"⚠️ **{filename}** is already indexed — skipping.")
            results["duplicate"] += 1
            continue

        st.markdown(f"**Processing: {filename}**")
        progress_bar = st.progress(0.0, text="Starting…")
        status_text = st.empty()

        def make_callback(pb, st_):
            def callback(step: str, pct: float):
                pb.progress(pct, text=step)
                st_.caption(step)
            return callback

        try:
            record, pages, chunks = ingest_document(
                filename=filename,
                file_bytes=file_bytes,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_model=EMBEDDING_MODEL_NAME,
                progress_callback=make_callback(progress_bar, status_text),
            )
            progress_bar.progress(1.0, text="✅ Done")

            if record.status == STATUS_INDEXED:
                st.success(
                    f"✅ **{filename}** indexed successfully — "
                    f"{record.chunk_count} chunks in {record.processing_time_s:.1f}s"
                )
                results["indexed"] += 1

                # Cache pages/chunks in session for the explorers
                st.session_state[f"pages_{doc_id}"] = pages
                st.session_state[f"chunks_{doc_id}"] = chunks

            elif record.status in ("empty_text", "empty"):
                st.warning(
                    f"⚠️ **{filename}**: No extractable text found. "
                    "The document may be image-only or scanned. OCR is not supported."
                )
                results["empty"] += 1
            else:
                st.error(f"❌ **{filename}** failed: {record.error_message}")
                results["failed"] += 1

        except Exception as exc:
            st.error(f"❌ **{filename}** — unexpected error: {exc}")
            results["failed"] += 1

    # Summary
    st.divider()
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("✅ Indexed", results["indexed"])
    s2.metric("⚠️ Duplicates", results["duplicate"])
    s3.metric("❌ Failed", results["failed"])
    s4.metric("📭 Empty", results["empty"])


def _render_document_manager():
    """List indexed documents with delete controls."""
    st.subheader("📚 Indexed Documents")

    docs = list_documents()
    if not docs:
        st.info("No documents in the workspace yet.")
        return

    for doc in docs:
        status_icon = {
            "indexed": "✅",
            "failed": "❌",
            "processing": "⏳",
            "pending": "🕐",
            "duplicate": "🔄",
            "empty_text": "⚠️",
        }.get(doc.status, "❓")

        with st.expander(
            f"{status_icon} {doc.filename}  —  {doc.chunk_count} chunks  |  `{doc.status}`",
            expanded=False,
        ):
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Document ID:** `{doc.document_id}`")
                st.write(f"**Filename:** {doc.filename}")
                st.write(f"**Size:** {doc.file_size_bytes / 1024:.1f} KB")
                st.write(f"**Pages:** {doc.page_count}")
            with c2:
                st.write(f"**Status:** `{doc.status}`")
                st.write(f"**Chunks:** {doc.chunk_count}")
                st.write(f"**Uploaded:** {doc.upload_time[:19]}")
                st.write(f"**Embedding model:** `{doc.embedding_model or 'N/A'}`")
                if doc.chunk_size:
                    st.write(
                        f"**Chunking:** size={doc.chunk_size}, overlap={doc.chunk_overlap}"
                    )
                if doc.error_message:
                    st.error(f"Error: {doc.error_message}")

            col_del, col_conf = st.columns([2, 4])
            with col_del:
                if st.button(
                    "🗑️ Delete Document",
                    key=f"del_{doc.document_id}",
                    help="Remove this document and all its vectors.",
                ):
                    st.session_state[f"confirm_delete_{doc.document_id}"] = True

            if st.session_state.get(f"confirm_delete_{doc.document_id}"):
                with col_conf:
                    st.warning(
                        f"⚠️ Delete **{doc.filename}** and all its {doc.chunk_count} vectors?"
                    )
                    yes, no = st.columns(2)
                    with yes:
                        if st.button("✅ Confirm", key=f"yes_{doc.document_id}"):
                            remaining = delete_document_chunks(doc.document_id)
                            mark_deleted(doc.document_id)
                            st.session_state.pop(
                                f"confirm_delete_{doc.document_id}", None
                            )
                            st.success(
                                f"Deleted **{doc.filename}**. {remaining} vectors remain."
                            )
                            st.rerun()
                    with no:
                        if st.button("❌ Cancel", key=f"no_{doc.document_id}"):
                            st.session_state.pop(
                                f"confirm_delete_{doc.document_id}", None
                            )
                            st.rerun()

    st.divider()
    if st.button("🧹 Clear Entire Workspace", type="secondary"):
        st.session_state["confirm_clear_all"] = True

    if st.session_state.get("confirm_clear_all"):
        st.warning(
            "⚠️ This will **permanently delete ALL documents, vectors, and history**. "
            "Are you absolutely sure?"
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Yes, clear everything"):
                from services.vector_store_service import clear_collection
                from storage.history_repository import clear_history
                for doc in docs:
                    delete_document(doc.document_id)
                clear_collection()
                clear_history()
                st.session_state.pop("confirm_clear_all", None)
                st.success("Workspace cleared.")
                st.rerun()
        with c2:
            if st.button("❌ No, cancel"):
                st.session_state.pop("confirm_clear_all", None)
                st.rerun()


def _find_existing(doc_id: str):
    from storage.document_registry import get_document
    return get_document(doc_id)
