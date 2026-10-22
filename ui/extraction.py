"""
Text Extraction Explorer UI.
Displays per-page extraction results with char counts and warnings.
"""

import streamlit as st

from services.pdf_service import (
    extract_text_from_bytes,
    summarise_extraction,
)
from storage.document_registry import list_documents


def render_extraction():
    st.header("📖 Text Extraction Explorer")
    st.markdown(
        "Inspect the raw text extracted from each page of your PDF documents. "
        "Pages without extractable text are flagged with explanatory warnings."
    )

    docs = [d for d in list_documents() if d.status in ("indexed", "empty_text")]
    if not docs:
        st.info("No processed documents found. Upload and process PDFs first.")
        return

    doc_names = {d.document_id: d.filename for d in docs}
    selected_id = st.selectbox(
        "Select document",
        options=[d.document_id for d in docs],
        format_func=lambda x: doc_names[x],
    )

    # Retrieve cached pages or re-extract
    cache_key = f"pages_{selected_id}"
    if cache_key not in st.session_state:
        st.info(
            "Extraction results not in memory (e.g. after a restart). "
            "Re-upload and process the document to see page-level results, "
            "or use the Document Upload page."
        )
        return

    pages = st.session_state[cache_key]
    summary = summarise_extraction(pages)

    # ── Summary metrics ────────────────────────────────────────────────────
    st.subheader("📊 Extraction Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Pages", summary["total_pages"])
    c2.metric("✅ OK", summary["ok_pages"])
    c3.metric("⚠️ Empty", summary["empty_pages"])
    c4.metric("❌ Errors", summary["error_pages"])
    c5.metric("Total Chars", f"{summary['total_chars']:,}")

    st.divider()

    # ── Per-page results ───────────────────────────────────────────────────
    st.subheader("📄 Per-Page Results")

    filter_status = st.radio(
        "Filter pages",
        ["All", "OK only", "Empty/Error only"],
        horizontal=True,
    )

    filtered = pages
    if filter_status == "OK only":
        filtered = [p for p in pages if p.status == "ok"]
    elif filter_status == "Empty/Error only":
        filtered = [p for p in pages if p.status != "ok"]

    if not filtered:
        st.info("No pages match the current filter.")
        return

    for page in filtered:
        status_icon = {"ok": "✅", "empty": "⚠️", "error": "❌"}.get(page.status, "❓")
        label = (
            f"{status_icon} Page {page.page_number}  "
            f"— {page.char_count:,} chars  "
            f"| status: `{page.status}`"
        )
        with st.expander(label, expanded=False):
            if page.warning:
                st.warning(page.warning)
            if page.text:
                st.text_area(
                    f"Extracted text (page {page.page_number})",
                    value=page.text,
                    height=200,
                    disabled=True,
                    key=f"page_text_{selected_id}_{page.page_number}",
                )
            else:
                st.caption("No text content on this page.")
