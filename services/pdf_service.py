"""
PDF text extraction service using PyMuPDF (fitz).
Extracts text page-by-page, preserving page boundaries and reporting warnings
for empty/image-only pages.
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from models.schemas import PageExtractionResult

logger = logging.getLogger(__name__)


def compute_sha256(file_bytes: bytes) -> str:
    """Return the SHA-256 hex digest of raw file content."""
    return hashlib.sha256(file_bytes).hexdigest()


def get_page_count(file_bytes: bytes) -> int:
    """Return page count without full extraction."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        count = doc.page_count
        doc.close()
        return count
    except Exception as exc:
        logger.warning("Could not determine page count: %s", exc)
        return 0


def extract_text_from_bytes(
    file_bytes: bytes,
    document_id: str,
    document_name: str,
) -> list[PageExtractionResult]:
    """
    Extract text from every page of a PDF supplied as raw bytes.

    Returns a list of PageExtractionResult objects (one per page).
    Pages with no text are flagged with a warning rather than silently omitted.
    """
    results: list[PageExtractionResult] = []

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        logger.error("Failed to open PDF '%s': %s", document_name, exc)
        raise ValueError(f"Cannot open PDF '{document_name}': {exc}") from exc

    try:
        for page_index in range(doc.page_count):
            page_number = page_index + 1  # 1-indexed
            try:
                page = doc.load_page(page_index)
                text = page.get_text("text")

                if not text or not text.strip():
                    warning = (
                        "No extractable text found on this page. "
                        "The page may be image-only or scanned. "
                        "OCR is not enabled in this build."
                    )
                    results.append(
                        PageExtractionResult(
                            document_id=document_id,
                            document_name=document_name,
                            page_number=page_number,
                            text="",
                            char_count=0,
                            status="empty",
                            warning=warning,
                        )
                    )
                else:
                    results.append(
                        PageExtractionResult(
                            document_id=document_id,
                            document_name=document_name,
                            page_number=page_number,
                            text=text,
                            char_count=len(text),
                            status="ok",
                        )
                    )
            except Exception as page_exc:
                logger.warning(
                    "Error extracting page %d from '%s': %s",
                    page_number, document_name, page_exc,
                )
                results.append(
                    PageExtractionResult(
                        document_id=document_id,
                        document_name=document_name,
                        page_number=page_number,
                        text="",
                        char_count=0,
                        status="error",
                        warning=str(page_exc),
                    )
                )
    finally:
        doc.close()

    return results


def has_any_text(extraction_results: list[PageExtractionResult]) -> bool:
    """Return True if at least one page yielded non-empty text."""
    return any(r.status == "ok" and r.char_count > 0 for r in extraction_results)


def summarise_extraction(results: list[PageExtractionResult]) -> dict:
    """Return a dict with aggregate statistics for display."""
    total_pages = len(results)
    ok_pages = sum(1 for r in results if r.status == "ok")
    empty_pages = sum(1 for r in results if r.status == "empty")
    error_pages = sum(1 for r in results if r.status == "error")
    total_chars = sum(r.char_count for r in results)
    return {
        "total_pages": total_pages,
        "ok_pages": ok_pages,
        "empty_pages": empty_pages,
        "error_pages": error_pages,
        "total_chars": total_chars,
    }
