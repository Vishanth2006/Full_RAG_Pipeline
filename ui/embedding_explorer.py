"""
Embedding Explorer UI.
Select any chunk and inspect its stored 384-dimensional embedding vector.
"""

import streamlit as st
import numpy as np

from config.settings import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL_NAME
from services.vector_store_service import get_all_records, get_collection
from storage.document_registry import list_documents


def render_embedding_explorer():
    st.header("🔢 Embedding Explorer")
    st.markdown(
        "Select a chunk to inspect its **stored embedding vector** retrieved from ChromaDB. "
        f"Model: `{EMBEDDING_MODEL_NAME}` | Dimensions: **{EMBEDDING_DIMENSIONS}**"
    )

    docs = [d for d in list_documents() if d.status == "indexed"]
    if not docs:
        st.info("No indexed documents. Upload and process PDFs first.")
        return

    # ── Document selector ─────────────────────────────────────────────────
    doc_options = {d.filename: d.document_id for d in docs}
    selected_name = st.selectbox("Select document", list(doc_options.keys()))
    selected_doc_id = doc_options[selected_name]

    # ── Fetch records from ChromaDB ───────────────────────────────────────
    try:
        records = get_all_records(document_id=selected_doc_id, limit=500)
    except Exception as exc:
        st.error(f"Failed to fetch records from ChromaDB: {exc}")
        return

    ids = records.get("ids", [])
    docs_text = records.get("documents", [])
    metadatas = records.get("metadatas", [])
    embeddings = records.get("embeddings", [])
    total = records.get("total", 0)

    if not ids:
        st.info("No records found for this document in ChromaDB.")
        return

    st.info(f"Found **{total}** indexed chunks for **{selected_name}**.")

    # ── Chunk selector ────────────────────────────────────────────────────
    chunk_labels = [
        f"Chunk {i}: {txt[:60]}…" if len(txt) > 60 else f"Chunk {i}: {txt}"
        for i, txt in enumerate(docs_text)
    ]
    selected_idx = st.selectbox(
        "Select chunk to inspect",
        range(len(chunk_labels)),
        format_func=lambda i: chunk_labels[i],
    )

    chunk_id = ids[selected_idx]
    chunk_text = docs_text[selected_idx]
    meta = metadatas[selected_idx] if selected_idx < len(metadatas) else {}
    embedding = embeddings[selected_idx] if embeddings and selected_idx < len(embeddings) else None

    st.divider()

    # ── Chunk info ────────────────────────────────────────────────────────
    st.subheader("📄 Chunk Information")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Chunk ID:** `{chunk_id}`")
        st.write(f"**Document:** {meta.get('document_name', 'N/A')}")
        st.write(f"**Pages:** {meta.get('source_page_numbers', 'N/A')}")
    with c2:
        st.write(f"**Characters:** {meta.get('char_count', len(chunk_text))}")
        st.write(f"**Embedding model:** `{meta.get('embedding_model', EMBEDDING_MODEL_NAME)}`")
        st.write(f"**Vector shape:** `({EMBEDDING_DIMENSIONS},)`")

    st.text_area(
        "Chunk Text",
        value=chunk_text,
        height=120,
        disabled=True,
        key=f"emb_exp_text_{chunk_id}",
    )

    st.divider()

    # ── Embedding display ─────────────────────────────────────────────────
    st.subheader("🔢 Embedding Vector")

    if embedding is None:
        st.warning(
            "Embedding not returned by ChromaDB for this chunk. "
            "Verify the collection was created with include=['embeddings']."
        )
        return

    emb_array = np.array(embedding, dtype=np.float32)

    st.write(f"**Shape:** `{emb_array.shape}`  |  **dtype:** `{emb_array.dtype}`")

    # Preview: first 10 values
    st.markdown("**First 10 values:**")
    preview_df_data = {
        "Index": list(range(10)),
        "Value": [f"{v:.6f}" for v in emb_array[:10]],
    }
    st.dataframe(preview_df_data, use_container_width=True, hide_index=True)

    # Full vector toggle
    if st.checkbox("Show all 384 values", key=f"show_all_{chunk_id}"):
        import pandas as pd
        df = pd.DataFrame(
            {"Index": range(len(emb_array)), "Value": emb_array.tolist()}
        )
        st.dataframe(df, use_container_width=True, height=400, hide_index=True)

    # ── Heatmap visualisation ─────────────────────────────────────────────
    st.subheader("🌡️ Vector Heatmap")
    st.markdown(
        "Each cell represents one dimension of the embedding vector. "
        "Brighter = higher value. The model uses cosine similarity, so "
        "relative magnitudes matter more than absolute values."
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        reshaped = emb_array.reshape(16, 24)  # 16×24 = 384
        fig, ax = plt.subplots(figsize=(10, 3.5))
        im = ax.imshow(reshaped, aspect="auto", cmap="RdYlGn", interpolation="nearest")
        ax.set_title(f"Embedding Heatmap — Chunk {selected_idx}", fontsize=11)
        ax.set_xlabel("Dimension (column)")
        ax.set_ylabel("Dimension (row ×24)")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except ImportError:
        st.info("Install matplotlib for the heatmap visualisation.")

    # ── Norm ──────────────────────────────────────────────────────────────
    norm = float(np.linalg.norm(emb_array))
    st.write(f"**L2 norm:** `{norm:.6f}` *(should be ≈ 1.0 for normalised embeddings)*")
