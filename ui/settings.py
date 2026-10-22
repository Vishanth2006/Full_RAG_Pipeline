"""
Settings UI.
Persistent configuration for chunking, retrieval, Gemini, and workspace management.
"""

import streamlit as st

from config.settings import (
    CHROMA_PERSIST_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    GEMINI_MODEL,
    MAX_TOP_K,
    MAX_UPLOAD_SIZE_MB,
    SQLITE_DB_PATH,
    SUPPORTED_GEMINI_MODELS,
)
from services.gemini_service import validate_api_config


def render_settings():
    st.header("⚙️ Settings")
    st.markdown(
        "Configure chunking, retrieval, and Gemini parameters. "
        "Changes take effect on the next processing or query operation."
    )

    tab_chunk, tab_retrieval, tab_gemini, tab_workspace = st.tabs(
        ["🧩 Chunking", "🔍 Retrieval", "🤖 Gemini", "🗄️ Workspace"]
    )

    # ── Chunking ──────────────────────────────────────────────────────────
    with tab_chunk:
        st.subheader("Chunking Settings")
        st.info(
            "⚠️ Changing chunk size or overlap requires **re-processing all documents**. "
            "Mixing chunk settings within the same collection is not supported."
        )
        chunk_size = st.slider(
            "Chunk size (characters)",
            min_value=200, max_value=2000,
            value=st.session_state.get("chunk_size", DEFAULT_CHUNK_SIZE),
            step=50,
        )
        chunk_overlap = st.slider(
            "Chunk overlap (characters)",
            min_value=0, max_value=chunk_size // 2,
            value=min(
                st.session_state.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
                chunk_size // 2,
            ),
            step=10,
        )
        st.write(f"Separators: `{['\\n\\n', '\\n', '. ', ' ', '']}`")
        if st.button("💾 Save Chunking Settings"):
            st.session_state["chunk_size"] = chunk_size
            st.session_state["chunk_overlap"] = chunk_overlap
            st.success(
                f"Saved: chunk_size={chunk_size}, overlap={chunk_overlap}. "
                "Re-upload and process documents to apply."
            )

    # ── Retrieval ─────────────────────────────────────────────────────────
    with tab_retrieval:
        st.subheader("Retrieval Settings")
        top_k = st.slider(
            "Default top-k results",
            min_value=1, max_value=MAX_TOP_K,
            value=st.session_state.get("top_k", DEFAULT_TOP_K),
        )
        if st.button("💾 Save Retrieval Settings"):
            st.session_state["top_k"] = top_k
            st.success(f"Saved: top_k={top_k}")

        st.subheader("Embedding Model")
        st.info(
            "⚠️ The embedding model cannot be changed in an existing collection "
            "without full re-embedding. Current model is fixed to ensure consistency."
        )
        st.code(
            f"Model     : {EMBEDDING_MODEL_NAME}\n"
            f"Dimensions: 384\n"
            f"Distance  : cosine (HNSW)"
        )

    # ── Gemini ────────────────────────────────────────────────────────────
    with tab_gemini:
        st.subheader("Gemini Configuration")

        api_valid, api_msg = validate_api_config()
        if api_valid:
            st.success(f"✅ {api_msg}")
        else:
            st.error(f"❌ {api_msg}")
            st.markdown(
                "Set `GEMINI_API_KEY` in your `.env` file. "
                "Get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey)."
            )

        st.markdown("**Model selection** (set in `.env` or below for session):")
        selected_model = st.selectbox(
            "Gemini model",
            SUPPORTED_GEMINI_MODELS,
            index=SUPPORTED_GEMINI_MODELS.index(GEMINI_MODEL)
            if GEMINI_MODEL in SUPPORTED_GEMINI_MODELS
            else 0,
        )

        temperature = st.slider(
            "Temperature",
            0.0, 1.0,
            value=st.session_state.get("temperature", DEFAULT_TEMPERATURE),
            step=0.05,
        )
        max_tokens = st.number_input(
            "Max output tokens",
            min_value=256, max_value=8192,
            value=st.session_state.get("max_tokens", DEFAULT_MAX_OUTPUT_TOKENS),
            step=256,
        )

        system_prompt_val = st.text_area(
            "Default system prompt",
            value=st.session_state.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
            height=160,
        )

        if st.button("💾 Save Gemini Settings"):
            st.session_state["temperature"] = temperature
            st.session_state["max_tokens"] = max_tokens
            st.session_state["system_prompt"] = system_prompt_val
            st.session_state["gemini_model"] = selected_model
            st.success("Gemini settings saved for this session.")

    # ── Workspace ─────────────────────────────────────────────────────────
    with tab_workspace:
        st.subheader("Storage Paths")
        st.code(
            f"ChromaDB   : {CHROMA_PERSIST_DIR}\n"
            f"SQLite     : {SQLITE_DB_PATH}\n"
            f"Max upload : {MAX_UPLOAD_SIZE_MB} MB"
        )

        st.subheader("Upload Limit")
        st.write(
            f"Current limit: **{MAX_UPLOAD_SIZE_MB} MB**. "
            "Change `MAX_UPLOAD_SIZE_MB` in `.env` to adjust."
        )
