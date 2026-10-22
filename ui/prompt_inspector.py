"""
Prompt Inspector UI.
Shows the exact system instruction, context block, and user message
that will be sent to Gemini — including an editable system prompt.
"""

import streamlit as st

from config.settings import DEFAULT_SYSTEM_PROMPT, GEMINI_MODEL, DEFAULT_TEMPERATURE, DEFAULT_MAX_OUTPUT_TOKENS
from services.gemini_service import generate_answer, validate_api_config
from services.history_service import record_query
from services.prompt_service import assemble_user_message, build_prompt_components, build_source_citations
from storage.document_registry import list_documents


def render_prompt_inspector():
    st.header("🔬 Prompt Inspector & Answer Generation")
    st.markdown(
        "Inspect the **exact components** sent to Gemini. "
        "The system instruction is passed via the SDK's `system_instruction` field; "
        "the context + question forms the `contents` field. "
        "**No API keys are shown here.**"
    )

    retrieval = st.session_state.get("last_retrieval")
    if not retrieval or not retrieval.get("results"):
        st.info(
            "No retrieval results available. "
            "Go to **Query & Retrieval**, ask a question, then return here."
        )
        return

    results = retrieval["results"]
    query_text = retrieval["query"]

    # ── System prompt editor ──────────────────────────────────────────────
    st.subheader("1️⃣ System Instruction")
    system_prompt = st.text_area(
        "Edit the system instruction (changes apply to the next Gemini call):",
        value=st.session_state.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        height=140,
        key="system_prompt_editor",
    )
    st.session_state["system_prompt"] = system_prompt

    st.divider()

    # ── Build prompt ──────────────────────────────────────────────────────
    components = build_prompt_components(query_text, results, system_prompt=system_prompt)

    # ── Section 2: Retrieved context ──────────────────────────────────────
    st.subheader("2️⃣ Retrieved Context Chunks")
    with st.expander("View context block (sent as part of user message)", expanded=True):
        st.code(components["context_block"], language=None)

    # ── Section 3: User question ──────────────────────────────────────────
    st.subheader("3️⃣ User Question")
    st.info(f"**{query_text}**")

    # ── Section 4: Full assembled prompt ─────────────────────────────────
    st.subheader("4️⃣ Assembled Readable Prompt")
    with st.expander("View full prompt (for inspection only)", expanded=False):
        st.code(components["assembled_readable"], language=None)

    col_copy, col_note = st.columns([1, 3])
    with col_copy:
        st.download_button(
            "📋 Download Prompt",
            data=components["assembled_readable"],
            file_name="prompt_inspection.txt",
            mime="text/plain",
        )
    with col_note:
        st.caption(
            "Note: The SDK sends system_instruction and contents as **separate fields**, "
            "not as one concatenated string."
        )

    st.divider()

    # ── Section 5: Generation config ──────────────────────────────────────
    st.subheader("5️⃣ Generation Configuration")
    c1, c2, c3 = st.columns(3)
    with c1:
        model_name = st.selectbox(
            "Gemini model",
            [GEMINI_MODEL, "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
            index=0,
            key="gen_model_select",
        )
    with c2:
        temperature = st.slider(
            "Temperature",
            0.0, 1.0,
            value=st.session_state.get("temperature", DEFAULT_TEMPERATURE),
            step=0.05,
            key="temperature_slider",
        )
    with c3:
        max_tokens = st.number_input(
            "Max output tokens",
            min_value=256,
            max_value=8192,
            value=st.session_state.get("max_tokens", DEFAULT_MAX_OUTPUT_TOKENS),
            step=256,
            key="max_tokens_input",
        )

    st.code(
        f"model            : {model_name}\n"
        f"temperature      : {temperature}\n"
        f"max_output_tokens: {max_tokens}\n"
        f"system_instruction: [see above]\n"
        f"contents         : [context + question — see above]",
        language=None,
    )

    st.divider()

    # ── API config status ─────────────────────────────────────────────────
    api_valid, api_msg = validate_api_config()
    if not api_valid:
        st.error(f"❌ Gemini not configured: {api_msg}")
        return

    # ── Generate answer ────────────────────────────────────────────────────
    st.subheader("6️⃣ Generate Answer")
    if st.button("🤖 Generate Answer with Gemini", type="primary"):
        user_message = assemble_user_message(query_text, results)

        with st.spinner("Calling Gemini API…"):
            response = generate_answer(
                system_prompt=system_prompt,
                user_message=user_message,
                model_name=model_name,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

        if response.get("error"):
            st.error(f"❌ Gemini error: {response['error']}")
            return

        # Store results in session
        answer = response["answer"]
        citations = build_source_citations(results)
        st.session_state["last_answer"] = {
            "answer": answer,
            "citations": citations,
            "query": query_text,
            "model": model_name,
            "token_usage": {
                "prompt": response["prompt_tokens"],
                "completion": response["completion_tokens"],
                "total": response["total_tokens"],
            },
        }

        # Save to history
        docs = [d for d in list_documents() if d.status == "indexed"]
        record_query(
            user_question=query_text,
            system_prompt=system_prompt,
            retrieved_chunks=results,
            gemini_model=model_name,
            generated_answer=answer,
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_k=retrieval.get("top_k", 4),
            document_filter=retrieval.get("doc_filter"),
            workspace_doc_ids=[d.document_id for d in docs],
            prompt_tokens=response["prompt_tokens"],
            completion_tokens=response["completion_tokens"],
            total_tokens=response["total_tokens"],
        )

    # ── Display answer ────────────────────────────────────────────────────
    last_answer = st.session_state.get("last_answer")
    if last_answer and last_answer.get("query") == query_text:
        _render_answer(last_answer)


def _render_answer(answer_data: dict):
    st.divider()
    st.subheader("💬 Generated Answer")

    st.markdown(answer_data["answer"])

    usage = answer_data.get("token_usage", {})
    if any(usage.values()):
        c1, c2, c3 = st.columns(3)
        c1.metric("Prompt Tokens", usage.get("prompt", 0))
        c2.metric("Completion Tokens", usage.get("completion", 0))
        c3.metric("Total Tokens", usage.get("total", 0))

    st.divider()
    st.subheader("📚 Source Citations")
    for cite in answer_data.get("citations", []):
        with st.expander(
            f"📄 [{cite['rank']}] {cite['document_name']} — Page(s) {cite['pages']}  "
            f"(similarity: {cite['similarity']:.4f})",
            expanded=False,
        ):
            st.text_area(
                "Supporting chunk text",
                value=cite["chunk_text"],
                height=120,
                disabled=True,
                key=f"cite_text_{cite['chunk_id']}",
            )
            st.caption(
                "⚠️ Similarity scores indicate semantic closeness, "
                "not factual correctness. Always verify information against original sources."
            )
