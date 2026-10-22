"""
Answer & Query History UI.
Browse, inspect, and compare past queries with full context reconstruction.
"""

import streamlit as st
from datetime import datetime

from config.settings import HISTORY_PER_PAGE
from services.history_service import reconstruct_retrieved_chunks
from services.prompt_service import build_prompt_components
from storage.document_registry import list_documents
from storage.history_repository import (
    clear_history,
    count_queries,
    delete_query,
    list_queries,
)


def render_answer_history():
    st.header("📜 Answer & Query History")
    st.markdown(
        "Every query is stored with its full context snapshot. "
        "Historical queries remain inspectable even after documents are uploaded or deleted."
    )

    total = count_queries()
    st.metric("Total Queries", total)

    if total == 0:
        st.info("No queries yet. Use Query & Retrieval and generate an answer first.")
        return

    # ── Load history ──────────────────────────────────────────────────────
    queries = list_queries(limit=200)

    # ── Compare mode ──────────────────────────────────────────────────────
    tab_browse, tab_compare, tab_manage = st.tabs(
        ["📋 Browse History", "⚖️ Compare Queries", "⚙️ Manage"]
    )

    with tab_browse:
        _render_browse(queries)

    with tab_compare:
        _render_compare(queries)

    with tab_manage:
        _render_manage()


def _render_browse(queries):
    # ── Pagination ────────────────────────────────────────────────────────
    total = len(queries)
    total_pages = max(1, (total + HISTORY_PER_PAGE - 1) // HISTORY_PER_PAGE)
    page_key = "history_page"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1

    col_prev, col_info, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("◀", key="h_prev") and st.session_state[page_key] > 1:
            st.session_state[page_key] -= 1
    with col_info:
        st.caption(f"Page {st.session_state[page_key]} of {total_pages} ({total} queries)")
    with col_next:
        if st.button("▶", key="h_next") and st.session_state[page_key] < total_pages:
            st.session_state[page_key] += 1

    start = (st.session_state[page_key] - 1) * HISTORY_PER_PAGE
    page_queries = queries[start : start + HISTORY_PER_PAGE]

    for q in page_queries:
        ts = q.timestamp[:19].replace("T", " ")
        label = (
            f"🗨️ {q.user_question[:70]}{'…' if len(q.user_question) > 70 else ''}  "
            f"| {ts} | {q.gemini_model}"
        )
        with st.expander(label, expanded=False):
            tab_ans, tab_ctx, tab_prompt = st.tabs(["Answer", "Retrieved Context", "Prompt"])

            with tab_ans:
                st.markdown(f"**Question:** {q.user_question}")
                st.markdown("**Answer:**")
                st.markdown(q.generated_answer)

                if q.total_tokens:
                    tc1, tc2, tc3 = st.columns(3)
                    tc1.metric("Prompt Tokens", q.prompt_tokens)
                    tc2.metric("Completion Tokens", q.completion_tokens)
                    tc3.metric("Total Tokens", q.total_tokens)

                st.caption(
                    f"Model: `{q.gemini_model}` | T={q.temperature} | "
                    f"MaxTok={q.max_output_tokens} | TopK={q.top_k}"
                )

            with tab_ctx:
                chunks = reconstruct_retrieved_chunks(q)
                if not chunks:
                    st.info("No context chunks stored for this query.")
                else:
                    for chunk in chunks:
                        pages_str = ", ".join(str(p) for p in chunk.source_page_numbers)
                        # Check if source still exists
                        active_ids = [d.document_id for d in list_documents()]
                        is_deleted = chunk.document_id not in active_ids
                        status_note = " ⚠️ *[source deleted]*" if is_deleted else ""

                        with st.expander(
                            f"Rank #{chunk.rank} | {chunk.document_name}{status_note} | "
                            f"Page(s): {pages_str} | Similarity: {chunk.cosine_similarity:.4f}"
                        ):
                            st.text_area(
                                "",
                                value=chunk.chunk_text,
                                height=130,
                                disabled=True,
                                key=f"hist_ctx_{q.query_id}_{chunk.rank}",
                            )

            with tab_prompt:
                chunks = reconstruct_retrieved_chunks(q)
                components = build_prompt_components(
                    q.user_question,
                    chunks,
                    system_prompt=q.system_prompt,
                )
                st.subheader("System Instruction")
                st.code(components["system_instruction"])
                st.subheader("User Message (context + question)")
                st.code(components["user_message"])


def _render_compare(queries):
    if len(queries) < 2:
        st.info("Need at least 2 queries to compare.")
        return

    all_labels = [
        f"{q.timestamp[:16]} | {q.user_question[:50]}" for q in queries
    ]

    c1, c2 = st.columns(2)
    with c1:
        idx1 = st.selectbox("Query A", range(len(all_labels)), format_func=lambda i: all_labels[i], key="cmp_a")
    with c2:
        idx2 = st.selectbox("Query B", range(len(all_labels)), format_func=lambda i: all_labels[i], key="cmp_b", index=min(1, len(queries)-1))

    q_a, q_b = queries[idx1], queries[idx2]

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"**Query A:** {q_a.user_question}")
        st.caption(f"Model: {q_a.gemini_model} | T={q_a.temperature}")
        st.markdown("**Answer:**")
        st.markdown(q_a.generated_answer[:500] + ("…" if len(q_a.generated_answer) > 500 else ""))
        st.caption(f"Retrieved {len(q_a.retrieved_chunk_ids)} chunks")

    with col_b:
        st.markdown(f"**Query B:** {q_b.user_question}")
        st.caption(f"Model: {q_b.gemini_model} | T={q_b.temperature}")
        st.markdown("**Answer:**")
        st.markdown(q_b.generated_answer[:500] + ("…" if len(q_b.generated_answer) > 500 else ""))
        st.caption(f"Retrieved {len(q_b.retrieved_chunk_ids)} chunks")


def _render_manage():
    st.markdown("### Danger Zone")
    if st.button("🗑️ Clear All Query History", type="secondary"):
        st.session_state["confirm_clear_hist"] = True

    if st.session_state.get("confirm_clear_hist"):
        st.error("This will delete **all** query history permanently.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Confirm Delete History"):
                clear_history()
                st.session_state.pop("confirm_clear_hist", None)
                st.success("History cleared.")
                st.rerun()
        with c2:
            if st.button("❌ Cancel"):
                st.session_state.pop("confirm_clear_hist", None)
                st.rerun()
