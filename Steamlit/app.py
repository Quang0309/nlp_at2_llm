import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama

from function.function import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_EMBED_MODEL,
    INDEX_PATH, DATASET_DIR,
    load_pdf_and_split, build_faiss_from_docs, load_index,
    make_retrieval_chain, list_index_sources,
    append_new_pdfs_to_index, compare_retrievers,
    is_jailbreak_attempt,
)

st.set_page_config(page_title="LangChain + Ollama RAG", page_icon=None, layout="wide")

# ===== Sidebar =====
st.sidebar.header("Ollama Config")
st.sidebar.write(f"Base URL: `{OLLAMA_BASE_URL}`")
st.sidebar.write(f"Chat Model: `{OLLAMA_MODEL}`")
st.sidebar.write(f"Embedding: `{OLLAMA_EMBED_MODEL}`")
st.sidebar.write(f"Index Path: `{INDEX_PATH}`")
st.sidebar.write(f"PDFs Path: `{DATASET_DIR}`")

# ===== Tabs =====
tab_chat, tab_index, tab_test = st.tabs(["Chat (RAG)", "Build / Inspect Index", "Compare Retrieval"])

# ----- Tab 1: RAG Chat -----
with tab_chat:
    st.subheader("Chat with your documents (RAG)")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    question = st.text_input("Your question:", placeholder="What will I learn in ANLP?")
    col1, col2 = st.columns([1, 1])
    run = col1.button("Ask", use_container_width=True)
    clear = col2.button("Clear history", use_container_width=True)

    if clear:
        st.session_state.chat_history = []
        st.session_state.pop("rag_answer", None)
        st.session_state.pop("rag_sources", None)
        st.session_state.pop("guardrail_status", None)

    if run and question.strip():
        # Guardrail check first
        llm_guard = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        print(f"Asking question: {question}")
        risky = is_jailbreak_attempt(llm_guard, question)
        status_text = "yes" if risky else "no"

        # show guardrail result in UI (before any answer)
        st.session_state["guardrail_status"] = status_text
        st.markdown(f"**Guardrail check: Is it an attack? : '{status_text}'**")

        if risky:
            blocked_msg = "This input appears unsafe or violates academic standards. Please rephrase."
            st.warning(blocked_msg)
            st.session_state.chat_history.append(HumanMessage(content=question))
            st.session_state.chat_history.append(AIMessage(content="Blocked by guardrail."))
        else:
            # Load index
            try:
                vs = load_index()
            except Exception as e:
                st.error(f"Failed to load index at `{INDEX_PATH}`. Build it first. Details: {e}")
                vs = None

            # Run RAG
            if vs:
                rag_chain = make_retrieval_chain(vs)
                with st.spinner("Retrieving and generating..."):
                    res = rag_chain.invoke({"input": question, "chat_history": st.session_state.chat_history})

                ans = res.get("answer", "")
                ctx_docs = res.get("context", [])

                st.session_state.chat_history.extend([HumanMessage(content=question), AIMessage(content=ans)])
                st.session_state["rag_answer"] = ans
                st.session_state["rag_sources"] = ctx_docs

    # Show last guardrail status line (if available)
    if "guardrail_status" in st.session_state:
        st.caption(f"Guardrail check: Is it an attack? : '{st.session_state['guardrail_status']}'")

    if "rag_answer" in st.session_state:
        st.markdown("### Answer")
        st.write(st.session_state["rag_answer"])

    if "rag_sources" in st.session_state and st.session_state["rag_sources"]:
        st.markdown("---")
        st.markdown("Sources")
        shown = set()
        for d in st.session_state["rag_sources"]:
            src = os.path.basename(d.metadata.get("source", ""))
            page = (d.metadata.get("page", -1) or -1) + 1
            key = (src, page)
            if key in shown:
                continue
            shown.add(key)
            st.caption(f"- {src} — page {page}")

# ----- Tab 2: Build / Inspect Index -----
with tab_index:
    st.subheader("Build / Overwrite FAISS Index")

    use_existing = st.checkbox("Use PDFs already in dataset/pdfs", value=True)
    uploads = None
    if not use_existing:
        uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    col_b1, col_b2, col_b3 = st.columns([1, 1, 1])

    # Build index (from existing or from uploaded)
    if col_b1.button("Build index", use_container_width=True):
        try:
            if use_existing:
                docs = load_pdf_and_split(DATASET_DIR)
                if not docs:
                    st.error(f"No PDFs found in {DATASET_DIR}. Untick the checkbox and upload files.")
                    st.stop()
            else:
                if not uploads:
                    st.error("Please upload at least one PDF, or tick 'Use PDFs already in dataset/pdfs'.")
                    st.stop()
                os.makedirs(DATASET_DIR, exist_ok=True)
                for f in uploads:
                    with open(os.path.join(DATASET_DIR, f.name), "wb") as fw:
                        fw.write(f.read())
                docs = load_pdf_and_split(DATASET_DIR)

            with st.spinner("Splitting, embedding, building FAISS..."):
                _, n_chunks = build_faiss_from_docs(docs, index_path=INDEX_PATH)
            st.success(f"Built index at `{INDEX_PATH}` with {n_chunks} chunks.")
        except Exception as e:
            st.error(f"Build failed: {e}")

    # Append new PDFs (do not rebuild)
    if col_b2.button("Append new PDFs to existing index", use_container_width=True):
        try:
            with st.spinner("Embedding and appending new PDFs..."):
                n_added = append_new_pdfs_to_index()
            if n_added == 0:
                st.info("No new PDFs detected. All sources already present in the index.")
            else:
                st.success(f"Appended {n_added} new chunks.")
        except Exception as e:
            st.error(f"Append failed: {e}")

    # Inspect index
    if col_b3.button("Inspect current index", use_container_width=True):
        try:
            vs = load_index()
            sources = sorted(s for s in list_index_sources(vs) if s)
            st.info(f"Chunks: {len(vs.docstore._dict)}")
            if sources:
                st.write("Files in index:")
                for s in sources:
                    st.caption(f"- {s}")
            else:
                st.write("No files recorded in index metadata.")
        except Exception as e:
            st.error(f"Failed to load index: {e}")

# ----- Tab 3: Compare Retrieval -----
with tab_test:
    st.subheader("Compare Basic vs MultiQuery Retriever")

    q = st.text_input("Your query to compare:", value="What is NLP?")
    c1, c2, c3 = st.columns([1, 1, 1])
    k = c1.slider("Top-k per retriever", 1, 10, 4, 1)
    show_content = c2.checkbox("Show snippet content", value=False)
    run_cmp = c3.button("Run Comparison", use_container_width=True)

    if run_cmp and q.strip():
        try:
            vs = load_index()
        except Exception as e:
            st.error(f"Failed to load index at `{INDEX_PATH}`. Build it first. Details: {e}")
            vs = None

        if vs:
            with st.spinner("Comparing retrievers..."):
                res = compare_retrievers(vs, q, k=k, show_content=show_content)

                basic_docs = res["basic_docs"]
                mq_docs = res["multiquery_docs"]
                overlap = res["overlap"]
                basic_sources = res["basic_sources"]
                mq_sources = res["mq_sources"]

            # Summary metrics
            st.markdown("### Summary")
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Basic: #docs", len(basic_docs))
            mcol2.metric("MultiQuery: #docs", len(mq_docs))
            mcol3.metric("Overlap (doc refs)", len(overlap))
            mcol4.metric("New in MultiQuery", len(set(mq_sources) - set(basic_sources)))

            # Side-by-side lists
            left, right = st.columns(2)
            with left:
                st.markdown("#### Basic Retriever")
                for i, s in enumerate(basic_sources, 1):
                    st.write(f"{i:02d}. {s}")
                    if show_content:
                        st.caption(basic_docs[i-1].page_content[:300] + "…")

            with right:
                st.markdown("#### MultiQuery Retriever")
                for i, s in enumerate(mq_sources, 1):
                    st.write(f"{i:02d}. {s}")
                    if show_content:
                        st.caption(mq_docs[i-1].page_content[:300] + "…")

            # Overlap & differences
            st.markdown("---")
            st.markdown("#### Overlap & Differences")
            o1, o2, o3 = st.columns(3)
            with o1:
                st.write("Overlap (both):")
                if overlap:
                    for s in sorted(overlap):
                        st.caption(f"- {s}")
                else:
                    st.caption("None")
            with o2:
                st.write("Only in MultiQuery:")
                only_mq = sorted(set(mq_sources) - set(basic_sources))
                if only_mq:
                    for s in only_mq:
                        st.caption(f"- {s}")
                else:
                    st.caption("None")
            with o3:
                st.write("Only in Basic:")
                only_basic = sorted(set(basic_sources) - set(mq_sources))
                if only_basic:
                    for s in only_basic:
                        st.caption(f"- {s}")
                else:
                    st.caption("None")
