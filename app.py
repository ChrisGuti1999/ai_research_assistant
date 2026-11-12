"""Launch with: streamlit run app.py"""
import streamlit as st

from rag import ingest, get_collection, answer_question
from agent import agent_answer

import re
from pathlib import Path



def display_source(filename: str) -> str:
    """Display friendly document names without changing source IDs."""

    if filename == "All documents":
        return filename

    name = Path(filename).stem

    # Remove numerical document prefix, e.g. 13.3.2.1_
    name = re.sub(r"^\d+(?:\.\d+)*_", "", name)
    name = name.replace("_", " ")

    # Shorten audited financial statement names
    match = re.fullmatch(
        r"(.+?) signed audited FS FY(\d{4})",
        name,
        flags=re.IGNORECASE
    )

    if match:
        return f"{match.group(1)} — FY{match.group(2)}"

    return name.title()


st.set_page_config(page_title='AI Research Assistant', page_icon='📚')
st.title('📚 AI Research Assistant')
st.caption('A learning project: Python + RAG + LLM API + optional tool-calling agent')

with st.sidebar:
    st.header('Document index')
    st.write('Put public .pdf or .txt reports in the `documents/` folder.')
    if st.button('Index documents', use_container_width=True):
        try:
            with st.spinner('Extracting text and creating embeddings...'):
                info = ingest()
            st.success(f"Indexed {info['documents']} documents, {info['chunks']} chunks.")
        except Exception as exc:
            st.error(str(exc))
    try:
        st.metric('Indexed chunks', get_collection().count())
    except Exception as exc:
        st.warning(str(exc))
    mode = st.radio('Mode', ['RAG (start here)', 'Agent (Phase 2)'])


    # Select how many chunks to retrieve
    top_k = st.select_slider(
        "Retrieval depth (top_k)",
        options=[3, 5],
        value=5,
        help="Number of evidence chunks supplied to the LLM."
    )

    # Retrieve filenames from indexed ChromaDB metadata
    try:
        collection = get_collection()

        metadata = collection.get(
            include=["metadatas"]
        )["metadatas"] or []

        available_sources = sorted({
            item["source"]
            for item in metadata
            if item and "source" in item
        })

    except Exception as exc:
        available_sources = []
        st.warning(f"Unable to load source documents: {exc}")

    # Let user search one document or the entire index

    selected_source = st.selectbox(
        "Source document",
        options=["All documents"] + available_sources,
        format_func=display_source,
        help="Select the document to search."
    )


    source_filter = (
        None
        if selected_source == "All documents"
        else selected_source
    )


question = st.text_input('Ask a question about your documents',
    placeholder='What drove Aurelia Analytics revenue growth in 2026?')
if st.button('Research', type='primary'):
    if not question.strip():
        st.warning('Type a question first.')
    else:
        try:
            with st.spinner('Retrieving evidence and generating an answer...'):

                if mode.startswith("RAG"):
                    result = answer_question(
                        question,
                        top_k=top_k,
                        source_filter=source_filter
                    )
                else:
                    result = agent_answer(
                        question,
                        top_k=top_k,
                        source_filter=source_filter
                    )

            st.subheader('Answer')
            st.write(result['answer'])
            st.caption(f"Mode: {result['mode']} | Latency: {result['latency_seconds']}s | "
                       f"Input tokens: {result['input_tokens']} | Output tokens: {result['output_tokens']}")
        
            st.caption(f"Retrieval depth: {top_k} | "
                        f"Source: {display_source(selected_source)}"
)
            st.subheader('Retrieved passages')
            for i, source in enumerate(result['sources'], 1):
                

                with st.expander(
                    f"[{i}] {display_source(source['source'])} "
                    f"— Page {source['page']}"
                ):
                    st.caption(f"Original source: {source['source']}")
                    st.write(source["text"])

        except Exception as exc:
            st.error(f'Problem: {exc}')

st.caption('Demo only: not suitable for confidential data or public production use without stronger controls.')
