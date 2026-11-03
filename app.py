"""Launch with: streamlit run app.py"""
import streamlit as st

from rag import ingest, get_collection, answer_question
from agent import agent_answer

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

question = st.text_input('Ask a question about your documents',
    placeholder='What drove Aurelia Analytics revenue growth in 2026?')
if st.button('Research', type='primary'):
    if not question.strip():
        st.warning('Type a question first.')
    else:
        try:
            with st.spinner('Retrieving evidence and generating an answer...'):
                result = (answer_question(question) if mode.startswith('RAG')
                          else agent_answer(question))
            st.subheader('Answer')
            st.write(result['answer'])
            st.caption(f"Mode: {result['mode']} | Latency: {result['latency_seconds']}s | "
                       f"Input tokens: {result['input_tokens']} | Output tokens: {result['output_tokens']}")
            st.subheader('Retrieved passages')
            for i, source in enumerate(result['sources'], 1):
                with st.expander(f"[{i}] {source['source']} – page {source['page']}"):
                    st.write(source['text'])
        except Exception as exc:
            st.error(f'Problem: {exc}')

st.caption('Demo only: not suitable for confidential data or public production use without stronger controls.')
