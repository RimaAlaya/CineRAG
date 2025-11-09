from main import rag_answer, query_system   # ensure rag_answer is available
import streamlit as st

st.title("Movie/Pop Culture RAG Demo")

st.title("Movie/Pop Culture RAG + LLaMA Demo")

question = st.text_input("Ask a question about movies or series:")

if question:
    answer = rag_answer(question, top_k=2)
    st.write("**Answer:**")
    st.write(answer)
