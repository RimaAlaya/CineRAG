# app.py
"""
CineRAG - Cinema Secrets Edition
Production Streamlit App with behind-the-scenes content
"""

import streamlit as st
import json
from cinema_secrets_rag import CinemaSecretsRAG
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="CineRAG - Cinema Secrets",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .secret-badge {
        background-color: #ff6b6b;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
    }
    .movie-title {
        font-size: 18px;
        font-weight: bold;
        color: #4ECDC4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize system (cached)
@st.cache_resource
def load_rag_system():
    return CinemaSecretsRAG()

rag = load_rag_system()

# Title
st.title("🎬 CineRAG - Cinema Secrets Encyclopedia")
st.markdown("""
**Beyond IMDB**: Get director info, plot summaries, AND behind-the-scenes secrets, 
production stories, and insider trivia that true cinema fans obsess over.
""")

# Sidebar - Stats
with st.sidebar:
    st.header("📊 System Stats")
    stats = rag.get_statistics()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Movies", stats['total_movies'])
        st.metric("Total Chunks", stats['total_chunks'])
    with col2:
        st.metric("🎬 Secrets", stats['secret_chunks'])
        st.metric("LLM", "✅ Groq" if stats['has_llm'] else "❌")

    st.divider()

    st.header("🎯 Features")
    st.markdown("""
    - **Hybrid Search** (BM25 + Semantic)
    - **Behind-the-Scenes Secrets**
    - **AI-Powered Answers** (Groq)
    - **252 Movies with Secrets**
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["🔍 Ask Questions", "💡 Examples", "ℹ️ About"])

# Tab 1: Main search
with tab1:
    st.header("Ask Me Anything About Movies")

    # Search mode
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Your question:",
            placeholder="e.g., What secrets are there about The Dark Knight?"
        )
    with col2:
        use_llm = st.checkbox("Use AI Answer", value=True)

    if query:
        with st.spinner("Searching cinema knowledge..."):
            result = rag.ask(query, top_k=5, use_llm=use_llm)

        # Show answer
        st.markdown("### 💬 Answer")
        st.info(result['answer'])

        # Show if secrets were found
        if result['has_secrets']:
            st.success("🎬 This answer includes behind-the-scenes secrets!")

        st.divider()

        # Show sources
        st.markdown("### 📚 Sources")

        for i, source in enumerate(result['sources'], 1):
            with st.expander(
                f"**{i}. {source['movie_title']}** [{source['chunk_type']}] "
                + ("🎬 SECRET" if source['is_secret'] else ""),
                expanded=(i <= 2)
            ):
                st.write(source['text'])
                st.caption(f"Relevance: {source['relevance_score']:.3f}")

# Tab 2: Examples
with tab2:
    st.header("💡 Try These Queries")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Basic Questions")
        basic_queries = [
            "Who directed Inception?",
            "Who stars in Titanic?",
            "What is The Matrix about?",
            "Leonardo DiCaprio movies"
        ]

        for q in basic_queries:
            if st.button(q, key=f"basic_{q}"):
                st.session_state.example_query = q

        st.markdown("### 🎬 Cinema Secrets")
        secret_queries = [
            "What secrets are there about Inception?",
            "Behind the scenes of The Dark Knight",
            "Tell me trivia about Titanic",
            "Production stories from The Matrix"
        ]

        for q in secret_queries:
            if st.button(q, key=f"secret_{q}"):
                st.session_state.example_query = q

    with col2:
        if 'example_query' in st.session_state:
            st.markdown(f"### Testing: *{st.session_state.example_query}*")

            with st.spinner("Searching..."):
                result = rag.ask(st.session_state.example_query, top_k=3, use_llm=True)

            st.markdown("**Answer:**")
            st.write(result['answer'])

            st.markdown("**Top Source:**")
            if result['sources']:
                source = result['sources'][0]
                st.write(f"**{source['movie_title']}** [{source['chunk_type']}]")
                st.caption(source['text'][:200] + "...")

# Tab 3: About
with tab3:
    st.header("About CineRAG")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🎯 What is This?
        
        **CineRAG** is a RAG system for true cinema lovers. While other movie 
        databases give you basic info (director, cast, plot), CineRAG reveals 
        the secrets that fans obsess over:
        
        - 🎬 Behind-the-scenes production stories
        - 🎭 Casting decisions and actor preparation
        - 🎥 Filming techniques and challenges
        - 📺 Reception, controversies, and legacy
        
        ### 🏗️ How It Works
        
        1. **Hybrid Search**: Combines semantic similarity (understanding meaning) 
           with keyword matching (exact phrases)
        2. **Multi-Source Data**: TMDB for basic info + Wikipedia for secrets
        3. **AI Generation**: Groq's Mixtral model formats answers naturally
        4. **Smart Chunking**: Separates plot, cast, crew, and secrets
        """)

    with col2:
        st.markdown("""
        ### 📊 Technical Details
        
        **Data:**
        - 466 movies from TMDB
        - 252 movies with behind-the-scenes secrets
        - 3,000+ searchable chunks
        
        **Tech Stack:**
        - **Vector Search**: FAISS + sentence-transformers
        - **Keyword Search**: BM25 (Okapi)
        - **LLM**: Groq (Mixtral-8x7b)
        - **Framework**: LangChain
        - **UI**: Streamlit
        
        **Performance:**
        - Average query time: <1 second
        - Hybrid search accuracy: 85.7% Recall@3
        - 64.8% improvement on cross-reference queries
        
        ### 🚀 Built By
        
        **Rima Alaya** - AI/ML Engineer
        
        [GitHub](https://github.com/RimaAlaya) | [LinkedIn](https://linkedin.com/in/rima-alaya)
        
        *Made with passion for cinema and technology* 🎬
        """)

# Footer
st.divider()
st.caption("🎬 CineRAG v2.0 - Cinema Secrets Edition | Data: TMDB + Wikipedia")