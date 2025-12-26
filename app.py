"""
CineRAG - Production Streamlit App
Showcases hybrid search with performance metrics
"""

import streamlit as st
import json
from hybrid_rag import HybridMovieRAG
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="CineRAG - Movie Search System",
    page_icon="🎬",
    layout="wide"
)


# Initialize system (cached)
@st.cache_resource
def load_rag_system():
    return HybridMovieRAG()


@st.cache_data
def load_comparison_data():
    with open('data/comparison_report.json', 'r') as f:
        return json.load(f)


# Load systems
rag = load_rag_system()
comparison_data = load_comparison_data()

# Title and description
st.title("🎬 CineRAG - Hybrid Movie Search System")
st.markdown("""
Advanced RAG system combining **semantic search** and **keyword matching** (BM25) 
for accurate movie information retrieval across 466+ movies.
""")

# Sidebar - System Info
with st.sidebar:
    st.header("📊 System Stats")
    stats = rag.get_statistics()
    st.metric("Total Movies", stats['total_movies'])
    st.metric("Total Chunks", stats['total_chunks'])
    st.metric("Embedding Dim", stats['embedding_dimension'])

    st.divider()

    st.header("🎯 Performance")
    hybrid_perf = comparison_data['hybrid']['overall']

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Recall@3", f"{hybrid_perf['recall@3']:.1%}")
    with col2:
        st.metric("Recall@5", f"{hybrid_perf['recall@5']:.1%}")

    st.metric("MRR", f"{hybrid_perf['mrr']:.3f}")
    st.metric("Avg Latency", f"{hybrid_perf['avg_latency_ms']:.0f}ms")

# Main content tabs
tab1, tab2, tab3 = st.tabs(["🔍 Search", "📊 Performance", "ℹ️ About"])

# Tab 1: Search Interface
with tab1:
    st.header("Search Movies")

    # Search method selector
    search_method = st.radio(
        "Search Method:",
        ["Hybrid (Best)", "Semantic Only", "BM25 Only"],
        horizontal=True
    )

    method_map = {
        "Hybrid (Best)": "hybrid",
        "Semantic Only": "semantic",
        "BM25 Only": "bm25"
    }

    # Search input
    query = st.text_input(
        "Ask anything about movies:",
        placeholder="e.g., Who directed Inception? or Leonardo DiCaprio movies"
    )

    # Number of results
    top_k = st.slider("Number of results:", 1, 10, 5)

    if query:
        with st.spinner("Searching..."):
            results = rag.search(query, top_k=top_k, method=method_map[search_method])

        st.success(f"Found {len(results)} results in {results[0].get('relevance_score', 0):.3f}s")

        # Display results
        for i, result in enumerate(results, 1):
            with st.expander(f"**{i}. {result['movie_title']}** [{result['chunk_type']}]", expanded=(i <= 3)):
                st.write(result['text'])

                # Show scores for hybrid
                if method_map[search_method] == 'hybrid':
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Final Score", f"{result['relevance_score']:.3f}")
                    with col2:
                        st.metric("Semantic", f"{result['semantic_contribution']:.3f}")
                    with col3:
                        st.metric("BM25", f"{result['bm25_contribution']:.3f}")
                else:
                    st.metric("Relevance Score", f"{result['relevance_score']:.3f}")

                # Metadata
                if result['metadata']:
                    st.caption(
                        f"📅 {result['metadata'].get('year', 'N/A')} | ⭐ {result['metadata'].get('rating', 'N/A')}")

    # Example queries
    st.divider()
    st.subheader("💡 Try these examples:")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Factual:**")
        st.markdown("- Who directed Inception?")
        st.markdown("- Who stars in Titanic?")
        st.markdown("- What year was The Matrix released?")

    with col2:
        st.markdown("**Descriptive:**")
        st.markdown("- What is Inception about?")
        st.markdown("- Describe The Matrix")
        st.markdown("- What happens in Titanic?")

    with col3:
        st.markdown("**Cross-Reference:**")
        st.markdown("- Leonardo DiCaprio movies")
        st.markdown("- Christopher Nolan films")
        st.markdown("- Tom Hanks movies")

# Tab 2: Performance Metrics
with tab2:
    st.header("Performance Analysis")

    st.markdown("""
    Evaluated on **164 test questions** across multiple categories and difficulty levels.
    """)

    # Overall comparison
    st.subheader("📈 Overall Performance: Baseline vs Hybrid")

    baseline = comparison_data['baseline']['overall']
    hybrid = comparison_data['hybrid']['overall']

    # Create comparison chart
    metrics = ['recall@1', 'recall@3', 'recall@5', 'mrr']
    baseline_vals = [baseline[m] * 100 for m in metrics]
    hybrid_vals = [hybrid[m] * 100 for m in metrics]

    fig = go.Figure(data=[
        go.Bar(name='Baseline (Semantic Only)', x=metrics, y=baseline_vals, marker_color='#FF6B6B'),
        go.Bar(name='Hybrid (BM25 + Semantic)', x=metrics, y=hybrid_vals, marker_color='#4ECDC4')
    ])

    fig.update_layout(
        barmode='group',
        yaxis_title='Score (%)',
        xaxis_title='Metric',
        height=400,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Key improvements
    col1, col2, col3 = st.columns(3)

    improvements = comparison_data['improvements']

    with col1:
        recall_imp = improvements['overall_recall@3'] * 100
        st.metric(
            "Recall@3 Improvement",
            f"{hybrid['recall@3']:.1%}",
            f"{recall_imp:+.1f}%"
        )

    with col2:
        st.metric(
            "Recall@5",
            f"{hybrid['recall@5']:.1%}",
            f"{(hybrid['recall@5'] - baseline['recall@5']) * 100:+.1f}%"
        )

    with col3:
        st.metric(
            "Hit@3",
            f"{hybrid['hit@3']:.1%}",
            f"{(hybrid['hit@3'] - baseline['hit@3']) * 100:+.1f}%"
        )

    # By category
    st.subheader("📋 Performance by Question Category")

    baseline_cat = comparison_data['baseline']['by_category']
    hybrid_cat = comparison_data['hybrid']['by_category']

    categories = list(baseline_cat.keys())
    baseline_cat_vals = [baseline_cat[c]['recall@3'] * 100 for c in categories]
    hybrid_cat_vals = [hybrid_cat[c]['recall@3'] * 100 for c in categories]

    fig2 = go.Figure(data=[
        go.Bar(name='Baseline', x=categories, y=baseline_cat_vals, marker_color='#FF6B6B'),
        go.Bar(name='Hybrid', x=categories, y=hybrid_cat_vals, marker_color='#4ECDC4')
    ])

    fig2.update_layout(
        barmode='group',
        yaxis_title='Recall@3 (%)',
        xaxis_title='Category',
        height=400
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Highlight biggest improvement
    st.success("🎯 **Biggest Improvement:** Cross-reference queries (6.2% → 71.0%, +64.8%)")
    st.info("💡 **Why?** BM25 excels at exact keyword matching (actor/director names)")

# Tab 3: About
with tab3:
    st.header("About CineRAG")

    st.markdown("""
    ### 🎯 System Overview

    CineRAG is an advanced Retrieval-Augmented Generation (RAG) system designed for 
    movie information retrieval. It combines multiple search techniques to provide 
    accurate answers to diverse movie-related queries.

    ### 🏗️ Architecture

    **Data Pipeline:**
    1. **Data Collection:** TMDB API (500+ movies)
    2. **Chunking:** Smart content segmentation (plot, cast, crew, metadata)
    3. **Indexing:** 
       - Semantic: all-MiniLM-L6-v2 embeddings + FAISS
       - Keyword: BM25 with Okapi scoring

    **Search Strategy:**
    - Retrieves top 20 candidates from each method
    - Normalizes and combines scores (50/50 weight)
    - Returns top K most relevant chunks

    ### 📊 Dataset
    - **Movies:** 466 films
    - **Chunks:** 1,982 searchable segments
    - **Test Questions:** 164 with ground truth labels

    ### 🔧 Technologies
    - **Embeddings:** sentence-transformers
    - **Vector DB:** FAISS
    - **Keyword Search:** rank-bm25
    - **Web Framework:** Streamlit
    - **Evaluation:** Custom metrics system

    ### 📈 Key Results
    - Overall Recall@3: **85.7%** (↑4.3% from baseline)
    - Cross-reference queries: **71.0%** (↑64.8% from baseline)
    - Average latency: **21ms** per query

    ### 🚀 Future Improvements
    - [ ] Cross-encoder reranking
    - [ ] Query classification for better routing
    - [ ] Metadata filtering for genre/year queries
    - [ ] Conversation memory
    - [ ] Multi-modal search (posters, images)

    ### 👨‍💻 Built by
    **[Rima ALAYA]** - AI/ML Engineer

    [GitHub](https://github.com/RimaAlaya/cinerag) | [LinkedIn](https://linkedin.com/in/rima-alaya)
    """)

# Footer
st.divider()
st.caption("🎬 CineRAG v2.0 - Hybrid Search System | Data from TMDB")