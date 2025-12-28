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
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Ask Questions", "💡 Examples", "📊 Metrics", "ℹ️ About"])
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

# Add after showing the answer
st.divider()

# Feedback section
st.markdown("### 📊 Was this answer helpful?")

col1, col2, col3 = st.columns([1, 1, 3])

with col1:
    if st.button("👍 Yes, helpful", key=f"good_{result['query_id']}"):
        rag.feedback.save_feedback(result['query_id'], rating=1)
        st.success("Thanks! This helps us improve.")
        st.rerun()

with col2:
    if st.button("👎 No, not helpful", key=f"bad_{result['query_id']}"):
        st.session_state[f'show_feedback_{result["query_id"]}'] = True
        st.rerun()

# Show feedback form if user clicked thumbs down
if st.session_state.get(f'show_feedback_{result["query_id"]}', False):
    with st.form(key=f'feedback_form_{result["query_id"]}'):
        feedback_text = st.text_area(
            "What was wrong with this answer?",
            placeholder="The answer was off-topic / Missing information / Wrong movie / Other..."
        )

        if st.form_submit_button("Submit Feedback"):
            rag.feedback.save_feedback(result['query_id'], rating=0, feedback_text=feedback_text)
            st.success("Thank you for your feedback! We'll use this to improve.")
            del st.session_state[f'show_feedback_{result["query_id"]}']
            st.rerun()

# Show performance metrics
st.caption(f"⚡ Query processed in {result['latency_ms']:.0f}ms")

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

with tab4:
    st.header("📊 System Metrics & Performance")

    from metrics_tracker import MetricsTracker

    tracker = MetricsTracker()

    # Time period selector
    days = st.selectbox("Time Period", [1, 7, 30], index=1)

    report = tracker.generate_report(days=days)

    # Overview metrics
    st.subheader("📈 Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Queries",
            report['overview']['total_queries']
        )
    with col2:
        st.metric(
            "Satisfaction Rate",
            f"{report['performance']['satisfaction_rate']:.1%}"
        )
    with col3:
        st.metric(
            "Avg Latency",
            f"{report['performance']['latency']['avg_ms']:.0f}ms"
        )
    with col4:
        st.metric(
            "Estimated Cost",
            f"${report['costs']['estimated_cost_usd']:.4f}"
        )

    st.divider()

    # Performance details
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⚡ Latency Distribution")
        latency_data = report['performance']['latency']
        st.write(f"**P95:** {latency_data['p95_ms']:.0f}ms")
        st.write(f"**P99:** {latency_data['p99_ms']:.0f}ms")
        st.write(f"**Max:** {latency_data['max_ms']:.0f}ms")

        # Simple latency chart (placeholder)
        st.bar_chart({
            'Average': [latency_data['avg_ms']],
            'P95': [latency_data['p95_ms']],
            'P99': [latency_data['p99_ms']]
        })

    with col2:
        st.subheader("🎬 Query Types")
        secrets = report['usage']['secrets_distribution']
        st.write(f"**With Secrets:** {secrets['secrets_pct']:.1f}%")
        st.write(f"**Basic Info:** {100 - secrets['secrets_pct']:.1f}%")

        # Pie chart data
        import plotly.graph_objects as go

        fig = go.Figure(data=[go.Pie(
            labels=['With Secrets', 'Basic Info'],
            values=[secrets['with_secrets'], secrets['without_secrets']],
            hole=.3
        )])
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Quality issues
    st.subheader("⚠️ Quality Analysis")
    failures = report['quality']

    if failures['total_failures'] > 0:
        st.warning(f"**{failures['total_failures']} queries** received negative feedback")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.write("**Failure Categories:**")
            for category, count in failures['failure_categories'].items():
                if count > 0:
                    st.write(f"- {category.replace('_', ' ').title()}: {count}")

        with col2:
            st.write("**Action Items:**")
            st.write("1. Review failed queries")
            st.write("2. Improve prompts")
            st.write("3. Add missing data")
    else:
        st.success("No negative feedback in this period!")

# Footer
st.divider()
st.caption("🎬 CineRAG v2.0 - Cinema Secrets Edition | Data: TMDB + Wikipedia")