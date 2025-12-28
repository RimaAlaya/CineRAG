# app.py
"""
CineRAG - GOSSIP MODE
Spilling Cinema's Hottest Secrets ☕
"""

import streamlit as st
import json
from cinema_secrets_rag import CinemaSecretsRAG
import plotly.graph_objects as go
import plotly.express as px
import time

# Page config
st.set_page_config(
    page_title="CineRAG - Cinema Gossip",
    page_icon="🍿",
    layout="wide"
)

# GOSSIP THEME - Make it POP
st.markdown("""
<style>
    /* 1. HIDE STREAMLIT BRANDING & DEPLOY BAR */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {display: none !important;}
        /* 3. REMOVE EXTRA WHITESPACE (The "Blank Bars") */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 95% !important;
    }
    div[data-testid="stVerticalBlock"] { gap: 1rem; }

    /* Cinema gossip vibe */
    .stApp {
        background-color: #1a0000;
        background-image: 
            repeating-linear-gradient(45deg, transparent, transparent 35px, rgba(139, 0, 0, 0.03) 35px, rgba(139, 0, 0, 0.03) 70px),
            url("data:image/svg+xml,%3Csvg width='60' height='60' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M30 0l5 10h10l-8 8 3 10-10-5-10 5 3-10-8-8h10z' fill='%238b0000' fill-opacity='0.03'/%3E%3C/svg%3E");
    }
    /* Gossip column header */
    .gossip-header {
        background: linear-gradient(135deg, #8b0000 0%, #dc143c 100%);
        border-radius: 20px;
        padding: 40px;
        margin: 20px 0 40px 0;
        box-shadow: 0 10px 40px rgba(220, 20, 60, 0.3);
        border: 3px solid #ff1744;
    }
    .gossip-title {
        font-size: 3.5rem;
        font-weight: 900;
        color: #fff;
        text-align: center;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.7);
        letter-spacing: 3px;
    }
    .gossip-subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #ffd700;
        margin-top: 10px;
        font-style: italic;
    }
    /* Sidebar stats - vertical */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a0000 0%, #2d0000 100%);
        border-right: 3px solid #8b0000;
    }
    .stat-box-vertical {
        background: linear-gradient(135deg, rgba(139, 0, 0, 0.3), rgba(220, 20, 60, 0.2));
        border: 2px solid #8b0000;
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        text-align: center;
        box-shadow: 0 5px 15px rgba(220, 20, 60, 0.2);
    }
    .stat-emoji {
        font-size: 2.5rem;
        margin-bottom: 10px;
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: 900;
        color: #ff1744;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    .stat-label {
        font-size: 0.95rem;
        color: #ffd700;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 8px;
    }
    /* Chat-style query box */
    .query-container {
        background: rgba(220, 20, 60, 0.1);
        border: 2px solid #8b0000;
        border-radius: 15px;
        padding: 25px;
        margin: 30px 0;
    }
    .stTextInput > div > div > input {
        background: rgba(0, 0, 0, 0.6);
        border: 2px solid #ff1744;
        border-radius: 12px;
        padding: 18px 25px;
        font-size: 1.2rem;
        color: white;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.5);
    }
    .stTextInput > div > div > input:focus {
        border-color: #ffd700;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.4);
    }
    /* Post button - Twitter style */
    .stButton > button {
        background: linear-gradient(90deg, #ff1744, #f50057);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 15px 50px;
        font-weight: 700;
        font-size: 1.15rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 5px 20px rgba(255, 23, 68, 0.5);
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(255, 23, 68, 0.7);
    }
    /* Answer - Gossip card style */
    .gossip-answer {
        background: linear-gradient(135deg, rgba(139, 0, 0, 0.4), rgba(220, 20, 60, 0.2));
        border-left: 6px solid #ff1744;
        border-radius: 15px;
        padding: 30px;
        margin: 25px 0;
        box-shadow: 0 8px 25px rgba(220, 20, 60, 0.3);
        border: 1px solid #8b0000;
    }
    .tea-emoji {
        font-size: 2rem;
        margin-right: 10px;
        animation: steam 2s infinite;
    }
    @keyframes steam {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
    }
    /* Secret badge - EXTRA */
    .secret-pill {
        background: linear-gradient(90deg, #ff1744, #ffd700);
        color: #000;
        padding: 8px 20px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 800;
        display: inline-block;
        margin: 10px 5px;
        box-shadow: 0 3px 10px rgba(255, 215, 0, 0.4);
        animation: glow 2s infinite;
    }
    @keyframes glow {
        0%, 100% { box-shadow: 0 3px 10px rgba(255, 215, 0, 0.4); }
        50% { box-shadow: 0 5px 20px rgba(255, 215, 0, 0.8); }
    }
    /* Sources - Cards */
    .source-gossip-card {
        background: rgba(0, 0, 0, 0.4);
        border: 2px solid #8b0000;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        transition: all 0.3s;
    }
    .source-gossip-card:hover {
        border-color: #ff1744;
        transform: translateX(5px);
        box-shadow: 0 5px 20px rgba(255, 23, 68, 0.3);
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(139, 0, 0, 0.3);
        border: 2px solid #8b0000;
        border-radius: 10px;
        color: white;
        padding: 12px 25px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #ff1744, #f50057);
        border-color: #ff1744;
    }
    /* Feedback - Twitter style */
    .feedback-row {
        display: flex;
        gap: 15px;
        margin: 20px 0;
    }
    /* Metrics plots */
    .plot-container {
        background: rgba(0, 0, 0, 0.4);
        border: 2px solid #8b0000;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize
@st.cache_resource
def load_rag_system():
    return CinemaSecretsRAG()


rag = load_rag_system()

# Sidebar with VERTICAL stats
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin: 20px 0;">
        <h2 style="color: #ff1744; font-size: 1.5rem;">🍿 GOSSIP STATS</h2>
    </div>
    """, unsafe_allow_html=True)

    stats = rag.get_statistics()

    st.markdown(f"""
    <div class="stat-box-vertical">
        <div class="stat-emoji">🎬</div>
        <div class="stat-number">{stats['total_movies']}</div>
        <div class="stat-label">Movies</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-box-vertical">
        <div class="stat-emoji">🤫</div>
        <div class="stat-number">{stats['secret_chunks']}</div>
        <div class="stat-label">Secrets</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-box-vertical">
        <div class="stat-emoji">📊</div>
        <div class="stat-number">{stats['total_chunks']}</div>
        <div class="stat-label">Total Facts</div>
    </div>
    """, unsafe_allow_html=True)

    llm_emoji = "🤖" if stats['has_llm'] else "⚠️"
    st.markdown(f"""
    <div class="stat-box-vertical">
        <div class="stat-emoji">{llm_emoji}</div>
        <div class="stat-number">{"AI" if stats['has_llm'] else "OFF"}</div>
        <div class="stat-label">Powered By</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔥 Quick Tea")
    if st.button("🎬 Inception Secrets", use_container_width=True):
        st.session_state.quick_query = "What secrets about Inception?"
    if st.button("🦇 Dark Knight Tea", use_container_width=True):
        st.session_state.quick_query = "Behind scenes Dark Knight"
    if st.button("⚡ Matrix Gossip", use_container_width=True):
        st.session_state.quick_query = "Matrix production secrets"

# Main header
st.markdown("""
<div class="gossip-header">
    <h1 class="gossip-title">🍿 CineRAG</h1>
    <p class="gossip-subtitle">Your Source for Cinema's Hottest Secrets & Gossip</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["☕ SPILL THE TEA", "📊 THE RECEIPTS", "ℹ️ ABOUT"])

# Tab 1: Main gossip
with tab1:
    st.markdown('<div class="query-container">', unsafe_allow_html=True)
    st.markdown("### 💬 What's the tea?")

    query = st.text_input(
        "",
        placeholder="Ask me ANYTHING about movies... (e.g., What REALLY happened on set of Titanic?)",
        label_visibility="collapsed",
        value=st.session_state.get('quick_query', '')
    )

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        post_button = st.button("🔥 SPILL IT", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if query and (post_button or st.session_state.get('quick_query')):
        with st.spinner("☕ Brewing the tea..."):
            start_time = time.time()
            result = rag.ask(query, top_k=5, use_llm=True)
            elapsed = time.time() - start_time

        if 'quick_query' in st.session_state:
            del st.session_state.quick_query

        # Answer
        st.markdown(f"""
        <div class="gossip-answer">
            <h3><span class="tea-emoji">☕</span> THE TEA</h3>
            <p style="font-size: 1.15rem; line-height: 1.8; color: #fff; margin-top: 15px;">
                {result['answer']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Badges
        if result['has_secrets']:
            st.markdown('<span class="secret-pill">🎬 INSIDER SECRETS REVEALED</span>', unsafe_allow_html=True)

        st.markdown(f'<span class="secret-pill">⚡ {elapsed:.2f}s</span>', unsafe_allow_html=True)

        # Feedback - Twitter style
        st.markdown("### 🗳️ Rate this tea")
        col1, col2, col3 = st.columns([1, 1, 3])

        with col1:
            if st.button("👍 HOT TEA", key="good", use_container_width=True):
                rag.feedback.save_feedback(result['query_id'], rating=1)
                st.success("Spicy! 🔥")

        with col2:
            if st.button("👎 LUKEWARM", key="bad", use_container_width=True):
                st.session_state.show_feedback = True

        if st.session_state.get('show_feedback', False):
            with st.form("feedback"):
                feedback = st.text_area("What went wrong?")
                if st.form_submit_button("Submit"):
                    rag.feedback.save_feedback(result['query_id'], rating=0, feedback_text=feedback)
                    st.success("Thanks for the feedback!")
                    st.session_state.show_feedback = False

        # Sources
        st.markdown("---")
        st.markdown("### 📰 SOURCES (Receipts)")

        for i, source in enumerate(result['sources'], 1):
            secret_badge = "🤫 EXCLUSIVE" if source['is_secret'] else ""

            st.markdown(f"""
            <div class="source-gossip-card">
                <h4 style="color: #ff1744;">{i}. {source['movie_title']} {secret_badge}</h4>
                <p style="color: #ffd700; font-size: 0.85rem; margin: 5px 0;">[{source['chunk_type']}]</p>
                <p style="color: #ddd; line-height: 1.6; margin: 10px 0;">{source['text'][:300]}...</p>
                <p style="color: #888; font-size: 0.85rem;">Relevance: {source['relevance_score']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

# Tab 2: Metrics with PLOTS
with tab2:
    st.markdown("### 📊 THE RECEIPTS - System Performance")

    from metrics_tracker import MetricsTracker

    tracker = MetricsTracker()

    days = st.selectbox("Time Period", [1, 7, 30], index=1)
    report = tracker.generate_report(days=days)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.markdown("#### 📈 Query Stats")

        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=report['overview']['total_queries'],
            title="Total Queries",
            delta={'reference': report['overview']['total_queries'] * 0.8}
        ))
        fig.update_layout(height=200, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.markdown("#### ⚡ Latency")
        latency = report['performance']['latency']

        fig = go.Figure(go.Bar(
            x=['Avg', 'P95', 'P99'],
            y=[latency['avg_ms'], latency['p95_ms'], latency['p99_ms']],
            marker_color=['#ff1744', '#f50057', '#8b0000']
        ))
        fig.update_layout(height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.markdown("#### 😊 Satisfaction")

        sat_rate = report['performance']['satisfaction_rate']
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=sat_rate * 100,
            title="Satisfaction %",
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#ff1744"},
                   'steps': [
                       {'range': [0, 50], 'color': "#8b0000"},
                       {'range': [50, 75], 'color': "#dc143c"}
                   ]}
        ))
        fig.update_layout(height=250, paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.markdown("#### 💰 Cost")
        st.metric("Estimated Cost", f"${report['costs']['estimated_cost_usd']:.4f}")
        st.metric("Per Query", f"${report['costs']['cost_per_query']:.6f}")
        st.markdown('</div>', unsafe_allow_html=True)

# Tab 3: About
with tab3:
    st.markdown("### About CineRAG")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **🍿 What's This?**
        Your gossip source for cinema secrets:
        - 🎬 Behind-the-scenes drama
        - 🤫 Production secrets
        - 💰 Budget controversies  
        - 🎭 Casting tea
        **🔧 Tech:**
        - Hybrid Search (BM25 + Semantic)
        - Groq AI (Mixtral-8x7b)
        - 85.7% accuracy
        - <1s response time
        """)

    with col2:
        st.markdown("""
        **📊 The Numbers:**
        - 466 movies
        - 252 with secrets
        - 3,000+ facts
        - Real-time AI
        **👩‍💻 Built By:**
        **Rima Alaya**
        AI/ML Engineer
        [GitHub](https://github.com/RimaAlaya) | [LinkedIn](https://linkedin.com/in/rima-alaya)
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <p style="color: #888;">🍿 Spilling cinema tea since 2025 | Built with ❤️ and ☕</p>
</div>
""", unsafe_allow_html=True)