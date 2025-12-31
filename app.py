# app.py
"""
CineRAG - GOSSIP MODE
Spilling Cinema's Hottest Secrets ☕
"""

import streamlit as st
from cinema_secrets_rag import CinemaSecretsRAG
from metrics_tracker import MetricsTracker
import plotly.graph_objects as go
import time

# IMPORTANT: Add this to your requirements.txt
# streamlit-confetti
try:
    import confetti
except ImportError:
    confetti = None

# Page config
st.set_page_config(
    page_title="CineRAG - Cinema Gossip",
    page_icon="🍿",
    layout="wide"
)

# GOSSIP THEME + REFINEMENTS
st.markdown("""
<style>
    /* HIDE STREAMLIT BRANDING */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {display: none !important;}

    /* TIGHT LAYOUT - NO SCROLLING + REMOVE BLANK BARS */
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 1rem !important;
        max-width: 95% !important;
    }
    div[data-testid="stVerticalBlock"] { gap: 0.8rem; }

    /* Cinema vibe */
    .stApp {
        background-color: #1a0000;
        background-image: 
            repeating-linear-gradient(45deg, transparent, transparent 35px, rgba(139, 0, 0, 0.03) 35px, rgba(139, 0, 0, 0.03) 70px),
            url("data:image/svg+xml,%3Csvg width='60' height='60' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M30 0l5 10h10l-8 8 3 10-10-5-10 5 3-10-8-8h10z' fill='%238b0000' fill-opacity='0.03'/%3E%3C/svg%3E");
    }

    /* BIGGER, CATCHIER HEADER */
    .gossip-header {
        background: linear-gradient(135deg, #8b0000 0%, #dc143c 100%);
        border-radius: 25px;
        padding: 50px 40px;
        margin: 10px 0 30px 0;
        box-shadow: 0 15px 50px rgba(220, 20, 60, 0.5);
        border: 4px solid #ff1744;
        text-align: center;
    }
    .gossip-title {
        font-size: 5rem !important;
        font-weight: 900;
        color: #fff;
        text-shadow: 4px 4px 10px rgba(0,0,0,0.8);
        letter-spacing: 5px;
        margin: 0;
    }
    .gossip-subtitle {
        font-size: 1.6rem;
        color: #ffd700;
        margin-top: 15px;
        font-style: italic;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.6);
    }

    /* Sidebar - Premium name */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a0000 0%, #2d0000 100%);
        border-right: 3px solid #8b0000;
        padding-top: 1rem;
    }
    .stat-box-vertical {
        background: linear-gradient(135deg, rgba(139, 0, 0, 0.3), rgba(220, 20, 60, 0.2));
        border: 2px solid #8b0000;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 5px 15px rgba(220, 20, 60, 0.2);
    }

    /* Query container - Tighter */
    .query-container {
        background: rgba(220, 20, 60, 0.1);
        border: 2px solid #8b0000;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0 20px 0;
    }

    /* Input & button */
    .stTextInput > div > div > input {
        background: rgba(0, 0, 0, 0.6);
        border: 2px solid #ff1744;
        border-radius: 12px;
        padding: 18px 25px;
        font-size: 1.2rem;
        color: white;
    }
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
    }

    /* Answer */
    .gossip-answer {
        background: linear-gradient(135deg, rgba(139, 0, 0, 0.4), rgba(220, 20, 60, 0.2));
        border-left: 6px solid #ff1744;
        border-radius: 15px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(220, 20, 60, 0.3);
        border: 1px solid #8b0000;
    }

    /* Secret pill */
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

    /* UPGRADED TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background: transparent;
        padding: 10px 0;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(139, 0, 0, 0.4);
        border: 3px solid #8b0000;
        border-radius: 20px;
        color: white;
        padding: 15px 35px;
        font-weight: 700;
        font-size: 1.1rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #ff1744, #f50057) !important;
        border-color: #ff1744 !important;
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(255, 23, 68, 0.6);
    }

    /* Feedback buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #ff1744, #f50057) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 20px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 5px 20px rgba(255, 23, 68, 0.5);
        height: 80px !important;
        white-space: nowrap !important;
    }
</style>
""", unsafe_allow_html=True)


# Initialize
@st.cache_resource
def load_rag_system():
    return CinemaSecretsRAG()


rag = load_rag_system()

# Sidebar - CINE STATS
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #ff1744; font-size: 1.8rem;'>🍿 CINE STATS</h2>",
                unsafe_allow_html=True)

    stats = rag.get_statistics()

    st.markdown(f"""
    <div class="stat-box-vertical">
        <div style="font-size: 2.8rem;">🎬</div>
        <div style="font-size: 2.4rem; font-weight: 900; color: #ff1744;">{stats['total_movies']}</div>
        <div style="color: #ffd700; text-transform: uppercase; letter-spacing: 2px;">Movies</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-box-vertical">
        <div style="font-size: 2.8rem;">🤫</div>
        <div style="font-size: 2.4rem; font-weight: 900; color: #ff1744;">{stats['secret_chunks']}</div>
        <div style="color: #ffd700; text-transform: uppercase; letter-spacing: 2px;">Secrets</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-box-vertical">
        <div style="font-size: 2.8rem;">📊</div>
        <div style="font-size: 2.4rem; font-weight: 900; color: #ff1744;">{stats['total_chunks']}</div>
        <div style="color: #ffd700; text-transform: uppercase; letter-spacing: 2px;">Total Facts</div>
    </div>
    """, unsafe_allow_html=True)

    llm_emoji = "🤖" if stats['has_llm'] else "⚠️"
    st.markdown(f"""
    <div class="stat-box-vertical">
        <div style="font-size: 2.8rem;">{llm_emoji}</div>
        <div style="font-size: 1.8rem; font-weight: 900; color: #ff1744;">{"AI" if stats['has_llm'] else "OFF"}</div>
        <div style="color: #ffd700; text-transform: uppercase; letter-spacing: 2px;">Powered By</div>
    </div>
    """, unsafe_allow_html=True)

# BIG HEADER
st.markdown("""
<div class="gossip-header">
    <h1 class="gossip-title">🍿 CineRAG</h1>
    <p class="gossip-subtitle">Your Source for Cinema's Hottest Secrets & Gossip</p>
</div>
""", unsafe_allow_html=True)

# TABS - NOW WITH A/B TESTING TAB
tab1, tab2, tab3, tab4 = st.tabs(["☕ SPILL THE TEA", "🎬 THE RECEIPTS", "🧪 A/B TESTS", "ℹ️ ABOUT"])

# Tab 1: Spill the Tea (with A/B testing support)
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

            # === A/B TESTING LOGIC ===
            experiment_name = None
            variant = None
            if st.session_state.get('ab_testing_enabled', False):
                experiment_name = 'prompt_style_test'
                variant = rag.ab_test.assign_variant(experiment_name)
                st.info(f"🧪 Testing variant: {variant.upper()}")

            result = rag.ask(
                query,
                top_k=5,
                use_llm=True,
                experiment_name=experiment_name,
                variant=variant
            )
            elapsed = time.time() - start_time

        if 'quick_query' in st.session_state:
            del st.session_state.quick_query

        st.session_state.current_result = result
        st.session_state.current_elapsed = elapsed
        st.session_state.current_query_id = result['query_id']

    if 'current_result' in st.session_state:
        result = st.session_state.current_result
        elapsed = st.session_state.current_elapsed
        query_id = st.session_state.current_query_id

        st.markdown(f"""
        <div class="gossip-answer">
            <h3><span style="font-size: 2rem; margin-right: 10px;">☕</span> THE TEA</h3>
            <p style="font-size: 1.15rem; line-height: 1.8; color: #fff; margin-top: 15px;">
                {result['answer']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        if result['has_secrets']:
            st.markdown('<span class="secret-pill">🎬 INSIDER SECRETS REVEALED</span>', unsafe_allow_html=True)

        st.markdown(f'<span class="secret-pill">⚡ {elapsed:.2f}s</span>', unsafe_allow_html=True)

        st.markdown("### 🗳️ Rate this tea")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("👍 HOT TEA", key=f"hot_{query_id}", use_container_width=True):
                rag.feedback.save_feedback(query_id, rating=1)
                st.success("Spicy! 🔥")
                if confetti:
                    confetti.fireworks()
                else:
                    st.balloons()

        with col2:
            if st.button("👎 LUKEWARM", key=f"lukewarm_{query_id}", use_container_width=True):
                st.session_state.show_negative_form = query_id

        if st.session_state.get('show_negative_form') == query_id:
            with st.form(key=f"negative_form_{query_id}"):
                feedback_text = st.text_area("What went wrong? Spill the real tea ☕")
                submitted = st.form_submit_button("Submit Feedback")
                if submitted:
                    rag.feedback.save_feedback(query_id, rating=0, feedback_text=feedback_text)
                    st.success("Thanks for the honest tea! We'll do better next time ❤️")
                    del st.session_state.show_negative_form

        st.markdown("---")
        st.markdown("### 📰 SOURCES (Receipts)")
        for i, source in enumerate(result['sources'], 1):
            secret_badge = "🤫 EXCLUSIVE" if source['is_secret'] else ""
            st.markdown(f"""
            <div class="source-gossip-card">
                <h4 style="color: #ff1744;">{i}. {source['movie_title']} {secret_badge}</h4>
                <p style="color: #ffd700; font-size: 0.85rem;">[{source['chunk_type']}]</p>
                <p style="color: #ddd; line-height: 1.6;">{source['text'][:300]}...</p>
                <p style="color: #888; font-size: 0.85rem;">Relevance: {source['relevance_score']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

# Tab 2: The Receipts (Metrics)
with tab2:
    st.markdown("### 📊 System Metrics")
    if st.button("🔄 Refresh Metrics"):
        st.cache_resource.clear()
        st.rerun()
    tracker = MetricsTracker()
    days = st.selectbox("Period", [1, 7, 30], index=1)
    report = tracker.generate_report(days=days)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Queries", report['overview']['total_queries'])
    with col2:
        st.metric("Satisfaction", f"{report['performance']['satisfaction_rate']:.0%}")
    with col3:
        st.metric("Latency", f"{report['performance']['latency']['avg_ms']:.0f}ms")
    with col4:
        st.metric("Cost", f"${report['costs']['estimated_cost_usd']:.4f}")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ⚡ Latency Distribution")
        latency = report['performance']['latency']
        fig = go.Figure(data=[go.Bar(x=['Avg', 'P95', 'P99', 'Max'],
                                     y=[latency['avg_ms'], latency['p95_ms'], latency['p99_ms'], latency['max_ms']],
                                     marker_color=['#ff1744', '#f50057', '#dc143c', '#8b0000'])])
        fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("#### 🎬 Query Types")
        secrets = report['usage']['secrets_distribution']
        fig = go.Figure(data=[
            go.Pie(labels=['With Secrets', 'Basic Info'], values=[secrets['with_secrets'], secrets['without_secrets']],
                   hole=0.4, marker_colors=['#ff1744', '#8b0000'])])
        fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font_color='white', showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.markdown("#### ⚠️ Quality Issues")
    failures = report['quality']
    if failures['total_failures'] > 0:
        st.warning(f"{failures['total_failures']} queries got negative feedback")
        for category, count in failures['failure_categories'].items():
            if count > 0: st.write(f"- {category.replace('_', ' ').title()}: {count}")
    else:
        st.success("No complaints in this period! 🎉")

# Tab 3: NEW A/B Testing Dashboard
with tab3:
    st.markdown("### 🧪 A/B Testing Lab")

    from ab_testing import ABTest

    ab = ABTest()

    # Show active experiment results
    st.markdown("#### Current Experiment: Prompt Style Test")
    st.info("Testing: Gossip Girl vs Documentary narrator style")

    results = ab.get_results('prompt_style_test')

    if results:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 🅰️ Variant A: Gossip Girl")
            st.metric("Queries", results['variant_a']['total_queries'])
            st.metric("Satisfaction", f"{results['variant_a']['satisfaction_rate']:.0%}")
            st.metric("Avg Latency", f"{results['variant_a']['avg_latency_ms']:.0f}ms")

        with col2:
            st.markdown("##### 🅱️ Variant B: Documentary")
            st.metric("Queries", results['variant_b']['total_queries'])
            st.metric("Satisfaction", f"{results['variant_b']['satisfaction_rate']:.0%}")
            st.metric("Avg Latency", f"{results['variant_b']['avg_latency_ms']:.0f}ms")

        # Analysis
        st.markdown("---")
        st.markdown("#### 📊 Analysis")

        analysis = results['analysis']
        winner = analysis['winner']

        if winner != 'tie':
            winner_name = results[f'variant_{winner.lower()}']['name']
            st.success(f"🏆 **Winner: Variant {winner}** ({winner_name})")
            st.metric("Performance Lift", f"{analysis['lift_percent']:+.1f}%")
        else:
            st.warning("📊 Too close to call - need more data")

        st.write(f"**Confidence:** {analysis['confidence']}")
        st.write(f"**Sample Size:** {analysis['sample_size']}")

        if analysis['sample_size'] < 30:
            st.warning("⚠️ Need at least 30 total queries for reliable results")
    else:
        st.info("No experiment data yet. Start asking questions!")

    # Enable experiment toggle
    st.markdown("---")
    if st.checkbox("🧪 Enable A/B testing for my queries"):
        st.session_state.ab_testing_enabled = True
        st.success("You'll be randomly assigned to variant A or B!")
    else:
        st.session_state.ab_testing_enabled = False

# Tab 4: About
with tab4:
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

st.markdown("---")
st.markdown(
    "<div style='text-align: center; padding: 15px; color: #888;'>🍿 Spilling cinema tea since 2025 | Built with ❤️ and ☕</div>",
    unsafe_allow_html=True)