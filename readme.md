---
title: CineRAG
emoji: "🎬"
colorFrom: "purple"
colorTo: "pink"
sdk: streamlit
sdk_version: "1.29.0"
app_file: app.py
pinned: false
---


# 🎬 CineRAG - Cinema Secrets Encyclopedia

**Beyond IMDB: A RAG system for true cinema lovers with behind-the-scenes secrets, production trivia, and insider stories.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-🦜-green.svg)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-⚡-orange.svg)](https://groq.com)

[🚀 Live Demo](#) | [📹 Video Demo](#) | [📊 Technical Details](#technical-details)

---

## 🎯 The Problem

**Movie databases give you the basics** - who directed it, who starred in it, what it's about.

**But cinema fans want more:**
- How did Heath Ledger prepare for the Joker?
- What went wrong during Titanic's filming?
- Why was the Inception hallway scene so hard to film?
- Did the cast of your favorite movie take anything from the set?

**This information is scattered** across Reddit threads, Wikipedia trivia sections, YouTube videos, and fan forums.

**CineRAG solves this** by aggregating behind-the-scenes secrets, production stories, and insider trivia into one searchable system.

---

## ✨ What Makes This Different

### Traditional Movie Database:
> **Q:** "Tell me about Inception"  
> **A:** "Inception (2010) directed by Christopher Nolan. Stars Leonardo DiCaprio. A thief who steals corporate secrets..."

### CineRAG:
> **Q:** "What secrets are there about Inception?"  
> **A:** "Nolan wrote Inception over 10 years. The rotating hallway fight? They actually built a rotating corridor - no CGI. Joseph Gordon-Levitt did his own stunts and got seriously dizzy filming those sequences."

**This is what cinema fans get excited about.**

---

## 🔥 Features

- **🎬 Cinema Secrets**: 252 movies with behind-the-scenes content
- **🔍 Hybrid Search**: BM25 + Semantic (85.7% Recall@3)
- **🤖 AI Answers**: Groq's Mixtral for natural responses
- **⚡ Fast**: <1 second per query
- **📚 Comprehensive**: 466 movies, 3,000+ chunks

---

## 🚀 Quick Start

### Installation
```bash
# Clone
git clone https://github.com/RimaAlaya/CineRAG.git
cd CineRAG

# Install dependencies
pip install -r requirements.txt

# Setup API keys
cp .env.example .env
# Add your GROQ_API_KEY and TMDB_API_KEY
```

### Run the App
```bash
streamlit run app.py
```

Visit `http://localhost:8501`

---

## 💡 Example Queries

### Basic Movie Info
- "Who directed Inception?"
- "What is The Matrix about?"
- "Leonardo DiCaprio movies"

### Cinema Secrets 🎬
- "What secrets are there about Inception?"
- "Behind the scenes of The Dark Knight"
- "Tell me trivia about Titanic filming"
- "Production stories from The Matrix"

---

## 🏗️ Architecture
```
User Query
    ↓
Hybrid Search Engine
    ├─ Semantic Search (FAISS + embeddings)
    └─ BM25 Keyword Search
    ↓
Score Fusion (0.5 + 0.5)
    ↓
Top K Relevant Chunks
    ↓
Groq LLM (Mixtral-8x7b)
    ↓
Natural Answer + Sources
```

### Data Pipeline
```
TMDB API → Movies Basic Info (466 movies)
    ↓
Wikipedia Scraping → Behind-the-Scenes Secrets (252 movies)
    ↓
Smart Chunking → 3,000+ Searchable Chunks
    ├─ plot
    ├─ cast
    ├─ crew
    ├─ metadata
    └─ secrets (production, filming, casting, reception)
    ↓
Dual Indexing
    ├─ FAISS (semantic)
    └─ BM25 (keyword)
```

---

## 📊 Technical Details

### Performance Metrics

| Metric | Baseline (Semantic Only) | Hybrid (BM25 + Semantic) | Improvement |
|--------|--------------------------|--------------------------|-------------|
| Recall@3 | 81.5% | **85.7%** | +4.3% ✅ |
| Recall@5 | 88.8% | **93.8%** | +5.0% ✅ |
| Cross-Reference | 6.2% | **71.0%** | +64.8% 🚀 |
| Avg Latency | 21ms | 21ms | - |

**Key Insight:** Hybrid search dramatically improved cross-reference queries (e.g., "Leonardo DiCaprio movies") through exact keyword matching.

### Tech Stack

- **Vector Search**: FAISS + sentence-transformers (all-MiniLM-L6-v2)
- **Keyword Search**: BM25 (rank-bm25)
- **LLM**: Groq (Mixtral-8x7b-32768)
- **Orchestration**: LangChain
- **Data Sources**: TMDB API + Wikipedia
- **UI**: Streamlit
- **Language**: Python 3.12+

### Dataset

- **Movies**: 466 from TMDB
- **Movies with Secrets**: 252
- **Total Chunks**: ~3,000
- **Secret Chunks**: 800+
- **Chunk Types**: plot, cast, crew, metadata, secrets_production, secrets_filming, secrets_casting, secrets_reception

---

## 📁 Project Structure
```
CineRAG/
├── app.py                      # Streamlit web interface
├── cinema_secrets_rag.py       # Main RAG system with Groq
├── hybrid_rag.py              # Baseline hybrid search
├── data_secrets_collector.py  # Wikipedia scraper
├── chunking_secrets.py        # Chunk creation with secrets
├── data/
│   ├── movies_full.json       # TMDB movie data
│   ├── movie_secrets.json     # Wikipedia secrets
│   ├── movie_chunks_with_secrets.json
│   ├── embeddings_with_secrets.npy
│   └── evaluation_dataset.json
├── requirements.txt
└── README.md
```

---

## 🎓 Why This Project Stands Out

### 1. Unique Value Proposition
**Not another generic RAG system.** Focuses on a specific niche (cinema secrets) that no one else addresses.

### 2. Technical Depth
- Hybrid search implementation (not just vector search)
- Systematic evaluation with 164 test questions
- Production-ready with proper error handling
- LangChain integration with Groq (cost-effective LLM)

### 3. Product Thinking
- Identified a gap: "IMDB doesn't have secrets"
- Built a solution: "Aggregate secrets into one place"
- Validated with user stories: "Cinema fans want behind-the-scenes content"

### 4. Execution Quality
- Clean, documented code
- Deployed and working
- Performance metrics tracked
- Iterative improvement (baseline → hybrid → LLM)

---

## 🔮 Future Improvements

- [ ] **More Sources**: Add Reddit (r/MovieDetails), YouTube transcripts
- [ ] **Actor/Director Profiles**: Dedicated pages for people
- [ ] **Cross-Encoder Reranking**: Improve top result accuracy
- [ ] **Conversation Memory**: Multi-turn dialogue
- [ ] **User Feedback Loop**: Learn from interactions
- [ ] **Multi-modal**: Add movie posters, stills

---

## 🤝 Contributing

Want to add more secrets? Improve the prompts? PRs welcome!

Areas for contribution:
1. Add more data sources (Reddit, YouTube, IMDb trivia)
2. Improve secret extraction quality
3. Add more evaluation metrics
4. Enhance UI/UX

---

## 📄 License

MIT License - Free to use for personal or commercial projects.

---

## 👨‍💻 Built By

**Rima Alaya**  
AI/ML Engineer passionate about cinema and intelligent systems

[GitHub](https://github.com/RimaAlaya) | [LinkedIn](https://linkedin.com/in/rima-alaya) | [Email](mailto:rimaalaya76@gmail.com)

---

## 📧 Contact

Questions? Ideas? Want to collaborate?

📧 rimaalaya76@gmail.com  
💼 [LinkedIn](https://linkedin.com/in/rima-alaya)  
🐙 [GitHub](https://github.com/RimaAlaya)

---

**Made with ❤️ for cinema lovers and built with cutting-edge AI technology**

*If you love movies and technology, star this repo ⭐*