# 🎬 CineRAG - Production-Grade Movie RAG System

**An advanced Retrieval-Augmented Generation system combining semantic and keyword search for movie information retrieval.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Key Achievement

**Improved cross-reference query accuracy by 64.8%** through hybrid search implementation, achieving 85.7% overall Recall@3 on a 164-question evaluation dataset.

---

## ✨ Features

- **Hybrid Search**: Combines BM25 keyword matching with semantic embeddings
- **Systematic Evaluation**: 164 test questions across 4 difficulty levels
- **Production-Ready**: 21ms average latency, 93.8% Recall@5
- **Interactive UI**: Streamlit dashboard with performance visualizations
- **Comprehensive Dataset**: 466 movies, 1,982 searchable chunks

---

## 📊 Performance Results

### Overall Metrics

| Metric | Baseline (Semantic Only) | Hybrid (BM25 + Semantic) | Improvement |
|--------|--------------------------|--------------------------|-------------|
| Recall@1 | 68.3% | 59.0% | -9.3% |
| **Recall@3** | **81.5%** | **85.7%** | **+4.3%** ✅ |
| **Recall@5** | **88.8%** | **93.8%** | **+5.0%** ✅ |
| MRR | 0.790 | 0.776 | -1.4% |
| Hit@3 | 82.9% | 87.2% | +4.3% ✅ |

### Performance by Query Category

| Category | Baseline | Hybrid | Improvement |
|----------|----------|--------|-------------|
| **Factual** | 99.2% | 97.9% | -1.3% |
| **Descriptive** | 46.7% | 53.3% | **+6.7%** ✅ |
| **Cross-Reference** | 6.2% | **71.0%** | **+64.8%** 🚀 |
| **Conceptual** | 0.0% | 0.0% | 0.0% |

**Key Insight:** Hybrid search dramatically improved cross-reference queries (e.g., "Leonardo DiCaprio movies") through exact keyword matching, while maintaining strong performance on factual queries.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Query                               │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │    Hybrid Search Engine        │
        └───────────┬───────────┬───────┘
                    │           │
        ┌───────────▼──┐    ┌──▼──────────────┐
        │  Semantic     │    │  BM25 Keyword   │
        │  (FAISS)      │    │  Search         │
        └───────────┬──┘    └──┬──────────────┘
                    │           │
                    │  Top 20   │  Top 20
                    │           │
        ┌───────────▼───────────▼──────────────┐
        │   Score Fusion (0.5 semantic +       │
        │              0.5 BM25)                │
        └───────────────┬──────────────────────┘
                        │
                        ▼ Top K Results
        ┌───────────────────────────────────────┐
        │  1,982 Chunks from 466 Movies         │
        │  (plot, cast, crew, metadata)         │
        └───────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/cinerag.git
cd cinerag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Collection

```bash
# Get TMDB API key from https://www.themoviedb.org/settings/api
# Add to data_collection.py line 8

# Collect movie data (5-7 minutes)
python data_collection.py

# Create chunks
python chunking.py
```

### Generate Evaluation Dataset

```bash
python generate_eval_dataset.py
```

### Run Evaluation

```bash
# Baseline evaluation
python evaluate_baseline.py

# Compare baseline vs hybrid
python compare_systems.py
```

### Launch Web Interface

```bash
streamlit run app.py
```

---

## 📦 Dependencies

```
streamlit==1.29.0
sentence-transformers==2.2.2
faiss-cpu==1.7.4
transformers==4.36.0
torch==2.1.0
numpy==1.24.3
rank-bm25==0.2.2
plotly==5.18.0
tqdm==4.66.1
```

---

## 📁 Project Structure

```
cinerag/
├── app.py                      # Streamlit web interface
├── main.py                     # Original RAG system (baseline)
├── hybrid_rag.py              # Hybrid search implementation
├── data_collection.py         # TMDB data fetching
├── chunking.py                # Document chunking logic
├── generate_eval_dataset.py   # Test question generator
├── evaluate_baseline.py       # Evaluation framework
├── compare_systems.py         # Performance comparison
├── test_rag.py               # Manual testing suite
├── data/
│   ├── movies_full.json       # Raw movie data
│   ├── movie_chunks.json      # Processed chunks
│   ├── embeddings.npy         # Cached embeddings
│   ├── evaluation_dataset.json # Test questions
│   ├── baseline_results.json  # Baseline metrics
│   ├── hybrid_results.json    # Hybrid metrics
│   └── comparison_report.json # Performance comparison
├── requirements.txt
└── README.md
```

---

## 🔬 Evaluation Methodology

### Test Dataset

- **164 questions** across 4 categories:
  - **Factual** (120): "Who directed Inception?"
  - **Descriptive** (30): "What is The Matrix about?"
  - **Cross-Reference** (10): "Leonardo DiCaprio movies"
  - **Conceptual** (4): "What are some Action movies?"

### Metrics

- **Recall@K**: Proportion of relevant chunks retrieved in top K results
- **MRR (Mean Reciprocal Rank)**: Average of 1/position of first relevant result
- **Hit@K**: Binary - did we retrieve any relevant chunk in top K?
- **Latency**: Average query processing time

### Ground Truth

Each question has labeled `relevant_chunk_ids` for objective evaluation.

---

## 🎓 Technical Insights

### Why Hybrid Search Works

**Problem with Semantic Search Alone:**
- Query: "Leonardo DiCaprio movies"
- Semantic search matches general concepts
- Misses exact actor name across multiple movies

**BM25 Keyword Search:**
- Exact match on "Leonardo DiCaprio" in cast chunks
- Retrieves all movies featuring the actor
- 10X improvement: 6.2% → 71.0%

**Hybrid Approach:**
- Combines both strengths
- Semantic handles conceptual queries
- BM25 handles exact matches
- Weighted fusion (50/50) balances both

### Trade-offs

- **Recall@1 decreased** (-9.3%): Top result sometimes less precise due to BM25 overweighting keywords
- **Recall@3+ improved**: Better overall coverage
- **Acceptable trade-off**: Users typically review top 3-5 results, not just the first

---

## 📈 Future Improvements

### High Priority
- [ ] **Cross-Encoder Reranking**: Use stronger model to re-score top 20 results
- [ ] **Query Classification**: Route queries to specialized retrievers
- [ ] **Metadata Filtering**: Enable genre/year-based search

### Medium Priority
- [ ] **Conversation Memory**: Track context across multiple queries
- [ ] **Actor/Director Entity Linking**: Improve cross-reference accuracy
- [ ] **Dynamic Chunk Sizing**: Optimize chunk length per content type

### Low Priority
- [ ] **Multi-modal Search**: Add movie poster and image search
- [ ] **User Feedback Loop**: Learn from user interactions
- [ ] **Cached Results**: Speed up common queries

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

1. **Reranking Module**: Implement cross-encoder scoring
2. **Query Understanding**: Add intent classification
3. **Visualization**: Enhanced performance dashboards
4. **Testing**: Expand evaluation dataset

---

## 📄 License

MIT License - feel free to use for personal or commercial projects.

---

## 🙏 Acknowledgments

- **TMDB** for movie data API
- **Sentence Transformers** for embedding models
- **FAISS** for vector similarity search
- **Streamlit** for web framework

---

## 📧 Contact

**[Rima ALAYA]**  
AI/ML Engineer | [GitHub](https://github.com/RimaAlaya) | [LinkedIn](https://linkedin.com/in/rima-alaya)

*Built as a demonstration of production-grade RAG system development with systematic evaluation and iterative improvement.*

---

## 📊 Citation

If you use this work in your research or projects, please cite:

```bibtex
@software{cinerag2025,
  author = {Rima ALAYA},
  title = {CineRAG: Production-Grade Movie RAG System},
  year = {2025},
  url = {https://github.com/RimaAlaya/cinerag}
}
```