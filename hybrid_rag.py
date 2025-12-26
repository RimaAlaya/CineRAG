"""
Hybrid Search RAG System
Combines BM25 (keyword) + Semantic (vector) search
This should improve Recall@3 by 5-10%
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re
from typing import List, Dict
from pathlib import Path


class HybridMovieRAG:
    """Enhanced RAG with hybrid search (BM25 + Semantic)"""

    def __init__(self, chunks_file: str = 'data/movie_chunks.json'):
        """Initialize hybrid RAG system"""
        self.chunks_file = chunks_file
        self.chunks = []
        self.index = None
        self.embeddings = None
        self.model = None
        self.bm25 = None
        self.tokenized_chunks = []

        print("🎬 Initializing Hybrid RAG System...")
        self._load_chunks()
        self._load_or_create_embeddings()
        self._build_faiss_index()
        self._build_bm25_index()
        print("✅ Hybrid RAG System ready!\n")

    def _load_chunks(self):
        """Load movie chunks"""
        print(f"📚 Loading chunks from {self.chunks_file}...")
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)

        # Add chunk IDs if not present
        for i, chunk in enumerate(self.chunks):
            if 'chunk_id' not in chunk:
                chunk['chunk_id'] = i

        print(f"✅ Loaded {len(self.chunks)} chunks")

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for BM25 (simple word splitting)"""
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens

    def _build_bm25_index(self):
        """Build BM25 index for keyword search"""
        print("🔨 Building BM25 index...")

        # Tokenize all chunks
        self.tokenized_chunks = [
            self._tokenize_text(chunk['text'])
            for chunk in self.chunks
        ]

        # Create BM25 index
        self.bm25 = BM25Okapi(self.tokenized_chunks)

        print("✅ BM25 index built")

    def _load_or_create_embeddings(self):
        """Load or create embeddings"""
        embeddings_file = 'data/embeddings.npy'
        model_cache = 'data/model_name.txt'

        print("🧠 Loading embedding model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        current_model_name = 'all-MiniLM-L6-v2'

        if Path(embeddings_file).exists() and Path(model_cache).exists():
            with open(model_cache, 'r') as f:
                saved_model_name = f.read().strip()

            if saved_model_name == current_model_name:
                print("📦 Loading cached embeddings...")
                self.embeddings = np.load(embeddings_file)
                print(f"✅ Loaded embeddings: {self.embeddings.shape}")
                return

        print("🔄 Creating embeddings...")
        texts = [chunk['text'] for chunk in self.chunks]
        self.embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)

        np.save(embeddings_file, self.embeddings)
        with open(model_cache, 'w') as f:
            f.write(current_model_name)

        print(f"✅ Created embeddings: {self.embeddings.shape}")

    def _build_faiss_index(self):
        """Build FAISS index"""
        print("🔨 Building FAISS index...")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        embeddings_np = self.embeddings.astype('float32')
        self.index.add(embeddings_np)
        print(f"✅ FAISS index built: {self.index.ntotal} vectors")

    def semantic_search(self, query: str, top_k: int = 20) -> List[Dict]:
        """Semantic search using embeddings"""
        query_embedding = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_embedding, k=top_k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            results.append({
                'chunk_id': int(idx),
                'score': float(distance),
                'method': 'semantic'
            })

        return results

    def bm25_search(self, query: str, top_k: int = 20) -> List[Dict]:
        """BM25 keyword search"""
        tokenized_query = self._tokenize_text(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top K indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                'chunk_id': int(idx),
                'score': float(scores[idx]),
                'method': 'bm25'
            })

        return results

    def hybrid_search(self, query: str, top_k: int = 5,
                      semantic_weight: float = 0.5,
                      bm25_weight: float = 0.5) -> List[Dict]:
        """
        Hybrid search combining semantic + BM25

        Args:
            query: Search query
            top_k: Number of final results
            semantic_weight: Weight for semantic scores (0-1)
            bm25_weight: Weight for BM25 scores (0-1)
        """
        # Get candidates from both methods
        semantic_results = self.semantic_search(query, top_k=20)
        bm25_results = self.bm25_search(query, top_k=20)

        # Normalize scores to 0-1 range
        # For semantic (L2 distance): lower is better, so invert
        max_sem_score = max([r['score'] for r in semantic_results]) if semantic_results else 1.0
        for r in semantic_results:
            # Invert and normalize (lower distance = higher score)
            r['normalized_score'] = 1.0 - (r['score'] / max_sem_score if max_sem_score > 0 else 0)

        # For BM25: higher is better, already positive
        max_bm25_score = max([r['score'] for r in bm25_results]) if bm25_results else 1.0
        for r in bm25_results:
            r['normalized_score'] = r['score'] / max_bm25_score if max_bm25_score > 0 else 0

        # Combine scores
        combined_scores = {}

        # Add semantic scores
        for result in semantic_results:
            chunk_id = result['chunk_id']
            combined_scores[chunk_id] = {
                'semantic': result['normalized_score'] * semantic_weight,
                'bm25': 0
            }

        # Add BM25 scores
        for result in bm25_results:
            chunk_id = result['chunk_id']
            if chunk_id not in combined_scores:
                combined_scores[chunk_id] = {'semantic': 0, 'bm25': 0}
            combined_scores[chunk_id]['bm25'] = result['normalized_score'] * bm25_weight

        # Calculate final scores
        final_scores = []
        for chunk_id, scores in combined_scores.items():
            final_score = scores['semantic'] + scores['bm25']
            final_scores.append({
                'chunk_id': chunk_id,
                'final_score': final_score,
                'semantic_score': scores['semantic'],
                'bm25_score': scores['bm25']
            })

        # Sort by final score (descending)
        final_scores.sort(key=lambda x: x['final_score'], reverse=True)

        # Get top K and add full chunk info
        results = []
        for item in final_scores[:top_k]:
            chunk = self.chunks[item['chunk_id']]
            results.append({
                'chunk_id': item['chunk_id'],
                'movie_title': chunk['movie_title'],
                'chunk_type': chunk['chunk_type'],
                'text': chunk['text'],
                'metadata': chunk['metadata'],
                'relevance_score': item['final_score'],
                'semantic_contribution': item['semantic_score'],
                'bm25_contribution': item['bm25_score']
            })

        return results

    def search(self, query: str, top_k: int = 5,
               method: str = 'hybrid') -> List[Dict]:
        """
        Main search interface

        Args:
            query: Search query
            top_k: Number of results
            method: 'semantic', 'bm25', or 'hybrid'
        """
        if method == 'semantic':
            results = self.semantic_search(query, top_k)
            return [self._enrich_result(r) for r in results]
        elif method == 'bm25':
            results = self.bm25_search(query, top_k)
            return [self._enrich_result(r) for r in results]
        else:  # hybrid
            return self.hybrid_search(query, top_k)

    def _enrich_result(self, result: Dict) -> Dict:
        """Add full chunk info to result"""
        chunk = self.chunks[result['chunk_id']]
        return {
            'chunk_id': result['chunk_id'],
            'movie_title': chunk['movie_title'],
            'chunk_type': chunk['chunk_type'],
            'text': chunk['text'],
            'metadata': chunk['metadata'],
            'relevance_score': result['score']
        }

    def get_statistics(self) -> Dict:
        """Get system statistics"""
        chunk_types = {}
        movies = set()

        for chunk in self.chunks:
            chunk_types[chunk['chunk_type']] = chunk_types.get(chunk['chunk_type'], 0) + 1
            movies.add(chunk['movie_title'])

        return {
            'total_chunks': len(self.chunks),
            'total_movies': len(movies),
            'chunk_types': chunk_types,
            'embedding_dimension': self.embeddings.shape[1],
            'has_bm25': self.bm25 is not None,
            'has_semantic': self.index is not None
        }


def compare_search_methods():
    """Compare semantic vs BM25 vs hybrid"""
    print("🔍 Comparing Search Methods")
    print("=" * 80)

    rag = HybridMovieRAG()

    test_queries = [
        "What happens in Miracle on 34th Street?",
        "Who directed Inception?",
        "Leonardo DiCaprio movies",
        "Action movies"
    ]

    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        print("-" * 80)

        # Semantic only
        sem_results = rag.search(query, top_k=3, method='semantic')
        print(f"\n   Semantic Top 1: {sem_results[0]['movie_title']} [{sem_results[0]['chunk_type']}]")

        # BM25 only
        bm25_results = rag.search(query, top_k=3, method='bm25')
        print(f"   BM25 Top 1: {bm25_results[0]['movie_title']} [{bm25_results[0]['chunk_type']}]")

        # Hybrid
        hybrid_results = rag.search(query, top_k=3, method='hybrid')
        print(f"   Hybrid Top 1: {hybrid_results[0]['movie_title']} [{hybrid_results[0]['chunk_type']}]")
        print(
            f"      (Semantic: {hybrid_results[0]['semantic_contribution']:.3f}, BM25: {hybrid_results[0]['bm25_contribution']:.3f})")


if __name__ == "__main__":
    compare_search_methods()