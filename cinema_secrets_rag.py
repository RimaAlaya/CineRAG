# cinema_secrets_rag.py
"""
Cinema Secrets RAG with LangChain + Groq
Hybrid search + LLM generation for cinema enthusiasts
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re
from typing import List, Dict
from pathlib import Path
import os
from dotenv import load_dotenv
from feedback_system import FeedbackSystem
import time
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("⚠️  LangChain not fully installed. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'langchain', 'langchain-groq', 'langchain-core'])
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
from ab_testing import ABTest, PROMPT_VARIANTS


load_dotenv()


class CinemaSecretsRAG:
    """Enhanced RAG with cinema secrets + Groq LLM"""

    def __init__(self, chunks_file: str = 'data/movie_chunks_with_secrets.json'):
        """Initialize Cinema Secrets RAG"""
        self.chunks_file = chunks_file
        self.chunks = []
        self.index = None
        self.embeddings = None
        self.model = None
        self.bm25 = None
        self.tokenized_chunks = []
        self.llm = None
        self.feedback = FeedbackSystem()

        print("🎬 Initializing Cinema Secrets RAG with Groq...")
        self._load_chunks()
        self._load_or_create_embeddings()
        self._build_faiss_index()
        self._build_bm25_index()
        self._initialize_llm()
        self.ab_test = ABTest()
        print("✅ Cinema Secrets RAG ready!\n")

    def _load_chunks(self):
        """Load movie chunks with secrets"""
        print(f"📚 Loading chunks from {self.chunks_file}...")
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)

        # Count secrets
        secret_chunks = sum(1 for c in self.chunks if 'secrets' in c['chunk_type'])
        print(f"✅ Loaded {len(self.chunks)} chunks ({secret_chunks} with secrets)")

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for BM25"""
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens

    def _build_bm25_index(self):
        """Build BM25 index"""
        print("🔨 Building BM25 index...")
        self.tokenized_chunks = [
            self._tokenize_text(chunk['text'])
            for chunk in self.chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        print("✅ BM25 index built")

    def _load_or_create_embeddings(self):
        """Load or create embeddings"""
        embeddings_file = 'data/embeddings_with_secrets.npy'

        print("🧠 Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        if Path(embeddings_file).exists():
            print("📦 Loading cached embeddings...")
            self.embeddings = np.load(embeddings_file)
            print(f"✅ Loaded embeddings: {self.embeddings.shape}")
        else:
            print("🔄 Creating embeddings (this takes 2-3 minutes)...")
            texts = [chunk['text'] for chunk in self.chunks]
            self.embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)

            np.save(embeddings_file, self.embeddings)
            print(f"✅ Created embeddings: {self.embeddings.shape}")

    def _build_faiss_index(self):
        """Build FAISS index"""
        print("🔨 Building FAISS index...")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        embeddings_np = self.embeddings.astype('float32')
        self.index.add(embeddings_np)
        print(f"✅ FAISS index built: {self.index.ntotal} vectors")

    def _initialize_llm(self):
        """Initialize Groq LLM"""
        print("🤖 Initializing Groq LLM...")

        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            print("⚠️  No GROQ_API_KEY found. LLM features disabled.")
            print("   Add GROQ_API_KEY to your .env file to enable.")
            self.llm = None
            return

        try:
            self.llm = ChatGroq(
                temperature=0.7,
                model_name="llama-3.3-70b-versatile",  # Fast and smart
                groq_api_key=api_key
            )
            print("✅ Groq LLM initialized (llama-3.3-70b-versatile)")
        except Exception as e:
            print(f"⚠️  Could not initialize Groq: {e}")
            self.llm = None

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
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                'chunk_id': int(idx),
                'score': float(scores[idx]),
                'method': 'bm25'
            })
        return results

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Hybrid search combining semantic + BM25"""
        # Get candidates
        semantic_results = self.semantic_search(query, top_k=20)
        bm25_results = self.bm25_search(query, top_k=20)

        # Normalize scores
        max_sem = max([r['score'] for r in semantic_results]) if semantic_results else 1.0
        for r in semantic_results:
            r['normalized_score'] = 1.0 - (r['score'] / max_sem if max_sem > 0 else 0)

        max_bm25 = max([r['score'] for r in bm25_results]) if bm25_results else 1.0
        for r in bm25_results:
            r['normalized_score'] = r['score'] / max_bm25 if max_bm25 > 0 else 0

        # Combine scores
        combined_scores = {}

        for result in semantic_results:
            chunk_id = result['chunk_id']
            combined_scores[chunk_id] = {'semantic': result['normalized_score'] * 0.5, 'bm25': 0}

        for result in bm25_results:
            chunk_id = result['chunk_id']
            if chunk_id not in combined_scores:
                combined_scores[chunk_id] = {'semantic': 0, 'bm25': 0}
            combined_scores[chunk_id]['bm25'] = result['normalized_score'] * 0.5

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

        final_scores.sort(key=lambda x: x['final_score'], reverse=True)

        # Enrich results
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
                'is_secret': 'secrets' in chunk['chunk_type']
            })

        return results

    def generate_answer(self, query: str, context_chunks: List[Dict],
                        experiment_name: str = None, variant: str = None) -> str:
        """Generate answer with optional A/B testing"""

        if not self.llm:
            return self._fallback_answer(context_chunks)

        context = "\n\n".join([
            f"[{c['movie_title']} - {c['chunk_type']}]\n{c['text']}"
            for c in context_chunks
        ])

        # A/B Testing: Pick variant
        if experiment_name:
            if not variant:
                variant = self.ab_test.assign_variant(experiment_name)

            # Get experiment config
            if variant == 'A':
                template = PROMPT_VARIANTS['gossip_girl']
            else:
                template = PROMPT_VARIANTS['documentary']
        else:
            # Default behavior (your current prompt)
            is_secrets_query = any(word in query.lower() for word in ['secret', 'behind', 'scene'])
            if is_secrets_query:
                template = """You're sharing insider cinema knowledge. Be brief and exciting.

    Context: {context}
    Question: {question}

    Answer (under 150 words):"""
            else:
                template = """Answer directly and precisely.

    Context: {context}
    Question: {question}

    Answer (2-3 sentences):"""

        # Generate
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain

        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        chain = LLMChain(llm=self.llm, prompt=prompt)

        try:
            response = chain.run(context=context, question=query)
            return response.strip()
        except Exception as e:
            return self._fallback_answer(context_chunks)

    def ask(self, query: str, top_k: int = 5, use_llm: bool = True) -> Dict:
        """
        Main interface with feedback tracking
        """
        start_time = time.time()

        # Search for relevant chunks
        chunks = self.hybrid_search(query, top_k=top_k)

        # Generate answer
        if use_llm and self.llm:
            answer = self.generate_answer(query, chunks)
        else:
            answer = chunks[0]['text'] if chunks else "Sorry, I couldn't find relevant information."

        latency_ms = (time.time() - start_time) * 1000

        # Prepare sources
        sources = [{
            'movie_title': c['movie_title'],
            'chunk_type': c['chunk_type'],
            'text': c['text'][:200] + '...',
            'is_secret': c['is_secret']
        } for c in chunks]

        has_secrets = any(c['is_secret'] for c in chunks)

        # Log to feedback system
        query_id = self.feedback.log_query(
            query=query,
            answer=answer,
            sources=sources,
            has_secrets=has_secrets,
            latency_ms=latency_ms
        )

        return {
            'query_id': query_id,
            'question': query,
            'answer': answer,
            'sources': chunks,
            'has_secrets': has_secrets,
            'latency_ms': round(latency_ms, 2)
        }

    def get_statistics(self) -> Dict:
        """Get system statistics"""
        chunk_types = {}
        movies = set()
        secret_count = 0

        for chunk in self.chunks:
            chunk_types[chunk['chunk_type']] = chunk_types.get(chunk['chunk_type'], 0) + 1
            movies.add(chunk['movie_title'])
            if 'secrets' in chunk['chunk_type']:
                secret_count += 1

        return {
            'total_chunks': len(self.chunks),
            'total_movies': len(movies),
            'secret_chunks': secret_count,
            'chunk_types': chunk_types,
            'has_llm': self.llm is not None
        }


def test_cinema_secrets():
    """Test the system"""
    print("\n" + "=" * 80)
    print("🧪 TESTING CINEMA SECRETS RAG")
    print("=" * 80)

    rag = CinemaSecretsRAG()

    # Stats
    stats = rag.get_statistics()
    print("\n📊 System Stats:")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Movies: {stats['total_movies']}")
    print(f"   Secret chunks: {stats['secret_chunks']}")
    print(f"   LLM enabled: {stats['has_llm']}")

    # Test queries
    test_queries = [
        "What secrets are there about Inception's production?",
        "Tell me behind the scenes stories from The Dark Knight",
        "Who directed Inception?",
        "Leonardo DiCaprio movies"
    ]

    print("\n" + "=" * 80)
    print("🔍 TEST QUERIES")
    print("=" * 80)

    for query in test_queries:
        print(f"\n❓ {query}")
        print("-" * 80)

        result = rag.ask(query, top_k=3, use_llm=True)

        print(f"\n💬 Answer:\n{result['answer']}\n")

        print("📚 Sources:")
        for i, source in enumerate(result['sources'], 1):
            secret_badge = "🎬" if source['is_secret'] else "  "
            print(f"   {i}. {secret_badge} {source['movie_title']} [{source['chunk_type']}]")


if __name__ == "__main__":
    test_cinema_secrets()