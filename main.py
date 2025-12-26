import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle
from typing import List, Dict
from tqdm import tqdm


class MovieRAGSystem:
    """Production-grade RAG system for movie queries"""

    def __init__(self, chunks_file: str = 'data/movie_chunks.json'):
        """Initialize the RAG system"""
        self.chunks_file = chunks_file
        self.chunks = []
        self.index = None
        self.embeddings = None
        self.model = None

        print("🎬 Initializing Movie RAG System...")
        self._load_chunks()
        self._load_or_create_embeddings()
        self._build_index()
        print("✅ RAG System ready!\n")

    def _load_chunks(self):
        """Load movie chunks from JSON"""
        print(f"📚 Loading chunks from {self.chunks_file}...")
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        print(f"✅ Loaded {len(self.chunks)} chunks")

    def _load_or_create_embeddings(self):
        """Load existing embeddings or create new ones"""
        embeddings_file = 'data/embeddings.npy'
        model_cache = 'data/model_name.txt'

        # Load embedding model
        print("🧠 Loading embedding model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        current_model_name = 'all-MiniLM-L6-v2'

        # Check if embeddings already exist
        if Path(embeddings_file).exists() and Path(model_cache).exists():
            with open(model_cache, 'r') as f:
                saved_model_name = f.read().strip()

            if saved_model_name == current_model_name:
                print("📦 Loading cached embeddings...")
                self.embeddings = np.load(embeddings_file)
                print(f"✅ Loaded embeddings: {self.embeddings.shape}")
                return

        # Create new embeddings
        print("🔄 Creating embeddings (this takes 1-2 minutes)...")
        texts = [chunk['text'] for chunk in self.chunks]

        # Create embeddings with progress bar
        self.embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )

        # Save embeddings
        print("💾 Saving embeddings for future use...")
        np.save(embeddings_file, self.embeddings)
        with open(model_cache, 'w') as f:
            f.write(current_model_name)

        print(f"✅ Created embeddings: {self.embeddings.shape}")

    def _build_index(self):
        """Build FAISS index for fast similarity search"""
        print("🔨 Building FAISS index...")

        # Get embedding dimension
        dimension = self.embeddings.shape[1]

        # Create index (L2 distance)
        self.index = faiss.IndexFlatL2(dimension)

        # Add embeddings to index
        embeddings_np = self.embeddings.astype('float32')
        self.index.add(embeddings_np)

        print(f"✅ FAISS index built: {self.index.ntotal} vectors")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant chunks

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of dicts with chunk info and relevance scores
        """
        # Encode query
        query_embedding = self.model.encode([query]).astype('float32')

        # Search index
        distances, indices = self.index.search(query_embedding, k=top_k)

        # Prepare results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            chunk = self.chunks[idx]
            results.append({
                'movie_title': chunk['movie_title'],
                'chunk_type': chunk['chunk_type'],
                'text': chunk['text'],
                'metadata': chunk['metadata'],
                'relevance_score': float(distance),  # Lower is better (L2 distance)
                'chunk_id': int(idx)
            })

        return results

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
            'index_size': self.index.ntotal
        }


def test_system():
    """Test the RAG system with sample queries"""
    print("\n" + "=" * 80)
    print("🧪 Testing RAG System")
    print("=" * 80)

    # Initialize system
    rag = MovieRAGSystem()

    # Print statistics
    stats = rag.get_statistics()
    print("\n📊 System Statistics:")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Total movies: {stats['total_movies']}")
    print(f"   Embedding dimension: {stats['embedding_dimension']}")
    print(f"\n   Chunk types:")
    for chunk_type, count in stats['chunk_types'].items():
        print(f"      {chunk_type}: {count}")

    # Test queries
    test_queries = [
        "Who directed Inception?",
        "What is The Matrix about?",
        "Who stars in Titanic?",
        "Best rated action movies",
    ]

    print("\n" + "=" * 80)
    print("🔍 Test Queries")
    print("=" * 80)

    for query in test_queries:
        print(f"\n❓ Query: '{query}'")
        print("-" * 80)

        results = rag.search(query, top_k=3)

        for i, result in enumerate(results, 1):
            print(f"\n   Result {i}: {result['movie_title']} [{result['chunk_type']}]")
            print(f"   Score: {result['relevance_score']:.4f}")
            print(f"   Text: {result['text'][:150]}...")

    print("\n" + "=" * 80)
    print("✅ Testing complete!")
    print("=" * 80)

    return rag


if __name__ == "__main__":
    # Run tests
    rag_system = test_system()

    # Interactive mode
    print("\n" + "=" * 80)
    print("💬 Interactive Mode (type 'quit' to exit)")
    print("=" * 80)

    while True:
        query = input("\n🎬 Your question: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break

        if not query:
            continue

        results = rag_system.search(query, top_k=3)

        print("\n📋 Results:")
        print("-" * 80)
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['movie_title']} [{result['chunk_type']}]")
            print(f"   {result['text']}")
            print(f"   (Score: {result['relevance_score']:.4f})")