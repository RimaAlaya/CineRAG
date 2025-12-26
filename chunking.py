import json
from pathlib import Path
from typing import List, Dict


def create_plot_chunk(movie: Dict) -> Dict:
    """Create a chunk for the movie plot/overview"""
    if not movie.get('overview'):
        return None

    text = f"{movie['title']} ({movie.get('release_date', 'Unknown')[:4]}): {movie['overview']}"

    if movie.get('tagline'):
        text = f"{movie['tagline']}\n\n{text}"

    return {
        'movie_id': movie['id'],
        'movie_title': movie['title'],
        'chunk_type': 'plot',
        'text': text,
        'metadata': {
            'year': movie.get('release_date', '')[:4],
            'genres': movie.get('genres', []),
            'rating': movie.get('vote_average'),
        }
    }


def create_cast_chunk(movie: Dict) -> Dict:
    """Create a chunk for cast information"""
    if not movie.get('cast'):
        return None

    # Build cast text
    cast_list = []
    for actor in movie['cast'][:8]:  # Top 8 actors
        cast_list.append(f"{actor['name']} as {actor['character']}")

    text = f"{movie['title']} stars: {', '.join(cast_list)}."

    return {
        'movie_id': movie['id'],
        'movie_title': movie['title'],
        'chunk_type': 'cast',
        'text': text,
        'metadata': {
            'year': movie.get('release_date', '')[:4],
            'num_cast': len(movie['cast'])
        }
    }


def create_crew_chunk(movie: Dict) -> Dict:
    """Create a chunk for director and crew"""
    if not movie.get('directors'):
        return None

    directors = ', '.join(movie['directors'])
    text = f"{movie['title']} was directed by {directors}."

    # Add genre and year context
    if movie.get('genres'):
        genres = ', '.join(movie['genres'])
        text += f" This {genres} film was released in {movie.get('release_date', '')[:4]}."

    return {
        'movie_id': movie['id'],
        'movie_title': movie['title'],
        'chunk_type': 'crew',
        'text': text,
        'metadata': {
            'directors': movie['directors'],
            'year': movie.get('release_date', '')[:4]
        }
    }


def create_metadata_chunk(movie: Dict) -> Dict:
    """Create a chunk for ratings, runtime, and other metadata"""
    text_parts = [f"{movie['title']}"]

    if movie.get('release_date'):
        text_parts.append(f"released in {movie['release_date'][:4]}")

    if movie.get('genres'):
        genres = ', '.join(movie['genres'])
        text_parts.append(f"Genre: {genres}")

    if movie.get('runtime'):
        text_parts.append(f"Runtime: {movie['runtime']} minutes")

    if movie.get('vote_average'):
        text_parts.append(f"Rating: {movie['vote_average']}/10 based on {movie.get('vote_count', 0)} votes")

    if movie.get('popularity'):
        text_parts.append(f"Popularity score: {movie['popularity']}")

    text = ". ".join(text_parts) + "."

    return {
        'movie_id': movie['id'],
        'movie_title': movie['title'],
        'chunk_type': 'metadata',
        'text': text,
        'metadata': {
            'rating': movie.get('vote_average'),
            'runtime': movie.get('runtime'),
            'year': movie.get('release_date', '')[:4],
            'genres': movie.get('genres', [])
        }
    }


def create_chunks_for_movie(movie: Dict) -> List[Dict]:
    """Create all chunks for a single movie"""
    chunks = []

    # Create each type of chunk
    chunk_creators = [
        create_plot_chunk,
        create_cast_chunk,
        create_crew_chunk,
        create_metadata_chunk
    ]

    for creator in chunk_creators:
        chunk = creator(movie)
        if chunk:  # Only add if chunk was created (has data)
            chunks.append(chunk)

    return chunks


def process_all_movies(input_file: str, output_file: str):
    """Process all movies and create chunks"""
    print("📚 Loading movies...")
    with open(input_file, 'r', encoding='utf-8') as f:
        movies = json.load(f)

    print(f"✅ Loaded {len(movies)} movies")
    print("\n🔪 Creating chunks...")

    all_chunks = []
    movies_with_chunks = 0
    chunk_type_counts = {
        'plot': 0,
        'cast': 0,
        'crew': 0,
        'metadata': 0
    }

    for movie in movies:
        chunks = create_chunks_for_movie(movie)
        if chunks:
            all_chunks.extend(chunks)
            movies_with_chunks += 1

            # Count chunk types
            for chunk in chunks:
                chunk_type_counts[chunk['chunk_type']] += 1

    print(f"\n✅ Created {len(all_chunks)} chunks from {movies_with_chunks} movies")
    print("\n📊 Chunk distribution:")
    for chunk_type, count in chunk_type_counts.items():
        print(f"   {chunk_type}: {count}")

    # Save chunks
    print(f"\n💾 Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print("✅ Chunks saved!")

    # Show sample chunks
    print("\n📄 Sample chunks:")
    print("=" * 80)
    for i, chunk in enumerate(all_chunks[:3]):
        print(f"\nChunk {i + 1} [{chunk['chunk_type']}]:")
        print(f"Movie: {chunk['movie_title']}")
        print(f"Text: {chunk['text'][:200]}...")
        print("-" * 80)

    return all_chunks


if __name__ == "__main__":
    print("🎬 Movie Chunking System")
    print("=" * 80)

    # Create output directory
    Path("data").mkdir(exist_ok=True)

    # Process movies
    chunks = process_all_movies(
        input_file='data/movies_full.json',
        output_file='data/movie_chunks.json'
    )

    print("\n" + "=" * 80)
    print("🎉 Chunking complete!")
    print(f"📊 Total chunks: {len(chunks)}")
    print(f"📁 Output: data/movie_chunks.json")
    print("\n💡 Next step: Create embeddings from these chunks!")