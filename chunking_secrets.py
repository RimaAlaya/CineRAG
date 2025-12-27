# chunking_secrets.py
"""
Create chunks from movie secrets
Combines with existing chunks
"""

import json
from pathlib import Path
from typing import List, Dict


def create_secrets_chunks(movie_secrets: dict, movie_data: dict) -> List[Dict]:
    """Create chunks from secrets for a single movie"""
    chunks = []
    movie_id = movie_secrets['movie_id']
    movie_title = movie_data.get('title', movie_secrets.get('wikipedia_title', 'Unknown'))

    # Production secrets chunk
    if movie_secrets.get('production'):
        chunks.append({
            'movie_id': movie_id,
            'movie_title': movie_title,
            'chunk_type': 'secrets_production',
            'text': f"Behind the scenes of {movie_title} - Production: {movie_secrets['production']}",
            'metadata': {
                'secret_type': 'production',
                'year': movie_data.get('release_date', '')[:4] if movie_data.get('release_date') else None,
                'has_secrets': True
            }
        })

    # Filming secrets chunk
    if movie_secrets.get('filming'):
        chunks.append({
            'movie_id': movie_id,
            'movie_title': movie_title,
            'chunk_type': 'secrets_filming',
            'text': f"Behind the scenes of {movie_title} - Filming: {movie_secrets['filming']}",
            'metadata': {
                'secret_type': 'filming',
                'year': movie_data.get('release_date', '')[:4] if movie_data.get('release_date') else None,
                'has_secrets': True
            }
        })

    # Casting secrets chunk
    if movie_secrets.get('casting'):
        chunks.append({
            'movie_id': movie_id,
            'movie_title': movie_title,
            'chunk_type': 'secrets_casting',
            'text': f"Behind the scenes of {movie_title} - Casting: {movie_secrets['casting']}",
            'metadata': {
                'secret_type': 'casting',
                'year': movie_data.get('release_date', '')[:4] if movie_data.get('release_date') else None,
                'has_secrets': True
            }
        })

    # Reception/controversies chunk
    reception_parts = []
    if movie_secrets.get('reception'):
        reception_parts.append(movie_secrets['reception'])
    if movie_secrets.get('controversies'):
        reception_parts.append(f"Controversies: {movie_secrets['controversies']}")
    if movie_secrets.get('legacy'):
        reception_parts.append(f"Legacy: {movie_secrets['legacy']}")

    if reception_parts:
        chunks.append({
            'movie_id': movie_id,
            'movie_title': movie_title,
            'chunk_type': 'secrets_reception',
            'text': f"Behind the scenes of {movie_title} - Reception and Impact: {' '.join(reception_parts)}",
            'metadata': {
                'secret_type': 'reception',
                'year': movie_data.get('release_date', '')[:4] if movie_data.get('release_date') else None,
                'has_secrets': True
            }
        })

    return chunks


def merge_with_existing_chunks():
    """Merge secret chunks with existing movie chunks"""
    print("🎬 CREATING SECRET CHUNKS")
    print("=" * 80)

    # Load data
    print("\n📚 Loading data...")

    with open('data/movie_secrets.json', 'r', encoding='utf-8') as f:
        all_secrets = json.load(f)

    with open('data/movies_full.json', 'r', encoding='utf-8') as f:
        movies = json.load(f)

    with open('data/movie_chunks.json', 'r', encoding='utf-8') as f:
        existing_chunks = json.load(f)

    print(f"✅ Loaded {len(all_secrets)} movie secrets")
    print(f"✅ Loaded {len(movies)} movies")
    print(f"✅ Loaded {len(existing_chunks)} existing chunks")

    # Create movie lookup dict
    movie_lookup = {m['id']: m for m in movies}

    # Create secret chunks
    print("\n🔪 Creating secret chunks...")
    secret_chunks = []

    for secret in all_secrets:
        movie_id = secret['movie_id']
        movie_data = movie_lookup.get(movie_id, {})

        chunks = create_secrets_chunks(secret, movie_data)
        secret_chunks.extend(chunks)

    print(f"✅ Created {len(secret_chunks)} secret chunks")

    # Combine with existing chunks
    print("\n🔗 Merging with existing chunks...")
    all_chunks = existing_chunks + secret_chunks

    # Add/update chunk IDs
    for i, chunk in enumerate(all_chunks):
        chunk['chunk_id'] = i

    print(f"✅ Total chunks: {len(all_chunks)}")

    # Save merged chunks
    output_file = 'data/movie_chunks_with_secrets.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved to: {output_file}")

    # Stats
    print("\n📊 Chunk breakdown:")
    chunk_types = {}
    for chunk in all_chunks:
        chunk_type = chunk['chunk_type']
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

    for chunk_type, count in sorted(chunk_types.items()):
        indicator = "🆕" if 'secrets' in chunk_type else "  "
        print(f"   {indicator} {chunk_type}: {count}")

    # Show sample secret chunk
    print("\n" + "=" * 80)
    print("📄 SAMPLE SECRET CHUNK:")
    print("=" * 80)

    secret_chunk = next((c for c in all_chunks if 'secrets' in c['chunk_type']), None)
    if secret_chunk:
        print(f"\nMovie: {secret_chunk['movie_title']}")
        print(f"Type: {secret_chunk['chunk_type']}")
        print(f"\nText preview:")
        print(secret_chunk['text'][:400] + "...")

    return all_chunks


if __name__ == "__main__":
    chunks = merge_with_existing_chunks()

    print("\n" + "=" * 80)
    print("🎉 SECRET CHUNKS CREATED!")
    print("=" * 80)
    print("\n💡 Next step: Update RAG system to use new chunks with secrets!")