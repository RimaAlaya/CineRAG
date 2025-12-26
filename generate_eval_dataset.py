"""
Generates evaluation dataset with test questions and ground truth answers
This is the FOUNDATION of the evaluation framework
"""

import json
import random
from pathlib import Path
from typing import List, Dict


def load_movies():
    """Load movie data"""
    with open('data/movies_full.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def load_chunks():
    """Load chunks with IDs"""
    with open('data/movie_chunks.json', 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    # Add chunk IDs
    for i, chunk in enumerate(chunks):
        chunk['chunk_id'] = i

    return chunks


def create_director_questions(movies: List[Dict], chunks: List[Dict]) -> List[Dict]:
    """Generate questions about directors"""
    questions = []

    for movie in movies:
        if not movie.get('directors'):
            continue

        director = movie['directors'][0]  # First director
        title = movie['title']

        # Find the relevant chunk IDs
        relevant_chunks = [
            c['chunk_id'] for c in chunks
            if c['movie_title'] == title and c['chunk_type'] == 'crew'
        ]

        if not relevant_chunks:
            continue

        # Create question
        questions.append({
            'question': f"Who directed {title}?",
            'movie': title,
            'expected_answer': director,
            'relevant_chunk_ids': relevant_chunks,
            'primary_chunk_type': 'crew',
            'difficulty': 'easy',
            'category': 'factual'
        })

    return questions


def create_cast_questions(movies: List[Dict], chunks: List[Dict]) -> List[Dict]:
    """Generate questions about cast members"""
    questions = []

    for movie in movies:
        if not movie.get('cast') or len(movie['cast']) < 2:
            continue

        title = movie['title']
        actor = movie['cast'][0]['name']  # Lead actor

        # Find relevant chunks
        relevant_chunks = [
            c['chunk_id'] for c in chunks
            if c['movie_title'] == title and c['chunk_type'] == 'cast'
        ]

        if not relevant_chunks:
            continue

        questions.append({
            'question': f"Who stars in {title}?",
            'movie': title,
            'expected_answer': actor,
            'relevant_chunk_ids': relevant_chunks,
            'primary_chunk_type': 'cast',
            'difficulty': 'easy',
            'category': 'factual'
        })

    return questions


def create_plot_questions(movies: List[Dict], chunks: List[Dict]) -> List[Dict]:
    """Generate questions about movie plots"""
    questions = []

    for movie in movies:
        if not movie.get('overview') or len(movie['overview']) < 50:
            continue

        title = movie['title']

        # Find relevant chunks
        relevant_chunks = [
            c['chunk_id'] for c in chunks
            if c['movie_title'] == title and c['chunk_type'] == 'plot'
        ]

        if not relevant_chunks:
            continue

        # Multiple question variations
        question_templates = [
            f"What is {title} about?",
            f"Describe the plot of {title}",
            f"What happens in {title}?",
            f"Summarize {title}"
        ]

        questions.append({
            'question': random.choice(question_templates),
            'movie': title,
            'expected_answer': movie['overview'][:100] + "...",
            'relevant_chunk_ids': relevant_chunks,
            'primary_chunk_type': 'plot',
            'difficulty': 'medium',
            'category': 'descriptive'
        })

    return questions


def create_metadata_questions(movies: List[Dict], chunks: List[Dict]) -> List[Dict]:
    """Generate questions about ratings, genres, runtime"""
    questions = []

    for movie in movies:
        title = movie['title']

        # Find relevant chunks
        relevant_chunks = [
            c['chunk_id'] for c in chunks
            if c['movie_title'] == title and c['chunk_type'] == 'metadata'
        ]

        if not relevant_chunks:
            continue

        # Year questions
        if movie.get('release_date'):
            year = movie['release_date'][:4]
            questions.append({
                'question': f"What year was {title} released?",
                'movie': title,
                'expected_answer': year,
                'relevant_chunk_ids': relevant_chunks,
                'primary_chunk_type': 'metadata',
                'difficulty': 'easy',
                'category': 'factual'
            })

        # Genre questions
        if movie.get('genres'):
            questions.append({
                'question': f"What genre is {title}?",
                'movie': title,
                'expected_answer': ', '.join(movie['genres']),
                'relevant_chunk_ids': relevant_chunks,
                'primary_chunk_type': 'metadata',
                'difficulty': 'easy',
                'category': 'factual'
            })

        # Rating questions
        if movie.get('vote_average'):
            questions.append({
                'question': f"What is the rating of {title}?",
                'movie': title,
                'expected_answer': str(movie['vote_average']),
                'relevant_chunk_ids': relevant_chunks,
                'primary_chunk_type': 'metadata',
                'difficulty': 'easy',
                'category': 'factual'
            })

    return questions


def create_conceptual_questions(movies: List[Dict], chunks: List[Dict]) -> List[Dict]:
    """Generate harder conceptual questions"""
    questions = []

    # Genre-based questions
    genre_movies = {}
    for movie in movies:
        for genre in movie.get('genres', []):
            if genre not in genre_movies:
                genre_movies[genre] = []
            genre_movies[genre].append(movie)

    # Pick popular genres
    for genre in ['Action', 'Science Fiction', 'Drama', 'Comedy']:
        if genre not in genre_movies or len(genre_movies[genre]) < 5:
            continue

        # Get relevant chunk IDs for this genre
        genre_movie_titles = [m['title'] for m in genre_movies[genre][:10]]
        relevant_chunks = [
            c['chunk_id'] for c in chunks
            if c['movie_title'] in genre_movie_titles
        ]

        questions.append({
            'question': f"What are some {genre} movies?",
            'movie': None,  # Multiple movies
            'expected_answer': ', '.join(genre_movie_titles[:5]),
            'relevant_chunk_ids': relevant_chunks[:20],  # Top 20 chunks
            'primary_chunk_type': 'any',
            'difficulty': 'hard',
            'category': 'conceptual'
        })

    return questions


def create_actor_questions(movies: List[Dict], chunks: List[Dict]) -> List[Dict]:
    """Generate questions about actors across movies"""
    questions = []

    # Build actor filmography
    actor_movies = {}
    for movie in movies:
        for cast_member in movie.get('cast', [])[:3]:  # Top 3 actors
            actor = cast_member['name']
            if actor not in actor_movies:
                actor_movies[actor] = []
            actor_movies[actor].append(movie['title'])

    # Pick actors with multiple movies
    for actor, movie_list in actor_movies.items():
        if len(movie_list) >= 2:
            # Find relevant chunks
            relevant_chunks = [
                c['chunk_id'] for c in chunks
                if c['movie_title'] in movie_list and c['chunk_type'] == 'cast'
            ]

            questions.append({
                'question': f"What movies has {actor} starred in?",
                'movie': None,
                'expected_answer': ', '.join(movie_list),
                'relevant_chunk_ids': relevant_chunks,
                'primary_chunk_type': 'cast',
                'difficulty': 'hard',
                'category': 'cross-reference'
            })

    return questions[:10]  # Limit to 10 actor questions


def generate_evaluation_dataset():
    """Main function to generate complete evaluation dataset"""
    print("🎬 Generating Evaluation Dataset")
    print("=" * 80)

    # Load data
    print("\n📚 Loading data...")
    movies = load_movies()
    chunks = load_chunks()
    print(f"✅ Loaded {len(movies)} movies and {len(chunks)} chunks")

    # Generate questions
    print("\n🔄 Generating questions...")

    all_questions = []

    # Director questions (easy, factual)
    director_q = create_director_questions(movies, chunks)
    print(f"   ✅ Director questions: {len(director_q)}")
    all_questions.extend(random.sample(director_q, min(50, len(director_q))))

    # Cast questions (easy, factual)
    cast_q = create_cast_questions(movies, chunks)
    print(f"   ✅ Cast questions: {len(cast_q)}")
    all_questions.extend(random.sample(cast_q, min(30, len(cast_q))))

    # Plot questions (medium)
    plot_q = create_plot_questions(movies, chunks)
    print(f"   ✅ Plot questions: {len(plot_q)}")
    all_questions.extend(random.sample(plot_q, min(30, len(plot_q))))

    # Metadata questions (easy)
    metadata_q = create_metadata_questions(movies, chunks)
    print(f"   ✅ Metadata questions: {len(metadata_q)}")
    all_questions.extend(random.sample(metadata_q, min(40, len(metadata_q))))

    # Conceptual questions (hard)
    conceptual_q = create_conceptual_questions(movies, chunks)
    print(f"   ✅ Conceptual questions: {len(conceptual_q)}")
    all_questions.extend(conceptual_q)

    # Actor questions (hard)
    actor_q = create_actor_questions(movies, chunks)
    print(f"   ✅ Actor questions: {len(actor_q)}")
    all_questions.extend(actor_q)

    # Shuffle
    random.shuffle(all_questions)

    # Add question IDs
    for i, q in enumerate(all_questions):
        q['question_id'] = i

    print(f"\n✅ Total questions generated: {len(all_questions)}")

    # Statistics
    print("\n📊 Question breakdown:")
    categories = {}
    difficulties = {}
    for q in all_questions:
        categories[q['category']] = categories.get(q['category'], 0) + 1
        difficulties[q['difficulty']] = difficulties.get(q['difficulty'], 0) + 1

    print("\n   By category:")
    for cat, count in categories.items():
        print(f"      {cat}: {count}")

    print("\n   By difficulty:")
    for diff, count in difficulties.items():
        print(f"      {diff}: {count}")

    # Save
    output_file = 'data/evaluation_dataset.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved to: {output_file}")

    # Show samples
    print("\n📄 Sample questions:")
    print("=" * 80)
    for i, q in enumerate(all_questions[:5], 1):
        print(f"\n{i}. [{q['difficulty'].upper()}] {q['question']}")
        print(f"   Expected: {q['expected_answer'][:100]}...")
        print(f"   Relevant chunks: {len(q['relevant_chunk_ids'])}")

    return all_questions


if __name__ == "__main__":
    dataset = generate_evaluation_dataset()

    print("\n" + "=" * 80)
    print("🎉 Evaluation dataset created!")
    print("=" * 80)
    print("\n💡 Next step: Build the metrics system to evaluate your RAG!")