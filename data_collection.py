import requests
import json
import time
from pathlib import Path
from tqdm import tqdm
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
TMDB_API_KEY = os.getenv('TMDB_API_KEY')

if not TMDB_API_KEY:
    raise ValueError(
        "TMDB_API_KEY not found! Please create a .env file with your API key.\n"
        "See .env.example for the required format."
    )

BASE_URL = "https://api.themoviedb.org/3"

# Create data directory
Path("data").mkdir(exist_ok=True)


def get_popular_movies(num_pages=25):
    """Get list of popular movie IDs (500 movies = 25 pages * 20 per page)"""
    movie_ids = []
    print(f"📥 Fetching popular movies...")

    for page in tqdm(range(1, num_pages + 1), desc="Getting movie list"):
        url = f"{BASE_URL}/movie/popular?api_key={TMDB_API_KEY}&page={page}"
        response = requests.get(url)

        if response.status_code == 200:
            results = response.json()['results']
            movie_ids.extend([movie['id'] for movie in results])
        else:
            print(f"❌ Error on page {page}: {response.status_code}")

        time.sleep(0.25)  # Rate limiting - be nice to API

    print(f"✅ Found {len(movie_ids)} movies")
    return movie_ids


def get_movie_details(movie_id):
    """Get full details for a single movie"""
    url = f"{BASE_URL}/movie/{movie_id}?api_key={TMDB_API_KEY}&append_to_response=credits,keywords"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    return None


def clean_movie_data(movie):
    """Extract only the fields we need"""
    if not movie:
        return None

    # Get top 10 cast members
    cast = []
    if 'credits' in movie and 'cast' in movie['credits']:
        cast = [
            {
                'name': person['name'],
                'character': person['character']
            }
            for person in movie['credits']['cast'][:10]
        ]

    # Get directors
    directors = []
    if 'credits' in movie and 'crew' in movie['credits']:
        directors = [
            person['name']
            for person in movie['credits']['crew']
            if person['job'] == 'Director'
        ]

    # Get genres
    genres = [g['name'] for g in movie.get('genres', [])]

    # Clean data
    cleaned = {
        'id': movie['id'],
        'title': movie['title'],
        'overview': movie.get('overview', ''),
        'release_date': movie.get('release_date', ''),
        'runtime': movie.get('runtime'),
        'vote_average': movie.get('vote_average'),
        'vote_count': movie.get('vote_count'),
        'popularity': movie.get('popularity'),
        'genres': genres,
        'cast': cast,
        'directors': directors,
        'tagline': movie.get('tagline', ''),
        'budget': movie.get('budget'),
        'revenue': movie.get('revenue'),
    }

    return cleaned


def collect_all_movies(num_movies=500):
    """Main function to collect all movie data"""
    # Get movie IDs
    movie_ids = get_popular_movies(num_pages=num_movies // 20)

    # Collect details for each movie
    movies = []
    print(f"\n📚 Collecting detailed information...")

    for movie_id in tqdm(movie_ids, desc="Fetching movie details"):
        movie = get_movie_details(movie_id)
        cleaned = clean_movie_data(movie)

        if cleaned and cleaned['overview']:  # Only keep movies with descriptions
            movies.append(cleaned)

        time.sleep(0.25)  # Rate limiting

        # Save progress every 50 movies
        if len(movies) % 50 == 0:
            with open('data/movies_progress.json', 'w', encoding='utf-8') as f:
                json.dump(movies, f, indent=2, ensure_ascii=False)

    # Final save
    with open('data/movies_full.json', 'w', encoding='utf-8') as f:
        json.dump(movies, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Collected {len(movies)} movies with full details")
    print(f"📁 Saved to: data/movies_full.json")

    # Print sample
    print("\n📊 Sample movie:")
    print(json.dumps(movies[0], indent=2, ensure_ascii=False)[:500] + "...")

    return movies


if __name__ == "__main__":
    print("🎬 TMDB Movie Data Collector")
    print("=" * 50)

    # Test connection first
    url = f"{BASE_URL}/movie/popular?api_key={TMDB_API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        print("✅ API connection successful!\n")

        # Start collection
        movies = collect_all_movies(num_movies=500)

        print("\n" + "=" * 50)
        print(f"🎉 Collection complete!")
        print(f"📊 Total movies: {len(movies)}")
        print(f"📁 File: data/movies_full.json")

    else:
        print(f"❌ API Error: {response.status_code}")
        print("Check your API key!")