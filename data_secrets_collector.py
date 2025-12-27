# data_secrets_collector.py
"""
Collect behind-the-scenes secrets and trivia for movies
Sources: Wikipedia (production, trivia, reception sections)
"""

import json
import time
import requests
from bs4 import BeautifulSoup
import re
from pathlib import Path
from tqdm import tqdm


class WikipediaSecretsCollector:
    """Scrape movie secrets from Wikipedia"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MovieRAG/1.0 (Educational Project)'
        })

    def search_wikipedia(self, movie_title: str, year: str = None) -> str:
        """Find Wikipedia page for a movie"""
        # Try with year first
        search_title = f"{movie_title} {year} film" if year else f"{movie_title} film"

        url = "https://en.wikipedia.org/w/api.php"
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': search_title,
            'format': 'json',
            'srlimit': 1
        }

        try:
            response = self.session.get(url, params=params, timeout=10)
            data = response.json()

            if data['query']['search']:
                page_title = data['query']['search'][0]['title']
                return page_title
        except:
            pass

        return None

    def get_page_content(self, page_title: str) -> dict:
        """Get full Wikipedia page content"""
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            'action': 'parse',
            'page': page_title,
            'format': 'json',
            'prop': 'text|sections'
        }

        try:
            response = self.session.get(url, params=params, timeout=15)
            data = response.json()

            if 'parse' in data:
                return {
                    'html': data['parse']['text']['*'],
                    'sections': data['parse']['sections']
                }
        except:
            pass

        return None

    def extract_section_text(self, html: str, section_name: str) -> str:
        """Extract text from a specific section"""
        soup = BeautifulSoup(html, 'html.parser')

        # Find the section heading
        section_patterns = [
            section_name,
            section_name.lower(),
            section_name.title(),
            section_name.upper()
        ]

        text_parts = []
        capturing = False

        for element in soup.find_all(['h2', 'h3', 'p', 'ul']):
            # Check if this is our target section
            if element.name in ['h2', 'h3']:
                heading_text = element.get_text().strip()

                # Start capturing if we found our section
                if any(pattern in heading_text for pattern in section_patterns):
                    capturing = True
                    continue
                # Stop if we hit another section
                elif capturing and element.name == 'h2':
                    break

            # Capture text if we're in the right section
            if capturing:
                if element.name == 'p':
                    text = element.get_text().strip()
                    if len(text) > 50:  # Skip short paragraphs
                        text_parts.append(text)

                elif element.name == 'ul':
                    # Get list items (often trivia)
                    for li in element.find_all('li'):
                        text = li.get_text().strip()
                        if len(text) > 30:
                            text_parts.append(f"• {text}")

        return '\n\n'.join(text_parts)

    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove citations [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)

        # Remove "edit" links
        text = re.sub(r'\[edit\]', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove "citation needed" etc
        text = re.sub(r'\[citation needed\]', '', text)

        return text.strip()

    def get_movie_secrets(self, movie_title: str, year: str = None) -> dict:
        """Get all secrets for a movie"""
        # Find Wikipedia page
        page_title = self.search_wikipedia(movie_title, year)

        if not page_title:
            return None

        # Get page content
        content = self.get_page_content(page_title)

        if not content:
            return None

        html = content['html']

        # Extract different sections
        secrets = {
            'movie_title': movie_title,
            'wikipedia_title': page_title,
            'production': self.clean_text(self.extract_section_text(html, 'Production')),
            'casting': self.clean_text(self.extract_section_text(html, 'Casting')),
            'filming': self.clean_text(self.extract_section_text(html, 'Filming')),
            'reception': self.clean_text(self.extract_section_text(html, 'Reception')),
            'trivia': self.clean_text(self.extract_section_text(html, 'Trivia')),
            'legacy': self.clean_text(self.extract_section_text(html, 'Legacy')),
            'controversies': self.clean_text(self.extract_section_text(html, 'Controversy'))
        }

        # Filter out empty sections
        secrets = {k: v for k, v in secrets.items() if v and len(v) > 100}

        return secrets if len(secrets) > 2 else None  # At least movie_title + 1 section


def collect_secrets_for_all_movies():
    """Main function: collect secrets for all movies in dataset"""
    print("🎬 CINEMA SECRETS COLLECTOR")
    print("=" * 80)

    # Load existing movies
    print("\n📚 Loading movies from CineRAG...")
    with open('data/movies_full.json', 'r', encoding='utf-8') as f:
        movies = json.load(f)

    print(f"✅ Loaded {len(movies)} movies")

    # Initialize collector
    collector = WikipediaSecretsCollector()

    # Collect secrets
    print("\n🔍 Collecting behind-the-scenes secrets from Wikipedia...")
    print("This will take 15-20 minutes (being nice to Wikipedia's servers)\n")

    all_secrets = []
    success_count = 0

    for movie in tqdm(movies, desc="Scraping secrets"):
        try:
            year = movie.get('release_date', '')[:4] if movie.get('release_date') else None
            secrets = collector.get_movie_secrets(movie['title'], year)

            if secrets:
                # Add movie ID for matching
                secrets['movie_id'] = movie['id']
                all_secrets.append(secrets)
                success_count += 1

            # Be nice to Wikipedia
            time.sleep(0.5)  # Half second between requests

            # Save progress every 50 movies
            if len(all_secrets) % 50 == 0 and all_secrets:
                with open('data/secrets_progress.json', 'w', encoding='utf-8') as f:
                    json.dump(all_secrets, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"\n   ⚠️  Error with {movie['title']}: {str(e)}")
            continue

    # Final save
    output_file = 'data/movie_secrets.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_secrets, f, indent=2, ensure_ascii=False)

    # Stats
    print("\n" + "=" * 80)
    print("📊 COLLECTION COMPLETE")
    print("=" * 80)
    print(f"\n✅ Successfully collected secrets for {success_count}/{len(movies)} movies")
    print(f"📁 Saved to: {output_file}")

    # Show what we got
    section_stats = {}
    for secret in all_secrets:
        for section in secret.keys():
            if section not in ['movie_title', 'movie_id', 'wikipedia_title']:
                section_stats[section] = section_stats.get(section, 0) + 1

    print("\n📋 Content breakdown:")
    for section, count in sorted(section_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"   {section}: {count} movies")

    # Show sample
    if all_secrets:
        print("\n" + "=" * 80)
        print("📄 SAMPLE SECRET:")
        print("=" * 80)
        sample = all_secrets[0]
        print(f"\nMovie: {sample['movie_title']}")
        for section, content in sample.items():
            if section not in ['movie_title', 'movie_id', 'wikipedia_title']:
                print(f"\n{section.upper()}:")
                print(content[:300] + "...")
                break

    return all_secrets


if __name__ == "__main__":
    secrets = collect_secrets_for_all_movies()

    print("\n" + "=" * 80)
    print("🎉 SECRET COLLECTION COMPLETE!")
    print("=" * 80)
    print("\n💡 Next step: Create secret chunks and add to your RAG system!")