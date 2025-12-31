# ab_testing.py
"""
A/B Testing Framework
Test different prompts/models and see what works better
"""

import random
import json
from datetime import datetime
from typing import Dict, List
import sqlite3


class ABTest:
    """Run A/B tests on prompts/models"""

    def __init__(self, db_path='data/feedback.db'):
        self.db_path = db_path
        self._init_experiments()

    def _init_experiments(self):
        """Create experiments table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS experiments
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           name
                           TEXT
                           NOT
                           NULL,
                           variant_a
                           TEXT
                           NOT
                           NULL,
                           variant_b
                           TEXT
                           NOT
                           NULL,
                           status
                           TEXT
                           DEFAULT
                           'active',
                           created_at
                           TEXT
                           NOT
                           NULL
                       )
                       ''')

        # Add variant column to queries if not exists
        cursor.execute('''
        PRAGMA table_info(queries)
        ''')

        columns = [col[1] for col in cursor.fetchall()]
        if 'experiment_variant' not in columns:
            cursor.execute('''
                           ALTER TABLE queries
                               ADD COLUMN experiment_variant TEXT
                           ''')

        if 'experiment_name' not in columns:
            cursor.execute('''
                           ALTER TABLE queries
                               ADD COLUMN experiment_name TEXT
                           ''')

        conn.commit()
        conn.close()

    def create_experiment(self, name: str, variant_a: str, variant_b: str):
        """Create new A/B test"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       INSERT INTO experiments (name, variant_a, variant_b, created_at)
                       VALUES (?, ?, ?, ?)
                       ''', (name, variant_a, variant_b, datetime.now().isoformat()))

        conn.commit()
        conn.close()

    def assign_variant(self, experiment_name: str) -> str:
        """Randomly assign user to variant A or B (50/50)"""
        return random.choice(['A', 'B'])

    def get_results(self, experiment_name: str) -> Dict:
        """Get A/B test results with statistical analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get experiment details
        cursor.execute('''
                       SELECT variant_a, variant_b
                       FROM experiments
                       WHERE name = ?
                       ''', (experiment_name,))

        exp = cursor.fetchone()
        if not exp:
            return None

        variant_a_name, variant_b_name = exp

        # Get results for variant A
        cursor.execute('''
                       SELECT COUNT(*)                               as total,
                              COUNT(CASE WHEN rating = 1 THEN 1 END) as positive,
                              AVG(latency_ms)                        as avg_latency
                       FROM queries
                       WHERE experiment_name = ?
                         AND experiment_variant = 'A'
                       ''', (experiment_name,))

        a_stats = cursor.fetchone()

        # Get results for variant B
        cursor.execute('''
                       SELECT COUNT(*)                               as total,
                              COUNT(CASE WHEN rating = 1 THEN 1 END) as positive,
                              AVG(latency_ms)                        as avg_latency
                       FROM queries
                       WHERE experiment_name = ?
                         AND experiment_variant = 'B'
                       ''', (experiment_name,))

        b_stats = cursor.fetchone()

        conn.close()

        # Calculate conversion rates
        a_rate = a_stats[1] / a_stats[0] if a_stats[0] > 0 else 0
        b_rate = b_stats[1] / b_stats[0] if b_stats[0] > 0 else 0

        # Simple statistical significance check (z-test approximation)
        lift = ((b_rate - a_rate) / a_rate * 100) if a_rate > 0 else 0

        # Determine confidence (simplified)
        total_samples = a_stats[0] + b_stats[0]
        if total_samples < 30:
            confidence = "insufficient_data"
        elif abs(b_rate - a_rate) > 0.1:  # 10% difference
            confidence = "high"
        elif abs(b_rate - a_rate) > 0.05:  # 5% difference
            confidence = "medium"
        else:
            confidence = "low"

        return {
            'experiment_name': experiment_name,
            'variant_a': {
                'name': variant_a_name,
                'total_queries': a_stats[0],
                'positive_ratings': a_stats[1],
                'satisfaction_rate': a_rate,
                'avg_latency_ms': round(a_stats[2], 2) if a_stats[2] else 0
            },
            'variant_b': {
                'name': variant_b_name,
                'total_queries': b_stats[0],
                'positive_ratings': b_stats[1],
                'satisfaction_rate': b_rate,
                'avg_latency_ms': round(b_stats[2], 2) if b_stats[2] else 0
            },
            'analysis': {
                'lift_percent': round(lift, 2),
                'winner': 'B' if b_rate > a_rate else 'A' if a_rate > b_rate else 'tie',
                'confidence': confidence,
                'sample_size': total_samples
            }
        }


# Pre-defined prompt variants
PROMPT_VARIANTS = {
    'gossip_girl': """You're spilling cinema tea like a gossip columnist. Be DRAMATIC, use caps for emphasis, and make it juicy.

Context: {context}
Question: {question}

Spill the tea (under 100 words):""",

    'documentary': """You're a documentary narrator. Be factual, clear, and authoritative.

Context: {context}
Question: {question}

Narrate the answer (under 100 words):""",

    'reddit': """You're a Reddit commenter sharing movie trivia. Be casual, use "lmao", "tbh", and sound like a real person.

Context: {context}
Question: {question}

Drop some knowledge (under 100 words):""",

    'professor': """You're a film professor teaching a class. Be educational but engaging.

Context: {context}
Question: {question}

Teach us (under 100 words):""",
}

if __name__ == "__main__":
    # Demo
    ab = ABTest()

    # Create experiment
    ab.create_experiment(
        name="prompt_style_test",
        variant_a="gossip_girl",
        variant_b="documentary"
    )

    print("✅ Experiment created!")
    print("Now run queries and see which variant users prefer...")