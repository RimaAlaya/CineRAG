# feedback_system.py
"""
User feedback tracking system
Stores ratings and improves over time
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class FeedbackSystem:
    """Track user feedback and query performance"""

    def __init__(self, db_path: str = 'data/feedback.db'):
        """Initialize feedback database"""
        self.db_path = db_path
        Path('data').mkdir(exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Create tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Queries table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS queries
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           timestamp
                           TEXT
                           NOT
                           NULL,
                           query
                           TEXT
                           NOT
                           NULL,
                           answer
                           TEXT
                           NOT
                           NULL,
                           sources
                           TEXT
                           NOT
                           NULL,
                           has_secrets
                           BOOLEAN,
                           latency_ms
                           REAL,
                           rating
                           INTEGER
                           DEFAULT
                           NULL,
                           feedback_text
                           TEXT
                           DEFAULT
                           NULL,
                           model_version
                           TEXT,
                           prompt_version
                           TEXT
                       )
                       ''')

        # Metrics table (aggregated)
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS daily_metrics
                       (
                           date
                           TEXT
                           PRIMARY
                           KEY,
                           total_queries
                           INTEGER,
                           positive_ratings
                           INTEGER,
                           negative_ratings
                           INTEGER,
                           avg_latency_ms
                           REAL,
                           avg_sources_returned
                           REAL,
                           secret_queries_pct
                           REAL
                       )
                       ''')

        conn.commit()
        conn.close()

    def log_query(self, query: str, answer: str, sources: List[Dict],
                  has_secrets: bool, latency_ms: float,
                  model_version: str = 'mixtral-8x7b',
                  prompt_version: str = 'v1') -> int:
        """Log a query and its response"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       INSERT INTO queries (timestamp, query, answer, sources, has_secrets,
                                            latency_ms, model_version, prompt_version)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       ''', (
                           datetime.now().isoformat(),
                           query,
                           answer,
                           json.dumps(sources),
                           has_secrets,
                           latency_ms,
                           model_version,
                           prompt_version
                       ))

        query_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return query_id

    def save_feedback(self, query_id: int, rating: int, feedback_text: str = None):
        """Save user feedback for a query"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       UPDATE queries
                       SET rating        = ?,
                           feedback_text = ?
                       WHERE id = ?
                       ''', (rating, feedback_text, query_id))

        conn.commit()
        conn.close()

    def get_satisfaction_rate(self, days: int = 7) -> float:
        """Get user satisfaction rate for last N days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       SELECT COUNT(CASE WHEN rating = 1 THEN 1 END) as positive,
                              COUNT(*)                               as total
                       FROM queries
                       WHERE rating IS NOT NULL
                         AND timestamp >= datetime('now'
                           , '-' || ? || ' days')
                       ''', (days,))

        result = cursor.fetchone()
        conn.close()

        if result and result[1] > 0:
            return result[0] / result[1]
        return 0.0

    def get_poor_queries(self, limit: int = 10) -> List[Dict]:
        """Get queries with negative feedback for improvement"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       SELECT id, query, answer, feedback_text, timestamp
                       FROM queries
                       WHERE rating = 0
                       ORDER BY timestamp DESC
                           LIMIT ?
                       ''', (limit,))

        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'query': row[1],
                'answer': row[2],
                'feedback': row[3],
                'timestamp': row[4]
            })

        conn.close()
        return results

    def get_stats(self) -> Dict:
        """Get overall system statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total queries
        cursor.execute('SELECT COUNT(*) FROM queries')
        total_queries = cursor.fetchone()[0]

        # Ratings distribution
        cursor.execute('''
                       SELECT COUNT(CASE WHEN rating = 1 THEN 1 END)     as positive,
                              COUNT(CASE WHEN rating = 0 THEN 1 END)     as negative,
                              COUNT(CASE WHEN rating IS NULL THEN 1 END) as no_rating
                       FROM queries
                       ''')
        ratings = cursor.fetchone()

        # Average latency
        cursor.execute('SELECT AVG(latency_ms) FROM queries')
        avg_latency = cursor.fetchone()[0] or 0

        # Queries with secrets
        cursor.execute('SELECT COUNT(*) FROM queries WHERE has_secrets = 1')
        secret_queries = cursor.fetchone()[0]

        conn.close()

        return {
            'total_queries': total_queries,
            'positive_ratings': ratings[0],
            'negative_ratings': ratings[1],
            'no_rating': ratings[2],
            'satisfaction_rate': ratings[0] / (ratings[0] + ratings[1]) if (ratings[0] + ratings[1]) > 0 else 0,
            'avg_latency_ms': round(avg_latency, 2),
            'secret_queries': secret_queries,
            'secret_queries_pct': round(secret_queries / total_queries * 100, 1) if total_queries > 0 else 0
        }


# Test it
if __name__ == "__main__":
    fb = FeedbackSystem()

    # Test logging
    query_id = fb.log_query(
        query="Who directed Inception?",
        answer="Christopher Nolan directed Inception (2010).",
        sources=[{'movie': 'Inception', 'type': 'crew'}],
        has_secrets=False,
        latency_ms=234.5
    )

    print(f"Logged query ID: {query_id}")

    # Test feedback
    fb.save_feedback(query_id, rating=1, feedback_text="Perfect answer!")

    # Test stats
    stats = fb.get_stats()
    print("\nStats:")
    print(json.dumps(stats, indent=2))