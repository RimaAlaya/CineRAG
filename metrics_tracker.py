# metrics_tracker.py
"""
Production metrics tracking
Tracks precision, recall, latency, costs, etc.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List
from feedback_system import FeedbackSystem
import sqlite3


class MetricsTracker:
    """Track and analyze system performance metrics"""

    def __init__(self):
        self.feedback = FeedbackSystem()

    def calculate_precision_at_k(self, k: int = 3) -> float:
        """
        Calculate precision based on user feedback
        Precision = relevant results / total results shown
        """
        conn = sqlite3.connect(self.feedback.db_path)
        cursor = conn.cursor()

        # Get queries with positive ratings
        cursor.execute('''
                       SELECT COUNT(*)
                       FROM queries
                       WHERE rating = 1
                       ''')
        relevant = cursor.fetchone()[0]

        # Get total rated queries
        cursor.execute('''
                       SELECT COUNT(*)
                       FROM queries
                       WHERE rating IS NOT NULL
                       ''')
        total = cursor.fetchone()[0]

        conn.close()

        return relevant / total if total > 0 else 0.0

    def get_latency_stats(self, days: int = 7) -> Dict:
        """Get latency statistics"""
        conn = sqlite3.connect(self.feedback.db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       SELECT AVG(latency_ms) as avg,
            MIN(latency_ms) as min,
            MAX(latency_ms) as max,
            COUNT(*) as count
                       FROM queries
                       WHERE timestamp >= datetime('now', '-' || ? || ' days')
                       ''', (days,))

        result = cursor.fetchone()
        conn.close()

        if result:
            return {
                'avg_ms': round(result[0], 2) if result[0] else 0,
                'min_ms': round(result[1], 2) if result[1] else 0,
                'max_ms': round(result[2], 2) if result[2] else 0,
                'p95_ms': self._get_percentile(95, days),
                'p99_ms': self._get_percentile(99, days),
                'total_queries': result[3]
            }
        return {}

    def _get_percentile(self, percentile: int, days: int) -> float:
        """Calculate latency percentile"""
        conn = sqlite3.connect(self.feedback.db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       SELECT latency_ms
                       FROM queries
                       WHERE timestamp >= datetime('now', '-' || ? || ' days')
                       ORDER BY latency_ms
                       ''', (days,))

        latencies = [row[0] for row in cursor.fetchall()]
        conn.close()

        if not latencies:
            return 0.0

        index = int(len(latencies) * percentile / 100)
        return round(latencies[min(index, len(latencies) - 1)], 2)

    def get_query_patterns(self, days: int = 7) -> Dict:
        """Analyze query patterns"""
        conn = sqlite3.connect(self.feedback.db_path)
        cursor = conn.cursor()

        # Most common query types
        cursor.execute('''
                       SELECT query, COUNT(*) as count
                       FROM queries
                       WHERE timestamp >= datetime('now', '-' || ? || ' days')
                       GROUP BY query
                       ORDER BY count DESC
                           LIMIT 10
                       ''', (days,))

        common_queries = [{'query': row[0], 'count': row[1]} for row in cursor.fetchall()]

        # Queries with secrets vs without
        cursor.execute('''
                       SELECT COUNT(CASE WHEN has_secrets = 1 THEN 1 END) as with_secrets,
                              COUNT(CASE WHEN has_secrets = 0 THEN 1 END) as without_secrets
                       FROM queries
                       WHERE timestamp >= datetime('now', '-' || ? || ' days')
                       ''', (days,))

        secrets_dist = cursor.fetchone()
        conn.close()

        return {
            'common_queries': common_queries,
            'secrets_distribution': {
                'with_secrets': secrets_dist[0],
                'without_secrets': secrets_dist[1],
                'secrets_pct': round(secrets_dist[0] / (secrets_dist[0] + secrets_dist[1]) * 100, 1)
                if (secrets_dist[0] + secrets_dist[1]) > 0 else 0
            }
        }

    def estimate_costs(self, days: int = 7) -> Dict:
        """Estimate API costs (Groq pricing)"""
        conn = sqlite3.connect(self.feedback.db_path)
        cursor = conn.cursor()

        # Count queries
        cursor.execute('''
                       SELECT COUNT(*)
                       FROM queries
                       WHERE timestamp >= datetime('now', '-' || ? || ' days')
                       ''', (days,))

        total_queries = cursor.fetchone()[0]
        conn.close()

        # Groq Mixtral pricing (approximate)
        # $0.27 per 1M input tokens, $0.27 per 1M output tokens
        # Estimate ~500 tokens per query (input + output)

        estimated_tokens = total_queries * 500
        estimated_cost = (estimated_tokens / 1_000_000) * 0.27

        return {
            'total_queries': total_queries,
            'estimated_tokens': estimated_tokens,
            'estimated_cost_usd': round(estimated_cost, 4),
            'cost_per_query': round(estimated_cost / total_queries, 6) if total_queries > 0 else 0
        }

    def get_failure_analysis(self) -> Dict:
        """Analyze failed queries (negative ratings)"""
        poor_queries = self.feedback.get_poor_queries(limit=20)

        # Categorize failures
        categories = {
            'off_topic': 0,
            'missing_info': 0,
            'wrong_movie': 0,
            'other': 0
        }

        for q in poor_queries:
            feedback = (q.get('feedback') or '').lower()
            if 'off-topic' in feedback or 'not relevant' in feedback:
                categories['off_topic'] += 1
            elif 'missing' in feedback or 'incomplete' in feedback:
                categories['missing_info'] += 1
            elif 'wrong' in feedback:
                categories['wrong_movie'] += 1
            else:
                categories['other'] += 1

        return {
            'total_failures': len(poor_queries),
            'failure_categories': categories,
            'recent_failures': poor_queries[:5]
        }

    def generate_report(self, days: int = 7) -> Dict:
        """Generate comprehensive metrics report"""
        stats = self.feedback.get_stats()
        latency = self.get_latency_stats(days)
        patterns = self.get_query_patterns(days)
        costs = self.estimate_costs(days)
        failures = self.get_failure_analysis()

        return {
            'report_date': datetime.now().isoformat(),
            'period_days': days,
            'overview': stats,
            'performance': {
                'precision': round(self.calculate_precision_at_k(), 3),
                'satisfaction_rate': round(stats['satisfaction_rate'], 3),
                'latency': latency
            },
            'usage': patterns,
            'costs': costs,
            'quality': failures
        }


if __name__ == "__main__":
    tracker = MetricsTracker()
    report = tracker.generate_report(days=30)

    print("📊 METRICS REPORT")
    print("=" * 80)
    print(json.dumps(report, indent=2))