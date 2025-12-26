"""
Evaluation Metrics System for RAG
Measures: Recall@K, MRR, Hit Rate, Latency
"""

import json
import time
from typing import List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm
import numpy as np
from main import MovieRAGSystem


class RAGEvaluator:
    """Evaluates RAG system performance with multiple metrics"""

    def __init__(self, eval_dataset_path: str = 'data/evaluation_dataset.json'):
        """Initialize evaluator with test questions"""
        print("📊 Initializing RAG Evaluator...")

        with open(eval_dataset_path, 'r', encoding='utf-8') as f:
            self.eval_dataset = json.load(f)

        print(f"✅ Loaded {len(self.eval_dataset)} evaluation questions")

        self.results = []

    def calculate_recall_at_k(self, retrieved_ids: List[int],
                              relevant_ids: List[int], k: int) -> float:
        """
        Calculate Recall@K

        Recall@K = (# relevant docs in top K) / (total # relevant docs)
        """
        top_k_ids = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)

        # How many relevant docs did we retrieve?
        retrieved_relevant = len(top_k_ids.intersection(relevant_set))

        # What's the maximum we could retrieve?
        total_relevant = len(relevant_set)

        if total_relevant == 0:
            return 0.0

        return retrieved_relevant / total_relevant

    def calculate_mrr(self, retrieved_ids: List[int],
                      relevant_ids: List[int]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)

        MRR = 1 / (position of first relevant result)
        If no relevant result found, MRR = 0
        """
        relevant_set = set(relevant_ids)

        for position, chunk_id in enumerate(retrieved_ids, start=1):
            if chunk_id in relevant_set:
                return 1.0 / position

        return 0.0  # No relevant result found

    def calculate_hit_at_k(self, retrieved_ids: List[int],
                           relevant_ids: List[int], k: int) -> float:
        """
        Calculate Hit@K (binary: did we get ANY relevant doc in top K?)

        Returns 1.0 if at least one relevant doc in top K, else 0.0
        """
        top_k_ids = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)

        return 1.0 if len(top_k_ids.intersection(relevant_set)) > 0 else 0.0

    def evaluate_single_question(self, rag: MovieRAGSystem,
                                 question: Dict, k: int = 5) -> Dict:
        """
        Evaluate a single question

        Returns metrics for this question
        """
        # Time the search
        start_time = time.time()
        results = rag.search(question['question'], top_k=k)
        latency = time.time() - start_time

        # Extract retrieved chunk IDs
        retrieved_ids = [r['chunk_id'] for r in results]
        relevant_ids = question['relevant_chunk_ids']

        # Calculate all metrics
        metrics = {
            'question_id': question['question_id'],
            'question': question['question'],
            'category': question['category'],
            'difficulty': question['difficulty'],
            'retrieved_chunk_ids': retrieved_ids,
            'relevant_chunk_ids': relevant_ids,
            'recall@1': self.calculate_recall_at_k(retrieved_ids, relevant_ids, 1),
            'recall@3': self.calculate_recall_at_k(retrieved_ids, relevant_ids, 3),
            'recall@5': self.calculate_recall_at_k(retrieved_ids, relevant_ids, 5),
            'mrr': self.calculate_mrr(retrieved_ids, relevant_ids),
            'hit@1': self.calculate_hit_at_k(retrieved_ids, relevant_ids, 1),
            'hit@3': self.calculate_hit_at_k(retrieved_ids, relevant_ids, 3),
            'hit@5': self.calculate_hit_at_k(retrieved_ids, relevant_ids, 5),
            'latency_ms': latency * 1000
        }

        return metrics

    def evaluate_all(self, rag: MovieRAGSystem, k: int = 5,
                     limit: int = None) -> Dict:
        """
        Evaluate all questions in dataset

        Args:
            rag: RAG system to evaluate
            k: Number of results to retrieve
            limit: Optional limit on number of questions (for testing)

        Returns:
            Dictionary with aggregated metrics
        """
        questions = self.eval_dataset[:limit] if limit else self.eval_dataset

        print(f"\n🔍 Evaluating {len(questions)} questions...")
        print(f"Retrieving top {k} results per question")
        print("=" * 80)

        self.results = []

        for question in tqdm(questions, desc="Evaluating"):
            result = self.evaluate_single_question(rag, question, k)
            self.results.append(result)

        # Aggregate metrics
        aggregated = self._aggregate_metrics()

        return aggregated

    def _aggregate_metrics(self) -> Dict:
        """Aggregate metrics across all questions"""

        if not self.results:
            return {}

        # Overall metrics
        overall = {
            'recall@1': np.mean([r['recall@1'] for r in self.results]),
            'recall@3': np.mean([r['recall@3'] for r in self.results]),
            'recall@5': np.mean([r['recall@5'] for r in self.results]),
            'mrr': np.mean([r['mrr'] for r in self.results]),
            'hit@1': np.mean([r['hit@1'] for r in self.results]),
            'hit@3': np.mean([r['hit@3'] for r in self.results]),
            'hit@5': np.mean([r['hit@5'] for r in self.results]),
            'avg_latency_ms': np.mean([r['latency_ms'] for r in self.results]),
            'total_questions': len(self.results)
        }

        # By category
        categories = {}
        for result in self.results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)

        by_category = {}
        for cat, results in categories.items():
            by_category[cat] = {
                'recall@1': np.mean([r['recall@1'] for r in results]),
                'recall@3': np.mean([r['recall@3'] for r in results]),
                'recall@5': np.mean([r['recall@5'] for r in results]),
                'mrr': np.mean([r['mrr'] for r in results]),
                'count': len(results)
            }

        # By difficulty
        difficulties = {}
        for result in self.results:
            diff = result['difficulty']
            if diff not in difficulties:
                difficulties[diff] = []
            difficulties[diff].append(result)

        by_difficulty = {}
        for diff, results in difficulties.items():
            by_difficulty[diff] = {
                'recall@1': np.mean([r['recall@1'] for r in results]),
                'recall@3': np.mean([r['recall@3'] for r in results]),
                'recall@5': np.mean([r['recall@5'] for r in results]),
                'mrr': np.mean([r['mrr'] for r in results]),
                'count': len(results)
            }

        return {
            'overall': overall,
            'by_category': by_category,
            'by_difficulty': by_difficulty
        }

    def print_results(self, aggregated: Dict):
        """Print evaluation results in a nice format"""

        print("\n" + "=" * 80)
        print("📊 EVALUATION RESULTS")
        print("=" * 80)

        # Overall metrics
        overall = aggregated['overall']
        print("\n📈 Overall Performance:")
        print(f"   Questions evaluated: {overall['total_questions']}")
        print(f"   Recall@1: {overall['recall@1']:.1%}")
        print(f"   Recall@3: {overall['recall@3']:.1%}")
        print(f"   Recall@5: {overall['recall@5']:.1%}")
        print(f"   MRR: {overall['mrr']:.3f}")
        print(f"   Hit@1: {overall['hit@1']:.1%}")
        print(f"   Hit@3: {overall['hit@3']:.1%}")
        print(f"   Hit@5: {overall['hit@5']:.1%}")
        print(f"   Avg Latency: {overall['avg_latency_ms']:.1f}ms")

        # By category
        print("\n📋 Performance by Category:")
        for cat, metrics in aggregated['by_category'].items():
            print(f"\n   {cat.upper()} ({metrics['count']} questions):")
            print(f"      Recall@3: {metrics['recall@3']:.1%}")
            print(f"      MRR: {metrics['mrr']:.3f}")

        # By difficulty
        print("\n🎯 Performance by Difficulty:")
        for diff, metrics in aggregated['by_difficulty'].items():
            print(f"\n   {diff.upper()} ({metrics['count']} questions):")
            print(f"      Recall@3: {metrics['recall@3']:.1%}")
            print(f"      MRR: {metrics['mrr']:.3f}")

        print("\n" + "=" * 80)

    def save_results(self, aggregated: Dict,
                     output_path: str = 'data/eval_results.json'):
        """Save detailed results to file"""

        results_data = {
            'aggregated_metrics': aggregated,
            'detailed_results': self.results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2)

        print(f"\n💾 Results saved to: {output_path}")

    def show_failures(self, top_n: int = 10):
        """Show questions where system performed poorly"""

        print("\n" + "=" * 80)
        print(f"❌ Top {top_n} Failures (Lowest MRR)")
        print("=" * 80)

        # Sort by MRR (ascending)
        sorted_results = sorted(self.results, key=lambda x: x['mrr'])

        for i, result in enumerate(sorted_results[:top_n], 1):
            print(f"\n{i}. {result['question']}")
            print(f"   Category: {result['category']} | Difficulty: {result['difficulty']}")
            print(f"   MRR: {result['mrr']:.3f} | Recall@3: {result['recall@3']:.1%}")
            print(f"   Expected {len(result['relevant_chunk_ids'])} chunk(s), got {len(result['retrieved_chunk_ids'])}")


def run_baseline_evaluation():
    """Run complete baseline evaluation"""

    print("🎬 RAG BASELINE EVALUATION")
    print("=" * 80)
    print("\nThis will measure your CURRENT system performance")
    print("We'll use these numbers as the baseline to improve against!\n")

    # Initialize systems
    print("Initializing RAG system...")
    rag = MovieRAGSystem()

    print("\nInitializing evaluator...")
    evaluator = RAGEvaluator()

    # Run evaluation
    aggregated = evaluator.evaluate_all(rag, k=5)

    # Print results
    evaluator.print_results(aggregated)

    # Show failures
    evaluator.show_failures(top_n=5)

    # Save results
    evaluator.save_results(aggregated, 'data/baseline_results.json')

    print("\n" + "=" * 80)
    print("✅ Baseline evaluation complete!")
    print("=" * 80)
    print("\n💡 Next steps:")
    print("   1. Review the metrics (especially failures)")
    print("   2. Implement improvements (hybrid search, reranking)")
    print("   3. Re-run evaluation to measure improvement")
    print("   4. Compare before/after metrics!")

    return aggregated, evaluator


if __name__ == "__main__":
    results, evaluator = run_baseline_evaluation()