"""
Evaluate Hybrid Search vs Baseline
Compare performance improvement
"""

import json
from evaluate_baseline import RAGEvaluator
from hybrid_rag import HybridMovieRAG


def compare_baseline_vs_hybrid():
    """Compare baseline semantic search vs hybrid search"""

    print("🎬 BASELINE vs HYBRID COMPARISON")
    print("=" * 80)
    print("\nEvaluating both approaches on the same test set...")
    print("This will take 3-4 minutes for 164 questions × 2 systems\n")

    # Initialize systems
    print("1️⃣ Initializing Hybrid RAG System...")
    hybrid_rag = HybridMovieRAG()

    # Initialize evaluator
    print("\n2️⃣ Loading evaluation dataset...")
    evaluator = RAGEvaluator()

    # Evaluate hybrid
    print("\n3️⃣ Evaluating Hybrid Search (BM25 + Semantic)...")
    print("-" * 80)
    hybrid_results = evaluator.evaluate_all(hybrid_rag, k=5)

    # Load baseline results
    print("\n4️⃣ Loading baseline results...")
    with open('data/baseline_results.json', 'r') as f:
        baseline_data = json.load(f)
        baseline_results = baseline_data['aggregated_metrics']

    # Print comparison
    print("\n" + "=" * 80)
    print("📊 PERFORMANCE COMPARISON")
    print("=" * 80)

    print("\n📈 Overall Metrics:")
    print(f"{'Metric':<20} {'Baseline':<15} {'Hybrid':<15} {'Change':<15}")
    print("-" * 65)

    baseline_overall = baseline_results['overall']
    hybrid_overall = hybrid_results['overall']

    metrics = ['recall@1', 'recall@3', 'recall@5', 'mrr', 'hit@3']

    for metric in metrics:
        baseline_val = baseline_overall[metric]
        hybrid_val = hybrid_overall[metric]
        change = hybrid_val - baseline_val

        # Format change with color indicator
        change_str = f"+{change:.1%}" if change > 0 else f"{change:.1%}"
        indicator = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"

        print(f"{metric:<20} {baseline_val:<14.1%} {hybrid_val:<14.1%} {indicator} {change_str}")

    # By category comparison
    print("\n📋 Performance by Category:")
    print(f"{'Category':<20} {'Baseline R@3':<15} {'Hybrid R@3':<15} {'Change':<15}")
    print("-" * 65)

    for category in baseline_results['by_category'].keys():
        baseline_cat = baseline_results['by_category'][category]['recall@3']
        hybrid_cat = hybrid_results['by_category'][category]['recall@3']
        change = hybrid_cat - baseline_cat

        change_str = f"+{change:.1%}" if change > 0 else f"{change:.1%}"
        indicator = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"

        print(f"{category:<20} {baseline_cat:<14.1%} {hybrid_cat:<14.1%} {indicator} {change_str}")

    # Key insights
    print("\n" + "=" * 80)
    print("💡 KEY INSIGHTS")
    print("=" * 80)

    overall_improvement = hybrid_overall['recall@3'] - baseline_overall['recall@3']

    if overall_improvement > 0.05:  # 5% improvement
        print(f"\n✅ SIGNIFICANT IMPROVEMENT: +{overall_improvement:.1%} in Recall@3")
        print("   Hybrid search is working! BM25 helps with keyword matching.")
    elif overall_improvement > 0:
        print(f"\n🟡 MODEST IMPROVEMENT: +{overall_improvement:.1%} in Recall@3")
        print("   Hybrid search helps slightly, but major issues remain.")
    else:
        print(f"\n❌ NO IMPROVEMENT: {overall_improvement:.1%} in Recall@3")
        print("   Hybrid search didn't help. Need different approach.")

    # Identify best improvements
    print("\n🎯 Best improvements:")
    for category in baseline_results['by_category'].keys():
        baseline_cat = baseline_results['by_category'][category]['recall@3']
        hybrid_cat = hybrid_results['by_category'][category]['recall@3']
        change = hybrid_cat - baseline_cat

        if change > 0.05:  # 5% improvement
            print(f"   ✅ {category}: +{change:.1%}")

    # Save hybrid results
    print("\n💾 Saving hybrid results...")
    evaluator.save_results(hybrid_results, 'data/hybrid_results.json')

    # Create comparison report
    comparison = {
        'baseline': baseline_results,
        'hybrid': hybrid_results,
        'improvements': {
            'overall_recall@3': overall_improvement,
            'overall_mrr': hybrid_overall['mrr'] - baseline_overall['mrr']
        }
    }

    with open('data/comparison_report.json', 'w') as f:
        json.dump(comparison, f, indent=2)

    print("✅ Saved to: data/comparison_report.json")

    print("\n" + "=" * 80)
    print("🎉 EVALUATION COMPLETE!")
    print("=" * 80)

    return baseline_results, hybrid_results, comparison


if __name__ == "__main__":
    baseline, hybrid, comparison = compare_baseline_vs_hybrid()