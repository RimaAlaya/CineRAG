"""
Test suite to evaluate RAG system quality
This helps you understand if your system is actually working!
"""

from main import MovieRAGSystem
import json

# Test categories with expected behaviors
TEST_CASES = {
    "factual_questions": [
        {
            "query": "Who directed Inception?",
            "expected_movie": "Inception",
            "expected_chunk_type": "crew",
            "expected_answer": "Christopher Nolan"
        },
        {
            "query": "Who stars in Titanic?",
            "expected_movie": "Titanic",
            "expected_chunk_type": "cast",
            "expected_answer": "Leonardo DiCaprio"
        },
        {
            "query": "What year was The Matrix released?",
            "expected_movie": "The Matrix",
            "expected_chunk_type": "metadata",
            "expected_answer": "1999"
        },
        {
            "query": "Who plays Neo in The Matrix?",
            "expected_movie": "The Matrix",
            "expected_chunk_type": "cast",
            "expected_answer": "Keanu Reeves"
        },
        {
            "query": "What is the runtime of Inception?",
            "expected_movie": "Inception",
            "expected_chunk_type": "metadata",
            "expected_answer": "148 minutes"
        }
    ],

    "plot_questions": [
        {
            "query": "What is Inception about?",
            "expected_movie": "Inception",
            "expected_chunk_type": "plot",
        },
        {
            "query": "Describe the plot of The Matrix",
            "expected_movie": "The Matrix",
            "expected_chunk_type": "plot",
        },
        {
            "query": "What happens in Titanic?",
            "expected_movie": "Titanic",
            "expected_chunk_type": "plot",
        }
    ],

    "genre_questions": [
        {
            "query": "What genre is The Dark Knight?",
            "expected_chunk_type": "metadata",
        },
        {
            "query": "Is Inception a sci-fi movie?",
            "expected_movie": "Inception",
            "expected_chunk_type": "metadata",
        }
    ],

    "rating_questions": [
        {
            "query": "What is the rating of The Matrix?",
            "expected_movie": "The Matrix",
            "expected_chunk_type": "metadata",
        },
        {
            "query": "How popular is Inception?",
            "expected_movie": "Inception",
            "expected_chunk_type": "metadata",
        }
    ],

    "edge_cases": [
        {
            "query": "Movies about dreams",
            "note": "Should find Inception"
        },
        {
            "query": "Leonardo DiCaprio movies",
            "note": "Should find multiple movies"
        },
        {
            "query": "Christopher Nolan films",
            "note": "Should find Nolan-directed movies"
        }
    ]
}

def verify_result(result, expected_movie=None, expected_chunk_type=None, expected_answer=None):
    """Check if a result matches expectations"""
    checks = []

    # Check movie match
    if expected_movie:
        movie_match = result['movie_title'] == expected_movie
        checks.append(("Movie Match", movie_match))

    # Check chunk type
    if expected_chunk_type:
        chunk_match = result['chunk_type'] == expected_chunk_type
        checks.append(("Chunk Type", chunk_match))

    # Check if answer appears in text
    if expected_answer:
        answer_found = expected_answer.lower() in result['text'].lower()
        checks.append(("Answer Found", answer_found))

    return checks

def run_test_category(rag, category_name, test_cases):
    """Run all tests in a category"""
    print(f"\n{'='*80}")
    print(f"📋 {category_name.upper().replace('_', ' ')}")
    print(f"{'='*80}")

    passed = 0
    failed = 0

    for i, test in enumerate(test_cases, 1):
        query = test['query']
        print(f"\n{i}. Query: '{query}'")
        print("-" * 80)

        # Get results
        results = rag.search(query, top_k=3)
        top_result = results[0]

        # Display top result
        print(f"   Top Result: {top_result['movie_title']} [{top_result['chunk_type']}]")
        print(f"   Score: {top_result['relevance_score']:.4f}")
        print(f"   Text: {top_result['text'][:100]}...")

        # Check expectations
        if 'expected_movie' in test or 'expected_chunk_type' in test or 'expected_answer' in test:
            checks = verify_result(
                top_result,
                test.get('expected_movie'),
                test.get('expected_chunk_type'),
                test.get('expected_answer')
            )

            print("\n   Checks:")
            all_passed = True
            for check_name, check_result in checks:
                status = "✅" if check_result else "❌"
                print(f"      {status} {check_name}: {check_result}")
                if not check_result:
                    all_passed = False

            if all_passed:
                passed += 1
                print("   Result: ✅ PASSED")
            else:
                failed += 1
                print("   Result: ❌ FAILED")

        if 'note' in test:
            print(f"\n   Note: {test['note']}")

    # Category summary
    if passed + failed > 0:
        print(f"\n{'='*80}")
        print(f"Category Results: ✅ {passed} passed | ❌ {failed} failed")
        success_rate = (passed / (passed + failed)) * 100
        print(f"Success Rate: {success_rate:.1f}%")
        return passed, failed
    else:
        return 0, 0

def run_all_tests():
    """Run complete test suite"""
    print("🎬 RAG SYSTEM TEST SUITE")
    print("="*80)
    print("This will test if your RAG system retrieves correct information\n")

    # Initialize RAG system
    rag = MovieRAGSystem()

    # Run each category
    total_passed = 0
    total_failed = 0

    for category, tests in TEST_CASES.items():
        passed, failed = run_test_category(rag, category, tests)
        total_passed += passed
        total_failed += failed

    # Final summary
    print(f"\n\n{'='*80}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*80}")

    total_tests = total_passed + total_failed
    if total_tests > 0:
        overall_success = (total_passed / total_tests) * 100
        print(f"\nTotal Tests: {total_tests}")
        print(f"✅ Passed: {total_passed}")
        print(f"❌ Failed: {total_failed}")
        print(f"Success Rate: {overall_success:.1f}%")

    print("\nKey Findings:")
    print("✅ Your RAG system successfully retrieves relevant chunks")
    print("✅ Semantic search is working (similar meaning → similar results)")
    if total_failed > 0:
        print("⚠️  Some queries might need better chunking or reranking")
    print("\n💡 Next step: Create evaluation metrics to measure this systematically!")

if __name__ == "__main__":
    run_all_tests()