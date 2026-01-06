#!/usr/bin/env python3
"""
Direct Model Decision Engine Testing
Bypasses conversation manager to test decision logic
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.append('/opt/tower-echo-brain')

from src.misc.model_decision_engine import get_decision_engine
import json

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "database": "echo_brain",
    "user": os.getenv("TOWER_USER", os.getenv("TOWER_USER", "patrick"))
}

async def test_individual_queries():
    """Test individual queries with different complexities"""

    # Get decision engine
    engine = get_decision_engine(DB_CONFIG)

    # Test cases with expected outcomes
    test_cases = [
        {
            "query": "What is 2+2?",
            "context": {},
            "expected_tier": "tiny",
            "description": "Simple math"
        },
        {
            "query": "Write a hello world function in Python",
            "context": {},
            "expected_tier": "small",
            "description": "Basic coding"
        },
        {
            "query": "Debug this async function that's causing memory leaks in my web service",
            "context": {"code_lines": 100},
            "expected_tier": "medium",
            "description": "Debugging task"
        },
        {
            "query": "Implement a REST API endpoint with JWT authentication and rate limiting",
            "context": {},
            "expected_tier": "medium",
            "description": "API implementation"
        },
        {
            "query": "Refactor this microservice architecture to improve scalability and implement distributed caching",
            "context": {"files": 20, "microservices": 8},
            "expected_tier": "large",
            "description": "Architecture refactoring"
        },
        {
            "query": "Design a distributed system with eventual consistency, implement CQRS pattern, and optimize for high throughput",
            "context": {},
            "expected_tier": "large",
            "description": "Complex system design"
        },
        {
            "query": "Analyze this entire codebase (100+ files) and suggest a complete restructuring with modern design patterns, implement CI/CD pipeline, and migrate to cloud-native architecture",
            "context": {"files": 100, "complexity": "extreme"},
            "expected_tier": "cloud",
            "description": "Massive refactoring"
        }
    ]

    print("=" * 80)
    print("DIRECT MODEL DECISION ENGINE TESTING")
    print("=" * 80)

    results = []

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['description']}")
        print(f"Query: {test['query'][:60]}...")

        # Make decision
        decision = await engine.decide_model(test['query'], test['context'])

        # Extract tier
        tier = decision['tier'].value if hasattr(decision['tier'], 'value') else decision['tier']

        # Check result
        if tier == test['expected_tier']:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"

        print(f"Expected: {test['expected_tier']} | Actual: {tier} | {status}")
        print(f"Complexity Score: {decision['complexity_score']:.1f}")
        print(f"Selected Model: {decision['model']}")
        print(f"Reason: {decision['reason']}")

        # Print feature breakdown
        features = decision['features']
        print("Feature Analysis:")
        for feature, value in features.items():
            if value > 0:
                print(f"  - {feature}: {value}")

        results.append({
            'test': test['description'],
            'expected': test['expected_tier'],
            'actual': tier,
            'complexity': decision['complexity_score'],
            'model': decision['model'],
            'passed': tier == test['expected_tier']
        })

    # Summary
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    score = (passed / total) * 100

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Tests Passed: {passed}/{total}")
    print(f"Score: {score:.1f}%")

    if score < 80:
        print("\n⚠️  PERFORMANCE ISSUES IDENTIFIED:")
        for r in results:
            if not r['passed']:
                print(f"  - {r['test']}: Expected {r['expected']}, got {r['actual']}")

    return results

async def analyze_thresholds():
    """Analyze current complexity thresholds"""

    engine = get_decision_engine(DB_CONFIG)

    print("\n" + "=" * 80)
    print("CURRENT COMPLEXITY THRESHOLDS")
    print("=" * 80)

    for tier, thresholds in engine.thresholds.items():
        print(f"{tier.upper()}: {thresholds['min']:.1f} - {thresholds['max']:.1f}")

    print("\nCURRENT FEATURE WEIGHTS:")
    print("-" * 40)
    for feature, weight in engine.complexity_analyzer.weights.items():
        print(f"{feature}: {weight}")

async def main():
    """Main test runner"""

    print("Starting Direct Model Decision Engine Testing...")

    # Run tests
    results = await test_individual_queries()

    # Analyze current settings
    await analyze_thresholds()

    # Calculate and suggest improvements
    failed_tests = [r for r in results if not r['passed']]

    if failed_tests:
        print("\n" + "=" * 80)
        print("OPTIMIZATION RECOMMENDATIONS")
        print("=" * 80)

        for test in failed_tests:
            expected_tier = test['expected']
            actual_tier = test['actual']
            complexity = test['complexity']

            print(f"\nTest: {test['test']}")
            print(f"Current complexity score: {complexity:.1f}")
            print(f"Got {actual_tier}, expected {expected_tier}")

            # Suggest threshold adjustments
            tier_order = ["tiny", "small", "medium", "large", "cloud"]
            expected_idx = tier_order.index(expected_tier)
            actual_idx = tier_order.index(actual_tier)

            if actual_idx < expected_idx:
                # Model too weak, need to increase thresholds or feature weights
                print(f"📈 Need to increase complexity scoring (current: {complexity:.1f})")
                print(f"   Suggestion: Increase feature weights or lower {expected_tier} threshold")
            else:
                # Model too strong, need to decrease thresholds
                print(f"📉 Need to decrease complexity scoring (current: {complexity:.1f})")
                print(f"   Suggestion: Decrease feature weights or raise {expected_tier} threshold")

if __name__ == "__main__":
    asyncio.run(main())