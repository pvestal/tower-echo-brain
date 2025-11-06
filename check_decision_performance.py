#!/usr/bin/env python3
"""
Check Decision Engine Performance Summary
"""

import os
import sys
sys.path.append('/opt/tower-echo-brain')

from model_decision_engine import get_decision_engine

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "database": "tower_consolidated",
    "user": os.getenv("TOWER_USER", os.getenv("TOWER_USER", "patrick"))
}

def main():
    """Check current performance summary"""

    engine = get_decision_engine(DB_CONFIG)

    print("Echo Brain Model Decision Engine Performance Summary")
    print("=" * 60)

    # Get performance summary
    summary = engine.get_performance_summary()

    if summary:
        print("OVERALL PERFORMANCE (Last 24 Hours):")
        print("-" * 40)
        overall = summary.get("summary", {})
        print(f"Total Decisions: {overall.get('total_decisions', 0)}")
        print(f"Average Complexity: {overall.get('avg_complexity', 0):.2f}")
        print(f"Average Response Time: {overall.get('avg_response_time', 0):.2f}s")
        print(f"Average Feedback: {overall.get('avg_feedback', 0):.2f}/5.0")
        print(f"API Fallbacks: {overall.get('api_fallbacks', 0)}")

        print("\nPER-MODEL PERFORMANCE:")
        print("-" * 40)
        models = summary.get("models", {})

        if models:
            for model, stats in models.items():
                print(f"{model}:")
                print(f"  Queries: {stats.get('queries', 0)}")
                avg_time = stats.get('avg_time') or 0
                avg_feedback = stats.get('avg_feedback') or 0
                print(f"  Avg Time: {avg_time:.2f}s")
                print(f"  Avg Feedback: {avg_feedback:.2f}/5.0")
                print()
        else:
            print("No model performance data available")
    else:
        print("❌ Failed to retrieve performance summary")

    # Show current thresholds
    print("CURRENT OPTIMIZED THRESHOLDS:")
    print("-" * 40)
    for tier, thresholds in engine.thresholds.items():
        print(f"{tier.upper()}: {thresholds['min']:.1f} - {thresholds['max']:.1f}")

    print("\nCURRENT OPTIMIZED WEIGHTS:")
    print("-" * 40)
    for feature, weight in engine.complexity_analyzer.weights.items():
        print(f"{feature}: {weight}")

if __name__ == "__main__":
    main()