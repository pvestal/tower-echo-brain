#!/usr/bin/env python3
"""
Quick benchmark to optimize model assignments in unified router.
Tests each model on representative tasks to find best performers.
"""

import json
import time
import subprocess
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class BenchmarkResult:
    model: str
    task_type: str
    query: str
    response_time: float
    quality_score: float
    success: bool

class ModelBenchmark:
    def __init__(self):
        # Models to test
        self.models = [
            "qwen2.5:3b",
            "llama3.2:3b",
            "deepseek-coder-v2:16b",
            "deepseek-r1:8b",
            "mistral:7b",
            "llama3.1:8b"
        ]

        # Test queries by category
        self.test_queries = {
            "coding": [
                "Write a Python function to reverse a linked list",
                "Debug this code: for i in range(10) print(i)",
                "Implement binary search in JavaScript"
            ],
            "reasoning": [
                "Analyze the trade-offs between microservices and monolithic architecture",
                "Design a scalable authentication system",
                "How does TCP congestion control work?"
            ],
            "creative": [
                "Create a cyberpunk character backstory",
                "Describe an anime battle scene",
                "Generate a story about a goblin merchant"
            ],
            "fast": [
                "What is 2+2?",
                "Define REST API",
                "List 3 benefits of Docker"
            ]
        }

        self.results = []

    def query_model(self, model: str, query: str, max_tokens: int = 100) -> tuple:
        """Query a model and measure response time."""
        start = time.time()

        try:
            # Use ollama directly for speed
            cmd = [
                "ollama", "run", model,
                "--verbose=false",
                query
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                env={"OLLAMA_MAX_TOKENS": str(max_tokens)}
            )

            elapsed = time.time() - start

            if result.returncode == 0:
                response = result.stdout.strip()
                return response, elapsed, True
            else:
                return "", elapsed, False

        except subprocess.TimeoutExpired:
            return "", 30.0, False
        except Exception as e:
            print(f"Error querying {model}: {e}")
            return "", 0, False

    def score_response(self, response: str, task_type: str) -> float:
        """Simple quality scoring based on response characteristics."""
        if not response:
            return 0.0

        score = 0.5  # Base score

        # Length check
        if len(response) > 50:
            score += 0.2

        # Task-specific scoring
        if task_type == "coding":
            # Check for code indicators
            if any(indicator in response.lower() for indicator in ['def ', 'function', 'return', '{', '}']):
                score += 0.3
        elif task_type == "reasoning":
            # Check for analytical language
            if any(word in response.lower() for word in ['because', 'therefore', 'however', 'analysis']):
                score += 0.3
        elif task_type == "creative":
            # Check for descriptive language
            if len(response) > 100:
                score += 0.3
        else:  # fast
            # Prefer concise answers
            if len(response) < 200:
                score += 0.3

        return min(score, 1.0)

    def run_benchmark(self):
        """Run full benchmark suite."""
        print("🚀 Starting model benchmark...")
        print(f"Testing {len(self.models)} models on {sum(len(q) for q in self.test_queries.values())} queries\n")

        for task_type, queries in self.test_queries.items():
            print(f"\n📊 Testing {task_type.upper()} tasks...")

            for model in self.models:
                print(f"  Testing {model}...")

                for query in queries[:2]:  # Limit to 2 queries per category for speed
                    response, elapsed, success = self.query_model(model, query)

                    if success:
                        quality = self.score_response(response, task_type)

                        result = BenchmarkResult(
                            model=model,
                            task_type=task_type,
                            query=query[:50],
                            response_time=elapsed,
                            quality_score=quality,
                            success=True
                        )

                        self.results.append(result)
                        print(f"    ✓ {elapsed:.1f}s, quality: {quality:.2f}")
                    else:
                        print(f"    ✗ Failed")

    def analyze_results(self) -> Dict:
        """Analyze results and recommend best model per task type."""
        recommendations = {}

        for task_type in self.test_queries.keys():
            # Filter results for this task type
            task_results = [r for r in self.results if r.task_type == task_type and r.success]

            if not task_results:
                continue

            # Calculate average scores per model
            model_scores = {}
            for result in task_results:
                if result.model not in model_scores:
                    model_scores[result.model] = {"times": [], "qualities": []}

                model_scores[result.model]["times"].append(result.response_time)
                model_scores[result.model]["qualities"].append(result.quality_score)

            # Compute combined score (quality weighted more than speed)
            best_model = None
            best_score = -1

            for model, scores in model_scores.items():
                avg_time = sum(scores["times"]) / len(scores["times"])
                avg_quality = sum(scores["qualities"]) / len(scores["qualities"])

                # Combined score: 70% quality, 30% speed
                speed_score = max(0, 1 - (avg_time / 10))  # Normalize to 0-1
                combined = (0.7 * avg_quality) + (0.3 * speed_score)

                if combined > best_score:
                    best_score = combined
                    best_model = model

            recommendations[task_type] = {
                "model": best_model,
                "score": best_score,
                "alternatives": list(model_scores.keys())
            }

        return recommendations

    def generate_router_update(self, recommendations: Dict) -> str:
        """Generate Python code to update the unified router."""
        code = '''# Benchmark-optimized model assignments
self.rules = {
'''

        mappings = {
            "reasoning": "ModelTier.REASONING",
            "coding": "ModelTier.CODING",
            "creative": "ModelTier.CREATIVE",
            "fast": "ModelTier.FAST"
        }

        for task_type, rec in recommendations.items():
            if task_type in mappings:
                code += f'    {mappings[task_type]}: "{rec["model"]}",\n'

        code += '}'
        return code

def main():
    """Run benchmark and generate recommendations."""
    benchmark = ModelBenchmark()

    # Check which models are available
    print("🔍 Checking available models...")
    available = []
    for model in benchmark.models:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True
        )
        if model in result.stdout:
            available.append(model)
            print(f"  ✓ {model}")
        else:
            print(f"  ✗ {model} (not available)")

    benchmark.models = available

    if len(available) < 2:
        print("\n❌ Not enough models available for meaningful benchmark")
        return

    # Run benchmark
    benchmark.run_benchmark()

    # Analyze results
    print("\n📊 ANALYZING RESULTS...")
    recommendations = benchmark.analyze_results()

    # Print recommendations
    print("\n🎯 RECOMMENDATIONS:")
    print("-" * 50)
    for task_type, rec in recommendations.items():
        print(f"{task_type.upper():10} → {rec['model']:20} (score: {rec['score']:.2f})")

    # Generate router update
    print("\n📝 ROUTER UPDATE CODE:")
    print("-" * 50)
    update_code = benchmark.generate_router_update(recommendations)
    print(update_code)

    # Save results
    with open("/opt/tower-echo-brain/benchmarks/results.json", "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "recommendations": recommendations,
            "raw_results": [
                {
                    "model": r.model,
                    "task": r.task_type,
                    "time": r.response_time,
                    "quality": r.quality_score
                }
                for r in benchmark.results
            ]
        }, f, indent=2)

    print("\n✅ Results saved to benchmarks/results.json")

if __name__ == "__main__":
    main()