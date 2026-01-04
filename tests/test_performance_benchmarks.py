#!/usr/bin/env python3
"""
Performance Benchmark Testing for Model Decision Engine
Tests response times and throughput for each model tier
"""

import asyncio
import aiohttp
import time
import json
import statistics
from typing import Dict, List

class PerformanceTester:
    """Test performance benchmarks for each model tier"""

    def __init__(self):
        self.ollama_base = "http://localhost:11434/api"
        self.benchmark_queries = {
            "tinyllama:latest": [
                "What is 2+2?",
                "Define recursion",
                "Hello world in Python"
            ],
            "deepseek-coder:latest": [
                "Write a Python function to sort a list",
                "Create a simple class in Python",
                "Implement binary search"
            ],
            "mistral:7b": [
                "Explain async/await in Python",
                "What are design patterns?",
                "Describe REST API principles"
            ],
            "codellama:13b": [
                "Implement a REST API with authentication",
                "Debug a memory leak in Python",
                "Write unit tests for a service"
            ],
            "qwen2.5-coder:32b": [
                "Design a microservice architecture",
                "Implement distributed caching",
                "Create a CI/CD pipeline"
            ]
        }

    async def benchmark_model(self, model: str, queries: List[str]) -> Dict:
        """Benchmark a specific model with given queries"""

        results = []

        print(f"\n🔍 Testing {model}...")

        async with aiohttp.ClientSession() as session:
            for i, query in enumerate(queries, 1):
                try:
                    start_time = time.time()

                    payload = {
                        "model": model,
                        "prompt": query,
                        "stream": False,
                        "options": {
                            "num_predict": 150,  # Consistent token count for fair comparison
                            "temperature": 0.7
                        }
                    }

                    print(f"  Query {i}/3: {query[:40]}...")

                    async with session.post(
                        f"{self.ollama_base}/generate",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:

                        if response.status == 200:
                            result = await response.json()
                            elapsed = time.time() - start_time

                            response_text = result.get("response", "")
                            tokens = len(response_text.split())
                            tokens_per_second = tokens / elapsed if elapsed > 0 else 0

                            results.append({
                                "query": query,
                                "response_time": elapsed,
                                "tokens": tokens,
                                "tokens_per_second": tokens_per_second,
                                "status": "success"
                            })

                            print(f"    ✅ {elapsed:.2f}s ({tokens_per_second:.1f} tok/s)")

                        else:
                            results.append({
                                "query": query,
                                "status": "failed",
                                "error": f"HTTP {response.status}"
                            })
                            print(f"    ❌ HTTP {response.status}")

                except asyncio.TimeoutError:
                    results.append({
                        "query": query,
                        "status": "timeout",
                        "error": "Request timeout"
                    })
                    print(f"    ⏱️ Timeout")

                except Exception as e:
                    results.append({
                        "query": query,
                        "status": "error",
                        "error": str(e)
                    })
                    print(f"    ❌ Error: {e}")

        # Calculate statistics
        successful_results = [r for r in results if r["status"] == "success"]

        if successful_results:
            response_times = [r["response_time"] for r in successful_results]
            tokens_per_sec = [r["tokens_per_second"] for r in successful_results]

            stats = {
                "model": model,
                "successful_queries": len(successful_results),
                "total_queries": len(queries),
                "success_rate": len(successful_results) / len(queries) * 100,
                "avg_response_time": statistics.mean(response_times),
                "median_response_time": statistics.median(response_times),
                "avg_tokens_per_second": statistics.mean(tokens_per_sec),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "results": results
            }
        else:
            stats = {
                "model": model,
                "successful_queries": 0,
                "total_queries": len(queries),
                "success_rate": 0,
                "error": "No successful queries",
                "results": results
            }

        return stats

    async def run_all_benchmarks(self) -> Dict:
        """Run benchmarks for all available models"""

        print("🚀 Starting Performance Benchmark Testing")
        print("=" * 60)

        all_results = {}

        for model, queries in self.benchmark_queries.items():
            result = await self.benchmark_model(model, queries)
            all_results[model] = result

        return all_results

    def analyze_performance(self, results: Dict) -> Dict:
        """Analyze performance results and provide recommendations"""

        # Performance thresholds (seconds)
        thresholds = {
            "tinyllama:latest": 3.0,
            "deepseek-coder:latest": 5.0,
            "mistral:7b": 8.0,
            "codellama:13b": 15.0,
            "qwen2.5-coder:32b": 30.0
        }

        analysis = {
            "performance_issues": [],
            "recommendations": [],
            "tier_performance": {},
            "overall_health": "healthy"
        }

        for model, result in results.items():
            if result.get("successful_queries", 0) == 0:
                analysis["performance_issues"].append(f"{model}: No successful queries")
                analysis["overall_health"] = "degraded"
                continue

            avg_time = result.get("avg_response_time", 0)
            threshold = thresholds.get(model, 10.0)

            tier_info = {
                "avg_response_time": avg_time,
                "threshold": threshold,
                "success_rate": result.get("success_rate", 0),
                "status": "healthy"
            }

            if avg_time > threshold:
                tier_info["status"] = "slow"
                analysis["performance_issues"].append(
                    f"{model}: {avg_time:.1f}s > {threshold}s threshold"
                )
                analysis["overall_health"] = "degraded"

            if result.get("success_rate", 0) < 100:
                tier_info["status"] = "unreliable"
                analysis["performance_issues"].append(
                    f"{model}: {result['success_rate']:.1f}% success rate"
                )
                analysis["overall_health"] = "degraded"

            analysis["tier_performance"][model] = tier_info

        # Generate recommendations
        if analysis["performance_issues"]:
            analysis["recommendations"].extend([
                "Consider adjusting model selection thresholds for slow models",
                "Implement model warming strategies for better response times",
                "Add automatic escalation for unreliable models",
                "Monitor system resources during peak usage"
            ])

        return analysis

    def generate_report(self, results: Dict, analysis: Dict) -> str:
        """Generate comprehensive performance report"""

        report = []
        report.append("=" * 80)
        report.append("MODEL PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 80)

        # Summary
        total_models = len(results)
        healthy_models = sum(1 for r in results.values() if r.get("success_rate", 0) >= 100)

        report.append(f"Overall Health: {analysis['overall_health'].upper()}")
        report.append(f"Models Tested: {total_models}")
        report.append(f"Healthy Models: {healthy_models}/{total_models}")
        report.append("")

        # Per-model results
        report.append("PER-MODEL PERFORMANCE:")
        report.append("-" * 60)

        for model, result in results.items():
            if result.get("successful_queries", 0) > 0:
                avg_time = result["avg_response_time"]
                tokens_per_sec = result["avg_tokens_per_second"]
                success_rate = result["success_rate"]

                status_icon = "✅" if success_rate >= 100 and model in analysis["tier_performance"] and analysis["tier_performance"][model]["status"] == "healthy" else "⚠️"

                report.append(f"{status_icon} {model}")
                report.append(f"   Response Time: {avg_time:.2f}s (range: {result['min_response_time']:.2f}-{result['max_response_time']:.2f}s)")
                report.append(f"   Throughput: {tokens_per_sec:.1f} tokens/second")
                report.append(f"   Success Rate: {success_rate:.1f}%")
            else:
                report.append(f"❌ {model}")
                report.append(f"   Status: Failed - {result.get('error', 'Unknown error')}")

            report.append("")

        # Issues and recommendations
        if analysis["performance_issues"]:
            report.append("⚠️  PERFORMANCE ISSUES:")
            for issue in analysis["performance_issues"]:
                report.append(f"  - {issue}")
            report.append("")

        if analysis["recommendations"]:
            report.append("💡 RECOMMENDATIONS:")
            for rec in analysis["recommendations"]:
                report.append(f"  - {rec}")
            report.append("")

        report.append("=" * 80)

        return "\n".join(report)

async def main():
    """Main benchmark testing function"""

    tester = PerformanceTester()

    # Run benchmarks
    results = await tester.run_all_benchmarks()

    # Analyze results
    analysis = tester.analyze_performance(results)

    # Generate and display report
    report = tester.generate_report(results, analysis)
    print("\n" + report)

    # Save results
    with open('/opt/tower-echo-brain/benchmark_results.json', 'w') as f:
        json.dump({
            "results": results,
            "analysis": analysis,
            "timestamp": time.time()
        }, f, indent=2)

    print(f"📊 Results saved to: /opt/tower-echo-brain/benchmark_results.json")

    # Return appropriate exit code
    return 0 if analysis["overall_health"] == "healthy" else 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))