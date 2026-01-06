#!/usr/bin/env python3
"""
Quick Performance Test for Model Decision Engine
Fast validation of model response times
"""

import asyncio
import aiohttp
import time

async def test_model_performance(model: str, query: str) -> dict:
    """Test a single model with a simple query"""

    try:
        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            payload = {
                "model": model,
                "prompt": query,
                "stream": False,
                "options": {"num_predict": 50}  # Short response
            }

            async with session.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=20)
            ) as response:

                if response.status == 200:
                    result = await response.json()
                    elapsed = time.time() - start_time
                    tokens = len(result.get("response", "").split())

                    return {
                        "model": model,
                        "response_time": elapsed,
                        "tokens": tokens,
                        "tokens_per_second": tokens / elapsed if elapsed > 0 else 0,
                        "status": "success"
                    }
                else:
                    return {
                        "model": model,
                        "status": "failed",
                        "error": f"HTTP {response.status}"
                    }

    except Exception as e:
        return {
            "model": model,
            "status": "error",
            "error": str(e)
        }

async def main():
    """Quick performance validation"""

    # Test key models with simple queries
    test_cases = [
        ("tinyllama:latest", "What is 2+2?"),
        ("deepseek-coder:latest", "Write hello world in Python"),
        ("mistral:7b", "Explain a function"),
        ("codellama:13b", "Debug this code"),
        ("qwen2.5-coder:32b", "Design an API")
    ]

    print("🚀 Quick Model Performance Test")
    print("=" * 50)

    results = []

    for model, query in test_cases:
        print(f"\nTesting {model}...")
        result = await test_model_performance(model, query)
        results.append(result)

        if result["status"] == "success":
            print(f"  ✅ {result['response_time']:.2f}s ({result['tokens_per_second']:.1f} tok/s)")
        else:
            print(f"  ❌ {result.get('error', 'Failed')}")

    # Summary
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)

    successful = [r for r in results if r["status"] == "success"]

    print(f"Models tested: {len(test_cases)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(test_cases) - len(successful)}")

    if successful:
        avg_response_time = sum(r["response_time"] for r in successful) / len(successful)
        print(f"Average response time: {avg_response_time:.2f}s")

        # Check thresholds
        thresholds = {
            "tinyllama:latest": 3.0,
            "deepseek-coder:latest": 5.0,
            "mistral:7b": 8.0,
            "codellama:13b": 15.0,
            "qwen2.5-coder:32b": 30.0
        }

        print("\nThreshold Analysis:")
        issues = []
        for result in successful:
            model = result["model"]
            time_taken = result["response_time"]
            threshold = thresholds.get(model, 10.0)

            if time_taken <= threshold:
                print(f"  ✅ {model}: {time_taken:.2f}s ≤ {threshold}s")
            else:
                print(f"  ⚠️ {model}: {time_taken:.2f}s > {threshold}s")
                issues.append(model)

        if not issues:
            print("\n✅ All models performing within thresholds!")
            return 0
        else:
            print(f"\n⚠️ {len(issues)} models exceeding thresholds")
            return 1
    else:
        print("\n❌ No models responded successfully")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))