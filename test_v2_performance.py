#!/usr/bin/env python3
"""
Performance comparison test between v1 and v2
"""

import requests
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
import json

def test_endpoint(url, iterations=10):
    """Test an endpoint multiple times"""
    times = []
    for _ in range(iterations):
        start = time.time()
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                times.append(time.time() - start)
            else:
                times.append(None)
        except:
            times.append(None)

    valid_times = [t for t in times if t is not None]
    if not valid_times:
        return None

    return {
        'avg': statistics.mean(valid_times),
        'min': min(valid_times),
        'max': max(valid_times),
        'success_rate': len(valid_times) / len(times)
    }

def concurrent_test(url, concurrent_requests=5):
    """Test concurrent requests"""
    def single_request():
        start = time.time()
        try:
            response = requests.get(url, timeout=5)
            return time.time() - start if response.status_code == 200 else None
        except:
            return None

    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = [executor.submit(single_request) for _ in range(concurrent_requests)]
        results = [f.result() for f in futures]

    valid_results = [r for r in results if r is not None]
    if not valid_results:
        return None

    return {
        'avg': statistics.mean(valid_results),
        'max': max(valid_results),
        'success_rate': len(valid_results) / len(results)
    }

def main():
    print("🔍 Testing current simple_echo.py performance...")

    # Test current version endpoints
    endpoints = {
        'health': 'http://localhost:8309/api/echo/health',
        'session_context': 'http://localhost:8309/api/echo/session-context'
    }

    results = {}

    for name, url in endpoints.items():
        print(f"\n📊 Testing {name}...")

        # Sequential tests
        sequential = test_endpoint(url, 10)
        if sequential:
            print(f"  Sequential (10x): avg={sequential['avg']:.3f}s, min={sequential['min']:.3f}s, max={sequential['max']:.3f}s")

        # Concurrent tests
        concurrent = concurrent_test(url, 5)
        if concurrent:
            print(f"  Concurrent (5x): avg={concurrent['avg']:.3f}s, max={concurrent['max']:.3f}s")

        results[name] = {
            'sequential': sequential,
            'concurrent': concurrent
        }

    # Test database operations
    print(f"\n💾 Testing conversation persistence...")
    test_query = {
        'query': f'Performance test at {time.time()}',
        'conversation_id': f'perf_test_{int(time.time())}'
    }

    start = time.time()
    response = requests.post('http://localhost:8309/api/echo/query', json=test_query, timeout=5)
    query_time = time.time() - start

    if response.status_code == 200:
        print(f"  Query processing: {query_time:.3f}s")
        results['query_processing'] = query_time
    else:
        print(f"  Query failed: {response.status_code}")

    # Memory usage check
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"\n📈 Memory usage: {memory_mb:.1f} MB")
        results['memory_mb'] = memory_mb
    except:
        print("\n📈 Memory usage: Could not determine")

    # Save results
    with open('/opt/tower-echo-brain/performance_baseline.json', 'w') as f:
        json.dump({
            'version': 'simple_echo_v1',
            'timestamp': time.time(),
            'results': results
        }, f, indent=2)

    print(f"\n✅ Baseline performance recorded")
    print(f"📄 Results saved to: /opt/tower-echo-brain/performance_baseline.json")

if __name__ == "__main__":
    main()