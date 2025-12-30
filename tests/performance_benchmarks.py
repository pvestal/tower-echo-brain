"""
Performance Benchmarks for Echo Brain
Measures execution times and resource usage for critical operations
"""

import asyncio
import time
import psutil
import statistics
from typing import Dict, Any, List
import aiohttp
import json
from pathlib import Path
import subprocess

class PerformanceBenchmark:
    """Comprehensive performance benchmarking"""

    def __init__(self):
        self.results = {}
        self.echo_url = "http://localhost:8309"

    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks"""

        print("\n" + "=" * 70)
        print("ECHO BRAIN PERFORMANCE BENCHMARKS")
        print("=" * 70)

        # 1. API Response Times
        print("\n[1/7] API Response Time Benchmarks...")
        self.results['api_response'] = await self._benchmark_api_responses()

        # 2. Query Processing
        print("\n[2/7] Query Processing Benchmarks...")
        self.results['query_processing'] = await self._benchmark_query_processing()

        # 3. Memory Operations
        print("\n[3/7] Memory System Benchmarks...")
        self.results['memory_ops'] = await self._benchmark_memory_operations()

        # 4. Code Execution
        print("\n[4/7] Code Execution Benchmarks...")
        self.results['code_execution'] = await self._benchmark_code_execution()

        # 5. ComfyUI Workflow
        print("\n[5/7] ComfyUI Workflow Benchmarks...")
        self.results['comfyui'] = await self._benchmark_comfyui()

        # 6. Database Operations
        print("\n[6/7] Database Operation Benchmarks...")
        self.results['database'] = await self._benchmark_database()

        # 7. Resource Usage
        print("\n[7/7] Resource Usage Analysis...")
        self.results['resources'] = await self._analyze_resource_usage()

        # Calculate summary
        self._calculate_summary()

        return self.results

    async def _benchmark_api_responses(self) -> Dict[str, Any]:
        """Benchmark API endpoint response times"""
        endpoints = [
            '/api/echo/health',
            '/api/echo/models/list',
            '/api/echo/brain'
        ]

        results = {}
        async with aiohttp.ClientSession() as session:
            for endpoint in endpoints:
                times = []
                for _ in range(10):  # 10 iterations
                    start = time.perf_counter()
                    try:
                        async with session.get(f"{self.echo_url}{endpoint}") as resp:
                            await resp.text()
                        times.append((time.perf_counter() - start) * 1000)
                    except:
                        times.append(999999)

                results[endpoint] = {
                    'mean_ms': statistics.mean(times),
                    'median_ms': statistics.median(times),
                    'min_ms': min(times),
                    'max_ms': max(times),
                    'stdev_ms': statistics.stdev(times) if len(times) > 1 else 0
                }

                print(f"  {endpoint}:")
                print(f"    Mean: {results[endpoint]['mean_ms']:.2f}ms")
                print(f"    Median: {results[endpoint]['median_ms']:.2f}ms")
                print(f"    Range: {results[endpoint]['min_ms']:.2f}-{results[endpoint]['max_ms']:.2f}ms")

        return results

    async def _benchmark_query_processing(self) -> Dict[str, Any]:
        """Benchmark query processing with different models"""
        queries = [
            ("simple", "What is 2+2?", "llama3.2:3b"),
            ("medium", "Explain recursion in programming", "llama3.2:3b"),
            ("complex", "Write a Python function to calculate fibonacci", "qwen2.5-coder:7b")
        ]

        results = {}
        async with aiohttp.ClientSession() as session:
            for query_type, query, model in queries:
                times = []
                for _ in range(3):  # 3 iterations (queries are slower)
                    start = time.perf_counter()
                    try:
                        async with session.post(
                            f"{self.echo_url}/api/echo/query",
                            json={"query": query, "model": model}
                        ) as resp:
                            await resp.text()
                        elapsed = (time.perf_counter() - start) * 1000
                        times.append(elapsed)
                    except Exception as e:
                        print(f"    Error: {e}")
                        times.append(999999)

                if times:
                    results[query_type] = {
                        'query': query[:50],
                        'model': model,
                        'mean_ms': statistics.mean(times),
                        'min_ms': min(times),
                        'max_ms': max(times)
                    }

                    print(f"  {query_type} query ({model}):")
                    print(f"    Mean: {results[query_type]['mean_ms']:.0f}ms")
                    print(f"    Range: {results[query_type]['min_ms']:.0f}-{results[query_type]['max_ms']:.0f}ms")

        return results

    async def _benchmark_memory_operations(self) -> Dict[str, Any]:
        """Benchmark persistent memory operations"""
        try:
            from src.capabilities.persistent_memory import PersistentMemorySystem

            memory = PersistentMemorySystem({
                'host': 'localhost',
                'user': 'patrick',
                'password': '***REMOVED***',
                'database': 'echo_brain'
            })

            await memory.connect()

            # Benchmark store operations
            store_times = []
            for i in range(10):
                start = time.perf_counter()
                await memory.store_memory(
                    f"benchmark_{i}",
                    "benchmark",
                    f"Test data {i}",
                    importance=0.5
                )
                store_times.append((time.perf_counter() - start) * 1000)

            # Benchmark retrieve operations
            retrieve_times = []
            for _ in range(10):
                start = time.perf_counter()
                await memory.retrieve_memory(category="benchmark", limit=5)
                retrieve_times.append((time.perf_counter() - start) * 1000)

            # Benchmark startup context
            start = time.perf_counter()
            context = await memory.get_startup_context()
            startup_time = (time.perf_counter() - start) * 1000

            await memory.close()

            results = {
                'store': {
                    'mean_ms': statistics.mean(store_times),
                    'median_ms': statistics.median(store_times)
                },
                'retrieve': {
                    'mean_ms': statistics.mean(retrieve_times),
                    'median_ms': statistics.median(retrieve_times)
                },
                'startup_context_ms': startup_time
            }

            print(f"  Store operation: {results['store']['mean_ms']:.2f}ms")
            print(f"  Retrieve operation: {results['retrieve']['mean_ms']:.2f}ms")
            print(f"  Startup context load: {results['startup_context_ms']:.2f}ms")

            return results

        except Exception as e:
            print(f"  Error: {e}")
            return {'error': str(e)}

    async def _benchmark_code_execution(self) -> Dict[str, Any]:
        """Benchmark code execution capabilities"""
        try:
            from src.capabilities.code_executor_fixed import CodeExecutor

            executor = CodeExecutor()

            # Simple code
            simple_code = "print(sum(range(100)))"
            start = time.perf_counter()
            result1 = await executor.execute_sandboxed(simple_code)
            simple_time = (time.perf_counter() - start) * 1000

            # Complex code
            complex_code = """
import math
primes = []
for n in range(2, 1000):
    is_prime = True
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            is_prime = False
            break
    if is_prime:
        primes.append(n)
print(f"Found {len(primes)} primes")
"""
            start = time.perf_counter()
            result2 = await executor.execute_sandboxed(complex_code)
            complex_time = (time.perf_counter() - start) * 1000

            results = {
                'simple_execution_ms': simple_time,
                'complex_execution_ms': complex_time,
                'simple_success': result1.get('success', False),
                'complex_success': result2.get('success', False)
            }

            print(f"  Simple code: {results['simple_execution_ms']:.0f}ms")
            print(f"  Complex code: {results['complex_execution_ms']:.0f}ms")

            return results

        except Exception as e:
            print(f"  Error: {e}")
            return {'error': str(e)}

    async def _benchmark_comfyui(self) -> Dict[str, Any]:
        """Benchmark ComfyUI workflow submission"""
        try:
            from src.capabilities.comfyui_integration import ComfyUIIntegration

            integrator = ComfyUIIntegration()

            # Check if ComfyUI is available
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get("http://localhost:8188/system_stats") as resp:
                        if resp.status != 200:
                            return {'available': False}
                except:
                    return {'available': False}

            # Load and inject workflow (without execution)
            start = time.perf_counter()
            workflow = await integrator.load_workflow_template()
            workflow = await integrator.inject_parameters(
                workflow,
                prompt="benchmark test",
                steps=5  # Minimal steps for benchmark
            )
            prep_time = (time.perf_counter() - start) * 1000

            # Note: Not executing full workflow in benchmark to save time
            results = {
                'available': True,
                'workflow_prep_ms': prep_time,
                'note': 'Full execution skipped for benchmarking'
            }

            print(f"  ComfyUI available: ✅")
            print(f"  Workflow preparation: {results['workflow_prep_ms']:.2f}ms")

            return results

        except Exception as e:
            print(f"  Error: {e}")
            return {'error': str(e)}

    async def _benchmark_database(self) -> Dict[str, Any]:
        """Benchmark database operations"""
        try:
            import asyncpg

            conn = await asyncpg.connect(
                host='localhost',
                user='patrick',
                password='***REMOVED***',
                database='echo_brain'
            )

            # Simple query
            times = []
            for _ in range(20):
                start = time.perf_counter()
                await conn.fetchval("SELECT 1")
                times.append((time.perf_counter() - start) * 1000)

            simple_mean = statistics.mean(times)

            # Complex query
            times = []
            for _ in range(10):
                start = time.perf_counter()
                await conn.fetch("""
                    SELECT id, importance, category, last_accessed
                    FROM persistent_memories
                    ORDER BY importance DESC, last_accessed DESC
                    LIMIT 10
                """)
                times.append((time.perf_counter() - start) * 1000)

            complex_mean = statistics.mean(times)

            # Count query
            start = time.perf_counter()
            count = await conn.fetchval("SELECT COUNT(*) FROM conversations")
            count_time = (time.perf_counter() - start) * 1000

            await conn.close()

            results = {
                'simple_query_ms': simple_mean,
                'complex_query_ms': complex_mean,
                'count_query_ms': count_time,
                'conversation_count': count
            }

            print(f"  Simple query: {results['simple_query_ms']:.2f}ms")
            print(f"  Complex query: {results['complex_query_ms']:.2f}ms")
            print(f"  Count query: {results['count_query_ms']:.2f}ms ({count} rows)")

            return results

        except Exception as e:
            print(f"  Error: {e}")
            return {'error': str(e)}

    async def _analyze_resource_usage(self) -> Dict[str, Any]:
        """Analyze system resource usage"""

        # CPU usage over 5 seconds
        cpu_samples = []
        for _ in range(5):
            cpu_samples.append(psutil.cpu_percent(interval=1))

        # Memory usage
        memory = psutil.virtual_memory()

        # Disk I/O
        disk_io_before = psutil.disk_io_counters()
        await asyncio.sleep(2)
        disk_io_after = psutil.disk_io_counters()

        # Process specific stats
        echo_process = None
        for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'cpu_percent']):
            if 'echo' in proc.info['name'].lower() or 'uvicorn' in proc.info['name'].lower():
                echo_process = proc
                break

        results = {
            'cpu': {
                'mean_percent': statistics.mean(cpu_samples),
                'max_percent': max(cpu_samples),
                'cores': psutil.cpu_count()
            },
            'memory': {
                'used_gb': memory.used / (1024**3),
                'total_gb': memory.total / (1024**3),
                'percent': memory.percent
            },
            'disk_io': {
                'read_mb_s': (disk_io_after.read_bytes - disk_io_before.read_bytes) / (2 * 1024 * 1024),
                'write_mb_s': (disk_io_after.write_bytes - disk_io_before.write_bytes) / (2 * 1024 * 1024)
            }
        }

        if echo_process:
            results['echo_process'] = {
                'memory_percent': echo_process.info['memory_percent'],
                'cpu_percent': echo_process.info['cpu_percent']
            }

        print(f"  CPU: {results['cpu']['mean_percent']:.1f}% (avg), {results['cpu']['max_percent']:.1f}% (max)")
        print(f"  Memory: {results['memory']['used_gb']:.1f}GB / {results['memory']['total_gb']:.1f}GB ({results['memory']['percent']:.1f}%)")
        print(f"  Disk I/O: {results['disk_io']['read_mb_s']:.2f} MB/s read, {results['disk_io']['write_mb_s']:.2f} MB/s write")

        if echo_process:
            print(f"  Echo Process: {results['echo_process']['cpu_percent']:.1f}% CPU, {results['echo_process']['memory_percent']:.1f}% Memory")

        return results

    def _calculate_summary(self):
        """Calculate performance summary and ratings"""

        ratings = {}

        # API performance rating
        if 'api_response' in self.results:
            avg_response = statistics.mean([
                v['mean_ms'] for v in self.results['api_response'].values()
            ])
            if avg_response < 10:
                ratings['api'] = 'Excellent'
            elif avg_response < 50:
                ratings['api'] = 'Good'
            elif avg_response < 100:
                ratings['api'] = 'Acceptable'
            else:
                ratings['api'] = 'Needs Improvement'

        # Database performance rating
        if 'database' in self.results and 'simple_query_ms' in self.results['database']:
            db_speed = self.results['database']['simple_query_ms']
            if db_speed < 1:
                ratings['database'] = 'Excellent'
            elif db_speed < 5:
                ratings['database'] = 'Good'
            elif db_speed < 10:
                ratings['database'] = 'Acceptable'
            else:
                ratings['database'] = 'Needs Improvement'

        # Resource usage rating
        if 'resources' in self.results:
            cpu = self.results['resources']['cpu']['mean_percent']
            mem = self.results['resources']['memory']['percent']
            if cpu < 20 and mem < 50:
                ratings['resources'] = 'Excellent'
            elif cpu < 50 and mem < 70:
                ratings['resources'] = 'Good'
            elif cpu < 80 and mem < 90:
                ratings['resources'] = 'Acceptable'
            else:
                ratings['resources'] = 'Needs Improvement'

        self.results['summary'] = {
            'ratings': ratings,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        print("\n" + "=" * 70)
        print("PERFORMANCE SUMMARY")
        print("-" * 70)
        for category, rating in ratings.items():
            symbol = "✅" if rating in ['Excellent', 'Good'] else "⚠️"
            print(f"{symbol} {category.title()}: {rating}")
        print("=" * 70)


async def main():
    """Run performance benchmarks"""
    benchmark = PerformanceBenchmark()
    results = await benchmark.run_all_benchmarks()

    # Save results
    output_file = Path('/opt/tower-echo-brain/tests/performance_benchmark_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/opt/tower-echo-brain')

    asyncio.run(main())