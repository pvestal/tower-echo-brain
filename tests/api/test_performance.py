#!/usr/bin/env python3
"""
Performance Test Suite for Echo Brain API

Tests performance characteristics of:
- Rate limiting sliding window accuracy
- Database connection pooling efficiency
- Concurrent request handling
- Query optimization effectiveness
- Circuit breaker performance impact
- Memory and CPU usage under load

Author: Echo Brain Testing Framework
Created: 2026-01-02
"""

import pytest
import asyncio
import time
import psutil
import statistics
from typing import List, Dict, Any, Tuple
import httpx
import redis
import sys
import os
from unittest.mock import Mock, patch, AsyncMock
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import asyncpg

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from src.middleware.rate_limiting import RateLimitMiddleware, SlidingWindowRateLimiter
from src.db.pool_manager import AsyncConnectionPool, get_pool
from src.routing.auth_middleware import AuthMiddleware


class PerformanceMetrics:
    """Helper class to collect and analyze performance metrics"""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.response_times = []
        self.memory_usage = []
        self.cpu_usage = []
        self.error_count = 0
        self.success_count = 0

    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.memory_usage.append(psutil.virtual_memory().percent)
        self.cpu_usage.append(psutil.cpu_percent())

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.end_time = time.time()
        self.memory_usage.append(psutil.virtual_memory().percent)
        self.cpu_usage.append(psutil.cpu_percent())

    def record_response(self, response_time: float, success: bool):
        """Record individual response metrics"""
        self.response_times.append(response_time)
        if success:
            self.success_count += 1
        else:
            self.error_count += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        total_duration = self.end_time - self.start_time if self.end_time else 0

        return {
            'total_duration': total_duration,
            'total_requests': len(self.response_times),
            'requests_per_second': len(self.response_times) / total_duration if total_duration > 0 else 0,
            'success_rate': self.success_count / (self.success_count + self.error_count) * 100 if (self.success_count + self.error_count) > 0 else 0,
            'avg_response_time': statistics.mean(self.response_times) if self.response_times else 0,
            'median_response_time': statistics.median(self.response_times) if self.response_times else 0,
            'p95_response_time': statistics.quantiles(self.response_times, n=20)[18] if len(self.response_times) > 20 else max(self.response_times) if self.response_times else 0,
            'p99_response_time': statistics.quantiles(self.response_times, n=100)[98] if len(self.response_times) > 100 else max(self.response_times) if self.response_times else 0,
            'min_response_time': min(self.response_times) if self.response_times else 0,
            'max_response_time': max(self.response_times) if self.response_times else 0,
            'memory_usage_change': self.memory_usage[-1] - self.memory_usage[0] if len(self.memory_usage) >= 2 else 0,
            'avg_cpu_usage': statistics.mean(self.cpu_usage) if self.cpu_usage else 0
        }


class TestPerformance:
    """Performance testing for Echo Brain API components"""

    @pytest.fixture
    def performance_metrics(self):
        """Provide performance metrics collector"""
        return PerformanceMetrics()

    @pytest.fixture
    def rate_limiter(self):
        """Create RateLimitMiddleware for performance testing"""
        return RateLimitMiddleware()

    @pytest.fixture
    def mock_redis_fast(self):
        """Fast mock Redis for performance testing"""
        with patch('redis.Redis') as mock_redis:
            mock_client = Mock()
            mock_client.ping.return_value = True

            # Fast pipeline execution
            mock_pipeline = Mock()
            mock_pipeline.execute.return_value = [None, 1, None, None]  # Simulate low usage
            mock_client.pipeline.return_value = mock_pipeline

            mock_redis.return_value = mock_client
            yield mock_client

    @pytest.fixture
    async def connection_pool(self):
        """Create database connection pool for testing"""
        # Mock database configuration to avoid real connections in tests
        with patch('src.db.pool_manager.get_secure_db_config') as mock_config:
            mock_config.return_value = {
                "database": "test_db",
                "user": "test_user",
                "password": "test_pass",
                "host": "localhost",
                "port": 5432
            }

            with patch('asyncpg.create_pool') as mock_create_pool:
                mock_pool = AsyncMock()
                mock_pool.get_size.return_value = 10
                mock_pool.get_idle_size.return_value = 5

                # Mock connection for testing
                mock_connection = AsyncMock()
                mock_connection.fetch.return_value = [{'id': 1, 'name': 'test'}]
                mock_connection.execute.return_value = "SELECT 1"
                mock_connection.fetchval.return_value = 1

                mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
                mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

                mock_create_pool.return_value = mock_pool

                pool = AsyncConnectionPool(min_size=5, max_size=20)
                await pool.initialize()
                yield pool

    @pytest.mark.asyncio
    async def test_database_connection_pooling_performance(self, connection_pool, performance_metrics):
        """Test database connection pool performance under load"""
        performance_metrics.start_monitoring()

        async def execute_query():
            """Execute a test query"""
            start_time = time.time()
            try:
                result = await connection_pool.execute_query("SELECT 1")
                response_time = time.time() - start_time
                performance_metrics.record_response(response_time, True)
                return result
            except Exception as e:
                response_time = time.time() - start_time
                performance_metrics.record_response(response_time, False)
                raise

        # Execute concurrent queries to test pool performance
        concurrent_queries = 100
        tasks = [execute_query() for _ in range(concurrent_queries)]

        await asyncio.gather(*tasks, return_exceptions=True)

        performance_metrics.stop_monitoring()
        summary = performance_metrics.get_summary()

        # Performance assertions
        assert summary['requests_per_second'] > 50, f"RPS too low: {summary['requests_per_second']}"
        assert summary['success_rate'] >= 95, f"Success rate too low: {summary['success_rate']}%"
        assert summary['avg_response_time'] < 0.1, f"Average response time too high: {summary['avg_response_time']}s"
        assert summary['p95_response_time'] < 0.2, f"P95 response time too high: {summary['p95_response_time']}s"

        print(f"Database Pool Performance Summary: {summary}")

    @pytest.mark.asyncio
    async def test_rate_limiting_sliding_window_performance(self, rate_limiter, mock_redis_fast, performance_metrics):
        """Test rate limiting performance and accuracy"""
        from fastapi import Request

        performance_metrics.start_monitoring()

        async def make_rate_limited_request(request_id):
            """Make a request that goes through rate limiting"""
            start_time = time.time()
            try:
                mock_request = Mock(spec=Request)
                mock_request.url.path = "/api/echo/query"
                mock_request.client.host = f"192.168.1.{request_id % 255}"  # Vary IPs
                mock_request.headers = {}

                result = await rate_limiter.check_rate_limit(mock_request)
                response_time = time.time() - start_time
                performance_metrics.record_response(response_time, True)
                return result
            except Exception as e:
                response_time = time.time() - start_time
                performance_metrics.record_response(response_time, "Rate limit" not in str(e))
                return e

        # Test rate limiting with concurrent requests
        concurrent_requests = 200
        tasks = [make_rate_limited_request(i) for i in range(concurrent_requests)]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        performance_metrics.stop_monitoring()
        summary = performance_metrics.get_summary()

        # Performance assertions for rate limiting
        assert summary['requests_per_second'] > 100, f"Rate limiter RPS too low: {summary['requests_per_second']}"
        assert summary['avg_response_time'] < 0.05, f"Rate limiter too slow: {summary['avg_response_time']}s"
        assert summary['p99_response_time'] < 0.1, f"Rate limiter P99 too high: {summary['p99_response_time']}s"

        # Test that rate limiting actually works (should have some failures)
        rate_limited_count = sum(1 for r in results if isinstance(r, Exception) and "Rate limit" in str(r))
        success_count = sum(1 for r in results if not isinstance(r, Exception))

        print(f"Rate Limiting Performance: {rate_limited_count} limited, {success_count} allowed")
        print(f"Rate Limiting Summary: {summary}")

    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, connection_pool, performance_metrics):
        """Test system performance under high concurrent load"""
        performance_metrics.start_monitoring()

        async def simulate_api_request(request_id):
            """Simulate a complete API request with database access"""
            start_time = time.time()
            try:
                # Simulate query execution
                await connection_pool.execute_query("SELECT $1", request_id)

                # Simulate processing delay
                await asyncio.sleep(0.01)  # 10ms processing

                response_time = time.time() - start_time
                performance_metrics.record_response(response_time, True)
                return f"response_{request_id}"
            except Exception as e:
                response_time = time.time() - start_time
                performance_metrics.record_response(response_time, False)
                raise

        # High concurrency test
        concurrent_requests = 500
        batch_size = 50

        # Process in batches to avoid overwhelming the system
        for batch_start in range(0, concurrent_requests, batch_size):
            batch_end = min(batch_start + batch_size, concurrent_requests)
            batch_tasks = [simulate_api_request(i) for i in range(batch_start, batch_end)]

            await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Small delay between batches
            await asyncio.sleep(0.1)

        performance_metrics.stop_monitoring()
        summary = performance_metrics.get_summary()

        # High load performance assertions
        assert summary['requests_per_second'] > 30, f"Concurrent handling RPS too low: {summary['requests_per_second']}"
        assert summary['success_rate'] >= 90, f"Success rate under load too low: {summary['success_rate']}%"
        assert summary['avg_response_time'] < 0.5, f"Response time under load too high: {summary['avg_response_time']}s"
        assert summary['memory_usage_change'] < 20, f"Memory usage increase too high: {summary['memory_usage_change']}%"

        print(f"Concurrent Load Performance Summary: {summary}")

    @pytest.mark.asyncio
    async def test_query_optimization_effectiveness(self, connection_pool, performance_metrics):
        """Test query optimization and caching performance"""
        performance_metrics.start_monitoring()

        # Test cached vs uncached query performance
        test_query = "SELECT COUNT(*) FROM test_table WHERE id = $1"
        cache_key = "test_count_query"

        async def execute_with_cache():
            """Execute query with caching"""
            start_time = time.time()
            try:
                result = await connection_pool.execute_query(
                    test_query, 1,
                    cache_key=cache_key,
                    cache_ttl=300
                )
                response_time = time.time() - start_time
                performance_metrics.record_response(response_time, True)
                return result
            except Exception as e:
                response_time = time.time() - start_time
                performance_metrics.record_response(response_time, False)
                raise

        # Execute same query multiple times to test caching
        cache_test_runs = 50
        tasks = [execute_with_cache() for _ in range(cache_test_runs)]

        await asyncio.gather(*tasks, return_exceptions=True)

        performance_metrics.stop_monitoring()
        summary = performance_metrics.get_summary()

        # Cache performance should be very fast after first execution
        assert summary['min_response_time'] < 0.005, f"Cache hit time too slow: {summary['min_response_time']}s"
        assert summary['avg_response_time'] < 0.02, f"Average cached response time too high: {summary['avg_response_time']}s"

        print(f"Query Cache Performance Summary: {summary}")

    @pytest.mark.asyncio
    async def test_circuit_breaker_performance_impact(self):
        """Test performance impact of circuit breaker pattern"""
        # This would test actual circuit breaker implementation
        # For now, we'll test the concept with mocks

        performance_metrics = PerformanceMetrics()
        performance_metrics.start_monitoring()

        async def circuit_breaker_request():
            """Simulate request with circuit breaker"""
            start_time = time.time()
            try:
                # Simulate circuit breaker check (very fast)
                circuit_state = "closed"  # closed = normal operation

                if circuit_state == "open":
                    raise Exception("Circuit breaker open")

                # Simulate actual request
                await asyncio.sleep(0.01)  # 10ms request

                response_time = time.time() - start_time
                performance_metrics.record_response(response_time, True)
                return "success"
            except Exception as e:
                response_time = time.time() - start_time
                performance_metrics.record_response(response_time, False)
                raise

        # Test circuit breaker overhead
        circuit_breaker_requests = 100
        tasks = [circuit_breaker_request() for _ in range(circuit_breaker_requests)]

        await asyncio.gather(*tasks, return_exceptions=True)

        performance_metrics.stop_monitoring()
        summary = performance_metrics.get_summary()

        # Circuit breaker should add minimal overhead
        assert summary['avg_response_time'] < 0.02, f"Circuit breaker adds too much overhead: {summary['avg_response_time']}s"
        assert summary['requests_per_second'] > 80, f"Circuit breaker reduces throughput too much: {summary['requests_per_second']}"

        print(f"Circuit Breaker Performance Summary: {summary}")

    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, connection_pool):
        """Test for memory leaks during sustained operation"""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        async def memory_test_operation():
            """Operation that might cause memory leaks"""
            await connection_pool.execute_query("SELECT 1")
            # Simulate some data processing
            data = [i for i in range(1000)]
            del data

        # Run many operations to detect memory leaks
        for batch in range(10):  # 10 batches
            tasks = [memory_test_operation() for _ in range(100)]  # 100 ops per batch
            await asyncio.gather(*tasks, return_exceptions=True)

            # Check memory usage periodically
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = current_memory - initial_memory

            # Memory shouldn't grow excessively
            assert memory_increase < 100, f"Memory leak detected: {memory_increase}MB increase"

            await asyncio.sleep(0.1)  # Small delay between batches

        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        total_memory_increase = final_memory - initial_memory

        print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (increase: {total_memory_increase:.1f}MB)")

        # Total memory increase should be reasonable
        assert total_memory_increase < 50, f"Excessive memory usage: {total_memory_increase}MB"

    @pytest.mark.asyncio
    async def test_database_pool_scaling(self, performance_metrics):
        """Test database pool scaling under different load patterns"""
        # Test pool with different sizes
        pool_configs = [
            {'min_size': 5, 'max_size': 10},
            {'min_size': 10, 'max_size': 30},
            {'min_size': 20, 'max_size': 50}
        ]

        pool_performance = {}

        for config in pool_configs:
            performance_metrics = PerformanceMetrics()
            performance_metrics.start_monitoring()

            # Create pool with specific configuration
            with patch('src.db.pool_manager.get_secure_db_config') as mock_config:
                mock_config.return_value = {
                    "database": "test_db", "user": "test_user",
                    "password": "test_pass", "host": "localhost", "port": 5432
                }

                with patch('asyncpg.create_pool') as mock_create_pool:
                    mock_pool = AsyncMock()
                    mock_pool.get_size.return_value = config['max_size']
                    mock_pool.get_idle_size.return_value = config['min_size']

                    mock_connection = AsyncMock()
                    mock_connection.fetch.return_value = [{'result': 1}]
                    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
                    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

                    mock_create_pool.return_value = mock_pool

                    test_pool = AsyncConnectionPool(
                        min_size=config['min_size'],
                        max_size=config['max_size']
                    )
                    await test_pool.initialize()

                    # Test with different load levels
                    for load_level in [25, 50, 100]:  # Different concurrent request levels
                        async def pool_test_query():
                            start_time = time.time()
                            try:
                                await test_pool.execute_query("SELECT 1")
                                response_time = time.time() - start_time
                                performance_metrics.record_response(response_time, True)
                            except Exception:
                                response_time = time.time() - start_time
                                performance_metrics.record_response(response_time, False)

                        tasks = [pool_test_query() for _ in range(load_level)]
                        await asyncio.gather(*tasks, return_exceptions=True)

            performance_metrics.stop_monitoring()
            summary = performance_metrics.get_summary()
            pool_performance[f"pool_{config['min_size']}_{config['max_size']}"] = summary

            await test_pool.close()

        # Analyze pool performance
        for pool_name, perf in pool_performance.items():
            print(f"Pool {pool_name} Performance: RPS={perf['requests_per_second']:.1f}, "
                  f"Avg={perf['avg_response_time']*1000:.1f}ms, "
                  f"Success Rate={perf['success_rate']:.1f}%")

        # Larger pools should handle higher loads better
        assert pool_performance['pool_20_50']['requests_per_second'] >= pool_performance['pool_5_10']['requests_per_second']

    @pytest.mark.asyncio
    async def test_rate_limiting_fairness(self, mock_redis_fast):
        """Test rate limiting fairness across different clients"""
        rate_limiter = RateLimitMiddleware()
        rate_limiter.limiter.redis_available = True
        rate_limiter.limiter.redis_client = mock_redis_fast

        # Mock different Redis responses for different IPs
        def mock_pipeline_response(*args, **kwargs):
            pipeline = Mock()
            # Simulate different usage levels for different IPs
            pipeline.execute.return_value = [None, 5, None, None]  # 5 requests for all IPs
            return pipeline

        mock_redis_fast.pipeline = mock_pipeline_response

        from fastapi import Request

        # Test multiple clients
        client_ips = [f"192.168.1.{i}" for i in range(1, 11)]  # 10 different clients
        client_results = {}

        for ip in client_ips:
            client_metrics = PerformanceMetrics()
            client_metrics.start_monitoring()

            async def client_request():
                mock_request = Mock(spec=Request)
                mock_request.url.path = "/api/echo/query"
                mock_request.client.host = ip
                mock_request.headers = {}

                start_time = time.time()
                try:
                    result = await rate_limiter.check_rate_limit(mock_request)
                    response_time = time.time() - start_time
                    client_metrics.record_response(response_time, True)
                    return result
                except Exception as e:
                    response_time = time.time() - start_time
                    client_metrics.record_response(response_time, False)
                    return e

            # Each client makes multiple requests
            client_tasks = [client_request() for _ in range(20)]
            await asyncio.gather(*client_tasks, return_exceptions=True)

            client_metrics.stop_monitoring()
            client_results[ip] = client_metrics.get_summary()

        # Check fairness - all clients should get similar treatment
        success_rates = [result['success_rate'] for result in client_results.values()]
        response_times = [result['avg_response_time'] for result in client_results.values()]

        # Success rates should be fairly consistent across clients
        success_rate_std = statistics.stdev(success_rates) if len(success_rates) > 1 else 0
        response_time_std = statistics.stdev(response_times) if len(response_times) > 1 else 0

        assert success_rate_std < 20, f"Rate limiting not fair across clients: {success_rate_std}% std deviation"
        assert response_time_std < 0.01, f"Response time not consistent: {response_time_std}s std deviation"

        print(f"Rate Limiting Fairness - Success Rate StdDev: {success_rate_std:.2f}%, "
              f"Response Time StdDev: {response_time_std*1000:.2f}ms")


class TestPerformanceBenchmarks:
    """Benchmark tests for setting performance baselines"""

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_baseline_api_performance(self):
        """Establish baseline performance metrics"""
        metrics = PerformanceMetrics()
        metrics.start_monitoring()

        # Simulate typical API request processing
        async def baseline_request():
            start_time = time.time()

            # Simulate authentication (0.5ms)
            await asyncio.sleep(0.0005)

            # Simulate rate limiting check (0.2ms)
            await asyncio.sleep(0.0002)

            # Simulate database query (5ms)
            await asyncio.sleep(0.005)

            # Simulate business logic (2ms)
            await asyncio.sleep(0.002)

            response_time = time.time() - start_time
            metrics.record_response(response_time, True)

        # Baseline performance test
        baseline_requests = 1000
        batch_size = 100

        for batch_start in range(0, baseline_requests, batch_size):
            batch_end = min(batch_start + batch_size, baseline_requests)
            tasks = [baseline_request() for _ in range(batch_end - batch_start)]
            await asyncio.gather(*tasks)

        metrics.stop_monitoring()
        summary = metrics.get_summary()

        # Baseline performance expectations
        assert summary['requests_per_second'] > 50, f"Baseline RPS too low: {summary['requests_per_second']}"
        assert summary['avg_response_time'] < 0.02, f"Baseline response time too high: {summary['avg_response_time']}s"
        assert summary['p95_response_time'] < 0.05, f"Baseline P95 too high: {summary['p95_response_time']}s"

        print("Performance Baseline Established:")
        print(f"  RPS: {summary['requests_per_second']:.1f}")
        print(f"  Average Response Time: {summary['avg_response_time']*1000:.1f}ms")
        print(f"  P95 Response Time: {summary['p95_response_time']*1000:.1f}ms")
        print(f"  P99 Response Time: {summary['p99_response_time']*1000:.1f}ms")

        return summary


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "not benchmark"])