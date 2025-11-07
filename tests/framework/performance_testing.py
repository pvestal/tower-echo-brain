"""
Performance Testing Suite for Echo Brain Load and Stress Testing
================================================================

This module provides comprehensive performance testing capabilities including
load testing, stress testing, memory profiling, concurrent user simulation,
and performance regression detection for Echo Brain's modernized architecture.

Features:
- Load testing with configurable user patterns
- Stress testing with breaking point detection
- Memory and CPU profiling
- Concurrent request simulation
- Performance baseline management
- Regression testing
- Real-time monitoring

Author: Development Testing Framework Agent
Created: 2025-11-06
"""

import asyncio
import time
import json
import statistics
import threading
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

import pytest
import psutil
import httpx
import numpy as np
from memory_profiler import profile

from .test_framework_core import TestFrameworkCore, TestMetrics


@dataclass
class LoadTestConfig:
    """Configuration for load testing."""
    name: str
    target_url: str
    method: str = "GET"
    payload: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    concurrent_users: int = 10
    duration_seconds: int = 60
    ramp_up_seconds: int = 10
    ramp_down_seconds: int = 5
    think_time_min: float = 0.1
    think_time_max: float = 1.0
    success_criteria: Dict[str, Any] = field(default_factory=lambda: {
        "max_response_time": 2.0,
        "min_success_rate": 0.95,
        "max_error_rate": 0.05
    })


@dataclass
class StressTestConfig:
    """Configuration for stress testing."""
    name: str
    target_url: str
    method: str = "GET"
    payload: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    min_users: int = 1
    max_users: int = 100
    step_size: int = 10
    step_duration: int = 30
    breaking_point_criteria: Dict[str, Any] = field(default_factory=lambda: {
        "max_response_time": 5.0,
        "min_success_rate": 0.8,
        "max_error_rate": 0.2
    })


@dataclass
class PerformanceMetrics:
    """Container for performance test metrics."""
    test_name: str
    test_type: str  # 'load', 'stress', 'spike', 'volume'
    start_time: datetime
    end_time: datetime
    duration: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    response_times: List[float]
    error_rates: List[float]
    throughput: float  # requests per second
    concurrent_users: int
    system_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful_requests / self.total_requests if self.total_requests > 0 else 0
    
    @property
    def avg_response_time(self) -> float:
        """Calculate average response time."""
        return statistics.mean(self.response_times) if self.response_times else 0
    
    @property
    def percentile_95(self) -> float:
        """Calculate 95th percentile response time."""
        return np.percentile(self.response_times, 95) if self.response_times else 0
    
    @property
    def percentile_99(self) -> float:
        """Calculate 99th percentile response time."""
        return np.percentile(self.response_times, 99) if self.response_times else 0


class SystemMonitor:
    """Real-time system monitoring during performance tests."""
    
    def __init__(self):
        """Initialize system monitor."""
        self.monitoring = False
        self.metrics_history: List[Dict[str, Any]] = []
        self.monitor_thread: Optional[threading.Thread] = None
        
    def start_monitoring(self, interval: float = 1.0):
        """Start system monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            
    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                network = psutil.net_io_counters()
                
                metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_mb': memory.available / (1024 * 1024),
                    'memory_used_mb': memory.used / (1024 * 1024),
                    'disk_percent': (disk.used / disk.total) * 100,
                    'network_bytes_sent': network.bytes_sent,
                    'network_bytes_recv': network.bytes_recv
                }
                
                self.metrics_history.append(metrics)
                
                # Keep only last 1000 entries to prevent memory issues
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                    
            except Exception as e:
                logging.error(f"System monitoring error: {e}")
                
            time.sleep(interval)
            
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        if not self.metrics_history:
            return {}
            
        cpu_values = [m['cpu_percent'] for m in self.metrics_history]
        memory_values = [m['memory_percent'] for m in self.metrics_history]
        
        return {
            'cpu': {
                'avg': statistics.mean(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values),
                'p95': np.percentile(cpu_values, 95)
            },
            'memory': {
                'avg': statistics.mean(memory_values),
                'max': max(memory_values),
                'min': min(memory_values),
                'p95': np.percentile(memory_values, 95)
            },
            'samples_collected': len(self.metrics_history)
        }


class LoadTester:
    """Load testing implementation."""
    
    def __init__(self, framework: TestFrameworkCore):
        """Initialize load tester."""
        self.framework = framework
        self.system_monitor = SystemMonitor()
        
    async def run_load_test(self, config: LoadTestConfig) -> PerformanceMetrics:
        """Execute load test based on configuration."""
        with self.framework.monitor_test(f"load_test_{config.name}"):
            self.framework.logger.info(f"Starting load test: {config.name}")
            
            # Start system monitoring
            self.system_monitor.start_monitoring()
            
            try:
                start_time = datetime.now()
                response_times = []
                successful_requests = 0
                failed_requests = 0
                
                # Create HTTP client
                async with httpx.AsyncClient() as client:
                    # Execute load test
                    semaphore = asyncio.Semaphore(config.concurrent_users)
                    tasks = []
                    
                    # Ramp up phase
                    test_start = time.time()
                    users_started = 0
                    
                    while time.time() - test_start < config.duration_seconds:
                        # Ramp up users gradually
                        if users_started < config.concurrent_users:
                            if time.time() - test_start < config.ramp_up_seconds:
                                users_to_add = int(
                                    (time.time() - test_start) / config.ramp_up_seconds * config.concurrent_users
                                ) - users_started
                                
                                for _ in range(users_to_add):
                                    task = asyncio.create_task(
                                        self._simulate_user(client, config, semaphore)
                                    )
                                    tasks.append(task)
                                    users_started += 1
                                    
                        await asyncio.sleep(0.1)
                        
                    # Wait for all tasks to complete
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results
                    for result in results:
                        if isinstance(result, dict):
                            response_times.extend(result.get('response_times', []))
                            successful_requests += result.get('successful', 0)
                            failed_requests += result.get('failed', 0)
                            
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                total_requests = successful_requests + failed_requests
                
                # Calculate throughput
                throughput = total_requests / duration if duration > 0 else 0
                
                # Get system metrics summary
                system_metrics = self.system_monitor.get_metrics_summary()
                
                return PerformanceMetrics(
                    test_name=config.name,
                    test_type='load',
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    total_requests=total_requests,
                    successful_requests=successful_requests,
                    failed_requests=failed_requests,
                    response_times=response_times,
                    error_rates=[],  # Calculate if needed
                    throughput=throughput,
                    concurrent_users=config.concurrent_users,
                    system_metrics=system_metrics
                )
                
            finally:
                # Stop system monitoring
                self.system_monitor.stop_monitoring()
                
    async def _simulate_user(
        self,
        client: httpx.AsyncClient,
        config: LoadTestConfig,
        semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        """Simulate a single user's behavior."""
        async with semaphore:
            response_times = []
            successful = 0
            failed = 0
            
            user_start = time.time()
            
            while time.time() - user_start < config.duration_seconds:
                try:
                    request_start = time.time()
                    
                    # Make request
                    if config.method.upper() == 'GET':
                        response = await client.get(
                            config.target_url,
                            headers=config.headers
                        )
                    elif config.method.upper() == 'POST':
                        response = await client.post(
                            config.target_url,
                            json=config.payload,
                            headers=config.headers
                        )
                    elif config.method.upper() == 'PUT':
                        response = await client.put(
                            config.target_url,
                            json=config.payload,
                            headers=config.headers
                        )
                    else:
                        raise ValueError(f"Unsupported method: {config.method}")
                        
                    request_duration = time.time() - request_start
                    response_times.append(request_duration)
                    
                    # Check success
                    if 200 <= response.status_code < 400:
                        successful += 1
                    else:
                        failed += 1
                        
                except Exception as e:
                    failed += 1
                    logging.debug(f"Request failed: {e}")
                    
                # Think time between requests
                think_time = np.random.uniform(
                    config.think_time_min,
                    config.think_time_max
                )
                await asyncio.sleep(think_time)
                
            return {
                'response_times': response_times,
                'successful': successful,
                'failed': failed
            }


class StressTester:
    """Stress testing implementation."""
    
    def __init__(self, framework: TestFrameworkCore):
        """Initialize stress tester."""
        self.framework = framework
        self.load_tester = LoadTester(framework)
        
    async def run_stress_test(self, config: StressTestConfig) -> List[PerformanceMetrics]:
        """Execute stress test with increasing load."""
        with self.framework.monitor_test(f"stress_test_{config.name}"):
            self.framework.logger.info(f"Starting stress test: {config.name}")
            
            results = []
            current_users = config.min_users
            breaking_point_reached = False
            
            while current_users <= config.max_users and not breaking_point_reached:
                self.framework.logger.info(f"Testing with {current_users} concurrent users")
                
                # Create load test config for this step
                load_config = LoadTestConfig(
                    name=f"{config.name}_users_{current_users}",
                    target_url=config.target_url,
                    method=config.method,
                    payload=config.payload,
                    headers=config.headers,
                    concurrent_users=current_users,
                    duration_seconds=config.step_duration,
                    ramp_up_seconds=min(10, config.step_duration // 3)
                )
                
                # Run load test for this step
                metrics = await self.load_tester.run_load_test(load_config)
                metrics.test_type = 'stress'
                results.append(metrics)
                
                # Check if breaking point criteria are met
                if self._check_breaking_point(metrics, config.breaking_point_criteria):
                    breaking_point_reached = True
                    self.framework.logger.warning(f"Breaking point reached at {current_users} users")
                    break
                    
                current_users += config.step_size
                
                # Cool down between steps
                await asyncio.sleep(5)
                
            return results
            
    def _check_breaking_point(
        self,
        metrics: PerformanceMetrics,
        criteria: Dict[str, Any]
    ) -> bool:
        """Check if breaking point criteria are met."""
        # Check response time
        if metrics.avg_response_time > criteria.get('max_response_time', float('inf')):
            return True
            
        # Check success rate
        if metrics.success_rate < criteria.get('min_success_rate', 0):
            return True
            
        # Check error rate
        error_rate = metrics.failed_requests / metrics.total_requests if metrics.total_requests > 0 else 0
        if error_rate > criteria.get('max_error_rate', 1):
            return True
            
        return False


class PerformanceTestSuite:
    """Main performance testing suite."""
    
    def __init__(self, framework: TestFrameworkCore):
        """Initialize performance test suite."""
        self.framework = framework
        self.load_tester = LoadTester(framework)
        self.stress_tester = StressTester(framework)
        self.baseline_metrics: Dict[str, PerformanceMetrics] = {}
        self.test_results: List[PerformanceMetrics] = []
        
    async def run_performance_suite(
        self,
        load_configs: List[LoadTestConfig],
        stress_configs: List[StressTestConfig]
    ) -> Dict[str, Any]:
        """Run complete performance test suite."""
        suite_results = {
            'load_tests': [],
            'stress_tests': [],
            'summary': {}
        }
        
        # Run load tests
        for config in load_configs:
            try:
                result = await self.load_tester.run_load_test(config)
                suite_results['load_tests'].append(result)
                self.test_results.append(result)
            except Exception as e:
                self.framework.logger.error(f"Load test {config.name} failed: {e}")
                
        # Run stress tests
        for config in stress_configs:
            try:
                results = await self.stress_tester.run_stress_test(config)
                suite_results['stress_tests'].extend(results)
                self.test_results.extend(results)
            except Exception as e:
                self.framework.logger.error(f"Stress test {config.name} failed: {e}")
                
        # Generate summary
        suite_results['summary'] = self._generate_suite_summary()
        
        return suite_results
        
    def set_performance_baseline(self, test_name: str, metrics: PerformanceMetrics):
        """Set performance baseline for regression testing."""
        self.baseline_metrics[test_name] = metrics
        
    def check_performance_regression(
        self,
        current_metrics: PerformanceMetrics,
        tolerance: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """Check for performance regression against baseline."""
        if tolerance is None:
            tolerance = {
                'response_time': 0.2,  # 20% increase
                'throughput': 0.1,     # 10% decrease
                'success_rate': 0.05   # 5% decrease
            }
            
        test_name = current_metrics.test_name
        if test_name not in self.baseline_metrics:
            return {
                'regression_detected': False,
                'reason': 'No baseline available'
            }
            
        baseline = self.baseline_metrics[test_name]
        regressions = []
        
        # Check response time regression
        response_time_increase = (
            current_metrics.avg_response_time - baseline.avg_response_time
        ) / baseline.avg_response_time
        
        if response_time_increase > tolerance['response_time']:
            regressions.append({
                'metric': 'response_time',
                'baseline': baseline.avg_response_time,
                'current': current_metrics.avg_response_time,
                'change_percent': response_time_increase * 100
            })
            
        # Check throughput regression
        throughput_decrease = (
            baseline.throughput - current_metrics.throughput
        ) / baseline.throughput
        
        if throughput_decrease > tolerance['throughput']:
            regressions.append({
                'metric': 'throughput',
                'baseline': baseline.throughput,
                'current': current_metrics.throughput,
                'change_percent': -throughput_decrease * 100
            })
            
        # Check success rate regression
        success_rate_decrease = baseline.success_rate - current_metrics.success_rate
        
        if success_rate_decrease > tolerance['success_rate']:
            regressions.append({
                'metric': 'success_rate',
                'baseline': baseline.success_rate,
                'current': current_metrics.success_rate,
                'change_percent': -success_rate_decrease * 100
            })
            
        return {
            'regression_detected': len(regressions) > 0,
            'regressions': regressions,
            'baseline_test': baseline.test_name,
            'current_test': current_metrics.test_name
        }
        
    def _generate_suite_summary(self) -> Dict[str, Any]:
        """Generate summary of all performance tests."""
        if not self.test_results:
            return {}
            
        load_tests = [r for r in self.test_results if r.test_type == 'load']
        stress_tests = [r for r in self.test_results if r.test_type == 'stress']
        
        summary = {
            'total_tests': len(self.test_results),
            'load_tests_count': len(load_tests),
            'stress_tests_count': len(stress_tests)
        }
        
        if load_tests:
            avg_response_times = [t.avg_response_time for t in load_tests]
            throughputs = [t.throughput for t in load_tests]
            success_rates = [t.success_rate for t in load_tests]
            
            summary['load_test_summary'] = {
                'avg_response_time': statistics.mean(avg_response_times),
                'max_response_time': max(avg_response_times),
                'avg_throughput': statistics.mean(throughputs),
                'max_throughput': max(throughputs),
                'avg_success_rate': statistics.mean(success_rates),
                'min_success_rate': min(success_rates)
            }
            
        if stress_tests:
            max_concurrent_users = max(t.concurrent_users for t in stress_tests)
            summary['stress_test_summary'] = {
                'max_concurrent_users_tested': max_concurrent_users
            }
            
        return summary
        
    def export_results(self, file_path: str):
        """Export test results to JSON file."""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'test_results': [
                {
                    'test_name': r.test_name,
                    'test_type': r.test_type,
                    'duration': r.duration,
                    'total_requests': r.total_requests,
                    'successful_requests': r.successful_requests,
                    'failed_requests': r.failed_requests,
                    'avg_response_time': r.avg_response_time,
                    'percentile_95': r.percentile_95,
                    'percentile_99': r.percentile_99,
                    'throughput': r.throughput,
                    'success_rate': r.success_rate,
                    'concurrent_users': r.concurrent_users,
                    'system_metrics': r.system_metrics
                }
                for r in self.test_results
            ],
            'summary': self._generate_suite_summary()
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)


# Export main classes
__all__ = [
    'PerformanceTestSuite',
    'LoadTester',
    'StressTester',
    'SystemMonitor',
    'LoadTestConfig',
    'StressTestConfig',
    'PerformanceMetrics'
]
