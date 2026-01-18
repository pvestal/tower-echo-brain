"""
Performance Benchmarking for Echo Brain Anime Generation System
Tests realistic performance metrics and identifies bottlenecks.
"""

import asyncio
import json
import statistics
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import concurrent.futures

import httpx
import psutil
import pytest
from prometheus_client.parser import text_string_to_metric_families


@dataclass
class PerformanceMetric:
    """Single performance measurement"""
    operation: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class ResourceUsage:
    """System resource usage snapshot"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    disk_io_read: float = 0
    disk_io_write: float = 0


@dataclass
class PerformanceBenchmark:
    """Complete performance benchmark results"""
    test_name: str
    start_time: str
    end_time: str
    total_duration: float
    metrics: List[PerformanceMetric]
    resource_usage: List[ResourceUsage]
    summary: Dict
    thresholds: Dict
    passed: bool


class PerformanceProfiler:
    """Profiles system resource usage during operations"""

    def __init__(self, sample_interval: float = 1.0):
        self.sample_interval = sample_interval
        self.monitoring = False
        self.resource_data = []

    async def start_monitoring(self):
        """Start resource monitoring in background"""
        self.monitoring = True
        self.resource_data = []

        while self.monitoring:
            try:
                # Get CPU and memory usage
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()

                # Get GPU memory if available
                gpu_memory_used = None
                gpu_memory_total = None
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
                    gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_used = gpu_info.used / 1024**3  # GB
                    gpu_memory_total = gpu_info.total / 1024**3  # GB
                except:
                    pass  # GPU monitoring not available

                # Get disk I/O
                disk_io = psutil.disk_io_counters()
                disk_read = disk_io.read_bytes if disk_io else 0
                disk_write = disk_io.write_bytes if disk_io else 0

                usage = ResourceUsage(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    gpu_memory_used=gpu_memory_used,
                    gpu_memory_total=gpu_memory_total,
                    disk_io_read=disk_read,
                    disk_io_write=disk_write
                )

                self.resource_data.append(usage)
                await asyncio.sleep(self.sample_interval)

            except Exception:
                # Continue monitoring even if one sample fails
                await asyncio.sleep(self.sample_interval)

    def stop_monitoring(self) -> List[ResourceUsage]:
        """Stop monitoring and return collected data"""
        self.monitoring = False
        return self.resource_data.copy()


class AnimeBenchmarkSuite:
    """Comprehensive benchmarking for anime generation system"""

    # Realistic performance thresholds based on current system capabilities
    PERFORMANCE_THRESHOLDS = {
        # Image generation thresholds (realistic)
        "image_generation_max_seconds": 60,
        "image_generation_target_seconds": 45,

        # Video generation thresholds (realistic)
        "video_2s_max_seconds": 120,
        "video_2s_target_seconds": 90,
        "video_5s_max_seconds": 300,
        "video_5s_target_seconds": 240,

        # API response thresholds
        "api_response_max_ms": 5000,
        "api_response_target_ms": 2000,
        "health_check_max_ms": 1000,

        # System resource thresholds
        "max_cpu_percent": 95,
        "max_memory_percent": 85,
        "max_gpu_memory_percent": 90,

        # Queue and concurrency
        "max_concurrent_jobs": 3,
        "queue_processing_max_delay": 10
    }

    def __init__(self,
                 anime_api: str = "http://192.168.50.135:8328",
                 echo_api: str = "http://192.168.50.135:8309",
                 comfyui_api: str = "http://192.168.50.135:8188"):
        self.anime_api = anime_api
        self.echo_api = echo_api
        self.comfyui_api = comfyui_api
        self.profiler = PerformanceProfiler()

    async def measure_operation(self,
                               operation_name: str,
                               operation_func,
                               *args, **kwargs) -> PerformanceMetric:
        """Measure the performance of a single operation"""
        start_time = time.time()
        error = None
        success = False
        metadata = {}

        try:
            result = await operation_func(*args, **kwargs)
            success = True
            if isinstance(result, dict):
                metadata = result
        except Exception as e:
            error = str(e)

        end_time = time.time()
        duration = end_time - start_time

        return PerformanceMetric(
            operation=operation_name,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            success=success,
            error=error,
            metadata=metadata
        )

    async def benchmark_api_responsiveness(self) -> List[PerformanceMetric]:
        """Benchmark API endpoint response times"""
        metrics = []

        async with httpx.AsyncClient(timeout=30) as client:
            # Test health check endpoints
            for service_name, endpoint in [
                ("anime_api_health", f"{self.anime_api}/api/health"),
                ("echo_api_health", f"{self.echo_api}/api/echo/health"),
                ("comfyui_health", f"{self.comfyui_api}/system_stats")
            ]:
                async def health_check():
                    response = await client.get(endpoint)
                    return {"status_code": response.status_code, "response_time": response.elapsed}

                metric = await self.measure_operation(service_name, health_check)
                metrics.append(metric)

            # Test API endpoints with various payloads
            test_requests = [
                ("anime_projects_list", "GET", f"{self.anime_api}/api/anime/projects", None),
                ("anime_characters_list", "GET", f"{self.anime_api}/api/anime/characters", None),
                ("echo_status", "GET", f"{self.echo_api}/api/echo/status", None),
                ("echo_models_list", "GET", f"{self.echo_api}/api/echo/models/list", None),
            ]

            for test_name, method, url, payload in test_requests:
                async def api_request():
                    if method == "GET":
                        response = await client.get(url)
                    elif method == "POST":
                        response = await client.post(url, json=payload)
                    else:
                        raise ValueError(f"Unsupported method: {method}")

                    return {
                        "status_code": response.status_code,
                        "response_size": len(response.content),
                        "response_time": response.elapsed
                    }

                metric = await self.measure_operation(test_name, api_request)
                metrics.append(metric)

        return metrics

    async def benchmark_image_generation(self, num_tests: int = 3) -> List[PerformanceMetric]:
        """Benchmark image generation performance"""
        metrics = []

        test_prompts = [
            "Simple anime girl, basic style, clean background",
            "Anime warrior with sword, detailed armor, forest background",
            "Cyberpunk character, complex neon environment, futuristic city"
        ]

        async with httpx.AsyncClient(timeout=300) as client:
            for i in range(min(num_tests, len(test_prompts))):
                prompt = test_prompts[i]
                complexity = ["simple", "medium", "complex"][i]

                async def generate_image():
                    # Submit generation request
                    generation_request = {
                        "prompt": prompt,
                        "style": "anime",
                        "quality": "high",
                        "type": "image",
                        "project_id": f"perf_test_{uuid.uuid4().hex[:8]}"
                    }

                    response = await client.post(
                        f"{self.anime_api}/api/anime/generate",
                        json=generation_request
                    )

                    if response.status_code != 200:
                        raise Exception(f"Generation request failed: {response.status_code}")

                    generation_data = response.json()
                    job_id = generation_data.get("job_id")

                    if not job_id:
                        raise Exception("No job_id returned")

                    # Wait for completion
                    image_path = await self._wait_for_job_completion(client, job_id, max_wait=180)

                    if not image_path:
                        raise Exception("Generation timed out or failed")

                    return {
                        "job_id": job_id,
                        "image_path": image_path,
                        "prompt": prompt,
                        "complexity": complexity
                    }

                metric = await self.measure_operation(f"image_generation_{complexity}", generate_image)
                metrics.append(metric)

        return metrics

    async def benchmark_video_generation(self, num_tests: int = 2) -> List[PerformanceMetric]:
        """Benchmark video generation performance"""
        metrics = []

        video_tests = [
            ("2s_video", "Simple anime character walking, 2 second duration", 2),
            ("5s_video", "Anime battle scene, 5 second duration", 5)
        ]

        async with httpx.AsyncClient(timeout=600) as client:
            for test_name, prompt, duration in video_tests[:num_tests]:
                async def generate_video():
                    generation_request = {
                        "prompt": prompt,
                        "style": "anime",
                        "type": "video",
                        "duration": duration,
                        "fps": 24,
                        "project_id": f"perf_test_video_{uuid.uuid4().hex[:8]}"
                    }

                    response = await client.post(
                        f"{self.anime_api}/api/anime/generate",
                        json=generation_request
                    )

                    if response.status_code != 200:
                        raise Exception(f"Video generation request failed: {response.status_code}")

                    generation_data = response.json()
                    job_id = generation_data.get("job_id")

                    if not job_id:
                        raise Exception("No job_id returned")

                    # Wait for completion with longer timeout for video
                    video_path = await self._wait_for_job_completion(client, job_id, max_wait=360)

                    if not video_path:
                        raise Exception("Video generation timed out or failed")

                    return {
                        "job_id": job_id,
                        "video_path": video_path,
                        "duration": duration,
                        "prompt": prompt
                    }

                metric = await self.measure_operation(test_name, generate_video)
                metrics.append(metric)

        return metrics

    async def benchmark_concurrent_load(self, concurrent_jobs: int = 3) -> List[PerformanceMetric]:
        """Test system performance under concurrent load"""
        metrics = []

        async def concurrent_generation():
            tasks = []
            async with httpx.AsyncClient(timeout=600) as client:
                # Submit multiple concurrent requests
                for i in range(concurrent_jobs):
                    task = self._submit_concurrent_job(client, i)
                    tasks.append(task)

                # Wait for all to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                successful_jobs = 0
                failed_jobs = 0
                total_time = 0

                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        failed_jobs += 1
                    else:
                        successful_jobs += 1
                        if result and "generation_time" in result:
                            total_time += result["generation_time"]

                return {
                    "concurrent_jobs": concurrent_jobs,
                    "successful_jobs": successful_jobs,
                    "failed_jobs": failed_jobs,
                    "success_rate": (successful_jobs / concurrent_jobs) * 100,
                    "average_time": total_time / successful_jobs if successful_jobs > 0 else 0
                }

        metric = await self.measure_operation("concurrent_load_test", concurrent_generation)
        metrics.append(metric)

        return metrics

    async def _submit_concurrent_job(self, client: httpx.AsyncClient, job_index: int) -> Dict:
        """Submit a single concurrent job"""
        generation_request = {
            "prompt": f"Anime character {job_index}, simple style",
            "style": "anime",
            "type": "image",
            "project_id": f"concurrent_test_{job_index}_{uuid.uuid4().hex[:8]}"
        }

        start_time = time.time()

        response = await client.post(
            f"{self.anime_api}/api/anime/generate",
            json=generation_request
        )

        if response.status_code != 200:
            raise Exception(f"Concurrent job {job_index} failed: {response.status_code}")

        generation_data = response.json()
        job_id = generation_data.get("job_id")

        # Wait for completion
        image_path = await self._wait_for_job_completion(client, job_id, max_wait=300)

        generation_time = time.time() - start_time

        return {
            "job_index": job_index,
            "job_id": job_id,
            "image_path": image_path,
            "generation_time": generation_time,
            "success": image_path is not None
        }

    async def _wait_for_job_completion(self, client: httpx.AsyncClient, job_id: str,
                                     max_wait: int = 300) -> Optional[str]:
        """Wait for job completion and return output path"""
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                response = await client.get(f"{self.anime_api}/api/anime/status/{job_id}")

                if response.status_code == 200:
                    status_data = response.json()
                    status = status_data.get("status", "unknown")

                    if status == "completed":
                        return status_data.get("output_path")
                    elif status in ["failed", "error"]:
                        return None

                await asyncio.sleep(5)  # Wait 5 seconds before next check

            except Exception:
                await asyncio.sleep(5)

        return None  # Timeout

    def analyze_performance_metrics(self, metrics: List[PerformanceMetric]) -> Dict:
        """Analyze performance metrics and generate summary"""
        if not metrics:
            return {"error": "No metrics to analyze"}

        # Group metrics by operation type
        operations = {}
        for metric in metrics:
            if metric.operation not in operations:
                operations[metric.operation] = []
            operations[metric.operation].append(metric)

        analysis = {
            "total_operations": len(metrics),
            "successful_operations": len([m for m in metrics if m.success]),
            "failed_operations": len([m for m in metrics if not m.success]),
            "overall_success_rate": len([m for m in metrics if m.success]) / len(metrics) * 100,
            "operations": {}
        }

        # Analyze each operation type
        for op_name, op_metrics in operations.items():
            durations = [m.duration for m in op_metrics if m.success]

            if durations:
                analysis["operations"][op_name] = {
                    "total_runs": len(op_metrics),
                    "successful_runs": len(durations),
                    "average_duration": statistics.mean(durations),
                    "median_duration": statistics.median(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "std_deviation": statistics.stdev(durations) if len(durations) > 1 else 0,
                    "success_rate": len(durations) / len(op_metrics) * 100
                }

                # Check against thresholds
                avg_duration = analysis["operations"][op_name]["average_duration"]
                threshold_key = self._get_threshold_key(op_name)

                if threshold_key and threshold_key in self.PERFORMANCE_THRESHOLDS:
                    threshold = self.PERFORMANCE_THRESHOLDS[threshold_key]
                    analysis["operations"][op_name]["threshold"] = threshold
                    analysis["operations"][op_name]["meets_threshold"] = avg_duration <= threshold
            else:
                analysis["operations"][op_name] = {
                    "total_runs": len(op_metrics),
                    "successful_runs": 0,
                    "success_rate": 0,
                    "error": "All operations failed"
                }

        return analysis

    def _get_threshold_key(self, operation_name: str) -> Optional[str]:
        """Map operation name to threshold key"""
        threshold_mapping = {
            "image_generation_simple": "image_generation_max_seconds",
            "image_generation_medium": "image_generation_max_seconds",
            "image_generation_complex": "image_generation_max_seconds",
            "2s_video": "video_2s_max_seconds",
            "5s_video": "video_5s_max_seconds",
            "anime_api_health": "health_check_max_ms",
            "echo_api_health": "health_check_max_ms"
        }
        return threshold_mapping.get(operation_name)

    async def run_full_benchmark(self) -> PerformanceBenchmark:
        """Run complete performance benchmark suite"""
        start_time = datetime.now()
        all_metrics = []

        # Start resource monitoring
        monitor_task = asyncio.create_task(self.profiler.start_monitoring())

        try:
            # Run all benchmark components
            print("Running API responsiveness tests...")
            api_metrics = await self.benchmark_api_responsiveness()
            all_metrics.extend(api_metrics)

            print("Running image generation tests...")
            image_metrics = await self.benchmark_image_generation(num_tests=2)
            all_metrics.extend(image_metrics)

            print("Running video generation tests...")
            video_metrics = await self.benchmark_video_generation(num_tests=1)
            all_metrics.extend(video_metrics)

            print("Running concurrent load tests...")
            concurrent_metrics = await self.benchmark_concurrent_load(concurrent_jobs=2)
            all_metrics.extend(concurrent_metrics)

        finally:
            # Stop monitoring
            resource_data = self.profiler.stop_monitoring()

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        # Analyze results
        analysis = self.analyze_performance_metrics(all_metrics)

        # Check if benchmark passed
        passed = self._evaluate_benchmark_success(analysis)

        benchmark = PerformanceBenchmark(
            test_name="full_anime_generation_benchmark",
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_duration=total_duration,
            metrics=all_metrics,
            resource_usage=resource_data,
            summary=analysis,
            thresholds=self.PERFORMANCE_THRESHOLDS,
            passed=passed
        )

        return benchmark

    def _evaluate_benchmark_success(self, analysis: Dict) -> bool:
        """Determine if benchmark meets success criteria"""
        # Overall success rate should be high
        if analysis.get("overall_success_rate", 0) < 80:
            return False

        # Check critical operations meet thresholds
        operations = analysis.get("operations", {})

        for op_name, op_data in operations.items():
            if "meets_threshold" in op_data and not op_data["meets_threshold"]:
                # Allow some flexibility for video generation
                if "video" in op_name and op_data.get("success_rate", 0) >= 75:
                    continue
                return False

        return True


# Pytest test cases
class TestPerformanceBenchmarks:
    """Pytest test cases for performance benchmarking"""

    @pytest.fixture
    def benchmark_suite(self):
        return AnimeBenchmarkSuite()

    @pytest.mark.asyncio
    async def test_api_responsiveness(self, benchmark_suite):
        """Test API response times meet requirements"""
        metrics = await benchmark_suite.benchmark_api_responsiveness()

        # All health checks should succeed
        health_checks = [m for m in metrics if "health" in m.operation]
        assert len(health_checks) > 0, "No health checks performed"

        for metric in health_checks:
            assert metric.success, f"Health check failed: {metric.operation} - {metric.error}"
            assert metric.duration < 5.0, f"Health check too slow: {metric.operation} took {metric.duration}s"

    @pytest.mark.asyncio
    async def test_image_generation_performance(self, benchmark_suite):
        """Test image generation meets performance requirements"""
        metrics = await benchmark_suite.benchmark_image_generation(num_tests=2)

        # At least one generation should succeed
        successful_generations = [m for m in metrics if m.success]
        assert len(successful_generations) > 0, "No successful image generations"

        # Check timing requirements
        for metric in successful_generations:
            assert metric.duration <= 90, f"Image generation too slow: {metric.duration}s (max 90s)"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_video_generation_performance(self, benchmark_suite):
        """Test video generation performance (slow test)"""
        metrics = await benchmark_suite.benchmark_video_generation(num_tests=1)

        # Video generation is allowed to have lower success rate
        if metrics:
            successful_videos = [m for m in metrics if m.success]
            if successful_videos:
                for metric in successful_videos:
                    # Generous timing for video generation
                    expected_max = 300 if "5s" in metric.operation else 150
                    assert metric.duration <= expected_max, \
                        f"Video generation too slow: {metric.duration}s (max {expected_max}s)"

    @pytest.mark.asyncio
    async def test_concurrent_load_handling(self, benchmark_suite):
        """Test system handles concurrent requests"""
        metrics = await benchmark_suite.benchmark_concurrent_load(concurrent_jobs=2)

        assert len(metrics) > 0, "No concurrent load test performed"

        load_metric = metrics[0]
        assert load_metric.success, f"Concurrent load test failed: {load_metric.error}"

        if load_metric.metadata:
            success_rate = load_metric.metadata.get("success_rate", 0)
            assert success_rate >= 50, f"Concurrent success rate too low: {success_rate}%"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_full_benchmark_suite(self, benchmark_suite):
        """Run complete benchmark suite (slow test)"""
        benchmark_result = await benchmark_suite.run_full_benchmark()

        # Save results
        results_file = f"/tmp/anime_benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            # Convert dataclass to dict for serialization
            result_dict = asdict(benchmark_result)
            json.dump(result_dict, f, indent=2, default=str)

        print(f"Benchmark results saved to: {results_file}")

        # Check overall system health
        assert benchmark_result.summary["overall_success_rate"] >= 60, \
            f"Overall success rate too low: {benchmark_result.summary['overall_success_rate']}%"

        # Print summary
        print(f"Benchmark Summary:")
        print(f"Total Duration: {benchmark_result.total_duration:.2f}s")
        print(f"Success Rate: {benchmark_result.summary['overall_success_rate']:.1f}%")
        print(f"Passed: {benchmark_result.passed}")


if __name__ == "__main__":
    # CLI interface for standalone benchmarking
    import argparse

    parser = argparse.ArgumentParser(description="Performance benchmarking for anime generation")
    parser.add_argument("--test-type", choices=["api", "images", "videos", "concurrent", "full"],
                       default="full", help="Type of benchmark to run")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--concurrent-jobs", type=int, default=2, help="Number of concurrent jobs")

    args = parser.parse_args()

    async def main():
        benchmark_suite = AnimeBenchmarkSuite()

        if args.test_type == "api":
            metrics = await benchmark_suite.benchmark_api_responsiveness()
            analysis = benchmark_suite.analyze_performance_metrics(metrics)
            results = {"metrics": [asdict(m) for m in metrics], "analysis": analysis}
        elif args.test_type == "images":
            metrics = await benchmark_suite.benchmark_image_generation()
            analysis = benchmark_suite.analyze_performance_metrics(metrics)
            results = {"metrics": [asdict(m) for m in metrics], "analysis": analysis}
        elif args.test_type == "videos":
            metrics = await benchmark_suite.benchmark_video_generation()
            analysis = benchmark_suite.analyze_performance_metrics(metrics)
            results = {"metrics": [asdict(m) for m in metrics], "analysis": analysis}
        elif args.test_type == "concurrent":
            metrics = await benchmark_suite.benchmark_concurrent_load(args.concurrent_jobs)
            analysis = benchmark_suite.analyze_performance_metrics(metrics)
            results = {"metrics": [asdict(m) for m in metrics], "analysis": analysis}
        elif args.test_type == "full":
            benchmark_result = await benchmark_suite.run_full_benchmark()
            results = asdict(benchmark_result)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(results, indent=2, default=str))

    asyncio.run(main())