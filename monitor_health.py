#!/usr/bin/env python3
"""
Echo Brain Health Monitor
Continuously monitors endpoint health and reports actual status
"""

import asyncio
import httpx
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

BASE_URL = "http://localhost:8309"

class EndpointHealthMonitor:
    def __init__(self):
        self.results = {}
        self.history = []

    async def check_endpoint(self, method: str, path: str, data: dict = None) -> Tuple[int, float]:
        """Check single endpoint health"""
        url = f"{BASE_URL}{path}"
        start = time.time()

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                if method == "GET":
                    response = await client.get(url)
                elif method == "POST":
                    response = await client.post(url, json=data or {})
                else:
                    return 0, 0

                elapsed = time.time() - start
                return response.status_code, elapsed
        except httpx.TimeoutException:
            return 408, 5.0  # Request Timeout
        except Exception:
            return 0, 0  # Connection failed

    async def scan_all_endpoints(self):
        """Scan all known endpoints"""
        # Critical endpoints to monitor
        endpoints = [
            ("GET", "/", None),
            ("GET", "/health", None),
            ("GET", "/api/status", None),
            ("POST", "/api/echo/query", {"query": "test"}),
            ("POST", "/api/echo/chat", {"query": "test"}),
            ("GET", "/api/echo/agents/status", None),
            ("GET", "/api/models/list", None),
            ("GET", "/api/autonomous/status", None),
            ("GET", "/api/echo/codebase/stats", None),
            ("GET", "/api/diagnostics/health", None),
            ("POST", "/api/context", {"query": "test"}),
            ("GET", "/api/collections", None),
            ("GET", "/api/coordination/services", None),
            ("GET", "/git/status", None),
        ]

        results = {}
        for method, path, data in endpoints:
            status, response_time = await self.check_endpoint(method, path, data)
            results[f"{method} {path}"] = {
                "status": status,
                "response_time": response_time,
                "healthy": 200 <= status < 400,
                "timestamp": datetime.now().isoformat()
            }

        return results

    def calculate_health_score(self, results: Dict) -> float:
        """Calculate overall health score (0-100)"""
        if not results:
            return 0

        healthy = sum(1 for r in results.values() if r.get("healthy", False))
        total = len(results)
        return (healthy / total) * 100

    def get_problem_endpoints(self, results: Dict) -> List[str]:
        """Get list of problematic endpoints"""
        problems = []
        for endpoint, data in results.items():
            if not data.get("healthy", False):
                problems.append(f"{endpoint}: {data.get('status', 0)}")
        return problems

    async def continuous_monitor(self, interval: int = 300):
        """Run continuous monitoring"""
        while True:
            print(f"\n[{datetime.now().isoformat()}] Starting health scan...")

            results = await self.scan_all_endpoints()
            health_score = self.calculate_health_score(results)
            problems = self.get_problem_endpoints(results)

            # Store results
            self.results = results
            self.history.append({
                "timestamp": datetime.now().isoformat(),
                "health_score": health_score,
                "total_endpoints": len(results),
                "healthy_endpoints": sum(1 for r in results.values() if r.get("healthy")),
                "problems": problems
            })

            # Print summary
            print(f"Health Score: {health_score:.1f}%")
            print(f"Healthy Endpoints: {sum(1 for r in results.values() if r.get('healthy'))}/{len(results)}")

            if problems:
                print(f"Problems detected:")
                for problem in problems[:5]:  # Show first 5 problems
                    print(f"  - {problem}")

            # Save to file
            self.save_results()

            # Alert if health drops below threshold
            if health_score < 50:
                print(f"⚠️ WARNING: Health score below 50%!")

            print(f"Next scan in {interval} seconds...")
            await asyncio.sleep(interval)

    def save_results(self):
        """Save monitoring results to file"""
        output = {
            "last_scan": datetime.now().isoformat(),
            "current_health_score": self.calculate_health_score(self.results),
            "endpoints": self.results,
            "recent_history": self.history[-10:]  # Keep last 10 scans
        }

        with open("/opt/tower-echo-brain/health_monitor.json", "w") as f:
            json.dump(output, f, indent=2)

async def main():
    """Main monitoring loop"""
    monitor = EndpointHealthMonitor()

    # Run initial scan
    print("Echo Brain Health Monitor Starting...")
    results = await monitor.scan_all_endpoints()
    health_score = monitor.calculate_health_score(results)

    print(f"\nInitial Health Check:")
    print(f"  Overall Health: {health_score:.1f}%")
    print(f"  Working Endpoints: {sum(1 for r in results.values() if r.get('healthy'))}/{len(results)}")

    problems = monitor.get_problem_endpoints(results)
    if problems:
        print(f"\n  Current Problems:")
        for problem in problems:
            print(f"    - {problem}")

    # Start continuous monitoring
    try:
        await monitor.continuous_monitor(interval=300)  # Check every 5 minutes
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    asyncio.run(main())