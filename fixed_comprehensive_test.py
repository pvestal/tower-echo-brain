#!/usr/bin/env python3
"""
Fixed comprehensive test suite for Echo Brain with correct API endpoints.
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import sys

BASE_URL = "http://localhost:8309"

class EchoBrainTester:
    def __init__(self):
        self.results = []
        self.test_conversation_id = f"test_{int(time.time())}"

    async def test_health_check(self) -> Tuple[bool, float, Dict]:
        """Test basic health endpoint"""
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{BASE_URL}/health") as response:
                    duration = time.time() - start_time
                    data = await response.json()
                    success = response.status == 200 and data.get("status") == "healthy"
                    return success, duration, {"status": data.get("status")}
            except Exception as e:
                duration = time.time() - start_time
                return False, duration, {"error": str(e)}

    async def test_basic_query(self) -> Tuple[bool, float, Dict]:
        """Test basic query endpoint with correct format"""
        start_time = time.time()
        payload = {
            "query": "What is 2 plus 2?",
            "conversation_id": self.test_conversation_id
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{BASE_URL}/api/echo/query",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    duration = time.time() - start_time
                    if response.status == 200:
                        data = await response.json()
                        success = "response" in data and len(data["response"]) > 0
                        return success, duration, {
                            "response_length": len(data.get("response", "")),
                            "model": data.get("model_used", "unknown")
                        }
                    else:
                        text = await response.text()
                        return False, duration, {
                            "status_code": response.status,
                            "error": text[:100]
                        }
            except Exception as e:
                duration = time.time() - start_time
                return False, duration, {"error": str(e)}

    async def test_model_listing(self) -> Tuple[bool, float, Dict]:
        """Test model listing endpoint"""
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{BASE_URL}/api/models/list",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    duration = time.time() - start_time
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("models", [])
                        return len(models) > 0, duration, {"model_count": len(models)}
                    else:
                        return False, duration, {"status_code": response.status}
            except asyncio.TimeoutError:
                duration = time.time() - start_time
                return False, duration, {"error": "timeout"}
            except Exception as e:
                duration = time.time() - start_time
                return False, duration, {"error": str(e)}

    async def test_conversation_history(self) -> Tuple[bool, float, Dict]:
        """Test conversation history retrieval"""
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{BASE_URL}/api/echo/conversation/{self.test_conversation_id}"
                ) as response:
                    duration = time.time() - start_time
                    success = response.status == 200
                    if success:
                        data = await response.json()
                        return True, duration, {"message_count": len(data.get("messages", []))}
                    else:
                        return False, duration, {"status_code": response.status}
            except Exception as e:
                duration = time.time() - start_time
                return False, duration, {"error": str(e)}

    async def test_complex_query(self) -> Tuple[bool, float, Dict]:
        """Test complex reasoning query"""
        start_time = time.time()
        payload = {
            "query": "Explain the difference between machine learning and deep learning in simple terms",
            "conversation_id": self.test_conversation_id,
            "intelligence_level": "advanced"
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{BASE_URL}/api/echo/query",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    duration = time.time() - start_time
                    if response.status == 200:
                        data = await response.json()
                        success = len(data.get("response", "")) > 100
                        return success, duration, {
                            "response_length": len(data.get("response", "")),
                            "intelligence_level": data.get("intelligence_level", "unknown")
                        }
                    else:
                        return False, duration, {"status_code": response.status}
            except asyncio.TimeoutError:
                duration = time.time() - start_time
                return False, duration, {"error": "timeout"}
            except Exception as e:
                duration = time.time() - start_time
                return False, duration, {"error": str(e)}

    async def test_brain_visualization(self) -> Tuple[bool, float, Dict]:
        """Test brain visualization endpoint"""
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{BASE_URL}/api/echo/brain") as response:
                    duration = time.time() - start_time
                    if response.status == 200:
                        data = await response.json()
                        return True, duration, {"has_activity": "activity" in data}
                    else:
                        return False, duration, {"status_code": response.status}
            except Exception as e:
                duration = time.time() - start_time
                return False, duration, {"error": str(e)}

    async def test_multimedia_status(self) -> Tuple[bool, float, Dict]:
        """Test multimedia service status"""
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{BASE_URL}/api/multimedia/status") as response:
                    duration = time.time() - start_time
                    if response.status == 200:
                        data = await response.json()
                        return True, duration, {"status": data.get("status", "unknown")}
                    else:
                        return False, duration, {"status_code": response.status}
            except Exception as e:
                duration = time.time() - start_time
                return False, duration, {"error": str(e)}

    async def run_all_tests(self):
        """Run all tests and generate report"""
        print("\n" + "="*80)
        print("🧪 ECHO BRAIN COMPREHENSIVE TEST SUITE")
        print("="*80)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Base URL: {BASE_URL}")
        print("-"*80 + "\n")

        tests = [
            ("Health Check", self.test_health_check),
            ("Basic Query", self.test_basic_query),
            ("Model Listing", self.test_model_listing),
            ("Conversation History", self.test_conversation_history),
            ("Complex Query", self.test_complex_query),
            ("Brain Visualization", self.test_brain_visualization),
            ("Multimedia Status", self.test_multimedia_status),
        ]

        for test_name, test_func in tests:
            print(f"Running: {test_name}...", end=" ")
            try:
                success, duration, details = await test_func()
                status = "✅ passed" if success else "❌ failed"
                print(f"{status} ({duration:.2f}s)")

                self.results.append({
                    "name": test_name,
                    "status": "passed" if success else "failed",
                    "duration": duration,
                    "details": details
                })
            except Exception as e:
                print(f"❌ error: {e}")
                self.results.append({
                    "name": test_name,
                    "status": "error",
                    "duration": 0,
                    "details": {"error": str(e)}
                })

        # Generate summary
        print("\n" + "="*80)
        print("📊 TEST SUMMARY")
        print("-"*80)

        passed = sum(1 for r in self.results if r["status"] == "passed")
        failed = sum(1 for r in self.results if r["status"] == "failed")
        errors = sum(1 for r in self.results if r["status"] == "error")
        total = len(self.results)

        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"🔥 Errors: {errors}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")

        # Show failed tests details
        if failed > 0 or errors > 0:
            print("\n🚨 FAILED TESTS DETAILS")
            print("-"*80)
            for result in self.results:
                if result["status"] in ["failed", "error"]:
                    print(f"• {result['name']}: {result['details']}")

        # Save results
        with open("/opt/tower-echo-brain/test_results_fixed.json", "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total": total,
                    "passed": passed,
                    "failed": failed,
                    "errors": errors,
                    "success_rate": (passed/total)*100
                },
                "tests": self.results
            }, f, indent=2)

        print(f"\nResults saved to: /opt/tower-echo-brain/test_results_fixed.json")
        print("="*80)

        return passed == total

async def main():
    tester = EchoBrainTester()
    success = await tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())