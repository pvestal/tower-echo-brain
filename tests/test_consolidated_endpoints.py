#!/usr/bin/env python3
"""
Comprehensive test suite for ALL consolidated endpoints
Tests EVERY SINGLE endpoint and reports results
"""

import asyncio
import httpx
import json
from datetime import datetime
from typing import Dict, List, Tuple

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

class EndpointTester:
    def __init__(self):
        self.results = []
        self.summary = {"passed": 0, "failed": 0, "total": 0}

    async def test_endpoint(self, method: str, url: str, data: Dict = None, description: str = ""):
        """Test a single endpoint"""
        self.summary["total"] += 1

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                if method == "GET":
                    response = await client.get(url)
                elif method == "POST":
                    response = await client.post(url, json=data or {})
                elif method == "PUT":
                    response = await client.put(url, json=data or {})
                elif method == "DELETE":
                    response = await client.delete(url)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                # Check status code
                success = response.status_code in [200, 201, 204]

                # Try to parse JSON response
                try:
                    response_data = response.json()
                except:
                    response_data = response.text[:100]

                result = {
                    "method": method,
                    "url": url,
                    "status": response.status_code,
                    "success": success,
                    "description": description,
                    "response": response_data
                }

                if success:
                    self.summary["passed"] += 1
                    print(f"{GREEN}✓{RESET} {method:6} {url:60} [{response.status_code}] {description}")
                else:
                    self.summary["failed"] += 1
                    print(f"{RED}✗{RESET} {method:6} {url:60} [{response.status_code}] {description}")
                    if response.status_code == 404:
                        print(f"  {YELLOW}→ Endpoint not found{RESET}")
                    elif response.status_code >= 500:
                        print(f"  {YELLOW}→ Server error: {response_data}{RESET}")

                self.results.append(result)
                return result

        except httpx.RequestError as e:
            self.summary["failed"] += 1
            print(f"{RED}✗{RESET} {method:6} {url:60} [ERROR] {description}")
            print(f"  {YELLOW}→ Connection error: {str(e)[:50]}{RESET}")

            self.results.append({
                "method": method,
                "url": url,
                "status": "ERROR",
                "success": False,
                "description": description,
                "error": str(e)
            })
            return None

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 80)
        print(f"{BLUE}TEST SUMMARY{RESET}")
        print("=" * 80)
        print(f"Total Tests: {self.summary['total']}")
        print(f"{GREEN}Passed: {self.summary['passed']}{RESET}")
        print(f"{RED}Failed: {self.summary['failed']}{RESET}")

        success_rate = (self.summary['passed'] / self.summary['total'] * 100) if self.summary['total'] > 0 else 0
        color = GREEN if success_rate >= 80 else YELLOW if success_rate >= 50 else RED
        print(f"{color}Success Rate: {success_rate:.1f}%{RESET}")

        # List failed endpoints
        if self.summary["failed"] > 0:
            print(f"\n{RED}FAILED ENDPOINTS:{RESET}")
            for result in self.results:
                if not result.get("success", False):
                    print(f"  - {result['method']} {result['url']}: {result.get('status', 'ERROR')}")

async def test_echo_brain_endpoints():
    """Test ALL Echo Brain consolidated endpoints"""
    print(f"\n{BLUE}{'=' * 80}{RESET}")
    print(f"{BLUE}TESTING ECHO BRAIN CONSOLIDATED API (port 8309){RESET}")
    print(f"{BLUE}{'=' * 80}{RESET}\n")

    tester = EndpointTester()
    base_url = "http://localhost:8309"

    # System endpoints (from system_router.py)
    print(f"\n{YELLOW}=== SYSTEM ENDPOINTS ==={RESET}")
    await tester.test_endpoint("GET", f"{base_url}/", description="Root endpoint")
    await tester.test_endpoint("GET", f"{base_url}/health", description="Health check")
    await tester.test_endpoint("GET", f"{base_url}/ready", description="Readiness check")
    await tester.test_endpoint("GET", f"{base_url}/alive", description="Liveness check")
    await tester.test_endpoint("GET", f"{base_url}/metrics", description="System metrics")
    await tester.test_endpoint("GET", f"{base_url}/metrics/history", description="Metrics history")
    await tester.test_endpoint("GET", f"{base_url}/diagnostics", description="System diagnostics")
    await tester.test_endpoint("GET", f"{base_url}/diagnostics/database", description="Database diagnostics")
    await tester.test_endpoint("GET", f"{base_url}/diagnostics/services", description="Services diagnostics")
    await tester.test_endpoint("GET", f"{base_url}/status", description="System status")
    await tester.test_endpoint("GET", f"{base_url}/status/logs", description="Recent logs")

    # Core endpoints (from core_router.py)
    print(f"\n{YELLOW}=== CORE AI ENDPOINTS ==={RESET}")
    await tester.test_endpoint("POST", f"{base_url}/api/echo/chat",
                              {"query": "Test query", "model": "llama3.1:8b"},
                              description="Chat endpoint")
    await tester.test_endpoint("POST", f"{base_url}/api/echo/query",
                              {"query": "Test query"},
                              description="Query endpoint")
    await tester.test_endpoint("GET", f"{base_url}/api/echo/thoughts/recent", description="Recent thoughts")
    await tester.test_endpoint("GET", f"{base_url}/api/echo/thoughts/test-id", description="Get thought by ID")
    await tester.test_endpoint("GET", f"{base_url}/api/echo/models", description="List models")
    await tester.test_endpoint("GET", f"{base_url}/api/echo/models/llama3.1:8b", description="Get model info")
    await tester.test_endpoint("POST", f"{base_url}/api/echo/models/test-model/load", description="Load model")
    await tester.test_endpoint("POST", f"{base_url}/api/echo/agents/execute",
                              {"task": "Test task", "agent_type": "general"},
                              description="Execute agent")
    await tester.test_endpoint("GET", f"{base_url}/api/echo/agents", description="List agents")
    await tester.test_endpoint("GET", f"{base_url}/api/echo/agents/test-id/status", description="Agent status")
    await tester.test_endpoint("GET", f"{base_url}/api/echo/conversations", description="List conversations")
    await tester.test_endpoint("GET", f"{base_url}/api/echo/conversation/test-id", description="Get conversation")
    await tester.test_endpoint("GET", f"{base_url}/api/echo/status", description="Echo status")

    # Production endpoints (from production_router.py)
    print(f"\n{YELLOW}=== PRODUCTION ENDPOINTS ==={RESET}")
    await tester.test_endpoint("POST", f"{base_url}/api/anime/generate",
                              {"prompt": "Test anime generation"},
                              description="Generate anime")
    await tester.test_endpoint("GET", f"{base_url}/api/anime/job/test-id", description="Get anime job status")
    await tester.test_endpoint("GET", f"{base_url}/api/anime/projects", description="List anime projects")
    await tester.test_endpoint("GET", f"{base_url}/api/anime/characters", description="List anime characters")
    await tester.test_endpoint("POST", f"{base_url}/api/lora/train",
                              {"dataset_path": "/tmp/test", "model_name": "test"},
                              description="Start LoRA training")
    await tester.test_endpoint("GET", f"{base_url}/api/lora/training/test-id", description="Get training status")
    await tester.test_endpoint("GET", f"{base_url}/api/lora/models", description="List LoRA models")
    await tester.test_endpoint("POST", f"{base_url}/api/video/workflow",
                              {"workflow_type": "test", "input_text": "Test"},
                              description="Execute video workflow")
    await tester.test_endpoint("GET", f"{base_url}/api/video/workflow/test-id", description="Get workflow status")
    await tester.test_endpoint("GET", f"{base_url}/api/video/templates", description="List video templates")

    # Integration endpoints (from integration_router.py)
    print(f"\n{YELLOW}=== INTEGRATION ENDPOINTS ==={RESET}")
    await tester.test_endpoint("GET", f"{base_url}/api/google/auth/status", description="Google auth status")
    await tester.test_endpoint("POST", f"{base_url}/api/google/auth/authenticate",
                              {"service": "gmail"},
                              description="Google authenticate")
    await tester.test_endpoint("GET", f"{base_url}/api/google/gmail/messages", description="Get Gmail messages")
    await tester.test_endpoint("GET", f"{base_url}/api/google/calendar/events", description="Get calendar events")
    await tester.test_endpoint("POST", f"{base_url}/api/google/calendar/event",
                              {"summary": "Test", "start_time": "2024-01-01T10:00:00", "end_time": "2024-01-01T11:00:00"},
                              description="Create calendar event")
    await tester.test_endpoint("GET", f"{base_url}/api/google/photos/albums", description="Get photo albums")
    await tester.test_endpoint("GET", f"{base_url}/api/vault/status", description="Vault status")
    await tester.test_endpoint("GET", f"{base_url}/api/vault/secret/test", description="Get vault secret")
    await tester.test_endpoint("POST", f"{base_url}/api/vault/secret",
                              {"path": "test", "key": "test", "value": "test"},
                              description="Store vault secret")
    await tester.test_endpoint("GET", f"{base_url}/api/vault/credentials/list", description="List credentials")
    await tester.test_endpoint("POST", f"{base_url}/api/knowledge/article",
                              {"title": "Test", "content": "Test content"},
                              description="Create knowledge article")
    await tester.test_endpoint("POST", f"{base_url}/api/knowledge/search",
                              {"query": "test"},
                              description="Search knowledge")
    await tester.test_endpoint("GET", f"{base_url}/api/knowledge/article/test-id", description="Get knowledge article")
    await tester.test_endpoint("GET", f"{base_url}/api/knowledge/categories", description="Get knowledge categories")
    await tester.test_endpoint("GET", f"{base_url}/api/preferences/test-user", description="Get user preferences")
    await tester.test_endpoint("PUT", f"{base_url}/api/preferences/test-user",
                              {"theme": "dark"},
                              description="Update preferences")

    tester.print_summary()
    return tester.results

async def test_anime_production_endpoints():
    """Test ALL Anime Production consolidated endpoints"""
    print(f"\n{BLUE}{'=' * 80}{RESET}")
    print(f"{BLUE}TESTING ANIME PRODUCTION CONSOLIDATED API (port 8328){RESET}")
    print(f"{BLUE}{'=' * 80}{RESET}\n")

    tester = EndpointTester()
    base_url = "http://localhost:8328"

    # System endpoints
    print(f"\n{YELLOW}=== SYSTEM ENDPOINTS ==={RESET}")
    await tester.test_endpoint("GET", f"{base_url}/", description="Root endpoint")
    await tester.test_endpoint("GET", f"{base_url}/health", description="Health check")

    # Generation endpoints
    print(f"\n{YELLOW}=== GENERATION ENDPOINTS ==={RESET}")
    await tester.test_endpoint("POST", f"{base_url}/api/anime/generate",
                              {"prompt": "Test generation"},
                              description="Generate anime")
    await tester.test_endpoint("GET", f"{base_url}/api/anime/jobs/test-id/status", description="Get job status")

    # Project endpoints
    print(f"\n{YELLOW}=== PROJECT ENDPOINTS ==={RESET}")
    await tester.test_endpoint("GET", f"{base_url}/api/anime/projects", description="List projects")
    await tester.test_endpoint("POST", f"{base_url}/api/anime/projects",
                              {"name": "Test Project", "description": "Test"},
                              description="Create project")

    # Episode endpoints
    print(f"\n{YELLOW}=== EPISODE ENDPOINTS ==={RESET}")
    await tester.test_endpoint("GET", f"{base_url}/api/anime/episodes", description="List episodes")
    await tester.test_endpoint("POST", f"{base_url}/api/anime/episodes?project_id=test",
                              {"title": "Test Episode", "description": "Test", "scenes": []},
                              description="Create episode")

    # Character endpoints
    print(f"\n{YELLOW}=== CHARACTER ENDPOINTS ==={RESET}")
    await tester.test_endpoint("GET", f"{base_url}/api/anime/characters", description="List characters")

    # LoRA endpoints
    print(f"\n{YELLOW}=== LORA ENDPOINTS ==={RESET}")
    await tester.test_endpoint("POST", f"{base_url}/api/lora/train",
                              {"name": "Test LoRA", "dataset_path": "/tmp/test"},
                              description="Start LoRA training")
    await tester.test_endpoint("GET", f"{base_url}/api/lora/training/test-id", description="Get training status")
    await tester.test_endpoint("GET", f"{base_url}/api/lora/models", description="List LoRA models")

    # Workflow endpoints
    print(f"\n{YELLOW}=== WORKFLOW ENDPOINTS ==={RESET}")
    await tester.test_endpoint("GET", f"{base_url}/api/video/workflows", description="List workflows")

    # File endpoints
    print(f"\n{YELLOW}=== FILE ENDPOINTS ==={RESET}")
    await tester.test_endpoint("GET", f"{base_url}/api/video/download/test.mp4", description="Download video")

    # Echo Brain integration
    print(f"\n{YELLOW}=== ECHO BRAIN INTEGRATION ==={RESET}")
    await tester.test_endpoint("GET", f"{base_url}/api/echo-brain/status", description="Echo Brain status")
    await tester.test_endpoint("POST", f"{base_url}/api/anime/projects/test-id/echo-suggest?prompt=test",
                              description="Echo suggest content")

    tester.print_summary()
    return tester.results

async def main():
    """Run all tests"""
    print(f"{BLUE}{'=' * 80}{RESET}")
    print(f"{BLUE}COMPREHENSIVE ENDPOINT TESTING - ALL CONSOLIDATED APIs{RESET}")
    print(f"{BLUE}Testing started at: {datetime.now().isoformat()}{RESET}")
    print(f"{BLUE}{'=' * 80}{RESET}")

    # Test Echo Brain
    echo_results = await test_echo_brain_endpoints()

    # Test Anime Production
    anime_results = await test_anime_production_endpoints()

    # Overall summary
    print(f"\n{BLUE}{'=' * 80}{RESET}")
    print(f"{BLUE}OVERALL TEST RESULTS{RESET}")
    print(f"{BLUE}{'=' * 80}{RESET}")

    total_tests = len(echo_results) + len(anime_results)
    total_passed = sum(1 for r in echo_results + anime_results if r.get("success", False))
    total_failed = total_tests - total_passed

    print(f"Total Tests Run: {total_tests}")
    print(f"{GREEN}Total Passed: {total_passed}{RESET}")
    print(f"{RED}Total Failed: {total_failed}{RESET}")

    overall_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    color = GREEN if overall_rate >= 80 else YELLOW if overall_rate >= 50 else RED
    print(f"{color}Overall Success Rate: {overall_rate:.1f}%{RESET}")

    print(f"\n{BLUE}Testing completed at: {datetime.now().isoformat()}{RESET}")

if __name__ == "__main__":
    asyncio.run(main())