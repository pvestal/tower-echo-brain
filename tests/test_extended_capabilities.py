#!/usr/bin/env python3
"""
Extended Echo Brain Capability Tests
Additional test coverage for integration endpoints, knowledge management, and more
"""

import asyncio
import json
import time
import subprocess
import psycopg2
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import random
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExtendedEchoBrainTests:
    """Extended test suite for Echo Brain integrations and advanced features"""

    def __init__(self):
        self.base_url = "http://localhost:8309"
        self.db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "echo_brain",
            "user": "patrick",
            "password": "RP78eIrW7cI2jYvL5akt1yurE"
        }
        self.test_results = []
        self.session = None

    async def setup(self):
        """Setup test environment"""
        self.session = aiohttp.ClientSession()
        logger.info("🔧 Extended test environment initialized")

    async def teardown(self):
        """Cleanup test environment"""
        if self.session:
            await self.session.close()
        logger.info("🧹 Extended test cleanup complete")

    # TEST 1: Knowledge Management API
    async def test_knowledge_management(self) -> Dict[str, Any]:
        """Test knowledge management CRUD operations"""
        test_name = "Knowledge Management API"
        logger.info(f"🧪 Testing: {test_name}")

        try:
            # Try to get knowledge items
            async with self.session.get(
                f"{self.base_url}/api/knowledge/items",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                status = response.status
                if status == 200:
                    data = await response.json()
                    has_data = isinstance(data, (list, dict))
                else:
                    has_data = False
                    data = None

            # Try to create a knowledge item
            test_item = {
                "title": f"Test Knowledge {int(time.time())}",
                "content": "Test content for Echo Brain knowledge base",
                "tags": ["test", "automated"],
                "user_id": "test_user"
            }

            can_create = False
            if status != 404:
                async with self.session.post(
                    f"{self.base_url}/api/knowledge/items",
                    json=test_item,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    can_create = response.status in [200, 201]

            result = {
                "test": test_name,
                "passed": status != 404,
                "endpoint_exists": status != 404,
                "can_read": has_data,
                "can_create": can_create,
                "status_code": status
            }

        except Exception as e:
            result = {
                "test": test_name,
                "passed": False,
                "error": str(e)
            }

        self.test_results.append(result)
        return result

    # TEST 2: Preferences API
    async def test_preferences_management(self) -> Dict[str, Any]:
        """Test user preferences management"""
        test_name = "Preferences Management API"
        logger.info(f"🧪 Testing: {test_name}")

        try:
            # Get preference profile
            async with self.session.get(
                f"{self.base_url}/api/preferences/profile",
                params={"user_id": "test_user"},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                status = response.status
                has_endpoint = status != 404

            # Try to get templates
            template_exists = False
            if has_endpoint:
                async with self.session.get(
                    f"{self.base_url}/api/preferences/templates",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    template_exists = response.status == 200

            result = {
                "test": test_name,
                "passed": has_endpoint,
                "endpoint_exists": has_endpoint,
                "templates_available": template_exists,
                "status_code": status
            }

        except Exception as e:
            result = {
                "test": test_name,
                "passed": False,
                "error": str(e)
            }

        self.test_results.append(result)
        return result

    # TEST 3: Vault Integration
    async def test_vault_integration(self) -> Dict[str, Any]:
        """Test HashiCorp Vault integration"""
        test_name = "Vault Integration"
        logger.info(f"🧪 Testing: {test_name}")

        try:
            # Test vault get endpoint
            async with self.session.get(
                f"{self.base_url}/api/vault/get/test/path",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                status = response.status
                has_endpoint = status != 404

                # Status 403 or 401 means endpoint exists but needs auth
                auth_required = status in [401, 403]

            result = {
                "test": test_name,
                "passed": has_endpoint,
                "endpoint_exists": has_endpoint,
                "auth_required": auth_required,
                "status_code": status
            }

        except Exception as e:
            result = {
                "test": test_name,
                "passed": False,
                "error": str(e)
            }

        self.test_results.append(result)
        return result

    # TEST 4: Agent Coordination
    async def test_agent_coordination(self) -> Dict[str, Any]:
        """Test agent coordination endpoints"""
        test_name = "Agent Coordination"
        logger.info(f"🧪 Testing: {test_name}")

        try:
            # Test coordination services
            async with self.session.get(
                f"{self.base_url}/api/coordination/services",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                coord_status = response.status
                coord_exists = coord_status != 404

            # Test theater agents
            async with self.session.get(
                f"{self.base_url}/api/theater/agents",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                theater_status = response.status
                theater_exists = theater_status != 404

            result = {
                "test": test_name,
                "passed": coord_exists or theater_exists,
                "coordination_endpoint": coord_exists,
                "theater_endpoint": theater_exists,
                "coord_status": coord_status,
                "theater_status": theater_status
            }

        except Exception as e:
            result = {
                "test": test_name,
                "passed": False,
                "error": str(e)
            }

        self.test_results.append(result)
        return result

    # TEST 5: Git Operations
    async def test_git_operations(self) -> Dict[str, Any]:
        """Test git operations endpoints"""
        test_name = "Git Operations"
        logger.info(f"🧪 Testing: {test_name}")

        try:
            # Test git status endpoint
            async with self.session.get(
                f"{self.base_url}/api/echo/git/status",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                status = response.status
                if status == 200:
                    data = await response.json()
                    has_commits = "commits" in data or "recent_commits" in data
                    has_branch = "branch" in data or "current_branch" in data
                else:
                    has_commits = False
                    has_branch = False

            result = {
                "test": test_name,
                "passed": status == 200,
                "endpoint_works": status == 200,
                "has_commit_info": has_commits,
                "has_branch_info": has_branch,
                "status_code": status
            }

        except Exception as e:
            result = {
                "test": test_name,
                "passed": False,
                "error": str(e)
            }

        self.test_results.append(result)
        return result

    # TEST 6: Solutions API
    async def test_solutions_api(self) -> Dict[str, Any]:
        """Test solutions generation API"""
        test_name = "Solutions API"
        logger.info(f"🧪 Testing: {test_name}")

        try:
            # Test solutions endpoint
            test_request = {
                "problem": "How to optimize database queries",
                "context": "PostgreSQL database with slow queries",
                "user_id": "test_user"
            }

            async with self.session.post(
                f"{self.base_url}/api/solutions/generate",
                json=test_request,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                status = response.status
                endpoint_exists = status != 404

            result = {
                "test": test_name,
                "passed": endpoint_exists,
                "endpoint_exists": endpoint_exists,
                "status_code": status
            }

        except Exception as e:
            result = {
                "test": test_name,
                "passed": False,
                "error": str(e)
            }

        self.test_results.append(result)
        return result

    # TEST 7: Delegation System
    async def test_delegation_system(self) -> Dict[str, Any]:
        """Test task delegation system"""
        test_name = "Delegation System"
        logger.info(f"🧪 Testing: {test_name}")

        try:
            # Test delegation endpoint
            delegation_request = {
                "task": "Analyze code quality",
                "target": "src/main.py",
                "user_id": "test_user"
            }

            async with self.session.post(
                f"{self.base_url}/api/echo/delegation/delegate",
                json=delegation_request,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                status = response.status
                endpoint_exists = status != 404

            result = {
                "test": test_name,
                "passed": endpoint_exists,
                "endpoint_exists": endpoint_exists,
                "status_code": status
            }

        except Exception as e:
            result = {
                "test": test_name,
                "passed": False,
                "error": str(e)
            }

        self.test_results.append(result)
        return result

    # TEST 8: Multi-Turn Conversation
    async def test_multi_turn_conversation(self) -> Dict[str, Any]:
        """Test multi-turn conversation with context retention"""
        test_name = "Multi-Turn Conversation"
        logger.info(f"🧪 Testing: {test_name}")

        conversation_id = f"multi_turn_{int(time.time())}"

        try:
            # First turn: Introduce context
            response1 = await self._make_echo_request(
                "I'm working on optimizing the Tower Dashboard performance",
                conversation_id
            )

            # Second turn: Reference previous context
            response2 = await self._make_echo_request(
                "What specific areas should I focus on?",
                conversation_id
            )

            # Third turn: Action request based on context
            response3 = await self._make_echo_request(
                "Can you check if the dashboard service is running efficiently?",
                conversation_id
            )

            # Check database for all turns
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM echo_unified_interactions
                WHERE conversation_id = %s
            """, (conversation_id,))
            turn_count = cursor.fetchone()[0]
            cursor.close()
            conn.close()

            # Check if context was maintained
            context_maintained = (
                turn_count >= 3 and
                len(response2.get("response", "")) > 20 and
                len(response3.get("response", "")) > 20
            )

            result = {
                "test": test_name,
                "passed": context_maintained,
                "turns_recorded": turn_count,
                "expected_turns": 3,
                "context_maintained": context_maintained,
                "final_response_preview": response3.get("response", "")[:100]
            }

        except Exception as e:
            result = {
                "test": test_name,
                "passed": False,
                "error": str(e)
            }

        self.test_results.append(result)
        return result

    # TEST 9: Concurrent Capability Execution
    async def test_concurrent_capabilities(self) -> Dict[str, Any]:
        """Test concurrent execution of multiple capabilities"""
        test_name = "Concurrent Capability Execution"
        logger.info(f"🧪 Testing: {test_name}")

        try:
            # Launch multiple capability requests simultaneously
            tasks = [
                self._make_echo_request(
                    "Send notification: Test concurrent execution",
                    f"concurrent_1_{int(time.time())}"
                ),
                self._make_echo_request(
                    "Check system health",
                    f"concurrent_2_{int(time.time())}"
                ),
                self._make_echo_request(
                    "What services are running?",
                    f"concurrent_3_{int(time.time())}"
                )
            ]

            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            execution_time = time.time() - start_time

            # Check if all completed without exceptions
            successful = sum(1 for r in results if isinstance(r, dict) and not r.get("error"))

            # Concurrent execution should be faster than sequential (< 5 seconds total)
            is_concurrent = execution_time < 5

            result = {
                "test": test_name,
                "passed": successful >= 2 and is_concurrent,
                "successful_requests": successful,
                "total_requests": len(tasks),
                "execution_time": execution_time,
                "is_concurrent": is_concurrent
            }

        except Exception as e:
            result = {
                "test": test_name,
                "passed": False,
                "error": str(e)
            }

        self.test_results.append(result)
        return result

    # TEST 10: Error Recovery
    async def test_error_recovery(self) -> Dict[str, Any]:
        """Test system's ability to recover from errors"""
        test_name = "Error Recovery"
        logger.info(f"🧪 Testing: {test_name}")

        try:
            # Send invalid request
            invalid_response = await self._make_echo_request(
                "restart nonexistent-service-xyz",
                f"error_test_{int(time.time())}"
            )

            # Check if system handles error gracefully
            has_error_handling = (
                "error" in invalid_response.get("response", "").lower() or
                "failed" in invalid_response.get("response", "").lower() or
                not invalid_response.get("reasoning", {}).get("success", True)
            )

            # Send valid request after error
            valid_response = await self._make_echo_request(
                "What is your status?",
                f"recovery_test_{int(time.time())}"
            )

            # System should still respond normally
            recovered = len(valid_response.get("response", "")) > 0

            result = {
                "test": test_name,
                "passed": has_error_handling and recovered,
                "handles_errors": has_error_handling,
                "recovers_properly": recovered,
                "error_response_preview": invalid_response.get("response", "")[:100],
                "recovery_response_preview": valid_response.get("response", "")[:100]
            }

        except Exception as e:
            result = {
                "test": test_name,
                "passed": False,
                "error": str(e)
            }

        self.test_results.append(result)
        return result

    # TEST 11: Performance Under Load
    async def test_performance_under_load(self) -> Dict[str, Any]:
        """Test system performance under load"""
        test_name = "Performance Under Load"
        logger.info(f"🧪 Testing: {test_name}")

        try:
            # Send 10 requests rapidly
            tasks = []
            for i in range(10):
                tasks.append(
                    self._make_echo_request(
                        f"Test request {i}",
                        f"load_test_{int(time.time())}_{i}"
                    )
                )

            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time

            # Calculate success rate and average response time
            successful = [r for r in results if isinstance(r, dict) and not r.get("error")]
            success_rate = len(successful) / len(tasks)
            avg_time = total_time / len(tasks)

            # Performance criteria: >80% success, <2s average response
            good_performance = success_rate > 0.8 and avg_time < 2

            result = {
                "test": test_name,
                "passed": good_performance,
                "success_rate": success_rate,
                "average_response_time": avg_time,
                "total_time": total_time,
                "requests_sent": len(tasks),
                "successful_responses": len(successful)
            }

        except Exception as e:
            result = {
                "test": test_name,
                "passed": False,
                "error": str(e)
            }

        self.test_results.append(result)
        return result

    # TEST 12: Integration Availability Check
    async def test_integration_availability(self) -> Dict[str, Any]:
        """Check which integrations are actually available"""
        test_name = "Integration Availability"
        logger.info(f"🧪 Testing: {test_name}")

        integrations = {
            "google_calendar": "/api/google-calendar/status",
            "home_assistant": "/api/home-assistant/status",
            "telegram": "/api/telegram/status",
            "notifications": "/api/notifications/status"
        }

        available = {}
        for name, endpoint in integrations.items():
            try:
                async with self.session.get(
                    f"{self.base_url}{endpoint}",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    available[name] = response.status != 404
            except:
                available[name] = False

        result = {
            "test": test_name,
            "passed": any(available.values()),
            "integrations": available,
            "total_available": sum(available.values()),
            "total_checked": len(integrations)
        }

        self.test_results.append(result)
        return result

    async def _make_echo_request(self, query: str, conversation_id: str) -> Dict:
        """Helper to make Echo Brain chat requests"""
        payload = {
            "query": query,
            "user_id": "test_user",
            "conversation_id": conversation_id
        }

        try:
            async with self.session.post(
                f"{self.base_url}/api/echo/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                return await response.json()
        except Exception as e:
            return {"error": str(e)}

    async def run_all_tests(self):
        """Run all extended tests"""
        logger.info("🚀 Starting Extended Echo Brain Tests")
        logger.info("=" * 60)

        await self.setup()

        # Run all extended tests
        tests = [
            self.test_knowledge_management(),
            self.test_preferences_management(),
            self.test_vault_integration(),
            self.test_agent_coordination(),
            self.test_git_operations(),
            self.test_solutions_api(),
            self.test_delegation_system(),
            self.test_multi_turn_conversation(),
            self.test_concurrent_capabilities(),
            self.test_error_recovery(),
            self.test_performance_under_load(),
            self.test_integration_availability()
        ]

        results = await asyncio.gather(*tests, return_exceptions=True)

        # Process results
        passed_tests = sum(1 for r in results if isinstance(r, dict) and r.get("passed", False))
        total_tests = len(tests)

        logger.info("=" * 60)
        logger.info(f"📊 Extended Test Results: {passed_tests}/{total_tests} passed")

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"❌ Test failed with exception: {result}")
            elif isinstance(result, dict):
                status = "✅" if result.get("passed") else "❌"
                logger.info(f"{status} {result.get('test', 'Unknown')}: {result.get('passed', False)}")
                if not result.get("passed") and result.get("test") != "Integration Availability":
                    logger.info(f"   Details: {json.dumps(result, indent=2)}")

        await self.teardown()

        # Save results
        timestamp = datetime.now().isoformat()
        summary = {
            "timestamp": timestamp,
            "test_type": "extended",
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "test_results": self.test_results
        }

        # Save to file
        with open("tests/extended_test_results.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Log to database
        self.log_test_run_to_database(summary)

        return summary

    def log_test_run_to_database(self, summary: Dict):
        """Log test run results to database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Insert test run
            cursor.execute("""
                INSERT INTO echo_test_runs
                (test_run_id, timestamp, total_tests, passed, failed, success_rate, test_type, details)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                f"extended_test_{int(time.time())}",
                summary["timestamp"],
                summary["total_tests"],
                summary["passed"],
                summary["failed"],
                summary["success_rate"],
                "extended_capabilities",
                json.dumps(summary["test_results"])
            ))

            conn.commit()
            cursor.close()
            conn.close()

            logger.info("✅ Extended test results logged to database")

        except Exception as e:
            logger.error(f"Failed to log test results: {e}")


if __name__ == "__main__":
    tester = ExtendedEchoBrainTests()
    asyncio.run(tester.run_all_tests())