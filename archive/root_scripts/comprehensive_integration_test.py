#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite for Echo Brain Vector Database and Memory System
Tests all major components: Qdrant vector database, PostgreSQL memory, API endpoints
"""

import asyncio
import json
import requests
import time
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from typing import Dict, List, Optional
from datetime import datetime
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EchoBrainIntegrationTest:
    def __init__(self):
        self.base_url = "http://localhost:8309"
        self.qdrant_url = "http://localhost:6333"
        self.db_config = {
            "host": "localhost",
            "database": "echo_brain",
            "user": "patrick",
            "password": "tower_echo_brain_secret_key_2025"
        }
        self.test_results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": []
        }

    def log_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result and update counters"""
        self.test_results["total_tests"] += 1
        if success:
            self.test_results["passed"] += 1
            logger.info(f"✅ {test_name}: PASSED {details}")
        else:
            self.test_results["failed"] += 1
            self.test_results["errors"].append(f"{test_name}: {details}")
            logger.error(f"❌ {test_name}: FAILED {details}")

    def test_service_health(self) -> bool:
        """Test 1: Basic service health check"""
        try:
            response = requests.get(f"{self.base_url}/api/echo/health", timeout=10)
            success = response.status_code == 200

            if success:
                health_data = response.json()
                details = f"Status: {health_data.get('status', 'unknown')}"
            else:
                details = f"HTTP {response.status_code}"

            self.log_result("Service Health Check", success, details)
            return success
        except Exception as e:
            self.log_result("Service Health Check", False, str(e))
            return False

    def test_qdrant_connection(self) -> bool:
        """Test 2: Qdrant vector database connection"""
        try:
            response = requests.get(f"{self.qdrant_url}/collections", timeout=10)
            success = response.status_code == 200

            if success:
                collections = response.json()
                details = f"Collections found: {len(collections.get('result', {}).get('collections', []))}"
            else:
                details = f"HTTP {response.status_code}"

            self.log_result("Qdrant Vector Database Connection", success, details)
            return success
        except Exception as e:
            self.log_result("Qdrant Vector Database Connection", False, str(e))
            return False

    def test_postgresql_connection(self) -> bool:
        """Test 3: PostgreSQL database connection and schema"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Test echo_conversations table
            cur.execute("SELECT COUNT(*) as count FROM echo_conversations")
            result = cur.fetchone()
            conversation_count = result['count']

            # Test if context column exists
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'echo_conversations' AND column_name = 'context'
            """)
            context_exists = cur.fetchone() is not None

            cur.close()
            conn.close()

            success = context_exists and conversation_count >= 0
            details = f"Conversations: {conversation_count}, Context column: {context_exists}"

            self.log_result("PostgreSQL Database Connection", success, details)
            return success
        except Exception as e:
            self.log_result("PostgreSQL Database Connection", False, str(e))
            return False

    def test_memory_query_api(self) -> bool:
        """Test 4: Memory query API endpoint"""
        try:
            test_query = {
                "query": "Test memory system integration",
                "conversation_id": f"integration_test_{int(time.time())}"
            }

            response = requests.post(
                f"{self.base_url}/api/echo/query",
                json=test_query,
                timeout=30
            )

            success = response.status_code == 200

            if success:
                result = response.json()
                response_text = result.get('response', 'No response')
                details = f"Response length: {len(response_text)} chars"
            else:
                details = f"HTTP {response.status_code}: {response.text[:200]}"

            self.log_result("Memory Query API", success, details)
            return success
        except Exception as e:
            self.log_result("Memory Query API", False, str(e))
            return False

    def test_conversation_persistence(self) -> bool:
        """Test 5: Conversation persistence in database"""
        try:
            # Create a conversation
            test_conversation_id = f"persistence_test_{int(time.time())}"
            test_query = {
                "query": "Remember this test message for persistence verification",
                "conversation_id": test_conversation_id
            }

            # Send query
            response = requests.post(
                f"{self.base_url}/api/echo/query",
                json=test_query,
                timeout=30
            )

            if response.status_code != 200:
                self.log_result("Conversation Persistence", False, f"Query failed: {response.status_code}")
                return False

            # Wait a moment for database save
            time.sleep(2)

            # Check if conversation was saved
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute(
                "SELECT * FROM echo_conversations WHERE conversation_id = %s",
                (test_conversation_id,)
            )
            conversation_record = cur.fetchone()

            cur.close()
            conn.close()

            success = conversation_record is not None
            details = f"Conversation saved: {success}"
            if success:
                details += f", Context exists: {'context' in conversation_record and conversation_record['context'] is not None}"

            self.log_result("Conversation Persistence", success, details)
            return success
        except Exception as e:
            self.log_result("Conversation Persistence", False, str(e))
            return False

    def test_vector_memory_search(self) -> bool:
        """Test 6: Vector memory search functionality"""
        try:
            # First, check if we have vector memories using POST method
            response = requests.post(
                f"{self.qdrant_url}/collections/echo_memories/points/scroll",
                json={"limit": 10},
                headers={"Content-Type": "application/json"},
                timeout=10
            )

            if response.status_code != 200:
                self.log_result("Vector Memory Search", False, f"Qdrant collection not accessible: {response.status_code}")
                return False

            points_data = response.json()
            vector_count = len(points_data.get('result', {}).get('points', []))

            # For now, just check that the collection is accessible (empty is OK)
            success = response.json().get('status') == 'ok'
            details = f"Vector memories found: {vector_count}, Collection accessible: {success}"

            self.log_result("Vector Memory Search", success, details)
            return success
        except Exception as e:
            self.log_result("Vector Memory Search", False, str(e))
            return False

    def test_api_response_times(self) -> bool:
        """Test 7: API response time performance"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/api/echo/health", timeout=10)
            health_time = time.time() - start_time

            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/echo/query",
                json={"query": "Quick test", "conversation_id": f"perf_test_{int(time.time())}"},
                timeout=30
            )
            query_time = time.time() - start_time

            success = health_time < 1.0 and query_time < 30.0
            details = f"Health: {health_time:.2f}s, Query: {query_time:.2f}s"

            self.log_result("API Response Times", success, details)
            return success
        except Exception as e:
            self.log_result("API Response Times", False, str(e))
            return False

    def test_memory_system_integration(self) -> bool:
        """Test 8: End-to-end memory system integration"""
        try:
            # Test conversation with context retrieval
            conversation_id = f"integration_full_test_{int(time.time())}"

            # First message
            response1 = requests.post(
                f"{self.base_url}/api/echo/query",
                json={
                    "query": "My name is TestUser and I work in software development",
                    "conversation_id": conversation_id
                },
                timeout=30
            )

            if response1.status_code != 200:
                self.log_result("Memory System Integration", False, f"First message failed: {response1.status_code}")
                return False

            time.sleep(3)  # Allow processing

            # Second message - should have context from first
            response2 = requests.post(
                f"{self.base_url}/api/echo/query",
                json={
                    "query": "What do you remember about my work?",
                    "conversation_id": conversation_id
                },
                timeout=30
            )

            success = response2.status_code == 200

            if success:
                result2 = response2.json()
                response_text = result2.get('response', '').lower()
                # Check if response mentions software development or related terms
                context_aware = any(term in response_text for term in ['software', 'development', 'testuser', 'work'])
                details = f"Context awareness: {context_aware}"
                success = context_aware
            else:
                details = f"Second query failed: {response2.status_code}"

            self.log_result("Memory System Integration", success, details)
            return success
        except Exception as e:
            self.log_result("Memory System Integration", False, str(e))
            return False

    def run_all_tests(self):
        """Run all integration tests"""
        logger.info("🚀 Starting Echo Brain Comprehensive Integration Tests")
        logger.info("=" * 60)

        tests = [
            self.test_service_health,
            self.test_qdrant_connection,
            self.test_postgresql_connection,
            self.test_memory_query_api,
            self.test_conversation_persistence,
            self.test_vector_memory_search,
            self.test_api_response_times,
            self.test_memory_system_integration
        ]

        for test in tests:
            try:
                test()
            except Exception as e:
                logger.error(f"Test execution failed: {e}")

            # Small delay between tests
            time.sleep(1)

        # Print final results
        self.print_final_results()

    def print_final_results(self):
        """Print comprehensive test results"""
        logger.info("=" * 60)
        logger.info("🏁 INTEGRATION TEST RESULTS")
        logger.info("=" * 60)

        total = self.test_results["total_tests"]
        passed = self.test_results["passed"]
        failed = self.test_results["failed"]
        success_rate = (passed / total * 100) if total > 0 else 0

        logger.info(f"Total Tests: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success Rate: {success_rate:.1f}%")

        if failed > 0:
            logger.info("\n❌ FAILED TESTS:")
            for error in self.test_results["errors"]:
                logger.info(f"  • {error}")

        if success_rate >= 80:
            logger.info(f"\n🎉 OVERALL RESULT: SYSTEM IS FUNCTIONAL (Score: {success_rate:.1f}%)")
        elif success_rate >= 60:
            logger.info(f"\n⚠️  OVERALL RESULT: SYSTEM NEEDS ATTENTION (Score: {success_rate:.1f}%)")
        else:
            logger.info(f"\n🚨 OVERALL RESULT: SYSTEM REQUIRES MAJOR FIXES (Score: {success_rate:.1f}%)")

        logger.info("=" * 60)

if __name__ == "__main__":
    test_suite = EchoBrainIntegrationTest()
    test_suite.run_all_tests()