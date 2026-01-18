#!/usr/bin/env python3
"""
Comprehensive Functional Test Suite for Echo Brain Capabilities
Tests actual execution, database logging, and improvement tracking
"""

import asyncio
import json
import time
import subprocess
import psycopg2
import aiohttp
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EchoBrainFunctionalTests:
    """Comprehensive functional testing for Echo Brain capabilities"""

    def __init__(self):
        self.base_url = "http://localhost:8309"
        self.db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "echo_brain",  # Use the actual database Echo Brain uses
            "user": "patrick",
            "password": "RP78eIrW7cI2jYvL5akt1yurE"
        }
        self.test_results = []
        self.session = None

    async def setup(self):
        """Setup test environment"""
        self.session = aiohttp.ClientSession()
        logger.info("🔧 Test environment setup complete")

    async def teardown(self):
        """Cleanup test environment"""
        if self.session:
            await self.session.close()
        logger.info("🧹 Test cleanup complete")

    async def make_request(self, query: str, conversation_id: str = None) -> Dict:
        """Make a request to Echo Brain"""
        if not conversation_id:
            conversation_id = f"test_{int(time.time())}"

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
            logger.error(f"Request failed: {e}")
            return {"error": str(e)}

    def verify_database_log(self, conversation_id: str) -> bool:
        """Verify interaction was logged in database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(*) FROM echo_unified_interactions
                WHERE conversation_id = %s
            """, (conversation_id,))

            count = cursor.fetchone()[0]
            cursor.close()
            conn.close()

            return count > 0
        except Exception as e:
            logger.error(f"Database verification failed: {e}")
            return False

    def verify_service_status(self, service_name: str, expected_status: str = "active") -> bool:
        """Verify actual service status"""
        try:
            result = subprocess.run(
                ["systemctl", "is-active", service_name],
                capture_output=True,
                text=True
            )
            actual_status = result.stdout.strip()
            return actual_status == expected_status
        except Exception as e:
            logger.error(f"Service status check failed: {e}")
            return False

    def check_logs_for_capability(self, capability: str, since_seconds: int = 10, conversation_id: str = None) -> bool:
        """Check system logs for capability execution"""
        try:
            result = subprocess.run([
                "sudo", "journalctl", "-u", "tower-echo-brain",
                "--since", f"{since_seconds} seconds ago",
                "--no-pager"
            ], capture_output=True, text=True)

            # If conversation_id provided, check if it appears near the capability
            if conversation_id:
                # Check if both the capability string AND the conversation ID appear
                lines = result.stdout.split('\n')
                for i, line in enumerate(lines):
                    if conversation_id in line:
                        # Check nearby lines for capability
                        for j in range(max(0, i-5), min(len(lines), i+5)):
                            if capability in lines[j]:
                                return True
                return False
            else:
                return capability in result.stdout
        except Exception as e:
            logger.error(f"Log check failed: {e}")
            return False

    async def test_service_restart_capability(self) -> Dict[str, Any]:
        """Test service restart capability with full validation"""
        test_name = "Service Restart Capability"
        logger.info(f"🧪 Testing: {test_name}")

        conversation_id = f"restart_test_{int(time.time())}"
        service_name = "tower-dashboard"

        # Get initial service uptime
        initial_status = subprocess.run(
            ["systemctl", "show", service_name, "--property=ActiveEnterTimestamp"],
            capture_output=True, text=True
        ).stdout.strip()

        # Execute restart request
        response = await self.make_request(
            f"Can you restart the {service_name} service?",
            conversation_id
        )

        # Wait for restart to complete
        await asyncio.sleep(3)

        # Get new service uptime
        new_status = subprocess.run(
            ["systemctl", "show", service_name, "--property=ActiveEnterTimestamp"],
            capture_output=True, text=True
        ).stdout.strip()

        # Validations
        validations = {
            "response_received": bool(response and "response" in response),
            "action_taken": response.get("reasoning", {}).get("action_taken", False),
            "service_restarted": initial_status != new_status,
            "service_active": self.verify_service_status(service_name),
            "database_logged": self.verify_database_log(conversation_id),
            "logs_show_execution": self.check_logs_for_capability("autonomous_repair")
        }

        success = all(validations.values())

        result = {
            "test_name": test_name,
            "success": success,
            "validations": validations,
            "response": response.get("response", "No response"),
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat()
        }

        self.test_results.append(result)
        return result

    async def test_system_diagnosis_capability(self) -> Dict[str, Any]:
        """Test system diagnosis capability"""
        test_name = "System Diagnosis Capability"
        logger.info(f"🧪 Testing: {test_name}")

        conversation_id = f"diagnosis_test_{int(time.time())}"

        # Execute diagnosis request
        response = await self.make_request(
            "Diagnose the system and check all services",
            conversation_id
        )

        # Parse response for service status
        response_text = response.get("response", "")

        validations = {
            "response_received": bool(response and "response" in response),
            "action_taken": response.get("reasoning", {}).get("action_taken", False),
            "contains_service_status": "Active Services" in response_text,
            "lists_services": all(
                service in response_text
                for service in ["tower-dashboard", "tower-echo-brain"]
            ),
            "database_logged": self.verify_database_log(conversation_id),
            "logs_show_execution": self.check_logs_for_capability("system_status")
        }

        success = all(validations.values())

        result = {
            "test_name": test_name,
            "success": success,
            "validations": validations,
            "response": response_text,
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat()
        }

        self.test_results.append(result)
        return result

    async def test_notification_capability(self) -> Dict[str, Any]:
        """Test notification sending capability"""
        test_name = "Notification Capability"
        logger.info(f"🧪 Testing: {test_name}")

        conversation_id = f"notify_test_{int(time.time())}"
        test_message = f"Test notification at {datetime.now().strftime('%H:%M:%S')}"

        # Execute notification request
        response = await self.make_request(
            f"Send a notification: {test_message}",
            conversation_id
        )

        validations = {
            "response_received": bool(response and "response" in response),
            "action_taken": response.get("reasoning", {}).get("action_taken", False),
            "success_message": "Notification sent" in response.get("response", ""),
            "database_logged": self.verify_database_log(conversation_id),
            "logs_show_execution": self.check_logs_for_capability("send_notification")
        }

        success = all(validations.values())

        result = {
            "test_name": test_name,
            "success": success,
            "validations": validations,
            "response": response.get("response", "No response"),
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat()
        }

        self.test_results.append(result)
        return result

    async def test_non_capability_query(self) -> Dict[str, Any]:
        """Test that non-capability queries don't trigger actions"""
        test_name = "Non-Capability Query Handling"
        logger.info(f"🧪 Testing: {test_name}")

        conversation_id = f"normal_test_{int(time.time())}"

        # Execute normal conversation request
        response = await self.make_request(
            "What is the weather like today?",
            conversation_id
        )

        validations = {
            "response_received": bool(response and "response" in response),
            "no_action_taken": not response.get("reasoning", {}).get("action_taken", False),
            "normal_response": len(response.get("response", "")) > 0,
            "database_logged": self.verify_database_log(conversation_id),
            "no_capability_execution": not self.check_logs_for_capability("Executing capability", conversation_id=conversation_id)
        }

        success = all(validations.values())

        result = {
            "test_name": test_name,
            "success": success,
            "validations": validations,
            "response": response.get("response", "No response")[:100],
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat()
        }

        self.test_results.append(result)
        return result

    async def test_database_persistence(self) -> Dict[str, Any]:
        """Test database persistence of interactions"""
        test_name = "Database Persistence"
        logger.info(f"🧪 Testing: {test_name}")

        conversation_id = f"persist_test_{int(time.time())}"

        # Make multiple requests in same conversation
        queries = [
            "What services are running?",
            "Can you restart tower-auth?",
            "Send a notification: Test complete"
        ]

        for query in queries:
            await self.make_request(query, conversation_id)
            await asyncio.sleep(1)

        # Verify all interactions were logged
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT query, response, model_used, intent, confidence
                FROM echo_unified_interactions
                WHERE conversation_id = %s
                ORDER BY timestamp
            """, (conversation_id,))

            rows = cursor.fetchall()
            cursor.close()
            conn.close()

            validations = {
                "all_queries_logged": len(rows) == len(queries),
                "queries_match": all(
                    rows[i][0] == queries[i] for i in range(min(len(rows), len(queries)))
                ),
                "responses_exist": all(row[1] for row in rows),
                "model_recorded": all(row[2] for row in rows),
                "intent_classified": all(row[3] for row in rows)
            }

        except Exception as e:
            logger.error(f"Database test failed: {e}")
            validations = {"database_error": str(e)}

        success = all(v for v in validations.values() if isinstance(v, bool))

        result = {
            "test_name": test_name,
            "success": success,
            "validations": validations,
            "conversation_id": conversation_id,
            "interactions_count": len(rows) if 'rows' in locals() else 0,
            "timestamp": datetime.now().isoformat()
        }

        self.test_results.append(result)
        return result

    async def test_capability_performance(self) -> Dict[str, Any]:
        """Test capability execution performance"""
        test_name = "Capability Performance"
        logger.info(f"🧪 Testing: {test_name}")

        performance_metrics = []

        test_queries = [
            ("Diagnose the system", "diagnosis"),
            ("Send a notification: Performance test", "notification"),
            ("What is 2+2?", "normal_query")
        ]

        for query, query_type in test_queries:
            start_time = time.time()
            response = await self.make_request(query)
            end_time = time.time()

            processing_time = end_time - start_time
            api_reported_time = response.get("processing_time", 0)

            performance_metrics.append({
                "query_type": query_type,
                "total_time": processing_time,
                "api_time": api_reported_time,
                "overhead": processing_time - api_reported_time if api_reported_time else 0
            })

        validations = {
            "all_under_5_seconds": all(m["total_time"] < 5 for m in performance_metrics),
            "capabilities_under_3_seconds": all(
                m["total_time"] < 3 for m in performance_metrics
                if m["query_type"] != "normal_query"
            ),
            "api_times_reported": all(m["api_time"] > 0 for m in performance_metrics)
        }

        success = all(validations.values())

        result = {
            "test_name": test_name,
            "success": success,
            "validations": validations,
            "performance_metrics": performance_metrics,
            "average_time": sum(m["total_time"] for m in performance_metrics) / len(performance_metrics),
            "timestamp": datetime.now().isoformat()
        }

        self.test_results.append(result)
        return result

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all functional tests"""
        logger.info("🚀 Starting Echo Brain Functional Test Suite")

        await self.setup()

        test_methods = [
            self.test_service_restart_capability,
            self.test_system_diagnosis_capability,
            self.test_notification_capability,
            self.test_non_capability_query,
            self.test_database_persistence,
            self.test_capability_performance
        ]

        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                logger.error(f"Test {test_method.__name__} failed: {e}")
                self.test_results.append({
                    "test_name": test_method.__name__,
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })

        await self.teardown()

        # Generate summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.get("success", False))

        summary = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "test_results": self.test_results,
            "timestamp": datetime.now().isoformat()
        }

        # Save results to file
        with open("/opt/tower-echo-brain/tests/test_results.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save to database for tracking
        self.save_test_results_to_db(summary)

        return summary

    def save_test_results_to_db(self, summary: Dict[str, Any]):
        """Save test results to database for improvement tracking"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Create test results table if needed
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS echo_test_results (
                    id SERIAL PRIMARY KEY,
                    test_run_id VARCHAR(100),
                    total_tests INTEGER,
                    passed INTEGER,
                    failed INTEGER,
                    success_rate FLOAT,
                    results JSONB,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Insert test results
            cursor.execute("""
                INSERT INTO echo_test_results
                (test_run_id, total_tests, passed, failed, success_rate, results)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                f"functional_test_{int(time.time())}",
                summary["total_tests"],
                summary["passed"],
                summary["failed"],
                summary["success_rate"],
                json.dumps(summary["test_results"])
            ))

            conn.commit()
            cursor.close()
            conn.close()

            logger.info("✅ Test results saved to database")

        except Exception as e:
            logger.error(f"Failed to save test results to database: {e}")

    def print_summary(self, summary: Dict[str, Any]):
        """Print formatted test summary"""
        print("\n" + "="*60)
        print("ECHO BRAIN FUNCTIONAL TEST RESULTS")
        print("="*60)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print("\nDetailed Results:")
        print("-"*60)

        for result in summary["test_results"]:
            status = "✅ PASS" if result.get("success") else "❌ FAIL"
            print(f"{status} - {result.get('test_name', 'Unknown Test')}")

            if not result.get("success") and "validations" in result:
                for key, value in result["validations"].items():
                    if not value:
                        print(f"  ❌ {key}")

        print("="*60)


async def main():
    """Main test runner"""
    tester = EchoBrainFunctionalTests()
    summary = await tester.run_all_tests()
    tester.print_summary(summary)

    # Return exit code based on test results
    exit_code = 0 if summary["success_rate"] == 100 else 1
    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)