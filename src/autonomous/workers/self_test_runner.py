"""Self-Test Runner Worker - Validates Echo Brain's output quality"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

import httpx
import asyncpg

logger = logging.getLogger(__name__)


# Test cases - data-driven for easy extension
SELF_TESTS = [
    {
        "name": "hardware_ram",
        "query": "How much RAM does Tower have?",
        "expected_contains": ["DDR6", "96GB"],
        "expected_not_contains": ["DDR5"],
        "endpoint": "/api/echo/ask",
        "method": "POST",
        "timeout_ms": 30000,
    },
    {
        "name": "hardware_gpu_nvidia",
        "query": "What NVIDIA GPU does Tower have?",
        "expected_contains": ["RTX 3060", "12GB"],
        "expected_not_contains": [],
        "endpoint": "/api/echo/ask",
        "method": "POST",
        "timeout_ms": 30000,
    },
    {
        "name": "hardware_gpu_amd",
        "query": "What AMD GPU does Tower have?",
        "expected_contains": ["RX 9070 XT", "16GB"],
        "expected_not_contains": [],
        "endpoint": "/api/echo/ask",
        "method": "POST",
        "timeout_ms": 30000,
    },
    {
        "name": "service_health",
        "query": None,  # Just GET the endpoint
        "expected_contains": ["healthy"],
        "expected_not_contains": [],
        "endpoint": "/health",
        "method": "GET",
        "timeout_ms": 5000,
    },
    {
        "name": "worker_status",
        "query": None,
        "expected_contains": ["true"],  # running: true
        "expected_not_contains": [],
        "endpoint": "/api/workers/status",
        "method": "GET",
        "timeout_ms": 5000,
    },
    {
        "name": "vector_search_functional",
        "query": "What is Echo Brain?",
        "expected_contains": ["personal", "AI"],
        "expected_not_contains": [],
        "endpoint": "/api/echo/ask",
        "method": "POST",
        "timeout_ms": 30000,
    },
    {
        "name": "echo_brain_purpose",
        "query": "What is the purpose of Echo Brain?",
        "expected_contains": ["memory", "knowledge"],
        "expected_not_contains": [],
        "endpoint": "/api/echo/ask",
        "method": "POST",
        "timeout_ms": 30000,
    },
    {
        "name": "patrick_info",
        "query": "Who is Patrick?",
        "expected_contains": ["developer", "engineer"],
        "expected_not_contains": [],
        "endpoint": "/api/echo/ask",
        "method": "POST",
        "timeout_ms": 30000,
    },
]


class SelfTestRunner:
    """Runs self-tests to validate Echo Brain's output quality"""

    def __init__(self):
        self.db_url = os.environ.get("DATABASE_URL",
            "postgresql://echo:echo_secure_password_123@localhost/echo_brain")
        self.base_url = "http://localhost:8309"
        self.config_file = "/opt/tower-echo-brain/config/self_tests.json"

    async def run_cycle(self):
        """Main worker cycle — called by scheduler"""
        logger.info("🧪 Self-Test Runner starting cycle")

        try:
            conn = await asyncpg.connect(self.db_url)

            # Load test cases (builtin + custom)
            test_cases = await self._load_test_cases()
            logger.info(f"Running {len(test_cases)} self-tests")

            # Track results
            tests_passed = 0
            tests_failed = 0
            regressions_detected = []

            async with httpx.AsyncClient(timeout=60.0) as client:
                for test in test_cases:
                    try:
                        # Run the test
                        result = await self._run_test(client, test)

                        # Store result
                        await conn.execute("""
                            INSERT INTO self_test_results
                            (test_name, test_query, expected_contains, expected_not_contains,
                             actual_response, passed, failure_reason, response_time_ms)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        """,
                            test['name'],
                            test.get('query'),
                            test['expected_contains'],
                            test['expected_not_contains'],
                            result['response'][:5000],  # Limit stored response size
                            result['passed'],
                            result.get('failure_reason'),
                            result['response_time_ms']
                        )

                        if result['passed']:
                            tests_passed += 1
                        else:
                            tests_failed += 1

                            # Check if this is a regression
                            previous_pass = await conn.fetchval("""
                                SELECT COUNT(*) > 0
                                FROM self_test_results
                                WHERE test_name = $1
                                    AND passed = true
                                    AND run_at > NOW() - INTERVAL '24 hours'
                            """, test['name'])

                            if previous_pass:
                                regressions_detected.append(test['name'])
                                await self._create_regression_issue(conn, test, result)

                    except Exception as e:
                        logger.error(f"Test '{test['name']}' execution failed: {e}")
                        tests_failed += 1

                        # Record the failure
                        await conn.execute("""
                            INSERT INTO self_test_results
                            (test_name, test_query, expected_contains, expected_not_contains,
                             actual_response, passed, failure_reason, response_time_ms)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        """,
                            test['name'],
                            test.get('query'),
                            test['expected_contains'],
                            test['expected_not_contains'],
                            '',
                            False,
                            f"Test execution error: {str(e)}",
                            0
                        )

            # Calculate pass rate
            total_tests = tests_passed + tests_failed
            pass_rate = (tests_passed / total_tests) if total_tests > 0 else 0.0

            # Store metrics
            await conn.execute("""
                INSERT INTO self_health_metrics (metric_name, metric_value, metadata)
                VALUES
                    ('self_test_pass_rate', $1, $2::jsonb),
                    ('self_test_regressions', $3, $4::jsonb)
            """,
                pass_rate,
                json.dumps({
                    "tests_passed": tests_passed,
                    "tests_failed": tests_failed,
                    "total_tests": total_tests
                }),
                float(len(regressions_detected)),
                json.dumps({"regressed_tests": regressions_detected})
            )

            # Create notification if pass rate is too low
            if pass_rate < 0.8 and total_tests > 0:
                await self._create_quality_alert(conn, pass_rate, tests_failed, regressions_detected)

            await conn.close()

            logger.info(f"✅ Self-Test Runner completed: {tests_passed}/{total_tests} passed "
                       f"({pass_rate:.1%}), {len(regressions_detected)} regressions")

        except Exception as e:
            logger.error(f"❌ Self-Test Runner cycle failed: {e}", exc_info=True)

            # Try to record the failure
            try:
                conn = await asyncpg.connect(self.db_url)
                await conn.execute("""
                    INSERT INTO self_detected_issues
                    (issue_type, severity, source, title, description, related_worker)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """,
                    "worker_failure", "critical", "self_test_runner",
                    "Self-Test Runner cycle failed",
                    str(e), "self_test_runner"
                )
                await conn.close()
            except:
                pass

    async def _load_test_cases(self) -> List[Dict[str, Any]]:
        """Load test cases from builtin + config file"""
        test_cases = list(SELF_TESTS)  # Start with builtin tests

        # Load additional tests from config if it exists
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    custom_tests = json.load(f)
                    if isinstance(custom_tests, list):
                        test_cases.extend(custom_tests)
                        logger.info(f"Loaded {len(custom_tests)} custom tests from config")
            except Exception as e:
                logger.warning(f"Failed to load custom tests: {e}")

        return test_cases

    async def _run_test(self, client: httpx.AsyncClient, test: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single test case"""
        start_time = time.time()
        url = f"{self.base_url}{test['endpoint']}"

        try:
            # Make the request
            if test['method'] == 'POST' and test.get('query'):
                response = await client.post(
                    url,
                    json={"question": test['query']},
                    timeout=test['timeout_ms'] / 1000
                )
            else:
                response = await client.get(
                    url,
                    timeout=test['timeout_ms'] / 1000
                )

            response_time_ms = int((time.time() - start_time) * 1000)

            # Parse response
            if response.status_code == 200:
                if response.headers.get('content-type', '').startswith('application/json'):
                    response_data = response.json()
                    # Extract the actual response text
                    response_text = str(response_data.get('answer') or
                                       response_data.get('response') or
                                       response_data.get('status') or
                                       json.dumps(response_data))
                else:
                    response_text = response.text
            else:
                return {
                    'passed': False,
                    'response': f"HTTP {response.status_code}: {response.text[:500]}",
                    'response_time_ms': response_time_ms,
                    'failure_reason': f"HTTP {response.status_code}"
                }

            # Validate response
            response_lower = response_text.lower()
            passed = True
            failure_reasons = []

            # Check expected_contains
            for expected in test['expected_contains']:
                if expected.lower() not in response_lower:
                    passed = False
                    failure_reasons.append(f"Missing expected: '{expected}'")

            # Check expected_not_contains
            for not_expected in test['expected_not_contains']:
                if not_expected.lower() in response_lower:
                    passed = False
                    failure_reasons.append(f"Contains unexpected: '{not_expected}'")

            return {
                'passed': passed,
                'response': response_text,
                'response_time_ms': response_time_ms,
                'failure_reason': '; '.join(failure_reasons) if failure_reasons else None
            }

        except httpx.TimeoutException:
            return {
                'passed': False,
                'response': '',
                'response_time_ms': test['timeout_ms'],
                'failure_reason': f"Timeout after {test['timeout_ms']}ms"
            }
        except Exception as e:
            return {
                'passed': False,
                'response': '',
                'response_time_ms': int((time.time() - start_time) * 1000),
                'failure_reason': f"Exception: {str(e)}"
            }

    async def _create_regression_issue(self, conn: asyncpg.Connection, test: Dict[str, Any], result: Dict[str, Any]):
        """Create an issue for a detected regression"""
        await conn.execute("""
            INSERT INTO self_detected_issues
            (issue_type, severity, source, title, description, related_worker, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
            "query_regression", "warning", "self_test_runner",
            f"Regression in test: {test['name']}",
            f"Test '{test['name']}' previously passed but now fails.\n"
            f"Query: {test.get('query', 'N/A')}\n"
            f"Failure: {result.get('failure_reason', 'Unknown')}\n"
            f"Response: {result['response'][:500]}",
            None, "open"
        )

    async def _create_quality_alert(self, conn: asyncpg.Connection, pass_rate: float,
                                   tests_failed: int, regressions: List[str]):
        """Create alert for low test pass rate"""
        await conn.execute("""
            INSERT INTO autonomous_notifications
            (title, message, notification_type, severity, metadata)
            VALUES ($1, $2, $3, $4, $5::jsonb)
        """,
            f"Self-test pass rate below threshold: {pass_rate:.1%}",
            f"Only {pass_rate:.1%} of self-tests are passing.\n"
            f"Failed tests: {tests_failed}\n"
            f"Regressions detected: {', '.join(regressions) if regressions else 'None'}",
            "alert", "high",
            json.dumps({
                "pass_rate": pass_rate,
                "tests_failed": tests_failed,
                "regressions": regressions
            })
        )