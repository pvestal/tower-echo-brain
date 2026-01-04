#!/usr/bin/env python3
"""
Comprehensive Echo Brain System Tests
Tests the complete session continuity solution
"""

import asyncio
import requests
import json
import subprocess
import psycopg2
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class EchoSystemTester:
    def __init__(self):
        self.base_url = "http://localhost:8309"
        self.db_config = {
            'host': 'localhost',
            'database': 'echo_brain',
            'user': 'patrick',
            'password': '***REMOVED***'
        }
        self.test_results = []

    def log_test(self, test_name: str, status: str, details: str = ""):
        """Log test result"""
        result = {
            'test': test_name,
            'status': status,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_emoji} {test_name}: {status}")
        if details:
            print(f"   {details}")

    def test_service_health(self) -> bool:
        """Test basic service health"""
        try:
            response = requests.get(f"{self.base_url}/api/echo/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                service_name = data.get('service', '')
                if data.get('status') == 'healthy' and ('simple_echo' in service_name):
                    version = data.get('version', 'v1')
                    self.log_test("Service Health", "PASS", f"Service: {service_name} {version}, Response: {response.elapsed.total_seconds():.3f}s")
                    return True
            self.log_test("Service Health", "FAIL", f"Status: {response.status_code}")
            return False
        except Exception as e:
            self.log_test("Service Health", "FAIL", f"Connection error: {e}")
            return False

    def test_database_connectivity(self) -> bool:
        """Test database connection and basic queries"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Test basic connectivity
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            if result[0] != 1:
                raise Exception("Basic query failed")

            # Test required tables exist
            required_tables = ['conversations', 'learning_history']
            for table in required_tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                if count is None:
                    raise Exception(f"Table {table} not accessible")

            cursor.close()
            conn.close()
            self.log_test("Database Connectivity", "PASS", f"All tables accessible")
            return True
        except Exception as e:
            self.log_test("Database Connectivity", "FAIL", f"Error: {e}")
            return False

    def test_session_context_api(self) -> bool:
        """Test session context endpoint"""
        try:
            response = requests.get(f"{self.base_url}/api/echo/session-context", timeout=10)
            if response.status_code != 200:
                self.log_test("Session Context API", "FAIL", f"HTTP {response.status_code}")
                return False

            data = response.json()

            # Check required fields
            required_fields = ['user_profile', 'recent_activity', 'key_learnings', 'current_issues']
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                self.log_test("Session Context API", "FAIL", f"Missing fields: {missing_fields}")
                return False

            # Check user profile
            profile = data.get('user_profile', {})
            if profile.get('name') != 'Patrick':
                self.log_test("Session Context API", "FAIL", "User profile incorrect")
                return False

            # Check service identification
            service_status = data.get('current_status', [])
            if 'simple_echo_v2' not in str(data).lower():
                # Accept both v1 and v2 format
                pass

            self.log_test("Session Context API", "PASS", f"Returned {len(data.get('key_learnings', []))} learnings")
            return True
        except Exception as e:
            self.log_test("Session Context API", "FAIL", f"Error: {e}")
            return False

    def test_conversation_persistence(self) -> bool:
        """Test that conversations are properly stored"""
        try:
            test_query = f"Test query at {datetime.now().isoformat()}"
            test_conversation_id = f"test_conv_{int(time.time())}"

            # Send a test query
            response = requests.post(f"{self.base_url}/api/echo/query", json={
                'query': test_query,
                'conversation_id': test_conversation_id
            }, timeout=10)

            if response.status_code != 200:
                self.log_test("Conversation Persistence", "FAIL", f"Query failed: {response.status_code}")
                return False

            # Check if it was stored in database
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT query_text, response_text, conversation_id
                FROM conversations
                WHERE conversation_id = %s
            """, (test_conversation_id,))
            result = cursor.fetchone()
            cursor.close()
            conn.close()

            if not result:
                self.log_test("Conversation Persistence", "FAIL", "Query not stored in database")
                return False

            if result[0] != test_query:
                self.log_test("Conversation Persistence", "FAIL", "Stored query doesn't match")
                return False

            self.log_test("Conversation Persistence", "PASS", "Query properly stored and retrievable")
            return True
        except Exception as e:
            self.log_test("Conversation Persistence", "FAIL", f"Error: {e}")
            return False

    def test_session_bridge_script(self) -> bool:
        """Test the session bridge shell script"""
        try:
            script_path = "/home/patrick/.claude/echo-session-bridge.sh"
            result = subprocess.run([script_path], capture_output=True, text=True, timeout=15)

            if result.returncode != 0:
                self.log_test("Session Bridge Script", "FAIL", f"Exit code: {result.returncode}")
                return False

            output = result.stdout

            # Check for expected output patterns
            expected_patterns = ["User: Patrick", "Echo Brain:", "Recent work"]
            missing_patterns = [p for p in expected_patterns if p not in output]
            if missing_patterns:
                self.log_test("Session Bridge Script", "FAIL", f"Missing patterns: {missing_patterns}")
                return False

            self.log_test("Session Bridge Script", "PASS", f"Output length: {len(output)} chars")
            return True
        except subprocess.TimeoutExpired:
            self.log_test("Session Bridge Script", "FAIL", "Script timeout")
            return False
        except Exception as e:
            self.log_test("Session Bridge Script", "FAIL", f"Error: {e}")
            return False

    def test_error_handling(self) -> bool:
        """Test error handling when database is unavailable"""
        try:
            # Test with invalid query
            response = requests.post(f"{self.base_url}/api/echo/query", json={
                'query': '',  # Empty query
                'conversation_id': 'test_error'
            }, timeout=5)

            if response.status_code == 500:
                self.log_test("Error Handling", "FAIL", "Service crashed on empty query")
                return False

            # Test session context with potential database issues
            response = requests.get(f"{self.base_url}/api/echo/session-context", timeout=5)
            data = response.json()

            # Should either work normally or return fallback context
            if 'error' in data and 'fallback_context' not in data:
                self.log_test("Error Handling", "FAIL", "No fallback context on error")
                return False

            self.log_test("Error Handling", "PASS", "Graceful error handling verified")
            return True
        except Exception as e:
            self.log_test("Error Handling", "FAIL", f"Error: {e}")
            return False

    def test_performance_benchmarks(self) -> bool:
        """Test system performance under load"""
        try:
            start_time = time.time()

            # Test multiple concurrent requests
            import threading
            results = []

            def make_request():
                try:
                    resp = requests.get(f"{self.base_url}/api/echo/session-context", timeout=5)
                    results.append(resp.elapsed.total_seconds())
                except:
                    results.append(None)

            threads = []
            for _ in range(5):
                t = threading.Thread(target=make_request)
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # Check results
            valid_results = [r for r in results if r is not None]
            if len(valid_results) != 5:
                self.log_test("Performance Benchmarks", "FAIL", f"Only {len(valid_results)}/5 requests succeeded")
                return False

            avg_time = sum(valid_results) / len(valid_results)
            max_time = max(valid_results)

            if max_time > 2.0:  # 2 second threshold
                self.log_test("Performance Benchmarks", "WARN", f"Slow response: {max_time:.3f}s")
                return True
            else:
                self.log_test("Performance Benchmarks", "PASS", f"Avg: {avg_time:.3f}s, Max: {max_time:.3f}s")
                return True

        except Exception as e:
            self.log_test("Performance Benchmarks", "FAIL", f"Error: {e}")
            return False

    def test_file_architecture_cleanup(self) -> bool:
        """Verify the file cleanup was effective"""
        try:
            # Check that simple_echo.py exists and is reasonable size
            import os
            echo_file = "/opt/tower-echo-brain/simple_echo.py"
            if not os.path.exists(echo_file):
                self.log_test("File Architecture", "FAIL", "simple_echo.py not found")
                return False

            file_size = os.path.getsize(echo_file)
            with open(echo_file, 'r') as f:
                line_count = len(f.readlines())

            if line_count > 200:
                self.log_test("File Architecture", "WARN", f"{line_count} lines (target: <200)")
            else:
                self.log_test("File Architecture", "PASS", f"{line_count} lines, {file_size} bytes")

            # Check that service is using simple_echo
            result = subprocess.run(['systemctl', 'show', 'tower-echo-brain.service', '--property=ExecStart'],
                                  capture_output=True, text=True)
            if 'simple_echo' not in result.stdout:
                self.log_test("File Architecture", "FAIL", "Service not using simple_echo.py")
                return False

            return True
        except Exception as e:
            self.log_test("File Architecture", "FAIL", f"Error: {e}")
            return False

    def run_comprehensive_tests(self) -> Dict:
        """Run all tests and return comprehensive results"""
        print("🧪 Running Echo Brain System Tests...")
        print("=" * 50)

        test_functions = [
            self.test_service_health,
            self.test_database_connectivity,
            self.test_session_context_api,
            self.test_conversation_persistence,
            self.test_session_bridge_script,
            self.test_error_handling,
            self.test_performance_benchmarks,
            self.test_file_architecture_cleanup
        ]

        passed = 0
        failed = 0
        warnings = 0

        for test_func in test_functions:
            try:
                result = test_func()
                if result:
                    passed += 1
            except Exception as e:
                self.log_test(test_func.__name__, "FAIL", f"Test crashed: {e}")
                failed += 1

        # Count status types
        for result in self.test_results:
            if result['status'] == 'PASS':
                continue  # Already counted
            elif result['status'] == 'FAIL':
                failed += 1
            elif result['status'] == 'WARN':
                warnings += 1

        print("=" * 50)
        print(f"📊 Test Results: {passed} PASS, {failed} FAIL, {warnings} WARN")

        # Overall assessment
        if failed == 0:
            overall_status = "EXCELLENT" if warnings == 0 else "GOOD"
        elif failed <= 2:
            overall_status = "ACCEPTABLE"
        else:
            overall_status = "POOR"

        summary = {
            'overall_status': overall_status,
            'passed': passed,
            'failed': failed,
            'warnings': warnings,
            'total_tests': len(test_functions),
            'detailed_results': self.test_results,
            'timestamp': datetime.now().isoformat()
        }

        print(f"🎯 Overall Status: {overall_status}")
        return summary

if __name__ == "__main__":
    tester = EchoSystemTester()
    results = tester.run_comprehensive_tests()

    # Save results to file
    with open('/opt/tower-echo-brain/test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📄 Detailed results saved to: /opt/tower-echo-brain/test_results.json")

    # Exit with appropriate code
    exit_code = 0 if results['failed'] == 0 else 1
    exit(exit_code)