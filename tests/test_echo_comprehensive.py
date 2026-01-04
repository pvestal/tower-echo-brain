#!/usr/bin/env python3
"""
Comprehensive Echo Brain Testing Suite
Tests all functionality: APIs, GUI, command execution, self-repair
"""

import pytest
import requests
import asyncio
import subprocess
import json
import time
from typing import Dict, Any, List
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import tempfile
import os

class EchoBrainTester:
    """Comprehensive Echo Brain test suite"""

    def __init__(self):
        self.base_url = "http://***REMOVED***:8309"
        self.dashboard_url = "https://***REMOVED***"
        self.test_results = {}

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all Echo Brain tests"""
        print("🧪 Starting Echo Brain Comprehensive Testing")

        results = {
            "api_tests": self.test_api_endpoints(),
            "gui_tests": self.test_gui_interface(),
            "command_execution": self.test_command_execution(),
            "self_repair": self.test_self_repair(),
            "task_queue": self.test_task_queue(),
            "conversation": self.test_conversation_system()
        }

        # Generate report
        self.generate_test_report(results)
        return results

    def test_api_endpoints(self) -> Dict[str, Any]:
        """Test all Echo API endpoints"""
        print("\n📡 Testing API Endpoints")

        endpoints = [
            ("GET", "/api/echo/health", {}),
            ("GET", "/api/echo/status", {}),
            ("GET", "/api/echo/system/metrics", {}),
            ("GET", "/api/echo/db/stats", {}),
            ("GET", "/api/echo/goals", {}),
            ("POST", "/api/echo/query", {"query": "Test query", "conversation_id": "test"}),
            ("POST", "/api/echo/chat", {"message": "Test message"}),
            ("GET", "/api/echo/brain", {}),
        ]

        results = {}
        for method, endpoint, data in endpoints:
            try:
                if method == "GET":
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=10, verify=False)
                else:
                    response = requests.post(f"{self.base_url}{endpoint}", json=data, timeout=10, verify=False)

                results[endpoint] = {
                    "status_code": response.status_code,
                    "success": 200 <= response.status_code < 300,
                    "response_size": len(response.text),
                    "has_json": self._is_json(response.text)
                }
                print(f"  ✅ {method} {endpoint}: {response.status_code}")
            except Exception as e:
                results[endpoint] = {
                    "status_code": None,
                    "success": False,
                    "error": str(e)
                }
                print(f"  ❌ {method} {endpoint}: {e}")

        return results

    def test_gui_interface(self) -> Dict[str, Any]:
        """Test GUI using Selenium"""
        print("\n🖥️  Testing GUI Interface")

        try:
            # Setup headless Chrome
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--ignore-ssl-errors=yes")
            chrome_options.add_argument("--ignore-certificate-errors")

            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(30)

            # Test dashboard loading
            driver.get(self.dashboard_url)

            # Wait for page to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            results = {
                "page_loads": True,
                "title": driver.title,
                "has_echo_elements": self._check_echo_elements(driver),
                "console_errors": self._get_console_errors(driver)
            }

            driver.quit()
            print("  ✅ GUI tests completed")
            return results

        except Exception as e:
            print(f"  ❌ GUI test failed: {e}")
            return {"success": False, "error": str(e)}

    def test_command_execution(self) -> Dict[str, Any]:
        """Test Echo's actual command execution"""
        print("\n⚡ Testing Command Execution")

        test_file = f"/tmp/echo_test_{int(time.time())}.txt"
        test_content = f"Echo test {time.time()}"

        # Test via query endpoint
        try:
            response = requests.post(
                f"{self.base_url}/api/echo/query",
                json={
                    "query": f"Execute this command: echo '{test_content}' > {test_file}",
                    "conversation_id": "command_test"
                },
                timeout=15,
                verify=False
            )

            # Check if file was actually created
            time.sleep(2)  # Give it time to execute
            file_exists = os.path.exists(test_file)
            file_content = ""

            if file_exists:
                with open(test_file, 'r') as f:
                    file_content = f.read().strip()
                os.remove(test_file)  # Cleanup

            return {
                "api_response": response.status_code == 200,
                "file_created": file_exists,
                "correct_content": file_content == test_content,
                "response_text": response.text[:200] if response.text else ""
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_self_repair(self) -> Dict[str, Any]:
        """Test Echo's self-repair capabilities"""
        print("\n🔧 Testing Self-Repair")

        # Test service repair endpoint
        repair_tests = [
            {"service_name": "echo-viz", "action": "restart"},
            {"service_name": "tower-echo-brain", "action": "status"}
        ]

        results = {}
        for test in repair_tests:
            try:
                response = requests.post(
                    f"{self.base_url}/api/service/repair",
                    json=test,
                    timeout=10,
                    verify=False
                )

                results[test["service_name"]] = {
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                    "response": response.json() if self._is_json(response.text) else response.text
                }

            except Exception as e:
                results[test["service_name"]] = {"error": str(e)}

        return results

    def test_task_queue(self) -> Dict[str, Any]:
        """Test Echo's task queue system"""
        print("\n📋 Testing Task Queue")

        # Test task implementation endpoint
        test_task = {
            "task_type": "SYSTEM_TEST",
            "priority": "LOW",
            "description": "Test task for verification",
            "payload": {"test": True}
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/echo/tasks/implement",
                json=test_task,
                timeout=10,
                verify=False
            )

            return {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "response": response.json() if self._is_json(response.text) else response.text
            }

        except Exception as e:
            return {"error": str(e)}

    def test_conversation_system(self) -> Dict[str, Any]:
        """Test Echo's conversation management"""
        print("\n💬 Testing Conversation System")

        try:
            # Test basic chat
            response = requests.post(
                f"{self.base_url}/api/echo/chat",
                json={"message": "Hello Echo, this is a test message"},
                timeout=15,
                verify=False
            )

            return {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "has_response": len(response.text) > 50,
                "response_preview": response.text[:100] if response.text else ""
            }

        except Exception as e:
            return {"error": str(e)}

    def _check_echo_elements(self, driver) -> bool:
        """Check for Echo-specific elements in GUI"""
        try:
            # Look for Echo-related elements
            echo_indicators = [
                "echo", "brain", "tower", "control", "dashboard"
            ]

            page_source = driver.page_source.lower()
            return any(indicator in page_source for indicator in echo_indicators)
        except:
            return False

    def _get_console_errors(self, driver) -> List[str]:
        """Get console errors from browser"""
        try:
            logs = driver.get_log('browser')
            return [log['message'] for log in logs if log['level'] == 'SEVERE']
        except:
            return []

    def _is_json(self, text: str) -> bool:
        """Check if text is valid JSON"""
        try:
            json.loads(text)
            return True
        except:
            return False

    def generate_test_report(self, results: Dict[str, Any]):
        """Generate comprehensive test report"""
        report = f"""
# Echo Brain Test Report - {time.strftime('%Y-%m-%d %H:%M:%S')}

## Summary
"""

        for category, data in results.items():
            if isinstance(data, dict):
                success_count = sum(1 for v in data.values() if isinstance(v, dict) and v.get('success', False))
                total_count = len([v for v in data.values() if isinstance(v, dict)])
                report += f"- **{category}**: {success_count}/{total_count} passed\n"

        report += "\n## Detailed Results\n"
        for category, data in results.items():
            report += f"\n### {category}\n```json\n{json.dumps(data, indent=2)}\n```\n"

        # Save report
        report_file = f"/opt/tower-echo-brain/tests/test_report_{int(time.time())}.md"
        with open(report_file, 'w') as f:
            f.write(report)

        print(f"\n📊 Test report saved: {report_file}")


def main():
    """Run Echo Brain comprehensive tests"""
    tester = EchoBrainTester()
    results = tester.run_all_tests()

    # Print summary
    print("\n" + "="*50)
    print("🧪 ECHO BRAIN TEST SUMMARY")
    print("="*50)

    for category, data in results.items():
        if isinstance(data, dict):
            if 'error' in data:
                print(f"❌ {category}: FAILED - {data['error']}")
            else:
                success_items = [k for k, v in data.items() if isinstance(v, dict) and v.get('success', False)]
                total_items = [k for k, v in data.items() if isinstance(v, dict)]
                print(f"📊 {category}: {len(success_items)}/{len(total_items)} passed")

    return results


if __name__ == "__main__":
    main()