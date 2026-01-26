#!/usr/bin/env python3
"""
REAL Echo Brain Functionality Test
No BS, no fake claims - actually test if things work
"""
import httpx
import json
import time
import uuid
import sys
from datetime import datetime

class EchoBrainTester:
    def __init__(self):
        self.client = httpx.Client(timeout=10.0)
        self.results = []
        self.test_id = str(uuid.uuid4())[:8]

    def test(self, name, func):
        """Run a test and record real results"""
        print(f"Testing: {name}")
        start_time = time.time()

        try:
            result = func()
            duration = time.time() - start_time

            if result['success']:
                print(f"  ✓ PASS ({duration:.2f}s)")
                self.results.append({
                    'name': name,
                    'status': 'PASS',
                    'duration': duration,
                    'details': result.get('details', '')
                })
            else:
                print(f"  ✗ FAIL ({duration:.2f}s): {result['error']}")
                self.results.append({
                    'name': name,
                    'status': 'FAIL',
                    'duration': duration,
                    'error': result['error']
                })

        except Exception as e:
            duration = time.time() - start_time
            print(f"  ✗ ERROR ({duration:.2f}s): {e}")
            self.results.append({
                'name': name,
                'status': 'ERROR',
                'duration': duration,
                'error': str(e)
            })

    def test_basic_connectivity(self):
        """Test if Echo Brain is even running"""
        try:
            resp = self.client.get("http://localhost:8309/health", timeout=5)
            if resp.status_code == 200:
                return {'success': True, 'details': 'Service responding'}
            else:
                return {'success': False, 'error': f'HTTP {resp.status_code}'}
        except Exception as e:
            return {'success': False, 'error': f'Connection failed: {e}'}

    def test_knowledge_creation(self):
        """Test if we can actually create knowledge"""
        test_fact = {
            "subject": f"test_{self.test_id}",
            "predicate": "verification_status",
            "object": "testing_creation",
            "confidence": 0.9
        }

        try:
            resp = self.client.post(
                "http://localhost:8309/api/knowledge/facts",
                json=test_fact
            )

            if resp.status_code != 200:
                return {'success': False, 'error': f'HTTP {resp.status_code}: {resp.text[:100]}'}

            result = resp.json()
            if 'id' in result and 'fact' in result:
                return {'success': True, 'details': f"Created fact ID: {result['id']}"}
            else:
                return {'success': False, 'error': f'Invalid response: {result}'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_knowledge_retrieval(self):
        """Test if we can retrieve the knowledge we created"""
        try:
            resp = self.client.get(
                f"http://localhost:8309/api/knowledge/facts?subject=test_{self.test_id}"
            )

            if resp.status_code != 200:
                return {'success': False, 'error': f'HTTP {resp.status_code}'}

            result = resp.json()
            if 'facts' in result and len(result['facts']) > 0:
                fact = result['facts'][0]
                if fact['object'] == 'testing_creation':
                    return {'success': True, 'details': f"Retrieved {len(result['facts'])} facts"}
                else:
                    return {'success': False, 'error': f'Wrong fact retrieved: {fact}'}
            else:
                return {'success': False, 'error': 'No facts found'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_mcp_server(self):
        """Test if MCP server actually works"""
        try:
            # Test fact storage
            mcp_data = {
                "method": "tools/call",
                "params": {
                    "name": "store_fact",
                    "arguments": {
                        "subject": f"mcp_test_{self.test_id}",
                        "predicate": "mcp_status",
                        "object": "testing_mcp",
                        "source": "real_test"
                    }
                }
            }

            resp = self.client.post("http://localhost:8312/mcp", json=mcp_data)

            if resp.status_code != 200:
                return {'success': False, 'error': f'MCP HTTP {resp.status_code}'}

            result = resp.json()
            if 'content' in result and len(result['content']) > 0:
                if 'Successfully stored' in result['content'][0]['text']:
                    return {'success': True, 'details': 'MCP fact storage working'}
                else:
                    return {'success': False, 'error': f'MCP returned: {result}'}
            else:
                return {'success': False, 'error': 'Invalid MCP response'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_query_processing(self):
        """Test if Echo Brain can actually process a query"""
        try:
            query_data = {
                "query": f"What do you know about test_{self.test_id}?"
            }

            resp = self.client.post(
                "http://localhost:8309/api/echo/query",
                json=query_data
            )

            if resp.status_code != 200:
                return {'success': False, 'error': f'HTTP {resp.status_code}'}

            result = resp.json()
            if 'response' in result and len(result.get('response', '')) > 10:
                return {'success': True, 'details': f"Response: {result['response'][:100]}..."}
            else:
                return {'success': False, 'error': f'Empty or invalid response: {result}'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_autonomous_tasks(self):
        """Test if autonomous tasks actually exist"""
        try:
            resp = self.client.get("http://localhost:8309/api/autonomous/tasks")

            if resp.status_code != 200:
                return {'success': False, 'error': f'HTTP {resp.status_code}'}

            result = resp.json()
            if isinstance(result, list):
                return {'success': True, 'details': f"Found {len(result)} tasks"}
            else:
                return {'success': False, 'error': f'Invalid response: {type(result)}'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_agents_status(self):
        """Test if agents are actually working"""
        try:
            resp = self.client.get("http://localhost:8309/api/agents/status")

            if resp.status_code != 200:
                return {'success': False, 'error': f'HTTP {resp.status_code}'}

            result = resp.json()
            if 'agents' in result and len(result['agents']) > 0:
                agent_count = len(result['agents'])
                return {'success': True, 'details': f"{agent_count} agents active"}
            else:
                return {'success': False, 'error': 'No agents found'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def run_all_tests(self):
        """Run all tests and provide real results"""
        print(f"Echo Brain REAL Functionality Test")
        print(f"Test ID: {self.test_id}")
        print(f"Started: {datetime.now()}")
        print("=" * 60)

        # Run tests in order
        self.test("Basic Connectivity", self.test_basic_connectivity)
        self.test("Knowledge Creation", self.test_knowledge_creation)
        self.test("Knowledge Retrieval", self.test_knowledge_retrieval)
        self.test("MCP Server", self.test_mcp_server)
        self.test("Query Processing", self.test_query_processing)
        self.test("Autonomous Tasks", self.test_autonomous_tasks)
        self.test("Agents Status", self.test_agents_status)

        # Calculate real results
        passed = len([r for r in self.results if r['status'] == 'PASS'])
        failed = len([r for r in self.results if r['status'] in ['FAIL', 'ERROR']])
        total = len(self.results)

        print("=" * 60)
        print(f"REAL RESULTS:")
        print(f"  PASSED: {passed}/{total} ({passed/total*100:.1f}%)")
        print(f"  FAILED: {failed}/{total}")
        print(f"  TOTAL TIME: {sum(r['duration'] for r in self.results):.2f}s")

        if passed == total:
            print("STATUS: ALL TESTS PASS - Actually Working")
        elif passed >= total * 0.8:
            print("STATUS: MOSTLY WORKING - Minor Issues")
        elif passed >= total * 0.5:
            print("STATUS: PARTIALLY WORKING - Major Issues")
        else:
            print("STATUS: MOSTLY BROKEN - Critical Failures")

        # Show failures
        failures = [r for r in self.results if r['status'] in ['FAIL', 'ERROR']]
        if failures:
            print(f"\nFAILURES:")
            for failure in failures:
                print(f"  ✗ {failure['name']}: {failure.get('error', 'Unknown error')}")

        return passed, total

    def cleanup(self):
        """Clean up test data"""
        try:
            # Try to delete the test fact we created
            resp = self.client.get(f"http://localhost:8309/api/knowledge/facts?subject=test_{self.test_id}")
            if resp.status_code == 200:
                facts = resp.json().get('facts', [])
                for fact in facts:
                    self.client.delete(f"http://localhost:8309/api/knowledge/facts/{fact['id']}")
        except:
            pass  # Cleanup is best effort

        self.client.close()

if __name__ == "__main__":
    tester = EchoBrainTester()
    try:
        passed, total = tester.run_all_tests()
        sys.exit(0 if passed == total else 1)
    finally:
        tester.cleanup()