#!/usr/bin/env python3
"""
Comprehensive Git/GitHub Integration Test Suite
Tests multi-repository git operations and GitHub features
"""
import httpx
import json
import time
import uuid
import sys
from datetime import datetime

class GitHubIntegrationTester:
    def __init__(self):
        self.client = httpx.Client(timeout=30.0)
        self.results = []
        self.test_id = str(uuid.uuid4())[:8]

    def test(self, name, func):
        """Run a test and record results"""
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

    def test_git_health(self):
        """Test git system health"""
        try:
            resp = self.client.get("http://localhost:8309/git/health")
            if resp.status_code == 200:
                result = resp.json()
                return {
                    'success': True,
                    'details': f"Git: {result.get('git_operations')}, GitHub: {result.get('github_auth')}, Services: {result.get('tower_services')}"
                }
            else:
                return {'success': False, 'error': f'HTTP {resp.status_code}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_tower_services_status(self):
        """Test tower ecosystem git status"""
        try:
            resp = self.client.get("http://localhost:8309/git/tower/status")
            if resp.status_code == 200:
                result = resp.json()
                total_services = result.get('total_services', 0)
                services = result.get('services', {})

                working_services = sum(1 for s in services.values() if s.get('error') is None)
                error_services = total_services - working_services

                return {
                    'success': working_services > 0,
                    'details': f"Services: {working_services} working, {error_services} with errors out of {total_services}"
                }
            else:
                return {'success': False, 'error': f'HTTP {resp.status_code}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_echo_brain_git_status(self):
        """Test git status on main echo brain repo"""
        try:
            resp = self.client.get("http://localhost:8309/git/status")
            if resp.status_code == 200:
                result = resp.json()
                status = result.get('status', {})
                return {
                    'success': True,
                    'details': f"Branch: {status.get('branch')}, Changes: {len(status.get('modified_files', []))}, Clean: {status.get('is_clean')}"
                }
            else:
                return {'success': False, 'error': f'HTTP {resp.status_code}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_github_integration_status(self):
        """Test GitHub integration and authentication"""
        try:
            resp = self.client.get("http://localhost:8309/git/github/status")
            if resp.status_code == 200:
                result = resp.json()
                auth_status = result.get('github_auth', False)
                current_branch = result.get('current_branch', 'unknown')
                open_prs = result.get('open_prs', 0)

                return {
                    'success': True,
                    'details': f"Auth: {auth_status}, Branch: {current_branch}, Open PRs: {open_prs}"
                }
            else:
                return {'success': False, 'error': f'HTTP {resp.status_code}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_tower_sync_dry_run(self):
        """Test tower repository sync (dry run)"""
        try:
            resp = self.client.post(
                "http://localhost:8309/git/tower/sync",
                json={"enable_auto_commit": False, "services": ["tower-echo-brain"]}
            )
            if resp.status_code == 200:
                result = resp.json()
                services_synced = result.get('services_synced', 0)
                results = result.get('results', {})

                success_count = sum(1 for r in results.values() if r.get('status') == 'success')

                return {
                    'success': services_synced > 0,
                    'details': f"Synced {services_synced} services, {success_count} successful"
                }
            else:
                return {'success': False, 'error': f'HTTP {resp.status_code}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_git_automation_status(self):
        """Test git automation features"""
        try:
            # Test enabling automation
            resp = self.client.post("http://localhost:8309/git/automation/enable")
            if resp.status_code != 200:
                return {'success': False, 'error': f'Enable failed: HTTP {resp.status_code}'}

            # Test disabling automation
            resp = self.client.post("http://localhost:8309/git/automation/disable")
            if resp.status_code == 200:
                result = resp.json()
                return {
                    'success': True,
                    'details': f"Automation: {result.get('status')}"
                }
            else:
                return {'success': False, 'error': f'Disable failed: HTTP {resp.status_code}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_git_logs_access(self):
        """Test git automation logs"""
        try:
            resp = self.client.get("http://localhost:8309/git/logs")
            if resp.status_code == 200:
                result = resp.json()
                logs = result.get('logs', [])
                total_entries = result.get('total_entries', 0)

                return {
                    'success': True,
                    'details': f"Logs accessible, {total_entries} entries available"
                }
            else:
                return {'success': False, 'error': f'HTTP {resp.status_code}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_service_coordination(self):
        """Test service coordination API"""
        try:
            resp = self.client.get("http://localhost:8309/api/coordination/services")
            if resp.status_code == 200:
                result = resp.json()
                services = result.get('services', [])
                running_services = [s for s in services if s.get('status') == 'running']

                return {
                    'success': len(running_services) > 0,
                    'details': f"{len(running_services)} services running out of {len(services)}"
                }
            else:
                return {'success': False, 'error': f'HTTP {resp.status_code}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def run_comprehensive_tests(self):
        """Run all git/GitHub integration tests"""
        print(f"Comprehensive Git/GitHub Integration Test")
        print(f"Test ID: {self.test_id}")
        print(f"Started: {datetime.now()}")
        print("=" * 70)

        # Run all tests
        self.test("Git System Health", self.test_git_health)
        self.test("Echo Brain Git Status", self.test_echo_brain_git_status)
        self.test("Tower Services Git Status", self.test_tower_services_status)
        self.test("GitHub Integration Status", self.test_github_integration_status)
        self.test("Tower Sync (Dry Run)", self.test_tower_sync_dry_run)
        self.test("Git Automation Features", self.test_git_automation_status)
        self.test("Git Logs Access", self.test_git_logs_access)
        self.test("Service Coordination", self.test_service_coordination)

        # Calculate results
        passed = len([r for r in self.results if r['status'] == 'PASS'])
        failed = len([r for r in self.results if r['status'] in ['FAIL', 'ERROR']])
        total = len(self.results)

        print("=" * 70)
        print(f"COMPREHENSIVE GIT/GITHUB TEST RESULTS:")
        print(f"  PASSED: {passed}/{total} ({passed/total*100:.1f}%)")
        print(f"  FAILED: {failed}/{total}")
        print(f"  TOTAL TIME: {sum(r['duration'] for r in self.results):.2f}s")

        # Status assessment
        if passed == total:
            print("STATUS: ALL TESTS PASS - Git/GitHub Fully Functional")
        elif passed >= total * 0.8:
            print("STATUS: MOSTLY WORKING - Minor Git/GitHub Issues")
        elif passed >= total * 0.5:
            print("STATUS: PARTIALLY WORKING - Major Git/GitHub Issues")
        else:
            print("STATUS: CRITICAL FAILURES - Git/GitHub System Broken")

        # Show failures
        failures = [r for r in self.results if r['status'] in ['FAIL', 'ERROR']]
        if failures:
            print(f"\nFAILED TESTS:")
            for failure in failures:
                print(f"  ✗ {failure['name']}: {failure.get('error', 'Unknown error')}")

        return passed, total

    def cleanup(self):
        """Clean up test resources"""
        self.client.close()

if __name__ == "__main__":
    tester = GitHubIntegrationTester()
    try:
        passed, total = tester.run_comprehensive_tests()
        sys.exit(0 if passed == total else 1)
    finally:
        tester.cleanup()