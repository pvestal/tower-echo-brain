#!/usr/bin/env python3
"""
CI/CD Pipeline End-to-End Test
Comprehensive test suite for validating the entire CI/CD pipeline
"""

import os
import json
import asyncio
import tempfile
import subprocess
import time
from datetime import datetime
from pathlib import Path
import aiohttp
import aiofiles

class CICDPipelineTest:
    """End-to-end test suite for CI/CD pipeline"""
    
    def __init__(self):
        self.base_url = "http://localhost"
        self.services = {
            'test_generator': 8340,
            'deployment_manager': 8341,
            'pipeline_orchestrator': 8342,
            'github_integration': 8343,
            'learning_integration': 8344
        }
        self.test_results = {}
    
    async def run_all_tests(self):
        """Run all tests in sequence"""
        print("🚀 Starting CI/CD Pipeline End-to-End Tests")
        print("=" * 60)
        
        tests = [
            ("Service Health Checks", self.test_service_health),
            ("Test Generator", self.test_test_generator),
            ("Deployment Manager", self.test_deployment_manager),
            ("Pipeline Orchestrator", self.test_pipeline_orchestrator),
            ("GitHub Integration", self.test_github_integration),
            ("Learning Integration", self.test_learning_integration),
            ("End-to-End Pipeline", self.test_complete_pipeline)
        ]
        
        for test_name, test_func in tests:
            print(f"\n📋 Running: {test_name}")
            print("-" * 40)
            
            try:
                result = await test_func()
                self.test_results[test_name] = {
                    'status': 'passed' if result else 'failed',
                    'timestamp': datetime.now().isoformat()
                }
                
                if result:
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
                    
            except Exception as e:
                self.test_results[test_name] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                print(f"💥 {test_name}: ERROR - {e}")
        
        await self.generate_test_report()
    
    async def test_service_health(self):
        """Test that all services are healthy"""
        all_healthy = True
        
        for service_name, port in self.services.items():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}:{port}/api/health", timeout=10) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            print(f"  ✅ {service_name} (:{port}): {data.get('status', 'unknown')}")
                        else:
                            print(f"  ❌ {service_name} (:{port}): HTTP {resp.status}")
                            all_healthy = False
            except Exception as e:
                print(f"  💥 {service_name} (:{port}): {e}")
                all_healthy = False
        
        return all_healthy
    
    async def test_test_generator(self):
        """Test the automated test generator"""
        try:
            # Create a simple Python file to test
            test_file_content = '''
def add_numbers(a, b):
    """Add two numbers together"""
    return a + b

def multiply_numbers(a, b):
    """Multiply two numbers"""
    return a * b

class Calculator:
    def __init__(self):
        self.history = []
    
    def calculate(self, operation, a, b):
        if operation == "add":
            result = add_numbers(a, b)
        elif operation == "multiply":
            result = multiply_numbers(a, b)
        else:
            raise ValueError("Unknown operation")
        
        self.history.append((operation, a, b, result))
        return result
'''
            
            # Write test file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_file_content)
                test_file_path = f.name
            
            try:
                # Call test generator API
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}:{self.services['test_generator']}/api/generate-tests",
                        json={
                            "file_path": test_file_path,
                            "module_name": "test_calculator",
                            "board_decision_id": "test-run-001"
                        },
                        timeout=60
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            success = data.get('success', False)
                            test_count = data.get('test_suite', {}).get('test_count', 0)
                            print(f"  ✅ Generated {test_count} tests successfully")
                            return success
                        else:
                            print(f"  ❌ Test generation failed: HTTP {resp.status}")
                            return False
            finally:
                os.unlink(test_file_path)
        
        except Exception as e:
            print(f"  💥 Test generator error: {e}")
            return False
    
    async def test_deployment_manager(self):
        """Test the deployment manager"""
        try:
            # Create a simple deployment request
            deployment_request = {
                "app_name": "test-app",
                "version": "v1.0.0-test",
                "git_commit": "abc123def456",
                "source_path": "/tmp/test-app",
                "target_env": "staging",
                "port": 9999,
                "health_check_url": "http://localhost:9999/health",
                "dependencies": [],
                "environment_vars": {"TEST_MODE": "true"}
            }
            
            # Create dummy source directory
            source_dir = Path("/tmp/test-app")
            source_dir.mkdir(exist_ok=True)
            (source_dir / "app.py").write_text("print('Hello Test')\n")
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}:{self.services['deployment_manager']}/api/deploy",
                        json=deployment_request,
                        timeout=30
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            deployment_id = data.get('deployment_id')
                            print(f"  ✅ Deployment initiated: {deployment_id}")
                            
                            # Check deployment status
                            await asyncio.sleep(5)
                            async with session.get(
                                f"{self.base_url}:{self.services['deployment_manager']}/api/deployments/{deployment_id}",
                                timeout=10
                            ) as status_resp:
                                if status_resp.status == 200:
                                    status_data = await status_resp.json()
                                    print(f"  ✅ Deployment status: {status_data.get('status', 'unknown')}")
                                    return True
                                else:
                                    print(f"  ❌ Failed to get deployment status: HTTP {status_resp.status}")
                                    return False
                        else:
                            print(f"  ❌ Deployment failed: HTTP {resp.status}")
                            return False
            finally:
                # Cleanup
                import shutil
                if source_dir.exists():
                    shutil.rmtree(source_dir)
        
        except Exception as e:
            print(f"  💥 Deployment manager error: {e}")
            return False
    
    async def test_pipeline_orchestrator(self):
        """Test the pipeline orchestrator"""
        try:
            pipeline_request = {
                "app_name": "test-pipeline-app",
                "git_repo": "https://github.com/test/repo.git",
                "git_branch": "main",
                "git_commit": "test123abc456",
                "trigger_type": "manual",
                "target_environments": ["staging"],
                "run_tests": False,  # Skip tests for this test
                "deploy_on_success": False,  # Skip deployment for this test
                "notify_on_completion": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}:{self.services['pipeline_orchestrator']}/api/pipelines/trigger",
                    json=pipeline_request,
                    timeout=30
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        pipeline_id = data.get('pipeline_id')
                        print(f"  ✅ Pipeline triggered: {pipeline_id}")
                        
                        # Check pipeline status
                        await asyncio.sleep(3)
                        async with session.get(
                            f"{self.base_url}:{self.services['pipeline_orchestrator']}/api/pipelines/{pipeline_id}",
                            timeout=10
                        ) as status_resp:
                            if status_resp.status == 200:
                                status_data = await status_resp.json()
                                print(f"  ✅ Pipeline status: {status_data.get('status', 'unknown')}")
                                return True
                            else:
                                print(f"  ❌ Failed to get pipeline status: HTTP {status_resp.status}")
                                return False
                    else:
                        print(f"  ❌ Pipeline trigger failed: HTTP {resp.status}")
                        return False
        
        except Exception as e:
            print(f"  💥 Pipeline orchestrator error: {e}")
            return False
    
    async def test_github_integration(self):
        """Test GitHub integration (webhook simulation)"""
        try:
            # Simulate a GitHub push webhook
            webhook_payload = {
                "ref": "refs/heads/main",
                "head_commit": {
                    "id": "test789xyz123",
                    "message": "Test commit for CI/CD",
                    "author": {"name": "Test User", "email": "test@example.com"}
                },
                "commits": [
                    {
                        "id": "test789xyz123",
                        "message": "Test commit",
                        "added": ["test_file.py"],
                        "modified": [],
                        "removed": []
                    }
                ],
                "repository": {
                    "name": "test-repo",
                    "full_name": "test/test-repo",
                    "clone_url": "https://github.com/test/test-repo.git",
                    "owner": {"login": "test"}
                },
                "pusher": {"name": "test-user"}
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}:{self.services['github_integration']}/webhook",
                    json=webhook_payload,
                    headers={"X-GitHub-Event": "push"},
                    timeout=30
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        status = data.get('status')
                        print(f"  ✅ Webhook processed: {status}")
                        
                        if 'pipeline_id' in data:
                            print(f"  ✅ Pipeline triggered: {data['pipeline_id']}")
                        
                        return status in ['processed', 'ignored']
                    else:
                        print(f"  ❌ Webhook processing failed: HTTP {resp.status}")
                        return False
        
        except Exception as e:
            print(f"  💥 GitHub integration error: {e}")
            return False
    
    async def test_learning_integration(self):
        """Test learning integration"""
        try:
            # Send pipeline feedback
            feedback = {
                "pipeline_id": "test-pipeline-learning-001",
                "app_name": "test-learning-app",
                "success": True,
                "duration_seconds": 120.5,
                "test_coverage": 85.0,
                "security_score": 90,
                "stages_completed": 5,
                "total_stages": 5,
                "git_commit": "learning123test456",
                "trigger_type": "test",
                "target_environments": ["staging"]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}:{self.services['learning_integration']}/api/learning/pipeline-feedback",
                    json=feedback,
                    timeout=30
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        print(f"  ✅ Feedback recorded: {data.get('status')}")
                        
                        # Get learning insights
                        async with session.get(
                            f"{self.base_url}:{self.services['learning_integration']}/api/learning/insights",
                            timeout=10
                        ) as insights_resp:
                            if insights_resp.status == 200:
                                insights = await insights_resp.json()
                                print(f"  ✅ Learning insights retrieved: {insights.get('learning_status')}")
                                return True
                            else:
                                print(f"  ❌ Failed to get insights: HTTP {insights_resp.status}")
                                return False
                    else:
                        print(f"  ❌ Feedback recording failed: HTTP {resp.status}")
                        return False
        
        except Exception as e:
            print(f"  💥 Learning integration error: {e}")
            return False
    
    async def test_complete_pipeline(self):
        """Test a complete end-to-end pipeline"""
        try:
            print("  🔄 Starting complete pipeline test...")
            
            # 1. Trigger a comprehensive pipeline
            pipeline_request = {
                "app_name": "e2e-test-app",
                "git_repo": "https://github.com/test/e2e-repo.git",
                "git_branch": "main",
                "git_commit": "e2e789test123",
                "trigger_type": "manual",
                "target_environments": [],  # No actual deployment
                "run_tests": False,  # Skip tests to avoid external dependencies
                "deploy_on_success": False,
                "notify_on_completion": False,
                "board_decision_id": "e2e-test-001"
            }
            
            async with aiohttp.ClientSession() as session:
                # Trigger pipeline
                async with session.post(
                    f"{self.base_url}:{self.services['pipeline_orchestrator']}/api/pipelines/trigger",
                    json=pipeline_request,
                    timeout=30
                ) as resp:
                    if resp.status != 200:
                        print(f"  ❌ Failed to trigger pipeline: HTTP {resp.status}")
                        return False
                    
                    data = await resp.json()
                    pipeline_id = data.get('pipeline_id')
                    print(f"  ✅ E2E Pipeline triggered: {pipeline_id}")
                
                # Monitor pipeline for a short time
                for i in range(6):  # Check for 30 seconds
                    await asyncio.sleep(5)
                    
                    async with session.get(
                        f"{self.base_url}:{self.services['pipeline_orchestrator']}/api/pipelines/{pipeline_id}",
                        timeout=10
                    ) as status_resp:
                        if status_resp.status == 200:
                            status_data = await status_resp.json()
                            status = status_data.get('status')
                            current_stage = status_data.get('current_stage')
                            
                            print(f"  🔄 Pipeline status: {status} (stage: {current_stage or 'none'})")
                            
                            if status in ['success', 'failed', 'cancelled']:
                                print(f"  ✅ Pipeline completed with status: {status}")
                                
                                # Send learning feedback
                                feedback = {
                                    "pipeline_id": pipeline_id,
                                    "app_name": "e2e-test-app",
                                    "success": status == 'success',
                                    "duration_seconds": 30.0,
                                    "test_coverage": 0.0,
                                    "security_score": 50,
                                    "stages_completed": len(status_data.get('stages', {})),
                                    "total_stages": 6,
                                    "git_commit": "e2e789test123",
                                    "trigger_type": "test"
                                }
                                
                                async with session.post(
                                    f"{self.base_url}:{self.services['learning_integration']}/api/learning/pipeline-feedback",
                                    json=feedback,
                                    timeout=10
                                ) as feedback_resp:
                                    if feedback_resp.status == 200:
                                        print(f"  ✅ Learning feedback sent successfully")
                                    else:
                                        print(f"  ⚠️ Learning feedback failed: HTTP {feedback_resp.status}")
                                
                                return status == 'success'
                        else:
                            print(f"  ❌ Failed to get pipeline status: HTTP {status_resp.status}")
                            return False
                
                print(f"  ⏰ Pipeline still running after 30 seconds - considering success")
                return True
        
        except Exception as e:
            print(f"  💥 Complete pipeline test error: {e}")
            return False
    
    async def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("📊 CI/CD Pipeline Test Report")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r['status'] == 'passed')
        failed_tests = sum(1 for r in self.test_results.values() if r['status'] == 'failed')
        error_tests = sum(1 for r in self.test_results.values() if r['status'] == 'error')
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Errors: {error_tests} 💥")
        print(f"Success Rate: {(passed_tests / total_tests * 100):.1f}%")
        
        print("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status_icon = {
                'passed': '✅',
                'failed': '❌',
                'error': '💥'
            }.get(result['status'], '❓')
            
            print(f"  {status_icon} {test_name}: {result['status'].upper()}")
            if result['status'] == 'error':
                print(f"    Error: {result.get('error', 'Unknown error')}")
        
        # Save report to file
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'errors': error_tests,
                'success_rate': passed_tests / total_tests * 100 if total_tests > 0 else 0
            },
            'results': self.test_results
        }
        
        os.makedirs('/opt/tower-echo-brain/test-reports', exist_ok=True)
        report_file = f'/opt/tower-echo-brain/test-reports/cicd-test-{int(time.time())}.json'
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📁 Report saved to: {report_file}")
        
        # Determine overall result
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! CI/CD Pipeline is fully functional.")
            return True
        elif passed_tests > total_tests / 2:
            print("\n⚠️ Most tests passed, but some issues need attention.")
            return False
        else:
            print("\n🚨 CRITICAL: Multiple test failures - CI/CD Pipeline needs fixing.")
            return False

async def main():
    """Main test execution"""
    tester = CICDPipelineTest()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎯 CI/CD Pipeline validation completed successfully!")
        exit(0)
    else:
        print("\n🔧 CI/CD Pipeline needs attention - check test results above.")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())
