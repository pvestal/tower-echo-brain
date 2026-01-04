#!/usr/bin/env python3
"""
Master test runner for Echo Brain and Telegram memory tests
Runs all test suites and generates a comprehensive report
"""

import asyncio
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from colorama import Fore, Style, init

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test modules
from tests.test_conversation_memory import ConversationMemoryTester
from tests.test_database_memory import DatabaseMemoryTester

init(autoreset=True)

class MemoryTestRunner:
    """Comprehensive test runner for all memory systems"""

    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
        self.report_path = Path("/opt/tower-echo-brain/tests/reports")
        self.report_path.mkdir(exist_ok=True)

    async def run_all_tests(self):
        """Run complete test suite"""
        self.start_time = datetime.now()

        print(f"\n{Fore.MAGENTA}{'='*70}")
        print(f"{Fore.MAGENTA}🧠 ECHO BRAIN & TELEGRAM MEMORY TEST SUITE")
        print(f"{Fore.MAGENTA}{'='*70}")
        print(f"\n{Fore.CYAN}Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Run conversation memory tests
        print(f"\n{Fore.BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"{Fore.BLUE}Running Conversation Memory Tests...")
        print(f"{Fore.BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        conv_tester = ConversationMemoryTester()
        await conv_tester.run_all_tests()
        self.results['conversation_memory'] = conv_tester.test_results

        # Run database memory tests
        print(f"\n{Fore.BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"{Fore.BLUE}Running Database Memory Tests...")
        print(f"{Fore.BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        db_tester = DatabaseMemoryTester()
        await db_tester.run_all_tests()
        self.results['database_memory'] = db_tester.test_results

        # Run integration tests
        print(f"\n{Fore.BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"{Fore.BLUE}Running Integration Tests...")
        print(f"{Fore.BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        await self.run_integration_tests()

        self.end_time = datetime.now()

        # Generate and display report
        self.generate_report()

    async def run_integration_tests(self):
        """Test integration between different memory systems"""
        import httpx

        integration_results = []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Test 1: Echo Brain health check
                print(f"\n{Fore.YELLOW}[INTEGRATION 1] Echo Brain Health Check")
                response = await client.get("http://localhost:8309/api/echo/health")

                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "healthy":
                        print(f"{Fore.GREEN}✓ Echo Brain is healthy")
                        integration_results.append(("Echo Brain Health", True, "Service healthy"))
                    else:
                        print(f"{Fore.RED}✗ Echo Brain unhealthy: {data}")
                        integration_results.append(("Echo Brain Health", False, f"Unhealthy: {data.get('status')}"))
                else:
                    print(f"{Fore.RED}✗ Echo Brain not responding")
                    integration_results.append(("Echo Brain Health", False, "Not responding"))

                # Test 2: Database connectivity
                print(f"\n{Fore.YELLOW}[INTEGRATION 2] Database Connectivity")
                response = await client.get("http://localhost:8309/api/echo/metrics/db")

                if response.status_code == 200:
                    data = response.json()
                    if data.get("database_status") == "healthy":
                        print(f"{Fore.GREEN}✓ Database connection healthy")
                        print(f"  - Conversations: {data.get('total_conversations', 0)}")
                        print(f"  - Vector memories: {data.get('vector_memories', 0)}")
                        integration_results.append(("Database Connection", True, "Connected"))
                    else:
                        print(f"{Fore.YELLOW}⚠ Database status unknown")
                        integration_results.append(("Database Connection", False, "Unknown status"))
                else:
                    print(f"{Fore.RED}✗ Could not check database status")
                    integration_results.append(("Database Connection", False, "Check failed"))

                # Test 3: Memory pipeline
                print(f"\n{Fore.YELLOW}[INTEGRATION 3] Memory Pipeline End-to-End")

                # Send a test message
                test_id = f"pipeline_{datetime.now().timestamp()}"
                response1 = await client.post(
                    "http://localhost:8309/api/echo/query",
                    json={
                        "query": f"Test pipeline message {test_id}",
                        "conversation_id": f"pipeline_test_{test_id}",
                        "username": "pipeline_tester"
                    }
                )

                if response1.status_code == 200:
                    # Wait for processing
                    await asyncio.sleep(2)

                    # Try to retrieve it
                    response2 = await client.post(
                        "http://localhost:8309/api/echo/query",
                        json={
                            "query": f"What was the pipeline test ID?",
                            "conversation_id": f"pipeline_recall_{test_id}",
                            "username": "pipeline_tester"
                        }
                    )

                    if response2.status_code == 200:
                        data = response2.json()
                        if test_id in str(data.get("response", "")):
                            print(f"{Fore.GREEN}✓ End-to-end memory pipeline working")
                            integration_results.append(("Memory Pipeline", True, "Complete pipeline working"))
                        else:
                            print(f"{Fore.YELLOW}⚠ Pipeline processed but recall failed")
                            integration_results.append(("Memory Pipeline", False, "Recall failed"))
                    else:
                        print(f"{Fore.RED}✗ Pipeline recall failed")
                        integration_results.append(("Memory Pipeline", False, "Recall error"))
                else:
                    print(f"{Fore.RED}✗ Pipeline input failed")
                    integration_results.append(("Memory Pipeline", False, "Input error"))

        except Exception as e:
            print(f"{Fore.RED}✗ Integration tests failed: {str(e)}")
            integration_results.append(("Integration Tests", False, str(e)))

        self.results['integration'] = integration_results

    def generate_report(self):
        """Generate comprehensive test report"""
        duration = (self.end_time - self.start_time).total_seconds()

        # Calculate statistics
        total_passed = 0
        total_failed = 0
        total_skipped = 0

        for category, results in self.results.items():
            for _, status, _ in results:
                if status is True:
                    total_passed += 1
                elif status is False:
                    total_failed += 1
                else:
                    total_skipped += 1

        total_tests = total_passed + total_failed + total_skipped
        pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

        # Display summary
        print(f"\n{Fore.MAGENTA}{'='*70}")
        print(f"{Fore.MAGENTA}TEST EXECUTION SUMMARY")
        print(f"{Fore.MAGENTA}{'='*70}\n")

        print(f"{Fore.CYAN}Duration: {duration:.2f} seconds")
        print(f"{Fore.CYAN}Total Tests: {total_tests}")
        print(f"{Fore.GREEN}Passed: {total_passed}")
        print(f"{Fore.RED}Failed: {total_failed}")
        print(f"{Fore.YELLOW}Skipped: {total_skipped}")
        print(f"{Fore.CYAN}Pass Rate: {pass_rate:.1f}%")

        # Detailed results by category
        print(f"\n{Fore.MAGENTA}DETAILED RESULTS BY CATEGORY:")
        print(f"{Fore.MAGENTA}{'─'*70}\n")

        for category, results in self.results.items():
            print(f"{Fore.BLUE}{category.replace('_', ' ').title()}:")
            for test_name, status, message in results:
                if status is True:
                    icon = f"{Fore.GREEN}✓"
                elif status is False:
                    icon = f"{Fore.RED}✗"
                else:
                    icon = f"{Fore.YELLOW}○"
                print(f"  {icon} {test_name}: {message}")
            print()

        # Save report to file
        report_file = self.report_path / f"memory_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_data = {
            "timestamp": self.start_time.isoformat(),
            "duration_seconds": duration,
            "summary": {
                "total_tests": total_tests,
                "passed": total_passed,
                "failed": total_failed,
                "skipped": total_skipped,
                "pass_rate": pass_rate
            },
            "results": self.results
        }

        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        print(f"{Fore.CYAN}Report saved to: {report_file}")

        # Return exit code based on failures
        if total_failed > 0:
            print(f"\n{Fore.RED}❌ TEST SUITE FAILED - {total_failed} tests failed")
            return 1
        else:
            print(f"\n{Fore.GREEN}✅ TEST SUITE PASSED - All tests successful!")
            return 0


async def main():
    """Main entry point"""
    runner = MemoryTestRunner()
    exit_code = await runner.run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())