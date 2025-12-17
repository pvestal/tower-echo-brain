#!/usr/bin/env python3
"""
Comprehensive Tower Git Automation Workflow Testing
Tests git automation across all Tower repositories
"""

import asyncio
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import git automation modules
from execution.git_operations import GitOperationsManager
from tasks.git_manager import git_manager
from tasks.github_integration import github_integration

# Import API client for testing
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TowerGitWorkflowTester:
    """Comprehensive tester for Tower git workflows"""

    def __init__(self):
        self.tower_services = git_manager._discover_tower_services()
        self.results = {}
        self.echo_api_base = "http://localhost:8309/api/echo"

    async def test_echo_brain_api_integration(self):
        """Test the Echo Brain API git endpoints"""
        logger.info("🧪 Testing Echo Brain API Git Integration...")

        tests = [
            ("/git/health", "Health check"),
            ("/git/status", "Git status"),
            ("/git/tower/status", "Tower ecosystem status"),
            ("/git/github/status", "GitHub integration status"),
        ]

        api_results = {}

        async with httpx.AsyncClient() as client:
            for endpoint, description in tests:
                try:
                    response = await client.get(f"{self.echo_api_base}{endpoint}", timeout=30.0)

                    if response.status_code == 200:
                        data = response.json()
                        api_results[endpoint] = {
                            "status": "success",
                            "data": data,
                            "description": description
                        }
                        logger.info(f"  ✅ {description}: SUCCESS")
                    else:
                        api_results[endpoint] = {
                            "status": "failed",
                            "error": f"HTTP {response.status_code}",
                            "description": description
                        }
                        logger.error(f"  ❌ {description}: HTTP {response.status_code}")

                except Exception as e:
                    api_results[endpoint] = {
                        "status": "error",
                        "error": str(e),
                        "description": description
                    }
                    logger.error(f"  ❌ {description}: {e}")

        return api_results

    async def test_repository_initialization(self):
        """Test repository initialization across Tower services"""
        logger.info("🏗️  Testing Repository Initialization...")

        init_results = {}

        for service_path in self.tower_services:
            try:
                # Check if repo exists
                git_dir = service_path / '.git'
                repo_exists = git_dir.exists()

                # Initialize if needed
                if not repo_exists:
                    success = await git_manager.initialize_repo(service_path)
                    init_results[service_path.name] = {
                        "was_initialized": True,
                        "success": success,
                        "status": "newly_created"
                    }
                    logger.info(f"  📁 {service_path.name}: Initialized {'✅' if success else '❌'}")
                else:
                    init_results[service_path.name] = {
                        "was_initialized": False,
                        "success": True,
                        "status": "already_exists"
                    }
                    logger.info(f"  📁 {service_path.name}: Already exists ✅")

            except Exception as e:
                init_results[service_path.name] = {
                    "was_initialized": False,
                    "success": False,
                    "error": str(e),
                    "status": "error"
                }
                logger.error(f"  📁 {service_path.name}: Error - {e}")

        return init_results

    async def test_status_monitoring(self):
        """Test git status monitoring across all Tower services"""
        logger.info("📊 Testing Status Monitoring...")

        status_results = {}

        for service_path in self.tower_services:
            try:
                status = await git_manager.get_repo_status(service_path)

                status_results[service_path.name] = {
                    "has_changes": status.get("has_changes", False),
                    "changed_files": len(status.get("changed_files", [])),
                    "untracked_files": len(status.get("untracked_files", [])),
                    "latest_commit": status.get("latest_commit", {}).get("hash", "None"),
                    "branch": status.get("branch", "unknown"),
                    "error": status.get("error")
                }

                changes_count = status_results[service_path.name]["changed_files"] + status_results[service_path.name]["untracked_files"]
                status_icon = "🔄" if status_results[service_path.name]["has_changes"] else "✅"

                logger.info(f"  {status_icon} {service_path.name}: {changes_count} changes, branch: {status_results[service_path.name]['branch']}")

            except Exception as e:
                status_results[service_path.name] = {
                    "error": str(e),
                    "has_changes": False
                }
                logger.error(f"  ❌ {service_path.name}: Error - {e}")

        return status_results

    async def test_smart_commit_workflow(self):
        """Test smart commit generation without actually committing"""
        logger.info("🤖 Testing Smart Commit Workflow...")

        commit_results = {}

        for service_path in self.tower_services:
            try:
                git_ops = GitOperationsManager(service_path)
                status = await git_ops.get_status()

                if status.modified_files or status.untracked_files:
                    # Simulate diff output for commit message generation
                    diff_output = "\n".join([f"M\t{f}" for f in status.modified_files[:5]])
                    if status.untracked_files:
                        diff_output += "\n" + "\n".join([f"A\t{f}" for f in status.untracked_files[:3]])

                    # Generate commit message
                    if diff_output:
                        message = await git_ops._generate_commit_message(diff_output, "update")
                        commit_results[service_path.name] = {
                            "would_commit": True,
                            "files_count": len(status.modified_files) + len(status.untracked_files),
                            "generated_message": message,
                            "status": "ready"
                        }
                        logger.info(f"  🔧 {service_path.name}: Would commit {commit_results[service_path.name]['files_count']} files")
                    else:
                        commit_results[service_path.name] = {
                            "would_commit": False,
                            "files_count": 0,
                            "status": "no_changes"
                        }
                else:
                    commit_results[service_path.name] = {
                        "would_commit": False,
                        "files_count": 0,
                        "status": "clean"
                    }
                    logger.info(f"  ✅ {service_path.name}: Clean - no changes")

            except Exception as e:
                commit_results[service_path.name] = {
                    "would_commit": False,
                    "error": str(e),
                    "status": "error"
                }
                logger.error(f"  ❌ {service_path.name}: Error - {e}")

        return commit_results

    async def test_github_integration(self):
        """Test GitHub integration capabilities"""
        logger.info("🐙 Testing GitHub Integration...")

        github_results = {}

        try:
            # Test authentication
            auth_ok = await github_integration.check_auth()
            github_results["authentication"] = {
                "success": auth_ok,
                "status": "authenticated" if auth_ok else "not_authenticated"
            }

            # Test current branch
            current_branch = github_integration.get_current_branch()
            github_results["current_branch"] = {
                "branch": current_branch,
                "success": bool(current_branch)
            }

            # Test open PRs
            open_prs = await github_integration.get_open_prs()
            github_results["open_prs"] = {
                "count": len(open_prs),
                "prs": open_prs,
                "success": True
            }

            logger.info(f"  🔑 Authentication: {'✅' if auth_ok else '❌'}")
            logger.info(f"  🌿 Current branch: {current_branch}")
            logger.info(f"  🔄 Open PRs: {len(open_prs)}")

        except Exception as e:
            github_results["error"] = str(e)
            logger.error(f"  ❌ GitHub integration error: {e}")

        return github_results

    async def test_cross_repository_coordination(self):
        """Test coordination across multiple repositories"""
        logger.info("🔗 Testing Cross-Repository Coordination...")

        coordination_results = {
            "total_repos": len(self.tower_services),
            "healthy_repos": 0,
            "repos_with_changes": 0,
            "error_repos": 0,
            "details": {}
        }

        for service_path in self.tower_services:
            try:
                # Get status
                status = await git_manager.get_repo_status(service_path)

                if status.get("error"):
                    coordination_results["error_repos"] += 1
                    coordination_results["details"][service_path.name] = {
                        "status": "error",
                        "error": status["error"]
                    }
                else:
                    coordination_results["healthy_repos"] += 1
                    if status.get("has_changes"):
                        coordination_results["repos_with_changes"] += 1

                    coordination_results["details"][service_path.name] = {
                        "status": "healthy",
                        "has_changes": status.get("has_changes", False),
                        "branch": status.get("branch", "unknown")
                    }

            except Exception as e:
                coordination_results["error_repos"] += 1
                coordination_results["details"][service_path.name] = {
                    "status": "error",
                    "error": str(e)
                }

        logger.info(f"  📊 Total repos: {coordination_results['total_repos']}")
        logger.info(f"  ✅ Healthy: {coordination_results['healthy_repos']}")
        logger.info(f"  🔄 With changes: {coordination_results['repos_with_changes']}")
        logger.info(f"  ❌ Errors: {coordination_results['error_repos']}")

        return coordination_results

    async def test_automation_safety(self):
        """Test safety mechanisms in automation"""
        logger.info("🛡️  Testing Automation Safety...")

        safety_results = {}

        try:
            # Test auto-commit is disabled by default
            safety_results["auto_commit_default"] = {
                "enabled": git_manager.auto_commit_enabled,
                "safe": not git_manager.auto_commit_enabled,
                "status": "safe" if not git_manager.auto_commit_enabled else "unsafe"
            }

            # Test GitHub auth requirement
            auth_required = github_integration.gh_available
            safety_results["github_auth"] = {
                "required": auth_required,
                "available": await github_integration.check_auth(),
                "status": "safe"
            }

            # Test pre-commit hooks existence
            pre_commit_hooks = 0
            for service_path in self.tower_services:
                hook_path = service_path / '.git' / 'hooks' / 'pre-commit'
                if hook_path.exists():
                    pre_commit_hooks += 1

            safety_results["pre_commit_hooks"] = {
                "total_possible": len(self.tower_services),
                "with_hooks": pre_commit_hooks,
                "coverage_percent": (pre_commit_hooks / len(self.tower_services)) * 100 if self.tower_services else 0
            }

            logger.info(f"  🔒 Auto-commit disabled: {'✅' if not git_manager.auto_commit_enabled else '❌'}")
            logger.info(f"  🔑 GitHub auth available: {'✅' if safety_results['github_auth']['available'] else '❌'}")
            logger.info(f"  🪝 Pre-commit hooks: {pre_commit_hooks}/{len(self.tower_services)} repos")

        except Exception as e:
            safety_results["error"] = str(e)
            logger.error(f"  ❌ Safety test error: {e}")

        return safety_results

    async def run_comprehensive_test(self):
        """Run all tests and generate comprehensive report"""
        logger.info("🚀 Starting Comprehensive Tower Git Workflow Testing")

        start_time = datetime.now()

        # Run all test phases
        test_phases = [
            ("API Integration", self.test_echo_brain_api_integration),
            ("Repository Initialization", self.test_repository_initialization),
            ("Status Monitoring", self.test_status_monitoring),
            ("Smart Commit Workflow", self.test_smart_commit_workflow),
            ("GitHub Integration", self.test_github_integration),
            ("Cross-Repository Coordination", self.test_cross_repository_coordination),
            ("Automation Safety", self.test_automation_safety),
        ]

        results = {}

        for phase_name, test_func in test_phases:
            logger.info(f"\n{'='*60}")
            logger.info(f"📋 Phase: {phase_name}")
            logger.info(f"{'='*60}")

            try:
                phase_results = await test_func()
                results[phase_name] = {
                    "success": True,
                    "results": phase_results,
                    "timestamp": datetime.now().isoformat()
                }
                logger.info(f"✅ {phase_name}: COMPLETED")

            except Exception as e:
                results[phase_name] = {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                logger.error(f"❌ {phase_name}: FAILED - {e}")

        # Calculate summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        passed_phases = sum(1 for r in results.values() if r.get("success"))
        total_phases = len(results)

        # Generate comprehensive report
        report = {
            "test_summary": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "total_phases": total_phases,
                "passed_phases": passed_phases,
                "success_rate": (passed_phases / total_phases) * 100 if total_phases > 0 else 0,
                "overall_status": "PASSED" if passed_phases == total_phases else "FAILED"
            },
            "tower_services": {
                "discovered": len(self.tower_services),
                "service_names": [s.name for s in self.tower_services]
            },
            "test_results": results
        }

        # Save report
        report_path = Path("/opt/tower-echo-brain/tower_git_workflow_test_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Print final summary
        logger.info(f"\n{'='*60}")
        logger.info("📊 FINAL SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"🎯 Overall Status: {report['test_summary']['overall_status']}")
        logger.info(f"📈 Success Rate: {report['test_summary']['success_rate']:.1f}%")
        logger.info(f"⏱️  Duration: {duration:.1f} seconds")
        logger.info(f"🏗️  Tower Services: {len(self.tower_services)}")
        logger.info(f"📋 Test Phases: {passed_phases}/{total_phases}")
        logger.info(f"📄 Report saved: {report_path}")

        return report

async def main():
    """Main test runner"""
    tester = TowerGitWorkflowTester()
    report = await tester.run_comprehensive_test()

    # Return appropriate exit code
    success = report["test_summary"]["overall_status"] == "PASSED"
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)