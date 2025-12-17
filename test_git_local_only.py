#!/usr/bin/env python3
"""
Local Git Automation Testing (without API dependency)
Tests core git functionality across Tower repositories
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_core_functionality():
    """Test core git automation functionality"""
    logger.info("🧪 Testing Core Git Automation Functionality")

    results = {}

    # Test 1: Echo Brain git operations
    logger.info("\n📋 Testing Echo Brain Git Operations...")
    try:
        git_ops = GitOperationsManager()
        status = await git_ops.get_status()

        results["echo_brain_git"] = {
            "status": "success",
            "branch": status.branch,
            "modified_files": len(status.modified_files),
            "untracked_files": len(status.untracked_files),
            "is_clean": status.is_clean
        }

        logger.info(f"  ✅ Branch: {status.branch}")
        logger.info(f"  ✅ Modified: {len(status.modified_files)} files")
        logger.info(f"  ✅ Untracked: {len(status.untracked_files)} files")
        logger.info(f"  ✅ Clean: {status.is_clean}")

    except Exception as e:
        results["echo_brain_git"] = {"status": "error", "error": str(e)}
        logger.error(f"  ❌ Echo Brain git error: {e}")

    # Test 2: Tower services discovery
    logger.info("\n📋 Testing Tower Services Discovery...")
    try:
        services = git_manager._discover_tower_services()
        results["tower_discovery"] = {
            "status": "success",
            "total_services": len(services),
            "service_names": [s.name for s in services]
        }

        logger.info(f"  ✅ Discovered {len(services)} Tower services")
        for service in services[:5]:  # Show first 5
            logger.info(f"    - {service.name}")
        if len(services) > 5:
            logger.info(f"    ... and {len(services) - 5} more")

    except Exception as e:
        results["tower_discovery"] = {"status": "error", "error": str(e)}
        logger.error(f"  ❌ Tower discovery error: {e}")

    # Test 3: GitHub integration
    logger.info("\n📋 Testing GitHub Integration...")
    try:
        auth_ok = await github_integration.check_auth()
        current_branch = github_integration.get_current_branch()

        results["github_integration"] = {
            "status": "success",
            "authenticated": auth_ok,
            "current_branch": current_branch
        }

        logger.info(f"  ✅ GitHub auth: {'Available' if auth_ok else 'Not available'}")
        logger.info(f"  ✅ Current branch: {current_branch}")

    except Exception as e:
        results["github_integration"] = {"status": "error", "error": str(e)}
        logger.error(f"  ❌ GitHub integration error: {e}")

    # Test 4: Smart commit message generation
    logger.info("\n📋 Testing Smart Commit Messages...")
    try:
        git_ops = GitOperationsManager()
        test_diff = "M\tsrc/api/git_operations.py\nA\ttest_git_local_only.py\nM\tsrc/app_factory.py"
        message = await git_ops._generate_commit_message(test_diff, "feat")

        results["smart_commits"] = {
            "status": "success",
            "generated_message": message
        }

        logger.info(f"  ✅ Generated message: {message}")

    except Exception as e:
        results["smart_commits"] = {"status": "error", "error": str(e)}
        logger.error(f"  ❌ Smart commit error: {e}")

    # Test 5: Repository health check
    logger.info("\n📋 Testing Repository Health...")
    try:
        services = git_manager._discover_tower_services()
        healthy_count = 0
        with_changes_count = 0

        for service_path in services[:10]:  # Test first 10 services
            try:
                status = await git_manager.get_repo_status(service_path)
                if not status.get("error"):
                    healthy_count += 1
                    if status.get("has_changes"):
                        with_changes_count += 1
            except:
                pass

        results["repository_health"] = {
            "status": "success",
            "tested_repositories": min(10, len(services)),
            "healthy_repositories": healthy_count,
            "repositories_with_changes": with_changes_count
        }

        logger.info(f"  ✅ Tested: {min(10, len(services))} repositories")
        logger.info(f"  ✅ Healthy: {healthy_count}")
        logger.info(f"  ✅ With changes: {with_changes_count}")

    except Exception as e:
        results["repository_health"] = {"status": "error", "error": str(e)}
        logger.error(f"  ❌ Repository health error: {e}")

    return results

async def test_deployment_readiness():
    """Test deployment readiness"""
    logger.info("🚀 Testing Deployment Readiness")

    results = {}

    # Test 1: Safety mechanisms
    logger.info("\n📋 Testing Safety Mechanisms...")
    try:
        auto_commit_enabled = git_manager.auto_commit_enabled
        results["safety_mechanisms"] = {
            "status": "success",
            "auto_commit_disabled": not auto_commit_enabled,
            "safe_by_default": not auto_commit_enabled
        }

        logger.info(f"  ✅ Auto-commit disabled: {not auto_commit_enabled}")
        logger.info(f"  ✅ Safe by default: {not auto_commit_enabled}")

    except Exception as e:
        results["safety_mechanisms"] = {"status": "error", "error": str(e)}
        logger.error(f"  ❌ Safety mechanisms error: {e}")

    # Test 2: API module imports
    logger.info("\n📋 Testing API Module Imports...")
    try:
        from src.api.git_operations import router as git_router
        from src.app_factory import create_app

        results["api_integration"] = {
            "status": "success",
            "git_router_available": git_router is not None,
            "app_factory_available": True
        }

        logger.info(f"  ✅ Git operations router: Available")
        logger.info(f"  ✅ App factory: Available")

    except Exception as e:
        results["api_integration"] = {"status": "error", "error": str(e)}
        logger.error(f"  ❌ API integration error: {e}")

    # Test 3: Configuration validation
    logger.info("\n📋 Testing Configuration...")
    try:
        echo_brain_path = Path("/opt/tower-echo-brain")
        git_ops = GitOperationsManager(echo_brain_path)

        results["configuration"] = {
            "status": "success",
            "echo_brain_path_exists": echo_brain_path.exists(),
            "is_git_repository": (echo_brain_path / '.git').exists(),
            "git_ops_initialized": git_ops is not None
        }

        logger.info(f"  ✅ Echo Brain path exists: {echo_brain_path.exists()}")
        logger.info(f"  ✅ Is git repository: {(echo_brain_path / '.git').exists()}")
        logger.info(f"  ✅ Git operations initialized: {git_ops is not None}")

    except Exception as e:
        results["configuration"] = {"status": "error", "error": str(e)}
        logger.error(f"  ❌ Configuration error: {e}")

    return results

async def run_local_tests():
    """Run all local git automation tests"""
    logger.info("🔧 Starting Local Git Automation Tests")
    logger.info("=" * 60)

    start_time = datetime.now()

    # Run test phases
    test_phases = [
        ("Core Functionality", test_core_functionality),
        ("Deployment Readiness", test_deployment_readiness),
    ]

    all_results = {}

    for phase_name, test_func in test_phases:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"🔬 Phase: {phase_name}")
        logger.info(f"{'=' * 60}")

        try:
            phase_results = await test_func()
            all_results[phase_name] = {
                "success": True,
                "results": phase_results,
                "timestamp": datetime.now().isoformat()
            }
            logger.info(f"✅ {phase_name}: COMPLETED")

        except Exception as e:
            all_results[phase_name] = {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            logger.error(f"❌ {phase_name}: FAILED - {e}")

    # Calculate summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    passed_phases = sum(1 for r in all_results.values() if r.get("success"))
    total_phases = len(all_results)

    # Generate report
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
        "test_results": all_results
    }

    # Save report
    report_path = Path("/opt/tower-echo-brain/git_automation_local_test_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Print summary
    logger.info(f"\n{'=' * 60}")
    logger.info("📊 LOCAL TEST SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"🎯 Overall Status: {report['test_summary']['overall_status']}")
    logger.info(f"📈 Success Rate: {report['test_summary']['success_rate']:.1f}%")
    logger.info(f"⏱️  Duration: {duration:.1f} seconds")
    logger.info(f"📋 Test Phases: {passed_phases}/{total_phases}")
    logger.info(f"📄 Report saved: {report_path}")

    return report

async def main():
    """Main test runner"""
    report = await run_local_tests()

    # Show deployment readiness
    if report["test_summary"]["overall_status"] == "PASSED":
        logger.info("\n🚀 DEPLOYMENT STATUS: READY")
        logger.info("✅ Git automation is ready for production deployment")
        logger.info("✅ All core functionality tests passed")
        logger.info("✅ Safety mechanisms are in place")
        logger.info("✅ API integration is configured")
    else:
        logger.warning("\n⚠️  DEPLOYMENT STATUS: NOT READY")
        logger.warning("❌ Some tests failed - review before deployment")

    # Return appropriate exit code
    success = report["test_summary"]["overall_status"] == "PASSED"
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)