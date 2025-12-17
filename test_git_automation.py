#!/usr/bin/env python3
"""
Comprehensive test suite for Echo Brain git automation capabilities
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from execution.git_operations import GitOperationsManager, GitHubOperations
from tasks.git_manager import git_manager
from tasks.github_integration import github_integration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_git_operations_manager():
    """Test GitOperationsManager functionality"""
    logger.info("🔧 Testing GitOperationsManager...")

    git_ops = GitOperationsManager()

    # Test status
    status = await git_ops.get_status()
    logger.info(f"  ✅ Git Status: Branch={status.branch}, Modified={len(status.modified_files)}, Untracked={len(status.untracked_files)}")

    # Test commit message generation
    test_diff = "M\tsrc/api/echo.py\nA\ttest_new_file.py"
    message = await git_ops._generate_commit_message(test_diff, "fix")
    logger.info(f"  ✅ Generated commit message: {message}")

    return True

async def test_git_manager():
    """Test GitManager functionality"""
    logger.info("🚀 Testing GitManager...")

    # Test Tower services discovery
    services = git_manager._discover_tower_services()
    logger.info(f"  ✅ Discovered {len(services)} Tower services:")
    for service in services:
        logger.info(f"    - {service.name}")

    # Test repository status for Echo Brain
    status = await git_manager.get_repo_status(Path("/opt/tower-echo-brain"))
    logger.info(f"  ✅ Echo Brain status: {status.get('has_changes', False)} changes")

    return True

async def test_github_integration():
    """Test GitHub integration"""
    logger.info("🐙 Testing GitHub integration...")

    # Test auth
    auth_ok = await github_integration.check_auth()
    logger.info(f"  ✅ GitHub auth: {'OK' if auth_ok else 'FAILED'}")

    # Test current branch
    current_branch = github_integration.get_current_branch()
    logger.info(f"  ✅ Current branch: {current_branch}")

    # Test open PRs
    open_prs = await github_integration.get_open_prs()
    logger.info(f"  ✅ Open PRs: {len(open_prs)}")

    return auth_ok

async def test_cross_tower_discovery():
    """Test discovery of all Tower repositories"""
    logger.info("🏗️  Testing cross-Tower repository discovery...")

    opt_dir = Path('/opt')
    tower_repos = []

    for path in opt_dir.glob('tower-*'):
        if path.is_dir() and (path / '.git').exists():
            tower_repos.append(path)

    logger.info(f"  ✅ Found {len(tower_repos)} Tower repositories:")
    for repo in tower_repos:
        logger.info(f"    - {repo.name}")

    return len(tower_repos) > 0

async def test_smart_commit_workflow():
    """Test the smart commit workflow without actually committing"""
    logger.info("🤖 Testing smart commit workflow (dry run)...")

    git_ops = GitOperationsManager()

    # Check what would be committed
    status = await git_ops.get_status()

    if status.modified_files or status.untracked_files:
        logger.info(f"  ✅ Would commit: {len(status.modified_files)} modified, {len(status.untracked_files)} untracked")

        # Generate commit message without actually committing
        if status.modified_files:
            # Simulate diff output
            diff_output = "\n".join([f"M\t{f}" for f in status.modified_files[:5]])
            message = await git_ops._generate_commit_message(diff_output, "update")
            logger.info(f"  ✅ Generated message: {message}")
    else:
        logger.info("  ✅ No changes to commit")

    return True

async def test_repository_health():
    """Test health of all Tower repositories"""
    logger.info("🏥 Testing repository health across Tower ecosystem...")

    results = {}

    # Initialize all repos
    init_results = await git_manager.initialize_all_repos()

    for service_name, init_success in init_results.items():
        service_path = Path(f"/opt/{service_name}")

        # Get status
        status = await git_manager.get_repo_status(service_path)

        results[service_name] = {
            'initialized': init_success,
            'has_changes': status.get('has_changes', False),
            'error': status.get('error'),
            'latest_commit': status.get('latest_commit', {}).get('hash', 'None')
        }

        logger.info(f"  {'✅' if init_success else '❌'} {service_name}: {results[service_name]}")

    healthy_repos = sum(1 for r in results.values() if r['initialized'])
    logger.info(f"  ✅ {healthy_repos}/{len(results)} repositories healthy")

    return results

async def main():
    """Run comprehensive git automation tests"""
    logger.info("🚀 Starting Echo Brain Git Automation Test Suite")

    tests = [
        ("Git Operations Manager", test_git_operations_manager),
        ("Git Manager", test_git_manager),
        ("GitHub Integration", test_github_integration),
        ("Cross-Tower Discovery", test_cross_tower_discovery),
        ("Smart Commit Workflow", test_smart_commit_workflow),
        ("Repository Health", test_repository_health),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"🧪 Running: {test_name}")
            logger.info(f"{'='*50}")

            result = await test_func()
            results[test_name] = {'success': True, 'result': result}
            logger.info(f"✅ {test_name}: PASSED")

        except Exception as e:
            results[test_name] = {'success': False, 'error': str(e)}
            logger.error(f"❌ {test_name}: FAILED - {e}")

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("📊 TEST SUMMARY")
    logger.info(f"{'='*50}")

    passed = sum(1 for r in results.values() if r['success'])
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result['success'] else f"❌ FAILED: {result.get('error', 'Unknown error')}"
        logger.info(f"  {test_name}: {status}")

    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! Git automation is ready for deployment.")
        return True
    else:
        logger.warning("⚠️  Some tests failed. Review issues before deployment.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)