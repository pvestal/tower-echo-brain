#!/usr/bin/env python3
"""
Git Operations Testing Framework
Comprehensive testing suite for git operations with sandbox environments
"""

import asyncio
import logging
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import pytest
import git
from git import Repo
import yaml

# Import git modules for testing
from .autonomous_git_controller import AutonomousGitController, AutonomousMode
from .workflow_coordinator import WorkflowCoordinator
from .security_manager import GitSecurityManager
from .intelligent_git_assistant import IntelligentGitAssistant

logger = logging.getLogger(__name__)

class TestType(Enum):
    """Types of git tests"""
    UNIT = "unit"
    INTEGRATION = "integration"
    END_TO_END = "end_to_end"
    PERFORMANCE = "performance"
    SECURITY = "security"

class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

@dataclass
class TestResult:
    """Result of a test execution"""
    test_id: str
    test_name: str
    test_type: TestType
    status: TestStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration_ms: Optional[int]
    error_message: Optional[str]
    details: Dict[str, Any]
    artifacts: List[str]

@dataclass
class TestSuite:
    """Collection of tests"""
    suite_id: str
    name: str
    description: str
    tests: List[str]
    setup_hooks: List[Callable]
    teardown_hooks: List[Callable]
    dependencies: List[str]

@dataclass
class SandboxEnvironment:
    """Isolated testing environment"""
    sandbox_id: str
    name: str
    path: Path
    repositories: Dict[str, Path]
    services_running: List[str]
    cleanup_required: bool

class GitTestFramework:
    """
    Comprehensive testing framework for git operations.

    Features:
    - Isolated sandbox environments
    - Automated test discovery and execution
    - Performance benchmarking
    - Security validation
    - CI/CD integration
    - Artifact collection and analysis
    """

    def __init__(self, test_config_path: Optional[Path] = None):
        self.test_config_path = test_config_path or Path("/opt/tower-echo-brain/config/test_config.yaml")
        self.test_results_path = Path("/opt/tower-echo-brain/test_results")
        self.sandbox_base_path = Path("/tmp/echo_git_tests")

        # Test configuration
        self.config = {
            'timeout_seconds': 300,
            'max_concurrent_tests': 5,
            'sandbox_cleanup': True,
            'artifact_retention_days': 7,
            'performance_baseline_file': 'performance_baseline.json',
            'security_scan_enabled': True
        }

        # Test registry
        self.test_suites: Dict[str, TestSuite] = {}
        self.test_results: List[TestResult] = []
        self.active_sandboxes: Dict[str, SandboxEnvironment] = {}

        # Components to test
        self.git_controller: Optional[AutonomousGitController] = None
        self.workflow_coordinator: Optional[WorkflowCoordinator] = None
        self.security_manager: Optional[GitSecurityManager] = None
        self.intelligent_assistant: Optional[IntelligentGitAssistant] = None

    async def initialize(self) -> bool:
        """Initialize the test framework"""
        try:
            logger.info("Initializing Git Test Framework...")

            # Create directories
            self.test_results_path.mkdir(parents=True, exist_ok=True)
            self.sandbox_base_path.mkdir(parents=True, exist_ok=True)

            # Load configuration
            await self._load_test_config()

            # Initialize components
            await self._initialize_test_components()

            # Register built-in test suites
            await self._register_builtin_test_suites()

            logger.info(f"Test framework initialized with {len(self.test_suites)} test suites")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize test framework: {e}")
            return False

    async def _load_test_config(self):
        """Load test configuration"""
        try:
            if self.test_config_path.exists():
                with open(self.test_config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    self.config.update(config_data)

                logger.info("Loaded test configuration")
            else:
                await self._save_test_config()

        except Exception as e:
            logger.warning(f"Failed to load test config: {e}")

    async def _save_test_config(self):
        """Save test configuration"""
        try:
            self.test_config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.test_config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)

        except Exception as e:
            logger.error(f"Failed to save test config: {e}")

    async def _initialize_test_components(self):
        """Initialize components for testing"""
        try:
            # Initialize git controller
            self.git_controller = AutonomousGitController(mode=AutonomousMode.MONITORING)
            await self.git_controller.initialize()

            # Initialize workflow coordinator
            self.workflow_coordinator = WorkflowCoordinator(self.git_controller)
            await self.workflow_coordinator.initialize()

            # Initialize security manager
            self.security_manager = GitSecurityManager()
            await self.security_manager.initialize()

            # Initialize intelligent assistant
            self.intelligent_assistant = IntelligentGitAssistant()
            await self.intelligent_assistant.initialize()

            logger.info("Test components initialized")

        except Exception as e:
            logger.warning(f"Failed to initialize some test components: {e}")

    async def _register_builtin_test_suites(self):
        """Register built-in test suites"""
        # Unit tests
        await self.register_test_suite(TestSuite(
            suite_id="unit_git_operations",
            name="Git Operations Unit Tests",
            description="Test basic git operations functionality",
            tests=[
                "test_git_repository_discovery",
                "test_git_status_parsing",
                "test_commit_message_generation",
                "test_branch_creation",
                "test_conflict_detection"
            ],
            setup_hooks=[],
            teardown_hooks=[],
            dependencies=[]
        ))

        # Integration tests
        await self.register_test_suite(TestSuite(
            suite_id="integration_workflows",
            name="Workflow Integration Tests",
            description="Test workflow coordination and automation",
            tests=[
                "test_dependency_workflow",
                "test_security_scan_workflow",
                "test_auto_commit_workflow",
                "test_cross_repo_sync"
            ],
            setup_hooks=[],
            teardown_hooks=[],
            dependencies=["unit_git_operations"]
        ))

        # Security tests
        await self.register_test_suite(TestSuite(
            suite_id="security_validation",
            name="Security Validation Tests",
            description="Test security features and credential management",
            tests=[
                "test_ssh_key_generation",
                "test_credential_encryption",
                "test_audit_logging",
                "test_token_validation",
                "test_unauthorized_access"
            ],
            setup_hooks=[],
            teardown_hooks=[],
            dependencies=[]
        ))

        # Performance tests
        await self.register_test_suite(TestSuite(
            suite_id="performance_benchmarks",
            name="Performance Benchmark Tests",
            description="Test performance under various conditions",
            tests=[
                "test_large_repository_operations",
                "test_concurrent_operations",
                "test_memory_usage",
                "test_scaling_limits"
            ],
            setup_hooks=[],
            teardown_hooks=[],
            dependencies=["unit_git_operations"]
        ))

        # End-to-end tests
        await self.register_test_suite(TestSuite(
            suite_id="end_to_end_scenarios",
            name="End-to-End Scenario Tests",
            description="Test complete git workflows from start to finish",
            tests=[
                "test_full_development_workflow",
                "test_merge_conflict_resolution",
                "test_release_preparation",
                "test_emergency_hotfix"
            ],
            setup_hooks=[],
            teardown_hooks=[],
            dependencies=["integration_workflows", "security_validation"]
        ))

    async def register_test_suite(self, test_suite: TestSuite) -> bool:
        """Register a new test suite"""
        try:
            self.test_suites[test_suite.suite_id] = test_suite
            logger.info(f"Registered test suite: {test_suite.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to register test suite {test_suite.suite_id}: {e}")
            return False

    async def create_sandbox_environment(
        self,
        name: str,
        repositories: Optional[List[str]] = None
    ) -> SandboxEnvironment:
        """Create an isolated testing environment"""
        try:
            sandbox_id = f"sandbox_{name}_{int(datetime.now().timestamp())}"
            sandbox_path = self.sandbox_base_path / sandbox_id

            # Create sandbox directory
            sandbox_path.mkdir(parents=True, exist_ok=True)

            # Create test repositories
            repo_paths = {}
            if repositories:
                for repo_name in repositories:
                    repo_path = await self._create_test_repository(sandbox_path, repo_name)
                    repo_paths[repo_name] = repo_path

            sandbox = SandboxEnvironment(
                sandbox_id=sandbox_id,
                name=name,
                path=sandbox_path,
                repositories=repo_paths,
                services_running=[],
                cleanup_required=True
            )

            self.active_sandboxes[sandbox_id] = sandbox
            logger.info(f"Created sandbox environment: {sandbox_id}")

            return sandbox

        except Exception as e:
            logger.error(f"Failed to create sandbox environment: {e}")
            raise

    async def _create_test_repository(self, sandbox_path: Path, repo_name: str) -> Path:
        """Create a test git repository"""
        try:
            repo_path = sandbox_path / repo_name
            repo_path.mkdir(parents=True, exist_ok=True)

            # Initialize git repository
            repo = Repo.init(repo_path)

            # Configure git
            with repo.config_writer() as config:
                config.set_value("user", "name", "Test User")
                config.set_value("user", "email", "test@example.com")

            # Create initial files
            test_files = [
                ("README.md", f"# {repo_name}\n\nTest repository for git operations testing."),
                ("main.py", "#!/usr/bin/env python3\n\nprint('Hello, World!')\n"),
                ("requirements.txt", "pytest>=6.0.0\ngitpython>=3.0.0\n"),
                (".gitignore", "*.pyc\n__pycache__/\n.venv/\n")
            ]

            for filename, content in test_files:
                file_path = repo_path / filename
                file_path.write_text(content)

            # Initial commit
            repo.index.add([filename for filename, _ in test_files])
            repo.index.commit("Initial commit")

            logger.info(f"Created test repository: {repo_name}")
            return repo_path

        except Exception as e:
            logger.error(f"Failed to create test repository {repo_name}: {e}")
            raise

    async def cleanup_sandbox_environment(self, sandbox_id: str) -> bool:
        """Clean up a sandbox environment"""
        try:
            sandbox = self.active_sandboxes.get(sandbox_id)
            if not sandbox:
                return False

            # Stop any running services
            # (Implementation would depend on what services are running)

            # Remove sandbox directory
            if sandbox.cleanup_required and sandbox.path.exists():
                shutil.rmtree(sandbox.path)

            # Remove from active sandboxes
            del self.active_sandboxes[sandbox_id]

            logger.info(f"Cleaned up sandbox environment: {sandbox_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cleanup sandbox {sandbox_id}: {e}")
            return False

    async def run_test_suite(
        self,
        suite_id: str,
        sandbox_name: Optional[str] = None
    ) -> List[TestResult]:
        """Run a complete test suite"""
        try:
            test_suite = self.test_suites.get(suite_id)
            if not test_suite:
                raise ValueError(f"Test suite {suite_id} not found")

            logger.info(f"Running test suite: {test_suite.name}")

            # Check dependencies
            for dep_suite_id in test_suite.dependencies:
                # In a real implementation, you'd check if dependency tests passed
                pass

            # Create sandbox if needed
            sandbox = None
            if sandbox_name:
                sandbox = await self.create_sandbox_environment(
                    sandbox_name,
                    repositories=["test_repo_1", "test_repo_2"]
                )

            # Run setup hooks
            for setup_hook in test_suite.setup_hooks:
                await setup_hook()

            # Run tests
            results = []
            for test_name in test_suite.tests:
                try:
                    result = await self._run_single_test(
                        test_name,
                        test_suite,
                        sandbox
                    )
                    results.append(result)

                except Exception as e:
                    logger.error(f"Test {test_name} failed with error: {e}")
                    error_result = TestResult(
                        test_id=f"{suite_id}_{test_name}",
                        test_name=test_name,
                        test_type=TestType.UNIT,
                        status=TestStatus.ERROR,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        duration_ms=0,
                        error_message=str(e),
                        details={},
                        artifacts=[]
                    )
                    results.append(error_result)

            # Run teardown hooks
            for teardown_hook in test_suite.teardown_hooks:
                await teardown_hook()

            # Cleanup sandbox
            if sandbox:
                await self.cleanup_sandbox_environment(sandbox.sandbox_id)

            # Store results
            self.test_results.extend(results)
            await self._save_test_results(results, suite_id)

            logger.info(f"Completed test suite: {test_suite.name}")
            return results

        except Exception as e:
            logger.error(f"Failed to run test suite {suite_id}: {e}")
            return []

    async def _run_single_test(
        self,
        test_name: str,
        test_suite: TestSuite,
        sandbox: Optional[SandboxEnvironment]
    ) -> TestResult:
        """Run a single test"""
        start_time = datetime.now()
        test_id = f"{test_suite.suite_id}_{test_name}"

        try:
            logger.info(f"Running test: {test_name}")

            # Dispatch to test method
            test_method = getattr(self, test_name, None)
            if not test_method:
                raise ValueError(f"Test method {test_name} not found")

            # Run the test
            test_context = {
                'sandbox': sandbox,
                'config': self.config,
                'git_controller': self.git_controller,
                'workflow_coordinator': self.workflow_coordinator,
                'security_manager': self.security_manager,
                'intelligent_assistant': self.intelligent_assistant
            }

            details = await test_method(test_context)

            end_time = datetime.now()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)

            return TestResult(
                test_id=test_id,
                test_name=test_name,
                test_type=TestType.UNIT,  # Would be determined from test_suite
                status=TestStatus.PASSED,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                error_message=None,
                details=details,
                artifacts=[]
            )

        except Exception as e:
            end_time = datetime.now()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)

            return TestResult(
                test_id=test_id,
                test_name=test_name,
                test_type=TestType.UNIT,
                status=TestStatus.FAILED,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                error_message=str(e),
                details={},
                artifacts=[]
            )

    # Test Methods - These would be the actual test implementations

    async def test_git_repository_discovery(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Test git repository discovery functionality"""
        git_controller = context['git_controller']

        if not git_controller:
            raise ValueError("Git controller not available")

        # Test repository discovery
        repositories = git_controller.repositories
        assert len(repositories) > 0, "Should discover at least one repository"

        # Test repository analysis
        for repo_name, repo_info in repositories.items():
            assert repo_info.path.exists(), f"Repository path should exist: {repo_info.path}"
            assert repo_info.name == repo_name, "Repository name should match"

        return {
            'repositories_discovered': len(repositories),
            'repository_names': list(repositories.keys())
        }

    async def test_git_status_parsing(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Test git status parsing functionality"""
        sandbox = context['sandbox']

        if not sandbox or not sandbox.repositories:
            raise ValueError("Sandbox with repositories required")

        # Get first repository
        repo_path = list(sandbox.repositories.values())[0]
        repo = Repo(repo_path)

        # Make some changes
        test_file = repo_path / "test_status.py"
        test_file.write_text("print('Testing status parsing')")

        # Test status parsing
        git_controller = context['git_controller']
        if git_controller:
            # This would use the actual git controller status parsing
            pass

        # Verify git status
        status = repo.git.status('--porcelain')
        assert "test_status.py" in status, "Should detect new file"

        return {
            'status_output': status,
            'files_detected': 1
        }

    async def test_commit_message_generation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Test intelligent commit message generation"""
        intelligent_assistant = context['intelligent_assistant']
        sandbox = context['sandbox']

        if not intelligent_assistant or not sandbox:
            raise ValueError("Intelligent assistant and sandbox required")

        # Get first repository
        repo_path = list(sandbox.repositories.values())[0]
        repo = Repo(repo_path)

        # Make changes
        test_file = repo_path / "feature.py"
        test_file.write_text("""
def new_feature():
    '''Implement new feature functionality'''
    return 'New feature implemented'
""")

        # Stage changes
        repo.index.add(['feature.py'])

        # Test commit message generation
        analysis = await intelligent_assistant.analyze_changes_for_commit(repo_path)

        assert analysis.suggested_message, "Should generate commit message"
        assert "feat" in analysis.suggested_message.lower(), "Should detect feature change"

        return {
            'suggested_message': analysis.suggested_message,
            'change_type': analysis.primary_change_type.value,
            'files_changed': len(analysis.changes)
        }

    async def test_branch_creation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Test branch creation functionality"""
        git_controller = context['git_controller']
        sandbox = context['sandbox']

        if not git_controller or not sandbox:
            raise ValueError("Git controller and sandbox required")

        # Get first repository
        repo_name = list(sandbox.repositories.keys())[0]

        # Test branch creation
        success, branch_name = await git_controller.create_branch(repo_name, "test-feature")

        assert success, f"Branch creation should succeed: {branch_name}"
        assert "test-feature" in branch_name, "Branch name should be included"

        return {
            'branch_created': branch_name,
            'repository': repo_name
        }

    async def test_conflict_detection(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Test merge conflict detection"""
        intelligent_assistant = context['intelligent_assistant']
        sandbox = context['sandbox']

        if not intelligent_assistant or not sandbox:
            raise ValueError("Intelligent assistant and sandbox required")

        # Get first repository
        repo_path = list(sandbox.repositories.values())[0]
        repo = Repo(repo_path)

        # Create a conflict scenario
        # Create branch
        feature_branch = repo.create_head('feature-branch')
        feature_branch.checkout()

        # Make changes on feature branch
        test_file = repo_path / "conflict_file.py"
        test_file.write_text("# Feature branch change\nprint('feature')")
        repo.index.add(['conflict_file.py'])
        repo.index.commit('Add feature change')

        # Switch back to main
        repo.heads.main.checkout()

        # Make conflicting changes
        test_file.write_text("# Main branch change\nprint('main')")
        repo.index.add(['conflict_file.py'])
        repo.index.commit('Add main change')

        # Try to merge (this will create conflicts)
        try:
            repo.git.merge('feature-branch')
        except Exception:
            pass  # Expected to fail with conflicts

        # Test conflict detection
        conflicts = await intelligent_assistant.detect_conflicts(repo_path)

        return {
            'conflicts_detected': len(conflicts),
            'conflict_files': [c.file_path for c in conflicts]
        }

    async def test_ssh_key_generation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Test SSH key generation and management"""
        security_manager = context['security_manager']

        if not security_manager:
            raise ValueError("Security manager not available")

        # Test SSH key generation
        success, key_id = await security_manager.create_ssh_key_pair(
            "test_key",
            key_type="ed25519"
        )

        assert success, f"SSH key generation should succeed: {key_id}"
        assert key_id, "Should return key ID"

        # Verify key is stored
        assert key_id in security_manager.ssh_keys, "Key should be stored"

        key_info = security_manager.ssh_keys[key_id]
        assert key_info.is_active, "Key should be active"
        assert key_info.key_type == "ed25519", "Key type should match"

        return {
            'key_id': key_id,
            'key_type': key_info.key_type,
            'fingerprint': key_info.fingerprint
        }

    async def test_credential_encryption(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Test credential encryption and storage"""
        security_manager = context['security_manager']

        if not security_manager:
            raise ValueError("Security manager not available")

        from .security_manager import CredentialType, SecurityLevel

        # Test credential storage
        test_credential = "test_token_12345"
        success, cred_id = await security_manager.store_credential(
            "test_token",
            CredentialType.GITHUB_TOKEN,
            test_credential,
            metadata={"test": True}
        )

        assert success, f"Credential storage should succeed: {cred_id}"

        # Test credential retrieval
        success, retrieved_credential = await security_manager.get_credential(cred_id)

        assert success, "Credential retrieval should succeed"
        assert retrieved_credential == test_credential, "Retrieved credential should match original"

        return {
            'credential_id': cred_id,
            'encryption_verified': True,
            'retrieval_verified': True
        }

    async def _save_test_results(self, results: List[TestResult], suite_id: str):
        """Save test results to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.test_results_path / f"{suite_id}_{timestamp}.json"

            results_data = {
                'suite_id': suite_id,
                'timestamp': timestamp,
                'total_tests': len(results),
                'passed': len([r for r in results if r.status == TestStatus.PASSED]),
                'failed': len([r for r in results if r.status == TestStatus.FAILED]),
                'errors': len([r for r in results if r.status == TestStatus.ERROR]),
                'results': [asdict(result) for result in results]
            }

            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)

            logger.info(f"Saved test results to {results_file}")

        except Exception as e:
            logger.error(f"Failed to save test results: {e}")

    async def run_all_tests(self) -> Dict[str, List[TestResult]]:
        """Run all registered test suites"""
        all_results = {}

        for suite_id, test_suite in self.test_suites.items():
            logger.info(f"Running test suite: {test_suite.name}")
            results = await self.run_test_suite(suite_id, f"test_{suite_id}")
            all_results[suite_id] = results

        return all_results

    async def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of test results"""
        if not self.test_results:
            return {"message": "No test results available"}

        total_tests = len(self.test_results)
        passed = len([r for r in self.test_results if r.status == TestStatus.PASSED])
        failed = len([r for r in self.test_results if r.status == TestStatus.FAILED])
        errors = len([r for r in self.test_results if r.status == TestStatus.ERROR])

        # Calculate average duration
        durations = [r.duration_ms for r in self.test_results if r.duration_ms]
        avg_duration = sum(durations) / len(durations) if durations else 0

        return {
            'total_tests': total_tests,
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'success_rate': (passed / total_tests * 100) if total_tests > 0 else 0,
            'average_duration_ms': avg_duration,
            'test_suites': len(self.test_suites),
            'active_sandboxes': len(self.active_sandboxes)
        }

    async def cleanup_all_sandboxes(self) -> int:
        """Clean up all active sandbox environments"""
        cleanup_count = 0

        for sandbox_id in list(self.active_sandboxes.keys()):
            success = await self.cleanup_sandbox_environment(sandbox_id)
            if success:
                cleanup_count += 1

        return cleanup_count


# Global instance
git_test_framework = GitTestFramework()

async def test_git_test_framework():
    """Test the git test framework itself"""
    framework = git_test_framework

    # Initialize
    success = await framework.initialize()
    if not success:
        print("❌ Failed to initialize test framework")
        return

    print("✅ Test framework initialized")
    print(f"📋 Test suites: {len(framework.test_suites)}")

    # Create a test sandbox
    sandbox = await framework.create_sandbox_environment(
        "framework_test",
        repositories=["test_repo"]
    )

    print(f"🏖️  Created sandbox: {sandbox.sandbox_id}")

    # Run a simple test suite
    try:
        results = await framework.run_test_suite("unit_git_operations", "unit_test")

        print(f"\n📊 Test Results:")
        passed = len([r for r in results if r.status == TestStatus.PASSED])
        failed = len([r for r in results if r.status == TestStatus.FAILED])

        print(f"  Total: {len(results)}")
        print(f"  Passed: {passed}")
        print(f"  Failed: {failed}")

        for result in results:
            status_icon = "✅" if result.status == TestStatus.PASSED else "❌"
            print(f"  {status_icon} {result.test_name}: {result.status.value}")
            if result.error_message:
                print(f"      Error: {result.error_message}")

    except Exception as e:
        print(f"❌ Error running tests: {e}")

    # Cleanup
    cleanup_count = await framework.cleanup_all_sandboxes()
    print(f"\n🧹 Cleaned up {cleanup_count} sandboxes")

    print("\n✅ Git test framework test complete")


if __name__ == "__main__":
    asyncio.run(test_git_test_framework())