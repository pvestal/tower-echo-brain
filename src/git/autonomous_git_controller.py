#!/usr/bin/env python3
"""
Autonomous Git Controller for Echo Brain
Comprehensive git control system for autonomous operations across Tower repositories
"""

import asyncio
import logging
import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import git
from git import Repo, InvalidGitRepositoryError
import aiohttp
import yaml

# Import existing Echo Brain modules
from ..execution.git_operations import GitOperationsManager, GitStatus, PRInfo
from ..tasks.git_manager import GitManager
from ..core.echo.llm_interface import LLMInterface
from ..auth.authentication import AuthManager
from ..db.database import DatabaseManager

logger = logging.getLogger(__name__)

class OperationPriority(Enum):
    """Priority levels for git operations"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class AutonomousMode(Enum):
    """Autonomous operation modes"""
    FULL = "full"           # Complete automation
    SUPERVISED = "supervised"  # Requires approval for commits/pushes
    MONITORING = "monitoring"  # Read-only monitoring
    DISABLED = "disabled"   # Manual operations only

@dataclass
class GitRepository:
    """Represents a Tower repository"""
    name: str
    path: Path
    remote_url: Optional[str]
    branch: str
    last_commit: Optional[str]
    has_uncommitted: bool
    status: str
    autonomous_enabled: bool
    priority: OperationPriority
    last_sync: Optional[datetime]
    health_score: float

@dataclass
class GitOperation:
    """Represents a git operation"""
    operation_id: str
    repo_name: str
    operation_type: str
    description: str
    priority: OperationPriority
    autonomous: bool
    created_at: datetime
    scheduled_at: Optional[datetime]
    completed_at: Optional[datetime]
    success: bool
    error_message: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class GitWorkflowConfig:
    """Configuration for git workflows"""
    auto_commit_enabled: bool
    auto_push_enabled: bool
    conflict_resolution: str
    commit_message_template: str
    branch_protection: List[str]
    review_required: List[str]
    ci_cd_triggers: List[str]

class AutonomousGitController:
    """
    Comprehensive git control system for Echo Brain autonomous operations.

    Features:
    - Multi-repository coordination
    - Autonomous commit and push workflows
    - Intelligent conflict resolution
    - Security and credential management
    - CI/CD integration
    - Real-time monitoring
    """

    def __init__(
        self,
        mode: AutonomousMode = AutonomousMode.SUPERVISED,
        config_path: Optional[Path] = None
    ):
        self.mode = mode
        self.config_path = config_path or Path("/opt/tower-echo-brain/config/git_config.yaml")
        self.repositories: Dict[str, GitRepository] = {}
        self.operations_queue: List[GitOperation] = []
        self.operation_history: List[GitOperation] = []
        self.workflow_config = GitWorkflowConfig(
            auto_commit_enabled=False,
            auto_push_enabled=False,
            conflict_resolution="manual",
            commit_message_template="[Echo] {category}: {description}",
            branch_protection=["main", "master", "production"],
            review_required=["main", "master"],
            ci_cd_triggers=["push", "pull_request"]
        )

        # Initialize components
        self.git_ops_manager = GitOperationsManager()
        self.git_manager = GitManager()
        self.auth_manager = AuthManager()
        self.db_manager = DatabaseManager()

        # Initialize LLM interface for intelligent operations
        self.llm = None  # Will be initialized when needed

        # Load configuration
        asyncio.create_task(self._load_configuration())

    async def _load_configuration(self):
        """Load git configuration from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)

                # Update workflow config from file
                for key, value in config_data.get('workflow', {}).items():
                    if hasattr(self.workflow_config, key):
                        setattr(self.workflow_config, key, value)

                # Update mode if specified
                if 'autonomous_mode' in config_data:
                    self.mode = AutonomousMode(config_data['autonomous_mode'])

                logger.info(f"Loaded git configuration from {self.config_path}")
            else:
                await self._save_configuration()
                logger.info(f"Created default git configuration at {self.config_path}")

        except Exception as e:
            logger.error(f"Failed to load git configuration: {e}")

    async def _save_configuration(self):
        """Save current configuration to file"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            config_data = {
                'autonomous_mode': self.mode.value,
                'workflow': {
                    'auto_commit_enabled': self.workflow_config.auto_commit_enabled,
                    'auto_push_enabled': self.workflow_config.auto_push_enabled,
                    'conflict_resolution': self.workflow_config.conflict_resolution,
                    'commit_message_template': self.workflow_config.commit_message_template,
                    'branch_protection': self.workflow_config.branch_protection,
                    'review_required': self.workflow_config.review_required,
                    'ci_cd_triggers': self.workflow_config.ci_cd_triggers
                }
            }

            with open(self.config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)

            logger.info(f"Saved git configuration to {self.config_path}")

        except Exception as e:
            logger.error(f"Failed to save git configuration: {e}")

    async def initialize(self) -> bool:
        """Initialize the autonomous git controller"""
        try:
            logger.info("Initializing Autonomous Git Controller...")

            # Discover and register all Tower repositories
            await self._discover_repositories()

            # Initialize git repos if needed
            await self._initialize_repositories()

            # Setup monitoring
            await self._setup_monitoring()

            # Initialize LLM interface for intelligent operations
            await self._initialize_llm()

            logger.info(f"Git Controller initialized with {len(self.repositories)} repositories")
            logger.info(f"Mode: {self.mode.value}")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize Git Controller: {e}")
            return False

    async def _discover_repositories(self) -> Dict[str, GitRepository]:
        """Discover all Tower repositories"""
        opt_dir = Path('/opt')
        discovered = {}

        for path in opt_dir.glob('tower-*'):
            if path.is_dir() and not path.name.endswith('.py'):
                repo_info = await self._analyze_repository(path)
                if repo_info:
                    discovered[repo_info.name] = repo_info

        # Add ComfyUI if it exists
        comfyui_path = Path('/opt/comfyui')
        if comfyui_path.exists():
            repo_info = await self._analyze_repository(comfyui_path)
            if repo_info:
                discovered[repo_info.name] = repo_info

        self.repositories = discovered
        logger.info(f"Discovered {len(discovered)} repositories")
        return discovered

    async def _analyze_repository(self, path: Path) -> Optional[GitRepository]:
        """Analyze a repository and create GitRepository object"""
        try:
            name = path.name

            # Check if git repository
            try:
                repo = Repo(path)
                has_git = True

                # Get basic info
                try:
                    branch = repo.active_branch.name
                    last_commit = repo.head.commit.hexsha[:8]
                    remote_url = repo.remotes.origin.url if repo.remotes else None
                except:
                    branch = "main"
                    last_commit = None
                    remote_url = None

                # Check for uncommitted changes
                has_uncommitted = repo.is_dirty(untracked_files=True)
                status = "dirty" if has_uncommitted else "clean"

            except InvalidGitRepositoryError:
                has_git = False
                branch = "main"
                last_commit = None
                remote_url = None
                has_uncommitted = False
                status = "no_git"

            # Determine priority based on repository name
            priority = self._determine_priority(name)

            # Check if autonomous operations are enabled
            autonomous_enabled = (
                self.mode in [AutonomousMode.FULL, AutonomousMode.SUPERVISED] and
                has_git and
                name not in self.workflow_config.branch_protection
            )

            # Calculate health score
            health_score = await self._calculate_health_score(path)

            return GitRepository(
                name=name,
                path=path,
                remote_url=remote_url,
                branch=branch,
                last_commit=last_commit,
                has_uncommitted=has_uncommitted,
                status=status,
                autonomous_enabled=autonomous_enabled,
                priority=priority,
                last_sync=None,
                health_score=health_score
            )

        except Exception as e:
            logger.error(f"Failed to analyze repository {path}: {e}")
            return None

    def _determine_priority(self, repo_name: str) -> OperationPriority:
        """Determine operation priority for a repository"""
        critical_repos = ['tower-echo-brain', 'tower-auth', 'tower-dashboard']
        high_repos = ['tower-anime-production', 'tower-kb', 'tower-agent-manager']

        if repo_name in critical_repos:
            return OperationPriority.CRITICAL
        elif repo_name in high_repos:
            return OperationPriority.HIGH
        else:
            return OperationPriority.MEDIUM

    async def _calculate_health_score(self, path: Path) -> float:
        """Calculate repository health score (0.0 to 1.0)"""
        score = 1.0

        try:
            # Check if git repository exists
            if not (path / '.git').exists():
                return 0.0

            repo = Repo(path)

            # Deduct for uncommitted changes
            if repo.is_dirty():
                score -= 0.2

            # Deduct for untracked files
            if repo.untracked_files:
                score -= 0.1

            # Deduct for no remote
            if not repo.remotes:
                score -= 0.3

            # Deduct for outdated branches
            try:
                # Check if behind remote
                commits_behind = list(repo.iter_commits(f'{repo.head.ref}..origin/{repo.head.ref}'))
                if len(commits_behind) > 10:
                    score -= 0.2
                elif len(commits_behind) > 0:
                    score -= 0.1
            except:
                pass

        except Exception as e:
            logger.warning(f"Error calculating health score for {path}: {e}")
            score = 0.5

        return max(0.0, score)

    async def _initialize_repositories(self) -> Dict[str, bool]:
        """Initialize git repositories if needed"""
        results = {}

        for name, repo_info in self.repositories.items():
            if repo_info.status == "no_git":
                success = await self.git_manager.initialize_repo(repo_info.path)
                results[name] = success
                if success:
                    # Update repository info
                    updated_info = await self._analyze_repository(repo_info.path)
                    if updated_info:
                        self.repositories[name] = updated_info

        return results

    async def _setup_monitoring(self):
        """Setup repository monitoring"""
        # Create monitoring task
        asyncio.create_task(self._monitoring_loop())

    async def _initialize_llm(self):
        """Initialize LLM interface for intelligent operations"""
        try:
            self.llm = LLMInterface()
            await self.llm.initialize()
            logger.info("LLM interface initialized for intelligent git operations")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM interface: {e}")
            self.llm = None

    async def _monitoring_loop(self):
        """Main monitoring loop for repositories"""
        while True:
            try:
                if self.mode != AutonomousMode.DISABLED:
                    # Check all repositories for changes
                    await self._check_repositories()

                    # Process operation queue
                    await self._process_operations_queue()

                    # Cleanup old operations
                    await self._cleanup_operations()

                # Sleep for monitoring interval
                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)

    async def _check_repositories(self):
        """Check all repositories for changes and queue operations"""
        for name, repo_info in self.repositories.items():
            try:
                if not repo_info.autonomous_enabled:
                    continue

                # Get current status
                current_status = await self._get_repository_status(repo_info)

                # Check if action is needed
                if current_status.get('needs_commit', False):
                    await self._queue_commit_operation(repo_info, current_status)

                if current_status.get('needs_push', False):
                    await self._queue_push_operation(repo_info, current_status)

                if current_status.get('needs_pull', False):
                    await self._queue_pull_operation(repo_info, current_status)

            except Exception as e:
                logger.error(f"Error checking repository {name}: {e}")

    async def _get_repository_status(self, repo_info: GitRepository) -> Dict[str, Any]:
        """Get detailed status of a repository"""
        try:
            repo = Repo(repo_info.path)

            # Check for uncommitted changes
            modified_files = [item.a_path for item in repo.index.diff(None)]
            untracked_files = repo.untracked_files
            staged_files = [item.a_path for item in repo.index.diff("HEAD")]

            needs_commit = (
                self.workflow_config.auto_commit_enabled and
                (modified_files or untracked_files or staged_files)
            )

            # Check if ahead/behind remote
            ahead = behind = 0
            needs_push = needs_pull = False

            try:
                if repo.remotes:
                    # Fetch latest
                    repo.remotes.origin.fetch()

                    # Count commits ahead/behind
                    ahead_commits = list(repo.iter_commits(f'origin/{repo.active_branch}..{repo.active_branch}'))
                    behind_commits = list(repo.iter_commits(f'{repo.active_branch}..origin/{repo.active_branch}'))

                    ahead = len(ahead_commits)
                    behind = len(behind_commits)

                    needs_push = self.workflow_config.auto_push_enabled and ahead > 0
                    needs_pull = behind > 0

            except Exception as e:
                logger.debug(f"Could not check remote status for {repo_info.name}: {e}")

            return {
                'modified_files': modified_files,
                'untracked_files': untracked_files,
                'staged_files': staged_files,
                'needs_commit': needs_commit,
                'needs_push': needs_push,
                'needs_pull': needs_pull,
                'ahead': ahead,
                'behind': behind,
                'last_check': datetime.now()
            }

        except Exception as e:
            logger.error(f"Failed to get status for {repo_info.name}: {e}")
            return {}

    async def _queue_commit_operation(self, repo_info: GitRepository, status: Dict[str, Any]):
        """Queue a commit operation"""
        operation_id = f"commit_{repo_info.name}_{int(datetime.now().timestamp())}"

        # Generate intelligent commit message
        commit_message = await self._generate_commit_message(repo_info, status)

        operation = GitOperation(
            operation_id=operation_id,
            repo_name=repo_info.name,
            operation_type="commit",
            description=commit_message,
            priority=repo_info.priority,
            autonomous=self.mode == AutonomousMode.FULL,
            created_at=datetime.now(),
            scheduled_at=None,
            completed_at=None,
            success=False,
            error_message=None,
            metadata={
                'modified_files': status.get('modified_files', []),
                'untracked_files': status.get('untracked_files', []),
                'commit_message': commit_message
            }
        )

        self.operations_queue.append(operation)
        logger.info(f"Queued commit operation for {repo_info.name}")

    async def _queue_push_operation(self, repo_info: GitRepository, status: Dict[str, Any]):
        """Queue a push operation"""
        operation_id = f"push_{repo_info.name}_{int(datetime.now().timestamp())}"

        operation = GitOperation(
            operation_id=operation_id,
            repo_name=repo_info.name,
            operation_type="push",
            description=f"Push {status.get('ahead', 0)} commits to remote",
            priority=repo_info.priority,
            autonomous=self.mode == AutonomousMode.FULL,
            created_at=datetime.now(),
            scheduled_at=None,
            completed_at=None,
            success=False,
            error_message=None,
            metadata={
                'commits_ahead': status.get('ahead', 0),
                'branch': repo_info.branch
            }
        )

        self.operations_queue.append(operation)
        logger.info(f"Queued push operation for {repo_info.name}")

    async def _queue_pull_operation(self, repo_info: GitRepository, status: Dict[str, Any]):
        """Queue a pull operation"""
        operation_id = f"pull_{repo_info.name}_{int(datetime.now().timestamp())}"

        operation = GitOperation(
            operation_id=operation_id,
            repo_name=repo_info.name,
            operation_type="pull",
            description=f"Pull {status.get('behind', 0)} commits from remote",
            priority=OperationPriority.HIGH,  # Pull operations are high priority
            autonomous=self.mode == AutonomousMode.FULL,
            created_at=datetime.now(),
            scheduled_at=None,
            completed_at=None,
            success=False,
            error_message=None,
            metadata={
                'commits_behind': status.get('behind', 0),
                'branch': repo_info.branch
            }
        )

        self.operations_queue.append(operation)
        logger.info(f"Queued pull operation for {repo_info.name}")

    async def _generate_commit_message(
        self,
        repo_info: GitRepository,
        status: Dict[str, Any]
    ) -> str:
        """Generate intelligent commit message"""
        try:
            if self.llm:
                # Use LLM to generate intelligent commit message
                modified_files = status.get('modified_files', [])
                untracked_files = status.get('untracked_files', [])

                file_list = modified_files + untracked_files

                prompt = f"""
Generate a concise git commit message for repository '{repo_info.name}' with these changes:
Modified files: {', '.join(modified_files[:5])}
New files: {', '.join(untracked_files[:5])}

Follow conventional commit format: type(scope): description
Types: feat, fix, docs, style, refactor, test, chore
Keep under 72 characters.
"""

                response = await self.llm.query(prompt, max_tokens=50)
                if response and len(response.strip()) > 0:
                    return response.strip()

            # Fallback to template-based generation
            modified_count = len(status.get('modified_files', []))
            new_count = len(status.get('untracked_files', []))

            if new_count > 0 and modified_count > 0:
                category = "feat"
                description = f"Add {new_count} files, update {modified_count} files"
            elif new_count > 0:
                category = "feat"
                description = f"Add {new_count} new files"
            elif modified_count > 0:
                category = "update"
                description = f"Update {modified_count} files"
            else:
                category = "chore"
                description = "Routine maintenance"

            return self.workflow_config.commit_message_template.format(
                category=category,
                description=description
            )

        except Exception as e:
            logger.error(f"Failed to generate commit message: {e}")
            return f"[Echo] Automated commit - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    async def _process_operations_queue(self):
        """Process queued operations"""
        if not self.operations_queue:
            return

        # Sort by priority and creation time
        self.operations_queue.sort(
            key=lambda op: (op.priority.value, op.created_at)
        )

        # Process operations
        while self.operations_queue:
            operation = self.operations_queue.pop(0)

            try:
                # Check if operation should be executed
                if not operation.autonomous and self.mode != AutonomousMode.FULL:
                    logger.info(f"Operation {operation.operation_id} requires approval")
                    # Move to pending approval (could implement approval system here)
                    continue

                # Execute operation
                success = await self._execute_operation(operation)
                operation.success = success
                operation.completed_at = datetime.now()

                # Add to history
                self.operation_history.append(operation)

                if success:
                    logger.info(f"Successfully executed {operation.operation_type} for {operation.repo_name}")
                else:
                    logger.error(f"Failed to execute {operation.operation_type} for {operation.repo_name}")

            except Exception as e:
                operation.error_message = str(e)
                operation.success = False
                operation.completed_at = datetime.now()
                self.operation_history.append(operation)
                logger.error(f"Error executing operation {operation.operation_id}: {e}")

    async def _execute_operation(self, operation: GitOperation) -> bool:
        """Execute a git operation"""
        try:
            repo_info = self.repositories.get(operation.repo_name)
            if not repo_info:
                raise ValueError(f"Repository {operation.repo_name} not found")

            repo = Repo(repo_info.path)

            if operation.operation_type == "commit":
                return await self._execute_commit(repo, operation)
            elif operation.operation_type == "push":
                return await self._execute_push(repo, operation)
            elif operation.operation_type == "pull":
                return await self._execute_pull(repo, operation)
            else:
                raise ValueError(f"Unknown operation type: {operation.operation_type}")

        except Exception as e:
            operation.error_message = str(e)
            logger.error(f"Failed to execute operation {operation.operation_id}: {e}")
            return False

    async def _execute_commit(self, repo: Repo, operation: GitOperation) -> bool:
        """Execute commit operation"""
        try:
            # Add all files
            repo.git.add(A=True)

            # Check if there are actually changes to commit
            if not repo.index.diff("HEAD") and not repo.untracked_files:
                logger.info(f"No changes to commit for {operation.repo_name}")
                return True

            # Commit with generated message
            commit_message = operation.metadata.get('commit_message', operation.description)
            repo.index.commit(commit_message)

            logger.info(f"Committed changes to {operation.repo_name}: {commit_message}")
            return True

        except Exception as e:
            logger.error(f"Failed to commit to {operation.repo_name}: {e}")
            return False

    async def _execute_push(self, repo: Repo, operation: GitOperation) -> bool:
        """Execute push operation"""
        try:
            if not repo.remotes:
                logger.warning(f"No remotes configured for {operation.repo_name}")
                return False

            # Push to origin
            repo.remotes.origin.push()

            logger.info(f"Pushed changes from {operation.repo_name} to remote")
            return True

        except Exception as e:
            logger.error(f"Failed to push {operation.repo_name}: {e}")
            return False

    async def _execute_pull(self, repo: Repo, operation: GitOperation) -> bool:
        """Execute pull operation"""
        try:
            if not repo.remotes:
                logger.warning(f"No remotes configured for {operation.repo_name}")
                return False

            # Check for conflicts before pulling
            if repo.is_dirty():
                logger.warning(f"Repository {operation.repo_name} has uncommitted changes")
                # Could implement stashing here
                return False

            # Pull from origin
            repo.remotes.origin.pull()

            logger.info(f"Pulled changes to {operation.repo_name} from remote")
            return True

        except Exception as e:
            logger.error(f"Failed to pull {operation.repo_name}: {e}")
            return False

    async def _cleanup_operations(self):
        """Cleanup old operations from history"""
        cutoff_date = datetime.now() - timedelta(days=30)

        self.operation_history = [
            op for op in self.operation_history
            if op.completed_at and op.completed_at > cutoff_date
        ]

    # Public API Methods

    async def get_repository_status(self, repo_name: Optional[str] = None) -> Dict[str, Any]:
        """Get status of repositories"""
        if repo_name:
            repo_info = self.repositories.get(repo_name)
            if not repo_info:
                return {"error": f"Repository {repo_name} not found"}

            status = await self._get_repository_status(repo_info)
            return {
                "repository": asdict(repo_info),
                "status": status
            }
        else:
            # Return status for all repositories
            all_status = {}
            for name, repo_info in self.repositories.items():
                status = await self._get_repository_status(repo_info)
                all_status[name] = {
                    "repository": asdict(repo_info),
                    "status": status
                }
            return all_status

    async def manual_commit(
        self,
        repo_name: str,
        message: Optional[str] = None,
        files: Optional[List[str]] = None
    ) -> Tuple[bool, str]:
        """Manually trigger a commit operation"""
        try:
            repo_info = self.repositories.get(repo_name)
            if not repo_info:
                return False, f"Repository {repo_name} not found"

            repo = Repo(repo_info.path)

            if files:
                # Add specific files
                for file in files:
                    repo.index.add([file])
            else:
                # Add all changes
                repo.git.add(A=True)

            # Generate message if not provided
            if not message:
                status = await self._get_repository_status(repo_info)
                message = await self._generate_commit_message(repo_info, status)

            # Commit
            repo.index.commit(message)

            # Update repository info
            updated_info = await self._analyze_repository(repo_info.path)
            if updated_info:
                self.repositories[repo_name] = updated_info

            return True, f"Successfully committed to {repo_name}"

        except Exception as e:
            return False, f"Failed to commit to {repo_name}: {e}"

    async def manual_push(self, repo_name: str) -> Tuple[bool, str]:
        """Manually trigger a push operation"""
        try:
            repo_info = self.repositories.get(repo_name)
            if not repo_info:
                return False, f"Repository {repo_name} not found"

            repo = Repo(repo_info.path)

            if not repo.remotes:
                return False, f"No remotes configured for {repo_name}"

            repo.remotes.origin.push()
            return True, f"Successfully pushed {repo_name} to remote"

        except Exception as e:
            return False, f"Failed to push {repo_name}: {e}"

    async def set_autonomous_mode(self, mode: AutonomousMode) -> bool:
        """Set autonomous operation mode"""
        try:
            self.mode = mode
            await self._save_configuration()

            logger.info(f"Set autonomous mode to {mode.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to set autonomous mode: {e}")
            return False

    async def enable_auto_commit(self, enabled: bool) -> bool:
        """Enable or disable automatic commits"""
        try:
            self.workflow_config.auto_commit_enabled = enabled
            await self._save_configuration()

            logger.info(f"Auto-commit {'enabled' if enabled else 'disabled'}")
            return True

        except Exception as e:
            logger.error(f"Failed to set auto-commit: {e}")
            return False

    async def enable_auto_push(self, enabled: bool) -> bool:
        """Enable or disable automatic pushes"""
        try:
            self.workflow_config.auto_push_enabled = enabled
            await self._save_configuration()

            logger.info(f"Auto-push {'enabled' if enabled else 'disabled'}")
            return True

        except Exception as e:
            logger.error(f"Failed to set auto-push: {e}")
            return False

    async def get_operations_status(self) -> Dict[str, Any]:
        """Get status of operations queue and history"""
        return {
            "queue_size": len(self.operations_queue),
            "history_size": len(self.operation_history),
            "mode": self.mode.value,
            "auto_commit_enabled": self.workflow_config.auto_commit_enabled,
            "auto_push_enabled": self.workflow_config.auto_push_enabled,
            "recent_operations": [
                asdict(op) for op in self.operation_history[-10:]
            ]
        }

    async def create_branch(self, repo_name: str, branch_name: str, from_branch: str = "main") -> Tuple[bool, str]:
        """Create a new branch in a repository"""
        try:
            repo_info = self.repositories.get(repo_name)
            if not repo_info:
                return False, f"Repository {repo_name} not found"

            repo = Repo(repo_info.path)

            # Create new branch
            new_branch = repo.create_head(branch_name, from_branch)
            new_branch.checkout()

            # Update repository info
            updated_info = await self._analyze_repository(repo_info.path)
            if updated_info:
                self.repositories[repo_name] = updated_info

            return True, f"Created and switched to branch {branch_name} in {repo_name}"

        except Exception as e:
            return False, f"Failed to create branch in {repo_name}: {e}"

    async def create_pull_request(
        self,
        repo_name: str,
        title: str,
        body: str = "",
        base: str = "main"
    ) -> Tuple[bool, Optional[str]]:
        """Create a pull request using GitHub CLI"""
        try:
            repo_info = self.repositories.get(repo_name)
            if not repo_info:
                return False, f"Repository {repo_name} not found"

            # Use existing GitHub integration
            pr_success, pr_info = await self.git_ops_manager.create_pull_request(
                title=title,
                body=body,
                base=base
            )

            if pr_success and pr_info:
                return True, pr_info.url
            else:
                return False, "Failed to create pull request"

        except Exception as e:
            return False, f"Failed to create PR for {repo_name}: {e}"

    async def shutdown(self):
        """Shutdown the git controller"""
        try:
            logger.info("Shutting down Autonomous Git Controller...")

            # Save final configuration
            await self._save_configuration()

            # Process any remaining operations
            if self.operations_queue:
                logger.info(f"Processing {len(self.operations_queue)} remaining operations...")
                await self._process_operations_queue()

            logger.info("Git Controller shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Singleton instance
autonomous_git_controller = AutonomousGitController()

async def test_autonomous_git_controller():
    """Test the autonomous git controller"""
    controller = autonomous_git_controller

    # Initialize
    success = await controller.initialize()
    if not success:
        print("❌ Failed to initialize git controller")
        return

    print("✅ Git controller initialized")

    # Get repository status
    status = await controller.get_repository_status()
    print(f"\n📊 Repository Status:")
    for name, info in status.items():
        repo_info = info['repository']
        repo_status = info['status']
        print(f"  {name}: {repo_info['status']} (Health: {repo_info['health_score']:.2f})")

    # Get operations status
    ops_status = await controller.get_operations_status()
    print(f"\n🔄 Operations Status:")
    print(f"  Mode: {ops_status['mode']}")
    print(f"  Queue: {ops_status['queue_size']} operations")
    print(f"  History: {ops_status['history_size']} operations")
    print(f"  Auto-commit: {ops_status['auto_commit_enabled']}")
    print(f"  Auto-push: {ops_status['auto_push_enabled']}")

    print("\n✅ Autonomous Git Controller test complete")


if __name__ == "__main__":
    asyncio.run(test_autonomous_git_controller())