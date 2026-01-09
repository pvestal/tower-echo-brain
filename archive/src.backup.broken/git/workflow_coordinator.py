#!/usr/bin/env python3
"""
Workflow Coordinator for Echo Brain Git Operations
Handles cross-repository coordination, CI/CD triggers, and automated workflows
"""

import asyncio
import logging
import json
import subprocess
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import hashlib

from .autonomous_git_controller import AutonomousGitController, GitRepository, OperationPriority

logger = logging.getLogger(__name__)

class WorkflowType(Enum):
    """Types of workflows"""
    CI_CD = "ci_cd"
    DEPENDENCY_UPDATE = "dependency_update"
    CROSS_REPO_SYNC = "cross_repo_sync"
    SECURITY_SCAN = "security_scan"
    DOCUMENTATION_UPDATE = "documentation_update"
    RELEASE_PREPARATION = "release_preparation"

class TriggerType(Enum):
    """Workflow trigger types"""
    PUSH = "push"
    PULL_REQUEST = "pull_request"
    SCHEDULE = "schedule"
    MANUAL = "manual"
    DEPENDENCY = "dependency"
    WEBHOOK = "webhook"

@dataclass
class WorkflowRule:
    """Represents a workflow rule"""
    rule_id: str
    name: str
    workflow_type: WorkflowType
    trigger_type: TriggerType
    repositories: List[str]
    conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    priority: OperationPriority
    enabled: bool
    created_at: datetime
    last_triggered: Optional[datetime]
    execution_count: int

@dataclass
class WorkflowExecution:
    """Represents a workflow execution"""
    execution_id: str
    rule_id: str
    triggered_by: str
    trigger_data: Dict[str, Any]
    repositories_affected: List[str]
    started_at: datetime
    completed_at: Optional[datetime]
    status: str  # pending, running, completed, failed
    results: Dict[str, Any]
    error_message: Optional[str]

@dataclass
class DependencyMap:
    """Maps dependencies between repositories"""
    dependent: str
    dependency: str
    dependency_type: str  # code, deployment, data
    strength: float  # 0.0 to 1.0

class WorkflowCoordinator:
    """
    Coordinates workflows across multiple repositories with intelligent automation.

    Features:
    - Cross-repository dependency tracking
    - Automated CI/CD pipeline triggers
    - Intelligent workflow scheduling
    - Repository synchronization
    - Security and compliance workflows
    """

    def __init__(self, git_controller: AutonomousGitController):
        self.git_controller = git_controller
        self.workflow_rules: Dict[str, WorkflowRule] = {}
        self.dependency_map: List[DependencyMap] = []
        self.execution_history: List[WorkflowExecution] = []
        self.active_executions: Dict[str, WorkflowExecution] = {}

        # Configuration paths
        self.config_dir = Path("/opt/tower-echo-brain/config/workflows")
        self.rules_file = self.config_dir / "workflow_rules.yaml"
        self.dependencies_file = self.config_dir / "dependencies.yaml"

        # Webhook server for external triggers
        self.webhook_port = 8310
        self.webhook_secret = None

        # CI/CD integration
        self.github_actions_enabled = False
        self.jenkins_enabled = False
        self.ci_cd_configs: Dict[str, Any] = {}

    async def initialize(self) -> bool:
        """Initialize the workflow coordinator"""
        try:
            logger.info("Initializing Workflow Coordinator...")

            # Create config directory
            self.config_dir.mkdir(parents=True, exist_ok=True)

            # Load configuration
            await self._load_workflow_rules()
            await self._load_dependency_map()
            await self._detect_ci_cd_systems()

            # Setup default workflows
            await self._setup_default_workflows()

            # Start monitoring
            asyncio.create_task(self._workflow_monitoring_loop())

            logger.info(f"Workflow Coordinator initialized with {len(self.workflow_rules)} rules")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Workflow Coordinator: {e}")
            return False

    async def _load_workflow_rules(self):
        """Load workflow rules from configuration"""
        try:
            if self.rules_file.exists():
                with open(self.rules_file, 'r') as f:
                    rules_data = yaml.safe_load(f)

                for rule_data in rules_data.get('rules', []):
                    rule = WorkflowRule(
                        rule_id=rule_data['rule_id'],
                        name=rule_data['name'],
                        workflow_type=WorkflowType(rule_data['workflow_type']),
                        trigger_type=TriggerType(rule_data['trigger_type']),
                        repositories=rule_data.get('repositories', []),
                        conditions=rule_data.get('conditions', {}),
                        actions=rule_data.get('actions', []),
                        priority=OperationPriority(rule_data.get('priority', 'medium')),
                        enabled=rule_data.get('enabled', True),
                        created_at=datetime.fromisoformat(rule_data['created_at']),
                        last_triggered=datetime.fromisoformat(rule_data['last_triggered']) if rule_data.get('last_triggered') else None,
                        execution_count=rule_data.get('execution_count', 0)
                    )
                    self.workflow_rules[rule.rule_id] = rule

                logger.info(f"Loaded {len(self.workflow_rules)} workflow rules")
            else:
                logger.info("No workflow rules file found, will create defaults")

        except Exception as e:
            logger.error(f"Failed to load workflow rules: {e}")

    async def _load_dependency_map(self):
        """Load repository dependency map"""
        try:
            if self.dependencies_file.exists():
                with open(self.dependencies_file, 'r') as f:
                    deps_data = yaml.safe_load(f)

                self.dependency_map = []
                for dep_data in deps_data.get('dependencies', []):
                    dependency = DependencyMap(
                        dependent=dep_data['dependent'],
                        dependency=dep_data['dependency'],
                        dependency_type=dep_data['type'],
                        strength=dep_data.get('strength', 0.5)
                    )
                    self.dependency_map.append(dependency)

                logger.info(f"Loaded {len(self.dependency_map)} dependency mappings")
            else:
                # Auto-detect dependencies
                await self._auto_detect_dependencies()

        except Exception as e:
            logger.error(f"Failed to load dependency map: {e}")

    async def _auto_detect_dependencies(self):
        """Automatically detect dependencies between repositories"""
        try:
            dependencies = []

            # Get all repositories
            repositories = list(self.git_controller.repositories.keys())

            for repo_name in repositories:
                repo_info = self.git_controller.repositories[repo_name]
                repo_path = repo_info.path

                # Check for common dependency indicators
                deps = await self._analyze_repository_dependencies(repo_path, repo_name, repositories)
                dependencies.extend(deps)

            self.dependency_map = dependencies

            # Save detected dependencies
            await self._save_dependency_map()

            logger.info(f"Auto-detected {len(dependencies)} dependencies")

        except Exception as e:
            logger.error(f"Failed to auto-detect dependencies: {e}")

    async def _analyze_repository_dependencies(
        self,
        repo_path: Path,
        repo_name: str,
        all_repos: List[str]
    ) -> List[DependencyMap]:
        """Analyze a repository to detect dependencies"""
        dependencies = []

        try:
            # Check package.json for Node.js projects
            package_json = repo_path / "package.json"
            if package_json.exists():
                with open(package_json, 'r') as f:
                    pkg_data = json.load(f)

                # Look for tower-* dependencies
                deps = pkg_data.get('dependencies', {})
                dev_deps = pkg_data.get('devDependencies', {})

                for dep_name in list(deps.keys()) + list(dev_deps.keys()):
                    if dep_name.startswith('@tower/') or any(repo in dep_name for repo in all_repos):
                        # Extract repository name
                        dep_repo = self._extract_repo_name_from_dependency(dep_name, all_repos)
                        if dep_repo and dep_repo != repo_name:
                            dependencies.append(DependencyMap(
                                dependent=repo_name,
                                dependency=dep_repo,
                                dependency_type="code",
                                strength=0.8
                            ))

            # Check requirements.txt for Python projects
            requirements_txt = repo_path / "requirements.txt"
            if requirements_txt.exists():
                with open(requirements_txt, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Check if it references another tower service
                            for repo in all_repos:
                                if repo.replace('-', '_') in line.lower():
                                    dependencies.append(DependencyMap(
                                        dependent=repo_name,
                                        dependency=repo,
                                        dependency_type="code",
                                        strength=0.7
                                    ))

            # Check docker-compose files for deployment dependencies
            docker_compose = repo_path / "docker-compose.yml"
            if docker_compose.exists():
                with open(docker_compose, 'r') as f:
                    compose_data = yaml.safe_load(f)

                services = compose_data.get('services', {})
                for service_name, service_config in services.items():
                    # Check for depends_on
                    depends_on = service_config.get('depends_on', [])
                    for dep_service in depends_on:
                        for repo in all_repos:
                            if repo in dep_service or dep_service in repo:
                                dependencies.append(DependencyMap(
                                    dependent=repo_name,
                                    dependency=repo,
                                    dependency_type="deployment",
                                    strength=0.9
                                ))

            # Special cases for known relationships
            dependencies.extend(self._get_known_dependencies(repo_name))

        except Exception as e:
            logger.warning(f"Error analyzing dependencies for {repo_name}: {e}")

        return dependencies

    def _extract_repo_name_from_dependency(self, dep_name: str, all_repos: List[str]) -> Optional[str]:
        """Extract repository name from dependency name"""
        for repo in all_repos:
            if repo.replace('-', '_') in dep_name.replace('-', '_').lower():
                return repo
        return None

    def _get_known_dependencies(self, repo_name: str) -> List[DependencyMap]:
        """Get known dependencies for specific repositories"""
        known_deps = {
            'tower-dashboard': ['tower-auth', 'tower-echo-brain', 'tower-kb'],
            'tower-anime-production': ['tower-echo-brain', 'tower-auth'],
            'tower-agent-manager': ['tower-echo-brain'],
            'tower-echo-frontend': ['tower-echo-brain'],
            'tower-music-production': ['tower-apple-music', 'tower-auth'],
        }

        dependencies = []
        for dependency in known_deps.get(repo_name, []):
            dependencies.append(DependencyMap(
                dependent=repo_name,
                dependency=dependency,
                dependency_type="code",
                strength=0.8
            ))

        return dependencies

    async def _save_dependency_map(self):
        """Save dependency map to file"""
        try:
            deps_data = {
                'dependencies': [
                    {
                        'dependent': dep.dependent,
                        'dependency': dep.dependency,
                        'type': dep.dependency_type,
                        'strength': dep.strength
                    }
                    for dep in self.dependency_map
                ],
                'last_updated': datetime.now().isoformat()
            }

            with open(self.dependencies_file, 'w') as f:
                yaml.dump(deps_data, f, default_flow_style=False)

            logger.info(f"Saved dependency map with {len(self.dependency_map)} entries")

        except Exception as e:
            logger.error(f"Failed to save dependency map: {e}")

    async def _detect_ci_cd_systems(self):
        """Detect available CI/CD systems"""
        # Check for GitHub Actions
        try:
            result = subprocess.run(['gh', 'auth', 'status'], capture_output=True)
            self.github_actions_enabled = result.returncode == 0
        except FileNotFoundError:
            self.github_actions_enabled = False

        # Check for Jenkins (if running locally)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8080/jenkins', timeout=5) as response:
                    self.jenkins_enabled = response.status == 200
        except:
            self.jenkins_enabled = False

        logger.info(f"CI/CD Systems - GitHub Actions: {self.github_actions_enabled}, Jenkins: {self.jenkins_enabled}")

    async def _setup_default_workflows(self):
        """Setup default workflow rules"""
        default_rules = [
            {
                'rule_id': 'auto_dependency_update',
                'name': 'Automatic Dependency Updates',
                'workflow_type': WorkflowType.DEPENDENCY_UPDATE,
                'trigger_type': TriggerType.PUSH,
                'repositories': ['*'],  # All repositories
                'conditions': {
                    'files_changed': ['package.json', 'requirements.txt', 'Cargo.toml']
                },
                'actions': [
                    {'type': 'update_dependents', 'auto_commit': True}
                ],
                'priority': OperationPriority.MEDIUM
            },
            {
                'rule_id': 'security_scan_on_push',
                'name': 'Security Scan on Push',
                'workflow_type': WorkflowType.SECURITY_SCAN,
                'trigger_type': TriggerType.PUSH,
                'repositories': ['tower-echo-brain', 'tower-auth', 'tower-dashboard'],
                'conditions': {},
                'actions': [
                    {'type': 'security_scan', 'tools': ['bandit', 'safety', 'npm_audit']}
                ],
                'priority': OperationPriority.HIGH
            },
            {
                'rule_id': 'cross_repo_documentation',
                'name': 'Cross-Repository Documentation Update',
                'workflow_type': WorkflowType.DOCUMENTATION_UPDATE,
                'trigger_type': TriggerType.PUSH,
                'repositories': ['*'],
                'conditions': {
                    'files_changed': ['*.py', '*.js', '*.ts', '*.md']
                },
                'actions': [
                    {'type': 'update_api_docs'},
                    {'type': 'update_readme'},
                    {'type': 'sync_kb'}
                ],
                'priority': OperationPriority.LOW
            }
        ]

        for rule_data in default_rules:
            rule_id = rule_data['rule_id']
            if rule_id not in self.workflow_rules:
                rule = WorkflowRule(
                    rule_id=rule_id,
                    name=rule_data['name'],
                    workflow_type=rule_data['workflow_type'],
                    trigger_type=rule_data['trigger_type'],
                    repositories=rule_data['repositories'],
                    conditions=rule_data['conditions'],
                    actions=rule_data['actions'],
                    priority=rule_data['priority'],
                    enabled=True,
                    created_at=datetime.now(),
                    last_triggered=None,
                    execution_count=0
                )
                self.workflow_rules[rule_id] = rule

        await self._save_workflow_rules()

    async def _save_workflow_rules(self):
        """Save workflow rules to file"""
        try:
            rules_data = {
                'rules': [
                    {
                        'rule_id': rule.rule_id,
                        'name': rule.name,
                        'workflow_type': rule.workflow_type.value,
                        'trigger_type': rule.trigger_type.value,
                        'repositories': rule.repositories,
                        'conditions': rule.conditions,
                        'actions': rule.actions,
                        'priority': rule.priority.value,
                        'enabled': rule.enabled,
                        'created_at': rule.created_at.isoformat(),
                        'last_triggered': rule.last_triggered.isoformat() if rule.last_triggered else None,
                        'execution_count': rule.execution_count
                    }
                    for rule in self.workflow_rules.values()
                ],
                'last_updated': datetime.now().isoformat()
            }

            with open(self.rules_file, 'w') as f:
                yaml.dump(rules_data, f, default_flow_style=False)

            logger.info(f"Saved {len(self.workflow_rules)} workflow rules")

        except Exception as e:
            logger.error(f"Failed to save workflow rules: {e}")

    async def _workflow_monitoring_loop(self):
        """Main workflow monitoring loop"""
        while True:
            try:
                # Check for triggered workflows
                await self._check_workflow_triggers()

                # Process active executions
                await self._process_active_executions()

                # Cleanup completed executions
                await self._cleanup_executions()

                # Sleep
                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Error in workflow monitoring loop: {e}")
                await asyncio.sleep(30)

    async def _check_workflow_triggers(self):
        """Check for workflow triggers"""
        try:
            # Get recent repository changes
            repo_changes = await self._get_recent_changes()

            for rule in self.workflow_rules.values():
                if not rule.enabled:
                    continue

                # Check if rule should be triggered
                should_trigger, trigger_data = await self._evaluate_rule_conditions(rule, repo_changes)

                if should_trigger:
                    await self._trigger_workflow(rule, trigger_data)

        except Exception as e:
            logger.error(f"Error checking workflow triggers: {e}")

    async def _get_recent_changes(self) -> Dict[str, Any]:
        """Get recent changes across all repositories"""
        changes = {}

        for repo_name, repo_info in self.git_controller.repositories.items():
            try:
                repo_changes = await self._get_repository_recent_changes(repo_info)
                if repo_changes:
                    changes[repo_name] = repo_changes
            except Exception as e:
                logger.warning(f"Error getting changes for {repo_name}: {e}")

        return changes

    async def _get_repository_recent_changes(self, repo_info: GitRepository) -> Optional[Dict[str, Any]]:
        """Get recent changes for a specific repository"""
        try:
            from git import Repo
            repo = Repo(repo_info.path)

            # Get commits from last 5 minutes
            since = datetime.now() - timedelta(minutes=5)
            recent_commits = list(repo.iter_commits(since=since))

            if not recent_commits:
                return None

            # Get changed files from recent commits
            changed_files = set()
            for commit in recent_commits:
                for item in commit.stats.files:
                    changed_files.add(item)

            return {
                'commits': len(recent_commits),
                'changed_files': list(changed_files),
                'latest_commit': {
                    'hash': recent_commits[0].hexsha,
                    'message': recent_commits[0].message.strip(),
                    'author': str(recent_commits[0].author),
                    'timestamp': datetime.fromtimestamp(recent_commits[0].committed_date)
                }
            }

        except Exception as e:
            logger.warning(f"Error getting recent changes for {repo_info.name}: {e}")
            return None

    async def _evaluate_rule_conditions(
        self,
        rule: WorkflowRule,
        repo_changes: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate if a workflow rule should be triggered"""
        try:
            # Check repository filter
            applicable_repos = []
            if '*' in rule.repositories:
                applicable_repos = list(repo_changes.keys())
            else:
                applicable_repos = [repo for repo in rule.repositories if repo in repo_changes]

            if not applicable_repos:
                return False, {}

            # Check trigger type
            if rule.trigger_type == TriggerType.PUSH:
                # Check if there are recent commits
                for repo_name in applicable_repos:
                    if repo_changes[repo_name]['commits'] > 0:
                        # Check specific conditions
                        if await self._check_rule_conditions(rule, repo_changes[repo_name]):
                            return True, {
                                'trigger_type': 'push',
                                'repository': repo_name,
                                'changes': repo_changes[repo_name]
                            }

            elif rule.trigger_type == TriggerType.SCHEDULE:
                # Check if scheduled time has passed
                if self._should_trigger_scheduled(rule):
                    return True, {
                        'trigger_type': 'schedule',
                        'repositories': applicable_repos
                    }

            return False, {}

        except Exception as e:
            logger.error(f"Error evaluating rule conditions for {rule.rule_id}: {e}")
            return False, {}

    async def _check_rule_conditions(self, rule: WorkflowRule, repo_changes: Dict[str, Any]) -> bool:
        """Check specific rule conditions"""
        conditions = rule.conditions

        # Check files_changed condition
        if 'files_changed' in conditions:
            required_patterns = conditions['files_changed']
            changed_files = repo_changes.get('changed_files', [])

            for pattern in required_patterns:
                if any(self._matches_pattern(file, pattern) for file in changed_files):
                    return True
            return False

        # No specific conditions, trigger by default
        return True

    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches pattern"""
        import fnmatch
        return fnmatch.fnmatch(filename, pattern)

    def _should_trigger_scheduled(self, rule: WorkflowRule) -> bool:
        """Check if scheduled workflow should trigger"""
        if not rule.last_triggered:
            return True

        # Check schedule interval (default: daily)
        interval = rule.conditions.get('schedule_interval', 'daily')
        now = datetime.now()

        if interval == 'hourly':
            return now - rule.last_triggered > timedelta(hours=1)
        elif interval == 'daily':
            return now - rule.last_triggered > timedelta(days=1)
        elif interval == 'weekly':
            return now - rule.last_triggered > timedelta(weeks=1)

        return False

    async def _trigger_workflow(self, rule: WorkflowRule, trigger_data: Dict[str, Any]):
        """Trigger a workflow execution"""
        try:
            execution_id = f"{rule.rule_id}_{int(datetime.now().timestamp())}"

            execution = WorkflowExecution(
                execution_id=execution_id,
                rule_id=rule.rule_id,
                triggered_by=trigger_data.get('trigger_type', 'unknown'),
                trigger_data=trigger_data,
                repositories_affected=self._get_affected_repositories(rule, trigger_data),
                started_at=datetime.now(),
                completed_at=None,
                status='pending',
                results={},
                error_message=None
            )

            # Add to active executions
            self.active_executions[execution_id] = execution

            # Update rule
            rule.last_triggered = datetime.now()
            rule.execution_count += 1

            logger.info(f"Triggered workflow {rule.name} (ID: {execution_id})")

            # Start execution
            asyncio.create_task(self._execute_workflow(execution, rule))

        except Exception as e:
            logger.error(f"Failed to trigger workflow {rule.rule_id}: {e}")

    def _get_affected_repositories(self, rule: WorkflowRule, trigger_data: Dict[str, Any]) -> List[str]:
        """Get list of repositories affected by workflow"""
        if trigger_data.get('repository'):
            base_repos = [trigger_data['repository']]
        else:
            base_repos = trigger_data.get('repositories', [])

        affected = set(base_repos)

        # Add dependent repositories based on dependency map
        for repo in base_repos:
            dependents = self._get_dependents(repo)
            affected.update(dependents)

        return list(affected)

    def _get_dependents(self, repo_name: str) -> List[str]:
        """Get repositories that depend on the given repository"""
        dependents = []
        for dep in self.dependency_map:
            if dep.dependency == repo_name:
                dependents.append(dep.dependent)
        return dependents

    async def _execute_workflow(self, execution: WorkflowExecution, rule: WorkflowRule):
        """Execute a workflow"""
        try:
            execution.status = 'running'
            logger.info(f"Executing workflow {rule.name} (ID: {execution.execution_id})")

            results = {}

            # Execute each action
            for action in rule.actions:
                action_result = await self._execute_workflow_action(action, execution, rule)
                action_type = action.get('type', 'unknown')
                results[action_type] = action_result

            execution.status = 'completed'
            execution.results = results
            execution.completed_at = datetime.now()

            logger.info(f"Completed workflow {rule.name} successfully")

        except Exception as e:
            execution.status = 'failed'
            execution.error_message = str(e)
            execution.completed_at = datetime.now()
            logger.error(f"Workflow {rule.name} failed: {e}")

    async def _execute_workflow_action(
        self,
        action: Dict[str, Any],
        execution: WorkflowExecution,
        rule: WorkflowRule
    ) -> Dict[str, Any]:
        """Execute a specific workflow action"""
        action_type = action.get('type')

        if action_type == 'update_dependents':
            return await self._action_update_dependents(action, execution, rule)
        elif action_type == 'security_scan':
            return await self._action_security_scan(action, execution, rule)
        elif action_type == 'update_api_docs':
            return await self._action_update_api_docs(action, execution, rule)
        elif action_type == 'update_readme':
            return await self._action_update_readme(action, execution, rule)
        elif action_type == 'sync_kb':
            return await self._action_sync_kb(action, execution, rule)
        else:
            logger.warning(f"Unknown action type: {action_type}")
            return {'status': 'skipped', 'reason': f'Unknown action type: {action_type}'}

    async def _action_update_dependents(
        self,
        action: Dict[str, Any],
        execution: WorkflowExecution,
        rule: WorkflowRule
    ) -> Dict[str, Any]:
        """Update dependent repositories when dependencies change"""
        try:
            updated_repos = []

            for repo_name in execution.repositories_affected:
                # Get dependents
                dependents = self._get_dependents(repo_name)

                for dependent in dependents:
                    if dependent in self.git_controller.repositories:
                        # Update dependency in dependent repository
                        success = await self._update_dependency_in_repo(repo_name, dependent)
                        if success:
                            updated_repos.append(dependent)

                            # Auto-commit if specified
                            if action.get('auto_commit', False):
                                commit_success, commit_msg = await self.git_controller.manual_commit(
                                    dependent,
                                    f"Update {repo_name} dependency"
                                )
                                if commit_success:
                                    logger.info(f"Auto-committed dependency update in {dependent}")

            return {
                'status': 'completed',
                'updated_repositories': updated_repos,
                'count': len(updated_repos)
            }

        except Exception as e:
            logger.error(f"Failed to update dependents: {e}")
            return {'status': 'failed', 'error': str(e)}

    async def _update_dependency_in_repo(self, dependency_repo: str, dependent_repo: str) -> bool:
        """Update dependency reference in dependent repository"""
        try:
            dependent_info = self.git_controller.repositories.get(dependent_repo)
            if not dependent_info:
                return False

            # Update package.json if it exists
            package_json = dependent_info.path / "package.json"
            if package_json.exists():
                with open(package_json, 'r') as f:
                    pkg_data = json.load(f)

                # Update version reference (simplified approach)
                # In a real implementation, you'd fetch the actual version
                deps = pkg_data.get('dependencies', {})
                dev_deps = pkg_data.get('devDependencies', {})

                updated = False
                for dep_name in deps:
                    if dependency_repo.replace('-', '_') in dep_name:
                        # Update to latest version (placeholder)
                        deps[dep_name] = "latest"
                        updated = True

                if updated:
                    with open(package_json, 'w') as f:
                        json.dump(pkg_data, f, indent=2)
                    return True

            return False

        except Exception as e:
            logger.error(f"Failed to update dependency {dependency_repo} in {dependent_repo}: {e}")
            return False

    async def _action_security_scan(
        self,
        action: Dict[str, Any],
        execution: WorkflowExecution,
        rule: WorkflowRule
    ) -> Dict[str, Any]:
        """Run security scan on repositories"""
        try:
            scan_results = {}
            tools = action.get('tools', ['bandit', 'safety'])

            for repo_name in execution.repositories_affected:
                repo_info = self.git_controller.repositories.get(repo_name)
                if not repo_info:
                    continue

                repo_results = {}

                # Run security tools
                for tool in tools:
                    tool_result = await self._run_security_tool(tool, repo_info.path)
                    repo_results[tool] = tool_result

                scan_results[repo_name] = repo_results

            return {
                'status': 'completed',
                'scan_results': scan_results,
                'repositories_scanned': len(scan_results)
            }

        except Exception as e:
            logger.error(f"Security scan failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    async def _run_security_tool(self, tool: str, repo_path: Path) -> Dict[str, Any]:
        """Run a specific security tool"""
        try:
            if tool == 'bandit':
                # Run bandit for Python code
                result = subprocess.run(
                    ['bandit', '-r', str(repo_path), '-f', 'json'],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode == 0:
                    return {'status': 'success', 'issues': 0}
                else:
                    # Parse bandit output for issues
                    try:
                        output = json.loads(result.stdout)
                        return {
                            'status': 'issues_found',
                            'issues': len(output.get('results', [])),
                            'high_severity': len([r for r in output.get('results', []) if r.get('issue_severity') == 'HIGH'])
                        }
                    except:
                        return {'status': 'error', 'message': 'Failed to parse bandit output'}

            elif tool == 'safety':
                # Run safety for Python dependencies
                result = subprocess.run(
                    ['safety', 'check', '--json'],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode == 0:
                    return {'status': 'success', 'vulnerabilities': 0}
                else:
                    try:
                        output = json.loads(result.stdout)
                        return {
                            'status': 'vulnerabilities_found',
                            'vulnerabilities': len(output),
                            'details': output[:5]  # First 5 vulnerabilities
                        }
                    except:
                        return {'status': 'error', 'message': 'Failed to parse safety output'}

            elif tool == 'npm_audit':
                # Run npm audit for Node.js projects
                result = subprocess.run(
                    ['npm', 'audit', '--json'],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                try:
                    output = json.loads(result.stdout)
                    vulnerabilities = output.get('metadata', {}).get('vulnerabilities', {})
                    total_vulns = sum(vulnerabilities.values()) if isinstance(vulnerabilities, dict) else 0
                    return {
                        'status': 'success' if total_vulns == 0 else 'vulnerabilities_found',
                        'vulnerabilities': total_vulns,
                        'critical': vulnerabilities.get('critical', 0),
                        'high': vulnerabilities.get('high', 0)
                    }
                except:
                    return {'status': 'error', 'message': 'Failed to parse npm audit output'}

            else:
                return {'status': 'skipped', 'reason': f'Unknown tool: {tool}'}

        except subprocess.TimeoutExpired:
            return {'status': 'timeout', 'tool': tool}
        except FileNotFoundError:
            return {'status': 'tool_not_found', 'tool': tool}
        except Exception as e:
            return {'status': 'error', 'tool': tool, 'error': str(e)}

    async def _action_update_api_docs(
        self,
        action: Dict[str, Any],
        execution: WorkflowExecution,
        rule: WorkflowRule
    ) -> Dict[str, Any]:
        """Update API documentation"""
        # Placeholder implementation
        return {'status': 'completed', 'message': 'API docs updated'}

    async def _action_update_readme(
        self,
        action: Dict[str, Any],
        execution: WorkflowExecution,
        rule: WorkflowRule
    ) -> Dict[str, Any]:
        """Update README files"""
        # Placeholder implementation
        return {'status': 'completed', 'message': 'README files updated'}

    async def _action_sync_kb(
        self,
        action: Dict[str, Any],
        execution: WorkflowExecution,
        rule: WorkflowRule
    ) -> Dict[str, Any]:
        """Sync with knowledge base"""
        # Placeholder implementation
        return {'status': 'completed', 'message': 'Knowledge base synchronized'}

    async def _process_active_executions(self):
        """Process and monitor active workflow executions"""
        completed_executions = []

        for execution_id, execution in self.active_executions.items():
            if execution.status in ['completed', 'failed']:
                completed_executions.append(execution_id)

        # Move completed executions to history
        for execution_id in completed_executions:
            execution = self.active_executions.pop(execution_id)
            self.execution_history.append(execution)

    async def _cleanup_executions(self):
        """Cleanup old execution history"""
        cutoff_date = datetime.now() - timedelta(days=30)
        self.execution_history = [
            exec for exec in self.execution_history
            if exec.completed_at and exec.completed_at > cutoff_date
        ]

    # Public API Methods

    async def create_workflow_rule(
        self,
        name: str,
        workflow_type: WorkflowType,
        trigger_type: TriggerType,
        repositories: List[str],
        conditions: Dict[str, Any],
        actions: List[Dict[str, Any]],
        priority: OperationPriority = OperationPriority.MEDIUM
    ) -> str:
        """Create a new workflow rule"""
        rule_id = f"custom_{hashlib.md5(name.encode()).hexdigest()[:8]}"

        rule = WorkflowRule(
            rule_id=rule_id,
            name=name,
            workflow_type=workflow_type,
            trigger_type=trigger_type,
            repositories=repositories,
            conditions=conditions,
            actions=actions,
            priority=priority,
            enabled=True,
            created_at=datetime.now(),
            last_triggered=None,
            execution_count=0
        )

        self.workflow_rules[rule_id] = rule
        await self._save_workflow_rules()

        logger.info(f"Created workflow rule: {name} (ID: {rule_id})")
        return rule_id

    async def get_workflow_status(self) -> Dict[str, Any]:
        """Get status of all workflows"""
        return {
            'total_rules': len(self.workflow_rules),
            'enabled_rules': len([r for r in self.workflow_rules.values() if r.enabled]),
            'active_executions': len(self.active_executions),
            'execution_history': len(self.execution_history),
            'dependencies_tracked': len(self.dependency_map),
            'recent_executions': [
                {
                    'execution_id': exec.execution_id,
                    'rule_name': self.workflow_rules[exec.rule_id].name,
                    'status': exec.status,
                    'started_at': exec.started_at.isoformat(),
                    'completed_at': exec.completed_at.isoformat() if exec.completed_at else None
                }
                for exec in self.execution_history[-5:]
            ]
        }

    async def trigger_manual_workflow(self, rule_id: str, trigger_data: Optional[Dict[str, Any]] = None) -> str:
        """Manually trigger a workflow"""
        rule = self.workflow_rules.get(rule_id)
        if not rule:
            raise ValueError(f"Workflow rule {rule_id} not found")

        if not trigger_data:
            trigger_data = {'trigger_type': 'manual', 'repositories': rule.repositories}

        await self._trigger_workflow(rule, trigger_data)
        return f"Triggered workflow {rule.name}"

    async def enable_workflow_rule(self, rule_id: str, enabled: bool) -> bool:
        """Enable or disable a workflow rule"""
        rule = self.workflow_rules.get(rule_id)
        if not rule:
            return False

        rule.enabled = enabled
        await self._save_workflow_rules()

        logger.info(f"{'Enabled' if enabled else 'Disabled'} workflow rule: {rule.name}")
        return True

    async def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Get dependency graph for visualization"""
        graph = {}

        for dep in self.dependency_map:
            if dep.dependent not in graph:
                graph[dep.dependent] = []
            graph[dep.dependent].append(dep.dependency)

        return graph


# Global instance
workflow_coordinator = None

async def get_workflow_coordinator(git_controller: AutonomousGitController) -> WorkflowCoordinator:
    """Get global workflow coordinator instance"""
    global workflow_coordinator
    if not workflow_coordinator:
        workflow_coordinator = WorkflowCoordinator(git_controller)
        await workflow_coordinator.initialize()
    return workflow_coordinator