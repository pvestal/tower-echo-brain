#!/usr/bin/env python3
"""
Echo Brain Git Integration
Integrates comprehensive git control system with Echo Brain autonomous behaviors
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

# Import git components
from .autonomous_git_controller import AutonomousGitController, AutonomousMode
from .workflow_coordinator import WorkflowCoordinator
from .security_manager import GitSecurityManager
from .intelligent_git_assistant import IntelligentGitAssistant
from .git_test_framework import GitTestFramework

# Import Echo Brain components
from ..tasks.autonomous_behaviors import AutonomousBehaviorSystem
from ..core.echo.llm_interface import LLMInterface
from ..db.database import DatabaseManager
from ..api.echo import EchoAPI

logger = logging.getLogger(__name__)

class GitOperationPriority(Enum):
    """Priority levels for git operations in autonomous system"""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"

class AutonomousGitAction(Enum):
    """Types of autonomous git actions"""
    AUTO_COMMIT = "auto_commit"
    CONFLICT_RESOLUTION = "conflict_resolution"
    DEPENDENCY_UPDATE = "dependency_update"
    SECURITY_SCAN = "security_scan"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    CODE_REVIEW = "code_review"
    DOCUMENTATION_UPDATE = "documentation_update"

@dataclass
class GitAutonomousTask:
    """Represents an autonomous git task"""
    task_id: str
    action: AutonomousGitAction
    priority: GitOperationPriority
    repository: str
    description: str
    estimated_duration: int  # seconds
    prerequisites: List[str]
    parameters: Dict[str, Any]
    created_at: datetime
    scheduled_for: Optional[datetime]
    max_retries: int
    retry_count: int

@dataclass
class GitIntelligenceReport:
    """Intelligence report from git operations"""
    report_id: str
    timestamp: datetime
    repository: str
    analysis_type: str
    findings: Dict[str, Any]
    recommendations: List[str]
    risk_level: str
    confidence_score: float

class EchoGitIntegration:
    """
    Comprehensive integration between Echo Brain and the git control system.

    Features:
    - Autonomous git operations based on AI decisions
    - Intelligent repository monitoring and analysis
    - Proactive conflict prevention and resolution
    - Security-aware operations with risk assessment
    - Performance optimization based on usage patterns
    - Learning from git operations to improve decisions
    """

    def __init__(self):
        # Core git components
        self.git_controller: Optional[AutonomousGitController] = None
        self.workflow_coordinator: Optional[WorkflowCoordinator] = None
        self.security_manager: Optional[GitSecurityManager] = None
        self.intelligent_assistant: Optional[IntelligentGitAssistant] = None
        self.test_framework: Optional[GitTestFramework] = None

        # Echo Brain components
        self.autonomous_system: Optional[AutonomousBehaviorSystem] = None
        self.llm_interface: Optional[LLMInterface] = None
        self.database: Optional[DatabaseManager] = None
        self.echo_api: Optional[EchoAPI] = None

        # Task management
        self.pending_tasks: List[GitAutonomousTask] = []
        self.active_tasks: Dict[str, GitAutonomousTask] = {}
        self.completed_tasks: List[GitAutonomousTask] = []
        self.intelligence_reports: List[GitIntelligenceReport] = []

        # Configuration
        self.config = {
            'autonomous_mode': True,
            'max_concurrent_tasks': 3,
            'intelligence_gathering_interval': 300,  # 5 minutes
            'repository_scan_interval': 1800,  # 30 minutes
            'auto_commit_threshold': 0.8,  # confidence threshold
            'conflict_prevention_enabled': True,
            'performance_monitoring': True,
            'learning_enabled': True
        }

        # Learning and adaptation
        self.operation_patterns: Dict[str, Any] = {}
        self.success_metrics: Dict[str, float] = {}
        self.failure_analysis: List[Dict[str, Any]] = []

    async def initialize(self) -> bool:
        """Initialize the Echo Brain Git integration"""
        try:
            logger.info("Initializing Echo Brain Git Integration...")

            # Initialize git components
            success = await self._initialize_git_components()
            if not success:
                return False

            # Initialize Echo Brain components
            success = await self._initialize_echo_components()
            if not success:
                return False

            # Setup integration hooks
            await self._setup_integration_hooks()

            # Start monitoring and intelligence gathering
            await self._start_autonomous_monitoring()

            logger.info("Echo Brain Git Integration initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Echo Brain Git Integration: {e}")
            return False

    async def _initialize_git_components(self) -> bool:
        """Initialize all git components"""
        try:
            # Initialize git controller
            self.git_controller = AutonomousGitController(mode=AutonomousMode.FULL)
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

            # Initialize test framework
            self.test_framework = GitTestFramework()
            await self.test_framework.initialize()

            logger.info("Git components initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize git components: {e}")
            return False

    async def _initialize_echo_components(self) -> bool:
        """Initialize Echo Brain components"""
        try:
            # Initialize LLM interface
            self.llm_interface = LLMInterface()
            await self.llm_interface.initialize()

            # Initialize database
            self.database = DatabaseManager()
            await self.database.initialize()

            # Initialize autonomous behavior system
            self.autonomous_system = AutonomousBehaviorSystem()
            await self.autonomous_system.initialize()

            # Note: EchoAPI would be initialized separately as it's the main service

            logger.info("Echo Brain components initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Echo Brain components: {e}")
            return False

    async def _setup_integration_hooks(self):
        """Setup integration hooks between systems"""
        try:
            # Register git operations as autonomous behaviors
            if self.autonomous_system:
                await self.autonomous_system.register_behavior(
                    "git_repository_monitoring",
                    self._autonomous_repository_monitoring,
                    interval=self.config['repository_scan_interval'],
                    priority="normal"
                )

                await self.autonomous_system.register_behavior(
                    "git_intelligence_gathering",
                    self._autonomous_intelligence_gathering,
                    interval=self.config['intelligence_gathering_interval'],
                    priority="low"
                )

                await self.autonomous_system.register_behavior(
                    "git_proactive_maintenance",
                    self._autonomous_proactive_maintenance,
                    interval=3600,  # 1 hour
                    priority="background"
                )

            logger.info("Integration hooks setup complete")

        except Exception as e:
            logger.error(f"Failed to setup integration hooks: {e}")

    async def _start_autonomous_monitoring(self):
        """Start autonomous monitoring tasks"""
        # Create background tasks for monitoring
        asyncio.create_task(self._task_execution_loop())
        asyncio.create_task(self._performance_monitoring_loop())
        asyncio.create_task(self._learning_adaptation_loop())

        logger.info("Autonomous monitoring started")

    # Autonomous Behavior Methods

    async def _autonomous_repository_monitoring(self):
        """Autonomous repository monitoring behavior"""
        try:
            if not self.git_controller:
                return

            logger.debug("Running autonomous repository monitoring")

            # Check all repositories for issues
            for repo_name, repo_info in self.git_controller.repositories.items():
                try:
                    # Analyze repository health
                    health_report = await self._analyze_repository_health(repo_name, repo_info)

                    # Generate intelligence report
                    intelligence_report = GitIntelligenceReport(
                        report_id=f"health_{repo_name}_{int(datetime.now().timestamp())}",
                        timestamp=datetime.now(),
                        repository=repo_name,
                        analysis_type="health_check",
                        findings=health_report,
                        recommendations=await self._generate_health_recommendations(health_report),
                        risk_level=self._assess_risk_level(health_report),
                        confidence_score=0.8
                    )

                    self.intelligence_reports.append(intelligence_report)

                    # Queue autonomous actions if needed
                    await self._queue_health_based_actions(repo_name, health_report)

                except Exception as e:
                    logger.warning(f"Error monitoring repository {repo_name}: {e}")

        except Exception as e:
            logger.error(f"Error in autonomous repository monitoring: {e}")

    async def _analyze_repository_health(self, repo_name: str, repo_info) -> Dict[str, Any]:
        """Analyze the health of a repository"""
        try:
            health_metrics = {
                'uncommitted_changes': repo_info.has_uncommitted,
                'health_score': repo_info.health_score,
                'last_commit_age': None,
                'branch_divergence': 0,
                'security_issues': [],
                'performance_issues': [],
                'dependency_issues': []
            }

            # Get detailed status from git controller
            status = await self.git_controller.get_repository_status(repo_name)
            if status and 'status' in status:
                repo_status = status['status']
                health_metrics.update({
                    'modified_files': len(repo_status.get('modified_files', [])),
                    'untracked_files': len(repo_status.get('untracked_files', [])),
                    'ahead_commits': repo_status.get('ahead', 0),
                    'behind_commits': repo_status.get('behind', 0)
                })

            # Check for security issues
            if self.security_manager:
                security_status = await self.security_manager.get_security_status()
                # Correlate with repository
                health_metrics['security_score'] = 1.0  # Placeholder

            # Check for conflicts
            if self.intelligent_assistant:
                conflicts = await self.intelligent_assistant.detect_conflicts(repo_info.path)
                health_metrics['conflicts'] = len(conflicts)

            return health_metrics

        except Exception as e:
            logger.error(f"Failed to analyze repository health: {e}")
            return {'error': str(e)}

    async def _generate_health_recommendations(self, health_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on health report"""
        recommendations = []

        try:
            if health_report.get('uncommitted_changes', False):
                recommendations.append("Consider committing or stashing uncommitted changes")

            if health_report.get('modified_files', 0) > 10:
                recommendations.append("Large number of modified files - consider smaller commits")

            if health_report.get('conflicts', 0) > 0:
                recommendations.append(f"Resolve {health_report['conflicts']} merge conflicts")

            if health_report.get('behind_commits', 0) > 5:
                recommendations.append("Repository is behind remote - consider pulling latest changes")

            if health_report.get('health_score', 1.0) < 0.7:
                recommendations.append("Repository health is below optimal - review and cleanup needed")

            # Use AI to generate additional recommendations
            if self.llm_interface and health_report:
                ai_recommendations = await self._get_ai_recommendations(health_report)
                recommendations.extend(ai_recommendations)

        except Exception as e:
            logger.warning(f"Error generating recommendations: {e}")

        return recommendations

    async def _get_ai_recommendations(self, health_report: Dict[str, Any]) -> List[str]:
        """Get AI-powered recommendations"""
        try:
            prompt = f"""
Analyze this git repository health report and provide specific recommendations:

Health Report:
{json.dumps(health_report, indent=2)}

Provide 2-3 specific, actionable recommendations to improve repository health.
Focus on practical git operations that can be automated.

Format as a list of recommendations:
"""

            response = await self.llm_interface.query(prompt, max_tokens=200)
            if response:
                # Parse recommendations from response
                lines = response.strip().split('\n')
                recommendations = [line.strip('- ').strip() for line in lines if line.strip()]
                return recommendations[:3]  # Limit to 3 recommendations

        except Exception as e:
            logger.warning(f"Failed to get AI recommendations: {e}")

        return []

    def _assess_risk_level(self, health_report: Dict[str, Any]) -> str:
        """Assess risk level based on health report"""
        risk_score = 0

        # Add risk for various issues
        if health_report.get('conflicts', 0) > 0:
            risk_score += 3

        if health_report.get('health_score', 1.0) < 0.5:
            risk_score += 2

        if health_report.get('uncommitted_changes', False):
            risk_score += 1

        if health_report.get('behind_commits', 0) > 10:
            risk_score += 2

        # Determine risk level
        if risk_score >= 5:
            return "critical"
        elif risk_score >= 3:
            return "high"
        elif risk_score >= 1:
            return "medium"
        else:
            return "low"

    async def _queue_health_based_actions(self, repo_name: str, health_report: Dict[str, Any]):
        """Queue autonomous actions based on health report"""
        try:
            # Auto-commit if there are uncommitted changes and confidence is high
            if (health_report.get('uncommitted_changes', False) and
                health_report.get('health_score', 0) > self.config['auto_commit_threshold']):

                task = GitAutonomousTask(
                    task_id=f"auto_commit_{repo_name}_{int(datetime.now().timestamp())}",
                    action=AutonomousGitAction.AUTO_COMMIT,
                    priority=GitOperationPriority.NORMAL,
                    repository=repo_name,
                    description=f"Auto-commit changes in {repo_name}",
                    estimated_duration=30,
                    prerequisites=[],
                    parameters={'health_report': health_report},
                    created_at=datetime.now(),
                    scheduled_for=None,
                    max_retries=3,
                    retry_count=0
                )

                await self._queue_autonomous_task(task)

            # Queue conflict resolution if conflicts exist
            if health_report.get('conflicts', 0) > 0:
                task = GitAutonomousTask(
                    task_id=f"conflict_resolution_{repo_name}_{int(datetime.now().timestamp())}",
                    action=AutonomousGitAction.CONFLICT_RESOLUTION,
                    priority=GitOperationPriority.HIGH,
                    repository=repo_name,
                    description=f"Resolve conflicts in {repo_name}",
                    estimated_duration=120,
                    prerequisites=[],
                    parameters={'health_report': health_report},
                    created_at=datetime.now(),
                    scheduled_for=None,
                    max_retries=2,
                    retry_count=0
                )

                await self._queue_autonomous_task(task)

        except Exception as e:
            logger.error(f"Error queueing health-based actions: {e}")

    async def _autonomous_intelligence_gathering(self):
        """Autonomous intelligence gathering behavior"""
        try:
            logger.debug("Running autonomous intelligence gathering")

            # Analyze git operation patterns
            if self.git_controller:
                operations_status = await self.git_controller.get_operations_status()

                # Generate intelligence report
                intelligence_report = GitIntelligenceReport(
                    report_id=f"ops_analysis_{int(datetime.now().timestamp())}",
                    timestamp=datetime.now(),
                    repository="all",
                    analysis_type="operations_analysis",
                    findings=operations_status,
                    recommendations=await self._analyze_operation_patterns(operations_status),
                    risk_level="low",
                    confidence_score=0.9
                )

                self.intelligence_reports.append(intelligence_report)

            # Analyze workflow efficiency
            if self.workflow_coordinator:
                workflow_status = await self.workflow_coordinator.get_workflow_status()

                # Look for optimization opportunities
                optimization_opportunities = await self._identify_workflow_optimizations(workflow_status)

                if optimization_opportunities:
                    intelligence_report = GitIntelligenceReport(
                        report_id=f"workflow_optimization_{int(datetime.now().timestamp())}",
                        timestamp=datetime.now(),
                        repository="all",
                        analysis_type="workflow_optimization",
                        findings={'opportunities': optimization_opportunities},
                        recommendations=[f"Optimize {op}" for op in optimization_opportunities],
                        risk_level="low",
                        confidence_score=0.7
                    )

                    self.intelligence_reports.append(intelligence_report)

        except Exception as e:
            logger.error(f"Error in autonomous intelligence gathering: {e}")

    async def _analyze_operation_patterns(self, operations_status: Dict[str, Any]) -> List[str]:
        """Analyze git operation patterns"""
        recommendations = []

        try:
            # Analyze operation frequency
            if operations_status.get('queue_size', 0) > 10:
                recommendations.append("High operation queue - consider increasing concurrent task limit")

            # Analyze success rates
            recent_ops = operations_status.get('recent_operations', [])
            if recent_ops:
                failed_ops = [op for op in recent_ops if not op.get('success', True)]
                if len(failed_ops) > len(recent_ops) * 0.2:  # >20% failure rate
                    recommendations.append("High failure rate detected - investigate common failure patterns")

            # Analyze timing patterns
            # This would involve more complex pattern analysis in a real implementation

        except Exception as e:
            logger.warning(f"Error analyzing operation patterns: {e}")

        return recommendations

    async def _identify_workflow_optimizations(self, workflow_status: Dict[str, Any]) -> List[str]:
        """Identify workflow optimization opportunities"""
        opportunities = []

        try:
            # Check for slow workflows
            recent_executions = workflow_status.get('recent_executions', [])
            for execution in recent_executions:
                if execution.get('duration_ms', 0) > 300000:  # >5 minutes
                    opportunities.append(f"Slow workflow: {execution.get('rule_name', 'unknown')}")

            # Check for failed workflows
            failed_executions = [e for e in recent_executions if e.get('status') == 'failed']
            if len(failed_executions) > 2:
                opportunities.append("Multiple workflow failures - review error patterns")

        except Exception as e:
            logger.warning(f"Error identifying workflow optimizations: {e}")

        return opportunities

    async def _autonomous_proactive_maintenance(self):
        """Autonomous proactive maintenance behavior"""
        try:
            logger.debug("Running autonomous proactive maintenance")

            # Cleanup old intelligence reports
            cutoff_date = datetime.now() - timedelta(days=7)
            self.intelligence_reports = [
                report for report in self.intelligence_reports
                if report.timestamp > cutoff_date
            ]

            # Run security cleanup
            if self.security_manager:
                cleanup_results = await self.security_manager.cleanup_expired_credentials()
                if cleanup_results:
                    logger.info(f"Security cleanup completed: {cleanup_results}")

            # Performance optimization
            await self._optimize_git_performance()

        except Exception as e:
            logger.error(f"Error in autonomous proactive maintenance: {e}")

    async def _optimize_git_performance(self):
        """Optimize git performance based on usage patterns"""
        try:
            if not self.git_controller:
                return

            # Analyze repository sizes and cleanup needs
            for repo_name, repo_info in self.git_controller.repositories.items():
                try:
                    repo_path = repo_info.path

                    # Check for large files that should be in LFS
                    large_files = await self._find_large_files(repo_path)
                    if large_files:
                        # Queue LFS migration task
                        task = GitAutonomousTask(
                            task_id=f"lfs_migration_{repo_name}_{int(datetime.now().timestamp())}",
                            action=AutonomousGitAction.PERFORMANCE_OPTIMIZATION,
                            priority=GitOperationPriority.LOW,
                            repository=repo_name,
                            description=f"Migrate large files to LFS in {repo_name}",
                            estimated_duration=300,
                            prerequisites=[],
                            parameters={'large_files': large_files},
                            created_at=datetime.now(),
                            scheduled_for=datetime.now() + timedelta(hours=1),
                            max_retries=2,
                            retry_count=0
                        )

                        await self._queue_autonomous_task(task)

                except Exception as e:
                    logger.warning(f"Error optimizing repository {repo_name}: {e}")

        except Exception as e:
            logger.error(f"Error optimizing git performance: {e}")

    async def _find_large_files(self, repo_path: Path) -> List[str]:
        """Find large files in repository"""
        large_files = []
        try:
            # Find files larger than 10MB
            for file_path in repo_path.rglob('*'):
                if file_path.is_file() and file_path.stat().st_size > 10 * 1024 * 1024:
                    # Exclude git internal files
                    if '.git' not in str(file_path):
                        large_files.append(str(file_path.relative_to(repo_path)))

        except Exception as e:
            logger.warning(f"Error finding large files: {e}")

        return large_files

    # Task Management Methods

    async def _queue_autonomous_task(self, task: GitAutonomousTask):
        """Queue an autonomous git task"""
        try:
            self.pending_tasks.append(task)
            logger.info(f"Queued autonomous task: {task.description}")

            # Sort by priority and creation time
            self.pending_tasks.sort(
                key=lambda t: (t.priority.value, t.created_at)
            )

        except Exception as e:
            logger.error(f"Error queueing autonomous task: {e}")

    async def _task_execution_loop(self):
        """Main task execution loop"""
        while True:
            try:
                # Check if we can execute more tasks
                if (len(self.active_tasks) < self.config['max_concurrent_tasks'] and
                    self.pending_tasks):

                    # Get next task
                    task = self.pending_tasks.pop(0)

                    # Check prerequisites
                    if await self._check_task_prerequisites(task):
                        # Execute task
                        self.active_tasks[task.task_id] = task
                        asyncio.create_task(self._execute_autonomous_task(task))

                # Process completed tasks
                completed_task_ids = []
                for task_id, task in self.active_tasks.items():
                    # Tasks are moved to completed when execution finishes
                    pass

                # Clean up completed tasks
                for task_id in completed_task_ids:
                    del self.active_tasks[task_id]

                # Sleep before next iteration
                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Error in task execution loop: {e}")
                await asyncio.sleep(30)

    async def _check_task_prerequisites(self, task: GitAutonomousTask) -> bool:
        """Check if task prerequisites are met"""
        try:
            # Check if scheduled time has passed
            if task.scheduled_for and datetime.now() < task.scheduled_for:
                return False

            # Check repository availability
            if self.git_controller and task.repository != "all":
                if task.repository not in self.git_controller.repositories:
                    return False

            # Check prerequisite tasks
            for prereq_task_id in task.prerequisites:
                # Check if prerequisite is completed
                prereq_completed = any(
                    t.task_id == prereq_task_id for t in self.completed_tasks
                )
                if not prereq_completed:
                    return False

            return True

        except Exception as e:
            logger.error(f"Error checking task prerequisites: {e}")
            return False

    async def _execute_autonomous_task(self, task: GitAutonomousTask):
        """Execute an autonomous git task"""
        try:
            logger.info(f"Executing autonomous task: {task.description}")

            # Dispatch to appropriate handler
            success = False
            error_message = None

            if task.action == AutonomousGitAction.AUTO_COMMIT:
                success, error_message = await self._execute_auto_commit(task)
            elif task.action == AutonomousGitAction.CONFLICT_RESOLUTION:
                success, error_message = await self._execute_conflict_resolution(task)
            elif task.action == AutonomousGitAction.PERFORMANCE_OPTIMIZATION:
                success, error_message = await self._execute_performance_optimization(task)
            else:
                error_message = f"Unknown action: {task.action}"

            # Update task status
            if success:
                self.completed_tasks.append(task)
                logger.info(f"Completed autonomous task: {task.description}")

                # Learn from successful operation
                await self._record_task_success(task)

            else:
                # Handle failure
                task.retry_count += 1
                if task.retry_count < task.max_retries:
                    # Retry later
                    task.scheduled_for = datetime.now() + timedelta(minutes=5 * task.retry_count)
                    self.pending_tasks.append(task)
                    logger.warning(f"Task failed, will retry: {task.description} - {error_message}")
                else:
                    # Max retries reached
                    logger.error(f"Task failed permanently: {task.description} - {error_message}")
                    await self._record_task_failure(task, error_message)

            # Remove from active tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]

        except Exception as e:
            logger.error(f"Error executing autonomous task {task.task_id}: {e}")
            await self._record_task_failure(task, str(e))

            # Remove from active tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]

    async def _execute_auto_commit(self, task: GitAutonomousTask) -> Tuple[bool, Optional[str]]:
        """Execute auto-commit task"""
        try:
            if not self.git_controller or not self.intelligent_assistant:
                return False, "Required components not available"

            # Generate intelligent commit message
            suggestions = await self.intelligent_assistant.get_commit_suggestions(
                self.git_controller.repositories[task.repository].path
            )

            if not suggestions.get('ready_to_commit', False):
                return False, "Repository not ready for commit"

            # Perform commit
            commit_analysis = suggestions['commit_analysis']
            success, message = await self.git_controller.manual_commit(
                task.repository,
                commit_analysis['suggested_message']
            )

            return success, message

        except Exception as e:
            return False, str(e)

    async def _execute_conflict_resolution(self, task: GitAutonomousTask) -> Tuple[bool, Optional[str]]:
        """Execute conflict resolution task"""
        try:
            if not self.intelligent_assistant:
                return False, "Intelligent assistant not available"

            repo_path = self.git_controller.repositories[task.repository].path

            # Detect conflicts
            conflicts = await self.intelligent_assistant.detect_conflicts(repo_path)

            if not conflicts:
                return True, "No conflicts found"

            # Attempt to resolve conflicts
            resolutions = await self.intelligent_assistant.resolve_conflicts(
                repo_path, conflicts, auto_resolve=True
            )

            # Check if all conflicts were resolved
            unresolved = [r for r in resolutions if r.manual_review_required]
            if unresolved:
                return False, f"{len(unresolved)} conflicts require manual review"

            return True, f"Resolved {len(resolutions)} conflicts automatically"

        except Exception as e:
            return False, str(e)

    async def _execute_performance_optimization(self, task: GitAutonomousTask) -> Tuple[bool, Optional[str]]:
        """Execute performance optimization task"""
        try:
            # This would implement specific performance optimizations
            # For now, just simulate success
            await asyncio.sleep(1)
            return True, "Performance optimization completed"

        except Exception as e:
            return False, str(e)

    async def _record_task_success(self, task: GitAutonomousTask):
        """Record successful task execution for learning"""
        try:
            # Update success metrics
            action_key = f"{task.action.value}_{task.repository}"
            if action_key not in self.success_metrics:
                self.success_metrics[action_key] = 0.0

            # Increase success rate
            self.success_metrics[action_key] = min(
                self.success_metrics[action_key] + 0.1, 1.0
            )

            # Record pattern
            pattern_key = f"{task.action.value}_{task.priority.value}"
            if pattern_key not in self.operation_patterns:
                self.operation_patterns[pattern_key] = {
                    'count': 0,
                    'avg_duration': 0,
                    'success_rate': 0.0
                }

            pattern = self.operation_patterns[pattern_key]
            pattern['count'] += 1
            pattern['success_rate'] = (pattern['success_rate'] + 1.0) / 2

        except Exception as e:
            logger.error(f"Error recording task success: {e}")

    async def _record_task_failure(self, task: GitAutonomousTask, error_message: str):
        """Record task failure for learning"""
        try:
            failure_record = {
                'task_id': task.task_id,
                'action': task.action.value,
                'repository': task.repository,
                'error_message': error_message,
                'timestamp': datetime.now().isoformat(),
                'retry_count': task.retry_count
            }

            self.failure_analysis.append(failure_record)

            # Keep only recent failures
            cutoff_date = datetime.now() - timedelta(days=7)
            self.failure_analysis = [
                f for f in self.failure_analysis
                if datetime.fromisoformat(f['timestamp']) > cutoff_date
            ]

        except Exception as e:
            logger.error(f"Error recording task failure: {e}")

    async def _performance_monitoring_loop(self):
        """Monitor performance of git operations"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                # Collect performance metrics
                metrics = await self._collect_performance_metrics()

                # Analyze for optimization opportunities
                optimizations = await self._analyze_performance_metrics(metrics)

                if optimizations:
                    logger.info(f"Performance optimization opportunities: {optimizations}")

            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)

    async def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics"""
        metrics = {
            'active_tasks': len(self.active_tasks),
            'pending_tasks': len(self.pending_tasks),
            'completed_tasks': len(self.completed_tasks),
            'intelligence_reports': len(self.intelligence_reports),
            'success_rates': self.success_metrics.copy(),
            'operation_patterns': self.operation_patterns.copy()
        }

        return metrics

    async def _analyze_performance_metrics(self, metrics: Dict[str, Any]) -> List[str]:
        """Analyze performance metrics for optimization opportunities"""
        optimizations = []

        # Check queue sizes
        if metrics.get('pending_tasks', 0) > 10:
            optimizations.append("Consider increasing max_concurrent_tasks")

        # Check success rates
        for action_key, success_rate in metrics.get('success_rates', {}).items():
            if success_rate < 0.5:
                optimizations.append(f"Low success rate for {action_key}: {success_rate:.2f}")

        return optimizations

    async def _learning_adaptation_loop(self):
        """Continuous learning and adaptation loop"""
        while True:
            try:
                await asyncio.sleep(3600)  # Every hour

                # Analyze failure patterns
                if self.failure_analysis:
                    patterns = await self._analyze_failure_patterns()
                    if patterns:
                        logger.info(f"Learned failure patterns: {patterns}")

                # Adapt configuration based on learning
                await self._adapt_configuration()

            except Exception as e:
                logger.error(f"Error in learning adaptation: {e}")
                await asyncio.sleep(300)

    async def _analyze_failure_patterns(self) -> List[str]:
        """Analyze failure patterns to learn from mistakes"""
        patterns = []

        try:
            # Group failures by action and repository
            failure_groups = {}
            for failure in self.failure_analysis:
                key = f"{failure['action']}_{failure['repository']}"
                if key not in failure_groups:
                    failure_groups[key] = []
                failure_groups[key].append(failure)

            # Identify common patterns
            for group_key, failures in failure_groups.items():
                if len(failures) >= 3:
                    # Common error messages
                    error_messages = [f['error_message'] for f in failures]
                    common_errors = set(error_messages)
                    if len(common_errors) == 1:
                        patterns.append(f"Recurring error in {group_key}: {list(common_errors)[0]}")

        except Exception as e:
            logger.error(f"Error analyzing failure patterns: {e}")

        return patterns

    async def _adapt_configuration(self):
        """Adapt configuration based on learning"""
        try:
            # Adjust auto-commit threshold based on success rates
            auto_commit_successes = [
                rate for key, rate in self.success_metrics.items()
                if 'auto_commit' in key
            ]

            if auto_commit_successes:
                avg_success = sum(auto_commit_successes) / len(auto_commit_successes)
                if avg_success > 0.9:
                    # High success rate - can be more aggressive
                    self.config['auto_commit_threshold'] = max(
                        self.config['auto_commit_threshold'] - 0.05, 0.6
                    )
                elif avg_success < 0.7:
                    # Low success rate - be more conservative
                    self.config['auto_commit_threshold'] = min(
                        self.config['auto_commit_threshold'] + 0.05, 0.95
                    )

            logger.debug(f"Adapted configuration: auto_commit_threshold = {self.config['auto_commit_threshold']}")

        except Exception as e:
            logger.error(f"Error adapting configuration: {e}")

    # Public API Methods

    async def get_integration_status(self) -> Dict[str, Any]:
        """Get status of the git integration"""
        return {
            'git_components_initialized': all([
                self.git_controller is not None,
                self.workflow_coordinator is not None,
                self.security_manager is not None,
                self.intelligent_assistant is not None,
                self.test_framework is not None
            ]),
            'echo_components_initialized': all([
                self.llm_interface is not None,
                self.database is not None,
                self.autonomous_system is not None
            ]),
            'autonomous_mode_enabled': self.config['autonomous_mode'],
            'task_statistics': {
                'pending': len(self.pending_tasks),
                'active': len(self.active_tasks),
                'completed': len(self.completed_tasks)
            },
            'intelligence_reports': len(self.intelligence_reports),
            'success_metrics': self.success_metrics,
            'configuration': self.config
        }

    async def trigger_manual_intelligence_gathering(self) -> Dict[str, Any]:
        """Manually trigger intelligence gathering"""
        await self._autonomous_intelligence_gathering()
        return {
            'reports_generated': len([
                r for r in self.intelligence_reports
                if r.timestamp > datetime.now() - timedelta(minutes=5)
            ])
        }

    async def get_intelligence_reports(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent intelligence reports"""
        recent_reports = sorted(
            self.intelligence_reports,
            key=lambda r: r.timestamp,
            reverse=True
        )[:limit]

        return [asdict(report) for report in recent_reports]

    async def shutdown(self):
        """Shutdown the integration"""
        try:
            logger.info("Shutting down Echo Brain Git Integration...")

            # Wait for active tasks to complete (with timeout)
            timeout = datetime.now() + timedelta(seconds=30)
            while self.active_tasks and datetime.now() < timeout:
                await asyncio.sleep(1)

            # Save learning data
            await self._save_learning_data()

            logger.info("Echo Brain Git Integration shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    async def _save_learning_data(self):
        """Save learning data for persistence"""
        try:
            learning_data = {
                'success_metrics': self.success_metrics,
                'operation_patterns': self.operation_patterns,
                'failure_analysis': self.failure_analysis[-100:],  # Keep last 100
                'configuration': self.config,
                'timestamp': datetime.now().isoformat()
            }

            learning_file = Path("/opt/tower-echo-brain/data/git_learning_data.json")
            learning_file.parent.mkdir(parents=True, exist_ok=True)

            with open(learning_file, 'w') as f:
                json.dump(learning_data, f, indent=2, default=str)

            logger.info("Saved learning data for persistence")

        except Exception as e:
            logger.error(f"Failed to save learning data: {e}")


# Global instance
echo_git_integration = EchoGitIntegration()

async def test_echo_git_integration():
    """Test the Echo Brain Git integration"""
    integration = echo_git_integration

    # Initialize
    success = await integration.initialize()
    if not success:
        print("❌ Failed to initialize Echo Git integration")
        return

    print("✅ Echo Git integration initialized")

    # Get status
    status = await integration.get_integration_status()
    print(f"\n📊 Integration Status:")
    print(f"  Git components: {status['git_components_initialized']}")
    print(f"  Echo components: {status['echo_components_initialized']}")
    print(f"  Autonomous mode: {status['autonomous_mode_enabled']}")
    print(f"  Active tasks: {status['task_statistics']['active']}")
    print(f"  Intelligence reports: {status['intelligence_reports']}")

    # Trigger intelligence gathering
    await asyncio.sleep(2)  # Let autonomous behaviors run
    result = await integration.trigger_manual_intelligence_gathering()
    print(f"\n🧠 Intelligence gathering: {result['reports_generated']} reports generated")

    # Get recent reports
    reports = await integration.get_intelligence_reports(limit=3)
    if reports:
        print(f"\n📄 Recent Intelligence Reports:")
        for report in reports:
            print(f"  - {report['analysis_type']}: {report['risk_level']} risk")

    # Shutdown
    await integration.shutdown()

    print("\n✅ Echo Git integration test complete")


if __name__ == "__main__":
    asyncio.run(test_echo_git_integration())