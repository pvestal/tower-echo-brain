#!/usr/bin/env python3
"""
Echo Task Decomposer - Autonomous Task Breakdown and Management
=============================================================

Echo's intelligent task decomposition system:
- Breaks down complex requests into manageable subtasks
- Analyzes dependencies and optimal execution order
- Manages task execution and tracks progress
- Learns from task completion patterns
- Provides intelligent task scheduling and resource allocation

This system enables Echo to handle complex, multi-step requests
autonomously while maintaining visibility and control over the process.
"""

import asyncio
import concurrent.futures
import json
import logging
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from bin.echo_board_manager import (BoardDecisionType, consult_board,
                                get_board_manager)
# Import learning and board management systems
from src.core.echo.echo_learning_system import (DecisionOutcome, LearningDomain,
                                  get_learning_system, record_decision,
                                  record_outcome)

# Configuration
TASK_DECOMPOSER_DB_PATH = "/opt/tower-echo-brain/data/echo_task_decomposer.db"
MAX_TASK_DEPTH = 5
MIN_TASK_COMPLEXITY = 0.3
MAX_PARALLEL_TASKS = 5
TASK_TIMEOUT_HOURS = 24


class TaskStatus(Enum):
    PENDING = "pending"
    ANALYZING = "analyzing"
    IN_PROGRESS = "in_progress"
    WAITING_DEPENDENCY = "waiting_dependency"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class TaskType(Enum):
    ANALYSIS = "analysis"
    IMPLEMENTATION = "implementation"
    RESEARCH = "research"
    DECISION = "decision"
    COMMUNICATION = "communication"
    VALIDATION = "validation"
    ORCHESTRATION = "orchestration"


class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


@dataclass
class Task:
    """Individual task within a larger workflow"""

    task_id: str
    parent_request_id: str
    title: str
    description: str
    task_type: TaskType
    priority: TaskPriority
    status: TaskStatus
    complexity_score: float
    estimated_duration: float  # hours
    dependencies: List[str]  # task_ids this task depends on
    blockers: List[str]  # task_ids this task blocks
    required_resources: List[str]
    assigned_tools: List[str]  # Tools/systems to use for this task
    context: Dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    result: Optional[Dict[str, Any]]
    error_info: Optional[Dict[str, Any]]
    progress_notes: List[Dict[str, Any]]
    learning_insights: List[str]


@dataclass
class TaskWorkflow:
    """Complete workflow containing multiple related tasks"""

    workflow_id: str
    original_request: str
    user_context: Dict[str, Any]
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    overall_status: TaskStatus
    estimated_completion: datetime
    actual_completion: Optional[datetime]
    decomposition_strategy: str
    execution_plan: Dict[str, Any]
    resource_allocation: Dict[str, Any]
    progress_history: List[Dict[str, Any]]
    final_result: Optional[Dict[str, Any]]
    learning_summary: Dict[str, Any]


@dataclass
class DecompositionPattern:
    """Learned pattern for task decomposition"""

    pattern_id: str
    request_type: str
    context_indicators: List[str]
    typical_breakdown: List[Dict[str, Any]]
    success_rate: float
    average_completion_time: float
    common_bottlenecks: List[str]
    optimization_suggestions: List[str]
    usage_count: int
    last_updated: datetime


class EchoTaskDecomposer:
    """
    Echo's Intelligent Task Decomposition System

    This system enables Echo to break down complex requests into manageable
    tasks, optimize execution order, and learn from completion patterns.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_path = TASK_DECOMPOSER_DB_PATH
        self._ensure_data_directory()
        self._init_database()

        # Active workflow management
        self.active_workflows: Dict[str, TaskWorkflow] = {}
        self.active_tasks: Dict[str, Task] = {}
        self.task_dependencies: nx.DiGraph = nx.DiGraph()

        # Decomposition patterns and learning
        self.decomposition_patterns: Dict[str, DecompositionPattern] = {}
        self.execution_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=MAX_PARALLEL_TASKS
        )

        # Integration with other systems
        self.learning_system = get_learning_system()
        self.board_manager = get_board_manager()

        self.logger.info("Echo Task Decomposer initialized")

    def _ensure_data_directory(self):
        """Ensure task decomposer data directory exists"""
        data_dir = Path(self.db_path).parent
        data_dir.mkdir(parents=True, exist_ok=True)

    def _init_database(self):
        """Initialize task decomposer database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    parent_request_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    complexity_score REAL NOT NULL,
                    estimated_duration REAL NOT NULL,
                    dependencies TEXT,
                    blockers TEXT,
                    required_resources TEXT,
                    assigned_tools TEXT,
                    context TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    result TEXT,
                    error_info TEXT,
                    progress_notes TEXT,
                    learning_insights TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workflows (
                    workflow_id TEXT PRIMARY KEY,
                    original_request TEXT NOT NULL,
                    user_context TEXT NOT NULL,
                    total_tasks INTEGER NOT NULL,
                    completed_tasks INTEGER NOT NULL,
                    failed_tasks INTEGER NOT NULL,
                    overall_status TEXT NOT NULL,
                    estimated_completion TEXT NOT NULL,
                    actual_completion TEXT,
                    decomposition_strategy TEXT NOT NULL,
                    execution_plan TEXT NOT NULL,
                    resource_allocation TEXT,
                    progress_history TEXT,
                    final_result TEXT,
                    learning_summary TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decomposition_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    request_type TEXT NOT NULL,
                    context_indicators TEXT NOT NULL,
                    typical_breakdown TEXT NOT NULL,
                    success_rate REAL NOT NULL,
                    average_completion_time REAL NOT NULL,
                    common_bottlenecks TEXT,
                    optimization_suggestions TEXT,
                    usage_count INTEGER NOT NULL,
                    last_updated TEXT NOT NULL
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_logs (
                    log_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    task_id TEXT,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    details TEXT,
                    performance_metrics TEXT
                )
            """
            )

            conn.commit()

    async def decompose_request(
        self,
        request: str,
        user_context: Dict[str, Any],
        priority: TaskPriority = TaskPriority.MEDIUM,
        deadline: Optional[datetime] = None,
    ) -> str:
        """
        Decompose a complex request into manageable tasks

        Args:
            request: The original user request
            user_context: Context about the request and user needs
            priority: Overall priority of the request
            deadline: Optional deadline for completion

        Returns:
            workflow_id: Unique identifier for the workflow
        """
        workflow_id = (
            f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )

        self.logger.info(f"Decomposing request: {workflow_id}")

        # Record this as a learning decision
        learning_decision_id = await record_decision(
            domain=LearningDomain.TASK_DECOMPOSITION,
            context={
                "request": request,
                "user_context": user_context,
                "priority": priority.value,
                "has_deadline": deadline is not None,
            },
            decision_factors={
                "request_complexity": self._analyze_request_complexity(
                    request, user_context
                ),
                "context_richness": len(user_context) / 10,
                "priority_level": priority.value / 5,
                "time_pressure": 1.0 if deadline else 0.5,
            },
            decision_made="request_decomposition",
            confidence=0.8,
            meta_data={"workflow_id": workflow_id},
        )

        try:
            # Analyze the request to determine decomposition strategy
            decomposition_analysis = await self._analyze_decomposition_requirements(
                request, user_context, priority, deadline
            )

            # Check for existing patterns
            matching_pattern = await self._find_matching_pattern(request, user_context)

            # Generate task breakdown
            tasks = await self._generate_task_breakdown(
                request, user_context, decomposition_analysis, matching_pattern
            )

            # Optimize task order and dependencies
            optimized_plan = await self._optimize_execution_plan(tasks, deadline)

            # Create workflow
            workflow = TaskWorkflow(
                workflow_id=workflow_id,
                original_request=request,
                user_context=user_context,
                total_tasks=len(tasks),
                completed_tasks=0,
                failed_tasks=0,
                overall_status=TaskStatus.PENDING,
                estimated_completion=optimized_plan["estimated_completion"],
                actual_completion=None,
                decomposition_strategy=decomposition_analysis["strategy"],
                execution_plan=optimized_plan,
                resource_allocation=optimized_plan["resource_allocation"],
                progress_history=[],
                final_result=None,
                learning_summary={},
            )

            # Store workflow and tasks
            self.active_workflows[workflow_id] = workflow
            for task in tasks:
                self.active_tasks[task.task_id] = task
                self._add_task_to_dependency_graph(task)

            await self._store_workflow(workflow)
            for task in tasks:
                await self._store_task(task)

            # Log decomposition
            await self._log_execution_event(
                workflow_id,
                None,
                "decomposition_complete",
                {
                    "total_tasks": len(tasks),
                    "strategy": decomposition_analysis["strategy"],
                },
            )

            # Record successful decomposition
            await record_outcome(
                learning_decision_id,
                DecisionOutcome.SUCCESS,
                {
                    "tasks_generated": len(tasks),
                    "complexity_handled": decomposition_analysis["complexity"],
                    "strategy_used": decomposition_analysis["strategy"],
                },
                f"Successfully decomposed into {len(tasks)} tasks",
            )

            self.logger.info(
                f"Request decomposed into {len(tasks)} tasks: {workflow_id}"
            )
            return workflow_id

        except Exception as e:
            self.logger.error(f"Request decomposition failed: {e}")

            # Record failure
            await record_outcome(
                learning_decision_id,
                DecisionOutcome.FAILURE,
                {"error": str(e), "decomposition_attempted": True},
                f"Decomposition failed: {e}",
            )

            raise

    async def _analyze_decomposition_requirements(
        self,
        request: str,
        user_context: Dict[str, Any],
        priority: TaskPriority,
        deadline: Optional[datetime],
    ) -> Dict[str, Any]:
        """Analyze what type of decomposition is needed"""
        # Analyze request complexity
        complexity = self._analyze_request_complexity(request, user_context)

        # Determine if we need board consultation
        needs_board_input = complexity > 0.7 or priority.value >= 4

        if needs_board_input:
            # Consult board for complex decomposition
            board_result = await consult_board(
                f"How should we decompose this complex request: {request}",
                {
                    "user_context": user_context,
                    "complexity": complexity,
                    "priority": priority.value,
                    "deadline": deadline.isoformat() if deadline else None,
                },
                decision_type=BoardDecisionType.EXPERT_REVIEW,
                domain="task_planning",
            )

            strategy = (
                "board_guided" if board_result["consensus_reached"] else "ai_autonomous"
            )
            board_insights = board_result.get("individual_insights", [])
        else:
            strategy = "pattern_based"
            board_insights = []

        return {
            "complexity": complexity,
            "strategy": strategy,
            "needs_parallel_execution": complexity > 0.6,
            "estimated_task_count": max(2, int(complexity * 10)),
            "board_insights": board_insights,
            "risk_factors": self._identify_risk_factors(request, user_context),
            "resource_requirements": self._analyze_resource_requirements(
                request, user_context
            ),
        }

    def _analyze_request_complexity(
        self, request: str, user_context: Dict[str, Any]
    ) -> float:
        """Analyze the complexity of a request (0.0 to 1.0)"""
        complexity_indicators = [
            len(request.split()) / 100,  # Length factor
            len(user_context) / 20,  # Context richness
            request.lower().count("and") / 10,  # Conjunction complexity
            request.lower().count("after") / 5,  # Sequence complexity
            int("integrate" in request.lower()) *
            0.3,  # Integration complexity
            int("analyze" in request.lower()) * 0.2,  # Analysis complexity
            int("implement" in request.lower()) *
            0.4,  # Implementation complexity
        ]

        # Check for complexity keywords
        complex_keywords = [
            "optimize",
            "comprehensive",
            "systematic",
            "automated",
            "intelligent",
        ]
        keyword_factor = sum(
            1 for keyword in complex_keywords if keyword in request.lower()
        ) / len(complex_keywords)

        raw_complexity = sum(complexity_indicators) + keyword_factor
        return min(1.0, raw_complexity)

    def _identify_risk_factors(
        self, request: str, user_context: Dict[str, Any]
    ) -> List[str]:
        """Identify potential risk factors in the request"""
        risks = []

        # Technical risks
        if "integrate" in request.lower():
            risks.append("integration_complexity")
        if "performance" in request.lower():
            risks.append("performance_impact")
        if "security" in request.lower():
            risks.append("security_implications")

        # Resource risks
        if user_context.get("resource_constraints"):
            risks.append("resource_limitations")
        if user_context.get("tight_deadline"):
            risks.append("time_constraints")

        # Scope risks
        if "comprehensive" in request.lower():
            risks.append("scope_creep")
        if "all" in request.lower() or "every" in request.lower():
            risks.append("overambitious_scope")

        return risks

    def _analyze_resource_requirements(
        self, request: str, user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze what resources will be needed"""
        resources = {
            "computational": 0.5,
            "storage": 0.3,
            "network": 0.3,
            "human_oversight": 0.4,
            "external_services": [],
        }

        # Adjust based on request content
        if "video" in request.lower() or "image" in request.lower():
            resources["computational"] = 0.9
            resources["storage"] = 0.8

        if "database" in request.lower() or "data" in request.lower():
            resources["storage"] = 0.7

        if "api" in request.lower() or "service" in request.lower():
            resources["network"] = 0.8

        if "critical" in request.lower() or "important" in request.lower():
            resources["human_oversight"] = 0.8

        # Identify external services
        service_keywords = {
            "comfyui": "comfyui",
            "anime": "anime_service",
            "music": "apple_music",
            "notification": "notification_service",
        }

        for keyword, service in service_keywords.items():
            if keyword in request.lower():
                resources["external_services"].append(service)

        return resources

    async def _find_matching_pattern(
        self, request: str, user_context: Dict[str, Any]
    ) -> Optional[DecompositionPattern]:
        """Find matching decomposition pattern from history"""
        # Analyze request characteristics
        request_type = self._classify_request_type(request)
        context_indicators = self._extract_context_indicators(user_context)

        # Look for matching patterns
        best_match = None
        best_score = 0.0

        for pattern in self.decomposition_patterns.values():
            if pattern.request_type == request_type:
                # Calculate similarity score
                indicator_overlap = len(
                    set(pattern.context_indicators) & set(context_indicators)
                )
                max_indicators = max(
                    len(pattern.context_indicators), len(context_indicators)
                )
                similarity = (
                    indicator_overlap / max_indicators if max_indicators > 0 else 0.0
                )

                # Weight by success rate and usage count
                score = (
                    similarity
                    * pattern.success_rate
                    * min(1.0, pattern.usage_count / 10)
                )

                if score > best_score and score > 0.5:  # Minimum similarity threshold
                    best_score = score
                    best_match = pattern

        if best_match:
            self.logger.info(
                f"Found matching pattern: {best_match.pattern_id} (score: {best_score:.2f})"
            )

        return best_match

    def _classify_request_type(self, request: str) -> str:
        """Classify the type of request"""
        request_lower = request.lower()

        # Define classification keywords
        classifications = {
            "analysis": ["analyze", "review", "evaluate", "assess"],
            "implementation": ["implement", "create", "build", "develop"],
            "optimization": ["optimize", "improve", "enhance", "tune"],
            "integration": ["integrate", "connect", "combine", "merge"],
            "maintenance": ["fix", "repair", "update", "maintain"],
            "research": ["research", "investigate", "explore", "study"],
        }

        # Find best match
        best_type = "general"
        max_matches = 0

        for req_type, keywords in classifications.items():
            matches = sum(
                1 for keyword in keywords if keyword in request_lower)
            if matches > max_matches:
                max_matches = matches
                best_type = req_type

        return best_type

    def _extract_context_indicators(self, user_context: Dict[str, Any]) -> List[str]:
        """Extract key indicators from user context"""
        indicators = []

        # Direct indicators
        for key, value in user_context.items():
            if isinstance(value, bool) and value:
                indicators.append(key)
            elif isinstance(value, str) and value:
                indicators.append(f"{key}_{value}")
            elif isinstance(value, (int, float)) and value > 0:
                indicators.append(key)

        # Derived indicators
        if "deadline" in user_context:
            indicators.append("time_sensitive")
        if "priority" in user_context and user_context["priority"] == "high":
            indicators.append("high_priority")
        if "resources" in user_context:
            indicators.append("resource_aware")

        return indicators

    async def _generate_task_breakdown(
        self,
        request: str,
        user_context: Dict[str, Any],
        analysis: Dict[str, Any],
        pattern: Optional[DecompositionPattern],
    ) -> List[Task]:
        """Generate the actual task breakdown"""
        tasks = []

        if pattern and pattern.success_rate > 0.7:
            # Use pattern-based decomposition
            tasks = await self._generate_tasks_from_pattern(
                request, user_context, pattern, analysis
            )
        else:
            # Generate tasks based on analysis
            tasks = await self._generate_tasks_from_analysis(
                request, user_context, analysis
            )

        # Add universal tasks if needed
        tasks.extend(await self._add_universal_tasks(request, user_context, analysis))

        # Assign IDs and metadata
        for i, task in enumerate(tasks):
            if not task.task_id:
                task.task_id = f"task_{i+1}_{uuid.uuid4().hex[:8]}"
            task.created_at = datetime.now()

        return tasks

    async def _generate_tasks_from_pattern(
        self,
        request: str,
        user_context: Dict[str, Any],
        pattern: DecompositionPattern,
        analysis: Dict[str, Any],
    ) -> List[Task]:
        """Generate tasks based on a successful pattern"""
        tasks = []

        for task_template in pattern.typical_breakdown:
            # Customize template for current request
            task = Task(
                task_id="",  # Will be assigned later
                parent_request_id="",  # Will be set by caller
                title=task_template["title"].format(request=request),
                description=task_template["description"],
                task_type=TaskType(task_template["type"]),
                priority=TaskPriority(task_template.get("priority", 2)),
                status=TaskStatus.PENDING,
                complexity_score=task_template.get("complexity", 0.5),
                estimated_duration=task_template.get("duration", 1.0),
                dependencies=task_template.get("dependencies", []),
                blockers=[],
                required_resources=task_template.get("resources", []),
                assigned_tools=task_template.get("tools", []),
                context={"pattern_derived": True,
                         "pattern_id": pattern.pattern_id},
                created_at=datetime.now(),
                started_at=None,
                completed_at=None,
                result=None,
                error_info=None,
                progress_notes=[],
                learning_insights=[],
            )
            tasks.append(task)

        # Update pattern usage
        pattern.usage_count += 1
        pattern.last_updated = datetime.now()
        await self._store_decomposition_pattern(pattern)

        return tasks

    async def _generate_tasks_from_analysis(
        self, request: str, user_context: Dict[str, Any], analysis: Dict[str, Any]
    ) -> List[Task]:
        """Generate tasks based on analysis when no pattern exists"""
        tasks = []
        strategy = analysis["strategy"]

        if strategy == "board_guided":
            # Use board insights to generate tasks
            tasks = await self._generate_board_guided_tasks(
                request, user_context, analysis
            )
        elif strategy == "ai_autonomous":
            # Use AI reasoning for complex decomposition
            tasks = await self._generate_ai_autonomous_tasks(
                request, user_context, analysis
            )
        else:
            # Use standard pattern-based approach
            tasks = await self._generate_standard_tasks(request, user_context, analysis)

        return tasks

    async def _generate_board_guided_tasks(
        self, request: str, user_context: Dict[str, Any], analysis: Dict[str, Any]
    ) -> List[Task]:
        """Generate tasks using board guidance"""
        tasks = []

        # Extract insights from board consultation
        board_insights = analysis.get("board_insights", [])

        # Create tasks based on board recommendations
        for insight in board_insights:
            if "insight" in insight:
                insight_text = insight["insight"]

                # Parse insight for actionable tasks
                if "analyze" in insight_text.lower():
                    tasks.append(self._create_analysis_task(
                        insight_text, user_context))
                elif "implement" in insight_text.lower():
                    tasks.append(
                        self._create_implementation_task(
                            insight_text, user_context)
                    )
                elif "validate" in insight_text.lower():
                    tasks.append(
                        self._create_validation_task(
                            insight_text, user_context)
                    )

        # Ensure we have at least basic tasks
        if not tasks:
            tasks = await self._generate_standard_tasks(request, user_context, analysis)

        return tasks

    async def _generate_ai_autonomous_tasks(
        self, request: str, user_context: Dict[str, Any], analysis: Dict[str, Any]
    ) -> List[Task]:
        """Generate tasks using autonomous AI reasoning"""
        tasks = []

        # Break down by logical components
        complexity = analysis["complexity"]
        estimated_tasks = analysis["estimated_task_count"]

        # Create phases
        if complexity > 0.8:
            # High complexity - create phases
            tasks.extend(
                [
                    self._create_task(
                        "Research and Analysis",
                        "Conduct thorough research and analysis of requirements",
                        TaskType.RESEARCH,
                        TaskPriority.HIGH,
                        complexity * 0.3,
                        2.0,
                    ),
                    self._create_task(
                        "Planning and Design",
                        "Create detailed implementation plan and design",
                        TaskType.ANALYSIS,
                        TaskPriority.HIGH,
                        complexity * 0.4,
                        3.0,
                    ),
                    self._create_task(
                        "Implementation",
                        "Execute the planned implementation",
                        TaskType.IMPLEMENTATION,
                        TaskPriority.MEDIUM,
                        complexity * 0.6,
                        5.0,
                    ),
                    self._create_task(
                        "Validation and Testing",
                        "Validate implementation and test functionality",
                        TaskType.VALIDATION,
                        TaskPriority.HIGH,
                        complexity * 0.4,
                        2.0,
                    ),
                ]
            )
        else:
            # Moderate complexity - create direct tasks
            if "analyze" in request.lower():
                tasks.append(
                    self._create_task(
                        "Analysis Task",
                        f"Analyze: {request}",
                        TaskType.ANALYSIS,
                        TaskPriority.MEDIUM,
                        complexity,
                        2.0,
                    )
                )

            if "implement" in request.lower():
                tasks.append(
                    self._create_task(
                        "Implementation Task",
                        f"Implement: {request}",
                        TaskType.IMPLEMENTATION,
                        TaskPriority.MEDIUM,
                        complexity,
                        4.0,
                    )
                )

        return tasks

    async def _generate_standard_tasks(
        self, request: str, user_context: Dict[str, Any], analysis: Dict[str, Any]
    ) -> List[Task]:
        """Generate standard task breakdown"""
        tasks = []

        # Basic task structure
        tasks.extend(
            [
                self._create_task(
                    "Initial Analysis",
                    f"Analyze requirements for: {request}",
                    TaskType.ANALYSIS,
                    TaskPriority.MEDIUM,
                    0.4,
                    1.0,
                ),
                self._create_task(
                    "Planning",
                    "Create implementation plan",
                    TaskType.ANALYSIS,
                    TaskPriority.MEDIUM,
                    0.5,
                    1.5,
                ),
                self._create_task(
                    "Execution",
                    "Execute the planned actions",
                    TaskType.IMPLEMENTATION,
                    TaskPriority.MEDIUM,
                    0.7,
                    3.0,
                ),
                self._create_task(
                    "Verification",
                    "Verify results and completion",
                    TaskType.VALIDATION,
                    TaskPriority.HIGH,
                    0.3,
                    0.5,
                ),
            ]
        )

        return tasks

    def _create_task(
        self,
        title: str,
        description: str,
        task_type: TaskType,
        priority: TaskPriority,
        complexity: float,
        duration: float,
    ) -> Task:
        """Create a task with standard parameters"""
        return Task(
            task_id="",  # Will be assigned later
            parent_request_id="",  # Will be set by caller
            title=title,
            description=description,
            task_type=task_type,
            priority=priority,
            status=TaskStatus.PENDING,
            complexity_score=complexity,
            estimated_duration=duration,
            dependencies=[],
            blockers=[],
            required_resources=[],
            assigned_tools=[],
            context={},
            created_at=datetime.now(),
            started_at=None,
            completed_at=None,
            result=None,
            error_info=None,
            progress_notes=[],
            learning_insights=[],
        )

    def _create_analysis_task(self, insight: str, context: Dict[str, Any]) -> Task:
        """Create an analysis task from board insight"""
        return self._create_task(
            "Board-Guided Analysis",
            f"Analysis based on board insight: {insight}",
            TaskType.ANALYSIS,
            TaskPriority.HIGH,
            0.6,
            2.0,
        )

    def _create_implementation_task(
        self, insight: str, context: Dict[str, Any]
    ) -> Task:
        """Create an implementation task from board insight"""
        return self._create_task(
            "Board-Guided Implementation",
            f"Implementation based on board insight: {insight}",
            TaskType.IMPLEMENTATION,
            TaskPriority.MEDIUM,
            0.7,
            4.0,
        )

    def _create_validation_task(self, insight: str, context: Dict[str, Any]) -> Task:
        """Create a validation task from board insight"""
        return self._create_task(
            "Board-Guided Validation",
            f"Validation based on board insight: {insight}",
            TaskType.VALIDATION,
            TaskPriority.HIGH,
            0.4,
            1.0,
        )

    async def _add_universal_tasks(
        self, request: str, user_context: Dict[str, Any], analysis: Dict[str, Any]
    ) -> List[Task]:
        """Add universal tasks that apply to most workflows"""
        universal_tasks = []

        # Add final reporting task
        universal_tasks.append(
            self._create_task(
                "Final Report",
                "Generate final report and summary",
                TaskType.COMMUNICATION,
                TaskPriority.LOW,
                0.2,
                0.5,
            )
        )

        # Add learning task
        universal_tasks.append(
            self._create_task(
                "Learning Integration",
                "Integrate lessons learned from this workflow",
                TaskType.ANALYSIS,
                TaskPriority.LOW,
                0.3,
                0.5,
            )
        )

        return universal_tasks

    async def _optimize_execution_plan(
        self, tasks: List[Task], deadline: Optional[datetime]
    ) -> Dict[str, Any]:
        """Optimize task execution order and dependencies"""
        # Set up dependency relationships
        for i, task in enumerate(tasks):
            task.parent_request_id = tasks[0].parent_request_id if tasks else "unknown"

            # Set basic dependencies (each task depends on previous for now)
            if i > 0:
                task.dependencies = [tasks[i - 1].task_id]
                tasks[i - 1].blockers = [task.task_id]

        # Calculate resource allocation
        total_computational = sum(
            0.5 for task in tasks if task.task_type == TaskType.IMPLEMENTATION
        )
        total_duration = sum(task.estimated_duration for task in tasks)

        # Estimate completion time
        if deadline:
            estimated_completion = deadline
        else:
            estimated_completion = datetime.now() + timedelta(hours=total_duration)

        # Create execution plan
        execution_plan = {
            "execution_order": [task.task_id for task in tasks],
            "parallel_groups": self._identify_parallel_groups(tasks),
            "critical_path": self._calculate_critical_path(tasks),
            "estimated_completion": estimated_completion,
            "resource_allocation": {
                "computational_load": min(1.0, total_computational),
                "storage_requirements": 0.5,  # Default estimate
                "network_usage": 0.3,  # Default estimate
                "human_oversight_needed": any(
                    task.task_type == TaskType.DECISION for task in tasks
                ),
            },
        }

        return execution_plan

    def _identify_parallel_groups(self, tasks: List[Task]) -> List[List[str]]:
        """Identify tasks that can be executed in parallel"""
        # For now, use simple heuristic - tasks with no dependencies can run in parallel
        parallel_groups = []
        independent_tasks = [
            task.task_id for task in tasks if not task.dependencies]

        if len(independent_tasks) > 1:
            parallel_groups.append(independent_tasks)

        return parallel_groups

    def _calculate_critical_path(self, tasks: List[Task]) -> List[str]:
        """Calculate the critical path through the tasks"""
        # Simple implementation - return linear path for now
        return [
            task.task_id for task in sorted(tasks, key=lambda t: len(t.dependencies))
        ]

    def _add_task_to_dependency_graph(self, task: Task):
        """Add task to the dependency graph"""
        self.task_dependencies.add_node(task.task_id, task=task)

        for dep_id in task.dependencies:
            if dep_id in self.active_tasks:
                self.task_dependencies.add_edge(dep_id, task.task_id)

    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute a workflow"""
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow = self.active_workflows[workflow_id]
        workflow.overall_status = TaskStatus.IN_PROGRESS

        self.logger.info(f"Starting workflow execution: {workflow_id}")

        try:
            # Get execution order
            execution_plan = workflow.execution_plan
            execution_order = execution_plan["execution_order"]

            # Execute tasks in order
            for task_id in execution_order:
                if task_id in self.active_tasks:
                    task = self.active_tasks[task_id]

                    # Check dependencies
                    if await self._check_task_dependencies(task):
                        result = await self._execute_task(task)

                        if result["success"]:
                            workflow.completed_tasks += 1
                        else:
                            workflow.failed_tasks += 1

                        # Update workflow progress
                        await self._update_workflow_progress(workflow)

            # Complete workflow
            workflow.overall_status = (
                TaskStatus.COMPLETED
                if workflow.failed_tasks == 0
                else TaskStatus.FAILED
            )
            workflow.actual_completion = datetime.now()

            # Generate learning summary
            workflow.learning_summary = await self._generate_workflow_learning_summary(
                workflow
            )

            await self._store_workflow(workflow)

            self.logger.info(f"Workflow execution complete: {workflow_id}")

            return {
                "workflow_id": workflow_id,
                "status": workflow.overall_status.value,
                "completed_tasks": workflow.completed_tasks,
                "failed_tasks": workflow.failed_tasks,
                "total_duration": (
                    (
                        workflow.actual_completion
                        - workflow.progress_history[0]["timestamp"]
                    ).total_seconds()
                    / 3600
                    if workflow.progress_history
                    else 0
                ),
                "learning_summary": workflow.learning_summary,
            }

        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            workflow.overall_status = TaskStatus.FAILED
            await self._store_workflow(workflow)
            raise

    async def _check_task_dependencies(self, task: Task) -> bool:
        """Check if task dependencies are satisfied"""
        for dep_id in task.dependencies:
            if dep_id in self.active_tasks:
                dep_task = self.active_tasks[dep_id]
                if dep_task.status != TaskStatus.COMPLETED:
                    return False
        return True

    async def _execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a single task"""
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()

        self.logger.info(f"Executing task: {task.task_id} - {task.title}")

        try:
            # Task execution logic based on type
            if task.task_type == TaskType.ANALYSIS:
                result = await self._execute_analysis_task(task)
            elif task.task_type == TaskType.IMPLEMENTATION:
                result = await self._execute_implementation_task(task)
            elif task.task_type == TaskType.RESEARCH:
                result = await self._execute_research_task(task)
            elif task.task_type == TaskType.DECISION:
                result = await self._execute_decision_task(task)
            elif task.task_type == TaskType.VALIDATION:
                result = await self._execute_validation_task(task)
            else:
                result = await self._execute_generic_task(task)

            if result["success"]:
                task.status = TaskStatus.COMPLETED
                task.result = result
            else:
                task.status = TaskStatus.FAILED
                task.error_info = {"error": result.get(
                    "error", "Unknown error")}

            task.completed_at = datetime.now()
            await self._store_task(task)

            return result

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_info = {"error": str(
                e), "timestamp": datetime.now().isoformat()}
            task.completed_at = datetime.now()
            await self._store_task(task)

            self.logger.error(f"Task execution failed: {task.task_id} - {e}")
            return {"success": False, "error": str(e)}

    async def _execute_analysis_task(self, task: Task) -> Dict[str, Any]:
        """Execute an analysis task"""
        # Simulate analysis work
        await asyncio.sleep(1)  # Simulate processing time

        analysis_result = {
            "analysis_type": "automated",
            "findings": [
                f"Analysis completed for: {task.description}",
                "Key insights identified",
                "Recommendations generated",
            ],
            "confidence": 0.8,
            "recommendations": [
                "Proceed with implementation",
                "Monitor progress closely",
            ],
        }

        return {
            "success": True,
            "result_type": "analysis",
            "data": analysis_result,
            "task_id": task.task_id,
        }

    async def _execute_implementation_task(self, task: Task) -> Dict[str, Any]:
        """Execute an implementation task"""
        # Simulate implementation work
        await asyncio.sleep(2)  # Simulate processing time

        implementation_result = {
            "implementation_type": "automated",
            "actions_taken": [
                f"Implemented: {task.description}",
                "Configuration updated",
                "System integration completed",
            ],
            "success_metrics": {
                "completion_rate": 0.95,
                "performance_impact": 0.1,
                "error_rate": 0.02,
            },
        }

        return {
            "success": True,
            "result_type": "implementation",
            "data": implementation_result,
            "task_id": task.task_id,
        }

    async def _execute_research_task(self, task: Task) -> Dict[str, Any]:
        """Execute a research task"""
        # Simulate research work
        await asyncio.sleep(1.5)  # Simulate processing time

        research_result = {
            "research_type": "automated",
            "sources_consulted": [
                "Internal knowledge base",
                "System documentation",
                "Best practices database",
            ],
            "findings": [
                f"Research completed for: {task.description}",
                "Multiple approaches identified",
                "Best practices determined",
            ],
            "confidence": 0.85,
        }

        return {
            "success": True,
            "result_type": "research",
            "data": research_result,
            "task_id": task.task_id,
        }

    async def _execute_decision_task(self, task: Task) -> Dict[str, Any]:
        """Execute a decision task"""
        # Use board consultation for decision tasks
        board_result = await consult_board(
            f"Decision needed for task: {task.description}",
            {"task_context": task.context, "task_type": task.task_type.value},
            decision_type=BoardDecisionType.CONSULTATION,
            domain="task_execution",
        )

        decision_result = {
            "decision_type": "board_consultation",
            "recommendation": board_result["recommendation"],
            "confidence": board_result["confidence"],
            "consensus": board_result["consensus_reached"],
            "contributing_directors": board_result.get("contributing_directors", []),
        }

        return {
            "success": board_result["confidence"] > 0.5,
            "result_type": "decision",
            "data": decision_result,
            "task_id": task.task_id,
        }

    async def _execute_validation_task(self, task: Task) -> Dict[str, Any]:
        """Execute a validation task"""
        # Simulate validation work
        await asyncio.sleep(0.5)  # Simulate processing time

        validation_result = {
            "validation_type": "automated",
            "checks_performed": [
                "Functional validation",
                "Performance validation",
                "Integration validation",
            ],
            "results": {
                "functional_pass": True,
                "performance_acceptable": True,
                "integration_successful": True,
            },
            "overall_status": "passed",
        }

        return {
            "success": True,
            "result_type": "validation",
            "data": validation_result,
            "task_id": task.task_id,
        }

    async def _execute_generic_task(self, task: Task) -> Dict[str, Any]:
        """Execute a generic task"""
        # Simulate generic work
        await asyncio.sleep(1)  # Simulate processing time

        generic_result = {
            "task_type": task.task_type.value,
            "description": task.description,
            "execution_notes": [
                "Task executed successfully",
                "Standard procedures followed",
            ],
            "completion_status": "success",
        }

        return {
            "success": True,
            "result_type": "generic",
            "data": generic_result,
            "task_id": task.task_id,
        }

    async def _update_workflow_progress(self, workflow: TaskWorkflow):
        """Update workflow progress tracking"""
        progress_entry = {
            "timestamp": datetime.now(),
            "completed_tasks": workflow.completed_tasks,
            "failed_tasks": workflow.failed_tasks,
            "progress_percentage": workflow.completed_tasks
            / workflow.total_tasks
            * 100,
            "estimated_remaining": workflow.estimated_completion - datetime.now(),
        }

        workflow.progress_history.append(progress_entry)
        await self._log_execution_event(
            workflow.workflow_id, None, "progress_update", {
                "progress": progress_entry}
        )

    async def _generate_workflow_learning_summary(
        self, workflow: TaskWorkflow
    ) -> Dict[str, Any]:
        """Generate learning summary from completed workflow"""
        summary = {
            "workflow_id": workflow.workflow_id,
            "completion_rate": workflow.completed_tasks / workflow.total_tasks,
            "time_efficiency": 1.0,  # Calculate based on estimated vs actual
            "strategy_effectiveness": workflow.decomposition_strategy,
            "key_insights": [],
            "improvement_opportunities": [],
            "pattern_candidates": [],
        }

        # Calculate time efficiency
        if workflow.actual_completion and workflow.progress_history:
            actual_duration = (
                workflow.actual_completion -
                workflow.progress_history[0]["timestamp"]
            ).total_seconds() / 3600
            estimated_duration = sum(
                task.estimated_duration
                for task in self.active_tasks.values()
                if task.parent_request_id == workflow.workflow_id
            )
            if estimated_duration > 0:
                summary["time_efficiency"] = estimated_duration / \
                    actual_duration

        # Generate insights
        if summary["completion_rate"] > 0.9:
            summary["key_insights"].append(
                "High completion rate indicates good task decomposition"
            )
        if summary["time_efficiency"] > 1.1:
            summary["key_insights"].append(
                "Tasks completed faster than estimated")
        if workflow.failed_tasks == 0:
            summary["key_insights"].append(
                "Zero failures indicates robust planning")

        # Generate improvement opportunities
        if summary["completion_rate"] < 0.8:
            summary["improvement_opportunities"].append(
                "Review task decomposition strategy"
            )
        if summary["time_efficiency"] < 0.8:
            summary["improvement_opportunities"].append(
                "Improve time estimation accuracy"
            )

        # Check if this should become a pattern
        if summary["completion_rate"] > 0.8 and summary["time_efficiency"] > 0.8:
            summary["pattern_candidates"].append(
                {
                    "pattern_type": workflow.decomposition_strategy,
                    "success_metrics": summary,
                    "should_create_pattern": True,
                }
            )

        return summary

    # Storage and retrieval methods

    async def _store_workflow(self, workflow: TaskWorkflow):
        """Store workflow in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO workflows
                (workflow_id, original_request, user_context, total_tasks, completed_tasks,
                 failed_tasks, overall_status, estimated_completion, actual_completion,
                 decomposition_strategy, execution_plan, resource_allocation, progress_history,
                 final_result, learning_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    workflow.workflow_id,
                    workflow.original_request,
                    json.dumps(workflow.user_context),
                    workflow.total_tasks,
                    workflow.completed_tasks,
                    workflow.failed_tasks,
                    workflow.overall_status.value,
                    workflow.estimated_completion.isoformat(),
                    (
                        workflow.actual_completion.isoformat()
                        if workflow.actual_completion
                        else None
                    ),
                    workflow.decomposition_strategy,
                    json.dumps(workflow.execution_plan),
                    json.dumps(workflow.resource_allocation),
                    json.dumps(workflow.progress_history, default=str),
                    (
                        json.dumps(workflow.final_result)
                        if workflow.final_result
                        else None
                    ),
                    json.dumps(workflow.learning_summary),
                ),
            )
            conn.commit()

    async def _store_task(self, task: Task):
        """Store task in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tasks
                (task_id, parent_request_id, title, description, task_type, priority, status,
                 complexity_score, estimated_duration, dependencies, blockers, required_resources,
                 assigned_tools, context, created_at, started_at, completed_at, result,
                 error_info, progress_notes, learning_insights)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    task.task_id,
                    task.parent_request_id,
                    task.title,
                    task.description,
                    task.task_type.value,
                    task.priority.value,
                    task.status.value,
                    task.complexity_score,
                    task.estimated_duration,
                    json.dumps(task.dependencies),
                    json.dumps(task.blockers),
                    json.dumps(task.required_resources),
                    json.dumps(task.assigned_tools),
                    json.dumps(task.context),
                    task.created_at.isoformat(),
                    task.started_at.isoformat() if task.started_at else None,
                    task.completed_at.isoformat() if task.completed_at else None,
                    json.dumps(task.result) if task.result else None,
                    json.dumps(task.error_info) if task.error_info else None,
                    json.dumps(task.progress_notes, default=str),
                    json.dumps(task.learning_insights),
                ),
            )
            conn.commit()

    async def _store_decomposition_pattern(self, pattern: DecompositionPattern):
        """Store decomposition pattern in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO decomposition_patterns
                (pattern_id, request_type, context_indicators, typical_breakdown,
                 success_rate, average_completion_time, common_bottlenecks,
                 optimization_suggestions, usage_count, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    pattern.pattern_id,
                    pattern.request_type,
                    json.dumps(pattern.context_indicators),
                    json.dumps(pattern.typical_breakdown),
                    pattern.success_rate,
                    pattern.average_completion_time,
                    json.dumps(pattern.common_bottlenecks),
                    json.dumps(pattern.optimization_suggestions),
                    pattern.usage_count,
                    pattern.last_updated.isoformat(),
                ),
            )
            conn.commit()

    async def _log_execution_event(
        self,
        workflow_id: str,
        task_id: Optional[str],
        event_type: str,
        details: Dict[str, Any],
    ):
        """Log execution event"""
        log_id = (
            f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO execution_logs
                (log_id, workflow_id, task_id, event_type, timestamp, details, performance_metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    log_id,
                    workflow_id,
                    task_id,
                    event_type,
                    datetime.now().isoformat(),
                    json.dumps(details),
                    json.dumps({}),  # Performance metrics placeholder
                ),
            )
            conn.commit()

    # Public API methods

    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow status"""
        if workflow_id not in self.active_workflows:
            return {"error": f"Workflow {workflow_id} not found"}

        workflow = self.active_workflows[workflow_id]

        return {
            "workflow_id": workflow_id,
            "status": workflow.overall_status.value,
            "progress": {
                "total_tasks": workflow.total_tasks,
                "completed_tasks": workflow.completed_tasks,
                "failed_tasks": workflow.failed_tasks,
                "progress_percentage": workflow.completed_tasks
                / workflow.total_tasks
                * 100,
            },
            "timing": {
                "estimated_completion": workflow.estimated_completion.isoformat(),
                "actual_completion": (
                    workflow.actual_completion.isoformat()
                    if workflow.actual_completion
                    else None
                ),
            },
            "strategy": workflow.decomposition_strategy,
            "recent_progress": (
                workflow.progress_history[-3:] if workflow.progress_history else []
            ),
        }

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status"""
        if task_id not in self.active_tasks:
            return {"error": f"Task {task_id} not found"}

        task = self.active_tasks[task_id]

        return {
            "task_id": task_id,
            "title": task.title,
            "status": task.status.value,
            "type": task.task_type.value,
            "priority": task.priority.value,
            "complexity": task.complexity_score,
            "progress": {
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": (
                    task.completed_at.isoformat() if task.completed_at else None
                ),
                "estimated_duration": task.estimated_duration,
            },
            "dependencies": task.dependencies,
            "result": task.result,
            "error_info": task.error_info,
        }

    async def get_decomposer_status(self) -> Dict[str, Any]:
        """Get overall decomposer status"""
        return {
            "active_workflows": len(self.active_workflows),
            "active_tasks": len(self.active_tasks),
            "learned_patterns": len(self.decomposition_patterns),
            "system_health": "operational",
        }


# Global instance
_task_decomposer = None


def get_task_decomposer() -> EchoTaskDecomposer:
    """Get global task decomposer instance"""
    global _task_decomposer
    if _task_decomposer is None:
        _task_decomposer = EchoTaskDecomposer()
    return _task_decomposer


# Convenience functions
async def decompose_request(
    request: str,
    user_context: Dict[str, Any],
    priority: TaskPriority = TaskPriority.MEDIUM,
    deadline: Optional[datetime] = None,
) -> str:
    """Convenience function to decompose a request"""
    decomposer = get_task_decomposer()
    return await decomposer.decompose_request(request, user_context, priority, deadline)


async def execute_workflow(workflow_id: str) -> Dict[str, Any]:
    """Convenience function to execute a workflow"""
    decomposer = get_task_decomposer()
    return await decomposer.execute_workflow(workflow_id)


async def get_workflow_status(workflow_id: str) -> Dict[str, Any]:
    """Convenience function to get workflow status"""
    decomposer = get_task_decomposer()
    return await decomposer.get_workflow_status(workflow_id)


if __name__ == "__main__":
    # Example usage
    async def example():
        decomposer = get_task_decomposer()

        # Decompose a complex request
        workflow_id = await decomposer.decompose_request(
            "Create a comprehensive video generation system with AI-powered content creation",
            {
                "complexity_indicators": ["AI", "video", "content creation", "system"],
                "priority": "high",
                "resource_constraints": False,
                "deadline": None,
            },
            priority=TaskPriority.HIGH,
        )

        print(f"Workflow created: {workflow_id}")

        # Check workflow status
        status = await decomposer.get_workflow_status(workflow_id)
        print("Workflow Status:", json.dumps(status, indent=2))

        # Execute the workflow
        result = await decomposer.execute_workflow(workflow_id)
        print("Execution Result:", json.dumps(result, indent=2))

    asyncio.run(example())
