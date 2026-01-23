"""
Autonomous Core Orchestrator for Echo Brain

The AutonomousCore class serves as the main orchestrator for all autonomous
operations, managing the lifecycle and coordination of all autonomous components.
"""

import asyncio
import logging
import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import asyncpg
from contextlib import asynccontextmanager
import signal
import time

# Import autonomous components
from .goals import GoalManager
from .scheduler import Scheduler, ScheduleConfig
from .executor import Executor, TaskResult
from .celery_executor import CeleryExecutor
from .events import EventWatcher
from .safety import SafetyController
from .audit import AuditLogger

logger = logging.getLogger(__name__)


class AutonomousState(Enum):
    """States of the autonomous core system"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class SystemStatus:
    """Current status of the autonomous system"""
    state: AutonomousState
    uptime_seconds: float
    cycles_completed: int
    tasks_executed: int
    last_cycle_time: Optional[datetime]
    last_error: Optional[str]
    components_status: Dict[str, bool]


class AutonomousCore:
    """
    Main orchestrator for Echo Brain's autonomous operations.

    Coordinates all autonomous components including goals, scheduling, execution,
    events, safety controls, and audit logging. Provides the main run loop and
    system lifecycle management.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the AutonomousCore with configuration."""
        self.config = config or {}

        # Database configuration - MUST be set before creating executors
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'echo_brain',
            'user': 'patrick',
            'password': os.environ.get('ECHO_BRAIN_DB_PASSWORD', 'RP78eIrW7cI2jYvL5akt1yurE')
        }

        # System state
        self.state = AutonomousState.STOPPED
        self.start_time = None
        self.cycles_completed = 0
        self.tasks_executed = 0
        self.last_cycle_time = None
        self.last_error = None

        # Core components
        self.goal_manager = GoalManager()
        self.scheduler = Scheduler(ScheduleConfig(
            max_concurrent_tasks=self.config.get('max_concurrent_tasks', 3),
            max_tasks_per_minute=self.config.get('max_tasks_per_minute', 10),
            max_tasks_per_hour=self.config.get('max_tasks_per_hour', 100)
        ))
        # Use Celery executor for distributed execution
        try:
            self.executor = CeleryExecutor(self.db_config)
            logger.info("Using CeleryExecutor for distributed task execution")
        except Exception as e:
            logger.warning(f"Failed to initialize CeleryExecutor: {e}, falling back to regular Executor")
            self.executor = Executor()
        self.event_watcher = EventWatcher()
        self.safety_controller = SafetyController()
        self.audit_logger = AuditLogger()

        # Control loop
        self.main_task = None
        self.cycle_interval = self.config.get('cycle_interval', 30)  # seconds
        self.running = False

        # Kill switch integration
        self.kill_switch_file = self.config.get('kill_switch_file', '/tmp/echo_brain_kill_switch')
        self.check_kill_switch = self.config.get('check_kill_switch', True)

        # Database pool (db_config already set above)
        self._pool = None

        # Performance tracking
        self.cycle_times = []
        self.max_cycle_time_history = 100

        logger.info(f"AutonomousCore initialized with cycle interval: {self.cycle_interval}s")

    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(**self.db_config, min_size=1, max_size=5)

        async with self._pool.acquire() as connection:
            yield connection

    async def start(self):
        """Start the autonomous core system."""
        if self.state != AutonomousState.STOPPED:
            logger.warning(f"Cannot start - current state is {self.state}")
            return False

        logger.info("Starting AutonomousCore...")
        self.state = AutonomousState.STARTING

        try:
            # Initialize database tables if needed
            await self.initialize_database()

            # Start all components
            await self.start_components()

            # Recover interrupted tasks and goals
            recovered = await self.recover_interrupted_state()
            if recovered['goals'] > 0 or recovered['tasks'] > 0:
                logger.info(f"Recovered {recovered['goals']} active goals and {recovered['tasks']} interrupted tasks")
                await self.audit_logger.log(
                    "state_recovered",
                    f"Recovered {recovered['goals']} goals, {recovered['tasks']} tasks",
                    details=recovered
                )

            # Setup signal handlers for graceful shutdown
            self.setup_signal_handlers()

            # Start the main control loop
            self.running = True
            self.start_time = datetime.now()
            self.main_task = asyncio.create_task(self.run_main_loop())

            self.state = AutonomousState.RUNNING
            await self.audit_logger.log("system_started", "Autonomous core started successfully")

            logger.info("AutonomousCore started successfully")
            return True

        except Exception as e:
            self.state = AutonomousState.ERROR
            self.last_error = str(e)
            logger.error(f"Failed to start AutonomousCore: {e}")
            await self.stop()
            return False

    async def stop(self):
        """Stop the autonomous core system."""
        if self.state == AutonomousState.STOPPED:
            return

        logger.info("Stopping AutonomousCore...")
        self.state = AutonomousState.STOPPING
        self.running = False

        try:
            # Save current state before stopping
            saved = await self.save_current_state()
            if saved['tasks'] > 0:
                logger.info(f"Saved state: {saved['tasks']} running tasks marked as interrupted")
                await self.audit_logger.log(
                    "state_saved",
                    f"Saved {saved['tasks']} interrupted tasks for recovery",
                    details=saved
                )

            # Cancel main loop
            if self.main_task:
                self.main_task.cancel()
                try:
                    await self.main_task
                except asyncio.CancelledError:
                    pass

            # Stop all components
            await self.stop_components()

            self.state = AutonomousState.STOPPED
            await self.audit_logger.log("system_stopped", "Autonomous core stopped")

            logger.info("AutonomousCore stopped successfully")

        except Exception as e:
            self.state = AutonomousState.ERROR
            self.last_error = str(e)
            logger.error(f"Error during shutdown: {e}")

    async def pause(self):
        """Pause autonomous operations."""
        if self.state != AutonomousState.RUNNING:
            logger.warning(f"Cannot pause - current state is {self.state}")
            return False

        logger.info("Pausing AutonomousCore...")
        self.state = AutonomousState.PAUSED
        await self.audit_logger.log("system_paused", "Autonomous core paused")

        return True

    async def resume(self):
        """Resume autonomous operations."""
        if self.state != AutonomousState.PAUSED:
            logger.warning(f"Cannot resume - current state is {self.state}")
            return False

        logger.info("Resuming AutonomousCore...")
        self.state = AutonomousState.RUNNING
        await self.audit_logger.log("system_resumed", "Autonomous core resumed")

        return True

    async def run_main_loop(self):
        """Main control loop for autonomous operations."""
        logger.info("Starting main autonomous loop")

        while self.running:
            try:
                # Check kill switch
                if self.check_kill_switch and await self.is_kill_switch_active():
                    logger.warning("Kill switch activated - stopping autonomous operations")
                    await self.stop()
                    break

                # Only run cycle if not paused
                if self.state == AutonomousState.RUNNING:
                    cycle_start = time.time()
                    await self.run_cycle()
                    cycle_time = time.time() - cycle_start

                    # Track performance
                    self.cycle_times.append(cycle_time)
                    if len(self.cycle_times) > self.max_cycle_time_history:
                        self.cycle_times.pop(0)

                    self.cycles_completed += 1
                    self.last_cycle_time = datetime.now()

                    if cycle_time > 10.0:  # Log slow cycles
                        logger.warning(f"Slow cycle detected: {cycle_time:.2f}s")

                # Wait for next cycle
                await asyncio.sleep(self.cycle_interval)

            except asyncio.CancelledError:
                logger.info("Main loop cancelled")
                break
            except Exception as e:
                self.last_error = str(e)
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)  # Brief pause on error

        logger.info("Main autonomous loop stopped")

    async def run_cycle(self) -> Dict[str, Any]:
        """
        Execute one cycle of autonomous operations.

        Returns:
            Dictionary with cycle execution details
        """
        cycle_start = datetime.now()
        cycle_results = {
            'timestamp': cycle_start,
            'tasks_processed': 0,
            'events_processed': 0,
            'goals_updated': 0,
            'errors': []
        }

        try:
            # 1. Process pending events
            await self.process_events(cycle_results)

            # 2. Update goal progress
            await self.update_goals(cycle_results)

            # 3. Execute scheduled tasks
            await self.execute_tasks(cycle_results)

            # 4. Perform maintenance operations
            await self.perform_maintenance(cycle_results)

            # Log successful cycle
            if cycle_results['tasks_processed'] > 0 or cycle_results['events_processed'] > 0:
                await self.audit_logger.log(
                    "cycle_completed",
                    f"Processed {cycle_results['tasks_processed']} tasks, "
                    f"{cycle_results['events_processed']} events",
                    details=cycle_results
                )

        except Exception as e:
            cycle_results['errors'].append(str(e))
            logger.error(f"Error in run_cycle: {e}")

        return cycle_results

    async def process_events(self, cycle_results: Dict[str, Any]):
        """Process pending events from the event watcher."""
        try:
            events = await self.event_watcher.check_events()

            for event in events:
                # Let the event watcher handle the event
                # This will create tasks as needed
                cycle_results['events_processed'] += 1

                # Audit the event processing
                await self.audit_logger.log(
                    "event_processed",
                    f"Processed event: {event.event_type}",
                    details={
                        'trigger_id': event.trigger_id,
                        'event_type': event.event_type,
                        'source': event.source
                    }
                )

        except Exception as e:
            cycle_results['errors'].append(f"Event processing: {e}")
            logger.error(f"Failed to process events: {e}")

    async def update_goals(self, cycle_results: Dict[str, Any]):
        """Update goal progress and create new tasks as needed."""
        try:
            # Get active goals
            active_goals = await self.goal_manager.get_goals(status='active')

            for goal in active_goals:
                # Update goal progress based on completed tasks
                await self.goal_manager.update_goal_progress(goal['id'])
                cycle_results['goals_updated'] += 1

                # Check if goal needs more tasks (autonomous task generation)
                await self._generate_tasks_for_goal(goal)

                # Check if goal is completed
                if goal['progress_percent'] >= 100.0:
                    await self.goal_manager.update_goal_status(goal['id'], 'completed')
                    await self.audit_logger.log(
                        "goal_completed",
                        f"Goal completed: {goal['name']}",
                        goal_id=goal['id']
                    )

        except Exception as e:
            cycle_results['errors'].append(f"Goal update: {e}")
            logger.error(f"Failed to update goals: {e}")

    async def execute_tasks(self, cycle_results: Dict[str, Any]):
        """Execute scheduled tasks."""
        try:
            # Get next task to execute
            while True:
                task = await self.scheduler.get_next_task()
                if not task:
                    break

                # Check safety requirements
                # Convert ScheduledTask to dict for safety evaluation
                if hasattr(task, 'id'):  # ScheduledTask object
                    task_dict = {
                        "id": task.id,
                        "name": task.name,
                        "task_type": task.task_type,
                        "goal_id": task.goal_id,
                        "priority": task.priority,
                        "safety_level": task.safety_level,
                        "metadata": task.metadata,
                        "status": task.status if hasattr(task, 'status') else 'pending'
                    }
                else:
                    task_dict = task  # Already a dict

                # Skip safety evaluation for already approved tasks
                if task_dict.get('status') == 'approved':
                    is_safe = True
                    safety_level = task_dict.get('safety_level', 'auto')
                    approval_data = None
                    logger.info(f"Task {task_dict['id']} is pre-approved, skipping safety evaluation")
                else:
                    is_safe, safety_level, approval_data = await self.safety_controller.evaluate_task_safety(task_dict)

                if not is_safe:
                    # Debug logging to identify the type error
                    logger.debug(f"Approval data type: {type(approval_data)}, value: {approval_data}")

                    # Safe handling of approval_data
                    if approval_data and isinstance(approval_data, dict):
                        reason = approval_data.get('reason', 'Safety check failed')
                    else:
                        reason = str(approval_data) if approval_data else 'Safety check failed'

                    # Safe handling of task_id extraction
                    if isinstance(task_dict, dict):
                        task_id = task_dict.get('id', 'unknown')
                    else:
                        task_id = getattr(task, 'id', 'unknown')

                    logger.info(f"Task {task_id} blocked by safety controller: {reason}")
                    await self.audit_logger.log(
                        "task_blocked",
                        f"Task {task_id} blocked: {reason}",
                        task_id=task_id,
                        safety_level=safety_level
                    )
                    continue

                # Execute the task
                self.scheduler.record_task_execution()
                result = await self.executor.execute(task.id)

                # Update statistics
                cycle_results['tasks_processed'] += 1
                self.tasks_executed += 1

                # Handle result
                if result.success:
                    await self.audit_logger.log(
                        "task_executed",
                        f"Task {task.id} executed successfully",
                        task_id=task.id,
                        details=result.metadata
                    )

                    # Handle recurring tasks
                    if task.metadata and task.metadata.get('recurring'):
                        await self.scheduler.handle_recurring_task_completion(task.id)

                else:
                    await self.audit_logger.log(
                        "task_failed",
                        f"Task {task.id} failed: {result.error}",
                        task_id=task.id,
                        outcome="failure",
                        details={'error': result.error}
                    )

                # Respect rate limits - only process one task per cycle for now
                break

        except Exception as e:
            cycle_results['errors'].append(f"Task execution: {e}")
            logger.error(f"Failed to execute tasks: {e}")

    async def perform_maintenance(self, cycle_results: Dict[str, Any]):
        """Perform periodic maintenance operations."""
        try:
            # Cleanup old audit logs (keep last 30 days)
            if self.cycles_completed % 100 == 0:  # Every 100 cycles
                await self.audit_logger.cleanup_old_logs(30)

            # Update system health metrics
            await self.update_health_metrics()

        except Exception as e:
            cycle_results['errors'].append(f"Maintenance: {e}")
            logger.error(f"Failed to perform maintenance: {e}")

    async def initialize_database(self):
        """Initialize database tables for autonomous operations."""
        try:
            async with self.get_connection() as conn:
                # Check if tables already exist
                table_check = await conn.fetchval("""
                    SELECT COUNT(*) FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'autonomous_goals'
                """)

                if table_check > 0:
                    logger.info("Autonomous tables already exist, skipping schema initialization")
                    return

                # Read and execute schema
                schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')

                if os.path.exists(schema_path):
                    with open(schema_path, 'r') as f:
                        schema_sql = f.read()

                    await conn.execute(schema_sql)
                    logger.info("Database schema initialized")
                else:
                    logger.warning("Schema file not found - database may not be properly initialized")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def start_components(self):
        """Start all autonomous components."""
        try:
            # Start event watcher
            await self.event_watcher.start_watching()
            logger.info("Event watcher started")

            # Initialize executor agents
            await self.executor.initialize_agents()
            logger.info("Executor agents initialized")

            logger.info("All components started successfully")

        except Exception as e:
            logger.error(f"Failed to start components: {e}")
            raise

    async def stop_components(self):
        """Stop all autonomous components."""
        try:
            # Stop event watcher
            await self.event_watcher.stop_watching()

            # Cleanup components
            await asyncio.gather(
                self.executor.cleanup(),
                self.scheduler.cleanup(),
                self.event_watcher.cleanup(),
                self.safety_controller.cleanup(),
                self.audit_logger.cleanup(),
                return_exceptions=True
            )

            # Close database pool
            if self._pool:
                await self._pool.close()

            logger.info("All components stopped")

        except Exception as e:
            logger.error(f"Error stopping components: {e}")

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self.signal_handler)
        if hasattr(signal, 'SIGINT'):
            signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        if self.running:
            asyncio.create_task(self.stop())

    async def is_kill_switch_active(self) -> bool:
        """Check if the kill switch file exists."""
        return os.path.exists(self.kill_switch_file)

    async def get_status(self) -> SystemStatus:
        """Get current system status."""
        uptime = 0.0
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()

        # Check component status
        components_status = {
            'goal_manager': True,  # Always available
            'scheduler': True,     # Always available
            'executor': True,      # Always available
            'event_watcher': self.event_watcher.running,
            'safety_controller': True,  # Always available
            'audit_logger': True        # Always available
        }

        return SystemStatus(
            state=self.state,
            uptime_seconds=uptime,
            cycles_completed=self.cycles_completed,
            tasks_executed=self.tasks_executed,
            last_cycle_time=self.last_cycle_time,
            last_error=self.last_error,
            components_status=components_status
        )

    async def update_health_metrics(self):
        """Update system health metrics."""
        try:
            status = await self.get_status()

            # Calculate average cycle time
            avg_cycle_time = sum(self.cycle_times) / len(self.cycle_times) if self.cycle_times else 0

            # Store health metrics
            async with self.get_connection() as conn:
                # Prepare health data with proper serialization
                health_data = {
                    'uptime_seconds': status.uptime_seconds,
                    'cycles_completed': status.cycles_completed,
                    'tasks_executed': status.tasks_executed,
                    'avg_cycle_time': avg_cycle_time,
                    'components_status': status.components_status,
                    'state': status.state.value,
                    'timestamp': datetime.now().isoformat()
                }

                await conn.execute("""
                    INSERT INTO autonomous_audit_log (event_type, action, details)
                    VALUES ('health_metrics', 'System health update', $1)
                """, json.dumps(health_data, default=str))

        except Exception as e:
            logger.error(f"Failed to update health metrics: {e}")

    async def create_goal(self, name: str, description: str, goal_type: str,
                         priority: int = 5, metadata: Optional[Dict] = None) -> Optional[int]:
        """Create a new autonomous goal."""
        try:
            goal_id = await self.goal_manager.create_goal(
                name=name,
                description=description,
                goal_type=goal_type,
                priority=priority,
                metadata=metadata
            )

            if goal_id:
                await self.audit_logger.log(
                    "goal_created",
                    f"New goal created: {name}",
                    goal_id=goal_id,
                    details={'goal_type': goal_type, 'priority': priority}
                )

            return goal_id

        except Exception as e:
            logger.error(f"Failed to create goal: {e}")
            return None

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        if not self.cycle_times:
            return {}

        return {
            'avg_cycle_time': sum(self.cycle_times) / len(self.cycle_times),
            'min_cycle_time': min(self.cycle_times),
            'max_cycle_time': max(self.cycle_times),
            'total_cycles': self.cycles_completed,
            'total_tasks': self.tasks_executed,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        }

    async def save_current_state(self) -> Dict[str, int]:
        """
        Save current state before shutdown.
        Marks running tasks as interrupted for later recovery.

        Returns:
            Dictionary with counts of saved items
        """
        saved = {'tasks': 0, 'goals': 0}

        try:
            async with self.get_connection() as conn:
                # Mark all in_progress tasks as interrupted
                result = await conn.execute("""
                    UPDATE autonomous_tasks
                    SET status = 'interrupted'
                    WHERE status = 'in_progress'
                """)

                # Extract count from result
                saved['tasks'] = int(result.split()[-1]) if result else 0

                # Mark paused goals for recovery
                result = await conn.execute("""
                    UPDATE autonomous_goals
                    SET metadata = jsonb_set(
                        COALESCE(metadata, '{}'),
                        '{needs_recovery}',
                        'true'
                    )
                    WHERE status = 'active'
                """)

                saved['goals'] = int(result.split()[-1]) if result else 0

                logger.info(f"State saved: {saved['tasks']} tasks, {saved['goals']} goals")

        except Exception as e:
            logger.error(f"Failed to save current state: {e}")

        return saved

    async def recover_interrupted_state(self) -> Dict[str, int]:
        """
        Recover interrupted tasks and goals after restart.

        Returns:
            Dictionary with counts of recovered items
        """
        recovered = {'tasks': 0, 'goals': 0}

        try:
            async with self.get_connection() as conn:
                # Get interrupted tasks
                interrupted_tasks = await conn.fetch("""
                    SELECT id, name, task_type, goal_id
                    FROM autonomous_tasks
                    WHERE status = 'interrupted'
                    ORDER BY priority ASC, created_at ASC
                """)

                # Reset interrupted tasks to pending for re-execution
                for task in interrupted_tasks:
                    await conn.execute("""
                        UPDATE autonomous_tasks
                        SET status = 'pending',
                            started_at = NULL,
                            result = NULL,
                            error = 'Task interrupted by system shutdown - will retry'
                        WHERE id = $1
                    """, task['id'])

                    recovered['tasks'] += 1

                    # Log recovery
                    await self.audit_logger.log(
                        "task_recovered",
                        f"Recovered interrupted task: {task['name']}",
                        task_id=task['id'],
                        details={'task_type': task['task_type'], 'goal_id': task['goal_id']}
                    )

                # Recover goals that need recovery
                goals_needing_recovery = await conn.fetch("""
                    SELECT id, name, goal_type
                    FROM autonomous_goals
                    WHERE status = 'active'
                    AND metadata->>'needs_recovery' = 'true'
                """)

                for goal in goals_needing_recovery:
                    # Remove recovery flag
                    await conn.execute("""
                        UPDATE autonomous_goals
                        SET metadata = metadata - 'needs_recovery'
                        WHERE id = $1
                    """, goal['id'])

                    recovered['goals'] += 1

                    # Log recovery
                    await self.audit_logger.log(
                        "goal_recovered",
                        f"Recovered active goal: {goal['name']}",
                        goal_id=goal['id'],
                        details={'goal_type': goal['goal_type']}
                    )

                if recovered['tasks'] > 0 or recovered['goals'] > 0:
                    logger.info(f"State recovery complete: {recovered['tasks']} tasks, {recovered['goals']} goals")

        except Exception as e:
            logger.error(f"Failed to recover interrupted state: {e}")

        return recovered

    async def _generate_tasks_for_goal(self, goal: Dict[str, Any]):
        """
        Generate tasks automatically for a goal that needs more tasks.

        Args:
            goal: Goal dictionary from database
        """
        try:
            # Check how many tasks this goal already has
            async with self.get_connection() as conn:
                task_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM autonomous_tasks
                    WHERE goal_id = $1
                """, goal['id'])

            goal_type = goal['goal_type']
            goal_id = goal['id']

            # Generate tasks based on goal type and current task count
            new_tasks = []

            if goal_type == 'testing' and task_count < 2:
                new_tasks = [
                    {
                        "name": "Run Self-Test Suite",
                        "task_type": "testing",
                        "description": "Execute comprehensive self-test validation",
                        "safety_level": "auto",
                        "priority": 3
                    },
                    {
                        "name": "Validate API Endpoints",
                        "task_type": "testing",
                        "description": "Test all autonomous API endpoints for functionality",
                        "safety_level": "auto",
                        "priority": 4
                    }
                ]
            elif goal_type == 'analysis' and task_count < 3:
                new_tasks = [
                    {
                        "name": "Analyze System Performance",
                        "task_type": "analysis",
                        "description": "Review system metrics and identify optimization opportunities",
                        "safety_level": "auto",
                        "priority": 5
                    },
                    {
                        "name": "Review Service Health Trends",
                        "task_type": "monitoring",
                        "description": "Analyze historical service health data for patterns",
                        "safety_level": "auto",
                        "priority": 6
                    }
                ]
            elif goal_type == 'research' and task_count < 2:
                new_tasks = [
                    {
                        "name": "Analyze User Interaction Patterns",
                        "task_type": "analysis",
                        "description": "Study user interaction data to identify learning opportunities",
                        "safety_level": "notify",
                        "priority": 7
                    }
                ]
            elif goal_type == 'coding' and task_count < 2:
                new_tasks = [
                    {
                        "name": "Review Code Quality Metrics",
                        "task_type": "analysis",
                        "description": "Analyze codebase for potential improvements and optimizations",
                        "safety_level": "notify",
                        "priority": 8
                    }
                ]

            # Create the tasks in database
            if new_tasks:
                async with self.get_connection() as conn:
                    for task in new_tasks:
                        task_id = await conn.fetchval("""
                            INSERT INTO autonomous_tasks
                            (goal_id, name, task_type, safety_level, priority, metadata)
                            VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                            RETURNING id
                        """,
                        goal_id,
                        task["name"],
                        task["task_type"],
                        task["safety_level"],
                        task["priority"],
                        json.dumps({"auto_generated": True, "description": task["description"]})
                        )

                        logger.info(f"Auto-generated task '{task['name']}' for goal '{goal['name']}' (ID: {task_id})")

                        # Audit the task creation
                        await self.audit_logger.log(
                            "task_created",
                            f"Auto-generated task: {task['name']}",
                            goal_id=goal_id,
                            task_id=task_id,
                            details={"auto_generated": True, "goal_type": goal_type}
                        )

        except Exception as e:
            logger.error(f"Failed to generate tasks for goal {goal['name']}: {e}")