"""
Procedure Library for Echo Brain
Stores and executes procedures - sequences of steps to accomplish tasks.
"""

import asyncio
import asyncpg
import json
import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import importlib

from .schemas import (
    Procedure, ProcedureExecution, Step, ActionPlan,
    ActionType, SafetyLevel, ProcedureCategory
)

logger = logging.getLogger(__name__)


class ProcedureLibrary:
    """
    Stores and executes procedures - sequences of steps to accomplish tasks.
    Procedures can be:
    - Diagnostic (figure out what's wrong)
    - Remediation (fix a problem)
    - Operational (deploy, restart, configure)
    """

    def __init__(self, db_config: Dict[str, str] = None):
        self.db_config = db_config or {
            "host": "localhost",
            "database": "echo_brain",
            "user": "patrick",
            "password": "RP78eIrW7cI2jYvL5akt1yurE"
        }
        self._pool = None
        self._executor = None
        self._system_model = None

    async def get_db_pool(self):
        """Get or create database connection pool"""
        if not self._pool:
            self._pool = await asyncpg.create_pool(
                **self.db_config,
                min_size=2,
                max_size=10,
                timeout=10
            )
        return self._pool

    async def close(self):
        """Clean up connections"""
        if self._pool:
            await self._pool.close()

    def _get_executor(self):
        """Get ActionExecutor instance"""
        if not self._executor:
            from .executor import get_action_executor
            self._executor = get_action_executor()
        return self._executor

    def _get_system_model(self):
        """Get SystemModel instance"""
        if not self._system_model:
            from .system_model import get_system_model
            self._system_model = get_system_model()
        return self._system_model

    async def initialize_procedures(self):
        """Initialize the procedure library with built-in procedures"""
        procedures = [
            await self._create_diagnose_service_failure_procedure(),
            await self._create_add_ollama_model_procedure(),
            await self._create_restart_service_procedure(),
            await self._create_analyze_error_logs_procedure(),
            await self._create_check_system_health_procedure(),
            await self._create_update_from_git_procedure(),
        ]

        pool = await self.get_db_pool()

        async with pool.acquire() as conn:
            for procedure in procedures:
                # Check if procedure exists
                existing = await conn.fetchrow(
                    "SELECT id FROM procedures WHERE name = $1", procedure.name
                )

                if not existing:
                    await conn.execute(
                        """INSERT INTO procedures
                           (name, description, trigger_patterns, category, steps)
                           VALUES ($1, $2, $3, $4, $5)""",
                        procedure.name,
                        procedure.description,
                        procedure.trigger_patterns,
                        procedure.category.value,
                        json.dumps([step.dict() for step in procedure.steps])
                    )
                    logger.info(f"Added procedure: {procedure.name}")

    async def _create_diagnose_service_failure_procedure(self) -> Procedure:
        """Create diagnostic procedure for service failures"""
        return Procedure(
            name="diagnose_service_failure",
            description="Systematically diagnose why a service is not working",
            trigger_patterns=[
                r".*service.*not.*working",
                r".*won't.*start",
                r".*failed.*start",
                r".*service.*down",
                r".*not.*responding"
            ],
            category=ProcedureCategory.DIAGNOSTIC,
            steps=[
                Step(
                    action=ActionType.SHELL,
                    target="systemd",
                    command="systemctl status {service_name}",
                    description="Check systemd service status",
                    safety_level=SafetyLevel.SAFE
                ),
                Step(
                    action=ActionType.SHELL,
                    target="systemd",
                    command="journalctl -u {service_name} --since '1 hour ago' -n 50",
                    description="Get recent service logs",
                    safety_level=SafetyLevel.SAFE
                ),
                Step(
                    action=ActionType.SHELL,
                    target="system",
                    command="ps aux | grep {service_name}",
                    description="Check if process is running",
                    safety_level=SafetyLevel.SAFE
                ),
                Step(
                    action=ActionType.SHELL,
                    target="network",
                    command="ss -tlnp | grep {port}",
                    description="Check if service port is listening",
                    safety_level=SafetyLevel.SAFE,
                    timeout_seconds=10
                ),
                Step(
                    action=ActionType.DB_QUERY,
                    target="echo_brain",
                    command="SELECT * FROM services WHERE name = '{service_name}'",
                    description="Get service configuration from database",
                    safety_level=SafetyLevel.SAFE
                )
            ]
        )

    async def _create_add_ollama_model_procedure(self) -> Procedure:
        """Create procedure for adding Ollama models"""
        return Procedure(
            name="add_ollama_model",
            description="Download and configure a new Ollama model",
            trigger_patterns=[
                r"add.*ollama.*model",
                r"install.*model",
                r"pull.*model",
                r"download.*llm"
            ],
            category=ProcedureCategory.OPERATIONAL,
            steps=[
                Step(
                    action=ActionType.SHELL,
                    target="disk",
                    command="df -h /var/lib/ollama",
                    description="Check available disk space",
                    safety_level=SafetyLevel.SAFE
                ),
                Step(
                    action=ActionType.SHELL,
                    target="ollama",
                    command="ollama list",
                    description="List currently installed models",
                    safety_level=SafetyLevel.SAFE
                ),
                Step(
                    action=ActionType.SHELL,
                    target="ollama",
                    command="ollama pull {model_name}",
                    description="Download the model",
                    safety_level=SafetyLevel.NEEDS_CONFIRM,
                    timeout_seconds=1800  # 30 minutes for large models
                ),
                Step(
                    action=ActionType.SHELL,
                    target="ollama",
                    command="ollama run {model_name} 'Hello, test'",
                    description="Test the model",
                    safety_level=SafetyLevel.SAFE,
                    timeout_seconds=60
                )
            ]
        )

    async def _create_restart_service_procedure(self) -> Procedure:
        """Create procedure for safely restarting services"""
        return Procedure(
            name="restart_service",
            description="Safely restart a systemd service",
            trigger_patterns=[
                r"restart.*service",
                r"reboot.*service",
                r"reload.*service"
            ],
            category=ProcedureCategory.OPERATIONAL,
            steps=[
                Step(
                    action=ActionType.SHELL,
                    target="systemd",
                    command="systemctl status {service_name}",
                    description="Check current service status",
                    safety_level=SafetyLevel.SAFE
                ),
                Step(
                    action=ActionType.SERVICE_MANAGE,
                    target="systemd",
                    command="restart {service_name}",
                    description="Restart the service",
                    safety_level=SafetyLevel.NEEDS_CONFIRM
                ),
                Step(
                    action=ActionType.SHELL,
                    target="system",
                    command="sleep 5",
                    description="Wait for service to start",
                    safety_level=SafetyLevel.SAFE
                ),
                Step(
                    action=ActionType.SHELL,
                    target="systemd",
                    command="systemctl status {service_name}",
                    description="Verify service is running",
                    safety_level=SafetyLevel.SAFE
                ),
                Step(
                    action=ActionType.API_CALL,
                    target="service",
                    command="GET http://localhost:{port}/health",
                    description="Test service endpoint if available",
                    safety_level=SafetyLevel.SAFE,
                    on_failure="continue"
                )
            ]
        )

    async def _create_analyze_error_logs_procedure(self) -> Procedure:
        """Create procedure for analyzing error logs"""
        return Procedure(
            name="analyze_error_logs",
            description="Parse and categorize recent error logs",
            trigger_patterns=[
                r"analyze.*logs",
                r"check.*errors",
                r"what.*wrong",
                r"debug.*issue"
            ],
            category=ProcedureCategory.DIAGNOSTIC,
            steps=[
                Step(
                    action=ActionType.SHELL,
                    target="systemd",
                    command="journalctl --since '24 hours ago' -p err --no-pager -n 100",
                    description="Get system error logs from last 24 hours",
                    safety_level=SafetyLevel.SAFE
                ),
                Step(
                    action=ActionType.SHELL,
                    target="systemd",
                    command="journalctl -u tower-* --since '24 hours ago' --no-pager -n 50",
                    description="Get Tower service logs",
                    safety_level=SafetyLevel.SAFE
                ),
                Step(
                    action=ActionType.SHELL,
                    target="system",
                    command="dmesg | tail -50",
                    description="Check kernel messages",
                    safety_level=SafetyLevel.SAFE
                ),
                Step(
                    action=ActionType.DB_QUERY,
                    target="echo_brain",
                    command="SELECT * FROM action_log WHERE success = false AND timestamp > NOW() - INTERVAL '24 hours' ORDER BY timestamp DESC LIMIT 20",
                    description="Get failed actions from Echo Brain",
                    safety_level=SafetyLevel.SAFE
                )
            ]
        )

    async def _create_check_system_health_procedure(self) -> Procedure:
        """Create procedure for comprehensive system health check"""
        return Procedure(
            name="check_system_health",
            description="Comprehensive health check of all Tower services",
            trigger_patterns=[
                r"health.*check",
                r"system.*status",
                r"check.*everything",
                r"overall.*health"
            ],
            category=ProcedureCategory.DIAGNOSTIC,
            steps=[
                Step(
                    action=ActionType.SHELL,
                    target="system",
                    command="systemctl list-units --type=service --state=failed tower-*",
                    description="Check for failed Tower services",
                    safety_level=SafetyLevel.SAFE
                ),
                Step(
                    action=ActionType.SHELL,
                    target="system",
                    command="df -h",
                    description="Check disk usage",
                    safety_level=SafetyLevel.SAFE
                ),
                Step(
                    action=ActionType.SHELL,
                    target="system",
                    command="free -h",
                    description="Check memory usage",
                    safety_level=SafetyLevel.SAFE
                ),
                Step(
                    action=ActionType.API_CALL,
                    target="echo_brain",
                    command="GET http://localhost:8309/health",
                    description="Check Echo Brain health",
                    safety_level=SafetyLevel.SAFE
                ),
                Step(
                    action=ActionType.API_CALL,
                    target="qdrant",
                    command="GET http://localhost:6333/",
                    description="Check Qdrant vector database",
                    safety_level=SafetyLevel.SAFE
                ),
                Step(
                    action=ActionType.API_CALL,
                    target="ollama",
                    command="GET http://localhost:11434/api/tags",
                    description="Check Ollama models",
                    safety_level=SafetyLevel.SAFE
                ),
                Step(
                    action=ActionType.DB_QUERY,
                    target="echo_brain",
                    command="SELECT COUNT(*) as vector_count FROM (SELECT 1) as dummy",
                    description="Check database connectivity",
                    safety_level=SafetyLevel.SAFE
                )
            ]
        )

    async def _create_update_from_git_procedure(self) -> Procedure:
        """Create procedure for updating from git"""
        return Procedure(
            name="update_from_git",
            description="Safely update service from git repository",
            trigger_patterns=[
                r"update.*git",
                r"pull.*latest",
                r"git.*update",
                r"sync.*code"
            ],
            category=ProcedureCategory.OPERATIONAL,
            steps=[
                Step(
                    action=ActionType.SHELL,
                    target="git",
                    command="git -C {repo_path} status --porcelain",
                    description="Check for uncommitted changes",
                    safety_level=SafetyLevel.SAFE
                ),
                Step(
                    action=ActionType.SHELL,
                    target="git",
                    command="git -C {repo_path} fetch origin",
                    description="Fetch latest changes",
                    safety_level=SafetyLevel.SAFE
                ),
                Step(
                    action=ActionType.SHELL,
                    target="git",
                    command="git -C {repo_path} pull origin main",
                    description="Pull latest changes",
                    safety_level=SafetyLevel.NEEDS_CONFIRM
                ),
                Step(
                    action=ActionType.SHELL,
                    target="service",
                    command="systemctl restart {service_name}",
                    description="Restart service with new code",
                    safety_level=SafetyLevel.NEEDS_CONFIRM
                ),
                Step(
                    action=ActionType.SHELL,
                    target="system",
                    command="sleep 10",
                    description="Wait for service to start",
                    safety_level=SafetyLevel.SAFE
                ),
                Step(
                    action=ActionType.API_CALL,
                    target="service",
                    command="GET http://localhost:{port}/health",
                    description="Verify service is healthy",
                    safety_level=SafetyLevel.SAFE,
                    expected_output="200"
                )
            ]
        )

    async def find_procedure(self, intent: str) -> Optional[Procedure]:
        """
        Match user intent to a procedure.
        """
        pool = await self.get_db_pool()

        try:
            async with pool.acquire() as conn:
                procedures = await conn.fetch(
                    "SELECT * FROM procedures ORDER BY success_count DESC"
                )

                intent_lower = intent.lower()

                # Try pattern matching first
                for proc_row in procedures:
                    trigger_patterns = proc_row['trigger_patterns'] or []

                    for pattern in trigger_patterns:
                        if re.search(pattern, intent_lower):
                            steps_data = json.loads(proc_row['steps'])
                            steps = [Step(**step) for step in steps_data]

                            return Procedure(
                                id=proc_row['id'],
                                name=proc_row['name'],
                                description=proc_row['description'],
                                trigger_patterns=trigger_patterns,
                                category=ProcedureCategory(proc_row['category']),
                                steps=steps,
                                created_at=proc_row['created_at'],
                                success_count=proc_row['success_count'],
                                failure_count=proc_row['failure_count']
                            )

                # Fallback to keyword matching
                keywords = {
                    'diagnose': ['diagnose_service_failure', 'analyze_error_logs', 'check_system_health'],
                    'restart': ['restart_service'],
                    'model': ['add_ollama_model'],
                    'health': ['check_system_health'],
                    'logs': ['analyze_error_logs'],
                    'update': ['update_from_git'],
                    'git': ['update_from_git']
                }

                for keyword, procedure_names in keywords.items():
                    if keyword in intent_lower:
                        for proc_row in procedures:
                            if proc_row['name'] in procedure_names:
                                steps_data = json.loads(proc_row['steps'])
                                steps = [Step(**step) for step in steps_data]

                                return Procedure(
                                    id=proc_row['id'],
                                    name=proc_row['name'],
                                    description=proc_row['description'],
                                    trigger_patterns=proc_row['trigger_patterns'] or [],
                                    category=ProcedureCategory(proc_row['category']),
                                    steps=steps,
                                    created_at=proc_row['created_at'],
                                    success_count=proc_row['success_count'],
                                    failure_count=proc_row['failure_count']
                                )

        except Exception as e:
            logger.error(f"Error finding procedure for intent '{intent}': {e}")

        return None

    async def execute_procedure(self, procedure: Procedure, context: Dict[str, Any] = None,
                              allow_dangerous: bool = False) -> Dict[str, Any]:
        """
        Execute steps, handling:
        - Conditional branches
        - Failure recovery
        - User confirmation for dangerous operations
        - Result collection
        """
        context = context or {}
        execution_id = None
        results = []
        success = True
        error_message = None

        pool = await self.get_db_pool()
        executor = self._get_executor()

        try:
            # Start execution record
            async with pool.acquire() as conn:
                execution_id = await conn.fetchval(
                    """INSERT INTO procedure_executions
                       (procedure_id, started_at, context)
                       VALUES ($1, $2, $3) RETURNING id""",
                    procedure.id, datetime.now(), json.dumps(context)
                )

            logger.info(f"Starting procedure execution: {procedure.name} (ID: {execution_id})")

            # Execute each step
            for i, step in enumerate(procedure.steps):
                try:
                    # Format command with context variables
                    formatted_command = step.command.format(**context)
                    formatted_target = step.target.format(**context) if step.target else step.target

                    logger.info(f"Step {i+1}: {step.description}")

                    # Check safety level
                    if step.safety_level == SafetyLevel.DANGEROUS and not allow_dangerous:
                        results.append({
                            'step': i + 1,
                            'action': step.action.value,
                            'command': formatted_command,
                            'success': False,
                            'message': 'Step blocked: dangerous operation requires explicit confirmation'
                        })
                        success = False
                        break

                    # Execute based on action type
                    step_result = None

                    if step.action == ActionType.SHELL:
                        safe_execution = allow_dangerous or step.safety_level == SafetyLevel.SAFE
                        shell_result = await executor.execute_shell(
                            formatted_command, safe=safe_execution, timeout=step.timeout_seconds
                        )
                        step_result = {
                            'success': shell_result.success,
                            'stdout': shell_result.stdout,
                            'stderr': shell_result.stderr,
                            'exit_code': shell_result.exit_code
                        }

                    elif step.action == ActionType.SERVICE_MANAGE:
                        parts = formatted_command.split()
                        if len(parts) >= 2:
                            action, service_name = parts[0], parts[1]
                            service_result = await executor.manage_service(
                                service_name, action, confirm_dangerous=allow_dangerous
                            )
                            step_result = {
                                'success': service_result.success,
                                'new_status': service_result.new_status,
                                'message': service_result.message
                            }

                    elif step.action == ActionType.API_CALL:
                        parts = formatted_command.split(' ', 1)
                        method = parts[0]
                        url = parts[1] if len(parts) > 1 else formatted_target

                        api_result = await executor.call_api(url, method, timeout=step.timeout_seconds)
                        step_result = {
                            'success': api_result.success,
                            'status_code': api_result.status_code,
                            'response_data': api_result.response_data
                        }

                    elif step.action == ActionType.DB_QUERY:
                        query_result = await executor.query_database(
                            formatted_target, formatted_command
                        )
                        step_result = {
                            'success': query_result.success,
                            'rows_affected': query_result.rows_affected,
                            'data': query_result.data
                        }

                    # Record step result
                    results.append({
                        'step': i + 1,
                        'action': step.action.value,
                        'description': step.description,
                        'command': formatted_command,
                        'target': formatted_target,
                        **step_result
                    })

                    # Check if step failed
                    if not step_result['success']:
                        if step.on_failure == 'continue':
                            logger.warning(f"Step {i+1} failed but continuing: {step.description}")
                        elif step.on_failure == 'retry':
                            logger.info(f"Retrying step {i+1}: {step.description}")
                            # Simple retry logic - could be enhanced
                            await asyncio.sleep(2)
                            # Re-execute step (simplified)
                            continue
                        else:  # abort
                            success = False
                            error_message = f"Step {i+1} failed: {step.description}"
                            break

                except Exception as e:
                    logger.error(f"Error executing step {i+1}: {e}")
                    results.append({
                        'step': i + 1,
                        'action': step.action.value,
                        'success': False,
                        'error': str(e)
                    })
                    success = False
                    error_message = str(e)
                    break

            # Update execution record
            if execution_id:
                async with pool.acquire() as conn:
                    await conn.execute(
                        """UPDATE procedure_executions
                           SET completed_at = $2, success = $3, result = $4, error_message = $5
                           WHERE id = $1""",
                        execution_id, datetime.now(), success,
                        json.dumps(results), error_message
                    )

                    # Update procedure statistics
                    if success:
                        await conn.execute(
                            "UPDATE procedures SET success_count = success_count + 1 WHERE id = $1",
                            procedure.id
                        )
                    else:
                        await conn.execute(
                            "UPDATE procedures SET failure_count = failure_count + 1 WHERE id = $1",
                            procedure.id
                        )

        except Exception as e:
            logger.error(f"Error during procedure execution: {e}")
            success = False
            error_message = str(e)

        return {
            'execution_id': execution_id,
            'procedure_name': procedure.name,
            'success': success,
            'steps_completed': len(results),
            'total_steps': len(procedure.steps),
            'results': results,
            'error_message': error_message
        }

    async def learn_procedure(self, task: str, steps: List[Dict[str, Any]], outcome: str):
        """Record a new procedure from observed actions"""
        pool = await self.get_db_pool()

        try:
            # Convert steps to Step objects
            step_objects = []
            for step_data in steps:
                step_objects.append(Step(
                    action=ActionType(step_data.get('action', 'shell')),
                    target=step_data.get('target', ''),
                    command=step_data.get('command', ''),
                    description=step_data.get('description', ''),
                    safety_level=SafetyLevel(step_data.get('safety_level', 'safe'))
                ))

            # Create procedure
            procedure_name = f"learned_{task.lower().replace(' ', '_')}"

            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO procedures
                       (name, description, trigger_patterns, category, steps)
                       VALUES ($1, $2, $3, $4, $5)""",
                    procedure_name,
                    f"Learned procedure: {task}",
                    [task.lower()],
                    ProcedureCategory.OPERATIONAL.value,
                    json.dumps([step.dict() for step in step_objects])
                )

            logger.info(f"Learned new procedure: {procedure_name}")

        except Exception as e:
            logger.error(f"Error learning procedure: {e}")

    async def list_procedures(self) -> List[Dict[str, Any]]:
        """List all available procedures"""
        pool = await self.get_db_pool()

        try:
            async with pool.acquire() as conn:
                procedures = await conn.fetch(
                    """SELECT name, description, category, success_count, failure_count
                       FROM procedures
                       ORDER BY name"""
                )

                return [
                    {
                        'name': row['name'],
                        'description': row['description'],
                        'category': row['category'],
                        'success_count': row['success_count'],
                        'failure_count': row['failure_count'],
                        'success_rate': (
                            row['success_count'] / (row['success_count'] + row['failure_count'])
                            if (row['success_count'] + row['failure_count']) > 0 else 0.0
                        )
                    }
                    for row in procedures
                ]

        except Exception as e:
            logger.error(f"Error listing procedures: {e}")
            return []


# Singleton instance
_procedure_library = None

def get_procedure_library() -> ProcedureLibrary:
    """Get or create singleton instance"""
    global _procedure_library
    if not _procedure_library:
        _procedure_library = ProcedureLibrary()
    return _procedure_library