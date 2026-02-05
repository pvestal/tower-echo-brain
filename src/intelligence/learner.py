"""
Learning Loop for Echo Brain
Updates Echo Brain's knowledge from experience.
NOT a training loop - this is about recording what worked/failed
and using that to improve future responses.
"""

import asyncio
import asyncpg
import json
import logging
import subprocess
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class LearningLoop:
    """
    Updates Echo Brain's knowledge from experience.
    NOT a training loop - this is about recording what worked/failed
    and using that to improve future responses.
    """

    def __init__(self, db_config: Dict[str, str] = None):
        self.db_config = db_config or {
            "host": "localhost",
            "database": "echo_brain",
            "user": "patrick",
            "password": "RP78eIrW7cI2jYvL5akt1yurE"
        }
        self._pool = None

    async def get_db_pool(self):
        """Get or create database connection pool"""
        if not self._pool:
            self._pool = await asyncpg.create_pool(
                **self.db_config,
                min_size=1,
                max_size=5,
                timeout=10
            )
        return self._pool

    async def close(self):
        """Clean up connections"""
        if self._pool:
            await self._pool.close()

    async def on_action_complete(self, action_type: str, command: str,
                                  success: bool, result: str, context: Dict):
        """Record action outcome in action_log table"""
        logger.info(f"Learning from action: {action_type} - {'Success' if success else 'Failed'}")

        try:
            pool = await self.get_db_pool()

            async with pool.acquire() as conn:
                # Record the action outcome
                await conn.execute(
                    """INSERT INTO action_log
                       (action_type, command, success, result, context, timestamp)
                       VALUES ($1, $2, $3, $4, $5, $6)""",
                    action_type, command, success, result,
                    json.dumps(context), datetime.now()
                )

                # Check for patterns - same action keeps failing
                if not success:
                    failure_count = await conn.fetchval(
                        """SELECT COUNT(*) FROM action_log
                           WHERE action_type = $1 AND command = $2 AND success = false
                           AND timestamp > NOW() - INTERVAL '24 hours'""",
                        action_type, command
                    )

                    if failure_count >= 3:
                        logger.warning(f"Action '{command}' has failed {failure_count} times in 24h")
                        await self._create_failure_pattern_fact(action_type, command, failure_count)

                # Update procedure success/failure counts if this was part of a procedure
                if context.get('procedure_id'):
                    if success:
                        await conn.execute(
                            "UPDATE procedures SET success_count = success_count + 1 WHERE id = $1",
                            context['procedure_id']
                        )
                    else:
                        await conn.execute(
                            "UPDATE procedures SET failure_count = failure_count + 1 WHERE id = $1",
                            context['procedure_id']
                        )

                # Create new facts from successful actions that solve issues
                if success and context.get('solved_issue'):
                    await self._create_solution_fact(command, context['solved_issue'])

        except Exception as e:
            logger.error(f"Error learning from action: {e}")

    async def on_code_change(self, paths: List[str]):
        """Re-index changed files"""
        logger.info(f"Learning from code changes in {len(paths)} files")

        try:
            from .code_index import get_code_intelligence
            code_intel = get_code_intelligence()

            # Re-index the changed files
            result = await code_intel.index_codebase(paths)

            logger.info(f"Re-indexed {len(paths)} changed files: {result.get('symbols_indexed', 0)} symbols")

            # Record the code change event
            pool = await self.get_db_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO code_changes
                       (paths, change_type, timestamp, symbols_reindexed)
                       VALUES ($1, $2, $3, $4)""",
                    json.dumps(paths), 'file_update', datetime.now(),
                    result.get('symbols_indexed', 0)
                )

        except Exception as e:
            logger.error(f"Error learning from code changes: {e}")

    async def on_service_change(self, service: str, old_status: str, new_status: str):
        """Record service state changes"""
        logger.info(f"Learning from service change: {service} {old_status} -> {new_status}")

        try:
            pool = await self.get_db_pool()

            async with pool.acquire() as conn:
                # Insert into service_health table
                await conn.execute(
                    """INSERT INTO service_health
                       (service_name, status, previous_status, timestamp, change_reason)
                       VALUES ($1, $2, $3, $4, $5)""",
                    service, new_status, old_status, datetime.now(), 'automatic_detection'
                )

                # If service went down, check if this is a pattern
                if new_status in ['failed', 'inactive', 'dead']:
                    failure_count = await conn.fetchval(
                        """SELECT COUNT(*) FROM service_health
                           WHERE service_name = $1 AND status IN ('failed', 'inactive', 'dead')
                           AND timestamp > NOW() - INTERVAL '24 hours'""",
                        service
                    )

                    if failure_count >= 3:
                        logger.warning(f"Service {service} has failed {failure_count} times in 24h")
                        await self._create_service_instability_fact(service, failure_count)

        except Exception as e:
            logger.error(f"Error learning from service change: {e}")

    async def run_maintenance(self):
        """
        Periodic maintenance - call this from a background task or cron:
        - Re-scan services
        - Check for code changes (git diff)
        - Clean old action logs (>30 days)
        - Update health metrics
        """
        logger.info("Running learning loop maintenance")

        try:
            # Clean old action logs (>30 days)
            await self._clean_old_logs()

            # Check for code changes
            await self._check_code_changes()

            # Re-scan services
            await self._scan_services()

            # Update health metrics
            await self._update_health_metrics()

            logger.info("Maintenance completed successfully")

        except Exception as e:
            logger.error(f"Error during maintenance: {e}")

    async def detect_patterns(self) -> List[Dict]:
        """
        Analyze action_log for patterns:
        - Commands that always fail
        - Services that keep restarting
        - Common error sequences
        """
        patterns = []

        try:
            pool = await self.get_db_pool()

            async with pool.acquire() as conn:
                # Find commands with high failure rates
                failing_commands = await conn.fetch(
                    """SELECT command, action_type,
                              COUNT(*) as total_attempts,
                              COUNT(*) FILTER (WHERE success = false) as failures,
                              ROUND(COUNT(*) FILTER (WHERE success = false) * 100.0 / COUNT(*), 2) as failure_rate
                       FROM action_log
                       WHERE timestamp > NOW() - INTERVAL '7 days'
                       GROUP BY command, action_type
                       HAVING COUNT(*) >= 3 AND
                              COUNT(*) FILTER (WHERE success = false) * 100.0 / COUNT(*) > 50
                       ORDER BY failure_rate DESC
                       LIMIT 10"""
                )

                for cmd in failing_commands:
                    patterns.append({
                        'type': 'failing_command',
                        'command': cmd['command'],
                        'action_type': cmd['action_type'],
                        'failure_rate': float(cmd['failure_rate']),
                        'total_attempts': cmd['total_attempts'],
                        'recommendation': f"Command '{cmd['command']}' fails {cmd['failure_rate']}% of the time"
                    })

                # Find services that restart frequently
                unstable_services = await conn.fetch(
                    """SELECT service_name,
                              COUNT(*) as status_changes,
                              COUNT(*) FILTER (WHERE status IN ('failed', 'inactive')) as failures
                       FROM service_health
                       WHERE timestamp > NOW() - INTERVAL '7 days'
                       GROUP BY service_name
                       HAVING COUNT(*) >= 5
                       ORDER BY status_changes DESC
                       LIMIT 5"""
                )

                for svc in unstable_services:
                    patterns.append({
                        'type': 'unstable_service',
                        'service': svc['service_name'],
                        'status_changes': svc['status_changes'],
                        'failures': svc['failures'],
                        'recommendation': f"Service '{svc['service_name']}' has {svc['status_changes']} status changes in 7 days"
                    })

                # Find common error sequences
                error_sequences = await conn.fetch(
                    """WITH error_actions AS (
                         SELECT command, action_type, result, timestamp,
                                LAG(command) OVER (ORDER BY timestamp) as prev_command
                         FROM action_log
                         WHERE success = false AND timestamp > NOW() - INTERVAL '7 days'
                       )
                       SELECT prev_command, command, COUNT(*) as occurrence_count
                       FROM error_actions
                       WHERE prev_command IS NOT NULL
                       GROUP BY prev_command, command
                       HAVING COUNT(*) >= 2
                       ORDER BY occurrence_count DESC
                       LIMIT 5"""
                )

                for seq in error_sequences:
                    patterns.append({
                        'type': 'error_sequence',
                        'sequence': f"{seq['prev_command']} -> {seq['command']}",
                        'occurrences': seq['occurrence_count'],
                        'recommendation': f"Command sequence often fails: {seq['prev_command']} followed by {seq['command']}"
                    })

        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            patterns.append({
                'type': 'error',
                'message': f"Could not analyze patterns: {str(e)}"
            })

        return patterns

    async def _create_failure_pattern_fact(self, action_type: str, command: str, failure_count: int):
        """Create a fact about a failing command pattern"""
        try:
            pool = await self.get_db_pool()

            async with pool.acquire() as conn:
                fact_content = f"Command '{command}' ({action_type}) has failed {failure_count} times in 24 hours"

                await conn.execute(
                    """INSERT INTO facts (content, source_type, source_id, confidence, tags, timestamp)
                       VALUES ($1, $2, $3, $4, $5, $6)
                       ON CONFLICT (content) DO UPDATE SET
                       confidence = EXCLUDED.confidence,
                       timestamp = EXCLUDED.timestamp""",
                    fact_content, 'learning_loop', 'failure_pattern', 0.9,
                    json.dumps(['failure', 'pattern', action_type]), datetime.now()
                )

        except Exception as e:
            logger.error(f"Error creating failure pattern fact: {e}")

    async def _create_service_instability_fact(self, service: str, failure_count: int):
        """Create a fact about an unstable service"""
        try:
            pool = await self.get_db_pool()

            async with pool.acquire() as conn:
                fact_content = f"Service '{service}' has failed {failure_count} times in 24 hours, indicating instability"

                await conn.execute(
                    """INSERT INTO facts (content, source_type, source_id, confidence, tags, timestamp)
                       VALUES ($1, $2, $3, $4, $5, $6)
                       ON CONFLICT (content) DO UPDATE SET
                       confidence = EXCLUDED.confidence,
                       timestamp = EXCLUDED.timestamp""",
                    fact_content, 'learning_loop', f'service_instability_{service}', 0.85,
                    json.dumps(['service', 'instability', service]), datetime.now()
                )

        except Exception as e:
            logger.error(f"Error creating service instability fact: {e}")

    async def _create_solution_fact(self, command: str, issue: str):
        """Create a fact about a successful solution"""
        try:
            pool = await self.get_db_pool()

            async with pool.acquire() as conn:
                fact_content = f"Command '{command}' successfully resolved issue: {issue}"

                await conn.execute(
                    """INSERT INTO facts (content, source_type, source_id, confidence, tags, timestamp)
                       VALUES ($1, $2, $3, $4, $5, $6)
                       ON CONFLICT (content) DO UPDATE SET
                       confidence = EXCLUDED.confidence + 0.1,
                       timestamp = EXCLUDED.timestamp""",
                    fact_content, 'learning_loop', 'solution', 0.8,
                    json.dumps(['solution', 'success']), datetime.now()
                )

        except Exception as e:
            logger.error(f"Error creating solution fact: {e}")

    async def _clean_old_logs(self):
        """Clean action logs older than 30 days"""
        try:
            pool = await self.get_db_pool()

            async with pool.acquire() as conn:
                # Delete old logs and get count separately
                result = await conn.execute(
                    """DELETE FROM action_log
                       WHERE timestamp < NOW() - INTERVAL '30 days'"""
                )
                deleted = int(result.split()[-1]) if result else 0

                if deleted > 0:
                    logger.info(f"Cleaned {deleted} old action log entries")

        except Exception as e:
            logger.error(f"Error cleaning old logs: {e}")

    async def _check_code_changes(self):
        """Check for code changes using git diff"""
        try:
            # Check main Echo Brain repo for changes
            repo_path = "/opt/tower-echo-brain"

            # Get list of changed files since last check
            result = subprocess.run(
                ["git", "-C", repo_path, "diff", "--name-only", "HEAD~1", "HEAD"],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0 and result.stdout.strip():
                changed_files = result.stdout.strip().split('\n')
                python_files = [f for f in changed_files if f.endswith('.py')]

                if python_files:
                    # Convert to full paths
                    full_paths = [f"{repo_path}/{f}" for f in python_files]
                    await self.on_code_change(full_paths)

        except Exception as e:
            logger.debug(f"Could not check code changes: {e}")

    async def _scan_services(self):
        """Re-scan system services"""
        try:
            from .system_model import get_system_model
            system_model = get_system_model()

            # Discover current services
            services = await system_model.discover_services()

            logger.debug(f"Scanned {len(services)} services during maintenance")

        except Exception as e:
            logger.error(f"Error scanning services: {e}")

    async def _update_health_metrics(self):
        """Update overall health metrics"""
        try:
            pool = await self.get_db_pool()

            async with pool.acquire() as conn:
                # Calculate success rate for last 24 hours
                stats = await conn.fetchrow(
                    """SELECT
                         COUNT(*) as total_actions,
                         COUNT(*) FILTER (WHERE success = true) as successful_actions,
                         ROUND(COUNT(*) FILTER (WHERE success = true) * 100.0 / COUNT(*), 2) as success_rate
                       FROM action_log
                       WHERE timestamp > NOW() - INTERVAL '24 hours'"""
                )

                if stats and stats['total_actions'] > 0:
                    # Store as a system health metric
                    await conn.execute(
                        """INSERT INTO system_metrics (metric_name, metric_value, timestamp)
                           VALUES ($1, $2, $3)""",
                        'action_success_rate_24h', float(stats['success_rate']), datetime.now()
                    )

                    logger.info(f"Health metrics: {stats['success_rate']}% success rate ({stats['successful_actions']}/{stats['total_actions']} actions)")

        except Exception as e:
            logger.error(f"Error updating health metrics: {e}")


# Singleton instance
_learning_loop = None

def get_learning_loop() -> LearningLoop:
    """Get or create singleton instance"""
    global _learning_loop
    if not _learning_loop:
        _learning_loop = LearningLoop()
    return _learning_loop