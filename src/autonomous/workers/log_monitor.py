"""Log Monitor Worker - Monitors Echo Brain service logs for errors and issues"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

import asyncpg

logger = logging.getLogger(__name__)


class LogMonitor:
    """Monitors Echo Brain's own service logs for errors and anomalies"""

    def __init__(self):
        self.db_url = os.environ.get("DATABASE_URL",
            "postgresql://echo:echo_secure_password_123@localhost/echo_brain")
        self.log_file = "/opt/tower-echo-brain/logs/echo_brain.log"
        self.minutes_to_scan = 15  # Look back 15 minutes each cycle

    async def run_cycle(self):
        """Main worker cycle — called by scheduler"""
        logger.info("📋 Log Monitor starting cycle")

        try:
            conn = await asyncpg.connect(self.db_url)

            # Read recent logs
            log_lines = await self._read_logs(self.minutes_to_scan)
            logger.info(f"Scanning {len(log_lines)} log lines")

            # Parse and classify issues
            issues_found = await self._parse_logs(log_lines)

            # Track statistics
            new_issues = 0
            duplicate_issues = 0

            for issue in issues_found:
                # Check for duplicates within the last hour
                existing = await conn.fetchrow("""
                    SELECT id FROM self_detected_issues
                    WHERE issue_type = $1
                        AND title = $2
                        AND created_at > NOW() - INTERVAL '1 hour'
                        AND status = 'open'
                    LIMIT 1
                """, issue['issue_type'], issue['title'])

                if existing:
                    duplicate_issues += 1
                    continue

                # Store new issue
                await conn.execute("""
                    INSERT INTO self_detected_issues
                    (issue_type, severity, source, title, description, related_file, related_worker, stack_trace, status)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                    issue['issue_type'], issue['severity'], 'log_monitor',
                    issue['title'], issue.get('description', ''),
                    issue.get('related_file'), issue.get('related_worker'),
                    issue.get('stack_trace'), 'open'
                )
                new_issues += 1

                # Create notification for critical issues
                if issue['severity'] == 'critical':
                    await self._create_notification(conn, issue)

            # Record metrics
            await conn.execute("""
                INSERT INTO self_health_metrics (metric_name, metric_value, metadata)
                VALUES
                    ('log_monitor_lines_scanned', $1, $2::jsonb),
                    ('log_monitor_issues_found', $3, $4::jsonb)
            """,
                float(len(log_lines)),
                json.dumps({"minutes_scanned": self.minutes_to_scan}),
                float(new_issues),
                json.dumps({"duplicates_skipped": duplicate_issues})
            )

            await conn.close()

            logger.info(f"✅ Log Monitor completed: {len(log_lines)} lines scanned, "
                       f"{new_issues} new issues, {duplicate_issues} duplicates skipped")

        except Exception as e:
            logger.error(f"❌ Log Monitor cycle failed: {e}", exc_info=True)

            # Try to record the failure
            try:
                conn = await asyncpg.connect(self.db_url)
                await conn.execute("""
                    INSERT INTO self_detected_issues
                    (issue_type, severity, source, title, description, related_worker)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """,
                    "worker_failure", "critical", "log_monitor",
                    "Log Monitor cycle failed",
                    str(e), "log_monitor"
                )
                await conn.close()
            except:
                pass

    async def _read_logs(self, minutes: int) -> List[str]:
        """Read recent logs, trying journalctl first, then log file"""
        lines = []

        # Try journalctl first (may not work for echo user)
        try:
            proc = await asyncio.create_subprocess_exec(
                'journalctl', '-u', 'tower-echo-brain',
                '--since', f'{minutes} minutes ago', '--no-pager',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                lines = stdout.decode('utf-8', errors='replace').splitlines()
                logger.info(f"Read {len(lines)} lines from journalctl")
                return lines
        except Exception as e:
            logger.debug(f"journalctl not available: {e}")

        # Fallback: read log file if it exists
        if os.path.exists(self.log_file):
            try:
                # Read file and filter by timestamp
                cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)

                with open(self.log_file, 'r', encoding='utf-8', errors='replace') as f:
                    # Read last 10000 lines (tail-like behavior)
                    all_lines = f.readlines()
                    recent_lines = all_lines[-10000:] if len(all_lines) > 10000 else all_lines

                    for line in recent_lines:
                        # Try to parse timestamp (assuming ISO format in logs)
                        # Example: 2026-02-06 12:34:56.789 INFO ...
                        timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})', line)
                        if timestamp_match:
                            try:
                                log_time = datetime.fromisoformat(timestamp_match.group(1).replace(' ', 'T'))
                                if log_time.tzinfo is None:
                                    log_time = log_time.replace(tzinfo=timezone.utc)
                                if log_time >= cutoff_time:
                                    lines.append(line.rstrip())
                            except:
                                lines.append(line.rstrip())  # Include if we can't parse timestamp
                        else:
                            # Include lines without timestamps (like stack traces)
                            if lines:  # Only if we already have some lines
                                lines.append(line.rstrip())

                logger.info(f"Read {len(lines)} recent lines from log file")
            except Exception as e:
                logger.error(f"Failed to read log file: {e}")

        # If no logs available, try to get systemd status
        if not lines:
            try:
                proc = await asyncio.create_subprocess_exec(
                    'systemctl', 'status', 'tower-echo-brain', '--no-pager', '-n', '50',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode in (0, 3):  # 3 = service not running
                    lines = stdout.decode('utf-8', errors='replace').splitlines()
                    logger.info(f"Read {len(lines)} lines from systemctl status")
            except:
                pass

        return lines

    async def _parse_logs(self, log_lines: List[str]) -> List[Dict[str, Any]]:
        """Parse log lines and extract issues"""
        issues = []
        current_traceback = []
        in_traceback = False

        for i, line in enumerate(log_lines):
            # Check for ERROR level
            if 'ERROR' in line or 'CRITICAL' in line:
                # Extract the error message
                error_match = re.search(r'(ERROR|CRITICAL)[:\s]+(.+)', line)
                if error_match:
                    severity = 'critical' if 'CRITICAL' in line else 'warning'
                    message = error_match.group(2).strip()

                    # Classify the error
                    issue_type = self._classify_error(message)

                    # Extract worker name if present
                    worker_match = re.search(r'(fact_extraction|conversation_watcher|knowledge_graph|codebase_indexer|schema_indexer|log_monitor|self_test_runner)', line, re.IGNORECASE)
                    worker_name = worker_match.group(1) if worker_match else None

                    # Extract file path if present
                    file_match = re.search(r'File "([^"]+)"', message)
                    file_path = file_match.group(1) if file_match else None

                    issues.append({
                        'issue_type': issue_type,
                        'severity': severity,
                        'title': message[:200],  # Truncate for title
                        'description': message,
                        'related_worker': worker_name,
                        'related_file': file_path
                    })

            # Check for WARNING level
            elif 'WARNING' in line or 'WARN' in line:
                warn_match = re.search(r'(WARNING|WARN)[:\s]+(.+)', line)
                if warn_match:
                    message = warn_match.group(2).strip()

                    # Only track specific warnings
                    if any(pattern in message.lower() for pattern in [
                        'deprecated', 'timeout', 'retry', 'failed to',
                        'connection', 'permission', 'not found'
                    ]):
                        issues.append({
                            'issue_type': self._classify_error(message),
                            'severity': 'warning',
                            'title': message[:200],
                            'description': message
                        })

            # Check for Python traceback
            elif 'Traceback' in line:
                in_traceback = True
                current_traceback = [line]
            elif in_traceback:
                current_traceback.append(line)
                # End of traceback
                if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    in_traceback = False
                    if current_traceback:
                        # Extract exception type and message
                        trace_text = '\n'.join(current_traceback)
                        exception_match = re.search(r'(\w+Error|\w+Exception): (.+)', trace_text)

                        if exception_match:
                            exception_type = exception_match.group(1)
                            exception_msg = exception_match.group(2).strip()

                            issues.append({
                                'issue_type': 'exception',
                                'severity': 'critical',
                                'title': f"{exception_type}: {exception_msg[:150]}",
                                'description': exception_msg,
                                'stack_trace': trace_text[:5000]  # Limit stack trace size
                            })
                        current_traceback = []

            # Check for specific patterns
            elif 'DDR5' in line:
                # Known bad data - DDR5 should be DDR6
                issues.append({
                    'issue_type': 'stale_data',
                    'severity': 'warning',
                    'title': 'DDR5 detected in response (should be DDR6)',
                    'description': f"Line contains incorrect DDR5 reference: {line[:500]}"
                })

            elif 'permission denied' in line.lower():
                issues.append({
                    'issue_type': 'permission_error',
                    'severity': 'critical',
                    'title': 'Permission denied error',
                    'description': line[:500]
                })

            elif re.search(r'No module named [\'"](\w+)[\'"]', line):
                module_match = re.search(r'No module named [\'"](\w+)[\'"]', line)
                if module_match:
                    issues.append({
                        'issue_type': 'import_error',
                        'severity': 'critical',
                        'title': f"Missing module: {module_match.group(1)}",
                        'description': line[:500]
                    })

        return issues

    def _classify_error(self, message: str) -> str:
        """Classify error message into issue type"""
        message_lower = message.lower()

        if 'permission' in message_lower or 'access denied' in message_lower:
            return 'permission_error'
        elif 'no module' in message_lower or 'import' in message_lower:
            return 'import_error'
        elif 'worker' in message_lower or 'cycle failed' in message_lower:
            return 'worker_failure'
        elif 'connection' in message_lower or 'timeout' in message_lower:
            return 'connection_error'
        elif 'sql' in message_lower or 'query' in message_lower:
            return 'database_error'
        elif '404' in message or 'not found' in message_lower:
            return 'not_found_error'
        elif 'method not allowed' in message_lower:
            return 'routing_error'
        elif 'ddr5' in message_lower:
            return 'stale_data'
        else:
            return 'general_error'

    async def _create_notification(self, conn: asyncpg.Connection, issue: Dict[str, Any]):
        """Create an autonomous notification for critical issues"""
        try:
            await conn.execute("""
                INSERT INTO autonomous_notifications
                (title, message, notification_type, severity, metadata)
                VALUES ($1, $2, $3, $4, $5::jsonb)
            """,
                f"Critical Issue: {issue['title'][:100]}",
                f"{issue.get('description', issue['title'])}\n\nDetected by: Log Monitor",
                'alert',
                'high',
                json.dumps({
                    'issue_type': issue['issue_type'],
                    'related_worker': issue.get('related_worker'),
                    'related_file': issue.get('related_file')
                })
            )
        except Exception as e:
            logger.error(f"Failed to create notification: {e}")