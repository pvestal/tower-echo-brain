#!/usr/bin/env python3
"""
Autonomous Repair Executor
Executes actual repairs with email notifications and detailed logging
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import sys
import time

# Import outcome learning system
sys.path.insert(0, '/opt/tower-echo-brain')
from scripts.tower_outcome_learning import TowerOutcomeLearning

logger = logging.getLogger(__name__)

class RepairExecutor:
    """Executes autonomous repairs with notifications and logging"""

    def __init__(self):
        self.repair_log_path = Path("/opt/tower-echo-brain/logs/autonomous_repairs.log")
        self.repair_history: List[Dict[str, Any]] = []
        self.smtp_config = {
            'server': 'smtp.gmail.com',
            'port': 587,
            'from_email': 'patrick.vestal.digital@gmail.com',
            'to_email': 'patrick.vestal@gmail.com'
        }

        # Load Gmail app password from vault or environment
        self.app_password = self._load_smtp_credentials()

        # Initialize outcome learning system
        self.learner = TowerOutcomeLearning()
        logger.info("✅ Outcome learning system connected to repair executor")

    def _load_smtp_credentials(self) -> Optional[str]:
        """Load SMTP credentials from Vault"""
        try:
            import hvac
            import os
            client = hvac.Client(url='http://127.0.0.1:8200', token=os.environ.get('VAULT_TOKEN'))
            # Try to get from vault
            response = client.secrets.kv.v2.read_secret_version(path='tower/gmail')
            return response['data']['data'].get('app_password')
        except Exception as e:
            logger.warning(f"Could not load SMTP credentials from Vault: {e}")
            return None

    async def execute_repair(self, repair_type: str, target: str, issue: str, **kwargs) -> Dict[str, Any]:
        """Execute a repair action with outcome learning"""

        logger.info(f"🔧 Starting repair: {repair_type} for {target}")

        # Check if learner has a better action based on past outcomes
        best_action = self.learner.get_best_action(target, issue)
        if best_action and best_action['confidence'] > 0.8:
            logger.info(f"📊 Using learned action: {best_action['action']} (confidence: {best_action['confidence']*100:.0f}%)")

        start_time = time.time()

        result = {
            'success': False,
            'repair_type': repair_type,
            'target': target,
            'issue': issue,
            'timestamp': datetime.now().isoformat(),
            'actions_taken': [],
            'error': None
        }

        try:
            # Execute repair based on type
            if repair_type == 'service_restart':
                result = await self._repair_service_restart(target, issue, **kwargs)
            elif repair_type == 'systemd_config_fix':
                result = await self._repair_systemd_config(target, issue, **kwargs)
            elif repair_type == 'disk_cleanup':
                result = await self._repair_disk_cleanup(target, issue, **kwargs)
            elif repair_type == 'process_kill':
                result = await self._repair_process_kill(target, issue, **kwargs)
            elif repair_type == 'log_rotation':
                result = await self._repair_log_rotation(target, issue, **kwargs)
            elif repair_type == "code_modification":
                result = await self._repair_code_modification(target, issue, **kwargs)
            elif repair_type == "style_update":
                result = await self._repair_style_update(target, issue, **kwargs)
            elif repair_type == 'vault_unseal':
                result = await self._repair_vault_unseal(target, issue, **kwargs)
            else:
                result['error'] = f"Unknown repair type: {repair_type}"

            # Log the repair
            await self._log_repair(result)

            # Send email notification
# DISABLED by Patrick -             await self._send_repair_notification(result)

            # Record outcome for learning
            time_taken = time.time() - start_time
            action_description = f"{repair_type}: {', '.join(result.get('actions_taken', []))}"

            self.learner.record_repair_attempt(
                service=target,
                issue=issue,
                action=action_description[:100],  # Limit action description length
                success=result.get('success', False),
                time_taken=time_taken,
                error=result.get('error')
            )

            if result.get('success'):
                logger.info(f"✅ Repair successful! Learned outcome recorded (time: {time_taken:.1f}s)")
            else:
                logger.info(f"❌ Repair failed. Learning from failure...")

            # Store in history
            self.repair_history.append(result)

            return result

        except Exception as e:
            logger.error(f"❌ Repair failed: {e}")
            result['error'] = str(e)
            await self._log_repair(result)
# DISABLED by Patrick -             await self._send_repair_notification(result)
            return result

    async def _repair_service_restart(self, service: str, issue: str, **kwargs) -> Dict[str, Any]:
        """Restart a systemd service"""

        result = {
            'success': False,
            'repair_type': 'service_restart',
            'target': service,
            'issue': issue,
            'timestamp': datetime.now().isoformat(),
            'actions_taken': [],
            'error': None
        }

        try:
            # Check if service exists
            check_cmd = f"systemctl list-unit-files | grep {service}"
            proc = await asyncio.create_subprocess_shell(
                check_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                result['error'] = f"Service {service} not found"
                return result

            result['actions_taken'].append(f"Verified service {service} exists")

            # Restart the service
            restart_cmd = f"sudo systemctl restart {service}"
            proc = await asyncio.create_subprocess_shell(
                restart_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                result['actions_taken'].append(f"Successfully restarted {service}")
                result['success'] = True
            else:
                result['error'] = f"Failed to restart: {stderr.decode()}"

            # Wait a moment for service to start
            await asyncio.sleep(3)

            # Check status
            status_cmd = f"systemctl is-active {service}"
            proc = await asyncio.create_subprocess_shell(
                status_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            status = stdout.decode().strip()
            result['actions_taken'].append(f"Service status after restart: {status}")

            if status == "active":
                result['success'] = True

            return result

        except Exception as e:
            result['error'] = str(e)
            return result

    async def _repair_systemd_config(self, service: str, issue: str, **kwargs) -> Dict[str, Any]:
        """Fix systemd service configuration - stub for now"""
        return {
            'success': False,
            'repair_type': 'systemd_config_fix',
            'target': service,
            'issue': issue,
            'timestamp': datetime.now().isoformat(),
            'actions_taken': ['Not yet implemented'],
            'error': 'systemd_config_fix not yet implemented - requires manual intervention'
        }

    async def _repair_disk_cleanup(self, target: str, issue: str, **kwargs) -> Dict[str, Any]:
        """Clean up zombie processes and other system maintenance"""

        result = {
            'success': False,
            'repair_type': 'disk_cleanup',
            'target': target,
            'issue': issue,
            'timestamp': datetime.now().isoformat(),
            'actions_taken': [],
            'error': None
        }

        try:
            if 'zombie' in issue.lower() or target == 'zombie_processes':
                # Check for actual zombie processes
                zombie_check_cmd = "ps aux | awk '$8 ~ /^Z/ { print $2, $11 }'"
                proc = await asyncio.create_subprocess_shell(
                    zombie_check_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()

                zombie_output = stdout.decode().strip()
                if zombie_output:
                    result['actions_taken'].append(f"Found zombie processes: {zombie_output}")
                    # Zombie processes can't be killed directly - they're already dead
                    # We need to notify their parent or wait for parent cleanup
                    result['actions_taken'].append("Zombie processes require parent process cleanup")
                    result['success'] = True  # We identified the issue
                else:
                    result['actions_taken'].append("No actual zombie processes found")
                    result['success'] = True
            else:
                result['actions_taken'].append(f"General disk cleanup for {target} not yet implemented")
                result['error'] = 'General disk cleanup not yet implemented'

            return result

        except Exception as e:
            result['error'] = str(e)
            return result

    async def _repair_process_kill(self, target: str, issue: str, **kwargs) -> Dict[str, Any]:
        """Kill a stuck process"""

        result = {
            'success': False,
            'repair_type': 'process_kill',
            'target': target,
            'issue': issue,
            'timestamp': datetime.now().isoformat(),
            'actions_taken': [],
            'error': None
        }

        try:
            pid = kwargs.get('pid')
            if not pid:
                result['error'] = "No PID provided"
                return result

            # Try graceful kill first
            kill_cmd = f"kill {pid}"
            proc = await asyncio.create_subprocess_shell(
                kill_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
            result['actions_taken'].append(f"Sent SIGTERM to PID {pid}")

            # Wait a moment
            await asyncio.sleep(3)

            # Check if process is still running
            check_cmd = f"ps -p {pid}"
            proc = await asyncio.create_subprocess_shell(
                check_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                # Process is gone
                result['actions_taken'].append(f"Process {pid} terminated successfully")
                result['success'] = True
            else:
                # Force kill
                force_kill_cmd = f"kill -9 {pid}"
                proc = await asyncio.create_subprocess_shell(
                    force_kill_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await proc.communicate()
                result['actions_taken'].append(f"Sent SIGKILL to PID {pid}")
                result['success'] = True

            return result

        except Exception as e:
            result['error'] = str(e)
            return result

    async def _repair_log_rotation(self, target: str, issue: str, **kwargs) -> Dict[str, Any]:
        """Rotate large log files - stub for now"""
        return {
            'success': False,
            'repair_type': 'log_rotation',
            'target': target,
            'issue': issue,
            'timestamp': datetime.now().isoformat(),
            'actions_taken': ['Not yet implemented'],
            'error': 'log_rotation not yet implemented - requires manual intervention'
        }

    async def _repair_vault_unseal(self, target: str, issue: str, **kwargs) -> Dict[str, Any]:
        """Unseal HashiCorp Vault using stored unseal keys"""

        result = {
            'success': False,
            'repair_type': 'vault_unseal',
            'target': target,
            'issue': issue,
            'timestamp': datetime.now().isoformat(),
            'actions_taken': [],
            'error': None
        }

        try:
            # Set Vault environment variables
            vault_env = {
                'VAULT_ADDR': 'http://127.0.0.1:8200',
                'PATH': '/usr/local/bin:/usr/bin:/bin'
            }

            # First check if Vault is sealed
            check_cmd = "vault status"
            proc = await asyncio.create_subprocess_shell(
                check_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**vault_env}
            )
            stdout, stderr = await proc.communicate()

            status_output = stdout.decode().strip()
            result['actions_taken'].append(f"Checked vault status: {status_output}")

            # Check if vault is actually sealed (status returns 2 when sealed)
            if proc.returncode != 2:
                # Vault is already unsealed or there's another issue
                if "Sealed: false" in status_output:
                    result['actions_taken'].append("Vault is already unsealed")
                    result['success'] = True
                    return result
                else:
                    result['error'] = f"Vault status check failed: {stderr.decode()}"
                    return result

            result['actions_taken'].append("Vault is sealed, proceeding with unseal")

            # Load unseal keys from backup file
            vault_init_file = "/home/patrick/.vault-init-backup.json"
            try:
                import json
                with open(vault_init_file, 'r') as f:
                    vault_init_data = json.load(f)

                unseal_keys = vault_init_data.get('unseal_keys_b64', [])
                root_token = vault_init_data.get('root_token')

                if not unseal_keys or len(unseal_keys) < 3:
                    result['error'] = "Insufficient unseal keys found in backup file"
                    return result

                result['actions_taken'].append(f"Loaded {len(unseal_keys)} unseal keys from backup")

            except Exception as e:
                result['error'] = f"Failed to load vault init data: {str(e)}"
                return result

            # Unseal vault using first 3 keys
            for i in range(3):
                if i >= len(unseal_keys):
                    break

                unseal_cmd = f"vault operator unseal {unseal_keys[i]}"
                proc = await asyncio.create_subprocess_shell(
                    unseal_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={**vault_env}
                )
                stdout, stderr = await proc.communicate()

                if proc.returncode == 0:
                    result['actions_taken'].append(f"Applied unseal key {i+1}/3")
                else:
                    result['error'] = f"Failed to apply unseal key {i+1}: {stderr.decode()}"
                    return result

            # Wait a moment for vault to fully unseal
            await asyncio.sleep(2)

            # Verify vault is now unsealed
            verify_cmd = "vault status"
            proc = await asyncio.create_subprocess_shell(
                verify_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**vault_env}
            )
            stdout, stderr = await proc.communicate()

            status_output = stdout.decode()
            if proc.returncode == 0 and "Sealed          false" in status_output:
                result['actions_taken'].append("Vault successfully unsealed")
                result['success'] = True

                # Restore vault token if available
                if root_token:
                    try:
                        token_file = "/opt/vault/.vault-token"
                        Path(token_file).parent.mkdir(parents=True, exist_ok=True)
                        with open(token_file, 'w') as f:
                            f.write(root_token)
                        result['actions_taken'].append("Restored vault root token")
                    except Exception as e:
                        result['actions_taken'].append(f"Warning: Failed to restore token: {str(e)}")
            else:
                result['error'] = f"Vault unseal verification failed: {stderr.decode()}"

            return result

        except Exception as e:
            result['error'] = str(e)
            return result

    async def _log_repair(self, result: Dict[str, Any]):
        """Log repair action to file"""

        try:
            log_entry = {
                'timestamp': result.get('timestamp'),
                'repair_type': result.get('repair_type'),
                'target': result.get('target'),
                'issue': result.get('issue'),
                'success': result.get('success'),
                'actions_taken': result.get('actions_taken'),
                'error': result.get('error')
            }

            # Ensure log directory exists
            self.repair_log_path.parent.mkdir(parents=True, exist_ok=True)

            # Append to log file
            with open(self.repair_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

            repair_type = result.get('repair_type', 'unknown')
            target = result.get('target', 'unknown')
            logger.info(f"📝 Logged repair: {repair_type} for {target}")

        except Exception as e:
            logger.error(f"Failed to log repair: {e}")

    async def _send_repair_notification(self, result: Dict[str, Any]):
        """Send email notification about repair"""

        try:
            # Create email content
            success = result.get('success', False)
            status_emoji = "✅" if success else "❌"
            status_text = "Success" if success else "Failed"

            repair_type = result.get('repair_type', 'unknown')
            subject = f"🔧 Echo Autonomous Repair: {repair_type} - {status_emoji} {status_text}"

            actions_taken = result.get('actions_taken', [])
            actions_text = '\n'.join(f"  - {action}" for action in actions_taken)

            error = result.get('error')
            error_text = f"\n\nError: {error}" if error else ""

            timestamp = result.get('timestamp', 'unknown')
            target = result.get('target', 'unknown')
            issue = result.get('issue', 'unknown')

            body = f"""Echo Brain Autonomous Repair Report
===================================

Timestamp: {timestamp}
Repair Type: {repair_type}
Target: {target}
Issue: {issue}
Status: {status_emoji} {status_text.upper()}

Actions Taken:
{actions_text}{error_text}

---
This repair was executed autonomously by Echo Brain.
Review the full repair log at: /opt/tower-echo-brain/logs/autonomous_repairs.log
"""

            # Send email
            await self._send_email(subject, body)

            logger.info("📧 Sent repair notification email")

        except Exception as e:
            logger.error(f"Failed to send repair notification: {e}")

    async def _send_email(self, subject: str, body: str):
        """Send email via centralized email manager"""

        try:
            # Import the centralized email manager
            from src.utils.email_manager import email_manager

            # Send via the email manager
            success = await email_manager.send_email(subject, body)

            if success:
                logger.info("✅ Email sent successfully via email manager")
            else:
                logger.warning("⚠️ Email could not be sent but was logged")

        except ImportError:
            # Fallback to logging if email manager not available
            logger.warning("Email manager not available - logging email content")
            logger.info(f"📧 Email Notification:\nSubject: {subject}\n{body}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")

# Global instance
repair_executor = RepairExecutor()
