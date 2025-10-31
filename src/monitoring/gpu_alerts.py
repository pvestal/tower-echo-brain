#!/usr/bin/env python3
"""
GPU VRAM Alert System
Monitors AMD and NVIDIA GPU VRAM usage and sends alerts when thresholds are exceeded
"""

import asyncio
import logging
import subprocess
from datetime import datetime, timedelta
from typing import Dict, Optional
import smtplib
from email.mime.text import MIMEText
from pathlib import Path

logger = logging.getLogger(__name__)

class GPUMonitor:
    """Monitors GPU VRAM usage and sends alerts"""

    def __init__(self):
        self.alert_thresholds = {
            "amd_vram_warning": 90.0,  # 90% VRAM usage
            "amd_vram_critical": 95.0, # 95% VRAM usage
            "nvidia_vram_warning": 85.0,
            "nvidia_vram_critical": 90.0
        }

        self.last_alert_times = {}
        self.alert_cooldown = 300  # 5 minutes between same alerts

        # Email configuration
        self.smtp_config = {
            'server': 'smtp.gmail.com',
            'port': 587,
            'from_email': 'patrick.vestal.digital@gmail.com',
            'to_email': 'patrick.vestal@gmail.com'
        }

    def _load_gmail_credentials(self) -> Optional[str]:
        """Load Gmail app password from vault or fallback"""
        try:
            # Try to load from vault first
            import hvac
            import os
            client = hvac.Client(url='http://127.0.0.1:8200', token=os.environ.get('VAULT_TOKEN'))
            response = client.secrets.kv.v2.read_secret_version(path='tower/gmail')
            return response['data']['data'].get('app_password')
        except Exception:
            # Fallback to file-based credential
            try:
                with open("/home/patrick/.gmail-app-password", "r") as f:
                    return f.read().strip()
            except Exception:
                return None

    async def get_gpu_stats(self) -> Dict:
        """Get current GPU statistics"""
        stats = {
            "nvidia": {"vram_used_gb": 0, "vram_total_gb": 12, "vram_percent": 0},
            "amd": {"vram_used_gb": 0, "vram_total_gb": 16, "vram_percent": 0}
        }

        # NVIDIA GPU
        try:
            result = await asyncio.create_subprocess_exec(
                'nvidia-smi', '--query-gpu=memory.used,memory.total',
                '--format=csv,noheader,nounits',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            if result.returncode == 0:
                vram_data = stdout.decode().strip().split(',')
                vram_used_mb = float(vram_data[0])
                vram_total_mb = float(vram_data[1])
                stats["nvidia"]["vram_used_gb"] = vram_used_mb / 1024
                stats["nvidia"]["vram_total_gb"] = vram_total_mb / 1024
                stats["nvidia"]["vram_percent"] = (vram_used_mb / vram_total_mb) * 100
        except Exception as e:
            logger.debug(f"NVIDIA GPU not available: {e}")

        # AMD GPU
        try:
            result = await asyncio.create_subprocess_exec(
                'rocm-smi', '--showmemuse',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            if result.returncode == 0:
                lines = stdout.decode().split('\n')
                for line in lines:
                    if 'GPU[0]' in line and 'GPU Memory Allocated (VRAM%)' in line:
                        try:
                            vram_percent = float(line.split(':')[-1].strip())
                            stats["amd"]["vram_percent"] = vram_percent
                            stats["amd"]["vram_used_gb"] = (vram_percent / 100) * 16
                            break
                        except:
                            pass
        except Exception as e:
            logger.debug(f"AMD GPU not available: {e}")

        return stats

    async def send_alert_email(self, subject: str, message: str) -> bool:
        """Send alert email"""
        try:
            app_password = self._load_gmail_credentials()
            if not app_password:
                logger.error("No Gmail credentials available for alerts")
                return False

            msg = MIMEText(message)
            msg['Subject'] = f"[Echo Brain Alert] {subject}"
            msg['From'] = self.smtp_config['from_email']
            msg['To'] = self.smtp_config['to_email']

            with smtplib.SMTP(self.smtp_config['server'], self.smtp_config['port']) as server:
                server.starttls()
                server.login(self.smtp_config['from_email'], app_password)
                server.send_message(msg)

            logger.info(f"Alert email sent: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send alert email: {e}")
            return False

    def _should_send_alert(self, alert_type: str) -> bool:
        """Check if enough time has passed since last alert of this type"""
        now = datetime.now()
        last_alert = self.last_alert_times.get(alert_type)

        if not last_alert:
            return True

        time_since_last = (now - last_alert).total_seconds()
        return time_since_last >= self.alert_cooldown

    async def check_and_alert(self) -> Dict:
        """Check GPU usage and send alerts if needed"""
        stats = await self.get_gpu_stats()
        alerts_sent = []

        # Check AMD GPU VRAM
        amd_vram_percent = stats["amd"]["vram_percent"]
        if amd_vram_percent >= self.alert_thresholds["amd_vram_critical"]:
            if self._should_send_alert("amd_vram_critical"):
                subject = f"CRITICAL: AMD GPU VRAM at {amd_vram_percent:.1f}%"
                message = f"""
CRITICAL GPU ALERT

AMD GPU VRAM usage has reached {amd_vram_percent:.1f}% ({stats["amd"]["vram_used_gb"]:.1f}GB / {stats["amd"]["vram_total_gb"]:.1f}GB)

This is above the critical threshold of {self.alert_thresholds["amd_vram_critical"]}%.
System may experience OOM errors or performance degradation.

Recommended actions:
1. Check Ollama model usage: ollama ps
2. Unload unnecessary models: ollama unload <model>
3. Restart Ollama service if needed: sudo systemctl restart ollama

Time: {datetime.now().isoformat()}
Server: Tower (192.168.50.135)
                """

                if await self.send_alert_email(subject, message):
                    self.last_alert_times["amd_vram_critical"] = datetime.now()
                    alerts_sent.append("amd_vram_critical")

        elif amd_vram_percent >= self.alert_thresholds["amd_vram_warning"]:
            if self._should_send_alert("amd_vram_warning"):
                subject = f"WARNING: AMD GPU VRAM at {amd_vram_percent:.1f}%"
                message = f"""
GPU WARNING

AMD GPU VRAM usage is at {amd_vram_percent:.1f}% ({stats["amd"]["vram_used_gb"]:.1f}GB / {stats["amd"]["vram_total_gb"]:.1f}GB)

This is above the warning threshold of {self.alert_thresholds["amd_vram_warning"]}%.
Consider unloading unused models or monitoring usage.

Time: {datetime.now().isoformat()}
                """

                if await self.send_alert_email(subject, message):
                    self.last_alert_times["amd_vram_warning"] = datetime.now()
                    alerts_sent.append("amd_vram_warning")

        # Check NVIDIA GPU VRAM
        nvidia_vram_percent = stats["nvidia"]["vram_percent"]
        if nvidia_vram_percent >= self.alert_thresholds["nvidia_vram_critical"]:
            if self._should_send_alert("nvidia_vram_critical"):
                subject = f"CRITICAL: NVIDIA GPU VRAM at {nvidia_vram_percent:.1f}%"
                message = f"""
CRITICAL GPU ALERT

NVIDIA GPU VRAM usage has reached {nvidia_vram_percent:.1f}% ({stats["nvidia"]["vram_used_gb"]:.1f}GB / {stats["nvidia"]["vram_total_gb"]:.1f}GB)

This is above the critical threshold of {self.alert_thresholds["nvidia_vram_critical"]}%.
ComfyUI video generation may fail.

Time: {datetime.now().isoformat()}
                """

                if await self.send_alert_email(subject, message):
                    self.last_alert_times["nvidia_vram_critical"] = datetime.now()
                    alerts_sent.append("nvidia_vram_critical")

        return {
            "gpu_stats": stats,
            "alerts_sent": alerts_sent,
            "alert_thresholds": self.alert_thresholds,
            "timestamp": datetime.now().isoformat()
        }

    async def continuous_monitoring(self, check_interval: int = 300):
        """Continuous GPU monitoring loop"""
        logger.info(f"Starting GPU VRAM monitoring (check every {check_interval}s)")

        while True:
            try:
                result = await self.check_and_alert()
                if result["alerts_sent"]:
                    logger.warning(f"GPU alerts sent: {result['alerts_sent']}")

                await asyncio.sleep(check_interval)

            except Exception as e:
                logger.error(f"GPU monitoring error: {e}")
                await asyncio.sleep(60)  # Shorter retry interval on errors

# Global GPU monitor instance
gpu_monitor = GPUMonitor()