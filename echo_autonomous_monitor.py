#!/usr/bin/env python3
"""
Echo Autonomous Monitoring & Self-Repair Service
Runs independently and monitors all Tower services including Telegram bot
"""

import asyncio
import aiohttp
import subprocess
import logging
import json
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EchoAutonomousMonitor:
    def __init__(self):
        self.services = {
            'echo': {
                'port': 8309,
                'health_endpoint': '/api/echo/health',
                'critical': True,
                'systemd_service': 'tower-echo-brain.service'
            },
            'telegram_bot': {
                'port': None,
                'process_name': 'tower_multiservice_bot.py',
                'log_file': '/opt/echo-telegram-bot/service_error.log',
                'critical': True,
                'systemd_service': 'tower-telegram-bot.service'
            },
            'kb': {
                'port': 8307,
                'health_endpoint': '/api/kb/health',
                'critical': True
            },
            'dashboard': {
                'port': 8080,
                'critical': False
            },
            'comfyui': {
                'port': 8188,
                'critical': False
            }
        }
        
        self.issues_detected = []
        self.repairs_attempted = []
        
    async def monitor_loop(self):
        """Main monitoring loop"""
        logger.info("🤖 Echo Autonomous Monitor starting...")
        
        while True:
            try:
                await self.check_all_services()
                await self.analyze_logs()
                await self.attempt_repairs()
                
                # Sleep 60 seconds between checks
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(60)
    
    async def check_all_services(self):
        """Check health of all services"""
        logger.info("🔍 Checking all services...")
        
        for name, config in self.services.items():
            try:
                healthy = await self.check_service_health(name, config)
                if not healthy and config['critical']:
                    self.issues_detected.append({
                        'timestamp': datetime.now().isoformat(),
                        'service': name,
                        'issue': 'service_unhealthy'
                    })
                    logger.warning(f"⚠️ Critical service {name} is unhealthy")
                    
            except Exception as e:
                logger.error(f"Error checking {name}: {e}")
    
    async def check_service_health(self, name: str, config: dict) -> bool:
        """Check if a service is healthy"""
        
        # Check systemd service if specified
        if 'systemd_service' in config:
            result = subprocess.run(
                ['systemctl', 'is-active', config['systemd_service']],
                capture_output=True, text=True
            )
            if result.stdout.strip() != 'active':
                logger.warning(f"Service {name} systemd status: {result.stdout.strip()}")
                return False
        
        # Check HTTP endpoint if specified
        if 'port' in config and config['port']:
            try:
                async with aiohttp.ClientSession() as session:
                    url = f"http://localhost:{config['port']}{config.get('health_endpoint', '/')}"
                    async with session.get(url, timeout=5) as resp:
                        if resp.status != 200:
                            return False
            except Exception as e:
                logger.error(f"HTTP check failed for {name}: {e}")
                return False
        
        return True
    
    async def analyze_logs(self):
        """Analyze log files for errors"""
        for name, config in self.services.items():
            if 'log_file' in config:
                try:
                    log_path = Path(config['log_file'])
                    if log_path.exists():
                        # Read last 50 lines
                        result = subprocess.run(
                            ['tail', '-50', str(log_path)],
                            capture_output=True, text=True
                        )
                        
                        # Look for common error patterns
                        if 'ERROR' in result.stdout or 'Field required' in result.stdout:
                            self.issues_detected.append({
                                'timestamp': datetime.now().isoformat(),
                                'service': name,
                                'issue': 'errors_in_logs',
                                'log_snippet': result.stdout[-500:]
                            })
                            logger.warning(f"⚠️ Errors detected in {name} logs")
                            
                except Exception as e:
                    logger.error(f"Error analyzing logs for {name}: {e}")
    
    async def attempt_repairs(self):
        """Attempt to repair detected issues"""
        if not self.issues_detected:
            return
        
        logger.info(f"🔧 Attempting to repair {len(self.issues_detected)} issues...")
        
        for issue in self.issues_detected[:]:  # Copy list to allow removal
            try:
                service_config = self.services.get(issue['service'])
                if not service_config:
                    continue
                
                if 'systemd_service' in service_config:
                    logger.info(f"🔄 Restarting {issue['service']}...")
                    result = subprocess.run(
                        ['sudo', 'systemctl', 'restart', service_config['systemd_service']],
                        capture_output=True, text=True
                    )
                    
                    if result.returncode == 0:
                        logger.info(f"✅ Successfully restarted {issue['service']}")
                        self.repairs_attempted.append({
                            'timestamp': datetime.now().isoformat(),
                            'service': issue['service'],
                            'action': 'restart',
                            'success': True
                        })
                        self.issues_detected.remove(issue)
                    else:
                        logger.error(f"❌ Failed to restart {issue['service']}: {result.stderr}")
                        
            except Exception as e:
                logger.error(f"Repair attempt failed for {issue['service']}: {e}")
        
        # Report status
        logger.info(f"📊 Repairs attempted: {len(self.repairs_attempted)}, Issues remaining: {len(self.issues_detected)}")

async def main():
    monitor = EchoAutonomousMonitor()
    await monitor.monitor_loop()

if __name__ == '__main__':
    asyncio.run(main())
