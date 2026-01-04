#!/opt/tower-echo-brain/venv/bin/python
"""
24-Hour Validation Test for Echo Brain Limited Autonomy Mode
Runs comprehensive checks every hour and enforces fail conditions
"""

import asyncio
import json
import psutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
import aiohttp
import asyncpg
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/tower-echo-brain/logs/24h_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sys.path.insert(0, '/opt/tower-echo-brain')
from config import AUTONOMY_CONFIG, FAIL_CONDITIONS, SUCCESS_METRICS, DATABASE_CONFIG

class ValidationMonitor:
    """24-hour validation monitoring system"""

    def __init__(self):
        self.start_time = datetime.now()
        self.test_results = []
        self.fail_conditions_triggered = []
        self.metrics_history = []
        self.alert_file = Path('/opt/tower-echo-brain/data/alerts/validation_alerts.json')
        self.alert_file.parent.mkdir(parents=True, exist_ok=True)

        # Baseline metrics
        self.baseline_metrics = None
        self.db_conn = None

    async def initialize(self):
        """Initialize monitoring and capture baseline"""
        logger.info("=== 24-HOUR VALIDATION TEST STARTED ===")
        logger.info(f"Deployment Mode: {AUTONOMY_CONFIG['mode']}")
        logger.info(f"Start Time: {self.start_time}")

        # Capture baseline metrics
        self.baseline_metrics = await self.capture_metrics()
        logger.info(f"Baseline captured: CPU={self.baseline_metrics['cpu_percent']:.1f}%, "
                   f"RAM={self.baseline_metrics['ram_gb']:.1f}GB, "
                   f"GPU={self.baseline_metrics['gpu_vram_gb']:.1f}GB")

        # Initialize database connection
        try:
            # Remove extra fields that asyncpg doesn't accept
            db_config = {
                'host': DATABASE_CONFIG['host'],
                'port': DATABASE_CONFIG['port'],
                'user': DATABASE_CONFIG['user'],
                'password': DATABASE_CONFIG['password'],
                'database': DATABASE_CONFIG['database']
            }
            self.db_conn = await asyncpg.connect(**db_config)
            await self.create_monitoring_table()
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")

    async def create_monitoring_table(self):
        """Create table for monitoring data"""
        await self.db_conn.execute('''
            CREATE TABLE IF NOT EXISTS validation_metrics (
                timestamp TIMESTAMP PRIMARY KEY,
                cpu_percent FLOAT,
                ram_gb FLOAT,
                gpu_vram_gb FLOAT,
                disk_percent FLOAT,
                tasks_completed INTEGER,
                tasks_failed INTEGER,
                anime_generated INTEGER,
                errors_logged INTEGER,
                services_running JSONB,
                alerts JSONB
            )
        ''')

    async def capture_metrics(self) -> Dict[str, Any]:
        """Capture current system metrics"""
        metrics = {}

        # CPU
        metrics['cpu_percent'] = psutil.cpu_percent(interval=1)

        # Memory
        memory = psutil.virtual_memory()
        metrics['ram_gb'] = memory.used / (1024**3)
        metrics['ram_percent'] = memory.percent

        # Disk
        disk = psutil.disk_usage('/opt/tower-echo-brain')
        metrics['disk_gb'] = disk.used / (1024**3)
        metrics['disk_percent'] = disk.percent

        # GPU
        try:
            result = subprocess.run(
                "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader",
                shell=True,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                metrics['gpu_vram_gb'] = float(parts[0].replace(' MiB', '')) / 1024
                metrics['gpu_vram_total_gb'] = float(parts[1].replace(' MiB', '')) / 1024
                metrics['gpu_utilization'] = float(parts[2].replace(' %', ''))
                metrics['gpu_vram_percent'] = (metrics['gpu_vram_gb'] / metrics['gpu_vram_total_gb']) * 100
        except:
            metrics['gpu_vram_gb'] = 0
            metrics['gpu_vram_percent'] = 0

        metrics['timestamp'] = datetime.now()

        return metrics

    async def check_services(self) -> Dict[str, bool]:
        """Check if all required services are running"""
        services = {}

        # Echo Brain
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8309/api/echo/health', timeout=5) as resp:
                    services['echo_brain'] = resp.status == 200
        except:
            services['echo_brain'] = False

        # PostgreSQL
        services['postgresql'] = self.db_conn is not None and not self.db_conn.is_closed()

        # Redis
        try:
            result = subprocess.run("redis-cli ping", shell=True, capture_output=True, timeout=2)
            services['redis'] = result.returncode == 0
        except:
            services['redis'] = False

        # Qdrant
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:6333/collections', timeout=5) as resp:
                    services['qdrant'] = resp.status == 200
        except:
            services['qdrant'] = False

        # ComfyUI
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8188/system_stats', timeout=5) as resp:
                    services['comfyui'] = resp.status == 200
        except:
            services['comfyui'] = False

        return services

    async def check_task_queue(self) -> Dict[str, int]:
        """Check task queue status"""
        stats = {'pending': 0, 'running': 0, 'completed': 0, 'failed': 0}

        try:
            rows = await self.db_conn.fetch('''
                SELECT status, COUNT(*) as count
                FROM task_queue
                WHERE created_at > $1
                GROUP BY status
            ''', self.start_time)

            for row in rows:
                stats[row['status']] = row['count']
        except Exception as e:
            logger.error(f"Failed to check task queue: {e}")

        return stats

    async def check_generated_content(self) -> Dict[str, int]:
        """Check for generated content"""
        content = {'anime_images': 0, 'reports': 0}

        # Check anime outputs
        validation_path = Path('/opt/tower-echo-brain/data/outputs/validation/')
        if validation_path.exists():
            content['anime_images'] = len(list(validation_path.glob('*.png')))

        # Check code reports
        reports_path = Path('/opt/tower-echo-brain/data/reports/')
        if reports_path.exists():
            content['reports'] = len(list(reports_path.glob('*.json')))

        return content

    async def analyze_logs(self) -> Dict[str, int]:
        """Analyze logs for errors"""
        errors = {'error_count': 0, 'critical_count': 0}

        log_file = Path('/opt/tower-echo-brain/logs/autonomous_loop.log')
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        if 'ERROR' in line:
                            errors['error_count'] += 1
                        if 'CRITICAL' in line:
                            errors['critical_count'] += 1
            except:
                pass

        return errors

    async def run_hourly_check(self) -> Dict[str, Any]:
        """Run comprehensive hourly validation check"""
        logger.info(f"\n=== HOURLY CHECK {len(self.test_results) + 1} ===")

        check_result = {
            'hour': len(self.test_results) + 1,
            'timestamp': datetime.now(),
            'metrics': await self.capture_metrics(),
            'services': await self.check_services(),
            'task_queue': await self.check_task_queue(),
            'content': await self.check_generated_content(),
            'logs': await self.analyze_logs(),
            'alerts': [],
            'passed': True
        }

        # Check fail conditions
        metrics = check_result['metrics']

        # GPU VRAM check
        if metrics['gpu_vram_percent'] > FAIL_CONDITIONS['gpu_vram_percent']:
            alert = f"GPU VRAM exceeded: {metrics['gpu_vram_percent']:.1f}% > {FAIL_CONDITIONS['gpu_vram_percent']}%"
            check_result['alerts'].append(alert)
            check_result['passed'] = False
            logger.critical(alert)

        # Disk check
        if metrics['disk_percent'] > FAIL_CONDITIONS['disk_percent']:
            alert = f"Disk usage exceeded: {metrics['disk_percent']:.1f}% > {FAIL_CONDITIONS['disk_percent']}%"
            check_result['alerts'].append(alert)
            check_result['passed'] = False
            logger.critical(alert)

        # CPU check (needs to be sustained)
        if metrics['cpu_percent'] > FAIL_CONDITIONS['cpu_percent_sustained']:
            # Check if CPU has been high for multiple checks
            high_cpu_count = sum(1 for r in self.test_results[-4:]
                               if r['metrics']['cpu_percent'] > FAIL_CONDITIONS['cpu_percent_sustained'])
            if high_cpu_count >= 4:  # 4+ consecutive high readings
                alert = f"CPU sustained high: {metrics['cpu_percent']:.1f}% for {high_cpu_count} checks"
                check_result['alerts'].append(alert)
                check_result['passed'] = False
                logger.critical(alert)

        # Service check
        for service, running in check_result['services'].items():
            if not running and service in ['echo_brain', 'postgresql']:
                alert = f"Critical service down: {service}"
                check_result['alerts'].append(alert)
                check_result['passed'] = False
                logger.critical(alert)

        # Failed tasks check
        if check_result['task_queue']['failed'] > FAIL_CONDITIONS['failed_task_threshold']:
            alert = f"Failed tasks exceeded: {check_result['task_queue']['failed']} > {FAIL_CONDITIONS['failed_task_threshold']}"
            check_result['alerts'].append(alert)
            check_result['passed'] = False
            logger.critical(alert)

        # Log results
        logger.info(f"Metrics: CPU={metrics['cpu_percent']:.1f}%, RAM={metrics['ram_gb']:.1f}GB, "
                   f"GPU={metrics['gpu_vram_gb']:.1f}GB, Disk={metrics['disk_percent']:.1f}%")
        logger.info(f"Services: {check_result['services']}")
        logger.info(f"Tasks: Completed={check_result['task_queue']['completed']}, "
                   f"Failed={check_result['task_queue']['failed']}")
        logger.info(f"Content: Anime={check_result['content']['anime_images']}, "
                   f"Reports={check_result['content']['reports']}")
        logger.info(f"Status: {'PASSED' if check_result['passed'] else 'FAILED'}")

        # Save to database
        if self.db_conn:
            try:
                await self.db_conn.execute('''
                    INSERT INTO validation_metrics
                    (timestamp, cpu_percent, ram_gb, gpu_vram_gb, disk_percent,
                     tasks_completed, tasks_failed, anime_generated, errors_logged,
                     services_running, alerts)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ''', metrics['timestamp'], metrics['cpu_percent'], metrics['ram_gb'],
                    metrics['gpu_vram_gb'], metrics['disk_percent'],
                    check_result['task_queue']['completed'], check_result['task_queue']['failed'],
                    check_result['content']['anime_images'],
                    check_result['logs']['error_count'] + check_result['logs']['critical_count'],
                    json.dumps(check_result['services']), json.dumps(check_result['alerts']))
            except Exception as e:
                logger.error(f"Failed to save metrics: {e}")

        return check_result

    def should_auto_shutdown(self, check_result: Dict[str, Any]) -> bool:
        """Determine if auto-shutdown should be triggered"""
        if not AUTONOMY_CONFIG['safety_features']['auto_shutdown_on_exceed']:
            return False

        return not check_result['passed'] and len(check_result['alerts']) > 0

    async def save_alert(self, alert: str):
        """Save alert to file"""
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'alert': alert,
            'metrics': self.metrics_history[-1] if self.metrics_history else None
        }

        alerts = []
        if self.alert_file.exists():
            with open(self.alert_file, 'r') as f:
                alerts = json.load(f)

        alerts.append(alert_data)

        with open(self.alert_file, 'w') as f:
            json.dump(alerts, f, indent=2, default=str)

    async def run_validation_loop(self):
        """Main validation loop - runs for 24 hours"""
        await self.initialize()

        end_time = self.start_time + timedelta(hours=24)
        check_interval = 3600  # 1 hour in seconds

        while datetime.now() < end_time:
            # Run hourly check
            check_result = await self.run_hourly_check()
            self.test_results.append(check_result)
            self.metrics_history.append(check_result['metrics'])

            # Save alerts
            for alert in check_result['alerts']:
                await self.save_alert(alert)

            # Check for auto-shutdown
            if self.should_auto_shutdown(check_result):
                logger.critical("AUTO-SHUTDOWN TRIGGERED DUE TO FAIL CONDITIONS")
                await self.generate_final_report()
                sys.exit(1)

            # Wait for next hour
            if datetime.now() < end_time:
                logger.info(f"Next check in {check_interval/60:.0f} minutes...")
                await asyncio.sleep(check_interval)

        # Generate final report
        await self.generate_final_report()

    async def generate_final_report(self):
        """Generate final 24-hour validation report"""
        logger.info("\n" + "=" * 60)
        logger.info("24-HOUR VALIDATION COMPLETE")
        logger.info("=" * 60)

        duration = datetime.now() - self.start_time
        total_checks = len(self.test_results)
        passed_checks = sum(1 for r in self.test_results if r['passed'])

        # Calculate totals
        total_tasks_completed = sum(r['task_queue']['completed'] for r in self.test_results)
        total_tasks_failed = sum(r['task_queue']['failed'] for r in self.test_results)
        total_anime_generated = max(r['content']['anime_images'] for r in self.test_results) if self.test_results else 0
        total_reports_generated = max(r['content']['reports'] for r in self.test_results) if self.test_results else 0

        # Check success metrics
        uptime_percent = (passed_checks / total_checks * 100) if total_checks > 0 else 0

        success = (
            uptime_percent >= SUCCESS_METRICS['required_uptime_percent'] and
            total_tasks_completed >= SUCCESS_METRICS['min_tasks_completed'] and
            total_anime_generated >= SUCCESS_METRICS['min_anime_generated'] and
            total_tasks_failed == 0
        )

        report = {
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_hours': duration.total_seconds() / 3600,
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'uptime_percent': uptime_percent,
            'total_tasks_completed': total_tasks_completed,
            'total_tasks_failed': total_tasks_failed,
            'total_anime_generated': total_anime_generated,
            'total_reports_generated': total_reports_generated,
            'baseline_metrics': self.baseline_metrics,
            'final_metrics': self.metrics_history[-1] if self.metrics_history else None,
            'alerts_triggered': sum(len(r['alerts']) for r in self.test_results),
            'success': success,
            'verdict': 'PASSED' if success else 'FAILED'
        }

        # Save report
        report_file = Path('/opt/tower-echo-brain/data/24h_validation_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Print summary
        logger.info(f"Duration: {duration.total_seconds()/3600:.1f} hours")
        logger.info(f"Checks Passed: {passed_checks}/{total_checks} ({uptime_percent:.1f}%)")
        logger.info(f"Tasks Completed: {total_tasks_completed}")
        logger.info(f"Tasks Failed: {total_tasks_failed}")
        logger.info(f"Anime Generated: {total_anime_generated}")
        logger.info(f"Reports Generated: {total_reports_generated}")
        logger.info(f"Alerts Triggered: {report['alerts_triggered']}")
        logger.info(f"\nFINAL VERDICT: {report['verdict']}")

        if report['success']:
            logger.info("✅ Echo Brain passed 24-hour validation in Limited Autonomy mode!")
        else:
            logger.error("❌ Echo Brain failed 24-hour validation")

        # Close database connection
        if self.db_conn:
            await self.db_conn.close()

async def main():
    """Main entry point"""
    monitor = ValidationMonitor()
    await monitor.run_validation_loop()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)