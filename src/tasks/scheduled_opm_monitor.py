#!/usr/bin/env python3
"""
Scheduled OPM and HR Training Monitor
Runs daily checks and sends reports
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from src.monitors.opm_hr_monitor import opm_monitor
from src.utils.email_manager import email_manager

logger = logging.getLogger(__name__)

class ScheduledOPMMonitor:
    """Scheduled task for OPM and HR training monitoring"""

    def __init__(self):
        self.enabled = True
        self.check_interval = 86400  # 24 hours
        self.last_run: Optional[datetime] = None

    async def run_daily_check(self):
        """Run daily OPM/HR compliance check"""

        try:
            logger.info("🏛️ Starting daily OPM/HR compliance check...")

            # Generate compliance report
            report = await opm_monitor.generate_compliance_report()

            # Check for critical updates
            results = await opm_monitor.run_monitoring_cycle()

            # Determine if email is needed
            send_email = False
            subject = "📋 Federal HR Compliance Report"

            if results['actions_needed']:
                send_email = True
                subject = f"⚠️ Federal HR Compliance - {len(results['actions_needed'])} Actions Required"
            elif results['opm_updates'] or results['cfr_updates']:
                send_email = True
                subject = "📰 Federal HR Updates Available"

            # Send report if needed
            if send_email:
                success = await email_manager.send_email(subject, report)
                if success:
                    logger.info(f"✅ OPM/HR compliance report sent: {subject}")
                else:
                    logger.warning("⚠️ Could not send compliance report")

            # Log completion
            self.last_run = datetime.now()
            logger.info(f"✅ Daily OPM/HR check complete - {len(results['actions_needed'])} actions needed")

            return results

        except Exception as e:
            logger.error(f"Daily OPM/HR check failed: {e}")
            return None

    async def start_monitoring(self):
        """Start the monitoring loop"""

        logger.info("🚀 Starting OPM/HR monitoring service...")

        while self.enabled:
            try:
                # Check if it's time for daily check
                now = datetime.now()

                if self.last_run is None or (now - self.last_run).total_seconds() >= self.check_interval:
                    await self.run_daily_check()
                else:
                    # Calculate time until next check
                    time_until_next = self.check_interval - (now - self.last_run).total_seconds()
                    logger.debug(f"Next OPM check in {time_until_next/3600:.1f} hours")

                # Sleep for 1 hour then check again
                await asyncio.sleep(3600)

            except asyncio.CancelledError:
                logger.info("OPM monitoring cancelled")
                break
            except Exception as e:
                logger.error(f"OPM monitoring error: {e}")
                await asyncio.sleep(3600)

        logger.info("OPM/HR monitoring stopped")

    async def run_immediate_check(self):
        """Run an immediate compliance check"""
        return await self.run_daily_check()

# Global instance
scheduled_opm_monitor = ScheduledOPMMonitor()