#!/usr/bin/env python3
"""
Reminder Worker — checks for due reminders and sends via NotificationService.
Registered in main.py startup_event() at 1-minute interval.
"""

import logging
import os
from datetime import datetime

import asyncpg

logger = logging.getLogger(__name__)


class ReminderWorker:
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable required")

    async def run_cycle(self):
        """Check for due reminders and send them."""
        conn = await asyncpg.connect(self.db_url)
        try:
            rows = await conn.fetch(
                """SELECT id, title, body, metadata
                   FROM notifications
                   WHERE status = 'pending'
                     AND scheduled_for IS NOT NULL
                     AND scheduled_for <= NOW()
                     AND category = 'reminder'
                   ORDER BY scheduled_for ASC
                   LIMIT 10"""
            )

            if not rows:
                return

            logger.info(f"Reminder worker: {len(rows)} due reminder(s)")

            from src.services.notification_service import (
                get_notification_service,
                NotificationType,
                NotificationChannel,
            )

            service = await get_notification_service()
            if not service:
                logger.warning("Reminder worker: notification service unavailable")
                return

            channel_map = {
                "telegram": NotificationChannel.TELEGRAM,
                "ntfy": NotificationChannel.NTFY,
                "email": NotificationChannel.EMAIL,
            }

            for row in rows:
                reminder_id = row["id"]
                title = row["title"] or "Reminder"
                body = row["body"] or ""
                metadata = row["metadata"] or {}
                channel_str = metadata.get("channel", "telegram")
                channel = channel_map.get(channel_str, NotificationChannel.TELEGRAM)

                try:
                    results = await service.send_notification(
                        message=body,
                        title=title,
                        notification_type=NotificationType.REMINDER,
                        channels=[channel],
                    )
                    sent = any(results.values())
                    new_status = "delivered" if sent else "failed"
                    await conn.execute(
                        """UPDATE notifications
                           SET status = $1, delivered_at = $2
                           WHERE id = $3""",
                        new_status,
                        datetime.now() if sent else None,
                        reminder_id,
                    )
                    logger.info(f"Reminder {reminder_id}: {new_status}")
                except Exception as e:
                    logger.error(f"Reminder {reminder_id} send failed: {e}")
                    await conn.execute(
                        "UPDATE notifications SET status = 'failed' WHERE id = $1",
                        reminder_id,
                    )
        finally:
            await conn.close()
