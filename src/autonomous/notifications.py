"""
Notification System for Echo Brain Autonomous Operations

Provides a simple notification system for tracking tasks that need human attention,
including approval requests and completed actions.
"""

import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncpg
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class NotificationManager:
    """
    Manages notifications for autonomous operations.

    Tracks approval requests, task completions, and other events
    that require human attention.
    """

    def __init__(self):
        """Initialize the NotificationManager with database configuration."""
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'echo_brain',
            'user': 'patrick',
            'password': os.environ.get('ECHO_BRAIN_DB_PASSWORD', os.getenv("TOWER_DB_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE"))
        }
        self._pool = None

    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(**self.db_config, min_size=2, max_size=10)

        async with self._pool.acquire() as connection:
            yield connection

    async def create_notification(self, notification_type: str, title: str,
                                message: str, task_id: Optional[int] = None,
                                metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Create a new notification.

        Args:
            notification_type: Type of notification
            title: Notification title
            message: Detailed message
            task_id: Associated task ID (optional)
            metadata: Additional metadata (optional)

        Returns:
            ID of created notification
        """
        try:
            async with self.get_connection() as conn:
                notification_id = await conn.fetchval("""
                    INSERT INTO autonomous_notifications
                    (notification_type, title, message, task_id, read, created_at)
                    VALUES ($1, $2, $3, $4, false, $5)
                    RETURNING id
                """, notification_type, title, message, task_id, datetime.now())

                logger.info(f"Created notification {notification_id}: {title}")
                return notification_id

        except Exception as e:
            logger.error(f"Failed to create notification: {e}")
            raise

    async def get_unread_notifications(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get all unread notifications.

        Args:
            limit: Maximum number of notifications to return

        Returns:
            List of notification dictionaries
        """
        try:
            async with self.get_connection() as conn:
                rows = await conn.fetch("""
                    SELECT id, notification_type, title, message, task_id,
                           read, created_at
                    FROM autonomous_notifications
                    WHERE read = false
                    ORDER BY created_at DESC
                    LIMIT $1
                """, limit)

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get unread notifications: {e}")
            return []

    async def get_all_notifications(self, limit: int = 100,
                                   notification_type: Optional[str] = None,
                                   start_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get notifications with optional filtering.

        Args:
            limit: Maximum number of notifications to return
            notification_type: Filter by type (optional)
            start_time: Filter from this time (optional)

        Returns:
            List of notification dictionaries
        """
        try:
            query_parts = ["SELECT * FROM autonomous_notifications WHERE 1=1"]
            params = []
            param_count = 0

            if notification_type:
                param_count += 1
                query_parts.append(f"AND notification_type = ${param_count}")
                params.append(notification_type)

            if start_time:
                param_count += 1
                query_parts.append(f"AND created_at >= ${param_count}")
                params.append(start_time)

            param_count += 1
            query = " ".join(query_parts) + f" ORDER BY created_at DESC LIMIT ${param_count}"
            params.append(limit)

            async with self.get_connection() as conn:
                rows = await conn.fetch(query, *params)
                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get notifications: {e}")
            return []

    async def mark_as_read(self, notification_id: int) -> bool:
        """
        Mark a notification as read.

        Args:
            notification_id: ID of notification to mark as read

        Returns:
            True if successful, False otherwise
        """
        try:
            async with self.get_connection() as conn:
                result = await conn.execute("""
                    UPDATE autonomous_notifications
                    SET read = true
                    WHERE id = $1
                """, notification_id)

                success = result.split()[-1] == '1'
                if success:
                    logger.info(f"Marked notification {notification_id} as read")
                return success

        except Exception as e:
            logger.error(f"Failed to mark notification {notification_id} as read: {e}")
            return False

    async def mark_all_as_read(self, notification_type: Optional[str] = None) -> int:
        """
        Mark all notifications as read.

        Args:
            notification_type: Only mark this type as read (optional)

        Returns:
            Number of notifications marked as read
        """
        try:
            async with self.get_connection() as conn:
                if notification_type:
                    result = await conn.execute("""
                        UPDATE autonomous_notifications
                        SET read = true
                        WHERE read = false AND notification_type = $1
                    """, notification_type)
                else:
                    result = await conn.execute("""
                        UPDATE autonomous_notifications
                        SET read = true
                        WHERE read = false
                    """)

                count = int(result.split()[-1])
                logger.info(f"Marked {count} notifications as read")
                return count

        except Exception as e:
            logger.error(f"Failed to mark notifications as read: {e}")
            return 0

    async def get_unread_count(self, notification_type: Optional[str] = None) -> int:
        """
        Get count of unread notifications.

        Args:
            notification_type: Count only this type (optional)

        Returns:
            Number of unread notifications
        """
        try:
            async with self.get_connection() as conn:
                if notification_type:
                    count = await conn.fetchval("""
                        SELECT COUNT(*) FROM autonomous_notifications
                        WHERE read = false AND notification_type = $1
                    """, notification_type)
                else:
                    count = await conn.fetchval("""
                        SELECT COUNT(*) FROM autonomous_notifications
                        WHERE read = false
                    """)

                return count or 0

        except Exception as e:
            logger.error(f"Failed to get unread count: {e}")
            return 0

    async def get_notifications_by_task(self, task_id: int) -> List[Dict[str, Any]]:
        """
        Get all notifications for a specific task.

        Args:
            task_id: Task ID to get notifications for

        Returns:
            List of notification dictionaries
        """
        try:
            async with self.get_connection() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM autonomous_notifications
                    WHERE task_id = $1
                    ORDER BY created_at DESC
                """, task_id)

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get notifications for task {task_id}: {e}")
            return []

    async def cleanup_old_notifications(self, days: int = 30) -> int:
        """
        Delete old read notifications.

        Args:
            days: Delete notifications older than this many days

        Returns:
            Number of notifications deleted
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            async with self.get_connection() as conn:
                result = await conn.execute("""
                    DELETE FROM autonomous_notifications
                    WHERE read = true AND created_at < $1
                """, cutoff_date)

                count = int(result.split()[-1])
                logger.info(f"Deleted {count} old notifications")
                return count

        except Exception as e:
            logger.error(f"Failed to cleanup old notifications: {e}")
            return 0

    async def cleanup(self):
        """Cleanup resources."""
        if self._pool:
            await self._pool.close()
        logger.info("NotificationManager cleaned up")


# Global instance for easy access
_notification_manager = None


def get_notification_manager() -> NotificationManager:
    """Get or create the global notification manager instance."""
    global _notification_manager
    if _notification_manager is None:
        _notification_manager = NotificationManager()
    return _notification_manager