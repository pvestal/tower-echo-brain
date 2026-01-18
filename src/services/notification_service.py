#!/usr/bin/env python3
"""
Unified Notification Service for Echo Brain
Coordinates notifications across multiple channels: ntfy, Telegram, Email
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class NotificationType(Enum):
    """Types of notifications"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"
    ALERT = "alert"
    REMINDER = "reminder"

class NotificationChannel(Enum):
    """Available notification channels"""
    NTFY = "ntfy"
    TELEGRAM = "telegram"
    EMAIL = "email"
    ALL = "all"

class NotificationService:
    """Unified notification service for Echo Brain"""

    def __init__(self):
        self.ntfy_client = None
        self.telegram_client = None
        self.email_client = None
        self.initialized = False

        # Channel availability
        self.channels_available = {
            NotificationChannel.NTFY: False,
            NotificationChannel.TELEGRAM: False,
            NotificationChannel.EMAIL: False
        }

    async def initialize(self) -> bool:
        """Initialize all notification clients"""
        try:
            logger.info("📢 Initializing notification service...")

            # Initialize ntfy
            try:
                from src.integrations.ntfy_client import get_ntfy_client
                self.ntfy_client = await get_ntfy_client()
                self.channels_available[NotificationChannel.NTFY] = self.ntfy_client is not None
                if self.ntfy_client:
                    logger.info("✅ ntfy channel available")
            except Exception as e:
                logger.warning(f"⚠️ ntfy channel unavailable: {e}")

            # Initialize Telegram
            try:
                from src.integrations.telegram_client import TelegramClient
                self.telegram_client = TelegramClient()
                telegram_ok = await self.telegram_client.initialize()
                self.channels_available[NotificationChannel.TELEGRAM] = telegram_ok
                if telegram_ok:
                    logger.info("✅ Telegram channel available")
            except Exception as e:
                logger.warning(f"⚠️ Telegram channel unavailable: {e}")

            # Initialize Email
            try:
                from src.integrations.email_client import EmailClient
                self.email_client = EmailClient()
                email_ok = await self.email_client.initialize() if hasattr(self.email_client, 'initialize') else True
                self.channels_available[NotificationChannel.EMAIL] = email_ok
                if email_ok:
                    logger.info("✅ Email channel available")
            except Exception as e:
                logger.warning(f"⚠️ Email channel unavailable: {e}")

            available_count = sum(self.channels_available.values())
            self.initialized = available_count > 0

            if self.initialized:
                logger.info(f"✅ Notification service initialized with {available_count} channels")
            else:
                logger.warning("⚠️ No notification channels available")

            return self.initialized

        except Exception as e:
            logger.error(f"❌ Failed to initialize notification service: {e}")
            return False

    async def send_notification(
        self,
        message: str,
        title: Optional[str] = None,
        notification_type: NotificationType = NotificationType.INFO,
        channels: Union[NotificationChannel, List[NotificationChannel]] = NotificationChannel.ALL,
        priority: Optional[int] = None,
        schedule: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, bool]:
        """
        Send notification across specified channels

        Args:
            message: Notification message
            title: Optional notification title
            notification_type: Type of notification for proper formatting
            channels: Channel(s) to send notification to
            priority: Notification priority (1-5)
            schedule: Future datetime to schedule notification
            metadata: Additional metadata for specific channels

        Returns:
            Dict mapping channel names to success status
        """
        if not self.initialized:
            logger.error("📢 Notification service not initialized")
            return {}

        # Normalize channels to list
        if isinstance(channels, NotificationChannel):
            if channels == NotificationChannel.ALL:
                channel_list = [ch for ch, available in self.channels_available.items() if available]
            else:
                channel_list = [channels] if self.channels_available.get(channels, False) else []
        else:
            channel_list = [ch for ch in channels if self.channels_available.get(ch, False)]

        if not channel_list:
            logger.warning("📢 No available channels for notification")
            return {}

        results = {}
        metadata = metadata or {}

        # Send to each channel
        for channel in channel_list:
            try:
                if channel == NotificationChannel.NTFY:
                    results[channel.value] = await self._send_ntfy_notification(
                        message, title, notification_type, priority, schedule, metadata
                    )
                elif channel == NotificationChannel.TELEGRAM:
                    results[channel.value] = await self._send_telegram_notification(
                        message, title, notification_type, metadata
                    )
                elif channel == NotificationChannel.EMAIL:
                    results[channel.value] = await self._send_email_notification(
                        message, title, notification_type, metadata
                    )
            except Exception as e:
                logger.error(f"❌ Failed to send {channel.value} notification: {e}")
                results[channel.value] = False

        success_count = sum(results.values())
        logger.info(f"📢 Notification sent to {success_count}/{len(channel_list)} channels")

        return results

    async def _send_ntfy_notification(
        self,
        message: str,
        title: Optional[str],
        notification_type: NotificationType,
        priority: Optional[int],
        schedule: Optional[datetime],
        metadata: Dict[str, Any]
    ) -> bool:
        """Send notification via ntfy"""
        if not self.ntfy_client:
            return False

        try:
            # Map notification type to ntfy priority and tags
            type_mapping = {
                NotificationType.INFO: (3, ["information_source"]),
                NotificationType.WARNING: (4, ["warning"]),
                NotificationType.ERROR: (5, ["rotating_light", "exclamation"]),
                NotificationType.SUCCESS: (3, ["white_check_mark"]),
                NotificationType.ALERT: (5, ["rotating_light"]),
                NotificationType.REMINDER: (3, ["bell"])
            }

            default_priority, default_tags = type_mapping.get(notification_type, (3, []))
            final_priority = priority or default_priority
            tags = metadata.get('tags', default_tags)

            from src.integrations.ntfy_client import Priority
            ntfy_priority = Priority(final_priority)

            return await self.ntfy_client.send_notification(
                message=message,
                title=title or f"Echo Brain {notification_type.value.title()}",
                priority=ntfy_priority,
                tags=tags,
                click_url=metadata.get('url'),
                schedule=schedule
            )

        except Exception as e:
            logger.error(f"❌ ntfy notification error: {e}")
            return False

    async def _send_telegram_notification(
        self,
        message: str,
        title: Optional[str],
        notification_type: NotificationType,
        metadata: Dict[str, Any]
    ) -> bool:
        """Send notification via Telegram"""
        if not self.telegram_client or not self.telegram_client.is_configured:
            return False

        try:
            # Format message with title if provided
            full_message = message
            if title:
                full_message = f"*{title}*\n\n{message}"

            # Add emoji based on notification type
            type_emojis = {
                NotificationType.INFO: "ℹ️",
                NotificationType.WARNING: "⚠️",
                NotificationType.ERROR: "❌",
                NotificationType.SUCCESS: "✅",
                NotificationType.ALERT: "🚨",
                NotificationType.REMINDER: "🔔"
            }

            emoji = type_emojis.get(notification_type, "📢")
            final_message = f"{emoji} {full_message}"

            return await self.telegram_client.send_message(
                message=final_message,
                parse_mode="Markdown",
                disable_web_page_preview=metadata.get('disable_preview', True)
            )

        except Exception as e:
            logger.error(f"❌ Telegram notification error: {e}")
            return False

    async def _send_email_notification(
        self,
        message: str,
        title: Optional[str],
        notification_type: NotificationType,
        metadata: Dict[str, Any]
    ) -> bool:
        """Send notification via Email"""
        if not self.email_client:
            return False

        try:
            subject = title or f"Echo Brain {notification_type.value.title()}"

            # Add notification type prefix to subject
            type_prefixes = {
                NotificationType.INFO: "[INFO]",
                NotificationType.WARNING: "[WARNING]",
                NotificationType.ERROR: "[ERROR]",
                NotificationType.SUCCESS: "[SUCCESS]",
                NotificationType.ALERT: "[ALERT]",
                NotificationType.REMINDER: "[REMINDER]"
            }

            prefix = type_prefixes.get(notification_type, "[NOTIFICATION]")
            final_subject = f"{prefix} {subject}"

            # Format message with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
            final_message = f"{message}\n\n---\nSent by Echo Brain at {timestamp}"

            return await self.email_client.send_email(
                to_email=metadata.get('to_email'),
                subject=final_subject,
                message=final_message,
                html_message=metadata.get('html_content')
            )

        except Exception as e:
            logger.error(f"❌ Email notification error: {e}")
            return False

    async def send_system_alert(self, message: str, title: str = "System Alert") -> Dict[str, bool]:
        """Send high-priority system alert"""
        return await self.send_notification(
            message=message,
            title=title,
            notification_type=NotificationType.ALERT,
            channels=NotificationChannel.ALL,
            priority=5
        )

    async def send_error_alert(self, error: str, context: str = "") -> Dict[str, bool]:
        """Send error notification"""
        message = f"Error: {error}"
        if context:
            message += f"\nContext: {context}"

        return await self.send_notification(
            message=message,
            title="Echo Brain Error",
            notification_type=NotificationType.ERROR,
            priority=5
        )

    async def send_success_message(self, message: str, title: str = "Task Completed") -> Dict[str, bool]:
        """Send success notification"""
        return await self.send_notification(
            message=message,
            title=title,
            notification_type=NotificationType.SUCCESS
        )

    async def send_reminder(self, message: str, when: datetime, title: str = "Reminder") -> Dict[str, bool]:
        """Schedule a reminder notification"""
        return await self.send_notification(
            message=message,
            title=title,
            notification_type=NotificationType.REMINDER,
            schedule=when
        )

    def get_status(self) -> Dict[str, Any]:
        """Get notification service status"""
        return {
            "initialized": self.initialized,
            "available_channels": {
                channel.value: available
                for channel, available in self.channels_available.items()
            },
            "total_channels": sum(self.channels_available.values())
        }

# Global notification service instance
_notification_service: Optional[NotificationService] = None

async def get_notification_service() -> Optional[NotificationService]:
    """Get or create notification service instance"""
    global _notification_service

    if _notification_service is None:
        _notification_service = NotificationService()
        initialized = await _notification_service.initialize()
        if not initialized:
            logger.warning("📢 Notification service not available")
            return None

    return _notification_service

# Convenience functions for common notifications

async def notify_info(message: str, title: str = "Information") -> bool:
    """Send info notification to all channels"""
    service = await get_notification_service()
    if service:
        results = await service.send_notification(message, title, NotificationType.INFO)
        return any(results.values())
    return False

async def notify_warning(message: str, title: str = "Warning") -> bool:
    """Send warning notification"""
    service = await get_notification_service()
    if service:
        results = await service.send_notification(message, title, NotificationType.WARNING)
        return any(results.values())
    return False

async def notify_error(message: str, title: str = "Error") -> bool:
    """Send error notification"""
    service = await get_notification_service()
    if service:
        results = await service.send_notification(message, title, NotificationType.ERROR)
        return any(results.values())
    return False

async def notify_success(message: str, title: str = "Success") -> bool:
    """Send success notification"""
    service = await get_notification_service()
    if service:
        results = await service.send_notification(message, title, NotificationType.SUCCESS)
        return any(results.values())
    return False

# CLI testing function
async def test_notification_service():
    """Test notification service"""
    print("📢 Testing notification service...")

    service = await get_notification_service()
    if service:
        status = service.get_status()
        print(f"✅ Notification service status: {status}")

        # Test different notification types
        test_notifications = [
            ("Test info message", "Test Info", NotificationType.INFO),
            ("Test warning message", "Test Warning", NotificationType.WARNING),
            ("Test success message", "Test Success", NotificationType.SUCCESS)
        ]

        for message, title, ntype in test_notifications:
            results = await service.send_notification(message, title, ntype)
            print(f"📢 {ntype.value} notification results: {results}")
            await asyncio.sleep(1)  # Rate limiting
    else:
        print("❌ Notification service initialization failed")

if __name__ == "__main__":
    asyncio.run(test_notification_service())