#!/usr/bin/env python3
"""
Telegram Integration for Echo Brain
Provides instant messaging and notification capabilities
Bot: @PatricksEchobot
"""

import aiohttp
import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class TelegramClient:
    """Telegram bot client for Echo Brain notifications"""

    def __init__(self):
        # Bot token for @PatricksEchobot
        self.bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")  # Patrick's chat ID
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.bot_username = "@PatricksEchobot"
        self.is_configured = False

    async def initialize(self):
        """Initialize and verify Telegram bot connection"""
        if not self.bot_token:
            logger.warning("⚠️ Telegram bot token not configured")
            return False

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/getMe") as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("ok"):
                            bot_info = data.get("result", {})
                            logger.info(f"✅ Connected to Telegram bot: @{bot_info.get('username')}")
                            self.is_configured = True

                            # Get updates to find chat ID if not set
                            if not self.chat_id:
                                await self._find_chat_id()
                            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Telegram: {e}")
        return False

    async def _find_chat_id(self):
        """Find Patrick's chat ID from recent messages"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/getUpdates") as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("ok"):
                            updates = data.get("result", [])
                            for update in updates:
                                message = update.get("message", {})
                                chat = message.get("chat", {})
                                if chat:
                                    self.chat_id = str(chat.get("id"))
                                    logger.info(f"📱 Found chat ID: {self.chat_id}")
                                    return
        except Exception as e:
            logger.error(f"Failed to find chat ID: {e}")

    async def send_message(self, text: str, chat_id: Optional[str] = None,
                          parse_mode: str = "Markdown") -> bool:
        """Send a message via Telegram"""
        if not self.is_configured:
            await self.initialize()

        if not self.is_configured:
            logger.warning("Telegram not configured, message not sent")
            return False

        target_chat = chat_id or self.chat_id
        if not target_chat:
            logger.error("No chat ID available")
            return False

        try:
            payload = {
                "chat_id": target_chat,
                "text": text,
                "parse_mode": parse_mode
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/sendMessage",
                    json=payload
                ) as response:
                    if response.status == 200:
                        logger.info(f"✅ Telegram message sent")
                        return True
                    else:
                        logger.error(f"Failed to send message: {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False

    async def send_notification(self, title: str, body: str,
                               priority: str = "normal") -> bool:
        """Send a formatted notification"""
        # Format message with emoji based on priority
        emoji = {
            "urgent": "🚨",
            "high": "⚠️",
            "normal": "ℹ️",
            "low": "💡"
        }.get(priority, "📢")

        message = f"{emoji} *{title}*\n\n{body}"

        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        message += f"\n\n_Sent at {timestamp}_"

        return await self.send_message(message)

    async def send_task_update(self, task_name: str, status: str,
                              details: Optional[str] = None) -> bool:
        """Send task status update"""
        status_emoji = {
            "started": "🔄",
            "completed": "✅",
            "failed": "❌",
            "warning": "⚠️",
            "info": "ℹ️"
        }.get(status, "📝")

        message = f"{status_emoji} *Task: {task_name}*\n"
        message += f"Status: {status}\n"

        if details:
            message += f"\nDetails:\n{details}"

        return await self.send_message(message)

    async def send_daily_report(self, stats: Dict[str, Any]) -> bool:
        """Send daily Echo Brain activity report"""
        message = "📊 *Echo Brain Daily Report*\n"
        message += "=" * 30 + "\n\n"

        # Tasks processed
        message += "📋 *Tasks Processed*\n"
        message += f"• Completed: {stats.get('tasks_completed', 0)}\n"
        message += f"• Failed: {stats.get('tasks_failed', 0)}\n"
        message += f"• In Progress: {stats.get('tasks_in_progress', 0)}\n\n"

        # Services monitored
        message += "🔧 *Services Monitored*\n"
        message += f"• Healthy: {stats.get('services_healthy', 0)}\n"
        message += f"• Issues Fixed: {stats.get('services_repaired', 0)}\n\n"

        # System status
        message += "💻 *System Status*\n"
        message += f"• CPU: {stats.get('cpu_percent', 0):.1f}%\n"
        message += f"• Memory: {stats.get('memory_percent', 0):.1f}%\n"
        message += f"• Disk: {stats.get('disk_percent', 0):.1f}%\n\n"

        # Learning & improvements
        message += "🧠 *Learning & Improvements*\n"
        message += f"• Conversations: {stats.get('conversations_processed', 0)}\n"
        message += f"• Patterns Learned: {stats.get('patterns_learned', 0)}\n"
        message += f"• Code Quality Checks: {stats.get('code_reviews', 0)}\n"

        return await self.send_message(message)

    async def send_alert(self, alert_type: str, message: str,
                        severity: str = "warning") -> bool:
        """Send an alert message"""
        severity_emoji = {
            "critical": "🚨🚨🚨",
            "high": "🚨",
            "warning": "⚠️",
            "info": "ℹ️"
        }.get(severity, "📢")

        alert_message = f"{severity_emoji} *ALERT: {alert_type}*\n\n"
        alert_message += f"{message}\n\n"
        alert_message += f"_Severity: {severity.upper()}_"

        return await self.send_message(alert_message)

    async def send_image(self, image_path: str, caption: Optional[str] = None) -> bool:
        """Send an image via Telegram"""
        if not self.is_configured or not self.chat_id:
            return False

        try:
            with open(image_path, 'rb') as photo:
                data = aiohttp.FormData()
                data.add_field('chat_id', self.chat_id)
                data.add_field('photo', photo)
                if caption:
                    data.add_field('caption', caption)

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/sendPhoto",
                        data=data
                    ) as response:
                        return response.status == 200

        except Exception as e:
            logger.error(f"Failed to send image: {e}")
            return False

    async def send_document(self, file_path: str, caption: Optional[str] = None) -> bool:
        """Send a document via Telegram"""
        if not self.is_configured or not self.chat_id:
            return False

        try:
            with open(file_path, 'rb') as doc:
                data = aiohttp.FormData()
                data.add_field('chat_id', self.chat_id)
                data.add_field('document', doc)
                if caption:
                    data.add_field('caption', caption)

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/sendDocument",
                        data=data
                    ) as response:
                        return response.status == 200

        except Exception as e:
            logger.error(f"Failed to send document: {e}")
            return False

# Singleton instance
telegram_client = TelegramClient()

async def get_telegram_client() -> TelegramClient:
    """Get initialized Telegram client"""
    if not telegram_client.is_configured:
        await telegram_client.initialize()
    return telegram_client