#!/usr/bin/env python3
"""
ntfy.sh Integration for Echo Brain
Provides instant push notifications with priority and scheduling
"""

import asyncio
import aiohttp
import logging
import json
import os
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class Priority(Enum):
    """ntfy notification priorities"""
    MIN = 1
    LOW = 2
    DEFAULT = 3
    HIGH = 4
    MAX = 5

class NtfyClient:
    """ntfy.sh client for push notifications"""

    def __init__(self, server_url: str = "https://ntfy.sh"):
        self.server_url = server_url.rstrip('/')
        self.default_topic = os.getenv("NTFY_TOPIC", "tower-echo-brain")
        self.auth_token = os.getenv("NTFY_TOKEN")
        self.username = os.getenv("NTFY_USERNAME")
        self.password = os.getenv("NTFY_PASSWORD")
        self.is_configured = False

    async def initialize(self) -> bool:
        """Initialize ntfy client and verify connection"""
        try:
            # Test basic connectivity
            await self._test_connection()
            self.is_configured = True
            logger.info(f"✅ ntfy client initialized - server: {self.server_url}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize ntfy client: {e}")
            return False

    async def _test_connection(self) -> bool:
        """Test connection to ntfy server"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.server_url}/v1/health"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        logger.debug("✅ ntfy server is healthy")
                        return True
                    else:
                        logger.warning(f"⚠️ ntfy server returned status {response.status}")
                        return False
        except Exception as e:
            logger.error(f"❌ ntfy health check failed: {e}")
            return False

    async def send_notification(
        self,
        message: str,
        title: Optional[str] = None,
        topic: Optional[str] = None,
        priority: Priority = Priority.DEFAULT,
        tags: Optional[List[str]] = None,
        click_url: Optional[str] = None,
        attach_url: Optional[str] = None,
        schedule: Optional[datetime] = None,
        email: Optional[str] = None
    ) -> bool:
        """
        Send notification via ntfy

        Args:
            message: Notification message body
            title: Optional notification title
            topic: ntfy topic (defaults to configured topic)
            priority: Notification priority level
            tags: List of emoji tags for the notification
            click_url: URL to open when notification is clicked
            attach_url: URL to attach file/image
            schedule: Future datetime to schedule notification
            email: Email address to also send notification to

        Returns:
            bool: True if notification sent successfully
        """
        if not self.is_configured:
            logger.error("❌ ntfy client not configured")
            return False

        try:
            topic = topic or self.default_topic
            url = f"{self.server_url}/{topic}"

            headers = {
                "Content-Type": "text/plain; charset=utf-8"
            }

            # Add authentication if configured
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            elif self.username and self.password:
                # Basic auth will be handled by aiohttp
                pass  # Authentication will be set later

            # Add optional headers
            if title:
                headers["X-Title"] = title

            if priority != Priority.DEFAULT:
                headers["X-Priority"] = str(priority.value)

            if tags:
                headers["X-Tags"] = ",".join(tags)

            if click_url:
                headers["X-Click"] = click_url

            if attach_url:
                headers["X-Attach"] = attach_url

            if schedule:
                # Convert to Unix timestamp
                timestamp = int(schedule.timestamp())
                headers["X-Delay"] = str(timestamp)

            if email:
                headers["X-Email"] = email

            # Prepare authentication for aiohttp
            auth = None
            if self.username and self.password and not self.auth_token:
                auth = aiohttp.BasicAuth(self.username, self.password)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    data=message.encode('utf-8'),
                    headers=headers,
                    auth=auth,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info(f"📱 ntfy notification sent to '{topic}': {title or message[:50]}")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ ntfy notification failed: {response.status} - {error_text}")
                        return False

        except Exception as e:
            logger.error(f"❌ Failed to send ntfy notification: {e}")
            return False

    async def subscribe_to_topic(self, topic: str, callback: callable) -> None:
        """
        Subscribe to ntfy topic for incoming notifications

        Args:
            topic: Topic to subscribe to
            callback: Function to call when notification received
        """
        if not self.is_configured:
            logger.error("❌ ntfy client not configured")
            return

        try:
            url = f"{self.server_url}/{topic}/sse"

            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"

            auth = None
            if self.username and self.password and not self.auth_token:
                auth = aiohttp.BasicAuth(self.username, self.password)

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, auth=auth) as response:
                    if response.status == 200:
                        logger.info(f"📱 Subscribed to ntfy topic: {topic}")
                        async for line in response.content:
                            if line:
                                try:
                                    # Parse SSE data
                                    line_str = line.decode('utf-8').strip()
                                    if line_str.startswith('data: '):
                                        data = json.loads(line_str[6:])  # Remove 'data: ' prefix
                                        await callback(data)
                                except Exception as e:
                                    logger.error(f"Error processing ntfy message: {e}")
                    else:
                        logger.error(f"Failed to subscribe to topic: {response.status}")

        except Exception as e:
            logger.error(f"❌ Failed to subscribe to ntfy topic: {e}")

    async def list_topics(self) -> List[str]:
        """List available topics (if server supports it)"""
        # Note: Public ntfy.sh doesn't provide topic listing
        # This would work with self-hosted ntfy servers
        logger.info("📱 Topic listing not supported on public ntfy.sh")
        return []

    def get_status(self) -> Dict[str, Any]:
        """Get ntfy client status"""
        return {
            "configured": self.is_configured,
            "server": self.server_url,
            "default_topic": self.default_topic,
            "has_auth": bool(self.auth_token or (self.username and self.password)),
            "auth_type": "token" if self.auth_token else "basic" if self.username else "none"
        }

# Global ntfy client instance
_ntfy_client: Optional[NtfyClient] = None

async def get_ntfy_client() -> Optional[NtfyClient]:
    """Get or create ntfy client instance"""
    global _ntfy_client

    if _ntfy_client is None:
        _ntfy_client = NtfyClient()
        initialized = await _ntfy_client.initialize()
        if not initialized:
            logger.warning("📱 ntfy client not available")
            return None

    return _ntfy_client

# Convenience functions for common notification types

async def send_info_notification(message: str, title: str = "Echo Brain Info") -> bool:
    """Send informational notification"""
    client = await get_ntfy_client()
    if client:
        return await client.send_notification(
            message=message,
            title=title,
            priority=Priority.DEFAULT,
            tags=["information_source"]
        )
    return False

async def send_warning_notification(message: str, title: str = "Echo Brain Warning") -> bool:
    """Send warning notification"""
    client = await get_ntfy_client()
    if client:
        return await client.send_notification(
            message=message,
            title=title,
            priority=Priority.HIGH,
            tags=["warning"]
        )
    return False

async def send_error_notification(message: str, title: str = "Echo Brain Error") -> bool:
    """Send error notification"""
    client = await get_ntfy_client()
    if client:
        return await client.send_notification(
            message=message,
            title=title,
            priority=Priority.MAX,
            tags=["rotating_light", "exclamation"]
        )
    return False

async def send_success_notification(message: str, title: str = "Echo Brain Success") -> bool:
    """Send success notification"""
    client = await get_ntfy_client()
    if client:
        return await client.send_notification(
            message=message,
            title=title,
            priority=Priority.DEFAULT,
            tags=["white_check_mark"]
        )
    return False

# CLI testing function
async def test_ntfy_connection():
    """Test ntfy connection from command line"""
    print("📱 Testing ntfy connection...")

    client = await get_ntfy_client()
    if client:
        status = client.get_status()
        print(f"✅ ntfy client status: {json.dumps(status, indent=2)}")

        # Send test notification
        success = await client.send_notification(
            message="Test message from Echo Brain ntfy integration",
            title="Echo Brain Test",
            tags=["test_tube"]
        )

        if success:
            print("✅ Test notification sent successfully")
        else:
            print("❌ Test notification failed")
    else:
        print("❌ ntfy client initialization failed")

if __name__ == "__main__":
    asyncio.run(test_ntfy_connection())