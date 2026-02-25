"""Telegram Bot client for Echo Brain notifications."""

import io
import json
import logging
import os
import urllib.request as _ur
from pathlib import Path

import httpx

logger = logging.getLogger("echo-brain.telegram")

# Load from env → vault fallback
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "")


class TelegramClient:
    """Telegram Bot API client for sending messages and photos."""

    def __init__(self):
        self.bot_token = BOT_TOKEN
        self.chat_id = CHAT_ID
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.is_configured = False

    async def initialize(self) -> bool:
        """Verify bot token is valid by calling getMe."""
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram bot token or chat ID not configured")
            return False
        try:
            req = _ur.Request(f"{self.api_url}/getMe", method="GET")
            resp = _ur.urlopen(req, timeout=5)
            data = json.loads(resp.read())
            if data.get("ok"):
                username = data["result"].get("username", "unknown")
                logger.info(f"Telegram bot connected: @{username}")
                self.is_configured = True
                return True
            return False
        except Exception as e:
            logger.warning(f"Telegram initialization failed: {e}")
            return False

    async def get_updates(self, offset: int = 0, timeout: int = 30) -> list[dict]:
        """Long-poll for incoming messages via getUpdates."""
        params: dict = {"timeout": timeout, "allowed_updates": ["message"]}
        if offset > 0:
            params["offset"] = offset
        try:
            async with httpx.AsyncClient(timeout=timeout + 10) as client:
                resp = await client.get(
                    f"{self.api_url}/getUpdates", params=params
                )
                data = resp.json()
                if data.get("ok"):
                    return data.get("result", [])
                logger.warning(f"getUpdates not ok: {data}")
                return []
        except Exception as e:
            logger.error(f"getUpdates failed: {e}")
            return []

    async def send_chat_action(self, action: str = "typing", chat_id: str | None = None) -> bool:
        """Send a chat action (e.g. 'typing' indicator)."""
        payload = json.dumps({
            "chat_id": chat_id or self.chat_id,
            "action": action,
        }).encode()
        try:
            req = _ur.Request(
                f"{self.api_url}/sendChatAction",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            resp = _ur.urlopen(req, timeout=5)
            return resp.getcode() < 300
        except Exception:
            return False

    async def send_message(
        self,
        message: str,
        parse_mode: str = "Markdown",
        disable_web_page_preview: bool = True,
        chat_id: str | None = None,
    ) -> bool:
        """Send a text message."""
        payload = json.dumps({
            "chat_id": chat_id or self.chat_id,
            "text": message,
            "parse_mode": parse_mode,
            "disable_web_page_preview": disable_web_page_preview,
        }).encode()

        try:
            req = _ur.Request(
                f"{self.api_url}/sendMessage",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            resp = _ur.urlopen(req, timeout=10)
            if resp.getcode() < 300:
                logger.info("Telegram message sent")
                return True
            return False
        except Exception as e:
            logger.error(f"Telegram send_message failed: {e}")
            return False

    async def send_photo(
        self,
        image_path: str | None = None,
        image_bytes: bytes | None = None,
        caption: str = "",
        parse_mode: str = "Markdown",
        chat_id: str | None = None,
    ) -> bool:
        """Send a photo with caption. Accepts file path or raw bytes."""
        if image_path:
            photo_data = Path(image_path).read_bytes()
            filename = Path(image_path).name
        elif image_bytes:
            photo_data = image_bytes
            filename = "snapshot.jpg"
        else:
            return await self.send_message(caption, parse_mode=parse_mode, chat_id=chat_id)

        try:
            boundary = "----EchoBrainTelegram"
            body = io.BytesIO()

            def field(name: str, value: str):
                body.write(f"--{boundary}\r\n".encode())
                body.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
                body.write(f"{value}\r\n".encode())

            def file_field(name: str, fname: str, data: bytes):
                body.write(f"--{boundary}\r\n".encode())
                body.write(f'Content-Disposition: form-data; name="{name}"; filename="{fname}"\r\n'.encode())
                body.write(b"Content-Type: image/jpeg\r\n\r\n")
                body.write(data)
                body.write(b"\r\n")

            field("chat_id", chat_id or self.chat_id)
            if caption:
                field("caption", caption)
                field("parse_mode", parse_mode)
            file_field("photo", filename, photo_data)
            body.write(f"--{boundary}--\r\n".encode())

            req = _ur.Request(
                f"{self.api_url}/sendPhoto",
                data=body.getvalue(),
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            )
            resp = _ur.urlopen(req, timeout=15)
            if resp.getcode() < 300:
                logger.info(f"Telegram photo sent: {filename}")
                return True
            return False
        except Exception as e:
            logger.error(f"Telegram send_photo failed: {e}")
            return False
