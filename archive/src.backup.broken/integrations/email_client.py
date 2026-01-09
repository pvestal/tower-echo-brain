#!/usr/bin/env python3
"""
Email Client for Echo Brain
Provides email notification capabilities
"""

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional, List
import os

logger = logging.getLogger(__name__)

class EmailClient:
    """Simple email client for notifications"""

    def __init__(self):
        # Try to get from environment first
        self.smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.environ.get("SMTP_PORT", 587))
        self.smtp_user = os.environ.get("SMTP_USER", "patrick.vestal@gmail.com")
        self.smtp_password = os.environ.get("SMTP_PASSWORD", "")
        self.from_email = os.environ.get("FROM_EMAIL", "patrick.vestal@gmail.com")
        self.to_email = os.environ.get("TO_EMAIL", "patrick.vestal@gmail.com")

    async def send_notification(self, subject: str, body: str,
                               to_email: Optional[str] = None) -> bool:
        """Send an email notification"""
        try:
            to_address = to_email or self.to_email

            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = to_address
            msg['Subject'] = f"[Echo Brain] {subject}"

            # Add timestamp to body
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            full_body = f"{body}\n\n---\nSent by Echo Brain at {timestamp}"

            msg.attach(MIMEText(full_body, 'plain'))

            # Send email if we have password configured
            if self.smtp_password:
                with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                    server.starttls()
                    server.login(self.smtp_user, self.smtp_password)
                    server.send_message(msg)

                logger.info(f"✅ Email sent: {subject}")
                return True
            else:
                logger.warning("⚠️ SMTP password not configured, email not sent")
                logger.info(f"📧 Would send: {subject}\n{body}")
                return False

        except Exception as e:
            logger.error(f"❌ Failed to send email: {e}")
            return False

    async def send_daily_digest(self, events: List[dict]) -> bool:
        """Send daily digest of Echo Brain activities"""
        if not events:
            return True

        subject = f"Daily Digest - {len(events)} events"

        body = "Echo Brain Daily Activity Report\n"
        body += "=" * 40 + "\n\n"

        # Group events by type
        by_type = {}
        for event in events:
            event_type = event.get("type", "unknown")
            if event_type not in by_type:
                by_type[event_type] = []
            by_type[event_type].append(event)

        # Format each type
        for event_type, items in by_type.items():
            body += f"\n{event_type.upper()} ({len(items)} events)\n"
            body += "-" * 30 + "\n"
            for item in items[:5]:  # Limit to 5 per type
                body += f"• {item.get('message', 'No message')}\n"
            if len(items) > 5:
                body += f"  ... and {len(items) - 5} more\n"

        body += "\n" + "=" * 40
        body += "\n\nSummary:\n"
        body += f"• Total Events: {len(events)}\n"
        body += f"• Event Types: {', '.join(by_type.keys())}\n"

        return await self.send_notification(subject, body)

    async def send_alert(self, alert_type: str, message: str,
                        details: Optional[dict] = None) -> bool:
        """Send an alert notification"""
        subject = f"ALERT: {alert_type}"

        body = f"Alert Type: {alert_type}\n"
        body += f"Message: {message}\n"

        if details:
            body += "\nDetails:\n"
            for key, value in details.items():
                body += f"• {key}: {value}\n"

        return await self.send_notification(subject, body)

# Singleton instance
email_client = EmailClient()

async def get_email_client() -> EmailClient:
    """Get email client instance"""
    return email_client