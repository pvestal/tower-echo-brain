#!/usr/bin/env python3
"""
Email Manager for Echo Brain
Handles email notifications with multiple backend support
"""

import smtplib
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

logger = logging.getLogger(__name__)

class EmailManager:
    """Centralized email management for Echo Brain"""

    def __init__(self):
        self.config = self._load_config()
        self.vault_token = None
        self.credentials = None
        self._load_credentials()

    def _load_config(self) -> Dict[str, Any]:
        """Load email configuration"""
        config = {
            'from_email': 'patrick.vestal.digital@gmail.com',
            'to_email': 'patrick.vestal@gmail.com',
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'use_tls': True,
            'fallback_to_log': True
        }

        # Try to load from config file
        config_file = Path("/opt/tower-echo-brain/config/email_config.json")
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config.update(json.load(f))
            except Exception as e:
                logger.warning(f"Could not load email config: {e}")

        return config

    def _load_credentials(self):
        """Load email credentials from multiple sources"""

        # Try 1: HashiCorp Vault
        try:
            import hvac
            import os

            vault_addr = os.environ.get('VAULT_ADDR', 'http://127.0.0.1:8200')
            vault_token = os.environ.get('VAULT_TOKEN')

            if not vault_token:
                # Try to read from file
                token_file = Path("/home/patrick/.vault-token")
                if token_file.exists():
                    vault_token = token_file.read_text().strip()

            if vault_token:
                client = hvac.Client(url=vault_addr, token=vault_token)
                if client.is_authenticated():
                    try:
                        response = client.secrets.kv.v2.read_secret_version(path='tower/gmail')
                        if response and 'data' in response:
                            self.credentials = response['data']['data'].get('app_password')
                            logger.info("✅ Loaded email credentials from Vault")
                            return
                    except Exception as e:
                        logger.debug(f"Vault read failed: {e}")
        except ImportError:
            logger.debug("hvac not installed, skipping Vault")
        except Exception as e:
            logger.debug(f"Vault connection failed: {e}")

        # Try 2: Tower credentials file
        tower_creds = Path("/home/patrick/.tower_credentials/vault.json")
        if tower_creds.exists():
            try:
                with open(tower_creds) as f:
                    vault_data = json.load(f)
                    gmail_config = vault_data.get('gmail', {})
                    self.credentials = gmail_config.get('password')
                    if self.credentials:
                        logger.info("✅ Loaded email credentials from Tower vault")
                        return
            except Exception as e:
                logger.debug(f"Tower vault read failed: {e}")

        # Try 3: Gmail app password file
        app_pass_file = Path("/home/patrick/.gmail-app-password")
        if app_pass_file.exists():
            try:
                self.credentials = app_pass_file.read_text().strip()
                logger.info("✅ Loaded email credentials from app password file")
                return
            except Exception as e:
                logger.debug(f"App password file read failed: {e}")

        logger.warning("⚠️ No email credentials found - will use fallback logging")

    async def send_email(self, subject: str, body: str, to_email: Optional[str] = None) -> bool:
        """Send email notification"""

        try:
            to_email = to_email or self.config['to_email']

            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config['from_email']
            msg['To'] = to_email
            msg['Subject'] = subject
            msg['Date'] = datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000")

            # Add Echo Brain signature
            body += "\n\n---\nSent by Echo Brain Autonomous System\n" + datetime.now().isoformat()
            msg.attach(MIMEText(body, 'plain'))

            # Try to send via SMTP
            if self.credentials:
                try:
                    with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                        if self.config.get('use_tls'):
                            server.starttls()
                        server.login(self.config['from_email'], self.credentials)
                        server.send_message(msg)
                        logger.info(f"✅ Email sent: {subject}")
                        return True
                except Exception as e:
                    logger.error(f"SMTP send failed: {e}")

            # Fallback: Try local sendmail if available
            try:
                import subprocess
                sendmail_path = "/usr/sbin/sendmail"
                if Path(sendmail_path).exists():
                    p = subprocess.Popen([sendmail_path, to_email], stdin=subprocess.PIPE)
                    p.communicate(msg.as_string().encode())
                    if p.returncode == 0:
                        logger.info(f"✅ Email sent via sendmail: {subject}")
                        return True
            except Exception as e:
                logger.debug(f"Sendmail failed: {e}")

            # Final fallback: Log the email
            if self.config.get('fallback_to_log', True):
                log_file = Path("/opt/tower-echo-brain/logs/email_notifications.log")
                log_file.parent.mkdir(parents=True, exist_ok=True)

                with open(log_file, 'a') as f:
                    f.write(f"\n{'='*50}\n")
                    f.write(f"Date: {datetime.now().isoformat()}\n")
                    f.write(f"To: {to_email}\n")
                    f.write(f"Subject: {subject}\n")
                    f.write(f"Body:\n{body}\n")

                logger.info(f"📝 Email logged to file: {subject}")
                return True

        except Exception as e:
            logger.error(f"Failed to send/log email: {e}")
            return False

    async def send_digest(self, entries: list, title: str = "Echo Brain Daily Digest") -> bool:
        """Send a digest email with multiple entries"""

        if not entries:
            return True

        body = f"{title}\n{'=' * len(title)}\n\n"
        body += f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        for i, entry in enumerate(entries, 1):
            body += f"{i}. {entry}\n"

        return await self.send_email(title, body)

# Global instance
email_manager = EmailManager()