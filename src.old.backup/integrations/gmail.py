"""
Gmail Integration for Echo Brain using App Password
"""
import asyncio
import imaplib
import smtplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import hvac
from pathlib import Path

logger = logging.getLogger(__name__)

class GmailIntegration:
    """Manages Gmail access using app password from Vault"""

    def __init__(self):
        self.vault_client = hvac.Client(url='http://127.0.0.1:8200')
        self.email_address = None
        self.app_password = None
        self.smtp_server = None
        self.smtp_port = None
        self._load_credentials()

    def _load_credentials(self):
        """Load Gmail credentials from Vault"""
        try:
            # Load Vault token
            token_path = Path('/opt/vault/.vault-token')
            if token_path.exists():
                self.vault_client.token = token_path.read_text().strip()

            # Get Gmail credentials from Vault
            gmail_data = self.vault_client.secrets.kv.v2.read_secret_version(
                path='tower/gmail'
            )

            if gmail_data:
                data = gmail_data['data']['data']
                self.email_address = data['email']
                self.app_password = data['app_password']
                self.smtp_server = data.get('smtp_server', 'smtp.gmail.com')
                self.smtp_port = int(data.get('smtp_port', 587))
                logger.info("Gmail credentials loaded from Vault")
            else:
                logger.error("Gmail credentials not found in Vault")

        except Exception as e:
            logger.error(f"Failed to load Gmail credentials: {e}")

    async def get_recent_emails(self, folder: str = 'INBOX', limit: int = 10, search_query: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent emails from Gmail"""
        if not self.email_address or not self.app_password:
            return []

        try:
            # Connect to Gmail IMAP
            mail = imaplib.IMAP4_SSL('imap.gmail.com')
            mail.login(self.email_address, self.app_password)
            mail.select(folder)

            # Search for emails
            if search_query:
                search_criteria = f'(TEXT "{search_query}")'
            else:
                # Get emails from last 7 days
                date = (datetime.now() - timedelta(days=7)).strftime("%d-%b-%Y")
                search_criteria = f'(SINCE {date})'

            _, message_ids = mail.search(None, search_criteria)
            message_ids = message_ids[0].split()

            emails = []
            for msg_id in message_ids[-limit:]:  # Get last N emails
                _, msg_data = mail.fetch(msg_id, '(RFC822)')
                raw_email = msg_data[0][1]
                msg = email.message_from_bytes(raw_email)

                # Extract email details
                email_dict = {
                    'id': msg_id.decode(),
                    'from': msg['From'],
                    'to': msg['To'],
                    'subject': msg['Subject'],
                    'date': msg['Date'],
                    'body': self._get_email_body(msg)[:500]  # First 500 chars
                }

                emails.append(email_dict)

            mail.close()
            mail.logout()

            return emails

        except Exception as e:
            logger.error(f"Failed to get emails: {e}")
            return []

    def _get_email_body(self, msg) -> str:
        """Extract body from email message"""
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    break
        else:
            body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
        return body

    async def send_email(self, to_email: str, subject: str, body: str, cc: Optional[List[str]] = None) -> bool:
        """Send an email using Gmail"""
        if not self.email_address or not self.app_password:
            return False

        try:
            # Create message
            message = MIMEMultipart()
            message['From'] = self.email_address
            message['To'] = to_email
            message['Subject'] = subject

            if cc:
                message['Cc'] = ', '.join(cc)

            # Add body
            message.attach(MIMEText(body, 'plain'))

            # Connect and send
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_address, self.app_password)

                recipients = [to_email]
                if cc:
                    recipients.extend(cc)

                server.send_message(message)

            logger.info(f"Email sent successfully to {to_email}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    async def check_for_important_emails(self) -> Dict[str, Any]:
        """Check for important emails that need attention"""
        important_senders = [
            'lucia',
            'volleyball',
            'school',
            'doctor',
            'appointment',
            'urgent',
            'important'
        ]

        important_emails = []

        for keyword in important_senders:
            emails = await self.get_recent_emails(search_query=keyword, limit=5)
            for email_msg in emails:
                # Check if already added
                if not any(e['id'] == email_msg['id'] for e in important_emails):
                    email_msg['keyword_match'] = keyword
                    important_emails.append(email_msg)

        return {
            'count': len(important_emails),
            'emails': important_emails[:10],  # Top 10 important emails
            'status': 'checked'
        }

    async def get_email_summary(self) -> str:
        """Get a summary of recent emails"""
        emails = await self.get_recent_emails(limit=20)

        if not emails:
            return "No recent emails found or Gmail not connected."

        # Group by sender
        senders = {}
        for email_msg in emails:
            sender = email_msg['from'].split('<')[0].strip()
            if sender not in senders:
                senders[sender] = []
            senders[sender].append(email_msg['subject'])

        summary = f"📧 **Email Summary** (Last {len(emails)} emails):\n\n"

        for sender, subjects in list(senders.items())[:5]:  # Top 5 senders
            summary += f"**From {sender}:**\n"
            for subject in subjects[:3]:  # First 3 subjects
                summary += f"  • {subject}\n"
            if len(subjects) > 3:
                summary += f"  • ...and {len(subjects) - 3} more\n"
            summary += "\n"

        # Check for important emails
        important = await self.check_for_important_emails()
        if important['count'] > 0:
            summary += f"⚠️ **{important['count']} potentially important emails need attention**\n"

        return summary