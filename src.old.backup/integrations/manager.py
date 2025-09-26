"""
Integration Manager for Echo Brain
Central hub for managing all personal data integrations
"""
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class IntegrationManager:
    """Manages all Echo Brain integrations with external services"""

    def __init__(self):
        self.integrations = {}
        self._load_integrations()

    def _load_integrations(self):
        """Lazy load integrations as needed"""
        self.google_calendar = None
        self.plaid = None
        self.apple_music = None
        self.gmail = None

    async def get_google_calendar(self):
        """Get or create Google Calendar integration"""
        if not self.google_calendar:
            try:
                from .google_calendar import GoogleCalendarIntegration
                self.google_calendar = GoogleCalendarIntegration()
                logger.info("Google Calendar integration loaded")
            except Exception as e:
                logger.error(f"Failed to load Google Calendar integration: {e}")
        return self.google_calendar

    async def get_gmail(self):
        """Get or create Gmail integration"""
        if not self.gmail:
            try:
                from .gmail import GmailIntegration
                self.gmail = GmailIntegration()
                logger.info("Gmail integration loaded")
            except Exception as e:
                logger.error(f"Failed to load Gmail integration: {e}")
        return self.gmail

    async def get_plaid(self):
        """Get or create Plaid integration"""
        if not self.plaid:
            try:
                from .plaid import PlaidIntegration
                self.plaid = PlaidIntegration()
                logger.info("Plaid integration loaded")
            except Exception as e:
                logger.error(f"Failed to load Plaid integration: {e}")
        return self.plaid

    async def check_partner_calendar(self) -> Dict[str, Any]:
        """Check partner's shared calendar for events like volleyball practice"""
        calendar = await self.get_google_calendar()
        if not calendar:
            return {'error': 'Google Calendar not available', 'found': False}

        try:
            # Get all shared calendars
            calendars = await calendar.get_calendar_list()
            partner_calendars = [c for c in calendars if c.get('shared', False)]

            result = {
                'shared_calendars_found': len(partner_calendars),
                'calendars': partner_calendars,
                'volleyball_events': []
            }

            # Check for volleyball practice specifically
            volleyball_check = await calendar.check_partner_volleyball_schedule()
            result['volleyball_events'] = volleyball_check.get('events', [])

            # Get upcoming events from partner calendars
            for cal in partner_calendars[:3]:  # Check first 3 shared calendars
                events = await calendar.get_upcoming_events(
                    calendar_id=cal['id'],
                    max_results=5
                )
                cal['upcoming_events'] = events

            return result

        except Exception as e:
            logger.error(f"Error checking partner calendar: {e}")
            return {'error': str(e), 'found': False}

    async def check_emails(self) -> Dict[str, Any]:
        """Check Gmail for important emails"""
        gmail = await self.get_gmail()
        if not gmail:
            return {'error': 'Gmail not available'}

        try:
            # Get email summary
            summary = await gmail.get_email_summary()

            # Check for important emails
            important = await gmail.check_for_important_emails()

            return {
                'summary': summary,
                'important_count': important['count'],
                'important_emails': important['emails'][:5]  # Top 5 important
            }

        except Exception as e:
            logger.error(f"Error checking emails: {e}")
            return {'error': str(e)}

    async def get_financial_overview(self, user_id: str = 'patrick') -> Dict[str, Any]:
        """Get comprehensive financial overview from Plaid"""
        plaid = await self.get_plaid()
        if not plaid:
            return {'error': 'Plaid not available'}

        try:
            # Get financial insights
            insights = await plaid.get_financial_insights()

            return {
                'status': 'connected' if not insights.get('demo_mode') else 'demo',
                'insights': insights,
                'accounts': await plaid.get_accounts(),
                'recent_transactions': await plaid.get_transactions()
            }

        except Exception as e:
            logger.error(f"Error getting financial overview: {e}")
            return {'error': str(e)}

    async def get_integration_status(self) -> Dict[str, Any]:
        """Get status of all integrations"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'integrations': {}
        }

        # Check Google Calendar
        try:
            calendar = await self.get_google_calendar()
            if calendar and calendar.service:
                calendars = await calendar.get_calendar_list()
                status['integrations']['google_calendar'] = {
                    'status': 'connected',
                    'calendars_count': len(calendars),
                    'has_shared': any(c.get('shared', False) for c in calendars)
                }
            else:
                status['integrations']['google_calendar'] = {'status': 'not_configured'}
        except:
            status['integrations']['google_calendar'] = {'status': 'error'}

        # Check Gmail
        try:
            gmail = await self.get_gmail()
            if gmail and gmail.email_address:
                status['integrations']['gmail'] = {
                    'status': 'connected',
                    'email': gmail.email_address
                }
            else:
                status['integrations']['gmail'] = {'status': 'not_configured'}
        except:
            status['integrations']['gmail'] = {'status': 'error'}

        # Check Plaid
        try:
            plaid = await self.get_plaid()
            if plaid and plaid.client_id:
                status['integrations']['plaid'] = {
                    'status': 'configured',
                    'environment': plaid.environment,
                    'demo_mode': plaid.environment == 'sandbox' or plaid.client_id == 'demo_client_id'
                }
            else:
                status['integrations']['plaid'] = {'status': 'not_configured'}
        except:
            status['integrations']['plaid'] = {'status': 'error'}

        # Check Apple Music
        status['integrations']['apple_music'] = {
            'status': 'pending_implementation',
            'note': 'MusicKit integration needed'
        }

        return status

    async def process_user_request(self, message: str) -> Dict[str, Any]:
        """Process natural language requests about integrations"""
        message_lower = message.lower()

        if 'calendar' in message_lower or 'volleyball' in message_lower or 'partner' in message_lower:
            return await self.check_partner_calendar()

        elif 'email' in message_lower or 'gmail' in message_lower or 'important' in message_lower:
            return await self.check_emails()

        elif 'financial' in message_lower or 'bank' in message_lower or 'money' in message_lower or 'plaid' in message_lower:
            return await self.get_financial_overview()

        elif 'status' in message_lower or 'integration' in message_lower:
            return await self.get_integration_status()

        else:
            return {
                'message': 'I can help you with calendar events, emails, financial data, and more.',
                'available_integrations': [
                    'Google Calendar (including shared calendars)',
                    'Gmail (email monitoring)',
                    'Plaid (financial accounts)',
                    'Apple Music (coming soon)'
                ]
            }