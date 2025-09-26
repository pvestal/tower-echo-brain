"""
Updated Google Calendar Integration using Vault tokens
"""
import asyncio
import datetime
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import hvac

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

class GoogleCalendarIntegration:
    """Manages Google Calendar access using tokens from Vault"""

    def __init__(self):
        self.vault_client = hvac.Client(url='http://127.0.0.1:8200')
        self.service = None
        self.creds = None
        self._initialize_from_vault()

    def _initialize_from_vault(self):
        """Initialize Google Calendar service with tokens from Vault"""
        try:
            # Load Vault token
            token_path = Path('/opt/vault/.vault-token')
            if token_path.exists():
                self.vault_client.token = token_path.read_text().strip()
            else:
                # Try root token for dev
                self.vault_client.token = 'root'

            # Get Google tokens from Vault
            token_data = self.vault_client.secrets.kv.v2.read_secret_version(
                path='tokens/google/patrick'
            )

            if token_data:
                data = token_data['data']['data']

                # Get credentials from Vault
                creds_data = self.vault_client.secrets.kv.v2.read_secret_version(
                    path='google/credentials'
                )

                if creds_data:
                    creds = creds_data['data']['data']

                    # Create Credentials object
                    self.creds = Credentials(
                        token=data['access_token'],
                        refresh_token=data['refresh_token'],
                        token_uri='https://oauth2.googleapis.com/token',
                        client_id=creds['client_id'],
                        client_secret=creds['client_secret'],
                        scopes=data['scope'].split(' ') if isinstance(data['scope'], str) else data['scope']
                    )

                    self.service = build('calendar', 'v3', credentials=self.creds)
                    logger.info("Google Calendar service initialized from Vault")
                else:
                    logger.error("Google credentials not found in Vault")
            else:
                logger.error("Google tokens not found in Vault")

        except Exception as e:
            logger.error(f"Failed to initialize from Vault: {e}")

    async def get_calendar_list(self) -> List[Dict[str, Any]]:
        """Get list of all calendars including shared ones"""
        if not self.service:
            return []

        try:
            calendar_list = self.service.calendarList().list().execute()
            calendars = calendar_list.get('items', [])

            result = []
            for calendar in calendars:
                cal_info = {
                    'id': calendar['id'],
                    'summary': calendar.get('summary', 'Unknown'),
                    'description': calendar.get('description', ''),
                    'accessRole': calendar.get('accessRole', 'reader'),
                    'primary': calendar.get('primary', False),
                    'backgroundColor': calendar.get('backgroundColor', '#4285f4')
                }

                # Check if this is a shared calendar
                if not cal_info['primary']:
                    cal_info['shared'] = True
                    cal_info['owner'] = calendar.get('summaryOverride', calendar.get('summary'))

                result.append(cal_info)

            logger.info(f"Retrieved {len(result)} calendars (including shared)")
            return result

        except HttpError as error:
            logger.error(f'Calendar list error: {error}')
            return []

    async def get_upcoming_events(
        self,
        calendar_id: str = 'primary',
        max_results: int = 10,
        time_min: Optional[datetime.datetime] = None,
        time_max: Optional[datetime.datetime] = None,
        search_query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get upcoming events from a specific calendar"""
        if not self.service:
            return []

        try:
            # Default to next 7 days if not specified
            if not time_min:
                time_min = datetime.datetime.now(datetime.timezone.utc)
            if not time_max:
                time_max = time_min + datetime.timedelta(days=7)

            events_result = self.service.events().list(
                calendarId=calendar_id,
                timeMin=time_min.isoformat(),
                timeMax=time_max.isoformat() if time_max else None,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime',
                q=search_query
            ).execute()

            events = events_result.get('items', [])

            result = []
            for event in events:
                event_info = {
                    'id': event['id'],
                    'summary': event.get('summary', 'No title'),
                    'description': event.get('description', ''),
                    'location': event.get('location', ''),
                    'start': event['start'].get('dateTime', event['start'].get('date')),
                    'end': event['end'].get('dateTime', event['end'].get('date')),
                    'attendees': [],
                    'organizer': event.get('organizer', {}).get('email', ''),
                    'htmlLink': event.get('htmlLink', '')
                }

                # Process attendees
                for attendee in event.get('attendees', []):
                    event_info['attendees'].append({
                        'email': attendee.get('email'),
                        'displayName': attendee.get('displayName', ''),
                        'responseStatus': attendee.get('responseStatus', 'needsAction')
                    })

                result.append(event_info)

            return result

        except HttpError as error:
            logger.error(f'Events fetch error: {error}')
            return []

    async def check_partner_volleyball_schedule(self) -> Dict[str, Any]:
        """Specifically check for partner's volleyball practice in shared calendars"""
        calendars = await self.get_calendar_list()
        volleyball_events = []

        for calendar in calendars:
            # Look for shared calendars or volleyball-related
            if calendar.get('shared', False) or 'volleyball' in calendar.get('summary', '').lower():
                events = await self.get_upcoming_events(
                    calendar_id=calendar['id'],
                    search_query='volleyball'
                )

                for event in events:
                    event['calendar_name'] = calendar['summary']
                    event['calendar_owner'] = calendar.get('owner', 'Unknown')
                    volleyball_events.append(event)

        # Also check primary calendar for volleyball
        primary_events = await self.get_upcoming_events(
            calendar_id='primary',
            search_query='volleyball'
        )

        for event in primary_events:
            event['calendar_name'] = 'Primary Calendar'
            event['calendar_owner'] = 'You'
            volleyball_events.append(event)

        return {
            'found': len(volleyball_events) > 0,
            'events': volleyball_events,
            'message': f"Found {len(volleyball_events)} volleyball events across all calendars"
        }