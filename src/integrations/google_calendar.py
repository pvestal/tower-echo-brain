#!/usr/bin/env python3
"""
Google Calendar Integration for Echo Brain
Provides calendar sync and event management capabilities
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

import google.auth.transport.requests
import google.oauth2.credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

class GoogleCalendarBridge:
    """Google Calendar integration for Echo Brain"""

    def __init__(self, vault_manager):
        self.vault_manager = vault_manager
        self.credentials = None
        self.service = None
        self.connected = False

    async def initialize(self) -> bool:
        """Initialize Google Calendar connection"""
        try:
            logger.info("📅 Initializing Google Calendar integration...")

            # Get Google credentials from vault
            google_creds = self.vault_manager.get_google_credentials()
            if not google_creds:
                logger.warning("📅 No Google credentials found in vault")
                return False

            # Create OAuth2 credentials
            if 'access_token' in google_creds:
                self.credentials = google.oauth2.credentials.Credentials(
                    token=google_creds.get('access_token'),
                    refresh_token=google_creds.get('refresh_token'),
                    client_id=google_creds.get('client_id'),
                    client_secret=google_creds.get('client_secret'),
                    token_uri=google_creds.get('token_uri', 'https://oauth2.googleapis.com/token'),
                    scopes=['https://www.googleapis.com/auth/calendar']
                )

                # Test credentials by attempting to refresh if needed
                if await self._refresh_credentials_if_needed():
                    # Build Calendar service
                    self.service = build('calendar', 'v3', credentials=self.credentials)

                    # Test connection with a simple API call
                    await self._test_connection()

                    if self.connected:
                        logger.info("✅ Google Calendar connected successfully")
                        return True

            logger.warning("📅 Google Calendar credentials not properly configured")
            return False

        except Exception as e:
            logger.error(f"❌ Failed to initialize Google Calendar: {e}")
            return False

    async def _refresh_credentials_if_needed(self) -> bool:
        """Refresh credentials if needed"""
        if not self.credentials:
            return False

        try:
            if self.credentials.expired:
                logger.info("📅 Refreshing Google Calendar credentials...")
                request = google.auth.transport.requests.Request()
                self.credentials.refresh(request)
                logger.info("✅ Google Calendar credentials refreshed")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to refresh Google Calendar credentials: {e}")
            return False

    async def _test_connection(self):
        """Test Google Calendar connection"""
        try:
            # Simple test - get calendar list
            calendars_result = self.service.calendarList().list(maxResults=1).execute()
            self.connected = True
            logger.debug("📅 Google Calendar connection test successful")
        except Exception as e:
            logger.error(f"❌ Google Calendar connection test failed: {e}")
            self.connected = False

    async def get_calendars(self) -> List[Dict[str, Any]]:
        """Get list of user's calendars"""
        if not self.connected:
            return []

        try:
            calendars_result = self.service.calendarList().list().execute()
            calendars = calendars_result.get('items', [])

            return [{
                'id': cal['id'],
                'summary': cal['summary'],
                'primary': cal.get('primary', False),
                'access_role': cal.get('accessRole', 'reader')
            } for cal in calendars]

        except Exception as e:
            logger.error(f"❌ Failed to get calendars: {e}")
            return []

    async def get_upcoming_events(self, hours_ahead: int = 24, max_results: int = 10, calendar_id: str = 'primary') -> List[Dict[str, Any]]:
        """Get upcoming events from calendar"""
        if not self.connected:
            return []

        try:
            now = datetime.utcnow()
            time_max = now + timedelta(hours=hours_ahead)

            events_result = self.service.events().list(
                calendarId=calendar_id,
                timeMin=now.isoformat() + 'Z',
                timeMax=time_max.isoformat() + 'Z',
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()

            events = events_result.get('items', [])

            processed_events = []
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                end = event['end'].get('dateTime', event['end'].get('date'))

                processed_events.append({
                    'id': event['id'],
                    'summary': event.get('summary', 'No title'),
                    'description': event.get('description', ''),
                    'start': start,
                    'end': end,
                    'location': event.get('location', ''),
                    'attendees': [att.get('email') for att in event.get('attendees', [])],
                    'calendar_id': calendar_id
                })

            logger.info(f"📅 Retrieved {len(processed_events)} upcoming events")
            return processed_events

        except Exception as e:
            logger.error(f"❌ Failed to get upcoming events: {e}")
            return []

    async def get_events_for_date(self, date: datetime, calendar_id: str = 'primary') -> List[Dict[str, Any]]:
        """Get events for a specific date"""
        if not self.connected:
            return []

        try:
            # Set time bounds for the full day
            start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = date.replace(hour=23, minute=59, second=59, microsecond=999999)

            events_result = self.service.events().list(
                calendarId=calendar_id,
                timeMin=start_of_day.isoformat() + 'Z',
                timeMax=end_of_day.isoformat() + 'Z',
                singleEvents=True,
                orderBy='startTime'
            ).execute()

            events = events_result.get('items', [])

            processed_events = []
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                end = event['end'].get('dateTime', event['end'].get('date'))

                processed_events.append({
                    'id': event['id'],
                    'summary': event.get('summary', 'No title'),
                    'description': event.get('description', ''),
                    'start': start,
                    'end': end,
                    'location': event.get('location', ''),
                    'attendees': [att.get('email') for att in event.get('attendees', [])],
                    'calendar_id': calendar_id
                })

            logger.info(f"📅 Retrieved {len(processed_events)} events for {date.strftime('%Y-%m-%d')}")
            return processed_events

        except Exception as e:
            logger.error(f"❌ Failed to get events for date: {e}")
            return []

    async def create_event(self, event_data: Dict[str, Any], calendar_id: str = 'primary') -> Optional[str]:
        """Create a new calendar event"""
        if not self.connected:
            return None

        try:
            # Convert event data to Google Calendar format
            google_event = {
                'summary': event_data.get('title', 'New Event'),
                'description': event_data.get('description', ''),
                'start': {
                    'dateTime': event_data['start_time'],
                    'timeZone': event_data.get('timezone', 'UTC'),
                },
                'end': {
                    'dateTime': event_data['end_time'],
                    'timeZone': event_data.get('timezone', 'UTC'),
                },
            }

            if event_data.get('location'):
                google_event['location'] = event_data['location']

            if event_data.get('attendees'):
                google_event['attendees'] = [{'email': email} for email in event_data['attendees']]

            # Create the event
            created_event = self.service.events().insert(
                calendarId=calendar_id,
                body=google_event
            ).execute()

            logger.info(f"📅 Created calendar event: {created_event['summary']}")
            return created_event['id']

        except Exception as e:
            logger.error(f"❌ Failed to create calendar event: {e}")
            return None

    async def get_calendar_status(self) -> Dict[str, Any]:
        """Get comprehensive calendar status for Echo Brain"""
        if not self.connected:
            return {
                "status": "disconnected",
                "message": "Google Calendar not connected"
            }

        try:
            # Get calendars
            calendars = await self.get_calendars()
            primary_calendar = next((cal for cal in calendars if cal.get('primary')), None)

            # Get today's events
            today = datetime.now()
            today_events = await self.get_events_for_date(today)

            # Get upcoming events (next 7 days)
            upcoming_events = await self.get_upcoming_events(hours_ahead=168, max_results=20)  # 7 days = 168 hours

            return {
                "status": "connected",
                "last_update": datetime.now().isoformat(),
                "calendars": {
                    "total": len(calendars),
                    "primary": primary_calendar['summary'] if primary_calendar else None,
                    "list": [cal['summary'] for cal in calendars[:5]]  # First 5 calendars
                },
                "events": {
                    "today": len(today_events),
                    "upcoming_week": len(upcoming_events),
                    "today_events": [event['summary'] for event in today_events[:3]],  # First 3 today
                    "next_events": [event['summary'] for event in upcoming_events[:5]]  # First 5 upcoming
                }
            }

        except Exception as e:
            logger.error(f"❌ Error getting calendar status: {e}")
            return {"status": "error", "error": str(e)}

# Global calendar bridge instance
_calendar_bridge: Optional[GoogleCalendarBridge] = None

async def get_calendar_bridge(vault_manager) -> Optional[GoogleCalendarBridge]:
    """Get or create Calendar bridge instance"""
    global _calendar_bridge

    if _calendar_bridge is None:
        _calendar_bridge = GoogleCalendarBridge(vault_manager)
        connected = await _calendar_bridge.initialize()
        if not connected:
            logger.warning("📅 Google Calendar bridge not available")
            return None

    return _calendar_bridge

async def get_calendar_status_for_echo(vault_manager) -> Dict[str, Any]:
    """Get calendar status formatted for Echo Brain queries"""
    bridge = await get_calendar_bridge(vault_manager)
    if not bridge:
        return {
            "available": False,
            "message": "Google Calendar integration not configured or not accessible"
        }

    status = await bridge.get_calendar_status()
    return {
        "available": True,
        "google_calendar": status
    }

# CLI testing function
async def test_calendar_connection(vault_manager):
    """Test Google Calendar connection from command line"""
    print("📅 Testing Google Calendar connection...")

    bridge = await get_calendar_bridge(vault_manager)
    if bridge:
        status = await bridge.get_calendar_status()
        print(f"✅ Google Calendar connected: {json.dumps(status, indent=2)}")

        # Get upcoming events
        events = await bridge.get_upcoming_events()
        print(f"📅 Found {len(events)} upcoming events")
        for event in events[:3]:
            print(f"  - {event['summary']} at {event['start']}")
    else:
        print("❌ Google Calendar connection failed")

if __name__ == "__main__":
    import sys
    import os
    sys.path.append('/opt/tower-echo-brain')
    from src.integrations.vault_manager import get_vault_manager

    async def main():
        vault_manager = await get_vault_manager()
        await test_calendar_connection(vault_manager)

    asyncio.run(main())