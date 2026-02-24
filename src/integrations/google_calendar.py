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

import google.oauth2.credentials
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

class GoogleCalendarBridge:
    """Google Calendar integration for Echo Brain via tower-auth SSO"""

    def __init__(self):
        self.credentials = None
        self.service = None
        self.connected = False

    async def initialize(self) -> bool:
        """Initialize Google Calendar connection via tower-auth"""
        try:
            from src.integrations.tower_auth_bridge import tower_auth

            logger.info("Initializing Google Calendar integration via tower-auth...")

            token = await tower_auth.get_valid_token('google')
            if not token:
                logger.warning("No Google token available from tower-auth")
                return False

            self.credentials = google.oauth2.credentials.Credentials(token=token)
            self.service = build('calendar', 'v3', credentials=self.credentials)

            # Test connection
            await self._test_connection()

            if self.connected:
                logger.info("Google Calendar connected successfully via tower-auth")
                return True

            logger.warning("Google Calendar connection test failed")
            return False

        except Exception as e:
            logger.error(f"Failed to initialize Google Calendar: {e}")
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
            # Set time bounds for the full day.
            # Convert to UTC so the isoformat+Z pattern stays valid even
            # when the caller passes a tz-aware datetime (e.g. Pacific).
            from zoneinfo import ZoneInfo
            utc = ZoneInfo("UTC")
            if date.tzinfo is not None:
                date = date.astimezone(utc)
            start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
            end_of_day = date.replace(hour=23, minute=59, second=59, microsecond=999999, tzinfo=None)

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
            # Detect all-day events (date-only strings like "2026-06-23")
            start_val = event_data['start_time']
            end_val = event_data['end_time']
            is_all_day = len(start_val) <= 10 and len(end_val) <= 10

            google_event = {
                'summary': event_data.get('title', 'New Event'),
                'description': event_data.get('description', ''),
            }

            if is_all_day:
                google_event['start'] = {'date': start_val}
                google_event['end'] = {'date': end_val}
            else:
                google_event['start'] = {
                    'dateTime': start_val,
                    'timeZone': event_data.get('timezone', 'UTC'),
                }
                google_event['end'] = {
                    'dateTime': end_val,
                    'timeZone': event_data.get('timezone', 'UTC'),
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

    async def get_events_for_month(self, year: int, month: int) -> Dict[str, Any]:
        """Get events from ALL calendars for a given month, with calendar colors.

        Query window covers the full visible grid: Sunday of the first week
        through Saturday of the last week, so multi-day events that start in
        adjacent months are captured.
        """
        if not self.connected:
            return {"calendars": [], "events": [], "year": year, "month": month}

        try:
            import calendar as cal_mod
            from datetime import date as date_cls

            first_of_month = date_cls(year, month, 1)
            _, days_in_month = cal_mod.monthrange(year, month)
            last_of_month = date_cls(year, month, days_in_month)

            # Grid starts on Sunday of the week containing the 1st
            grid_start = first_of_month - timedelta(days=first_of_month.weekday() + 1)
            if first_of_month.weekday() == 6:  # Sunday
                grid_start = first_of_month
            # Grid ends on Saturday of the week containing the last day
            days_to_sat = (5 - last_of_month.weekday()) % 7
            if last_of_month.weekday() == 6:  # Sunday → need 6 more days
                days_to_sat = 6
            grid_end = last_of_month + timedelta(days=days_to_sat)

            time_min = f"{grid_start.isoformat()}T00:00:00Z"
            time_max = f"{grid_end.isoformat()}T23:59:59Z"

            # Get calendar list with colors
            cal_list_result = self.service.calendarList().list().execute()
            cal_items = cal_list_result.get("items", [])

            calendars_meta = []
            all_events = []

            for cal_entry in cal_items:
                cal_id = cal_entry["id"]
                cal_summary = cal_entry.get("summary", cal_id)
                bg_color = cal_entry.get("backgroundColor", "#238636")

                calendars_meta.append({
                    "id": cal_id,
                    "summary": cal_summary,
                    "color": bg_color,
                })

                try:
                    events_result = self.service.events().list(
                        calendarId=cal_id,
                        timeMin=time_min,
                        timeMax=time_max,
                        singleEvents=True,
                        orderBy="startTime",
                        maxResults=2500,
                    ).execute()
                except Exception as e:
                    logger.warning(f"Failed to fetch events for {cal_summary}: {e}")
                    continue

                for event in events_result.get("items", []):
                    start = event.get("start", {}).get(
                        "dateTime", event.get("start", {}).get("date", "")
                    )
                    end = event.get("end", {}).get(
                        "dateTime", event.get("end", {}).get("date", "")
                    )
                    all_events.append({
                        "id": event["id"],
                        "summary": event.get("summary", "No title"),
                        "start": start,
                        "end": end,
                        "location": event.get("location", ""),
                        "calendar_id": cal_id,
                        "calendar_name": cal_summary,
                        "calendar_color": bg_color,
                    })

            # Sort by start time
            all_events.sort(key=lambda e: e["start"])

            return {
                "calendars": calendars_meta,
                "events": all_events,
                "year": year,
                "month": month,
                "grid_start": grid_start.isoformat(),
                "grid_end": grid_end.isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get events for month {year}-{month:02d}: {e}")
            return {"calendars": [], "events": [], "year": year, "month": month}

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

async def get_calendar_bridge() -> Optional[GoogleCalendarBridge]:
    """Get or create Calendar bridge instance, retrying if previously disconnected"""
    global _calendar_bridge

    if _calendar_bridge is None or not _calendar_bridge.connected:
        _calendar_bridge = GoogleCalendarBridge()
        connected = await _calendar_bridge.initialize()
        if not connected:
            logger.warning("Google Calendar bridge not available")
            _calendar_bridge = None
            return None

    return _calendar_bridge

async def get_calendar_status_for_echo() -> Dict[str, Any]:
    """Get calendar status formatted for Echo Brain queries"""
    bridge = await get_calendar_bridge()
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
async def test_calendar_connection():
    """Test Google Calendar connection from command line"""
    print("Testing Google Calendar connection...")

    bridge = await get_calendar_bridge()
    if bridge:
        status = await bridge.get_calendar_status()
        print(f"Google Calendar connected: {json.dumps(status, indent=2)}")

        events = await bridge.get_upcoming_events()
        print(f"Found {len(events)} upcoming events")
        for event in events[:3]:
            print(f"  - {event['summary']} at {event['start']}")
    else:
        print("Google Calendar connection failed")

if __name__ == "__main__":
    import sys
    sys.path.append('/opt/tower-echo-brain')

    async def main():
        from src.integrations.tower_auth_bridge import tower_auth
        await tower_auth.initialize()
        await tower_auth.load_existing_tokens()
        await test_calendar_connection()

    asyncio.run(main())