#!/usr/bin/env python3
"""
Unified Calendar Interface for Echo Brain
Manages multiple calendar accounts (Google, Apple) with sync and merge capabilities
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import httpx
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class CalendarEvent:
    """Unified calendar event structure"""
    id: str
    title: str
    start: datetime
    end: datetime
    location: Optional[str] = None
    description: Optional[str] = None
    attendees: List[str] = None
    source: str = 'google'  # google, apple, outlook
    calendar_id: str = None
    all_day: bool = False
    reminders: List[int] = None  # Minutes before event
    recurring: bool = False
    color: str = None

class UnifiedCalendarManager:
    """Manages multiple calendar accounts with unified interface"""

    def __init__(self):
        self.tower_auth_url = "http://localhost:8088"
        self.calendars = {}
        self.events_cache = []
        self.sync_interval = 300  # 5 minutes

    async def add_google_calendar(self, email: str, calendar_id: str = 'primary'):
        """Add a Google calendar to sync"""
        self.calendars[f"google_{email}"] = {
            'type': 'google',
            'email': email,
            'calendar_id': calendar_id,
            'last_sync': None
        }
        logger.info(f"Added Google calendar for {email}")

    async def add_apple_calendar(self, apple_id: str):
        """Add an Apple calendar to sync"""
        self.calendars[f"apple_{apple_id}"] = {
            'type': 'apple',
            'apple_id': apple_id,
            'last_sync': None
        }
        logger.info(f"Added Apple calendar for {apple_id}")

    async def sync_all_calendars(self) -> Dict[str, Any]:
        """Sync all configured calendars"""
        results = {
            'synced': [],
            'failed': [],
            'total_events': 0,
            'events': []
        }

        tasks = []
        for cal_key, cal_info in self.calendars.items():
            if cal_info['type'] == 'google':
                tasks.append(self._sync_google_calendar(cal_key, cal_info))
            elif cal_info['type'] == 'apple':
                tasks.append(self._sync_apple_calendar(cal_key, cal_info))

        # Run all syncs in parallel
        sync_results = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, (cal_key, result) in enumerate(zip(self.calendars.keys(), sync_results)):
            if isinstance(result, Exception):
                logger.error(f"Failed to sync {cal_key}: {result}")
                results['failed'].append(cal_key)
            else:
                results['synced'].append(cal_key)
                results['events'].extend(result)
                results['total_events'] += len(result)

        # Sort all events by start time
        results['events'].sort(key=lambda x: x.start)

        # Cache the events
        self.events_cache = results['events']

        return results

    async def _sync_google_calendar(self, cal_key: str, cal_info: Dict) -> List[CalendarEvent]:
        """Sync a Google calendar"""
        try:
            # Get auth token from Tower Auth
            async with httpx.AsyncClient() as client:
                token_response = await client.get(
                    f"{self.tower_auth_url}/tokens/google",
                    headers={'X-Calendar-Email': cal_info['email']}
                )

                if token_response.status_code != 200:
                    raise Exception("Failed to get Google token")

                token = token_response.json().get('access_token')

            # Fetch calendar events
            headers = {'Authorization': f'Bearer {token}'}
            params = {
                'calendarId': cal_info['calendar_id'],
                'timeMin': datetime.utcnow().isoformat() + 'Z',
                'timeMax': (datetime.utcnow() + timedelta(days=30)).isoformat() + 'Z',
                'singleEvents': True,
                'orderBy': 'startTime'
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://www.googleapis.com/calendar/v3/calendars/{cal_info['calendar_id']}/events",
                    headers=headers,
                    params=params
                )

                if response.status_code == 200:
                    events_data = response.json().get('items', [])
                    events = []

                    for event in events_data:
                        # Convert to unified format
                        start = event.get('start', {})
                        end = event.get('end', {})

                        # Handle all-day events
                        if 'date' in start:
                            start_dt = datetime.fromisoformat(start['date'])
                            end_dt = datetime.fromisoformat(end['date'])
                            all_day = True
                        else:
                            start_dt = datetime.fromisoformat(start.get('dateTime', '').replace('Z', '+00:00'))
                            end_dt = datetime.fromisoformat(end.get('dateTime', '').replace('Z', '+00:00'))
                            all_day = False

                        unified_event = CalendarEvent(
                            id=event['id'],
                            title=event.get('summary', 'Untitled'),
                            start=start_dt,
                            end=end_dt,
                            location=event.get('location'),
                            description=event.get('description'),
                            attendees=[a.get('email') for a in event.get('attendees', [])],
                            source='google',
                            calendar_id=cal_info['email'],
                            all_day=all_day,
                            recurring='recurringEventId' in event
                        )
                        events.append(unified_event)

                    cal_info['last_sync'] = datetime.now()
                    return events

        except Exception as e:
            logger.error(f"Google calendar sync failed for {cal_key}: {e}")
            raise

    async def _sync_apple_calendar(self, cal_key: str, cal_info: Dict) -> List[CalendarEvent]:
        """Sync an Apple calendar using CalDAV"""
        try:
            # Apple Calendar uses CalDAV protocol
            # This would require caldav library and Apple credentials
            from caldav import DAVClient

            # Get Apple credentials from Tower Auth
            async with httpx.AsyncClient() as client:
                creds_response = await client.get(
                    f"{self.tower_auth_url}/tokens/apple",
                    headers={'X-Apple-ID': cal_info['apple_id']}
                )

                if creds_response.status_code != 200:
                    raise Exception("Failed to get Apple credentials")

                creds = creds_response.json()

            # Connect to iCloud CalDAV
            url = f"https://caldav.icloud.com/{creds['user_id']}/calendars"
            client = DAVClient(
                url=url,
                username=cal_info['apple_id'],
                password=creds['app_specific_password']
            )

            principal = client.principal()
            calendars = principal.calendars()

            events = []
            for calendar in calendars:
                # Get events from last 7 days to next 30 days
                start = datetime.now() - timedelta(days=7)
                end = datetime.now() + timedelta(days=30)

                cal_events = calendar.date_search(start, end)

                for event in cal_events:
                    vevents = event.icalendar_instance.subcomponents

                    for vevent in vevents:
                        if vevent.name == 'VEVENT':
                            unified_event = CalendarEvent(
                                id=str(vevent.get('uid')),
                                title=str(vevent.get('summary')),
                                start=vevent.get('dtstart').dt,
                                end=vevent.get('dtend').dt,
                                location=str(vevent.get('location', '')),
                                description=str(vevent.get('description', '')),
                                source='apple',
                                calendar_id=cal_info['apple_id'],
                                all_day='DATE' in str(vevent.get('dtstart').params),
                                recurring=vevent.get('rrule') is not None
                            )
                            events.append(unified_event)

            cal_info['last_sync'] = datetime.now()
            return events

        except Exception as e:
            logger.error(f"Apple calendar sync failed for {cal_key}: {e}")
            # Fallback to EventKit Bridge if available
            return await self._sync_apple_calendar_eventkit(cal_key, cal_info)

    async def _sync_apple_calendar_eventkit(self, cal_key: str, cal_info: Dict) -> List[CalendarEvent]:
        """Fallback sync using EventKit Bridge API"""
        # This would connect to a macOS service exposing EventKit
        # For now, return empty list
        logger.warning(f"EventKit bridge not available for {cal_key}")
        return []

    async def get_upcoming_events(self, hours: int = 24) -> List[CalendarEvent]:
        """Get upcoming events across all calendars"""
        if not self.events_cache:
            await self.sync_all_calendars()

        cutoff_time = datetime.now() + timedelta(hours=hours)
        upcoming = [e for e in self.events_cache if e.start <= cutoff_time and e.end >= datetime.now()]

        return sorted(upcoming, key=lambda x: x.start)

    async def search_events(self, query: str) -> List[CalendarEvent]:
        """Search events by title or description"""
        if not self.events_cache:
            await self.sync_all_calendars()

        query_lower = query.lower()
        matching = []

        for event in self.events_cache:
            if (query_lower in event.title.lower() or
                (event.description and query_lower in event.description.lower()) or
                (event.location and query_lower in event.location.lower())):
                matching.append(event)

        return sorted(matching, key=lambda x: x.start)

    async def create_event(self, event: CalendarEvent, calendar_email: str = None) -> Dict[str, Any]:
        """Create an event in specified calendar"""
        # Determine which calendar to use
        if calendar_email:
            target_cal = f"google_{calendar_email}"
        else:
            # Use first Google calendar by default
            target_cal = next((k for k in self.calendars if k.startswith('google_')), None)

        if not target_cal:
            return {'error': 'No calendar available for creating events'}

        cal_info = self.calendars[target_cal]

        if cal_info['type'] == 'google':
            return await self._create_google_event(event, cal_info)
        elif cal_info['type'] == 'apple':
            return await self._create_apple_event(event, cal_info)

    async def _create_google_event(self, event: CalendarEvent, cal_info: Dict) -> Dict[str, Any]:
        """Create event in Google Calendar"""
        try:
            # Get auth token
            async with httpx.AsyncClient() as client:
                token_response = await client.get(
                    f"{self.tower_auth_url}/tokens/google",
                    headers={'X-Calendar-Email': cal_info['email']}
                )
                token = token_response.json().get('access_token')

            # Build event body
            event_body = {
                'summary': event.title,
                'location': event.location,
                'description': event.description,
                'start': {
                    'dateTime': event.start.isoformat(),
                    'timeZone': 'America/Los_Angeles'
                },
                'end': {
                    'dateTime': event.end.isoformat(),
                    'timeZone': 'America/Los_Angeles'
                }
            }

            if event.attendees:
                event_body['attendees'] = [{'email': email} for email in event.attendees]

            if event.reminders:
                event_body['reminders'] = {
                    'useDefault': False,
                    'overrides': [{'method': 'popup', 'minutes': m} for m in event.reminders]
                }

            # Create the event
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://www.googleapis.com/calendar/v3/calendars/{cal_info['calendar_id']}/events",
                    headers={'Authorization': f'Bearer {token}'},
                    json=event_body
                )

                if response.status_code == 200:
                    created_event = response.json()
                    logger.info(f"✅ Event created: {created_event['id']}")
                    return {'success': True, 'event_id': created_event['id']}
                else:
                    return {'error': f"Failed to create event: {response.text}"}

        except Exception as e:
            logger.error(f"Failed to create Google event: {e}")
            return {'error': str(e)}

    async def _create_apple_event(self, event: CalendarEvent, cal_info: Dict) -> Dict[str, Any]:
        """Create event in Apple Calendar"""
        # Would implement CalDAV event creation
        logger.warning("Apple event creation not yet implemented")
        return {'error': 'Apple event creation pending implementation'}

    async def detect_conflicts(self, new_event: CalendarEvent) -> List[CalendarEvent]:
        """Detect scheduling conflicts"""
        if not self.events_cache:
            await self.sync_all_calendars()

        conflicts = []
        for existing in self.events_cache:
            # Check for time overlap
            if (new_event.start < existing.end and new_event.end > existing.start):
                conflicts.append(existing)

        return conflicts

    async def suggest_meeting_times(self, duration_minutes: int, attendee_emails: List[str],
                                   days_ahead: int = 7) -> List[Dict[str, Any]]:
        """Suggest available meeting times for multiple attendees"""
        # This would check calendars of all attendees
        # For now, return available slots in primary calendar

        if not self.events_cache:
            await self.sync_all_calendars()

        suggestions = []
        current = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
        end_date = current + timedelta(days=days_ahead)

        while current < end_date:
            # Skip weekends
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue

            # Check business hours (9 AM - 5 PM)
            if current.hour >= 9 and current.hour < 17:
                slot_end = current + timedelta(minutes=duration_minutes)

                # Check if slot is free
                is_free = True
                for event in self.events_cache:
                    if (current < event.end and slot_end > event.start):
                        is_free = False
                        break

                if is_free:
                    suggestions.append({
                        'start': current.isoformat(),
                        'end': slot_end.isoformat(),
                        'available': True
                    })

                    if len(suggestions) >= 5:  # Return top 5 suggestions
                        break

            # Move to next slot
            current += timedelta(minutes=30)

        return suggestions

# Global instance
unified_calendar = UnifiedCalendarManager()