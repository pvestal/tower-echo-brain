#!/usr/bin/env python3
"""
Google Calendar API endpoints for Echo Brain
Provides calendar sync and event management through natural language interface
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Import calendar integration
try:
    from src.integrations.google_calendar import get_calendar_bridge, get_calendar_status_for_echo
    from src.integrations.vault_manager import get_vault_manager
    CALENDAR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Google Calendar integration not available: {e}")
    CALENDAR_AVAILABLE = False

router = APIRouter(prefix="/api/calendar", tags=["google_calendar"])

class EventRequest(BaseModel):
    """Calendar event creation request"""
    title: str
    description: Optional[str] = ""
    start_time: str  # ISO format
    end_time: str    # ISO format
    location: Optional[str] = ""
    attendees: Optional[List[str]] = []
    timezone: Optional[str] = "UTC"

class CalendarQuery(BaseModel):
    """Calendar query for natural language processing"""
    query: str
    date_range: Optional[str] = "today"  # today, tomorrow, week, month
    calendar_id: Optional[str] = "primary"

@router.get("/status")
async def get_calendar_status():
    """Get comprehensive calendar status"""
    if not CALENDAR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Google Calendar integration not available")

    try:
        vault_manager = await get_vault_manager()
        status = await get_calendar_status_for_echo(vault_manager)
        return status
    except Exception as e:
        logger.error(f"Error getting calendar status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get calendar status: {str(e)}")

@router.get("/calendars")
async def get_calendars():
    """Get list of user's calendars"""
    if not CALENDAR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Google Calendar integration not available")

    try:
        vault_manager = await get_vault_manager()
        bridge = await get_calendar_bridge(vault_manager)
        if not bridge:
            raise HTTPException(status_code=503, detail="Calendar not connected")

        calendars = await bridge.get_calendars()
        return {"calendars": calendars, "count": len(calendars)}
    except Exception as e:
        logger.error(f"Error getting calendars: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get calendars: {str(e)}")

@router.get("/events/upcoming")
async def get_upcoming_events(hours: int = 24, max_results: int = 10, calendar_id: str = "primary"):
    """Get upcoming calendar events"""
    if not CALENDAR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Google Calendar integration not available")

    try:
        vault_manager = await get_vault_manager()
        bridge = await get_calendar_bridge(vault_manager)
        if not bridge:
            raise HTTPException(status_code=503, detail="Calendar not connected")

        events = await bridge.get_upcoming_events(hours_ahead=hours, max_results=max_results, calendar_id=calendar_id)
        return {"events": events, "count": len(events)}
    except Exception as e:
        logger.error(f"Error getting upcoming events: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get upcoming events: {str(e)}")

@router.get("/events/today")
async def get_today_events(calendar_id: str = "primary"):
    """Get today's calendar events"""
    if not CALENDAR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Google Calendar integration not available")

    try:
        vault_manager = await get_vault_manager()
        bridge = await get_calendar_bridge(vault_manager)
        if not bridge:
            raise HTTPException(status_code=503, detail="Calendar not connected")

        today = datetime.now()
        events = await bridge.get_events_for_date(today, calendar_id=calendar_id)
        return {"events": events, "count": len(events), "date": today.strftime('%Y-%m-%d')}
    except Exception as e:
        logger.error(f"Error getting today's events: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get today's events: {str(e)}")

@router.get("/events/date/{date}")
async def get_events_for_date(date: str, calendar_id: str = "primary"):
    """Get events for specific date (YYYY-MM-DD)"""
    if not CALENDAR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Google Calendar integration not available")

    try:
        # Parse date
        event_date = datetime.strptime(date, '%Y-%m-%d')

        vault_manager = await get_vault_manager()
        bridge = await get_calendar_bridge(vault_manager)
        if not bridge:
            raise HTTPException(status_code=503, detail="Calendar not connected")

        events = await bridge.get_events_for_date(event_date, calendar_id=calendar_id)
        return {"events": events, "count": len(events), "date": date}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except Exception as e:
        logger.error(f"Error getting events for date: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get events for date: {str(e)}")

@router.post("/events")
async def create_event(event: EventRequest, calendar_id: str = "primary"):
    """Create a new calendar event"""
    if not CALENDAR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Google Calendar integration not available")

    try:
        vault_manager = await get_vault_manager()
        bridge = await get_calendar_bridge(vault_manager)
        if not bridge:
            raise HTTPException(status_code=503, detail="Calendar not connected")

        # Convert request to internal format
        event_data = {
            "title": event.title,
            "description": event.description,
            "start_time": event.start_time,
            "end_time": event.end_time,
            "location": event.location,
            "attendees": event.attendees,
            "timezone": event.timezone
        }

        event_id = await bridge.create_event(event_data, calendar_id=calendar_id)
        if event_id:
            return {"success": True, "event_id": event_id, "message": "Event created successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to create event")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating event: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create event: {str(e)}")

@router.post("/query")
async def query_calendar(query: CalendarQuery):
    """Query calendar with natural language"""
    if not CALENDAR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Google Calendar integration not available")

    try:
        vault_manager = await get_vault_manager()
        bridge = await get_calendar_bridge(vault_manager)
        if not bridge:
            raise HTTPException(status_code=503, detail="Calendar not connected")

        # Process natural language query
        result = await _process_calendar_query(bridge, query)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing calendar query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process calendar query: {str(e)}")

async def _process_calendar_query(bridge, query: CalendarQuery) -> Dict[str, Any]:
    """Process natural language calendar query"""
    query_lower = query.query.lower()

    # Today's events
    if any(word in query_lower for word in ["today", "today's", "what's today"]):
        events = await bridge.get_events_for_date(datetime.now(), query.calendar_id)
        return {
            "category": "today",
            "query": query.query,
            "events": events,
            "count": len(events),
            "summary": f"You have {len(events)} events today"
        }

    # Tomorrow's events
    elif any(word in query_lower for word in ["tomorrow", "tomorrow's"]):
        from datetime import timedelta
        tomorrow = datetime.now() + timedelta(days=1)
        events = await bridge.get_events_for_date(tomorrow, query.calendar_id)
        return {
            "category": "tomorrow",
            "query": query.query,
            "events": events,
            "count": len(events),
            "summary": f"You have {len(events)} events tomorrow"
        }

    # This week's events
    elif any(word in query_lower for word in ["week", "weekly", "this week"]):
        events = await bridge.get_upcoming_events(hours_ahead=168, max_results=50, calendar_id=query.calendar_id)  # 7 days
        return {
            "category": "week",
            "query": query.query,
            "events": events,
            "count": len(events),
            "summary": f"You have {len(events)} events this week"
        }

    # Upcoming events
    elif any(word in query_lower for word in ["upcoming", "next", "soon"]):
        events = await bridge.get_upcoming_events(hours_ahead=72, max_results=20, calendar_id=query.calendar_id)  # 3 days
        return {
            "category": "upcoming",
            "query": query.query,
            "events": events,
            "count": len(events),
            "summary": f"You have {len(events)} upcoming events in the next 3 days"
        }

    # General status
    else:
        status = await bridge.get_calendar_status()
        return {
            "category": "general",
            "query": query.query,
            "status": status,
            "summary": "Calendar status retrieved"
        }