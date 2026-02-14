from fastapi import APIRouter, HTTPException
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/google", tags=["google_data"])

async def get_google_credentials_from_tower_auth():
    """Get Google OAuth credentials via tower-auth SSO"""
    try:
        from src.integrations.tower_auth_bridge import tower_auth
        token = await tower_auth.get_valid_token('google')
        if not token:
            logger.warning("No Google token available from tower-auth")
            return None
        return Credentials(token=token)
    except Exception as e:
        logger.error(f"Error getting credentials from tower-auth: {e}")
        return None

@router.get("/emails/count")
async def get_email_count():
    """Get count of emails in Gmail"""
    try:
        creds = await get_google_credentials_from_tower_auth()
        if not creds:
            raise HTTPException(status_code=503, detail="Google credentials not available")

        service = build('gmail', 'v1', credentials=creds)

        # Get count of messages in inbox
        results = service.users().messages().list(userId='me', q='in:inbox').execute()
        messages = results.get('messages', [])

        # Get total count (Gmail API provides this)
        profile = service.users().getProfile(userId='me').execute()
        total_count = profile.get('messagesTotal', 0)

        return {
            "count": total_count,
            "inbox_count": len(messages),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting email count: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get email count: {str(e)}")

@router.get("/photos/count")
async def get_photos_count():
    """Get count of photos in Google Photos"""
    try:
        creds = await get_google_credentials_from_tower_auth()
        if not creds:
            raise HTTPException(status_code=503, detail="Google credentials not available")

        service = build(
            'photoslibrary', 'v1', credentials=creds,
            discoveryServiceUrl='https://photoslibrary.googleapis.com/$discovery/rest?version=v1',
            cache_discovery=False,
        )

        # Get media items count
        # Note: Google Photos API doesn't provide direct count, so we estimate
        results = service.mediaItems().list(pageSize=50).execute()
        items = results.get('mediaItems', [])

        # For a rough count, we could paginate through all items
        # For now, return a sample count
        estimated_count = len(items) * 10  # Very rough estimate

        return {
            "count": estimated_count,
            "sample_items": len(items),
            "timestamp": datetime.now().isoformat(),
            "note": "Estimated count - Google Photos API requires full enumeration"
        }

    except HttpError as e:
        if e.resp.status in (403, 401):
            logger.warning(f"Google Photos API access denied: {e}")
            raise HTTPException(status_code=503, detail="Google Photos API access denied — OAuth scope may not include Photos")
        raise HTTPException(status_code=500, detail=f"Failed to get photos count: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting photos count: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get photos count: {str(e)}")

@router.get("/calendar/count")
async def get_calendar_count():
    """Get count of calendar events"""
    try:
        creds = await get_google_credentials_from_tower_auth()
        if not creds:
            raise HTTPException(status_code=503, detail="Google credentials not available")

        service = build('calendar', 'v3', credentials=creds)

        # Get events from the last 30 days and next 30 days
        now = datetime.utcnow()
        time_min = (now - timedelta(days=30)).isoformat() + 'Z'
        time_max = (now + timedelta(days=30)).isoformat() + 'Z'

        events_result = service.events().list(
            calendarId='primary',
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        events = events_result.get('items', [])

        # Also get upcoming events (next 7 days)
        upcoming_max = (now + timedelta(days=7)).isoformat() + 'Z'
        upcoming_result = service.events().list(
            calendarId='primary',
            timeMin=now.isoformat() + 'Z',
            timeMax=upcoming_max,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        upcoming_events = upcoming_result.get('items', [])

        return {
            "count": len(events),
            "upcoming_count": len(upcoming_events),
            "period": "60 days total (30 past, 30 future)",
            "upcoming_period": "7 days",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting calendar count: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get calendar count: {str(e)}")

@router.get("/summary")
async def get_google_summary():
    """Get summary of all Google data counts"""
    try:
        # Collect all counts — degrade gracefully per service
        summary: dict = {"timestamp": datetime.now().isoformat()}

        try:
            email_data = await get_email_count()
            summary["emails"] = {"total": email_data["count"], "inbox": email_data["inbox_count"]}
        except HTTPException:
            summary["emails"] = {"error": "unavailable"}

        try:
            photos_data = await get_photos_count()
            summary["photos"] = {"estimated_total": photos_data["count"], "note": photos_data["note"]}
        except HTTPException:
            summary["photos"] = {"error": "unavailable — OAuth scope may not include Photos"}

        try:
            calendar_data = await get_calendar_count()
            summary["calendar"] = {"events_60_days": calendar_data["count"], "upcoming_7_days": calendar_data["upcoming_count"]}
        except HTTPException:
            summary["calendar"] = {"error": "unavailable"}

        return summary

    except Exception as e:
        logger.error(f"Error getting Google summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get Google summary: {str(e)}")