from fastapi import APIRouter, HTTPException
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/google", tags=["google_data"])

GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/photoslibrary.readonly",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/drive.file",
]

async def get_google_credentials_from_tower_auth():
    """Get Google OAuth credentials via tower-auth SSO.

    Always fetches a fresh token from tower-auth (which handles refresh
    internally via Vault-backed refresh tokens).  The returned Credentials
    object carries an expiry so that googleapiclient never silently uses a
    stale token.
    """
    try:
        from src.integrations.tower_auth_bridge import tower_auth
        token = await tower_auth.get_valid_token('google')
        if not token:
            logger.warning("No Google token available from tower-auth")
            return None

        # Pull expiry from the cached token info (set by tower-auth after
        # refresh).  Default to 55 min from now — Google access tokens live
        # 60 min, and this avoids the edge-case where the token expires
        # between credential creation and the API call.
        token_info = tower_auth.cached_tokens.get('google', {})
        expiry = token_info.get('expires_at', datetime.utcnow() + timedelta(minutes=55))

        return Credentials(token=token, expiry=expiry, scopes=GOOGLE_SCOPES)
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

        # Google Photos API has no "total count" endpoint — we can only
        # enumerate pages.  Fetch one page of 100 as a sample.
        results = service.mediaItems().list(pageSize=100).execute()
        items = results.get('mediaItems', [])
        has_more = 'nextPageToken' in results

        return {
            "sample_count": len(items),
            "has_more": has_more,
            "timestamp": datetime.now().isoformat(),
            "note": "Google Photos API does not expose a total count; this is one page of results"
        }

    except HttpError as e:
        if e.resp.status in (403, 401):
            detail_msg = str(e)
            if "insufficient authentication scopes" in detail_msg.lower():
                hint = ("Google Photos Library API may not be enabled in the "
                        "Cloud Console project.  Enable it at: "
                        "https://console.cloud.google.com/apis/library/photoslibrary.googleapis.com")
            else:
                hint = "OAuth scope may not include photoslibrary.readonly"
            logger.warning(f"Google Photos API access denied: {e}")
            raise HTTPException(status_code=503, detail=f"Google Photos API access denied — {hint}")
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
            summary["photos"] = {"sample_count": photos_data["sample_count"], "has_more": photos_data["has_more"], "note": photos_data["note"]}
        except HTTPException as exc:
            summary["photos"] = {"error": exc.detail}

        try:
            calendar_data = await get_calendar_count()
            summary["calendar"] = {"events_60_days": calendar_data["count"], "upcoming_7_days": calendar_data["upcoming_count"]}
        except HTTPException:
            summary["calendar"] = {"error": "unavailable"}

        return summary

    except Exception as e:
        logger.error(f"Error getting Google summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get Google summary: {str(e)}")