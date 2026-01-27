from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import json
import os
from datetime import datetime, timedelta

router = APIRouter(prefix="/google", tags=["google_data"])
security = HTTPBearer()

def load_google_credentials():
    """Load Google OAuth credentials from config file"""
    try:
        with open('/opt/tower-echo-brain/config/google_credentials.json') as f:
            cred_data = json.load(f)

        return Credentials(
            token=cred_data['token'],
            refresh_token=cred_data.get('refresh_token'),
            token_uri=cred_data.get('token_uri', 'https://oauth2.googleapis.com/token'),
            client_id=cred_data.get('client_id'),
            client_secret=cred_data.get('client_secret'),
            scopes=cred_data.get('scopes', [])
        )
    except Exception as e:
        print(f"Error loading credentials: {e}")
        return None

async def verify_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple auth verification - in production this would validate JWT tokens"""
    # For now, just check if token exists
    if not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing token")
    return credentials.credentials

@router.get("/emails/count")
async def get_email_count(token: str = Depends(verify_auth)):
    """Get count of emails in Gmail"""
    try:
        creds = load_google_credentials()
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
        print(f"Error getting email count: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get email count: {str(e)}")

@router.get("/photos/count")
async def get_photos_count(token: str = Depends(verify_auth)):
    """Get count of photos in Google Photos"""
    try:
        creds = load_google_credentials()
        if not creds:
            raise HTTPException(status_code=503, detail="Google credentials not available")

        service = build('photoslibrary', 'v1', credentials=creds)

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

    except Exception as e:
        print(f"Error getting photos count: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get photos count: {str(e)}")

@router.get("/calendar/count")
async def get_calendar_count(token: str = Depends(verify_auth)):
    """Get count of calendar events"""
    try:
        creds = load_google_credentials()
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
        print(f"Error getting calendar count: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get calendar count: {str(e)}")

@router.get("/summary")
async def get_google_summary(token: str = Depends(verify_auth)):
    """Get summary of all Google data counts"""
    try:
        # Collect all counts
        email_data = await get_email_count(token)
        photos_data = await get_photos_count(token)
        calendar_data = await get_calendar_count(token)

        return {
            "emails": {
                "total": email_data["count"],
                "inbox": email_data["inbox_count"]
            },
            "photos": {
                "estimated_total": photos_data["count"],
                "note": photos_data["note"]
            },
            "calendar": {
                "events_60_days": calendar_data["count"],
                "upcoming_7_days": calendar_data["upcoming_count"]
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"Error getting Google summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get Google summary: {str(e)}")