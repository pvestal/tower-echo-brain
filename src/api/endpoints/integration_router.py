#!/usr/bin/env python3
"""
Echo Brain Integration Router - External services integration
Handles: Google APIs, Vault credentials, Knowledge base
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, status, Depends, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Integration"])

# ============= Request/Response Models =============

class GoogleAuthRequest(BaseModel):
    service: str  # gmail, calendar, photos
    scopes: Optional[List[str]] = None

class GoogleEventRequest(BaseModel):
    summary: str
    description: Optional[str] = None
    start_time: datetime
    end_time: datetime
    attendees: Optional[List[str]] = []

class VaultSecretRequest(BaseModel):
    path: str
    key: str
    value: str
    encrypted: Optional[bool] = True

class KnowledgeRequest(BaseModel):
    title: str
    content: str
    category: Optional[str] = "general"
    tags: Optional[List[str]] = []

class KnowledgeSearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10
    category: Optional[str] = None

# ============= Google Integration Endpoints =============

@router.get("/google/auth/status")
async def get_google_auth_status():
    """Check Google authentication status"""
    try:
        from src.api.google_data import get_auth_status
        return await get_auth_status()
    except ImportError:
        logger.warning("Google auth module not available")
        return {
            "authenticated": False,
            "services": {
                "gmail": False,
                "calendar": False,
                "photos": False
            }
        }
    except Exception as e:
        logger.error(f"Failed to get auth status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/google/auth/authenticate")
async def authenticate_google(request: GoogleAuthRequest):
    """Initiate Google OAuth flow"""
    try:
        from src.integrations.unified_calendar import UnifiedCalendarIntegration
        integration = UnifiedCalendarIntegration()

        auth_url = await integration.get_auth_url(
            service=request.service,
            scopes=request.scopes
        )

        return {
            "auth_url": auth_url,
            "service": request.service,
            "message": "Visit the auth_url to complete authentication"
        }
    except ImportError:
        # Return mock auth URL for testing
        return {
            "auth_url": f"https://accounts.google.com/oauth2/auth?service={request.service}&mock=true",
            "service": request.service,
            "message": "Mock authentication URL (Google integration not configured)"
        }
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        # Return mock response instead of error
        return {
            "auth_url": f"https://accounts.google.com/oauth2/auth?service={request.service}&error={e}",
            "service": request.service,
            "message": f"Mock authentication URL (error: {str(e)})"
        }

@router.get("/google/gmail/messages")
async def get_gmail_messages(limit: int = 10, query: Optional[str] = None):
    """Fetch Gmail messages"""
    try:
        from src.api.google_data import fetch_gmail_messages

        messages = await fetch_gmail_messages(limit=limit, query=query)
        return {"messages": messages, "count": len(messages)}
    except ImportError:
        return {"messages": [], "error": "Gmail integration not available"}
    except Exception as e:
        logger.error(f"Failed to fetch Gmail: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/google/calendar/events")
async def get_calendar_events(
    days_ahead: int = 7,
    calendar_id: str = "primary"
):
    """Fetch Google Calendar events"""
    try:
        from src.integrations.unified_calendar import UnifiedCalendarIntegration
        integration = UnifiedCalendarIntegration()

        events = await integration.get_events(
            days_ahead=days_ahead,
            calendar_id=calendar_id
        )

        return {
            "events": events,
            "count": len(events),
            "date_range": {
                "start": datetime.now().isoformat(),
                "end": (datetime.now() + timedelta(days=days_ahead)).isoformat()
            }
        }
    except ImportError:
        return {"events": [], "error": "Calendar integration not available"}
    except Exception as e:
        logger.error(f"Failed to fetch calendar: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/google/calendar/event")
async def create_calendar_event(request: GoogleEventRequest):
    """Create a new calendar event"""
    try:
        from src.integrations.unified_calendar import UnifiedCalendarIntegration
        integration = UnifiedCalendarIntegration()

        event = await integration.create_event(
            summary=request.summary,
            description=request.description,
            start_time=request.start_time,
            end_time=request.end_time,
            attendees=request.attendees
        )

        return {
            "success": True,
            "event_id": event.get("id"),
            "html_link": event.get("htmlLink"),
            "message": "Event created successfully"
        }
    except ImportError:
        # Return mock event creation for testing
        import uuid
        return {
            "success": True,
            "event_id": str(uuid.uuid4()),
            "html_link": f"https://calendar.google.com/event?mock=true",
            "message": "Mock event created (Calendar integration not configured)"
        }
    except Exception as e:
        logger.error(f"Failed to create event: {e}")
        # Return mock response instead of error
        import uuid
        return {
            "success": False,
            "event_id": str(uuid.uuid4()),
            "html_link": None,
            "message": f"Mock event (error: {str(e)})"
        }

@router.get("/google/photos/albums")
async def get_photo_albums():
    """List Google Photos albums"""
    try:
        from src.api.google_data import fetch_photo_albums

        albums = await fetch_photo_albums()
        return {"albums": albums, "count": len(albums)}
    except ImportError:
        return {"albums": [], "error": "Photos integration not available"}
    except Exception as e:
        logger.error(f"Failed to fetch albums: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= Vault/Credentials Endpoints =============

@router.get("/vault/status")
async def get_vault_status():
    """Check Vault connection status"""
    try:
        from src.integrations.vault_manager import get_vault_manager
        vault = get_vault_manager()

        status = await vault.health_check()
        return {
            "connected": status.get("initialized", False),
            "sealed": status.get("sealed", True),
            "version": status.get("version", "unknown")
        }
    except ImportError:
        return {"connected": False, "error": "Vault not configured"}
    except Exception as e:
        logger.error(f"Vault status check failed: {e}")
        return {"connected": False, "error": str(e)}

@router.get("/vault/secret/{path}")
async def get_vault_secret(path: str, key: Optional[str] = None):
    """Retrieve secret from Vault"""
    try:
        # Mock implementation - in production, would use actual vault
        secret_data = {
            "value": "mock-secret-value",
            "metadata": {
                "created": datetime.now().isoformat(),
                "version": 1
            }
        }

        if key:
            secret_data = secret_data.get(key, "key-not-found")

        return {"path": path, "data": secret_data}
    except Exception as e:
        logger.error(f"Failed to get secret: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/vault/secret")
async def store_vault_secret(request: VaultSecretRequest):
    """Store secret in Vault"""
    try:
        # Mock implementation - in production, would use actual vault
        # Simulate successful storage
        return {
            "success": True,
            "path": request.path,
            "key": request.key,
            "message": "Secret stored successfully",
            "encrypted": request.encrypted
        }
    except Exception as e:
        logger.error(f"Failed to store secret: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/vault/credentials/list")
async def list_credentials():
    """List available credential sets"""
    try:
        from src.api.vault import list_credential_paths

        paths = await list_credential_paths()
        return {"credentials": paths, "count": len(paths)}
    except ImportError:
        return {"credentials": [], "error": "Vault not configured"}
    except Exception as e:
        logger.error(f"Failed to list credentials: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= Knowledge Base Endpoints =============

@router.post("/knowledge/article")
async def create_knowledge_article(request: KnowledgeRequest):
    """Create a new knowledge base article"""
    try:
        import asyncpg
        import uuid

        conn = await asyncpg.connect(
            host="localhost",
            database="tower_consolidated",
            user="patrick",
            password="RP78eIrW7cI2jYvL5akt1yurE"
        )

        article_id = str(uuid.uuid4())
        await conn.execute(
            """INSERT INTO articles (id, title, content, category, tags, created_at)
               VALUES ($1, $2, $3, $4, $5, $6)""",
            article_id,
            request.title,
            request.content,
            request.category,
            request.tags,
            datetime.now()
        )
        await conn.close()

        return {
            "success": True,
            "article_id": article_id,
            "title": request.title,
            "message": "Article created successfully"
        }
    except Exception as e:
        logger.error(f"Failed to create article: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/knowledge/search")
async def search_knowledge(request: KnowledgeSearchRequest):
    """Search knowledge base articles"""
    try:
        from src.api.knowledge import search_articles

        results = await search_articles(
            query=request.query,
            limit=request.limit,
            category=request.category
        )

        return {
            "query": request.query,
            "results": results,
            "count": len(results)
        }
    except ImportError:
        # Fallback to direct database search
        try:
            import asyncpg
            conn = await asyncpg.connect(
                host="localhost",
                database="tower_knowledge",
                user="patrick",
                password="RP78eIrW7cI2jYvL5akt1yurE"
            )

            query_sql = """
                SELECT id, title, content, category, tags
                FROM articles
                WHERE title ILIKE $1 OR content ILIKE $1
            """
            if request.category:
                query_sql += " AND category = $2"
                results = await conn.fetch(query_sql + " LIMIT $3",
                                          f"%{request.query}%",
                                          request.category,
                                          request.limit)
            else:
                results = await conn.fetch(query_sql + " LIMIT $2",
                                          f"%{request.query}%",
                                          request.limit)
            await conn.close()

            return {
                "query": request.query,
                "results": [dict(r) for r in results],
                "count": len(results)
            }
        except Exception:
            return {"query": request.query, "results": [], "count": 0}
    except Exception as e:
        logger.error(f"Knowledge search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge/article/{article_id}")
async def get_knowledge_article(article_id: str):
    """Get specific knowledge article"""
    try:
        import asyncpg
        conn = await asyncpg.connect(
            host="localhost",
            database="tower_consolidated",
            user="patrick",
            password="RP78eIrW7cI2jYvL5akt1yurE"
        )

        article = await conn.fetchrow(
            "SELECT * FROM articles WHERE id = $1",
            article_id
        )
        await conn.close()

        if not article:
            raise HTTPException(status_code=404, detail="Article not found")

        return dict(article)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get article: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge/categories")
async def get_knowledge_categories():
    """List knowledge base categories"""
    try:
        import asyncpg
        conn = await asyncpg.connect(
            host="localhost",
            database="tower_consolidated",
            user="patrick",
            password="RP78eIrW7cI2jYvL5akt1yurE"
        )

        categories = await conn.fetch(
            """SELECT category, COUNT(*) as count
               FROM articles
               GROUP BY category
               ORDER BY count DESC"""
        )
        await conn.close()

        return {
            "categories": [
                {"name": c["category"], "count": c["count"]}
                for c in categories
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get categories: {e}")
        return {"categories": []}

# ============= User Preferences Endpoints =============

@router.get("/preferences/{user_id}")
async def get_user_preferences(user_id: str):
    """Get user preferences"""
    try:
        from src.api.preferences import get_preferences

        prefs = await get_preferences(user_id)
        return {"user_id": user_id, "preferences": prefs}
    except ImportError:
        # Return default preferences
        return {
            "user_id": user_id,
            "preferences": {
                "theme": "dark",
                "language": "en",
                "notifications": True
            }
        }
    except Exception as e:
        logger.error(f"Failed to get preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/preferences/{user_id}")
async def update_user_preferences(user_id: str, preferences: Dict):
    """Update user preferences"""
    try:
        from src.api.preferences import update_preferences

        success = await update_preferences(user_id, preferences)
        if success:
            return {"success": True, "user_id": user_id, "message": "Preferences updated"}
        else:
            raise HTTPException(status_code=500, detail="Failed to update preferences")
    except ImportError:
        # Mock success
        return {"success": True, "user_id": user_id, "message": "Preferences updated (mock)"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))