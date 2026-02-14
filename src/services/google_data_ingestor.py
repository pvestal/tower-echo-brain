"""
Google Data Ingestor — Echo Brain
Ingests Google Calendar events, Gmail messages, and Drive file metadata
into Qdrant for semantic search via Echo Brain.
"""
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

import asyncpg
import httpx

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "echo_memory")
DB_URL = os.getenv("DATABASE_URL", "postgresql://echo:echo_secure_password_123@localhost/echo_brain")


class GoogleDataIngestor:
    """Ingests Google Calendar, Gmail, and Drive data into Qdrant."""

    async def _ensure_schema(self, conn: asyncpg.Connection):
        """Create tracking table if it doesn't exist."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS google_ingested_items (
                id              SERIAL PRIMARY KEY,
                source          TEXT NOT NULL,
                source_id       TEXT UNIQUE NOT NULL,
                content_text    TEXT NOT NULL,
                metadata        JSONB DEFAULT '{}',
                qdrant_point_id TEXT,
                ingested_at     TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_gii_source ON google_ingested_items(source)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_gii_source_id ON google_ingested_items(source_id)
        """)

    async def _get_conn(self) -> asyncpg.Connection:
        return await asyncpg.connect(DB_URL)

    async def _embed_text(self, text: str) -> Optional[List[float]]:
        """Embed text via Ollama nomic-embed-text."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{OLLAMA_URL}/api/embeddings",
                    json={"model": EMBED_MODEL, "prompt": text},
                )
                return resp.json().get("embedding")
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None

    async def _store_in_qdrant(
        self, point_id: str, vector: List[float], payload: Dict[str, Any]
    ) -> bool:
        """Store a single point in Qdrant."""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.put(
                    f"{QDRANT_URL}/collections/{COLLECTION}/points",
                    json={
                        "points": [
                            {"id": point_id, "vector": vector, "payload": payload}
                        ]
                    },
                )
                return resp.status_code == 200
        except Exception as e:
            logger.error(f"Qdrant store failed: {e}")
            return False

    async def _already_ingested(self, conn: asyncpg.Connection, source_id: str) -> bool:
        row = await conn.fetchval(
            "SELECT 1 FROM google_ingested_items WHERE source_id = $1", source_id
        )
        return row is not None

    # ------------------------------------------------------------------
    # Calendar ingestion
    # ------------------------------------------------------------------
    async def ingest_calendar_events(
        self, months_back: int = 3, months_forward: int = 3
    ) -> Dict[str, Any]:
        """Ingest calendar events from ALL calendars into Qdrant."""
        from src.integrations.google_calendar import get_calendar_bridge

        bridge = await get_calendar_bridge()
        if not bridge:
            return {"error": "Google Calendar not connected", "ingested": 0}

        conn = await self._get_conn()
        await self._ensure_schema(conn)

        now = datetime.utcnow()
        time_min = (now - timedelta(days=months_back * 30)).isoformat() + "Z"
        time_max = (now + timedelta(days=months_forward * 30)).isoformat() + "Z"

        calendars = await bridge.get_calendars()
        ingested = 0
        skipped = 0
        errors = 0

        try:
            for cal in calendars:
                cal_id = cal["id"]
                cal_name = cal["summary"]
                try:
                    events_result = bridge.service.events().list(
                        calendarId=cal_id,
                        timeMin=time_min,
                        timeMax=time_max,
                        singleEvents=True,
                        orderBy="startTime",
                        maxResults=2500,
                    ).execute()
                except Exception as e:
                    logger.warning(f"Failed to list events for calendar {cal_name}: {e}")
                    errors += 1
                    continue

                for event in events_result.get("items", []):
                    source_id = f"gcal:{event['id']}"
                    if await self._already_ingested(conn, source_id):
                        skipped += 1
                        continue

                    start = event.get("start", {}).get(
                        "dateTime", event.get("start", {}).get("date", "")
                    )
                    end = event.get("end", {}).get(
                        "dateTime", event.get("end", {}).get("date", "")
                    )
                    summary = event.get("summary", "No title")
                    location = event.get("location", "")
                    attendees = [
                        a.get("email", "") for a in event.get("attendees", [])
                    ]
                    description = event.get("description", "")

                    parts = [f"Calendar event: {summary}"]
                    if start:
                        parts.append(f"on {start}")
                    if location:
                        parts.append(f"Location: {location}")
                    if attendees:
                        parts.append(f"Attendees: {', '.join(attendees[:5])}")
                    parts.append(f"Calendar: {cal_name}")
                    if description:
                        parts.append(f"Details: {description[:200]}")
                    content_text = ". ".join(parts)

                    vector = await self._embed_text(content_text)
                    if not vector:
                        errors += 1
                        continue

                    point_id = str(uuid4())
                    payload = {
                        "content": content_text,
                        "text": content_text,
                        "type": "calendar_event",
                        "category": "google:calendar",
                        "source": f"gcal:{cal_name}",
                        "calendar_id": cal_id,
                        "calendar_name": cal_name,
                        "event_id": event["id"],
                        "event_start": start,
                        "event_end": end,
                        "event_summary": summary,
                        "ingested_at": datetime.now().isoformat(),
                    }

                    if await self._store_in_qdrant(point_id, vector, payload):
                        await conn.execute(
                            """INSERT INTO google_ingested_items
                               (source, source_id, content_text, metadata, qdrant_point_id)
                               VALUES ($1, $2, $3, $4, $5)""",
                            "calendar",
                            source_id,
                            content_text,
                            json.dumps({
                                "calendar_name": cal_name,
                                "event_summary": summary,
                                "event_start": start,
                            }),
                            point_id,
                        )
                        ingested += 1
                    else:
                        errors += 1
        finally:
            await conn.close()

        logger.info(
            f"Calendar ingestion: {ingested} ingested, {skipped} skipped, {errors} errors"
        )
        return {
            "ingested": ingested,
            "skipped": skipped,
            "errors": errors,
            "calendars_scanned": len(calendars),
        }

    # ------------------------------------------------------------------
    # Gmail ingestion
    # ------------------------------------------------------------------
    async def ingest_gmail_messages(
        self, max_messages: int = 500, query: str = "newer_than:90d"
    ) -> Dict[str, Any]:
        """Ingest Gmail message metadata + snippets into Qdrant."""
        from src.integrations.tower_auth_bridge import tower_auth

        service = await tower_auth.get_gmail_service()
        if not service:
            return {"error": "Gmail not connected", "ingested": 0}

        conn = await self._get_conn()
        await self._ensure_schema(conn)

        ingested = 0
        skipped = 0
        errors = 0

        try:
            page_token = None
            fetched = 0

            while fetched < max_messages:
                try:
                    result = service.users().messages().list(
                        userId="me",
                        q=query,
                        maxResults=min(100, max_messages - fetched),
                        pageToken=page_token,
                    ).execute()
                except Exception as e:
                    logger.error(f"Gmail list failed: {e}")
                    errors += 1
                    break

                messages = result.get("messages", [])
                if not messages:
                    break

                for msg_stub in messages:
                    msg_id = msg_stub["id"]
                    source_id = f"gmail:{msg_id}"

                    if await self._already_ingested(conn, source_id):
                        skipped += 1
                        fetched += 1
                        continue

                    try:
                        msg = (
                            service.users()
                            .messages()
                            .get(
                                userId="me",
                                id=msg_id,
                                format="metadata",
                                metadataHeaders=["From", "To", "Subject", "Date"],
                            )
                            .execute()
                        )
                    except Exception as e:
                        logger.warning(f"Gmail get {msg_id} failed: {e}")
                        errors += 1
                        fetched += 1
                        continue

                    headers = {
                        h["name"]: h["value"]
                        for h in msg.get("payload", {}).get("headers", [])
                    }
                    snippet = msg.get("snippet", "")
                    from_addr = headers.get("From", "unknown")
                    to_addr = headers.get("To", "unknown")
                    subject = headers.get("Subject", "(no subject)")
                    date_str = headers.get("Date", "")

                    content_text = (
                        f"Email from {from_addr} to {to_addr} on {date_str}. "
                        f"Subject: {subject}. {snippet}"
                    )

                    vector = await self._embed_text(content_text)
                    if not vector:
                        errors += 1
                        fetched += 1
                        continue

                    point_id = str(uuid4())
                    payload = {
                        "content": content_text,
                        "text": content_text,
                        "type": "email",
                        "category": "google:email",
                        "source": "gmail",
                        "email_from": from_addr,
                        "email_to": to_addr,
                        "email_subject": subject,
                        "email_date": date_str,
                        "gmail_id": msg_id,
                        "ingested_at": datetime.now().isoformat(),
                    }

                    if await self._store_in_qdrant(point_id, vector, payload):
                        await conn.execute(
                            """INSERT INTO google_ingested_items
                               (source, source_id, content_text, metadata, qdrant_point_id)
                               VALUES ($1, $2, $3, $4, $5)""",
                            "email",
                            source_id,
                            content_text,
                            json.dumps({
                                "from": from_addr,
                                "subject": subject,
                                "date": date_str,
                            }),
                            point_id,
                        )
                        ingested += 1
                    else:
                        errors += 1

                    fetched += 1

                page_token = result.get("nextPageToken")
                if not page_token:
                    break
        finally:
            await conn.close()

        logger.info(
            f"Gmail ingestion: {ingested} ingested, {skipped} skipped, {errors} errors"
        )
        return {"ingested": ingested, "skipped": skipped, "errors": errors}

    # ------------------------------------------------------------------
    # Drive ingestion
    # ------------------------------------------------------------------
    async def ingest_drive_files(self, max_files: int = 1000) -> Dict[str, Any]:
        """Ingest Google Drive file metadata into Qdrant.

        NOTE: drive.file scope only sees files the app created.
        Full access requires drive.readonly scope (re-auth needed).
        """
        from src.integrations.tower_auth_bridge import tower_auth

        token = await tower_auth.get_valid_token("google")
        if not token:
            return {"error": "Google token not available", "ingested": 0}

        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        creds = Credentials(token=token)
        try:
            service = build("drive", "v3", credentials=creds)
        except Exception as e:
            return {"error": f"Failed to build Drive service: {e}", "ingested": 0}

        conn = await self._get_conn()
        await self._ensure_schema(conn)

        ingested = 0
        skipped = 0
        errors = 0

        try:
            page_token = None
            fetched = 0

            while fetched < max_files:
                try:
                    result = (
                        service.files()
                        .list(
                            pageSize=min(100, max_files - fetched),
                            pageToken=page_token,
                            fields="nextPageToken, files(id,name,mimeType,modifiedTime,size,description,webViewLink)",
                        )
                        .execute()
                    )
                except Exception as e:
                    logger.error(f"Drive list failed: {e}")
                    errors += 1
                    break

                files = result.get("files", [])
                if not files:
                    break

                for f in files:
                    file_id = f["id"]
                    source_id = f"gdrive:{file_id}"

                    if await self._already_ingested(conn, source_id):
                        skipped += 1
                        fetched += 1
                        continue

                    name = f.get("name", "untitled")
                    mime = f.get("mimeType", "unknown")
                    modified = f.get("modifiedTime", "")
                    description = f.get("description", "")
                    link = f.get("webViewLink", "")

                    parts = [f"Google Drive file: {name} ({mime})"]
                    if modified:
                        parts.append(f"Modified: {modified}")
                    if description:
                        parts.append(f"Description: {description[:200]}")
                    content_text = ". ".join(parts)

                    vector = await self._embed_text(content_text)
                    if not vector:
                        errors += 1
                        fetched += 1
                        continue

                    point_id = str(uuid4())
                    payload = {
                        "content": content_text,
                        "text": content_text,
                        "type": "drive_file",
                        "category": "google:drive",
                        "source": "gdrive",
                        "file_name": name,
                        "mime_type": mime,
                        "modified_time": modified,
                        "web_link": link,
                        "ingested_at": datetime.now().isoformat(),
                    }

                    if await self._store_in_qdrant(point_id, vector, payload):
                        await conn.execute(
                            """INSERT INTO google_ingested_items
                               (source, source_id, content_text, metadata, qdrant_point_id)
                               VALUES ($1, $2, $3, $4, $5)""",
                            "drive",
                            source_id,
                            content_text,
                            json.dumps({"name": name, "mime": mime, "modified": modified}),
                            point_id,
                        )
                        ingested += 1
                    else:
                        errors += 1

                    fetched += 1

                page_token = result.get("nextPageToken")
                if not page_token:
                    break
        finally:
            await conn.close()

        logger.info(
            f"Drive ingestion: {ingested} ingested, {skipped} skipped, {errors} errors"
        )
        return {
            "ingested": ingested,
            "skipped": skipped,
            "errors": errors,
            "note": "drive.file scope — only app-created files visible. Re-auth with drive.readonly for full access.",
        }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    async def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get counts by source type."""
        conn = await self._get_conn()
        try:
            await self._ensure_schema(conn)
            rows = await conn.fetch(
                """SELECT source, COUNT(*) as count
                   FROM google_ingested_items
                   GROUP BY source
                   ORDER BY source"""
            )
            total = await conn.fetchval(
                "SELECT COUNT(*) FROM google_ingested_items"
            )
            last = await conn.fetchval(
                "SELECT MAX(ingested_at) FROM google_ingested_items"
            )
            return {
                "total": total or 0,
                "by_source": {r["source"]: r["count"] for r in rows},
                "last_ingested_at": last.isoformat() if last else None,
            }
        finally:
            await conn.close()
