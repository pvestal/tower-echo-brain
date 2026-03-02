"""
Photo & Video Memory Service — Echo Brain
Scans local photos, Google Takeout media (photos + videos), performs
vision analysis with Gemma 3, face detection with InsightFace, and
ingests everything into Qdrant for semantic search.

PERSONAL MEDIA ONLY — no anime, ComfyUI, LoRA, or training content.
"""
import base64
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import asyncpg
import httpx

logger = logging.getLogger(__name__)

# --- Configuration ---
LOCAL_PHOTOS_ROOT = Path("/home/patrick/Pictures")
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".gif", ".bmp", ".tiff", ".tif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp", ".wmv"}
MEDIA_EXTENSIONS = PHOTO_EXTENSIONS | VIDEO_EXTENSIONS
EXCLUDED_PATH_SEGMENTS = {"ComfyUI", "comfyui", "lora", "training", "anime", "datasets", "workflow", "output", "temp"}
TAKEOUT_PHOTOS_ROOT = Path("/mnt/10TB2/Google_Takeout_2025/Takeout/Google Photos")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
VISION_MODEL = "gemma3:12b"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "echo_memory")
DB_URL = os.getenv("DATABASE_URL", "postgresql://echo:echo_secure_password_123@localhost/echo_brain")
AUTH_SERVICE_URL = "http://localhost:8088"

GOOGLE_PHOTOS_API = "https://photoslibrary.googleapis.com/v1"

MAX_VIDEO_KEYFRAMES = 30  # Cap at 30 frames (5 min of video at 10s interval)

# Vision prompt for personal photo analysis (Gemma 3) — enhanced
_VISION_PHOTO_PROMPT = """Analyze this personal photo briefly:
1. SCENE: What is the main subject or scene? (one sentence)
2. PEOPLE: Are there people? How many? General description (no names).
3. LOCATION: Where was this likely taken? (indoor/outdoor, type of place)
4. MOOD: What mood or emotion does this photo convey? (one word)
5. OBJECTS: List specific notable objects visible (e.g. dog, car, mountain, cake, guitar)
6. ACTIVITY: What activity is happening? (e.g. cooking, hiking, eating, posing, swimming)
7. TEXT: Any visible text or signs? Quote briefly or "none"
8. TIME_OF_DAY: Estimated time of day and weather if outdoor (e.g. "sunny afternoon", "overcast evening", "indoor")
9. CATEGORIES: List 2-4 categories from: family, friends, travel, nature, food, pets, celebration, selfie, architecture, sports, work, art, vehicles, documents, screenshot, beach, concert, wedding, holiday, garden, cityscape

Respond in this exact format:
SCENE: ...
PEOPLE: ...
LOCATION: ...
MOOD: ...
OBJECTS: ...
ACTIVITY: ...
TEXT: ...
TIME_OF_DAY: ...
CATEGORIES: ..."""

# Vision prompt for video keyframe analysis
_VISION_VIDEO_FRAME_PROMPT = """Analyze this video frame briefly:
1. SCENE: What is happening in this frame? (one sentence)
2. PEOPLE: Are there people? How many? General description (no names).
3. LOCATION: Where does this appear to be? (indoor/outdoor, type of place)
4. MOOD: What mood does this frame convey? (one word)
5. OBJECTS: List specific notable objects visible
6. ACTIVITY: What activity is happening?
7. TEXT: Any visible text or signs? Quote briefly or "none"

Respond in this exact format:
SCENE: ...
PEOPLE: ...
LOCATION: ...
MOOD: ...
OBJECTS: ...
ACTIVITY: ...
TEXT: ..."""


class PhotoDedupService:
    """Manages local photo/video scanning, Takeout import, vision analysis,
    face detection, and Echo Brain ingestion."""

    def __init__(self):
        self.stats = {
            "local_scanned": 0, "local_new": 0, "local_skipped": 0,
            "cloud_fetched": 0, "matched": 0, "analyzed": 0,
            "embedded": 0, "errors": 0
        }
        self._insightface_app = None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    async def _ensure_schema(self, conn: asyncpg.Connection):
        """Create tables if they don't exist."""
        # AGE puts ag_catalog first in search_path; pin DDL to public
        await conn.execute("SET search_path TO public")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS photos (
                id              SERIAL PRIMARY KEY,
                file_path       TEXT UNIQUE NOT NULL,
                filename        TEXT NOT NULL,
                sha256          TEXT,
                perceptual_hash TEXT,
                file_size       BIGINT,
                width           INT,
                height          INT,
                date_taken      TIMESTAMPTZ,
                year_folder     TEXT,
                camera_model    TEXT,
                location_text   TEXT,
                llava_analysis  JSONB,
                llava_description TEXT,
                qdrant_point_id TEXT,
                match_type      TEXT DEFAULT 'unmatched',
                google_photo_id TEXT,
                created_at      TIMESTAMPTZ DEFAULT NOW(),
                updated_at      TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_photos_sha256 ON photos(sha256);
            CREATE INDEX IF NOT EXISTS idx_photos_filename ON photos(filename);
            CREATE INDEX IF NOT EXISTS idx_photos_date ON photos(date_taken);
            CREATE INDEX IF NOT EXISTS idx_photos_match ON photos(match_type);
            CREATE INDEX IF NOT EXISTS idx_photos_qdrant ON photos(qdrant_point_id);
            CREATE INDEX IF NOT EXISTS idx_photos_year ON photos(year_folder);

            CREATE TABLE IF NOT EXISTS google_photos_cloud (
                id              SERIAL PRIMARY KEY,
                google_photo_id TEXT UNIQUE NOT NULL,
                filename        TEXT,
                mime_type       TEXT,
                creation_time   TIMESTAMPTZ,
                width           INT,
                height          INT,
                camera_make     TEXT,
                camera_model    TEXT,
                description     TEXT,
                matched_local_id INT REFERENCES photos(id),
                fetched_at      TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_gpc_filename ON google_photos_cloud(filename);
            CREATE INDEX IF NOT EXISTS idx_gpc_creation ON google_photos_cloud(creation_time);
            CREATE INDEX IF NOT EXISTS idx_gpc_matched ON google_photos_cloud(matched_local_id);

            CREATE TABLE IF NOT EXISTS photo_dedup_runs (
                id              SERIAL PRIMARY KEY,
                run_type        TEXT NOT NULL,
                started_at      TIMESTAMPTZ DEFAULT NOW(),
                finished_at     TIMESTAMPTZ,
                items_processed INT DEFAULT 0,
                items_new       INT DEFAULT 0,
                items_skipped   INT DEFAULT 0,
                items_error     INT DEFAULT 0,
                details         JSONB
            );

            CREATE TABLE IF NOT EXISTS face_clusters (
                id              SERIAL PRIMARY KEY,
                cluster_name    TEXT,
                centroid_embedding BYTEA,
                sample_photo_ids INT[],
                photo_count     INT DEFAULT 0,
                created_at      TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS photo_faces (
                id              SERIAL PRIMARY KEY,
                photo_id        INT REFERENCES photos(id),
                face_index      INT,
                bbox            JSONB,
                embedding       BYTEA,
                cluster_id      INT REFERENCES face_clusters(id),
                confidence      FLOAT,
                created_at      TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_photo_faces_photo ON photo_faces(photo_id);
            CREATE INDEX IF NOT EXISTS idx_photo_faces_cluster ON photo_faces(cluster_id);
        """)

        # Idempotent ALTER TABLE for new columns on existing photos table
        for col_def in [
            ("media_type", "TEXT DEFAULT 'photo'"),
            ("duration_seconds", "FLOAT"),
            ("keyframe_count", "INT"),
            ("source_root", "TEXT"),
            ("takeout_metadata", "JSONB"),
            ("face_embeddings", "JSONB"),
            ("face_count", "INT"),
        ]:
            col_name, col_type = col_def
            try:
                await conn.execute(
                    f"ALTER TABLE photos ADD COLUMN IF NOT EXISTS {col_name} {col_type}"
                )
            except Exception:
                pass  # Column already exists

        # Additional indexes for new columns
        for idx in [
            "CREATE INDEX IF NOT EXISTS idx_photos_media_type ON photos(media_type)",
            "CREATE INDEX IF NOT EXISTS idx_photos_source_root ON photos(source_root)",
            "CREATE INDEX IF NOT EXISTS idx_photos_face_count ON photos(face_count)",
        ]:
            try:
                await conn.execute(idx)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Local scan
    # ------------------------------------------------------------------
    async def scan_local_photos(self, batch_size: int = 500) -> Dict[str, Any]:
        """Walk ~/Pictures/, compute SHA256, extract EXIF, insert to photos table."""
        conn = await asyncpg.connect(DB_URL)
        try:
            await self._ensure_schema(conn)

            run_id = await conn.fetchval(
                "INSERT INTO photo_dedup_runs (run_type) VALUES ('local_scan') RETURNING id")

            scanned = 0
            new = 0
            skipped = 0
            errors = 0

            if not LOCAL_PHOTOS_ROOT.exists():
                logger.error(f"Photos root not found: {LOCAL_PHOTOS_ROOT}")
                return {"error": f"{LOCAL_PHOTOS_ROOT} does not exist"}

            batch = []
            for photo_path in LOCAL_PHOTOS_ROOT.rglob("*"):
                if not photo_path.is_file():
                    continue
                if photo_path.suffix.lower() not in PHOTO_EXTENSIONS:
                    continue
                if any(seg in photo_path.parts for seg in EXCLUDED_PATH_SEGMENTS):
                    continue

                scanned += 1

                # Check if already in DB
                existing = await conn.fetchval(
                    "SELECT id FROM photos WHERE file_path = $1", str(photo_path))
                if existing:
                    skipped += 1
                    continue

                try:
                    data = await self._process_local_photo(photo_path)
                    data["source_root"] = "local"
                    data["media_type"] = "photo"
                    batch.append(data)

                    if len(batch) >= batch_size:
                        inserted = await self._insert_photo_batch(conn, batch)
                        new += inserted
                        batch = []

                except Exception as e:
                    logger.error(f"Error processing {photo_path}: {e}")
                    errors += 1

            # Insert remaining
            if batch:
                inserted = await self._insert_photo_batch(conn, batch)
                new += inserted

            await conn.execute("""
                UPDATE photo_dedup_runs
                SET finished_at = NOW(), items_processed = $2, items_new = $3,
                    items_skipped = $4, items_error = $5
                WHERE id = $1
            """, run_id, scanned, new, skipped, errors)

            result = {
                "scanned": scanned, "new": new, "skipped": skipped,
                "errors": errors, "run_id": run_id
            }
            logger.info(f"[PhotoDedup] Local scan: {result}")
            return result

        finally:
            await conn.close()

    async def _process_local_photo(self, photo_path: Path) -> Dict[str, Any]:
        """Compute SHA256 and extract EXIF for a single photo."""
        sha256 = self._compute_sha256(photo_path)
        file_size = photo_path.stat().st_size

        # Determine year folder from path
        year_folder = None
        for part in photo_path.parts:
            if re.match(r"^\d{4}$", part):
                year_folder = part
                break

        # Extract EXIF
        exif = self._extract_exif(photo_path)

        # Compute perceptual hash
        phash = self._compute_perceptual_hash(photo_path)

        # Get dimensions
        width, height = self._get_dimensions(photo_path)

        return {
            "file_path": str(photo_path),
            "filename": photo_path.name,
            "sha256": sha256,
            "perceptual_hash": phash,
            "file_size": file_size,
            "width": width,
            "height": height,
            "date_taken": exif.get("date_taken"),
            "year_folder": year_folder,
            "camera_model": exif.get("camera_model"),
            "location_text": exif.get("location"),
        }

    async def _insert_photo_batch(self, conn: asyncpg.Connection, batch: List[Dict]) -> int:
        """Insert a batch of photo records."""
        inserted = 0
        for data in batch:
            try:
                await conn.execute("""
                    INSERT INTO photos (file_path, filename, sha256, perceptual_hash,
                        file_size, width, height, date_taken, year_folder,
                        camera_model, location_text, media_type, source_root,
                        duration_seconds, keyframe_count, takeout_metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                    ON CONFLICT (file_path) DO NOTHING
                """,
                    data["file_path"], data["filename"], data.get("sha256"),
                    data.get("perceptual_hash"), data.get("file_size"),
                    data.get("width"), data.get("height"), data.get("date_taken"),
                    data.get("year_folder"), data.get("camera_model"), data.get("location_text"),
                    data.get("media_type", "photo"), data.get("source_root", "local"),
                    data.get("duration_seconds"), data.get("keyframe_count"),
                    json.dumps(data["takeout_metadata"]) if data.get("takeout_metadata") else None,
                )
                inserted += 1
            except Exception as e:
                logger.error(f"Insert error for {data['file_path']}: {e}")
        return inserted

    # ------------------------------------------------------------------
    # Takeout media scanning
    # ------------------------------------------------------------------
    async def scan_takeout_media(self, batch_size: int = 500) -> Dict[str, Any]:
        """Walk Google Takeout photos directory for all media files."""
        if not TAKEOUT_PHOTOS_ROOT.exists():
            return {"error": f"Takeout root not found: {TAKEOUT_PHOTOS_ROOT}"}

        conn = await asyncpg.connect(DB_URL)
        try:
            await self._ensure_schema(conn)

            run_id = await conn.fetchval(
                "INSERT INTO photo_dedup_runs (run_type) VALUES ('takeout_scan') RETURNING id")

            scanned = 0
            new = 0
            skipped = 0
            errors = 0

            batch = []
            for media_path in TAKEOUT_PHOTOS_ROOT.rglob("*"):
                if not media_path.is_file():
                    continue
                suffix = media_path.suffix.lower()
                if suffix not in MEDIA_EXTENSIONS:
                    continue

                scanned += 1

                existing = await conn.fetchval(
                    "SELECT id FROM photos WHERE file_path = $1", str(media_path))
                if existing:
                    skipped += 1
                    continue

                try:
                    is_video = suffix in VIDEO_EXTENSIONS
                    media_type = "video" if is_video else "photo"

                    # Parse companion Takeout JSON
                    takeout_meta = self._parse_takeout_metadata(media_path)

                    # Determine date_taken from Takeout metadata or EXIF
                    date_taken = None
                    if takeout_meta and takeout_meta.get("photoTakenTime"):
                        try:
                            ts = int(takeout_meta["photoTakenTime"]["timestamp"])
                            date_taken = datetime.fromtimestamp(ts)
                        except (ValueError, KeyError, TypeError):
                            pass

                    # Year folder from path
                    year_folder = None
                    for part in media_path.parts:
                        if re.match(r"^\d{4}$", part):
                            year_folder = part
                            break
                    # Try from date_taken
                    if not year_folder and date_taken:
                        year_folder = str(date_taken.year)

                    # Location from Takeout geo
                    location_text = None
                    if takeout_meta and takeout_meta.get("geoData"):
                        geo = takeout_meta["geoData"]
                        lat = geo.get("latitude", 0)
                        lon = geo.get("longitude", 0)
                        if lat != 0 or lon != 0:
                            location_text = f"GPS: {lat}, {lon}"

                    data = {
                        "file_path": str(media_path),
                        "filename": media_path.name,
                        "file_size": media_path.stat().st_size,
                        "media_type": media_type,
                        "source_root": "takeout",
                        "date_taken": date_taken,
                        "year_folder": year_folder,
                        "location_text": location_text,
                        "takeout_metadata": takeout_meta,
                    }

                    if is_video:
                        # Use ffprobe for duration + dimensions
                        probe = self._ffprobe_video(media_path)
                        data["duration_seconds"] = probe.get("duration")
                        data["width"] = probe.get("width")
                        data["height"] = probe.get("height")
                    else:
                        # Photo: SHA256 + EXIF + dimensions
                        data["sha256"] = self._compute_sha256(media_path)
                        if not date_taken:
                            exif = self._extract_exif(media_path)
                            data["date_taken"] = data["date_taken"] or exif.get("date_taken")
                            data["camera_model"] = exif.get("camera_model")
                        data["perceptual_hash"] = self._compute_perceptual_hash(media_path)
                        w, h = self._get_dimensions(media_path)
                        data["width"] = w
                        data["height"] = h

                    batch.append(data)

                    if len(batch) >= batch_size:
                        inserted = await self._insert_photo_batch(conn, batch)
                        new += inserted
                        batch = []

                except Exception as e:
                    logger.error(f"Takeout processing error {media_path}: {e}")
                    errors += 1

            if batch:
                inserted = await self._insert_photo_batch(conn, batch)
                new += inserted

            await conn.execute("""
                UPDATE photo_dedup_runs
                SET finished_at = NOW(), items_processed = $2, items_new = $3,
                    items_skipped = $4, items_error = $5
                WHERE id = $1
            """, run_id, scanned, new, skipped, errors)

            result = {"scanned": scanned, "new": new, "skipped": skipped,
                      "errors": errors, "run_id": run_id}
            logger.info(f"[PhotoDedup] Takeout scan: {result}")
            return result

        finally:
            await conn.close()

    @staticmethod
    def _parse_takeout_metadata(media_path: Path) -> Optional[Dict]:
        """Try all 3 Takeout JSON naming variants for a media file."""
        stem = media_path.name
        parent = media_path.parent

        # Google Takeout JSON naming variants
        candidates = [
            parent / f"{stem}.supplemental-metadata.json",
            parent / f"{stem}.suppl.json",
            parent / f"{stem}.supplement.json",
        ]

        for candidate in candidates:
            if candidate.exists():
                try:
                    return json.loads(candidate.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
        return None

    @staticmethod
    def _ffprobe_video(video_path: Path) -> Dict[str, Any]:
        """Get video duration and dimensions via ffprobe."""
        result = {"duration": None, "width": None, "height": None}
        try:
            proc = subprocess.run(
                [
                    "ffprobe", "-v", "quiet", "-print_format", "json",
                    "-show_format", "-show_streams", str(video_path)
                ],
                capture_output=True, text=True, timeout=15
            )
            if proc.returncode != 0:
                return result

            data = json.loads(proc.stdout)

            # Duration from format
            fmt = data.get("format", {})
            if fmt.get("duration"):
                result["duration"] = float(fmt["duration"])

            # Dimensions from first video stream
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    result["width"] = stream.get("width")
                    result["height"] = stream.get("height")
                    if not result["duration"] and stream.get("duration"):
                        result["duration"] = float(stream["duration"])
                    break

        except Exception as e:
            logger.debug(f"ffprobe failed for {video_path}: {e}")
        return result

    # ------------------------------------------------------------------
    # Cloud metadata fetch
    # ------------------------------------------------------------------
    async def fetch_cloud_metadata(self) -> Dict[str, Any]:
        """Paginate Google Photos API, store metadata to google_photos_cloud."""
        conn = await asyncpg.connect(DB_URL)
        try:
            await self._ensure_schema(conn)

            token = await self._get_google_token()
            if not token:
                return {"error": "No Google OAuth token. Complete OAuth first."}

            run_id = await conn.fetchval(
                "INSERT INTO photo_dedup_runs (run_type) VALUES ('cloud_fetch') RETURNING id")

            fetched = 0
            new = 0
            errors = 0
            next_page_token = None

            async with httpx.AsyncClient(timeout=30.0) as client:
                while True:
                    params = {"pageSize": 100}
                    if next_page_token:
                        params["pageToken"] = next_page_token

                    try:
                        resp = await client.get(
                            f"{GOOGLE_PHOTOS_API}/mediaItems",
                            headers={"Authorization": f"Bearer {token}"},
                            params=params
                        )

                        if resp.status_code == 401:
                            # Token expired, try refresh
                            token = await self._refresh_google_token()
                            if not token:
                                logger.error("Google Photos 401 and refresh failed — need re-auth")
                                errors += 1
                                break
                            continue

                        if resp.status_code != 200:
                            logger.error(f"Google Photos API error: {resp.status_code} {resp.text[:200]}")
                            errors += 1
                            break

                        data = resp.json()
                        items = data.get("mediaItems", [])

                        for item in items:
                            fetched += 1
                            meta = item.get("mediaMetadata", {})
                            creation_time = None
                            if meta.get("creationTime"):
                                try:
                                    creation_time = datetime.fromisoformat(
                                        meta["creationTime"].replace("Z", "+00:00"))
                                except (ValueError, TypeError):
                                    pass

                            photo_meta = meta.get("photo", {})

                            try:
                                await conn.execute("""
                                    INSERT INTO google_photos_cloud
                                        (google_photo_id, filename, mime_type, creation_time,
                                         width, height, camera_make, camera_model, description)
                                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                                    ON CONFLICT (google_photo_id) DO UPDATE SET
                                        filename = EXCLUDED.filename,
                                        creation_time = EXCLUDED.creation_time,
                                        fetched_at = NOW()
                                """,
                                    item.get("id"),
                                    item.get("filename"),
                                    item.get("mimeType"),
                                    creation_time,
                                    int(meta.get("width", 0)) or None,
                                    int(meta.get("height", 0)) or None,
                                    photo_meta.get("cameraMake"),
                                    photo_meta.get("cameraModel"),
                                    item.get("description"),
                                )
                                new += 1
                            except Exception as e:
                                logger.error(f"Cloud insert error: {e}")
                                errors += 1

                        next_page_token = data.get("nextPageToken")
                        if not next_page_token:
                            break

                    except httpx.TimeoutException:
                        logger.error("Google Photos API timeout")
                        errors += 1
                        break

            await conn.execute("""
                UPDATE photo_dedup_runs
                SET finished_at = NOW(), items_processed = $2, items_new = $3, items_error = $4
                WHERE id = $1
            """, run_id, fetched, new, errors)

            result = {"fetched": fetched, "new": new, "errors": errors, "run_id": run_id}
            logger.info(f"[PhotoDedup] Cloud fetch: {result}")
            return result

        finally:
            await conn.close()

    # ------------------------------------------------------------------
    # Dedup matching
    # ------------------------------------------------------------------
    async def run_dedup_matching(self) -> Dict[str, Any]:
        """Three-tier matching: SHA256 local dupes, filename+date+dims, filename-only fallback."""
        conn = await asyncpg.connect(DB_URL)
        try:
            await self._ensure_schema(conn)

            run_id = await conn.fetchval(
                "INSERT INTO photo_dedup_runs (run_type) VALUES ('dedup_match') RETURNING id")

            results = {
                "local_sha256_groups": 0,
                "local_sha256_dupes": 0,
                "cloud_filename_date_dims": 0,
                "cloud_filename_only": 0,
                "only_local": 0,
                "only_cloud": 0,
            }

            # --- Tier 1: Local SHA256 exact dupes ---
            dupe_groups = await conn.fetch("""
                SELECT sha256, COUNT(*) as cnt
                FROM photos
                WHERE sha256 IS NOT NULL
                GROUP BY sha256
                HAVING COUNT(*) > 1
            """)
            for group in dupe_groups:
                results["local_sha256_groups"] += 1
                results["local_sha256_dupes"] += group["cnt"] - 1
                # Mark all but the first (by id) as sha256_dupe
                dupes = await conn.fetch("""
                    SELECT id FROM photos WHERE sha256 = $1 ORDER BY id
                """, group["sha256"])
                for dupe in dupes[1:]:
                    await conn.execute(
                        "UPDATE photos SET match_type = 'sha256_dupe' WHERE id = $1",
                        dupe["id"])

            # --- Tier 2: Local vs Cloud — filename + date + dimensions ---
            matched_2 = await conn.execute("""
                UPDATE photos p
                SET match_type = 'cloud_matched',
                    google_photo_id = g.google_photo_id,
                    updated_at = NOW()
                FROM google_photos_cloud g
                WHERE p.filename = g.filename
                  AND p.date_taken IS NOT NULL
                  AND g.creation_time IS NOT NULL
                  AND ABS(EXTRACT(EPOCH FROM (p.date_taken - g.creation_time))) < 86400
                  AND p.width = g.width
                  AND p.height = g.height
                  AND p.match_type = 'unmatched'
                  AND g.matched_local_id IS NULL
            """)
            results["cloud_filename_date_dims"] = int(matched_2.split()[-1]) if matched_2 else 0

            # Link cloud records back
            await conn.execute("""
                UPDATE google_photos_cloud g
                SET matched_local_id = p.id
                FROM photos p
                WHERE p.google_photo_id = g.google_photo_id
                  AND g.matched_local_id IS NULL
            """)

            # --- Tier 3: Filename-only fallback ---
            matched_3 = await conn.execute("""
                UPDATE photos p
                SET match_type = 'cloud_filename_match',
                    google_photo_id = g.google_photo_id,
                    updated_at = NOW()
                FROM google_photos_cloud g
                WHERE p.filename = g.filename
                  AND p.match_type = 'unmatched'
                  AND g.matched_local_id IS NULL
            """)
            results["cloud_filename_only"] = int(matched_3.split()[-1]) if matched_3 else 0

            await conn.execute("""
                UPDATE google_photos_cloud g
                SET matched_local_id = p.id
                FROM photos p
                WHERE p.google_photo_id = g.google_photo_id
                  AND g.matched_local_id IS NULL
            """)

            # --- Count unmatched ---
            results["only_local"] = await conn.fetchval(
                "SELECT COUNT(*) FROM photos WHERE match_type = 'unmatched'")
            results["only_cloud"] = await conn.fetchval(
                "SELECT COUNT(*) FROM google_photos_cloud WHERE matched_local_id IS NULL")

            await conn.execute("""
                UPDATE photo_dedup_runs
                SET finished_at = NOW(), items_processed = $2,
                    details = $3::jsonb
                WHERE id = $1
            """, run_id,
                results["cloud_filename_date_dims"] + results["cloud_filename_only"],
                json.dumps(results))

            logger.info(f"[PhotoDedup] Matching: {results}")
            return results

        finally:
            await conn.close()

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    async def generate_report(self) -> Dict[str, Any]:
        """Summary stats with by-year breakdown and top dupe groups."""
        conn = await asyncpg.connect(DB_URL)
        try:
            await self._ensure_schema(conn)

            total_local = await conn.fetchval("SELECT COUNT(*) FROM photos")
            total_cloud = await conn.fetchval("SELECT COUNT(*) FROM google_photos_cloud")

            match_breakdown = await conn.fetch("""
                SELECT match_type, COUNT(*) as cnt
                FROM photos GROUP BY match_type ORDER BY cnt DESC
            """)

            by_year = await conn.fetch("""
                SELECT year_folder, COUNT(*) as cnt,
                       COUNT(*) FILTER (WHERE llava_description IS NOT NULL) as analyzed,
                       COUNT(*) FILTER (WHERE qdrant_point_id IS NOT NULL) as embedded
                FROM photos
                WHERE year_folder IS NOT NULL
                GROUP BY year_folder ORDER BY year_folder
            """)

            top_dupe_groups = await conn.fetch("""
                SELECT sha256, COUNT(*) as cnt,
                       ARRAY_AGG(file_path ORDER BY id) as paths
                FROM photos
                WHERE sha256 IN (
                    SELECT sha256 FROM photos
                    WHERE sha256 IS NOT NULL
                    GROUP BY sha256 HAVING COUNT(*) > 1
                )
                GROUP BY sha256
                ORDER BY cnt DESC
                LIMIT 10
            """)

            recent_runs = await conn.fetch("""
                SELECT id, run_type, started_at, finished_at,
                       items_processed, items_new, items_skipped, items_error
                FROM photo_dedup_runs ORDER BY id DESC LIMIT 10
            """)

            # Media type breakdown
            media_breakdown = await conn.fetch("""
                SELECT media_type, source_root, COUNT(*) as cnt
                FROM photos
                GROUP BY media_type, source_root
                ORDER BY cnt DESC
            """)

            return {
                "total_local": total_local,
                "total_cloud": total_cloud,
                "match_breakdown": {r["match_type"]: r["cnt"] for r in match_breakdown},
                "media_breakdown": [dict(r) for r in media_breakdown],
                "by_year": [dict(r) for r in by_year],
                "top_dupe_groups": [
                    {"sha256": r["sha256"][:16] + "...", "count": r["cnt"],
                     "paths": r["paths"][:5]}
                    for r in top_dupe_groups
                ],
                "recent_runs": [dict(r) for r in recent_runs],
            }
        finally:
            await conn.close()

    # ------------------------------------------------------------------
    # Vision analysis (Gemma 3) — photos and videos
    # ------------------------------------------------------------------
    async def analyze_photos_batch(self, batch_size: int = 50,
                                   media_type: Optional[str] = None) -> Dict[str, Any]:
        """Run vision analysis on un-analyzed media, newest first.
        media_type: None (both), "photo", or "video"
        """
        conn = await asyncpg.connect(DB_URL)
        try:
            await self._ensure_schema(conn)

            type_filter = ""
            if media_type == "photo":
                type_filter = "AND (media_type = 'photo' OR media_type IS NULL)"
            elif media_type == "video":
                type_filter = "AND media_type = 'video'"

            photos = await conn.fetch(f"""
                SELECT id, file_path, filename, year_folder, media_type
                FROM photos
                WHERE llava_description IS NULL
                  AND match_type != 'sha256_dupe'
                  {type_filter}
                ORDER BY date_taken DESC NULLS LAST, id DESC
                LIMIT $1
            """, batch_size)

            if not photos:
                return {"analyzed": 0, "message": f"No {media_type or 'media'} pending analysis"}

            run_id = await conn.fetchval(
                "INSERT INTO photo_dedup_runs (run_type) VALUES ('vision_analysis') RETURNING id")

            analyzed = 0
            errors = 0

            # Free VRAM: evict mistral:7b so gemma3:12b has headroom
            await self._prepare_gpu_for_vision()

            for photo in photos:
                try:
                    is_video = photo.get("media_type") == "video"
                    if is_video:
                        result = await self._analyze_single_video(photo["file_path"])
                    else:
                        result = await self._analyze_single_photo(photo["file_path"])

                    if result:
                        update_fields = {
                            "llava_analysis": json.dumps(result),
                            "llava_description": result.get("description", ""),
                        }
                        # For videos, also store keyframe_count
                        if is_video and result.get("keyframe_count"):
                            await conn.execute("""
                                UPDATE photos
                                SET llava_analysis = $2::jsonb,
                                    llava_description = $3,
                                    keyframe_count = $4,
                                    updated_at = NOW()
                                WHERE id = $1
                            """, photo["id"],
                                update_fields["llava_analysis"],
                                update_fields["llava_description"],
                                result["keyframe_count"])
                        else:
                            await conn.execute("""
                                UPDATE photos
                                SET llava_analysis = $2::jsonb,
                                    llava_description = $3,
                                    updated_at = NOW()
                                WHERE id = $1
                            """, photo["id"],
                                update_fields["llava_analysis"],
                                update_fields["llava_description"])
                        analyzed += 1
                    else:
                        errors += 1
                except Exception as e:
                    logger.error(f"Vision analysis error for {photo['file_path']}: {e}")
                    errors += 1

            # Let gemma3:12b unload after 5min idle, freeing VRAM for mistral
            await self._release_vision_model()

            await conn.execute("""
                UPDATE photo_dedup_runs
                SET finished_at = NOW(), items_processed = $2, items_new = $3, items_error = $4
                WHERE id = $1
            """, run_id, len(photos), analyzed, errors)

            result = {"analyzed": analyzed, "errors": errors, "batch_size": len(photos),
                      "media_type": media_type or "all", "run_id": run_id}
            logger.info(f"[PhotoDedup] Vision analysis: {result}")
            return result

        finally:
            await conn.close()

    async def _prepare_gpu_for_vision(self):
        """Evict mistral:7b from VRAM to make room for gemma3:12b vision model.
        nomic-embed-text (0.3GB) stays pinned; freeing mistral (5.1GB) gives
        gemma3 ~15.7GB of headroom on the 16GB AMD RX 9070 XT.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": "mistral:7b", "keep_alive": 0, "prompt": "", "stream": False},
                )
                if resp.status_code == 200:
                    logger.info("[PhotoDedup] Unloading mistral:7b for vision analysis")
                else:
                    logger.warning(f"[PhotoDedup] Failed to unload mistral:7b: {resp.status_code}")
        except Exception as e:
            logger.warning(f"[PhotoDedup] Could not unload mistral:7b: {e}")

    async def _release_vision_model(self):
        """Set gemma3:12b to a 5-minute idle timeout so it unloads after the
        vision batch completes, freeing VRAM for other models."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": VISION_MODEL, "keep_alive": "5m", "prompt": "", "stream": False},
                )
                if resp.status_code == 200:
                    logger.info(f"[PhotoDedup] Set {VISION_MODEL} keep_alive=5m (will unload after idle)")
                else:
                    logger.warning(f"[PhotoDedup] Failed to set {VISION_MODEL} keep_alive: {resp.status_code}")
        except Exception as e:
            logger.warning(f"[PhotoDedup] Could not set {VISION_MODEL} keep_alive: {e}")

    async def _analyze_single_photo(self, file_path: str) -> Optional[Dict]:
        """Send a photo to Gemma 3 for vision analysis."""
        path = Path(file_path)
        if not path.exists():
            return None

        if path.stat().st_size > 20_000_000:  # Skip >20MB
            return None

        # Convert to JPEG for Ollama compatibility (HEIC, TIFF, BMP not supported natively)
        image_b64 = self._load_image_as_jpeg_b64(path)
        if not image_b64:
            return None

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": VISION_MODEL,
                    "prompt": _VISION_PHOTO_PROMPT,
                    "images": [image_b64],
                    "stream": False,
                    "keep_alive": "5m",
                    "options": {"temperature": 0.3},
                }
            )
            if resp.status_code != 200:
                logger.error(f"Vision API error: {resp.status_code}")
                return None

            raw = resp.json().get("response", "")

        return self._parse_vision_response(raw, path)

    async def _analyze_single_video(self, file_path: str) -> Optional[Dict]:
        """Extract keyframes from video, analyze each with Gemma 3, aggregate."""
        path = Path(file_path)
        if not path.exists():
            return None

        frames, duration = self._extract_keyframes(path)
        if not frames:
            return None

        try:
            frame_analyses = []
            for frame_path in frames:
                analysis = await self._analyze_video_frame(frame_path)
                if analysis:
                    frame_analyses.append(analysis)

            if not frame_analyses:
                return None

            return self._aggregate_video_analysis(
                frame_analyses, duration, len(frames))
        finally:
            # Clean up temp frames
            for frame_path in frames:
                try:
                    frame_path.unlink(missing_ok=True)
                except Exception:
                    pass
            # Remove temp directory
            if frames:
                try:
                    frames[0].parent.rmdir()
                except Exception:
                    pass

    def _extract_keyframes(self, video_path: Path, interval: int = 10) -> Tuple[List[Path], float]:
        """Extract keyframes using ffmpeg, 1 frame per interval seconds.
        Returns (list of frame paths, duration in seconds).
        """
        try:
            # Get duration first
            probe = self._ffprobe_video(video_path)
            duration = probe.get("duration", 0) or 0

            # Create temp directory for frames
            tmp_dir = Path(tempfile.mkdtemp(prefix="echo_keyframes_"))

            proc = subprocess.run(
                [
                    "ffmpeg", "-i", str(video_path),
                    "-vf", f"fps=1/{interval}",
                    "-frames:v", str(MAX_VIDEO_KEYFRAMES),
                    "-q:v", "2",
                    str(tmp_dir / "frame_%04d.jpg"),
                ],
                capture_output=True, timeout=120
            )

            if proc.returncode != 0:
                logger.warning(f"ffmpeg keyframe extraction failed for {video_path}")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return [], duration

            frames = sorted(tmp_dir.glob("frame_*.jpg"))
            return frames, duration

        except Exception as e:
            logger.error(f"Keyframe extraction error for {video_path}: {e}")
            return [], 0

    async def _analyze_video_frame(self, frame_path: Path) -> Optional[Dict]:
        """Analyze a single video keyframe with Gemma 3."""
        image_b64 = self._load_image_as_jpeg_b64(frame_path)
        if not image_b64:
            return None

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": VISION_MODEL,
                    "prompt": _VISION_VIDEO_FRAME_PROMPT,
                    "images": [image_b64],
                    "stream": False,
                    "keep_alive": "5m",
                    "options": {"temperature": 0.3},
                }
            )
            if resp.status_code != 200:
                return None

            raw = resp.json().get("response", "")

        return self._parse_video_frame_response(raw)

    def _parse_video_frame_response(self, raw: str) -> Dict:
        """Parse video frame vision response."""
        result = {
            "scene": "", "people": "", "location": "",
            "mood": "", "objects": "", "activity": "", "text": "",
        }

        for line in raw.split("\n"):
            line_s = line.strip()
            upper = line_s.upper()
            if upper.startswith("SCENE:"):
                result["scene"] = line_s.split(":", 1)[1].strip()
            elif upper.startswith("PEOPLE:"):
                result["people"] = line_s.split(":", 1)[1].strip()
            elif upper.startswith("LOCATION:"):
                result["location"] = line_s.split(":", 1)[1].strip()
            elif upper.startswith("MOOD:"):
                result["mood"] = line_s.split(":", 1)[1].strip()
            elif upper.startswith("OBJECTS:"):
                result["objects"] = line_s.split(":", 1)[1].strip()
            elif upper.startswith("ACTIVITY:"):
                result["activity"] = line_s.split(":", 1)[1].strip()
            elif upper.startswith("TEXT:"):
                result["text"] = line_s.split(":", 1)[1].strip()

        return result

    def _aggregate_video_analysis(self, frame_analyses: List[Dict],
                                   duration: float, keyframe_count: int) -> Dict:
        """Combine frame analyses into a unified video analysis."""
        all_scenes = []
        all_people = set()
        all_locations = set()
        all_moods = []
        all_objects = set()
        all_activities = set()
        all_text = set()

        for fa in frame_analyses:
            if fa.get("scene"):
                all_scenes.append(fa["scene"])
            if fa.get("people") and fa["people"].lower() not in ("none", "no", "no people"):
                all_people.add(fa["people"])
            if fa.get("location"):
                all_locations.add(fa["location"])
            if fa.get("mood"):
                all_moods.append(fa["mood"].lower())
            if fa.get("objects"):
                for obj in fa["objects"].split(","):
                    obj = obj.strip().lower()
                    if obj and obj != "none":
                        all_objects.add(obj)
            if fa.get("activity"):
                act = fa["activity"].strip().lower()
                if act and act != "none":
                    all_activities.add(act)
            if fa.get("text"):
                txt = fa["text"].strip()
                if txt.lower() not in ("none", "no text", "no visible text"):
                    all_text.add(txt)

        # Dominant mood (most frequent)
        dominant_mood = ""
        if all_moods:
            mood_counts = defaultdict(int)
            for m in all_moods:
                mood_counts[m] += 1
            dominant_mood = max(mood_counts, key=mood_counts.get)

        # Infer categories from content
        categories = set()
        combined_text = " ".join(all_scenes).lower()
        category_keywords = {
            "travel": ["travel", "beach", "mountain", "tourist", "landmark", "hotel"],
            "family": ["family", "children", "baby", "kid"],
            "nature": ["nature", "forest", "lake", "river", "tree", "sunset", "ocean"],
            "food": ["food", "restaurant", "cooking", "eating", "dinner", "lunch"],
            "celebration": ["party", "celebration", "birthday", "wedding", "cake"],
            "sports": ["sport", "running", "swimming", "game", "playing"],
            "pets": ["dog", "cat", "pet", "animal"],
            "cityscape": ["city", "building", "street", "downtown"],
        }
        for cat, keywords in category_keywords.items():
            if any(kw in combined_text for kw in keywords):
                categories.add(cat)
        if not categories:
            categories.add("video")

        # Build description
        parts = []
        # Use first 3 unique scenes
        unique_scenes = list(dict.fromkeys(all_scenes))[:3]
        if unique_scenes:
            parts.append("Video: " + "; ".join(unique_scenes))
        if all_people:
            parts.append(f"People: {', '.join(list(all_people)[:3])}")
        if all_locations:
            parts.append(f"Location: {', '.join(list(all_locations)[:2])}")
        if dominant_mood:
            parts.append(f"Mood: {dominant_mood}")
        if all_objects:
            parts.append(f"Objects: {', '.join(list(all_objects)[:6])}")
        if all_activities:
            parts.append(f"Activities: {', '.join(list(all_activities)[:3])}")
        if duration:
            mins = int(duration // 60)
            secs = int(duration % 60)
            parts.append(f"Duration: {mins}m{secs}s")

        return {
            "scene": "; ".join(unique_scenes),
            "people": ", ".join(list(all_people)[:3]) if all_people else "None",
            "location": ", ".join(list(all_locations)[:2]),
            "mood": dominant_mood,
            "objects": list(all_objects),
            "activities": list(all_activities),
            "text_visible": list(all_text),
            "categories": list(categories),
            "duration_seconds": duration,
            "keyframe_count": keyframe_count,
            "frames_analyzed": len(frame_analyses),
            "description": ". ".join(parts) if parts else "Video",
            "raw": f"Aggregated from {len(frame_analyses)} frames",
        }

    def _parse_vision_response(self, raw: str, path: Path) -> Dict:
        """Parse structured vision response into dict (enhanced with new fields)."""
        result = {
            "scene": "", "people": "", "location": "",
            "mood": "", "categories": [], "objects": "",
            "activity": "", "text": "", "time_of_day": "",
            "raw": raw
        }

        for line in raw.split("\n"):
            line_s = line.strip()
            upper = line_s.upper()
            if upper.startswith("SCENE:"):
                result["scene"] = line_s.split(":", 1)[1].strip()
            elif upper.startswith("PEOPLE:"):
                result["people"] = line_s.split(":", 1)[1].strip()
            elif upper.startswith("LOCATION:"):
                result["location"] = line_s.split(":", 1)[1].strip()
            elif upper.startswith("MOOD:"):
                result["mood"] = line_s.split(":", 1)[1].strip()
            elif upper.startswith("OBJECTS:"):
                result["objects"] = line_s.split(":", 1)[1].strip()
            elif upper.startswith("ACTIVITY:"):
                result["activity"] = line_s.split(":", 1)[1].strip()
            elif upper.startswith("TEXT:"):
                result["text"] = line_s.split(":", 1)[1].strip()
            elif upper.startswith("TIME_OF_DAY:"):
                result["time_of_day"] = line_s.split(":", 1)[1].strip()
            elif upper.startswith("CATEGORIES:"):
                cats_str = line_s.split(":", 1)[1].strip()
                result["categories"] = [c.strip().lower() for c in cats_str.split(",") if c.strip()]

        # Build a searchable description
        parts = []
        if result["scene"]:
            parts.append(result["scene"])
        if result["people"]:
            parts.append(f"People: {result['people']}")
        if result["location"]:
            parts.append(f"Location: {result['location']}")
        if result["mood"]:
            parts.append(f"Mood: {result['mood']}")
        if result["objects"] and result["objects"].lower() != "none":
            parts.append(f"Objects: {result['objects']}")
        if result["activity"] and result["activity"].lower() != "none":
            parts.append(f"Activity: {result['activity']}")
        if result["time_of_day"]:
            parts.append(f"Time: {result['time_of_day']}")

        year_match = re.search(r"/(\d{4})/", str(path))
        if year_match:
            parts.append(f"Year: {year_match.group(1)}")

        result["description"] = ". ".join(parts) if parts else path.name

        return result

    @staticmethod
    def _load_image_as_jpeg_b64(path: Path) -> Optional[str]:
        """Load any image format and return as JPEG base64 for Ollama.
        Handles HEIC, TIFF, BMP, and other formats Ollama can't read natively.
        """
        try:
            from PIL import Image
            # Register HEIC opener if available
            try:
                from pillow_heif import register_heif_opener
                register_heif_opener()
            except ImportError:
                pass

            suffix = path.suffix.lower()
            # Formats Ollama handles natively
            native_formats = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

            if suffix in native_formats:
                # Send as-is for native formats
                return base64.b64encode(path.read_bytes()).decode("utf-8")

            # Convert non-native formats (HEIC, TIFF, BMP, etc.) to JPEG
            import io
            with Image.open(path) as img:
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                return base64.b64encode(buf.getvalue()).decode("utf-8")

        except Exception as e:
            logger.error(f"Failed to load image {path}: {e}")
            return None

    # ------------------------------------------------------------------
    # Face detection pipeline (InsightFace)
    # ------------------------------------------------------------------
    def _get_insightface_app(self):
        """Lazily initialize InsightFace model (downloads buffalo_l on first use)."""
        if self._insightface_app is None:
            try:
                import insightface
                self._insightface_app = insightface.app.FaceAnalysis(
                    name="buffalo_l",
                    providers=["CPUExecutionProvider"],
                )
                self._insightface_app.prepare(ctx_id=-1, det_size=(640, 640))
                logger.info("[PhotoDedup] InsightFace buffalo_l loaded (CPU)")
            except Exception as e:
                logger.error(f"InsightFace init failed: {e}")
                return None
        return self._insightface_app

    async def detect_faces_batch(self, batch_size: int = 100) -> Dict[str, Any]:
        """Detect faces in photos that have been vision-analyzed but not face-scanned."""
        app = self._get_insightface_app()
        if app is None:
            return {"error": "InsightFace not available"}

        conn = await asyncpg.connect(DB_URL)
        try:
            await self._ensure_schema(conn)

            photos = await conn.fetch("""
                SELECT id, file_path, filename
                FROM photos
                WHERE llava_description IS NOT NULL
                  AND face_count IS NULL
                  AND (media_type = 'photo' OR media_type IS NULL)
                  AND match_type != 'sha256_dupe'
                ORDER BY id
                LIMIT $1
            """, batch_size)

            if not photos:
                return {"processed": 0, "message": "No photos pending face detection"}

            run_id = await conn.fetchval(
                "INSERT INTO photo_dedup_runs (run_type) VALUES ('face_detection') RETURNING id")

            processed = 0
            total_faces = 0
            errors = 0

            import cv2
            import numpy as np

            for photo in photos:
                try:
                    path = Path(photo["file_path"])
                    if not path.exists():
                        await conn.execute(
                            "UPDATE photos SET face_count = 0 WHERE id = $1", photo["id"])
                        processed += 1
                        continue

                    img = cv2.imread(str(path))
                    if img is None:
                        await conn.execute(
                            "UPDATE photos SET face_count = 0 WHERE id = $1", photo["id"])
                        processed += 1
                        continue

                    faces = app.get(img)
                    face_count = len(faces)

                    for i, face in enumerate(faces):
                        bbox = face.bbox.tolist()
                        embedding_bytes = face.embedding.astype(np.float32).tobytes()
                        confidence = float(face.det_score) if hasattr(face, 'det_score') else None

                        await conn.execute("""
                            INSERT INTO photo_faces (photo_id, face_index, bbox, embedding, confidence)
                            VALUES ($1, $2, $3::jsonb, $4, $5)
                        """, photo["id"], i,
                            json.dumps({"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]}),
                            embedding_bytes, confidence)

                    await conn.execute(
                        "UPDATE photos SET face_count = $2, updated_at = NOW() WHERE id = $1",
                        photo["id"], face_count)

                    total_faces += face_count
                    processed += 1

                except Exception as e:
                    logger.error(f"Face detection error for {photo['file_path']}: {e}")
                    errors += 1

            await conn.execute("""
                UPDATE photo_dedup_runs
                SET finished_at = NOW(), items_processed = $2, items_new = $3, items_error = $4
                WHERE id = $1
            """, run_id, processed, total_faces, errors)

            result = {"processed": processed, "faces_detected": total_faces,
                      "errors": errors, "run_id": run_id}
            logger.info(f"[PhotoDedup] Face detection: {result}")
            return result

        finally:
            await conn.close()

    async def cluster_faces(self, distance_threshold: float = 0.4) -> Dict[str, Any]:
        """Cluster face embeddings using cosine distance."""
        conn = await asyncpg.connect(DB_URL)
        try:
            await self._ensure_schema(conn)
            import numpy as np

            # Load all face embeddings
            faces = await conn.fetch("""
                SELECT id, embedding FROM photo_faces
                WHERE embedding IS NOT NULL
                ORDER BY id
            """)

            if not faces:
                return {"clusters": 0, "message": "No face embeddings to cluster"}

            # Parse embeddings
            face_ids = []
            embeddings = []
            for face in faces:
                face_ids.append(face["id"])
                emb = np.frombuffer(face["embedding"], dtype=np.float32)
                # Normalize for cosine distance
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                embeddings.append(emb)

            embeddings = np.array(embeddings)

            # Simple greedy clustering
            clusters = []  # list of (centroid, face_id_list)
            assigned = [False] * len(face_ids)

            for i in range(len(face_ids)):
                if assigned[i]:
                    continue

                # Start a new cluster
                cluster_ids = [i]
                assigned[i] = True
                centroid = embeddings[i].copy()

                for j in range(i + 1, len(face_ids)):
                    if assigned[j]:
                        continue
                    # Cosine distance = 1 - cosine_similarity
                    sim = np.dot(centroid, embeddings[j])
                    if (1 - sim) < distance_threshold:
                        cluster_ids.append(j)
                        assigned[j] = True
                        # Update centroid (running average)
                        n = len(cluster_ids)
                        centroid = centroid * ((n - 1) / n) + embeddings[j] / n
                        norm = np.linalg.norm(centroid)
                        if norm > 0:
                            centroid = centroid / norm

                clusters.append((centroid, cluster_ids))

            # Store clusters in DB
            # Clear existing clusters
            await conn.execute("UPDATE photo_faces SET cluster_id = NULL")
            await conn.execute("DELETE FROM face_clusters")

            cluster_count = 0
            for centroid, member_indices in clusters:
                member_face_ids = [face_ids[idx] for idx in member_indices]

                # Get sample photo_ids for this cluster
                sample_photos = await conn.fetch("""
                    SELECT DISTINCT photo_id FROM photo_faces
                    WHERE id = ANY($1::int[])
                    LIMIT 5
                """, member_face_ids)
                sample_photo_ids = [r["photo_id"] for r in sample_photos]

                cluster_id = await conn.fetchval("""
                    INSERT INTO face_clusters (centroid_embedding, sample_photo_ids, photo_count)
                    VALUES ($1, $2, $3)
                    RETURNING id
                """, centroid.astype(np.float32).tobytes(),
                    sample_photo_ids,
                    len(set(r["photo_id"] for r in await conn.fetch(
                        "SELECT photo_id FROM photo_faces WHERE id = ANY($1::int[])", member_face_ids
                    ))))

                # Update face records
                await conn.execute("""
                    UPDATE photo_faces SET cluster_id = $1
                    WHERE id = ANY($2::int[])
                """, cluster_id, member_face_ids)

                cluster_count += 1

            result = {"clusters": cluster_count, "total_faces": len(face_ids)}
            logger.info(f"[PhotoDedup] Face clustering: {result}")
            return result

        finally:
            await conn.close()

    async def name_face_cluster(self, cluster_id: int, name: str) -> Dict[str, Any]:
        """Assign a name to a face cluster."""
        conn = await asyncpg.connect(DB_URL)
        try:
            result = await conn.execute(
                "UPDATE face_clusters SET cluster_name = $2 WHERE id = $1",
                cluster_id, name)
            if "UPDATE 0" in result:
                return {"error": f"Cluster {cluster_id} not found"}
            return {"cluster_id": cluster_id, "name": name, "status": "updated"}
        finally:
            await conn.close()

    async def get_face_clusters(self) -> List[Dict[str, Any]]:
        """Return all face clusters with counts and sample photos."""
        conn = await asyncpg.connect(DB_URL)
        try:
            await self._ensure_schema(conn)

            clusters = await conn.fetch("""
                SELECT fc.id, fc.cluster_name, fc.photo_count, fc.sample_photo_ids,
                       fc.created_at
                FROM face_clusters fc
                ORDER BY fc.photo_count DESC
            """)

            results = []
            for c in clusters:
                # Get sample file paths
                sample_paths = []
                if c["sample_photo_ids"]:
                    photos = await conn.fetch("""
                        SELECT file_path, filename FROM photos
                        WHERE id = ANY($1::int[])
                    """, c["sample_photo_ids"])
                    sample_paths = [p["file_path"] for p in photos]

                results.append({
                    "id": c["id"],
                    "name": c["cluster_name"] or f"Person #{c['id']}",
                    "photo_count": c["photo_count"],
                    "sample_photos": sample_paths,
                    "created_at": c["created_at"].isoformat() if c["created_at"] else None,
                })

            return results
        finally:
            await conn.close()

    # ------------------------------------------------------------------
    # Qdrant ingestion
    # ------------------------------------------------------------------
    async def ingest_to_qdrant(self, batch_size: int = 100) -> Dict[str, Any]:
        """Embed analyzed media metadata with nomic-embed-text, store in echo_memory."""
        conn = await asyncpg.connect(DB_URL)
        try:
            await self._ensure_schema(conn)

            photos = await conn.fetch("""
                SELECT p.id, p.file_path, p.filename, p.year_folder, p.date_taken,
                       p.camera_model, p.llava_description, p.llava_analysis,
                       p.match_type, p.width, p.height, p.media_type,
                       p.duration_seconds, p.keyframe_count, p.source_root,
                       p.takeout_metadata, p.location_text, p.face_count
                FROM photos p
                WHERE p.llava_description IS NOT NULL
                  AND p.qdrant_point_id IS NULL
                  AND p.match_type != 'sha256_dupe'
                ORDER BY p.date_taken DESC NULLS LAST
                LIMIT $1
            """, batch_size)

            if not photos:
                return {"embedded": 0, "message": "No media pending embedding"}

            run_id = await conn.fetchval(
                "INSERT INTO photo_dedup_runs (run_type) VALUES ('qdrant_ingest') RETURNING id")

            # Pre-fetch face cluster names for photos with faces
            face_cluster_map = {}
            photo_ids_with_faces = [p["id"] for p in photos if p.get("face_count") and p["face_count"] > 0]
            if photo_ids_with_faces:
                face_data = await conn.fetch("""
                    SELECT pf.photo_id, fc.cluster_name
                    FROM photo_faces pf
                    JOIN face_clusters fc ON pf.cluster_id = fc.id
                    WHERE pf.photo_id = ANY($1::int[])
                      AND fc.cluster_name IS NOT NULL
                """, photo_ids_with_faces)
                for fd in face_data:
                    face_cluster_map.setdefault(fd["photo_id"], set()).add(fd["cluster_name"])

            embedded = 0
            errors = 0

            for photo in photos:
                try:
                    people_names = list(face_cluster_map.get(photo["id"], []))
                    text = self._build_embedding_text(photo, people_names)
                    point_id = await self._embed_and_store(text, photo, people_names)

                    if point_id:
                        await conn.execute("""
                            UPDATE photos SET qdrant_point_id = $2, updated_at = NOW()
                            WHERE id = $1
                        """, photo["id"], point_id)
                        embedded += 1
                    else:
                        errors += 1

                except Exception as e:
                    logger.error(f"Qdrant ingest error for {photo['file_path']}: {e}")
                    errors += 1

            await conn.execute("""
                UPDATE photo_dedup_runs
                SET finished_at = NOW(), items_processed = $2, items_new = $3, items_error = $4
                WHERE id = $1
            """, run_id, len(photos), embedded, errors)

            result = {"embedded": embedded, "errors": errors, "batch_size": len(photos), "run_id": run_id}
            logger.info(f"[PhotoDedup] Qdrant ingest: {result}")
            return result

        finally:
            await conn.close()

    def _build_embedding_text(self, photo: dict, people_names: Optional[List[str]] = None) -> str:
        """Build rich text for embedding from photo/video metadata."""
        media_type = photo.get("media_type", "photo") or "photo"
        parts = [f"Personal {media_type}: {photo['filename']}"]

        if photo.get("llava_description"):
            parts.append(photo["llava_description"])

        if photo.get("year_folder"):
            parts.append(f"Taken in {photo['year_folder']}")

        if photo.get("date_taken"):
            parts.append(f"Date: {photo['date_taken'].strftime('%Y-%m-%d')}")

        if photo.get("camera_model"):
            parts.append(f"Camera: {photo['camera_model']}")

        if photo.get("width") and photo.get("height"):
            parts.append(f"Resolution: {photo['width']}x{photo['height']}")

        # Video-specific fields
        if media_type == "video":
            if photo.get("duration_seconds"):
                dur = photo["duration_seconds"]
                mins = int(dur // 60)
                secs = int(dur % 60)
                parts.append(f"Duration: {mins}m{secs}s")
            if photo.get("keyframe_count"):
                parts.append(f"Keyframes analyzed: {photo['keyframe_count']}")

        # Categories, objects, activities from analysis
        analysis = photo.get("llava_analysis")
        if analysis and isinstance(analysis, dict):
            cats = analysis.get("categories", [])
            if cats:
                parts.append(f"Categories: {', '.join(cats)}")
            # Objects (can be string or list)
            objects = analysis.get("objects", "")
            if objects:
                if isinstance(objects, list):
                    objects = ", ".join(objects)
                if objects.lower() != "none":
                    parts.append(f"Objects: {objects}")
            # Activities
            activities = analysis.get("activities", analysis.get("activity", ""))
            if activities:
                if isinstance(activities, list):
                    activities = ", ".join(activities)
                if activities.lower() != "none":
                    parts.append(f"Activities: {activities}")

        # Takeout geo/description
        if photo.get("location_text"):
            parts.append(f"Location: {photo['location_text']}")
        takeout = photo.get("takeout_metadata")
        if takeout and isinstance(takeout, dict):
            desc = takeout.get("description")
            if desc:
                parts.append(f"Description: {desc}")

        # Face cluster names
        if people_names:
            parts.append(f"People: {', '.join(people_names)}")

        return "\n".join(parts)

    async def _embed_and_store(self, text: str, photo: dict,
                                people_names: Optional[List[str]] = None) -> Optional[str]:
        """Embed text and store in Qdrant echo_memory."""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{OLLAMA_URL}/api/embeddings",
                    json={"model": EMBED_MODEL, "prompt": text}
                )
                if resp.status_code != 200:
                    return None
                embedding = resp.json().get("embedding")
                if not embedding:
                    return None

            point_id = str(uuid4())

            # Parse categories from analysis JSONB
            categories = []
            mood = ""
            objects = []
            activities = []
            analysis = photo.get("llava_analysis")
            if analysis and isinstance(analysis, dict):
                categories = analysis.get("categories", [])
                mood = analysis.get("mood", "")
                raw_objects = analysis.get("objects", "")
                if isinstance(raw_objects, list):
                    objects = raw_objects
                elif raw_objects and raw_objects.lower() != "none":
                    objects = [o.strip() for o in raw_objects.split(",") if o.strip()]
                raw_activities = analysis.get("activities", analysis.get("activity", ""))
                if isinstance(raw_activities, list):
                    activities = raw_activities
                elif raw_activities and raw_activities.lower() != "none":
                    activities = [a.strip() for a in raw_activities.split(",") if a.strip()]

            media_type = photo.get("media_type", "photo") or "photo"

            payload = {
                "content": text[:10000],
                "type": media_type,
                "category": "photos:personal",
                "source": photo["file_path"],
                "filename": photo["filename"],
                "year": photo.get("year_folder"),
                "date_taken": photo["date_taken"].isoformat() if photo.get("date_taken") else None,
                "camera": photo.get("camera_model"),
                "match_type": photo.get("match_type"),
                "photo_categories": categories,
                "resolution": f"{photo.get('width', '?')}x{photo.get('height', '?')}",
                "ingested_at": datetime.now().isoformat(),
                "mood": mood,
                "objects": objects,
                "activities": activities,
                "people": people_names or [],
                "source_root": photo.get("source_root"),
            }

            # Video-specific payload fields
            if media_type == "video":
                payload["duration_seconds"] = photo.get("duration_seconds")
                payload["keyframe_count"] = photo.get("keyframe_count")

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.put(
                    f"{QDRANT_URL}/collections/{COLLECTION}/points",
                    json={"points": [{"id": point_id, "vector": embedding, "payload": payload}]}
                )
                if resp.status_code not in (200, 201):
                    logger.error(f"Qdrant store error: {resp.status_code} {resp.text[:200]}")
                    return None

            return point_id

        except Exception as e:
            logger.error(f"Embed/store error: {e}")
            return None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    async def search_media(self, query: str, media_type: Optional[str] = None,
                           year: Optional[str] = None, category: Optional[str] = None,
                           person: Optional[str] = None,
                           date_from: Optional[str] = None, date_to: Optional[str] = None,
                           limit: int = 20) -> List[Dict[str, Any]]:
        """Semantic search for photos/videos in Qdrant."""
        try:
            # Embed query
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{OLLAMA_URL}/api/embeddings",
                    json={"model": EMBED_MODEL, "prompt": query}
                )
                if resp.status_code != 200:
                    return []
                embedding = resp.json().get("embedding")
                if not embedding:
                    return []

            # Build Qdrant filter
            must_conditions = [
                {"key": "category", "match": {"value": "photos:personal"}}
            ]

            if media_type:
                must_conditions.append({"key": "type", "match": {"value": media_type}})

            if year:
                must_conditions.append({"key": "year", "match": {"value": year}})

            if person:
                must_conditions.append({
                    "key": "people",
                    "match": {"any": [person]}
                })

            if category:
                must_conditions.append({
                    "key": "photo_categories",
                    "match": {"any": [category]}
                })

            qdrant_filter = {"must": must_conditions} if must_conditions else None

            # Date range filter
            if date_from or date_to:
                date_condition = {"key": "date_taken"}
                range_val = {}
                if date_from:
                    range_val["gte"] = date_from
                if date_to:
                    range_val["lte"] = date_to
                date_condition["range"] = range_val
                must_conditions.append(date_condition)

            # Search Qdrant
            search_body = {
                "vector": embedding,
                "limit": limit,
                "with_payload": True,
            }
            if qdrant_filter:
                search_body["filter"] = qdrant_filter

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
                    json=search_body
                )
                if resp.status_code != 200:
                    logger.error(f"Qdrant search error: {resp.status_code}")
                    return []

                data = resp.json()
                results = []
                for hit in data.get("result", []):
                    payload = hit.get("payload", {})
                    results.append({
                        "file_path": payload.get("source", ""),
                        "filename": payload.get("filename", ""),
                        "description": payload.get("content", "")[:500],
                        "categories": payload.get("photo_categories", []),
                        "people": payload.get("people", []),
                        "mood": payload.get("mood", ""),
                        "objects": payload.get("objects", []),
                        "activities": payload.get("activities", []),
                        "year": payload.get("year"),
                        "date_taken": payload.get("date_taken"),
                        "media_type": payload.get("type", "photo"),
                        "duration_seconds": payload.get("duration_seconds"),
                        "score": hit.get("score", 0),
                    })

                return results

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    async def get_stats(self) -> Dict[str, Any]:
        """Overall stats: local count, cloud count, matched, analyzed, embedded."""
        conn = await asyncpg.connect(DB_URL)
        try:
            await self._ensure_schema(conn)

            total_local = await conn.fetchval("SELECT COUNT(*) FROM photos") or 0
            total_cloud = await conn.fetchval("SELECT COUNT(*) FROM google_photos_cloud") or 0
            matched = await conn.fetchval(
                "SELECT COUNT(*) FROM photos WHERE match_type IN ('cloud_matched', 'cloud_filename_match')") or 0
            sha256_dupes = await conn.fetchval(
                "SELECT COUNT(*) FROM photos WHERE match_type = 'sha256_dupe'") or 0
            analyzed = await conn.fetchval(
                "SELECT COUNT(*) FROM photos WHERE llava_description IS NOT NULL") or 0
            embedded = await conn.fetchval(
                "SELECT COUNT(*) FROM photos WHERE qdrant_point_id IS NOT NULL") or 0
            unmatched_local = await conn.fetchval(
                "SELECT COUNT(*) FROM photos WHERE match_type = 'unmatched'") or 0
            unmatched_cloud = await conn.fetchval(
                "SELECT COUNT(*) FROM google_photos_cloud WHERE matched_local_id IS NULL") or 0
            total_size = await conn.fetchval("SELECT SUM(file_size) FROM photos") or 0

            # Media type counts
            photos_count = await conn.fetchval(
                "SELECT COUNT(*) FROM photos WHERE media_type = 'photo' OR media_type IS NULL") or 0
            videos_count = await conn.fetchval(
                "SELECT COUNT(*) FROM photos WHERE media_type = 'video'") or 0

            # Source breakdown
            local_count = await conn.fetchval(
                "SELECT COUNT(*) FROM photos WHERE source_root = 'local' OR source_root IS NULL") or 0
            takeout_count = await conn.fetchval(
                "SELECT COUNT(*) FROM photos WHERE source_root = 'takeout'") or 0

            # Face stats
            faces_detected = await conn.fetchval(
                "SELECT COUNT(*) FROM photos WHERE face_count IS NOT NULL AND face_count > 0") or 0
            face_clusters = await conn.fetchval(
                "SELECT COUNT(*) FROM face_clusters") or 0

            return {
                "local_photos": total_local,
                "cloud_photos": total_cloud,
                "cloud_matched": matched,
                "sha256_dupes": sha256_dupes,
                "analyzed": analyzed,
                "embedded_in_qdrant": embedded,
                "unmatched_local": unmatched_local,
                "unmatched_cloud": unmatched_cloud,
                "total_size_gb": round(total_size / (1024**3), 2),
                "scan_complete": total_local > 0,
                "cloud_fetched": total_cloud > 0,
                "photos_count": photos_count,
                "videos_count": videos_count,
                "local_source": local_count,
                "takeout_source": takeout_count,
                "photos_with_faces": faces_detected,
                "face_clusters": face_clusters,
            }
        finally:
            await conn.close()

    # ------------------------------------------------------------------
    # Google OAuth helpers
    # ------------------------------------------------------------------
    async def get_oauth_status(self) -> Dict[str, Any]:
        """Check if Google OAuth token exists."""
        token = await self._get_google_token()
        if token:
            return {"has_token": True, "status": "ready"}
        return {
            "has_token": False,
            "status": "needs_auth",
            "login_url": f"{AUTH_SERVICE_URL}/api/auth/oauth/google/login",
        }

    async def _get_google_token(self) -> Optional[str]:
        """Get Google OAuth token via tower-auth bridge.
        Returns None if no refresh_token exists (access_token alone is likely expired).
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{AUTH_SERVICE_URL}/tokens/list")
                if resp.status_code == 200:
                    tokens = resp.json()
                    google = tokens.get("google", {})
                    access = google.get("access_token")
                    refresh = google.get("refresh_token")
                    if access and refresh:
                        return access
                    if access and not refresh:
                        logger.warning(
                            "Google access_token exists but no refresh_token — "
                            "token is likely expired and cannot be refreshed. "
                            "Re-authenticate at /api/auth/oauth/google/login"
                        )
                        return None
        except Exception as e:
            logger.warning(f"Failed to get Google token: {e}")
        return None

    async def _refresh_google_token(self) -> Optional[str]:
        """Refresh Google token via tower-auth."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Get current refresh token
                resp = await client.get(f"{AUTH_SERVICE_URL}/tokens/list")
                if resp.status_code != 200:
                    return None
                tokens = resp.json()
                refresh = tokens.get("google", {}).get("refresh_token")
                if not refresh:
                    return None

                resp = await client.post(
                    f"{AUTH_SERVICE_URL}/api/auth/oauth/google/refresh",
                    json={"refresh_token": refresh}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("access_token")
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
        return None

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _compute_perceptual_hash(path: Path) -> Optional[str]:
        try:
            from PIL import Image
            import imagehash
            img = Image.open(path)
            return str(imagehash.phash(img))
        except Exception:
            return None

    @staticmethod
    def _get_dimensions(path: Path) -> tuple:
        try:
            from PIL import Image
            with Image.open(path) as img:
                return img.size
        except Exception:
            return (None, None)

    @staticmethod
    def _extract_exif(path: Path) -> Dict[str, Any]:
        result = {"date_taken": None, "camera_model": None, "location": None}
        try:
            import exifread
            with open(path, "rb") as f:
                tags = exifread.process_file(f, details=False)

            if "EXIF DateTimeOriginal" in tags:
                date_str = str(tags["EXIF DateTimeOriginal"])
                try:
                    result["date_taken"] = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    pass

            if "Image Model" in tags:
                result["camera_model"] = str(tags["Image Model"]).strip()

            # GPS coordinates
            lat = tags.get("GPS GPSLatitude")
            lon = tags.get("GPS GPSLongitude")
            if lat and lon:
                result["location"] = f"GPS: {lat}, {lon}"

        except Exception:
            pass
        return result
