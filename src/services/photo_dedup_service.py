"""
Photo Dedup Service — Echo Brain
Deduplicates ~/Pictures/ against Google Photos cloud library,
then ingests photo metadata into Echo Brain for semantic search.

PERSONAL PHOTOS ONLY — no anime, ComfyUI, LoRA, or training content.
"""
import base64
import hashlib
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import asyncpg
import httpx

logger = logging.getLogger(__name__)

# --- Configuration ---
LOCAL_PHOTOS_ROOT = Path("/home/patrick/Pictures")
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".gif", ".bmp", ".tiff", ".tif"}
EXCLUDED_PATH_SEGMENTS = {"ComfyUI", "comfyui", "lora", "training", "anime", "datasets", "workflow", "output", "temp"}

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLAVA_MODEL = "llava:7b"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "echo_memory")
DB_URL = os.getenv("DATABASE_URL", "postgresql://echo:echo_secure_password_123@localhost/echo_brain")
AUTH_SERVICE_URL = "http://localhost:8088"

GOOGLE_PHOTOS_API = "https://photoslibrary.googleapis.com/v1"

# LLaVA prompt for personal photo analysis
_LLAVA_PHOTO_PROMPT = """Analyze this personal photo briefly:
1. SCENE: What is the main subject or scene? (one sentence)
2. PEOPLE: Are there people? How many? General description (no names).
3. LOCATION: Where was this likely taken? (indoor/outdoor, type of place)
4. MOOD: What mood or emotion does this photo convey? (one word)
5. CATEGORIES: List 2-4 categories from: family, friends, travel, nature, food, pets, celebration, selfie, architecture, sports, work, art, vehicles, documents, screenshot

Respond in this exact format:
SCENE: ...
PEOPLE: ...
LOCATION: ...
MOOD: ...
CATEGORIES: ..."""


class PhotoDedupService:
    """Manages local photo scanning, Google Photos cloud dedup, and Echo Brain ingestion."""

    def __init__(self):
        self.stats = {
            "local_scanned": 0, "local_new": 0, "local_skipped": 0,
            "cloud_fetched": 0, "matched": 0, "analyzed": 0,
            "embedded": 0, "errors": 0
        }

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    async def _ensure_schema(self, conn: asyncpg.Connection):
        """Create tables if they don't exist."""
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
        """)

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
                        camera_model, location_text)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (file_path) DO NOTHING
                """,
                    data["file_path"], data["filename"], data["sha256"],
                    data["perceptual_hash"], data["file_size"],
                    data["width"], data["height"], data["date_taken"],
                    data["year_folder"], data["camera_model"], data["location_text"]
                )
                inserted += 1
            except Exception as e:
                logger.error(f"Insert error for {data['file_path']}: {e}")
        return inserted

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
                __import__("json").dumps(results))

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

            return {
                "total_local": total_local,
                "total_cloud": total_cloud,
                "match_breakdown": {r["match_type"]: r["cnt"] for r in match_breakdown},
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
    # LLaVA analysis
    # ------------------------------------------------------------------
    async def analyze_photos_batch(self, batch_size: int = 50) -> Dict[str, Any]:
        """Run LLaVA vision analysis on un-analyzed photos, newest first."""
        conn = await asyncpg.connect(DB_URL)
        try:
            await self._ensure_schema(conn)

            photos = await conn.fetch("""
                SELECT id, file_path, filename, year_folder
                FROM photos
                WHERE llava_description IS NULL
                  AND match_type != 'sha256_dupe'
                ORDER BY date_taken DESC NULLS LAST, id DESC
                LIMIT $1
            """, batch_size)

            if not photos:
                return {"analyzed": 0, "message": "No photos pending analysis"}

            run_id = await conn.fetchval(
                "INSERT INTO photo_dedup_runs (run_type) VALUES ('llava_analysis') RETURNING id")

            analyzed = 0
            errors = 0

            for photo in photos:
                try:
                    result = await self._analyze_single_photo(photo["file_path"])
                    if result:
                        await conn.execute("""
                            UPDATE photos
                            SET llava_analysis = $2::jsonb,
                                llava_description = $3,
                                updated_at = NOW()
                            WHERE id = $1
                        """, photo["id"],
                            __import__("json").dumps(result),
                            result.get("description", ""))
                        analyzed += 1
                    else:
                        errors += 1
                except Exception as e:
                    logger.error(f"LLaVA error for {photo['file_path']}: {e}")
                    errors += 1

            await conn.execute("""
                UPDATE photo_dedup_runs
                SET finished_at = NOW(), items_processed = $2, items_new = $3, items_error = $4
                WHERE id = $1
            """, run_id, len(photos), analyzed, errors)

            result = {"analyzed": analyzed, "errors": errors, "batch_size": len(photos), "run_id": run_id}
            logger.info(f"[PhotoDedup] LLaVA analysis: {result}")
            return result

        finally:
            await conn.close()

    async def _analyze_single_photo(self, file_path: str) -> Optional[Dict]:
        """Send a photo to LLaVA for analysis."""
        path = Path(file_path)
        if not path.exists():
            return None

        # Read and encode image
        image_data = path.read_bytes()
        if len(image_data) > 20_000_000:  # Skip >20MB
            return None
        image_b64 = base64.b64encode(image_data).decode("utf-8")

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": LLAVA_MODEL,
                    "prompt": _LLAVA_PHOTO_PROMPT,
                    "images": [image_b64],
                    "stream": False,
                    "options": {"temperature": 0.3},
                }
            )
            if resp.status_code != 200:
                logger.error(f"LLaVA API error: {resp.status_code}")
                return None

            raw = resp.json().get("response", "")

        return self._parse_llava_response(raw, path)

    def _parse_llava_response(self, raw: str, path: Path) -> Dict:
        """Parse structured LLaVA response into dict."""
        result = {
            "scene": "", "people": "", "location": "",
            "mood": "", "categories": [], "raw": raw
        }

        for line in raw.split("\n"):
            line_s = line.strip()
            if line_s.upper().startswith("SCENE:"):
                result["scene"] = line_s.split(":", 1)[1].strip()
            elif line_s.upper().startswith("PEOPLE:"):
                result["people"] = line_s.split(":", 1)[1].strip()
            elif line_s.upper().startswith("LOCATION:"):
                result["location"] = line_s.split(":", 1)[1].strip()
            elif line_s.upper().startswith("MOOD:"):
                result["mood"] = line_s.split(":", 1)[1].strip()
            elif line_s.upper().startswith("CATEGORIES:"):
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

        year_match = re.search(r"/(\d{4})/", str(path))
        if year_match:
            parts.append(f"Year: {year_match.group(1)}")

        result["description"] = ". ".join(parts) if parts else path.name

        return result

    # ------------------------------------------------------------------
    # Qdrant ingestion
    # ------------------------------------------------------------------
    async def ingest_to_qdrant(self, batch_size: int = 100) -> Dict[str, Any]:
        """Embed analyzed photo metadata with nomic-embed-text, store in echo_memory."""
        conn = await asyncpg.connect(DB_URL)
        try:
            await self._ensure_schema(conn)

            photos = await conn.fetch("""
                SELECT id, file_path, filename, year_folder, date_taken,
                       camera_model, llava_description, llava_analysis,
                       match_type, width, height
                FROM photos
                WHERE llava_description IS NOT NULL
                  AND qdrant_point_id IS NULL
                  AND match_type != 'sha256_dupe'
                ORDER BY date_taken DESC NULLS LAST
                LIMIT $1
            """, batch_size)

            if not photos:
                return {"embedded": 0, "message": "No photos pending embedding"}

            run_id = await conn.fetchval(
                "INSERT INTO photo_dedup_runs (run_type) VALUES ('qdrant_ingest') RETURNING id")

            embedded = 0
            errors = 0

            for photo in photos:
                try:
                    # Build text for embedding
                    text = self._build_embedding_text(photo)
                    point_id = await self._embed_and_store(text, photo)

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

    def _build_embedding_text(self, photo: dict) -> str:
        """Build rich text for embedding from photo metadata."""
        parts = [f"Personal photo: {photo['filename']}"]

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

        # Add categories from analysis
        analysis = photo.get("llava_analysis")
        if analysis and isinstance(analysis, dict):
            cats = analysis.get("categories", [])
            if cats:
                parts.append(f"Categories: {', '.join(cats)}")

        return "\n".join(parts)

    async def _embed_and_store(self, text: str, photo: dict) -> Optional[str]:
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
            analysis = photo.get("llava_analysis")
            if analysis and isinstance(analysis, dict):
                categories = analysis.get("categories", [])

            payload = {
                "content": text[:10000],
                "type": "photo",
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
            }

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
        """Get Google OAuth token via tower-auth bridge."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{AUTH_SERVICE_URL}/tokens/list")
                if resp.status_code == 200:
                    tokens = resp.json()
                    google = tokens.get("google", {})
                    if google.get("access_token"):
                        return google["access_token"]
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
                    f"{AUTH_SERVICE_URL}/oauth/google/refresh",
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
