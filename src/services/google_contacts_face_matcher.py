"""
Google Contacts → Face Cluster Auto-Matcher

Pulls Google Contacts (with profile photos), computes face embeddings
via InsightFace, and matches against existing face cluster centroids.
High-confidence matches are auto-named; uncertain ones are flagged for review.
"""
import asyncio
import logging
import os
from io import BytesIO
from typing import Dict, Any, List, Optional, Tuple

import asyncpg
import httpx
import numpy as np

logger = logging.getLogger(__name__)

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://echo:echo_secure_password_123@localhost/echo_brain",
)

# Thresholds for cosine similarity (1 = identical, 0 = unrelated)
AUTO_NAME_THRESHOLD = 0.55   # Above this: auto-name without review
REVIEW_THRESHOLD = 0.40      # Between review and auto: flag for review
# Below REVIEW_THRESHOLD: no match


class GoogleContactsFaceMatcher:
    """Fetches Google Contacts, matches profile photos to face clusters."""

    def __init__(self):
        self._insightface_app = None
        self.tower_auth_url = "http://localhost:8088"

    def _get_insightface_app(self):
        """Lazily initialize InsightFace."""
        if self._insightface_app is None:
            try:
                import insightface
                self._insightface_app = insightface.app.FaceAnalysis(
                    name="buffalo_l",
                    providers=["CPUExecutionProvider"],
                )
                self._insightface_app.prepare(ctx_id=-1, det_size=(640, 640))
                logger.info("InsightFace loaded for contact matching")
            except Exception as e:
                logger.error(f"InsightFace init failed: {e}")
                return None
        return self._insightface_app

    async def _get_google_token(self) -> Optional[str]:
        """Get a valid Google access token.
        Tries tower-auth bridge first, falls back to direct DB read."""
        # Try bridge first
        try:
            from src.integrations.tower_auth_bridge import tower_auth
            token = await tower_auth.get_valid_token("google")
            if token:
                return token
        except Exception as e:
            logger.debug(f"Bridge token fetch failed: {e}")

        # Fallback: read directly from auth_sessions in tower_consolidated
        try:
            tc_url = os.getenv(
                "TOWER_CONSOLIDATED_URL",
                f"postgresql://patrick:{os.getenv('DB_PASSWORD', 'RP78eIrW7cI2jYvL5akt1yurE')}@localhost/tower_consolidated",
            )
            conn = await asyncpg.connect(tc_url)
            try:
                row = await conn.fetchrow("""
                    SELECT access_token FROM auth_sessions
                    WHERE provider = 'google' AND is_active = true
                    ORDER BY last_accessed DESC LIMIT 1
                """)
                if row and row["access_token"]:
                    logger.info("Got Google token from auth_sessions (direct DB)")
                    return row["access_token"]
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Direct DB token fetch failed: {e}")

        return None

    async def fetch_contacts(self, max_contacts: int = 2000) -> List[Dict[str, Any]]:
        """Fetch Google Contacts with profile photos via People API."""
        token = await self._get_google_token()
        if not token:
            return []

        contacts = []
        next_page = None
        headers = {"Authorization": f"Bearer {token}"}

        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                params = {
                    "personFields": "names,emailAddresses,phoneNumbers,photos,birthdays,organizations",
                    "pageSize": 100,
                    "sortOrder": "LAST_MODIFIED_DESCENDING",
                }
                if next_page:
                    params["pageToken"] = next_page

                resp = await client.get(
                    "https://people.googleapis.com/v1/people/me/connections",
                    headers=headers,
                    params=params,
                )

                if resp.status_code == 403:
                    logger.error(
                        "Google Contacts API 403 — contacts.readonly scope missing. "
                        "Re-authenticate at /api/auth/oauth/google/login"
                    )
                    return []

                if resp.status_code != 200:
                    logger.error(f"Contacts API error {resp.status_code}: {resp.text[:200]}")
                    break

                data = resp.json()
                for person in data.get("connections", []):
                    names = person.get("names", [])
                    photos = person.get("photos", [])
                    emails = person.get("emailAddresses", [])
                    phones = person.get("phoneNumbers", [])
                    birthdays = person.get("birthdays", [])
                    orgs = person.get("organizations", [])

                    if not names:
                        continue

                    name = names[0].get("displayName", "")
                    if not name:
                        continue

                    # Get the best photo URL (skip default silhouette)
                    photo_url = None
                    for p in photos:
                        if not p.get("metadata", {}).get("source", {}).get("type") == "DOMAIN_PROFILE":
                            if not p.get("default", False):
                                photo_url = p.get("url")
                                break

                    contacts.append({
                        "name": name,
                        "photo_url": photo_url,
                        "emails": [e.get("value") for e in emails if e.get("value")],
                        "phones": [p.get("value") for p in phones if p.get("value")],
                        "birthday": birthdays[0].get("date") if birthdays else None,
                        "organization": orgs[0].get("name") if orgs else None,
                        "resource_name": person.get("resourceName"),
                    })

                next_page = data.get("nextPageToken")
                if not next_page or len(contacts) >= max_contacts:
                    break

                await asyncio.sleep(0.2)

        logger.info(f"Fetched {len(contacts)} Google contacts ({sum(1 for c in contacts if c['photo_url'])} with photos)")
        return contacts

    async def _download_photo_embedding(
        self, photo_url: str, client: httpx.AsyncClient
    ) -> Optional[np.ndarray]:
        """Download a contact photo and compute its face embedding."""
        try:
            resp = await client.get(photo_url, timeout=10)
            if resp.status_code != 200:
                return None

            from PIL import Image
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            img_array = np.array(img)

            app = self._get_insightface_app()
            if app is None:
                return None

            faces = app.get(img_array)
            if not faces:
                return None

            # Use the largest face (most prominent)
            largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            emb = largest.embedding.astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb
        except Exception as e:
            logger.debug(f"Photo embedding failed for {photo_url[:60]}: {e}")
            return None

    async def match_contacts_to_clusters(self) -> Dict[str, Any]:
        """
        Main pipeline:
        1. Fetch contacts
        2. Store contacts in DB
        3. Compute embeddings for contact photos
        4. Match against cluster centroids
        5. Auto-name high-confidence matches
        6. Return review list for uncertain matches
        """
        # Step 1: Fetch contacts
        contacts = await self.fetch_contacts()
        if not contacts:
            return {"error": "No contacts fetched — check Google token and contacts.readonly scope"}

        conn = await asyncpg.connect(DB_URL)
        try:
            # Step 2: Ensure contacts table exists and store
            await conn.execute("SET search_path TO public")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS google_contacts (
                    id SERIAL PRIMARY KEY,
                    resource_name TEXT UNIQUE,
                    name TEXT NOT NULL,
                    emails TEXT[],
                    phones TEXT[],
                    birthday JSONB,
                    organization TEXT,
                    photo_url TEXT,
                    face_embedding BYTEA,
                    matched_cluster_id INT REFERENCES face_clusters(id),
                    match_confidence FLOAT,
                    match_status TEXT DEFAULT 'pending',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_contacts_resource
                ON google_contacts(resource_name)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_contacts_cluster
                ON google_contacts(matched_cluster_id)
            """)

            # Upsert contacts
            import json as _json
            inserted = 0
            for c in contacts:
                bday = c.get("birthday")
                if bday and not isinstance(bday, str):
                    bday = _json.dumps(bday)
                result = await conn.execute("""
                    INSERT INTO google_contacts (resource_name, name, emails, phones,
                        birthday, organization, photo_url, updated_at)
                    VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, NOW())
                    ON CONFLICT (resource_name) DO UPDATE SET
                        name = EXCLUDED.name,
                        emails = EXCLUDED.emails,
                        phones = EXCLUDED.phones,
                        birthday = EXCLUDED.birthday,
                        organization = EXCLUDED.organization,
                        photo_url = EXCLUDED.photo_url,
                        updated_at = NOW()
                """,
                    c["resource_name"], c["name"],
                    c["emails"] or [], c["phones"] or [],
                    bday, c.get("organization"),
                    c["photo_url"],
                )
                if "INSERT" in result:
                    inserted += 1

            logger.info(f"Stored {inserted} new / {len(contacts)} total contacts")

            # Step 3: Get contacts with photos that don't have embeddings yet
            contacts_to_embed = await conn.fetch("""
                SELECT id, name, photo_url FROM google_contacts
                WHERE photo_url IS NOT NULL
                  AND face_embedding IS NULL
            """)

            logger.info(f"Computing face embeddings for {len(contacts_to_embed)} contact photos")

            embedded_count = 0
            async with httpx.AsyncClient() as client:
                for contact in contacts_to_embed:
                    emb = await self._download_photo_embedding(contact["photo_url"], client)
                    if emb is not None:
                        await conn.execute("""
                            UPDATE google_contacts SET face_embedding = $1
                            WHERE id = $2
                        """, emb.tobytes(), contact["id"])
                        embedded_count += 1
                    await asyncio.sleep(0.1)

            logger.info(f"Computed {embedded_count} contact face embeddings")

            # Step 4: Load cluster centroids
            clusters = await conn.fetch("""
                SELECT id, cluster_name, centroid_embedding, photo_count
                FROM face_clusters
                WHERE centroid_embedding IS NOT NULL
                  AND photo_count >= 3
                ORDER BY photo_count DESC
            """)

            cluster_centroids = []
            for cl in clusters:
                centroid = np.frombuffer(cl["centroid_embedding"], dtype=np.float32)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                cluster_centroids.append({
                    "id": cl["id"],
                    "name": cl["cluster_name"],
                    "centroid": centroid,
                    "photo_count": cl["photo_count"],
                })

            if not cluster_centroids:
                return {
                    "contacts_fetched": len(contacts),
                    "embedded": embedded_count,
                    "error": "No face clusters with centroids found",
                }

            # Step 5: Match each contact embedding against centroids
            contacts_with_embs = await conn.fetch("""
                SELECT id, name, face_embedding, photo_url
                FROM google_contacts
                WHERE face_embedding IS NOT NULL
                  AND match_status = 'pending'
            """)

            auto_named = 0
            needs_review = 0
            no_match = 0

            for contact in contacts_with_embs:
                contact_emb = np.frombuffer(contact["face_embedding"], dtype=np.float32)
                norm = np.linalg.norm(contact_emb)
                if norm > 0:
                    contact_emb = contact_emb / norm

                # Find best matching cluster
                best_sim = -1.0
                best_cluster_id = None
                best_cluster_count = 0

                for cl in cluster_centroids:
                    sim = float(np.dot(contact_emb, cl["centroid"]))
                    if sim > best_sim:
                        best_sim = sim
                        best_cluster_id = cl["id"]
                        best_cluster_count = cl["photo_count"]

                if best_sim >= AUTO_NAME_THRESHOLD:
                    # High confidence — auto-name the cluster
                    await conn.execute("""
                        UPDATE face_clusters SET cluster_name = $2
                        WHERE id = $1 AND cluster_name IS NULL
                    """, best_cluster_id, contact["name"])
                    await conn.execute("""
                        UPDATE google_contacts
                        SET matched_cluster_id = $1, match_confidence = $2,
                            match_status = 'auto_named'
                        WHERE id = $3
                    """, best_cluster_id, best_sim, contact["id"])
                    auto_named += 1
                    logger.info(
                        f"Auto-named cluster {best_cluster_id} "
                        f"({best_cluster_count} photos) as '{contact['name']}' "
                        f"(sim={best_sim:.3f})"
                    )

                elif best_sim >= REVIEW_THRESHOLD:
                    # Uncertain — flag for review
                    await conn.execute("""
                        UPDATE google_contacts
                        SET matched_cluster_id = $1, match_confidence = $2,
                            match_status = 'needs_review'
                        WHERE id = $3
                    """, best_cluster_id, best_sim, contact["id"])
                    needs_review += 1

                else:
                    await conn.execute("""
                        UPDATE google_contacts
                        SET match_confidence = $1, match_status = 'no_match'
                        WHERE id = $2
                    """, best_sim, contact["id"])
                    no_match += 1

            result = {
                "contacts_fetched": len(contacts),
                "contacts_with_photos": sum(1 for c in contacts if c["photo_url"]),
                "embeddings_computed": embedded_count,
                "clusters_checked": len(cluster_centroids),
                "auto_named": auto_named,
                "needs_review": needs_review,
                "no_match": no_match,
            }
            logger.info(f"Contact-face matching complete: {result}")
            return result

        finally:
            await conn.close()

    async def get_review_queue(self) -> List[Dict[str, Any]]:
        """Get contacts that need manual review (uncertain matches)."""
        conn = await asyncpg.connect(DB_URL)
        try:
            rows = await conn.fetch("""
                SELECT gc.id as contact_id, gc.name as contact_name,
                       gc.photo_url, gc.match_confidence,
                       gc.emails, gc.organization,
                       fc.id as cluster_id, fc.photo_count,
                       fc.sample_photo_ids, fc.cluster_name
                FROM google_contacts gc
                JOIN face_clusters fc ON fc.id = gc.matched_cluster_id
                WHERE gc.match_status = 'needs_review'
                ORDER BY gc.match_confidence DESC
            """)
            return [dict(r) for r in rows]
        finally:
            await conn.close()

    async def confirm_match(self, contact_id: int, cluster_id: int, name: str) -> Dict:
        """Confirm a review match — name the cluster."""
        conn = await asyncpg.connect(DB_URL)
        try:
            await conn.execute(
                "UPDATE face_clusters SET cluster_name = $2 WHERE id = $1",
                cluster_id, name,
            )
            await conn.execute("""
                UPDATE google_contacts
                SET match_status = 'confirmed', matched_cluster_id = $1
                WHERE id = $2
            """, cluster_id, contact_id)
            return {"status": "confirmed", "contact_id": contact_id, "cluster_id": cluster_id, "name": name}
        finally:
            await conn.close()

    async def reject_match(self, contact_id: int) -> Dict:
        """Reject a suggested match."""
        conn = await asyncpg.connect(DB_URL)
        try:
            await conn.execute("""
                UPDATE google_contacts
                SET match_status = 'rejected', matched_cluster_id = NULL
                WHERE id = $1
            """, contact_id)
            return {"status": "rejected", "contact_id": contact_id}
        finally:
            await conn.close()

    async def get_stats(self) -> Dict[str, Any]:
        """Get current matching stats."""
        conn = await asyncpg.connect(DB_URL)
        try:
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name='google_contacts')"
            )
            if not exists:
                return {"contacts": 0, "message": "No contacts synced yet"}

            stats = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total_contacts,
                    COUNT(*) FILTER (WHERE photo_url IS NOT NULL) as with_photos,
                    COUNT(*) FILTER (WHERE face_embedding IS NOT NULL) as with_embeddings,
                    COUNT(*) FILTER (WHERE match_status = 'auto_named') as auto_named,
                    COUNT(*) FILTER (WHERE match_status = 'needs_review') as needs_review,
                    COUNT(*) FILTER (WHERE match_status = 'confirmed') as confirmed,
                    COUNT(*) FILTER (WHERE match_status = 'rejected') as rejected,
                    COUNT(*) FILTER (WHERE match_status = 'no_match') as no_match
                FROM google_contacts
            """)
            named_clusters = await conn.fetchval(
                "SELECT COUNT(*) FROM face_clusters WHERE cluster_name IS NOT NULL AND cluster_name != '__skipped__'"
            )
            return {**dict(stats), "named_clusters": named_clusters}
        finally:
            await conn.close()
