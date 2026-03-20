"""
PersonID Service — Temporal Face Recognition
Graph-based identity clustering that replaces greedy single-pass clustering.

Architecture:
  persons (stable identity)
    └── person_cluster_map (many clusters → one person)
          └── face_clusters (atomic embedding groups)
                └── photo_faces (individual detections)

Key innovation: temporal similarity thresholds + Louvain community detection
finds transitive identity chains across decades (2005→2007→2010→2015→2025).
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import numpy as np

logger = logging.getLogger(__name__)

DB_URL = "postgresql://echo:echo_secure_password_123@localhost/echo_brain"


def _temporal_threshold(date_a: Optional[datetime], date_b: Optional[datetime]) -> float:
    """Return cosine similarity threshold based on temporal gap between two photos."""
    if date_a is None or date_b is None:
        return 0.60  # No date info — moderate threshold

    gap_days = abs((date_a - date_b).days)

    if gap_days < 1:
        return 0.50   # Same event
    elif gap_days < 365:
        return 0.55   # Same year
    elif gap_days < 730:
        return 0.58   # Adjacent years (2yr)
    elif gap_days < 3650:
        return 0.62   # 3-10 year gap
    else:
        return 0.68   # 10+ year gap


class PersonIDService:
    """Graph-based face identity service with temporal awareness."""

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    async def ensure_schema(self, conn: asyncpg.Connection):
        """Create persons/person_cluster_map/merge_history tables and new columns."""
        await conn.execute("SET search_path TO public")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id              SERIAL PRIMARY KEY,
                name            TEXT,
                contact_id      INT,
                is_confirmed    BOOLEAN DEFAULT FALSE,
                birth_year      INT,
                notes           TEXT,
                created_at      TIMESTAMPTZ DEFAULT NOW(),
                updated_at      TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS person_cluster_map (
                id              SERIAL PRIMARY KEY,
                person_id       INT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
                cluster_id      INT NOT NULL REFERENCES face_clusters(id) ON DELETE CASCADE,
                confidence      FLOAT,
                source          TEXT DEFAULT 'auto',
                created_at      TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(cluster_id)
            );

            CREATE TABLE IF NOT EXISTS merge_history (
                id              SERIAL PRIMARY KEY,
                action          TEXT NOT NULL,
                person_id       INT REFERENCES persons(id),
                cluster_ids     INT[],
                similarity      FLOAT,
                reason          TEXT,
                created_at      TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_pcm_person ON person_cluster_map(person_id);
            CREATE INDEX IF NOT EXISTS idx_pcm_cluster ON person_cluster_map(cluster_id);
            CREATE INDEX IF NOT EXISTS idx_merge_history_person ON merge_history(person_id);
        """)

        # New columns on face_clusters
        for col_def in [
            ("date_range_start", "TIMESTAMPTZ"),
            ("date_range_end", "TIMESTAMPTZ"),
            ("is_locked", "BOOLEAN DEFAULT FALSE"),
        ]:
            try:
                await conn.execute(
                    f"ALTER TABLE face_clusters ADD COLUMN IF NOT EXISTS {col_def[0]} {col_def[1]}"
                )
            except Exception:
                pass

        # Shortcut column on photo_faces
        try:
            await conn.execute(
                "ALTER TABLE photo_faces ADD COLUMN IF NOT EXISTS person_id INT REFERENCES persons(id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_photo_faces_person ON photo_faces(person_id)"
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Migration — create Person records for existing named clusters
    # ------------------------------------------------------------------
    async def migrate_named_clusters(self, conn: asyncpg.Connection) -> Dict[str, Any]:
        """One-time migration: create Person records for existing named clusters."""
        named = await conn.fetch("""
            SELECT id, cluster_name FROM face_clusters
            WHERE cluster_name IS NOT NULL
              AND cluster_name != '__skipped__'
              AND id NOT IN (SELECT cluster_id FROM person_cluster_map)
        """)

        created = 0
        for cluster in named:
            # Check if a person with this name already exists
            existing = await conn.fetchrow(
                "SELECT id FROM persons WHERE name = $1", cluster["cluster_name"]
            )
            if existing:
                person_id = existing["id"]
            else:
                person_id = await conn.fetchval("""
                    INSERT INTO persons (name, is_confirmed, created_at, updated_at)
                    VALUES ($1, TRUE, NOW(), NOW())
                    RETURNING id
                """, cluster["cluster_name"])
                created += 1

            # Map cluster to person
            await conn.execute("""
                INSERT INTO person_cluster_map (person_id, cluster_id, confidence, source)
                VALUES ($1, $2, 1.0, 'migration')
                ON CONFLICT (cluster_id) DO NOTHING
            """, person_id, cluster["id"])

            # Lock the cluster
            await conn.execute(
                "UPDATE face_clusters SET is_locked = TRUE WHERE id = $1",
                cluster["id"]
            )

            # Set person_id on all faces in this cluster
            await conn.execute(
                "UPDATE photo_faces SET person_id = $1 WHERE cluster_id = $2",
                person_id, cluster["id"]
            )

        # Log migration
        if created > 0:
            await conn.execute("""
                INSERT INTO merge_history (action, reason, created_at)
                VALUES ('migration', $1, NOW())
            """, f"Created {created} person records from named clusters")

        return {"persons_created": created, "clusters_mapped": len(named)}

    # ------------------------------------------------------------------
    # Full graph clustering
    # ------------------------------------------------------------------
    async def run_full_graph_cluster(self, conn: asyncpg.Connection) -> Dict[str, Any]:
        """Build similarity graph on all faces, run Louvain community detection,
        create/update person records. Respects locked clusters."""
        import networkx as nx
        from sklearn.neighbors import NearestNeighbors

        logger.info("[PersonID] Starting full graph clustering...")

        # Load all face embeddings with photo dates
        faces = await conn.fetch("""
            SELECT pf.id, pf.embedding, pf.cluster_id, pf.person_id,
                   p.date_taken
            FROM photo_faces pf
            JOIN photos p ON p.id = pf.photo_id
            WHERE pf.embedding IS NOT NULL
            ORDER BY pf.id
        """)

        if not faces:
            return {"status": "no_faces", "persons": 0}

        logger.info(f"[PersonID] Loading {len(faces)} face embeddings...")

        face_ids = []
        embeddings = []
        dates = []
        existing_person = {}  # face_id → person_id (for locked clusters)

        for f in faces:
            fid = f["id"]
            face_ids.append(fid)
            emb = np.frombuffer(f["embedding"], dtype=np.float32).copy()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm
            embeddings.append(emb)
            dates.append(f["date_taken"])
            if f["person_id"]:
                existing_person[fid] = f["person_id"]

        embeddings_arr = np.array(embeddings, dtype=np.float32)
        n_faces = len(face_ids)
        logger.info(f"[PersonID] Building sparse similarity graph for {n_faces} faces...")

        # Build sparse graph using radius query (cosine distance < 0.50 = similarity > 0.50)
        nn = NearestNeighbors(metric='cosine', radius=0.50, algorithm='brute', n_jobs=-1)
        nn.fit(embeddings_arr)
        distances, indices = nn.radius_neighbors(embeddings_arr)

        # Build NetworkX graph with temporal thresholds
        G = nx.Graph()
        G.add_nodes_from(range(n_faces))

        edges_added = 0
        for i in range(n_faces):
            for j_idx, dist in zip(indices[i], distances[i]):
                if j_idx <= i:
                    continue  # Skip self and duplicates

                sim = 1.0 - dist
                threshold = _temporal_threshold(dates[i], dates[j_idx])

                if sim >= threshold:
                    G.add_edge(i, j_idx, weight=sim)
                    edges_added += 1

        logger.info(f"[PersonID] Graph: {n_faces} nodes, {edges_added} edges")

        # Louvain community detection
        communities = nx.community.louvain_communities(G, resolution=1.2, seed=42)
        logger.info(f"[PersonID] Found {len(communities)} communities")

        # Get locked person mappings (cluster_id → person_id)
        locked = await conn.fetch("""
            SELECT pcm.cluster_id, pcm.person_id
            FROM person_cluster_map pcm
            JOIN face_clusters fc ON fc.id = pcm.cluster_id
            WHERE fc.is_locked = TRUE
        """)
        locked_cluster_person = {r["cluster_id"]: r["person_id"] for r in locked}

        # Get face → cluster mapping
        face_cluster = {}
        for f in faces:
            if f["cluster_id"]:
                face_cluster[f["id"]] = f["cluster_id"]

        persons_created = 0
        persons_updated = 0
        faces_assigned = 0

        for community in communities:
            member_face_ids = [face_ids[idx] for idx in community]

            # Check if any face in this community belongs to a locked person
            locked_person_ids = set()
            for fid in member_face_ids:
                cid = face_cluster.get(fid)
                if cid and cid in locked_cluster_person:
                    locked_person_ids.add(locked_cluster_person[cid])
                elif fid in existing_person:
                    locked_person_ids.add(existing_person[fid])

            if len(locked_person_ids) == 1:
                # Community maps to one locked person
                person_id = locked_person_ids.pop()
                persons_updated += 1
            elif len(locked_person_ids) > 1:
                # Conflict — multiple locked persons in same community
                # Keep the largest locked person, log for review
                logger.warning(
                    f"[PersonID] Community has {len(locked_person_ids)} locked persons: "
                    f"{locked_person_ids}. Skipping auto-merge."
                )
                continue
            else:
                # New unnamed person
                person_id = await conn.fetchval("""
                    INSERT INTO persons (created_at, updated_at)
                    VALUES (NOW(), NOW())
                    RETURNING id
                """)
                persons_created += 1

            # Compute proper centroid for this community
            member_embeddings = embeddings_arr[list(community)]
            centroid = member_embeddings.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid /= norm

            # Get date range
            member_dates = [dates[idx] for idx in community if dates[idx] is not None]
            date_start = min(member_dates) if member_dates else None
            date_end = max(member_dates) if member_dates else None

            # Get photo IDs for sample
            sample_face_ids = member_face_ids[:10]
            sample_photos = await conn.fetch("""
                SELECT DISTINCT photo_id FROM photo_faces
                WHERE id = ANY($1::int[])
                LIMIT 5
            """, sample_face_ids)
            sample_photo_ids = [r["photo_id"] for r in sample_photos]

            # Photo count for this community
            photo_count_row = await conn.fetchval("""
                SELECT COUNT(DISTINCT photo_id) FROM photo_faces
                WHERE id = ANY($1::int[])
            """, member_face_ids)

            # Create or update cluster for this community
            # Check if we can reuse an existing cluster
            existing_clusters = set()
            for fid in member_face_ids:
                cid = face_cluster.get(fid)
                if cid:
                    existing_clusters.add(cid)

            if existing_clusters:
                # Use the largest existing cluster
                cluster_counts = await conn.fetch("""
                    SELECT id, photo_count FROM face_clusters
                    WHERE id = ANY($1::int[])
                    ORDER BY photo_count DESC
                """, list(existing_clusters))

                if cluster_counts:
                    primary_cluster_id = cluster_counts[0]["id"]
                    merge_cluster_ids = [r["id"] for r in cluster_counts[1:]]
                else:
                    primary_cluster_id = None
                    merge_cluster_ids = []
            else:
                primary_cluster_id = None
                merge_cluster_ids = []

            if primary_cluster_id is None:
                # Create new cluster
                primary_cluster_id = await conn.fetchval("""
                    INSERT INTO face_clusters (
                        centroid_embedding, sample_photo_ids, photo_count,
                        date_range_start, date_range_end
                    ) VALUES ($1, $2, $3, $4, $5)
                    RETURNING id
                """, centroid.astype(np.float32).tobytes(),
                    sample_photo_ids, photo_count_row or 0,
                    date_start, date_end)
            else:
                # Update existing cluster
                await conn.execute("""
                    UPDATE face_clusters SET
                        centroid_embedding = $2,
                        sample_photo_ids = $3,
                        photo_count = $4,
                        date_range_start = $5,
                        date_range_end = $6
                    WHERE id = $1
                """, primary_cluster_id,
                    centroid.astype(np.float32).tobytes(),
                    sample_photo_ids, photo_count_row or 0,
                    date_start, date_end)

            # Reassign faces from merge clusters to primary
            if merge_cluster_ids:
                await conn.execute("""
                    UPDATE photo_faces SET cluster_id = $1
                    WHERE cluster_id = ANY($2::int[])
                """, primary_cluster_id, merge_cluster_ids)

                # Remove old cluster mappings
                await conn.execute("""
                    DELETE FROM person_cluster_map
                    WHERE cluster_id = ANY($1::int[])
                """, merge_cluster_ids)

                # Delete merged clusters (only unlocked ones)
                await conn.execute("""
                    DELETE FROM face_clusters
                    WHERE id = ANY($1::int[]) AND (is_locked IS NULL OR is_locked = FALSE)
                """, merge_cluster_ids)

            # Assign all community faces to primary cluster
            await conn.execute("""
                UPDATE photo_faces SET cluster_id = $1, person_id = $2
                WHERE id = ANY($3::int[])
            """, primary_cluster_id, person_id, member_face_ids)

            # Map cluster → person
            await conn.execute("""
                INSERT INTO person_cluster_map (person_id, cluster_id, confidence, source)
                VALUES ($1, $2, 1.0, 'graph_cluster')
                ON CONFLICT (cluster_id)
                DO UPDATE SET person_id = $1, confidence = 1.0, source = 'graph_cluster'
            """, person_id, primary_cluster_id)

            faces_assigned += len(member_face_ids)

        # Log
        await conn.execute("""
            INSERT INTO merge_history (action, reason, created_at)
            VALUES ('full_graph_cluster', $1, NOW())
        """, f"Created {persons_created}, updated {persons_updated} persons. "
             f"{faces_assigned}/{n_faces} faces assigned from {len(communities)} communities.")

        result = {
            "communities": len(communities),
            "persons_created": persons_created,
            "persons_updated": persons_updated,
            "faces_assigned": faces_assigned,
            "total_faces": n_faces,
            "edges": edges_added,
        }
        logger.info(f"[PersonID] Full graph clustering complete: {result}")
        return result

    # ------------------------------------------------------------------
    # Incremental assignment — for new faces (30-min worker cycle)
    # ------------------------------------------------------------------
    async def assign_new_faces(self, conn: asyncpg.Connection) -> Dict[str, Any]:
        """Match unassigned faces against existing cluster centroids using temporal thresholds."""
        # Get faces without a person assignment
        unassigned = await conn.fetch("""
            SELECT pf.id, pf.embedding, p.date_taken
            FROM photo_faces pf
            JOIN photos p ON p.id = pf.photo_id
            WHERE pf.person_id IS NULL
              AND pf.embedding IS NOT NULL
            ORDER BY pf.id
            LIMIT 500
        """)

        if not unassigned:
            return {"assigned": 0, "new_singletons": 0}

        # Load cluster centroids
        clusters = await conn.fetch("""
            SELECT fc.id AS cluster_id, fc.centroid_embedding,
                   fc.date_range_start, fc.date_range_end,
                   pcm.person_id
            FROM face_clusters fc
            LEFT JOIN person_cluster_map pcm ON pcm.cluster_id = fc.id
            WHERE fc.centroid_embedding IS NOT NULL
        """)

        if not clusters:
            # No clusters yet — create singletons
            return await self._create_singletons(conn, unassigned)

        # Parse cluster centroids
        cluster_ids = []
        cluster_embeddings = []
        cluster_dates = []  # (start, end)
        cluster_person_ids = []

        for c in clusters:
            cluster_ids.append(c["cluster_id"])
            emb = np.frombuffer(c["centroid_embedding"], dtype=np.float32).copy()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm
            cluster_embeddings.append(emb)
            # Use midpoint of date range for threshold calculation
            start = c["date_range_start"]
            end = c["date_range_end"]
            mid = None
            if start and end:
                mid = start + (end - start) / 2
            elif start:
                mid = start
            elif end:
                mid = end
            cluster_dates.append(mid)
            cluster_person_ids.append(c["person_id"])

        cluster_embeddings_arr = np.array(cluster_embeddings, dtype=np.float32)

        assigned = 0
        new_singletons = 0

        for face in unassigned:
            emb = np.frombuffer(face["embedding"], dtype=np.float32).copy()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm

            face_date = face["date_taken"]

            # Compute similarity to all centroids
            sims = cluster_embeddings_arr @ emb

            # Apply temporal thresholds
            best_idx = None
            best_sim = 0.0
            for i, sim in enumerate(sims):
                threshold = _temporal_threshold(face_date, cluster_dates[i])
                if sim >= threshold and sim > best_sim:
                    best_sim = sim
                    best_idx = i

            if best_idx is not None:
                cid = cluster_ids[best_idx]
                pid = cluster_person_ids[best_idx]

                await conn.execute("""
                    UPDATE photo_faces SET cluster_id = $1, person_id = $2
                    WHERE id = $3
                """, cid, pid, face["id"])

                # Update cluster photo count
                await conn.execute("""
                    UPDATE face_clusters SET
                        photo_count = (
                            SELECT COUNT(DISTINCT photo_id)
                            FROM photo_faces WHERE cluster_id = $1
                        )
                    WHERE id = $1
                """, cid)

                assigned += 1
            else:
                # No match — create singleton cluster + unnamed person
                centroid_bytes = emb.astype(np.float32).tobytes()

                photo_id = await conn.fetchval(
                    "SELECT photo_id FROM photo_faces WHERE id = $1", face["id"]
                )

                new_cluster_id = await conn.fetchval("""
                    INSERT INTO face_clusters (
                        centroid_embedding, sample_photo_ids, photo_count,
                        date_range_start, date_range_end
                    ) VALUES ($1, $2, 1, $3, $3)
                    RETURNING id
                """, centroid_bytes, [photo_id] if photo_id else [],
                    face_date)

                new_person_id = await conn.fetchval("""
                    INSERT INTO persons (created_at, updated_at)
                    VALUES (NOW(), NOW())
                    RETURNING id
                """)

                await conn.execute("""
                    INSERT INTO person_cluster_map (person_id, cluster_id, confidence, source)
                    VALUES ($1, $2, 1.0, 'singleton')
                """, new_person_id, new_cluster_id)

                await conn.execute("""
                    UPDATE photo_faces SET cluster_id = $1, person_id = $2
                    WHERE id = $3
                """, new_cluster_id, new_person_id, face["id"])

                new_singletons += 1

        result = {"assigned": assigned, "new_singletons": new_singletons,
                  "total_processed": len(unassigned)}
        if assigned > 0 or new_singletons > 0:
            logger.info(f"[PersonID] Incremental assignment: {result}")
        return result

    async def _create_singletons(self, conn, faces) -> Dict[str, Any]:
        """Create singleton cluster + person for each unassigned face."""
        created = 0
        for face in faces:
            emb = np.frombuffer(face["embedding"], dtype=np.float32).copy()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm

            photo_id = await conn.fetchval(
                "SELECT photo_id FROM photo_faces WHERE id = $1", face["id"]
            )

            new_cluster_id = await conn.fetchval("""
                INSERT INTO face_clusters (
                    centroid_embedding, sample_photo_ids, photo_count,
                    date_range_start, date_range_end
                ) VALUES ($1, $2, 1, $3, $3)
                RETURNING id
            """, emb.astype(np.float32).tobytes(),
                [photo_id] if photo_id else [],
                face["date_taken"])

            new_person_id = await conn.fetchval("""
                INSERT INTO persons (created_at, updated_at)
                VALUES (NOW(), NOW())
                RETURNING id
            """)

            await conn.execute("""
                INSERT INTO person_cluster_map (person_id, cluster_id, confidence, source)
                VALUES ($1, $2, 1.0, 'singleton')
            """, new_person_id, new_cluster_id)

            await conn.execute("""
                UPDATE photo_faces SET cluster_id = $1, person_id = $2
                WHERE id = $3
            """, new_cluster_id, new_person_id, face["id"])

            created += 1

        return {"assigned": 0, "new_singletons": created}

    # ------------------------------------------------------------------
    # Periodic cluster review — merge suggestions (6-hour cycle)
    # ------------------------------------------------------------------
    async def review_clusters(self, conn: asyncpg.Connection) -> Dict[str, Any]:
        """Build cluster-level similarity graph, propose merges for unnamed persons.
        Auto-merge unnamed; queue named for review."""
        import networkx as nx

        clusters = await conn.fetch("""
            SELECT fc.id AS cluster_id, fc.centroid_embedding,
                   fc.date_range_start, fc.date_range_end,
                   fc.is_locked, fc.cluster_name,
                   pcm.person_id
            FROM face_clusters fc
            LEFT JOIN person_cluster_map pcm ON pcm.cluster_id = fc.id
            WHERE fc.centroid_embedding IS NOT NULL
        """)

        if len(clusters) < 2:
            return {"merges": 0, "suggestions": 0}

        # Parse
        cids = []
        embs = []
        c_dates = []
        c_locked = []
        c_person = []
        c_named = []

        for c in clusters:
            cids.append(c["cluster_id"])
            emb = np.frombuffer(c["centroid_embedding"], dtype=np.float32).copy()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm
            embs.append(emb)
            start = c["date_range_start"]
            end = c["date_range_end"]
            mid = None
            if start and end:
                mid = start + (end - start) / 2
            elif start:
                mid = start
            elif end:
                mid = end
            c_dates.append(mid)
            c_locked.append(bool(c["is_locked"]))
            c_person.append(c["person_id"])
            c_named.append(c["cluster_name"] is not None and c["cluster_name"] != '__skipped__')

        embs_arr = np.array(embs, dtype=np.float32)
        n = len(cids)

        # Build cluster-level graph
        G = nx.Graph()
        G.add_nodes_from(range(n))

        # Pairwise similarities (cluster-level, so manageable)
        sim_matrix = embs_arr @ embs_arr.T

        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i, j])
                threshold = _temporal_threshold(c_dates[i], c_dates[j])
                # Use slightly higher threshold for cluster merging
                threshold = max(threshold, 0.55)
                if sim >= threshold:
                    G.add_edge(i, j, weight=sim)

        # Find communities
        if G.number_of_edges() == 0:
            return {"merges": 0, "suggestions": 0}

        communities = nx.community.louvain_communities(G, resolution=1.2, seed=42)

        auto_merges = 0
        suggestions_created = 0

        for community in communities:
            if len(community) < 2:
                continue

            member_cids = [cids[idx] for idx in community]
            member_persons = set(c_person[idx] for idx in community if c_person[idx] is not None)
            has_named = any(c_named[idx] for idx in community)
            has_locked = any(c_locked[idx] for idx in community)

            # Compute average similarity within community
            sims = []
            for i_idx in community:
                for j_idx in community:
                    if j_idx > i_idx:
                        sims.append(float(sim_matrix[i_idx, j_idx]))
            avg_sim = sum(sims) / len(sims) if sims else 0.0

            if len(member_persons) <= 1 and not has_locked and not has_named:
                # Auto-merge unnamed clusters
                target_person = member_persons.pop() if member_persons else None
                if target_person is None:
                    target_person = await conn.fetchval("""
                        INSERT INTO persons (created_at, updated_at)
                        VALUES (NOW(), NOW()) RETURNING id
                    """)

                for cid in member_cids:
                    await conn.execute("""
                        INSERT INTO person_cluster_map (person_id, cluster_id, confidence, source)
                        VALUES ($1, $2, $3, 'auto_merge')
                        ON CONFLICT (cluster_id)
                        DO UPDATE SET person_id = $1, confidence = $3, source = 'auto_merge'
                    """, target_person, cid, avg_sim)

                    await conn.execute(
                        "UPDATE photo_faces SET person_id = $1 WHERE cluster_id = $2",
                        target_person, cid
                    )

                await conn.execute("""
                    INSERT INTO merge_history (action, person_id, cluster_ids, similarity, reason)
                    VALUES ('auto_merge', $1, $2, $3, $4)
                """, target_person, member_cids, avg_sim,
                    f"Auto-merged {len(member_cids)} unnamed clusters")

                auto_merges += 1
            else:
                # Queue as suggestion for review
                await conn.execute("""
                    INSERT INTO merge_history (action, cluster_ids, similarity, reason)
                    VALUES ('suggestion', $1, $2, $3)
                """, member_cids, avg_sim,
                    f"Suggested merge of {len(member_cids)} clusters "
                    f"(named={has_named}, locked={has_locked})")

                suggestions_created += 1

        result = {"merges": auto_merges, "suggestions": suggestions_created,
                  "communities_found": len(communities)}
        logger.info(f"[PersonID] Cluster review: {result}")
        return result

    # ------------------------------------------------------------------
    # Naming propagation
    # ------------------------------------------------------------------
    async def name_person(self, conn: asyncpg.Connection, person_id: int, name: str) -> Dict[str, Any]:
        """Name a person and propagate to all mapped clusters and faces."""
        person = await conn.fetchrow("SELECT id, name FROM persons WHERE id = $1", person_id)
        if not person:
            return {"error": f"Person {person_id} not found"}

        old_name = person["name"]

        # Update person
        await conn.execute("""
            UPDATE persons SET name = $2, is_confirmed = TRUE, updated_at = NOW()
            WHERE id = $1
        """, person_id, name)

        # Update all mapped clusters
        cluster_ids = [r["cluster_id"] for r in await conn.fetch(
            "SELECT cluster_id FROM person_cluster_map WHERE person_id = $1", person_id
        )]

        for cid in cluster_ids:
            await conn.execute(
                "UPDATE face_clusters SET cluster_name = $2 WHERE id = $1",
                cid, name
            )

        # Update all faces
        updated_faces = await conn.execute(
            "UPDATE photo_faces SET person_id = $1 WHERE cluster_id = ANY($2::int[])",
            person_id, cluster_ids
        )

        # Audit
        await conn.execute("""
            INSERT INTO merge_history (action, person_id, cluster_ids, reason)
            VALUES ('name', $1, $2, $3)
        """, person_id, cluster_ids,
            f"Named '{name}' (was '{old_name}')")

        return {
            "person_id": person_id,
            "name": name,
            "clusters_updated": len(cluster_ids),
            "status": "named",
        }

    # ------------------------------------------------------------------
    # Merge two persons
    # ------------------------------------------------------------------
    async def merge_persons(self, conn: asyncpg.Connection,
                            source_id: int, target_id: int) -> Dict[str, Any]:
        """Merge source person into target. Moves all clusters and faces."""
        source = await conn.fetchrow("SELECT id, name FROM persons WHERE id = $1", source_id)
        target = await conn.fetchrow("SELECT id, name FROM persons WHERE id = $1", target_id)
        if not source or not target:
            return {"error": "Person not found"}

        # Move cluster mappings
        await conn.execute("""
            UPDATE person_cluster_map SET person_id = $1
            WHERE person_id = $2
        """, target_id, source_id)

        # Update faces
        await conn.execute("""
            UPDATE photo_faces SET person_id = $1
            WHERE person_id = $2
        """, target_id, source_id)

        # Get all cluster IDs now under target
        cluster_ids = [r["cluster_id"] for r in await conn.fetch(
            "SELECT cluster_id FROM person_cluster_map WHERE person_id = $1", target_id
        )]

        # Update cluster names if target is named
        if target["name"]:
            for cid in cluster_ids:
                await conn.execute(
                    "UPDATE face_clusters SET cluster_name = $2 WHERE id = $1",
                    cid, target["name"]
                )

        # Audit
        await conn.execute("""
            INSERT INTO merge_history (action, person_id, cluster_ids, reason)
            VALUES ('merge_persons', $1, $2, $3)
        """, target_id, cluster_ids,
            f"Merged person {source_id} ('{source['name']}') into {target_id} ('{target['name']}')")

        # Delete source person
        await conn.execute("DELETE FROM persons WHERE id = $1", source_id)

        return {
            "target_id": target_id,
            "source_id": source_id,
            "clusters_moved": len(cluster_ids),
            "status": "merged",
        }

    # ------------------------------------------------------------------
    # Split clusters from a person
    # ------------------------------------------------------------------
    async def split_clusters(self, conn: asyncpg.Connection,
                             person_id: int, cluster_ids: List[int]) -> Dict[str, Any]:
        """Detach clusters from person into a new unnamed person."""
        # Verify clusters belong to this person
        owned = await conn.fetch("""
            SELECT cluster_id FROM person_cluster_map
            WHERE person_id = $1 AND cluster_id = ANY($2::int[])
        """, person_id, cluster_ids)

        if not owned:
            return {"error": "No matching clusters found for this person"}

        owned_ids = [r["cluster_id"] for r in owned]

        # Create new person
        new_person_id = await conn.fetchval("""
            INSERT INTO persons (created_at, updated_at)
            VALUES (NOW(), NOW()) RETURNING id
        """)

        # Move cluster mappings
        await conn.execute("""
            UPDATE person_cluster_map SET person_id = $1
            WHERE person_id = $2 AND cluster_id = ANY($3::int[])
        """, new_person_id, person_id, owned_ids)

        # Update faces
        await conn.execute("""
            UPDATE photo_faces SET person_id = $1
            WHERE cluster_id = ANY($2::int[])
        """, new_person_id, owned_ids)

        # Clear cluster names on split clusters
        await conn.execute("""
            UPDATE face_clusters SET cluster_name = NULL
            WHERE id = ANY($1::int[])
        """, owned_ids)

        # Audit
        await conn.execute("""
            INSERT INTO merge_history (action, person_id, cluster_ids, reason)
            VALUES ('split', $1, $2, $3)
        """, person_id, owned_ids,
            f"Split {len(owned_ids)} clusters from person {person_id} → new person {new_person_id}")

        return {
            "original_person_id": person_id,
            "new_person_id": new_person_id,
            "clusters_split": owned_ids,
            "status": "split",
        }

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------
    async def list_persons(self, conn: asyncpg.Connection,
                           limit: int = 50, offset: int = 0,
                           named_only: bool = False) -> Dict[str, Any]:
        """List persons with face count and date range."""
        name_filter = "WHERE p.name IS NOT NULL" if named_only else ""

        total = await conn.fetchval(f"""
            SELECT COUNT(*) FROM persons p {name_filter}
        """)

        rows = await conn.fetch(f"""
            SELECT p.id, p.name, p.is_confirmed, p.birth_year, p.notes,
                   p.created_at, p.updated_at,
                   COALESCE(agg.face_count, 0) AS face_count,
                   COALESCE(agg.cluster_count, 0) AS cluster_count,
                   agg.date_start, agg.date_end
            FROM persons p
            LEFT JOIN LATERAL (
                SELECT
                    COUNT(pf.id) AS face_count,
                    COUNT(DISTINCT pcm.cluster_id) AS cluster_count,
                    MIN(fc.date_range_start) AS date_start,
                    MAX(fc.date_range_end) AS date_end
                FROM person_cluster_map pcm
                JOIN face_clusters fc ON fc.id = pcm.cluster_id
                LEFT JOIN photo_faces pf ON pf.cluster_id = fc.id
                WHERE pcm.person_id = p.id
            ) agg ON TRUE
            {name_filter}
            ORDER BY agg.face_count DESC NULLS LAST
            LIMIT $1 OFFSET $2
        """, limit, offset)

        persons = []
        for r in rows:
            persons.append({
                "id": r["id"],
                "name": r["name"],
                "is_confirmed": r["is_confirmed"],
                "birth_year": r["birth_year"],
                "notes": r["notes"],
                "face_count": r["face_count"],
                "cluster_count": r["cluster_count"],
                "date_range": {
                    "start": r["date_start"].isoformat() if r["date_start"] else None,
                    "end": r["date_end"].isoformat() if r["date_end"] else None,
                },
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            })

        return {"persons": persons, "total": total}

    async def get_person_detail(self, conn: asyncpg.Connection,
                                person_id: int) -> Optional[Dict[str, Any]]:
        """Get person detail with clusters and sample photos."""
        person = await conn.fetchrow("""
            SELECT id, name, is_confirmed, birth_year, notes, contact_id,
                   created_at, updated_at
            FROM persons WHERE id = $1
        """, person_id)

        if not person:
            return None

        clusters = await conn.fetch("""
            SELECT fc.id, fc.cluster_name, fc.photo_count,
                   fc.sample_photo_ids, fc.is_locked,
                   fc.date_range_start, fc.date_range_end,
                   pcm.confidence, pcm.source
            FROM person_cluster_map pcm
            JOIN face_clusters fc ON fc.id = pcm.cluster_id
            WHERE pcm.person_id = $1
            ORDER BY fc.photo_count DESC
        """, person_id)

        face_count = await conn.fetchval("""
            SELECT COUNT(*) FROM photo_faces WHERE person_id = $1
        """, person_id)

        photo_count = await conn.fetchval("""
            SELECT COUNT(DISTINCT photo_id) FROM photo_faces WHERE person_id = $1
        """, person_id)

        return {
            "id": person["id"],
            "name": person["name"],
            "is_confirmed": person["is_confirmed"],
            "birth_year": person["birth_year"],
            "notes": person["notes"],
            "contact_id": person["contact_id"],
            "face_count": face_count or 0,
            "photo_count": photo_count or 0,
            "clusters": [
                {
                    "id": c["id"],
                    "name": c["cluster_name"],
                    "photo_count": c["photo_count"],
                    "sample_photo_ids": list(c["sample_photo_ids"] or []),
                    "is_locked": c["is_locked"],
                    "confidence": c["confidence"],
                    "source": c["source"],
                    "date_range": {
                        "start": c["date_range_start"].isoformat() if c["date_range_start"] else None,
                        "end": c["date_range_end"].isoformat() if c["date_range_end"] else None,
                    },
                }
                for c in clusters
            ],
            "created_at": person["created_at"].isoformat() if person["created_at"] else None,
            "updated_at": person["updated_at"].isoformat() if person["updated_at"] else None,
        }

    async def get_person_timeline(self, conn: asyncpg.Connection,
                                  person_id: int) -> Dict[str, Any]:
        """Get photos by year for a person."""
        rows = await conn.fetch("""
            SELECT EXTRACT(YEAR FROM p.date_taken)::int AS year,
                   COUNT(*) AS count,
                   ARRAY_AGG(p.id ORDER BY p.date_taken LIMIT 5) AS sample_ids
            FROM photo_faces pf
            JOIN photos p ON p.id = pf.photo_id
            WHERE pf.person_id = $1
              AND p.date_taken IS NOT NULL
            GROUP BY EXTRACT(YEAR FROM p.date_taken)
            ORDER BY year
        """, person_id)

        # Fallback for photos without dates
        no_date_count = await conn.fetchval("""
            SELECT COUNT(*) FROM photo_faces pf
            JOIN photos p ON p.id = pf.photo_id
            WHERE pf.person_id = $1 AND p.date_taken IS NULL
        """, person_id)

        timeline = [
            {"year": r["year"], "count": r["count"],
             "sample_photo_ids": list(r["sample_ids"] or [])}
            for r in rows
        ]

        return {
            "person_id": person_id,
            "timeline": timeline,
            "undated_count": no_date_count or 0,
        }

    async def get_merge_suggestions(self, conn: asyncpg.Connection,
                                    limit: int = 20) -> List[Dict[str, Any]]:
        """Get pending merge suggestions from merge_history."""
        rows = await conn.fetch("""
            SELECT id, cluster_ids, similarity, reason, created_at
            FROM merge_history
            WHERE action = 'suggestion'
            ORDER BY similarity DESC, created_at DESC
            LIMIT $1
        """, limit)

        suggestions = []
        for r in rows:
            # Get cluster details
            cluster_details = await conn.fetch("""
                SELECT fc.id, fc.cluster_name, fc.photo_count, fc.sample_photo_ids,
                       pcm.person_id, p.name AS person_name
                FROM face_clusters fc
                LEFT JOIN person_cluster_map pcm ON pcm.cluster_id = fc.id
                LEFT JOIN persons p ON p.id = pcm.person_id
                WHERE fc.id = ANY($1::int[])
            """, r["cluster_ids"])

            suggestions.append({
                "id": r["id"],
                "cluster_ids": list(r["cluster_ids"]),
                "similarity": r["similarity"],
                "reason": r["reason"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "clusters": [
                    {
                        "id": c["id"],
                        "name": c["cluster_name"],
                        "photo_count": c["photo_count"],
                        "sample_photo_ids": list(c["sample_photo_ids"] or []),
                        "person_id": c["person_id"],
                        "person_name": c["person_name"],
                    }
                    for c in cluster_details
                ],
            })

        return suggestions

    async def accept_suggestion(self, conn: asyncpg.Connection,
                                suggestion_id: int) -> Dict[str, Any]:
        """Accept a merge suggestion — merge all clusters' persons into one."""
        row = await conn.fetchrow("""
            SELECT id, cluster_ids, similarity FROM merge_history
            WHERE id = $1 AND action = 'suggestion'
        """, suggestion_id)

        if not row:
            return {"error": f"Suggestion {suggestion_id} not found"}

        cluster_ids = list(row["cluster_ids"])

        # Find all persons for these clusters
        person_rows = await conn.fetch("""
            SELECT DISTINCT pcm.person_id, p.name
            FROM person_cluster_map pcm
            JOIN persons p ON p.id = pcm.person_id
            WHERE pcm.cluster_id = ANY($1::int[])
        """, cluster_ids)

        if not person_rows:
            return {"error": "No persons found for these clusters"}

        # Pick the named person (or largest) as target
        target = None
        for pr in person_rows:
            if pr["name"]:
                target = pr["person_id"]
                break
        if target is None:
            target = person_rows[0]["person_id"]

        # Merge all other persons into target
        for pr in person_rows:
            if pr["person_id"] != target:
                await self.merge_persons(conn, pr["person_id"], target)

        # Mark suggestion as accepted
        await conn.execute("""
            UPDATE merge_history SET action = 'suggestion_accepted'
            WHERE id = $1
        """, suggestion_id)

        return {"status": "accepted", "target_person_id": target,
                "merged_persons": len(person_rows) - 1}

    async def reject_suggestion(self, conn: asyncpg.Connection,
                                suggestion_id: int) -> Dict[str, Any]:
        """Reject a merge suggestion."""
        result = await conn.execute("""
            UPDATE merge_history SET action = 'suggestion_rejected'
            WHERE id = $1 AND action = 'suggestion'
        """, suggestion_id)

        if "UPDATE 0" in result:
            return {"error": f"Suggestion {suggestion_id} not found"}

        return {"status": "rejected", "suggestion_id": suggestion_id}

    # ------------------------------------------------------------------
    # Phase 5: Singleton Pruning
    # ------------------------------------------------------------------
    async def prune_singletons(self, conn: asyncpg.Connection,
                               min_faces: int = 3) -> Dict[str, Any]:
        """Remove unnamed, unconfirmed persons with fewer than min_faces total faces.
        Preserves face embeddings but clears person/cluster assignments.
        Never touches named, confirmed, or locked persons."""

        # Find singleton persons
        singletons = await conn.fetch("""
            SELECT p.id, COUNT(pf.id) AS face_count
            FROM persons p
            JOIN person_cluster_map pcm ON pcm.person_id = p.id
            JOIN face_clusters fc ON fc.id = pcm.cluster_id
            LEFT JOIN photo_faces pf ON pf.cluster_id = fc.id
            WHERE p.name IS NULL
              AND p.is_confirmed = FALSE
              AND (fc.is_locked IS NULL OR fc.is_locked = FALSE)
            GROUP BY p.id
            HAVING COUNT(pf.id) < $1
        """, min_faces)

        if not singletons:
            return {"pruned_persons": 0, "faces_freed": 0}

        person_ids = [s["id"] for s in singletons]
        total_faces_freed = sum(s["face_count"] for s in singletons)

        # Get cluster IDs for these persons
        cluster_rows = await conn.fetch("""
            SELECT pcm.cluster_id FROM person_cluster_map pcm
            WHERE pcm.person_id = ANY($1::int[])
        """, person_ids)
        cluster_ids = [r["cluster_id"] for r in cluster_rows]

        # Clear person_id and cluster_id on faces (preserve embeddings)
        if cluster_ids:
            await conn.execute("""
                UPDATE photo_faces SET person_id = NULL, cluster_id = NULL
                WHERE cluster_id = ANY($1::int[])
            """, cluster_ids)

        # Delete mappings, clusters, persons
        await conn.execute(
            "DELETE FROM person_cluster_map WHERE person_id = ANY($1::int[])",
            person_ids)
        if cluster_ids:
            await conn.execute(
                "DELETE FROM face_clusters WHERE id = ANY($1::int[]) AND (is_locked IS NULL OR is_locked = FALSE)",
                cluster_ids)
        await conn.execute(
            "DELETE FROM persons WHERE id = ANY($1::int[])", person_ids)

        # Log
        await conn.execute("""
            INSERT INTO merge_history (action, reason, created_at)
            VALUES ('prune_singletons', $1, NOW())
        """, f"Pruned {len(person_ids)} singleton persons (< {min_faces} faces), "
             f"freed {total_faces_freed} face records for reclustering")

        result = {"pruned_persons": len(person_ids), "faces_freed": total_faces_freed,
                  "clusters_deleted": len(cluster_ids)}
        logger.info(f"[PersonID] Singleton pruning: {result}")
        return result

    # ------------------------------------------------------------------
    # Phase 6: Prepare for Recluster
    # ------------------------------------------------------------------
    async def prepare_for_recluster(self, conn: asyncpg.Connection) -> Dict[str, Any]:
        """Clear non-locked cluster/person assignments, then recluster.
        Locked clusters (named/confirmed) are preserved."""

        # Count faces to be cleared
        cleared_faces = await conn.fetchval("""
            SELECT COUNT(*) FROM photo_faces
            WHERE cluster_id IN (
                SELECT id FROM face_clusters
                WHERE is_locked IS NULL OR is_locked = FALSE
            )
        """) or 0

        # Clear non-locked face assignments
        await conn.execute("""
            UPDATE photo_faces SET person_id = NULL, cluster_id = NULL
            WHERE cluster_id IN (
                SELECT id FROM face_clusters
                WHERE is_locked IS NULL OR is_locked = FALSE
            )
        """)

        # Delete non-locked cluster mappings
        await conn.execute("""
            DELETE FROM person_cluster_map
            WHERE cluster_id IN (
                SELECT id FROM face_clusters
                WHERE is_locked IS NULL OR is_locked = FALSE
            )
        """)

        # Count and delete non-locked clusters
        deleted_clusters = await conn.fetchval("""
            SELECT COUNT(*) FROM face_clusters
            WHERE is_locked IS NULL OR is_locked = FALSE
        """) or 0
        await conn.execute("""
            DELETE FROM face_clusters
            WHERE is_locked IS NULL OR is_locked = FALSE
        """)

        # Count and delete unnamed persons with no remaining clusters
        deleted_persons = await conn.fetchval("""
            SELECT COUNT(*) FROM persons
            WHERE name IS NULL
              AND NOT EXISTS (
                SELECT 1 FROM person_cluster_map pcm WHERE pcm.person_id = persons.id
              )
        """) or 0
        await conn.execute("""
            DELETE FROM persons
            WHERE name IS NULL
              AND NOT EXISTS (
                SELECT 1 FROM person_cluster_map pcm WHERE pcm.person_id = persons.id
              )
        """)

        # Log
        await conn.execute("""
            INSERT INTO merge_history (action, reason, created_at)
            VALUES ('prepare_recluster', $1, NOW())
        """, f"Cleared {cleared_faces} face assignments, deleted {deleted_clusters} clusters, "
             f"{deleted_persons} unnamed persons. Ready for full recluster.")

        prep_result = {
            "faces_cleared": cleared_faces,
            "clusters_deleted": deleted_clusters,
            "persons_deleted": deleted_persons,
        }
        logger.info(f"[PersonID] Recluster prep: {prep_result}")

        # Now run full recluster
        cluster_result = await self.run_full_graph_cluster(conn)

        return {"preparation": prep_result, "recluster": cluster_result}
