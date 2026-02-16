"""
Governor Worker (HMLR Pipeline)
Resolves conflicting facts by comparing effective scores
(confidence + recency), keeping the winner and demoting others.
Logs decisions for audit trail.
"""
import logging
import os
import asyncpg
from datetime import datetime, timezone
import json

logger = logging.getLogger("echo.workers.governor")

AMBIGUITY_THRESHOLD = 0.1  # If effective score spread < this, flag for human review


class Governor:
    """Resolves conflicting facts. Part of the HMLR pipeline."""

    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable required")

    async def _ensure_table(self, conn):
        """Create governor_decisions table if it doesn't exist."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS governor_decisions (
                id SERIAL PRIMARY KEY,
                decision_type VARCHAR(100) NOT NULL,
                subject TEXT,
                predicate TEXT,
                conflicting_facts JSONB,
                winner_fact_id TEXT,
                reasoning TEXT,
                outcome VARCHAR(50),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        # Migrate winner_fact_id from INTEGER to TEXT if needed
        col_type = await conn.fetchval("""
            SELECT data_type FROM information_schema.columns
            WHERE table_name = 'governor_decisions' AND column_name = 'winner_fact_id'
        """)
        if col_type and col_type != "text":
            await conn.execute(
                "ALTER TABLE governor_decisions ALTER COLUMN winner_fact_id TYPE TEXT USING winner_fact_id::TEXT"
            )

    def _recency_score(self, created_at, now) -> float:
        """Compute recency score: 1.0 for brand-new, 0.1 for 90+ days old."""
        if not created_at:
            return 0.5
        try:
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            age_days = (now - created_at).total_seconds() / 86400
            return max(0.1, 1.0 - (age_days / 90.0))
        except Exception:
            return 0.5

    async def run_cycle(self):
        resolved = 0
        flagged = 0

        try:
            conn = await asyncpg.connect(self.db_url)
            try:
                await self._ensure_table(conn)
                now = datetime.now(timezone.utc)

                # Find facts grouped by (subject, predicate) with multiple different objects
                conflicts = await conn.fetch("""
                    SELECT subject, predicate,
                           array_agg(id) as fact_ids,
                           array_agg(object) as objects,
                           array_agg(confidence) as confidences,
                           array_agg(created_at) as created_ats
                    FROM facts
                    WHERE subject IS NOT NULL AND predicate IS NOT NULL
                    GROUP BY subject, predicate
                    HAVING COUNT(DISTINCT object) > 1
                """)

                if not conflicts:
                    logger.info("Governor: no conflicting facts found")
                    return

                logger.info(f"Governor: found {len(conflicts)} conflict groups")

                for group in conflicts:
                    subject = group["subject"]
                    predicate = group["predicate"]
                    fact_ids = list(group["fact_ids"])
                    objects = list(group["objects"])
                    confidences = [float(c) for c in group["confidences"]]
                    created_ats = list(group["created_ats"])

                    # Compute effective scores
                    effective_scores = []
                    fact_details = []
                    for i, fid in enumerate(fact_ids):
                        conf = confidences[i]
                        recency = self._recency_score(created_ats[i], now)
                        eff = conf * 0.6 + recency * 0.4
                        effective_scores.append(eff)
                        fact_details.append({
                            "fact_id": str(fid),
                            "object": objects[i],
                            "confidence": conf,
                            "recency": round(recency, 3),
                            "effective_score": round(eff, 3),
                        })

                    # Sort by effective score descending
                    ranked = sorted(
                        zip(effective_scores, fact_details, fact_ids),
                        key=lambda x: x[0],
                        reverse=True,
                    )

                    best_score = ranked[0][0]
                    second_score = ranked[1][0] if len(ranked) > 1 else 0

                    if best_score - second_score < AMBIGUITY_THRESHOLD:
                        # Ambiguous — flag for human review
                        await conn.execute("""
                            INSERT INTO governor_decisions
                                (decision_type, subject, predicate, conflicting_facts, reasoning, outcome)
                            VALUES ($1, $2, $3, $4, $5, $6)
                        """,
                            "conflict_resolution",
                            subject,
                            predicate,
                            json.dumps([d for _, d, _ in ranked]),
                            f"Ambiguous: spread={best_score - second_score:.3f} < {AMBIGUITY_THRESHOLD}",
                            "flagged_for_review",
                        )
                        flagged += 1
                        continue

                    # Winner is the top-ranked fact
                    winner_id = ranked[0][2]
                    winner_detail = ranked[0][1]

                    # Demote losers
                    for eff, detail, fid in ranked[1:]:
                        old_conf = detail["confidence"]
                        new_conf = max(0.2, round(old_conf * 0.3, 4))
                        await conn.execute(
                            "UPDATE facts SET confidence = $1 WHERE id = $2",
                            new_conf, fid,
                        )

                    # Build reasoning string
                    reasoning_parts = [
                        f"kept fact_id {winner_id} (eff={winner_detail['effective_score']}, "
                        f"conf={winner_detail['confidence']}, recency={winner_detail['recency']})"
                    ]
                    for _, detail, fid in ranked[1:]:
                        reasoning_parts.append(
                            f"demoted fact_id {fid} (eff={detail['effective_score']}, "
                            f"conf={detail['confidence']}, recency={detail['recency']})"
                        )

                    await conn.execute("""
                        INSERT INTO governor_decisions
                            (decision_type, subject, predicate, conflicting_facts,
                             winner_fact_id, reasoning, outcome)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                        "conflict_resolution",
                        subject,
                        predicate,
                        json.dumps([d for _, d, _ in ranked]),
                        str(winner_id),
                        "; ".join(reasoning_parts),
                        "resolved",
                    )
                    resolved += 1

            finally:
                await conn.close()

        except Exception as e:
            logger.error(f"Governor error: {e}")

        logger.info(f"Governor cycle: resolved={resolved}, flagged={flagged}")
