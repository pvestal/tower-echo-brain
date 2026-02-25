"""
Conversation Compactor for Echo Brain

Summarizes old conversation turns when a session exceeds a character threshold,
persists the summary to session_summaries, and replaces old turns with a compact
system message so the LLM stays within its context window.
"""
import asyncpg
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger("echo.core.compactor")

# Threshold in total characters across all turns before compaction kicks in
COMPACTION_THRESHOLD = 16_000  # ~4,000 tokens at 4 chars/token

# How many recent turns to always keep verbatim
KEEP_RECENT = 6

SUMMARIZE_SYSTEM = (
    "You are a conversation summarizer. Given a transcript of conversation turns, "
    "produce a concise summary that preserves: (1) the main topics discussed, "
    "(2) any decisions or conclusions reached, (3) any files or code mentioned, "
    "(4) the user's current intent or open questions. "
    "Be factual and compact. Output plain text, no markdown headers."
)


class ConversationCompactor:
    """Compacts long conversation sessions into summaries."""

    def __init__(self):
        self._dsn = None

    def _get_dsn(self) -> str:
        if not self._dsn:
            password = os.environ.get('PGPASSWORD', os.environ.get('DB_PASSWORD', ''))
            self._dsn = f"postgresql://patrick:{password}@localhost/echo_brain"
        return self._dsn

    async def maybe_compact(self, session_id: str, turns: List[Dict]) -> List[Dict]:
        """Entry point: compact if total content exceeds threshold.

        Args:
            session_id: The conversation session identifier
            turns: List of {role, content, timestamp} dicts, oldest-first

        Returns:
            Original turns if under threshold, or [summary_turn] + recent turns
        """
        if not turns or len(turns) <= KEEP_RECENT:
            return turns

        total_chars = sum(len(t.get("content", "")) for t in turns)
        if total_chars <= COMPACTION_THRESHOLD:
            return turns

        logger.info(
            f"Compacting session {session_id}: {len(turns)} turns, "
            f"{total_chars} chars (threshold={COMPACTION_THRESHOLD})"
        )

        try:
            return await self._compact(session_id, turns)
        except Exception as e:
            logger.warning(f"Compaction failed for {session_id}, using safe fallback: {e}")
            # Safe fallback: just return recent turns
            return turns[-KEEP_RECENT:]

    async def _compact(self, session_id: str, turns: List[Dict]) -> List[Dict]:
        """Perform the actual compaction."""
        to_summarize = turns[:-KEEP_RECENT]
        to_keep = turns[-KEEP_RECENT:]

        # Build transcript from old turns
        transcript_lines = []
        for turn in to_summarize:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            transcript_lines.append(f"{role}: {content}")
        transcript = "\n".join(transcript_lines)

        # Truncate transcript if absurdly long (keep under ~8K chars for summarizer)
        if len(transcript) > 8000:
            transcript = transcript[:8000] + "\n\n[...transcript truncated...]"

        # Generate summary via LLM
        from src.services.llm_service import get_llm_service
        llm = get_llm_service()

        prompt = f"Summarize this conversation transcript:\n\n{transcript}"
        result = await llm.generate(
            prompt=prompt,
            model="mistral:7b",
            system=SUMMARIZE_SYSTEM,
            temperature=0.1,
            max_tokens=1024
        )
        summary_text = result.content.strip()

        if not summary_text:
            logger.warning("LLM returned empty summary, skipping compaction")
            return turns[-KEEP_RECENT:]

        # Extract structured fields from summary
        topics = self._extract_topics(summary_text)
        decisions = self._extract_decisions(summary_text)
        files = self._extract_files(summary_text)

        # Persist to database
        compacted_at = datetime.now()
        turns_summarized = len(to_summarize)

        try:
            conn = await asyncpg.connect(self._get_dsn(), timeout=5)
            try:
                async with conn.transaction():
                    # Delete old turns from conversation_messages
                    oldest_kept_ts = to_keep[0].get("timestamp")
                    if oldest_kept_ts:
                        # Parse ISO string to datetime if needed
                        if isinstance(oldest_kept_ts, str):
                            oldest_kept_ts = datetime.fromisoformat(oldest_kept_ts)
                        await conn.execute("""
                            DELETE FROM conversation_messages
                            WHERE conversation_id = $1
                              AND timestamp < $2
                        """, session_id, oldest_kept_ts)

                    # Insert summary as a system message
                    metadata = json.dumps({
                        "type": "compaction_summary",
                        "compacted_at": compacted_at.isoformat(),
                        "turns_summarized": turns_summarized
                    })
                    await conn.execute("""
                        INSERT INTO conversation_messages
                            (conversation_id, role, content, metadata)
                        VALUES ($1, 'system', $2, $3::jsonb)
                    """, session_id, summary_text, metadata)

                    # Insert into session_summaries
                    await conn.execute("""
                        INSERT INTO session_summaries
                            (session_id, summary, topics, key_decisions, files_modified)
                        VALUES ($1, $2, $3, $4, $5)
                    """, session_id, summary_text, topics, decisions, files)

            finally:
                await conn.close()

            logger.info(
                f"Compacted {turns_summarized} turns for session {session_id} "
                f"(topics={topics}, decisions={len(decisions)})"
            )
        except Exception as e:
            logger.warning(f"DB persistence failed during compaction: {e}")
            # Still return the compacted turns even if DB fails

        # Build summary turn
        summary_turn = {
            "role": "system",
            "content": f"[Previous conversation summary]\n{summary_text}",
            "timestamp": compacted_at.isoformat()
        }

        return [summary_turn] + to_keep

    def _extract_topics(self, summary: str) -> List[str]:
        """Extract topic keywords from summary text."""
        # Simple heuristic: look for noun phrases after "topics:" or just grab key words
        topics = []
        lower = summary.lower()

        # Check for explicit topic mentions
        topic_keywords = [
            "echo brain", "tower", "anime", "comfyui", "framepack", "ollama",
            "database", "postgresql", "qdrant", "api", "frontend", "nginx",
            "authentication", "generation", "training", "video", "music",
            "compaction", "memory", "agent", "model", "deployment"
        ]
        for kw in topic_keywords:
            if kw in lower:
                topics.append(kw)

        return topics[:10]  # Cap at 10

    def _extract_decisions(self, summary: str) -> List[str]:
        """Extract decision statements from summary."""
        decisions = []
        for line in summary.split("\n"):
            line = line.strip()
            lower = line.lower()
            if any(marker in lower for marker in [
                "decided", "decision", "chose", "agreed", "will use",
                "switched to", "replaced", "conclusion"
            ]):
                decisions.append(line)
        return decisions[:5]

    def _extract_files(self, summary: str) -> List[str]:
        """Extract file paths mentioned in summary."""
        import re
        files = re.findall(r'(?:/[\w.-]+)+(?:\.\w+)?', summary)
        # Deduplicate and filter out obvious non-paths
        seen = set()
        result = []
        for f in files:
            if f not in seen and len(f) > 3 and '.' in f.split('/')[-1]:
                seen.add(f)
                result.append(f)
        return result[:10]


# Singleton
_compactor: Optional[ConversationCompactor] = None


def get_compactor() -> ConversationCompactor:
    """Get the singleton ConversationCompactor instance."""
    global _compactor
    if _compactor is None:
        _compactor = ConversationCompactor()
    return _compactor
