#!/usr/bin/env python3
"""
Learning Middleware for Echo Brain
Automatically saves important conversations and extracts knowledge
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import psycopg2
from psycopg2.extras import Json
import re
import json

logger = logging.getLogger(__name__)


class LearningMiddleware:
    """Middleware to capture and learn from conversations"""

    def __init__(self):
        self.db_config = {
            "host": "localhost",
            "database": "echo_brain",
            "user": "patrick",
            "password": "***REMOVED***"
        }
        self.importance_threshold = 0.3

    async def process_conversation(self, request_data: Dict, response_data: Dict) -> None:
        """Process a conversation exchange for learning"""
        try:
            # Extract key information
            query = request_data.get("query", "")
            response = response_data.get("response", "")
            conversation_id = request_data.get("conversation_id", "unknown")
            model_used = response_data.get("model_used", "unknown")
            confidence = response_data.get("confidence", 0.5)

            # Calculate importance
            importance = self._calculate_importance(query, response, confidence)

            if importance >= self.importance_threshold:
                # Save to episodic memory
                await self._save_episodic_memory(
                    conversation_id, query, response, model_used, importance
                )

                # Extract and save facts
                facts = self._extract_facts(query, response)
                if facts:
                    await self._save_learned_facts(conversation_id, facts)

                # Update learning history
                await self._update_learning_history(query, response, importance)

                logger.info(f"Learned from conversation {conversation_id} (importance: {importance:.2f})")

        except Exception as e:
            logger.error(f"Learning middleware error: {e}")

    def _calculate_importance(self, query: str, response: str, confidence: float) -> float:
        """Calculate the importance of a conversation"""
        importance = confidence

        # Boost importance for certain topics
        important_keywords = [
            "photo", "video", "google", "takeout", "memory", "remember",
            "anime", "comfyui", "personal", "data", "file", "search"
        ]

        query_lower = query.lower()
        for keyword in important_keywords:
            if keyword in query_lower:
                importance += 0.1

        # Boost for questions
        if "?" in query:
            importance += 0.1

        # Boost for long, detailed responses
        if len(response) > 500:
            importance += 0.1

        # Boost if numbers or paths are mentioned
        if re.search(r'\d+', response) or re.search(r'/[^\s]+', response):
            importance += 0.15

        return min(importance, 1.0)  # Cap at 1.0

    def _extract_facts(self, query: str, response: str) -> str:
        """Extract important facts from the exchange"""
        facts = []

        # Extract file paths
        paths = re.findall(r'/[^\s]+', response)
        if paths:
            facts.append(f"Files referenced: {', '.join(paths[:3])}")

        # Extract numbers with context
        numbers = re.findall(r'(\d+)\s+([a-zA-Z]+)', response)
        for number, context in numbers[:3]:
            facts.append(f"{number} {context}")

        # Extract key phrases
        if "found" in response.lower():
            found_match = re.search(r'found\s+([\d,]+)\s+(\w+)', response.lower())
            if found_match:
                facts.append(f"Found {found_match.group(1)} {found_match.group(2)}")

        # Extract dates or years
        years = re.findall(r'20\d{2}', response)
        if years:
            facts.append(f"Years mentioned: {', '.join(set(years))}")

        # Query type classification
        if "?" in query:
            if "how many" in query.lower():
                facts.append("User asked for count/quantity")
            elif "where" in query.lower():
                facts.append("User asked for location")
            elif "when" in query.lower():
                facts.append("User asked for time/date")
            elif "what" in query.lower():
                facts.append("User asked for information")

        return "; ".join(facts) if facts else ""

    async def _save_episodic_memory(self, conversation_id: str, query: str,
                                   response: str, model_used: str, importance: float) -> None:
        """Save to episodic memory table"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            memory_type = self._classify_memory_type(query)
            emotional_valence = self._calculate_sentiment(response)
            learned_fact = self._extract_facts(query, response)

            cur.execute("""
                INSERT INTO echo_episodic_memory
                (conversation_id, memory_type, content, emotional_valence, importance,
                 user_query, echo_response, model_used, learned_fact, created_at, access_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), 1)
            """, (
                conversation_id,
                memory_type,
                f"Q: {query[:200]}... A: {response[:200]}...",
                emotional_valence,
                importance,
                query,
                response,
                model_used,
                learned_fact
            ))

            conn.commit()
            cur.close()
            conn.close()

        except Exception as e:
            logger.error(f"Error saving episodic memory: {e}")

    async def _save_learned_facts(self, conversation_id: str, facts: str) -> None:
        """Save extracted facts to learning history"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO learning_history
                (input_text, learned_output, model_used, quality_score, created_at, metadata)
                VALUES (%s, %s, %s, %s, NOW(), %s)
            """, (
                conversation_id,
                facts,
                "learning_middleware",
                0.8,  # Default quality score
                Json({"source": "conversation", "auto_extracted": True})
            ))

            conn.commit()
            cur.close()
            conn.close()

        except Exception as e:
            logger.error(f"Error saving learned facts: {e}")

    async def _update_learning_history(self, query: str, response: str, importance: float) -> None:
        """Update learning history with conversation summary"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            # Create a summary
            summary = f"User asked: {query[:100]}... System responded with information about: "

            # Extract topics from response
            topics = []
            if "photo" in response.lower():
                topics.append("photos")
            if "video" in response.lower():
                topics.append("videos")
            if "google" in response.lower():
                topics.append("Google data")
            if "found" in response.lower():
                match = re.search(r'found\s+(\d+)', response.lower())
                if match:
                    topics.append(f"{match.group(1)} items")

            summary += ", ".join(topics) if topics else "general information"

            cur.execute("""
                INSERT INTO learning_history
                (input_text, learned_output, model_used, quality_score, created_at, metadata)
                VALUES (%s, %s, %s, %s, NOW(), %s)
            """, (
                query[:500],
                summary,
                "auto_learning",
                importance,
                Json({
                    "type": "conversation_summary",
                    "response_length": len(response),
                    "topics": topics
                })
            ))

            conn.commit()
            cur.close()
            conn.close()

        except Exception as e:
            logger.error(f"Error updating learning history: {e}")

    def _classify_memory_type(self, query: str) -> str:
        """Classify the type of memory based on content"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["photo", "picture", "image"]):
            return "personal_media"
        elif any(word in query_lower for word in ["video", "movie", "animation"]):
            return "video_content"
        elif any(word in query_lower for word in ["anime", "comfyui", "generate"]):
            return "creative_work"
        elif any(word in query_lower for word in ["google", "takeout", "backup"]):
            return "data_archive"
        elif any(word in query_lower for word in ["code", "program", "function"]):
            return "technical"
        else:
            return "general"

    def _calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment score (-1 to 1)"""
        positive_words = ["good", "great", "found", "success", "complete", "working"]
        negative_words = ["error", "fail", "not", "missing", "broken", "issue"]

        text_lower = text.lower()
        positive = sum(1 for word in positive_words if word in text_lower)
        negative = sum(1 for word in negative_words if word in text_lower)

        total = positive + negative
        if total == 0:
            return 0.0

        return (positive - negative) / total


# Global instance
learning_middleware = LearningMiddleware()


async def apply_learning(request_data: Dict, response_data: Dict) -> None:
    """Apply learning middleware to a conversation"""
    await learning_middleware.process_conversation(request_data, response_data)