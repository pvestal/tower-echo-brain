#!/usr/bin/env python3
"""
Conversation Management System for Echo Brain
Integrates all conversation, memory, and context systems
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import psycopg2
from psycopg2.extras import Json, RealDictCursor
import asyncio
import hashlib

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Single source of truth for all conversation handling in Echo
    Merges conversation_manager, user_context, episodic memory, and learning
    """

    def __init__(self):
        self.db_config = {
            "host": "localhost",
            "database": "echo_brain",
            "user": "patrick",
            "password": "***REMOVED***"
        }

        # Cache for conversation data
        self.conversation_cache = {}

        # Unified cache for all context data
        self.unified_cache = {}

        # Components we're unifying
        self.components = {
            "conversation_history": [],
            "episodic_memories": [],
            "user_context": {},
            "learning_pipeline": None,
            "vector_memory": None
        }

        logger.info("🔄 Conversation Manager initialized")

    async def process_message(self, conversation_id: str, query_text: str,
                             user_id: str = None, metadata: Dict = None) -> Dict:
        """
        Main entry point for all conversation processing
        This replaces multiple separate systems
        """

        start_time = datetime.now()
        print(f"\n🔄🔄🔄 UNIFIED CONVERSATION CALLED: {conversation_id}: {query_text[:50]}...\n")
        logger.info(f"🔄 UNIFIED CONVERSATION: Processing message for {conversation_id}: {query_text[:50]}")

        # Step 1: Load or create unified conversation context
        context = await self._get_unified_context(conversation_id, user_id)

        # Step 2: Enhance query with context
        enhanced_query = await self._enhance_with_context(query_text, context)

        # Step 3: Process query through intelligence system
        # (This would integrate with existing Echo intelligence)

        # Step 4: Save EVERYTHING
        await self._save_complete_turn(
            conversation_id,
            user_id,
            query_text,
            enhanced_query,
            context,
            metadata
        )

        # Step 5: Update all memory systems
        await self._update_all_memories(
            conversation_id,
            user_id,
            query_text,
            context
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        return {
            "enhanced_query": enhanced_query,
            "context": context,
            "processing_time": processing_time,
            "memories_updated": True
        }

    async def _get_unified_context(self, conversation_id: str, user_id: str = None) -> Dict:
        """
        Get ALL context for a conversation from ALL sources
        """

        # Check cache first
        cache_key = f"{conversation_id}:{user_id}"
        if cache_key in self.unified_cache:
            return self.unified_cache[cache_key]

        context = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "conversation_history": [],
            "episodic_memories": [],
            "learned_facts": [],
            "user_preferences": {},
            "active_tasks": [],
            "entities": {},
            "topics": [],
            "temporal_context": {},
            "media_references": []
        }

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # 1. Get conversation history
            cur.execute("""
                SELECT user_query, response, timestamp, model_used, confidence, metadata
                FROM conversations
                WHERE conversation_id = %s
                ORDER BY timestamp DESC
                LIMIT 20
            """, (conversation_id,))

            history = cur.fetchall()
            for msg in reversed(history):
                context["conversation_history"].append({
                    "user": msg["user_query"],
                    "assistant": msg["response"],
                    "timestamp": msg["timestamp"].isoformat() if msg["timestamp"] else None,
                    "model": msg["model_used"],
                    "confidence": msg["confidence"]
                })

            # 2. Get episodic memories
            cur.execute("""
                SELECT memory_type, content, learned_fact, importance, created_at
                FROM echo_episodic_memory
                WHERE conversation_id = %s OR conversation_id LIKE %s
                ORDER BY importance DESC, created_at DESC
                LIMIT 10
            """, (conversation_id, f"%{user_id}%"))

            memories = cur.fetchall()
            for mem in memories:
                context["episodic_memories"].append({
                    "type": mem["memory_type"],
                    "content": mem["content"],
                    "fact": mem["learned_fact"],
                    "importance": mem["importance"],
                    "timestamp": mem["created_at"].isoformat() if mem["created_at"] else None
                })

                # Extract learned facts
                if mem["learned_fact"]:
                    context["learned_facts"].append(mem["learned_fact"])

            # 3. Get learning history
            cur.execute("""
                SELECT learned_fact, confidence, metadata
                FROM learning_history
                WHERE conversation_id = %s
                ORDER BY created_at DESC
                LIMIT 5
            """, (conversation_id,))

            learning = cur.fetchall()
            for item in learning:
                if item["learned_fact"]:
                    context["learned_facts"].append(item["learned_fact"])

            # 4. Extract entities and topics from metadata
            for msg in history:
                if msg["metadata"]:
                    if "entities" in msg["metadata"]:
                        context["entities"].update(msg["metadata"]["entities"])
                    if "topics" in msg["metadata"]:
                        context["topics"].extend(msg["metadata"]["topics"])

            # Remove duplicates
            context["topics"] = list(set(context["topics"]))
            context["learned_facts"] = list(set(context["learned_facts"]))

            cur.close()
            conn.close()

            # Cache the context
            self.unified_cache[cache_key] = context

            logger.info(f"📚 Loaded unified context: {len(context['conversation_history'])} messages, "
                       f"{len(context['episodic_memories'])} memories, "
                       f"{len(context['learned_facts'])} facts")

        except Exception as e:
            logger.error(f"Error loading unified context: {e}")

        return context

    async def _enhance_with_context(self, query: str, context: Dict) -> str:
        """
        Enhance the query with all available context
        """

        enhanced_parts = []

        # Add conversation history summary
        if context["conversation_history"]:
            recent = context["conversation_history"][-3:]  # Last 3 exchanges
            if recent:
                enhanced_parts.append("[Recent conversation]")
                for msg in recent:
                    if msg["user"]:
                        enhanced_parts.append(f"User: {msg['user'][:100]}")
                    if msg["assistant"]:
                        enhanced_parts.append(f"You: {msg['assistant'][:100]}")

        # Add learned facts
        if context["learned_facts"]:
            enhanced_parts.append(f"\n[Known facts]: {'; '.join(context['learned_facts'][:5])}")

        # Add entities
        if context["entities"]:
            enhanced_parts.append(f"\n[Entities]: {', '.join(list(context['entities'].keys())[:10])}")

        # Add the current query
        enhanced_parts.append(f"\n[Current message]: {query}")

        return "\n".join(enhanced_parts)

    async def _save_complete_turn(self, conversation_id: str, user_id: str,
                                  original_query: str, enhanced_query: str,
                                  context: Dict, metadata: Dict = None):
        """
        Save conversation turn to ALL relevant tables
        """

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            # Prepare metadata
            full_metadata = {
                "user_id": user_id,
                "original_query": original_query,
                "enhanced_query": enhanced_query,
                "context_size": len(context.get("conversation_history", [])),
                "entities": context.get("entities", {}),
                "topics": context.get("topics", []),
                "timestamp": datetime.now().isoformat()
            }

            if metadata:
                full_metadata.update(metadata)

            # 1. Save/update conversation record
            cur.execute("""
                INSERT INTO echo_conversations
                (conversation_id, user_id, context, last_interaction)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (conversation_id) DO UPDATE SET
                    context = EXCLUDED.context,
                    last_interaction = NOW(),
                    message_count = echo_conversations.message_count + 1
            """, (
                conversation_id,
                user_id,
                Json(full_metadata)
            ))

            # 2. Save the actual message to echo_unified_interactions
            cur.execute("""
                INSERT INTO echo_unified_interactions
                (conversation_id, user_id, query, response, model_used, processing_time, metadata, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                conversation_id,
                user_id,
                original_query,
                "",  # Response will be updated later
                "unified_system",  # Default model name
                0.0,  # Default processing time
                Json(full_metadata)
            ))

            # 2. Save to episodic memory
            importance = self._calculate_importance(original_query, context)

            cur.execute("""
                INSERT INTO echo_episodic_memory
                (conversation_id, memory_type, content, importance, user_query,
                 echo_response, model_used, learned_fact, created_at, access_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), 1)
            """, (
                conversation_id,
                "conversation",
                f"User said: {original_query[:200]}",
                importance,
                original_query,
                "",  # Will be updated when response is generated
                "unified_system",
                self._extract_key_fact(original_query)
            ))

            # 3. Save to learnings if important
            if importance > 0.5:
                cur.execute("""
                    INSERT INTO learning_history
                    (conversation_id, learned_fact, fact_type, confidence, learning_type, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    conversation_id,
                    self._extract_key_fact(original_query),
                    "pattern",
                    importance,
                    "unified_system",
                    Json({"source": "unified_system", "query": original_query, "user_id": user_id})
                ))

            conn.commit()
            cur.close()
            conn.close()

            logger.info(f"💾 Saved complete conversation turn (importance: {importance:.2f})")

        except Exception as e:
            logger.error(f"Error saving conversation turn: {e}")

    async def _update_all_memories(self, conversation_id: str, user_id: str,
                                   query: str, context: Dict):
        """
        Update all memory systems with new information
        """

        # Extract new entities
        new_entities = self._extract_entities(query)
        if new_entities:
            context["entities"].update(new_entities)

        # Extract topics
        new_topics = self._extract_topics(query)
        if new_topics:
            context["topics"].extend(new_topics)
            context["topics"] = list(set(context["topics"]))

        # Update cache
        cache_key = f"{conversation_id}:{user_id}"
        self.unified_cache[cache_key] = context

        logger.info(f"🔄 Updated all memory systems")

    def _calculate_importance(self, query: str, context: Dict) -> float:
        """
        Calculate importance score for a message
        """

        importance = 0.3  # Base importance

        # Boost for questions
        if "?" in query:
            importance += 0.2

        # Boost for keywords
        important_keywords = ["remember", "important", "don't forget", "need", "must", "critical"]
        if any(kw in query.lower() for kw in important_keywords):
            importance += 0.3

        # Boost for media/data references
        media_keywords = ["photo", "image", "video", "file", "data", "document"]
        if any(kw in query.lower() for kw in media_keywords):
            importance += 0.2

        # Boost if continuing a conversation
        if len(context.get("conversation_history", [])) > 2:
            importance += 0.1

        return min(importance, 1.0)

    def _extract_key_fact(self, text: str) -> str:
        """
        Extract a key fact from text
        """

        # Simple extraction - could be enhanced with NLP
        facts = []

        # Look for "I am", "I have", "I like" patterns
        personal_patterns = ["i am", "i have", "i like", "my name", "i want", "i need"]
        text_lower = text.lower()

        for pattern in personal_patterns:
            if pattern in text_lower:
                # Extract the sentence containing this pattern
                sentences = text.split(".")
                for sentence in sentences:
                    if pattern in sentence.lower():
                        facts.append(sentence.strip())
                        break

        # Look for definitions or declarations
        if " is " in text_lower or " are " in text_lower:
            facts.append(text[:100])

        return "; ".join(facts) if facts else text[:100]

    def _extract_entities(self, text: str) -> Dict[str, str]:
        """
        Extract named entities from text
        """

        entities = {}

        # Simple entity extraction - could use NER model
        # Look for capitalized words (potential names)
        words = text.split()
        for word in words:
            if word and word[0].isupper() and len(word) > 1:
                entities[word] = "name"

        return entities

    def _extract_topics(self, text: str) -> List[str]:
        """
        Extract topics from text
        """

        topics = []
        topic_keywords = {
            "technology": ["code", "program", "software", "computer", "ai"],
            "media": ["photo", "video", "image", "picture", "media"],
            "anime": ["anime", "manga", "character", "comfyui"],
            "personal": ["i", "me", "my", "name", "like"],
            "data": ["data", "memory", "database", "information"]
        }

        text_lower = text.lower()
        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                topics.append(topic)

        return topics

    async def save_response(self, conversation_id: str, response: str, metadata: Dict = None):
        """
        Save the assistant's response and update all relevant tables
        """

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            # Update the most recent interaction with response
            cur.execute("""
                UPDATE echo_unified_interactions
                SET response = %s,
                    model_used = %s,
                    processing_time = %s,
                    confidence = %s
                WHERE id = (
                    SELECT id FROM echo_unified_interactions
                    WHERE conversation_id = %s
                      AND (response = '' OR response IS NULL)
                    ORDER BY timestamp DESC
                    LIMIT 1
                )
            """, (
                response,
                metadata.get("model_used", "unified_system"),
                metadata.get("processing_time", 0),
                metadata.get("confidence", 0.5),
                conversation_id
            ))

            # Update episodic memory with response
            cur.execute("""
                UPDATE echo_episodic_memory
                SET echo_response = %s
                WHERE id = (
                    SELECT id FROM echo_episodic_memory
                    WHERE conversation_id = %s
                      AND echo_response = ''
                    ORDER BY created_at DESC
                    LIMIT 1
                )
            """, (response, conversation_id))

            conn.commit()
            cur.close()
            conn.close()

            logger.info(f"💾 Saved response for {conversation_id}")

        except Exception as e:
            logger.error(f"Error saving response: {e}")


# Global instance
conversation_handler = ConversationManager()


# Helper functions for easy integration
async def process_message(conversation_id: str, query_text: str,
                         user_id: str = None, metadata: Dict = None) -> Dict:
    """Process a message through the conversation system"""
    return await conversation_handler.process_message(
        conversation_id, query_text, user_id, metadata
    )


async def save_response(conversation_id: str, response: str, metadata: Dict = None):
    """Save response through the conversation system"""
    await conversation_handler.save_response(conversation_id, response, metadata)