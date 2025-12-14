#!/usr/bin/env python3
"""
Conversation Context Middleware for Echo Brain
Maintains conversation history and context across messages
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor, Json
import asyncio
from collections import deque

logger = logging.getLogger(__name__)


class ConversationContextManager:
    """Manages conversation history and context retrieval"""

    def __init__(self):
        self.db_config = {
            "host": "localhost",
            "database": "echo_brain",
            "user": "patrick",
            "password": "***REMOVED***"
        }
        # In-memory cache for recent conversations
        self.context_cache = {}
        self.max_context_messages = 10  # Keep last 10 messages per conversation
        self.context_window_hours = 24  # Look back 24 hours for context

    async def get_conversation_context(self, conversation_id: str,
                                      current_query: str = None) -> Dict[str, Any]:
        """Retrieve full conversation context"""
        try:
            # Check cache first
            if conversation_id in self.context_cache:
                cached = self.context_cache[conversation_id]
                # Update cache timestamp
                cached['last_accessed'] = datetime.now()
            else:
                # Load from database
                cached = await self._load_conversation_from_db(conversation_id)
                self.context_cache[conversation_id] = cached

            # Build context summary
            context = {
                "conversation_id": conversation_id,
                "message_count": len(cached.get('messages', [])),
                "messages": cached.get('messages', [])[-self.max_context_messages:],
                "topics": cached.get('topics', []),
                "entities": cached.get('entities', {}),
                "user_preferences": cached.get('preferences', {}),
                "session_start": cached.get('session_start'),
                "last_message_time": cached.get('last_message_time'),
                "current_context": self._build_context_string(cached, current_query)
            }

            return context

        except Exception as e:
            logger.error(f"Error retrieving conversation context: {e}")
            return {
                "conversation_id": conversation_id,
                "message_count": 0,
                "messages": [],
                "current_context": ""
            }

    async def _load_conversation_from_db(self, conversation_id: str) -> Dict:
        """Load conversation history from database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Get recent messages for this conversation
            cur.execute("""
                SELECT id, user_query, response, model_used, timestamp,
                       intent, confidence, metadata, escalation_path
                FROM conversations
                WHERE conversation_id = %s
                  AND timestamp > NOW() - INTERVAL '%s hours'
                ORDER BY timestamp DESC
                LIMIT %s
            """, (conversation_id, self.context_window_hours, 50))

            messages = cur.fetchall()

            # Process messages
            processed_messages = []
            topics = set()
            entities = {}

            for msg in reversed(messages):  # Reverse to get chronological order
                processed_msg = {
                    "id": msg['id'],
                    "user": msg['user_query'],
                    "assistant": msg['response'],
                    "timestamp": msg['timestamp'].isoformat() if msg['timestamp'] else None,
                    "model": msg['model_used'],
                    "intent": msg['intent'],
                    "confidence": msg['confidence']
                }
                processed_messages.append(processed_msg)

                # Extract topics and entities from metadata
                if msg['metadata']:
                    if 'topics' in msg['metadata']:
                        topics.update(msg['metadata']['topics'])
                    if 'entities' in msg['metadata']:
                        entities.update(msg['metadata']['entities'])

            # Get user preferences from episodic memory
            cur.execute("""
                SELECT learned_fact, importance
                FROM echo_episodic_memory
                WHERE conversation_id = %s
                  AND importance > 0.6
                ORDER BY created_at DESC
                LIMIT 10
            """, (conversation_id,))

            learned_facts = cur.fetchall()
            preferences = {}
            for fact in learned_facts:
                if fact['learned_fact']:
                    preferences[fact['learned_fact'][:50]] = fact['importance']

            cur.close()
            conn.close()

            return {
                "messages": processed_messages,
                "topics": list(topics),
                "entities": entities,
                "preferences": preferences,
                "session_start": messages[-1]['timestamp'] if messages else datetime.now(),
                "last_message_time": messages[0]['timestamp'] if messages else datetime.now(),
                "last_accessed": datetime.now()
            }

        except Exception as e:
            logger.error(f"Error loading conversation from DB: {e}")
            return {
                "messages": [],
                "topics": [],
                "entities": {},
                "preferences": {},
                "last_accessed": datetime.now()
            }

    def _build_context_string(self, cached_data: Dict, current_query: str = None) -> str:
        """Build a context string for the LLM"""
        context_parts = []

        # Add conversation summary
        messages = cached_data.get('messages', [])
        if messages:
            context_parts.append(f"[Conversation History - Last {len(messages)} messages]")

            # Include last few exchanges
            for msg in messages[-5:]:
                if msg.get('user'):
                    context_parts.append(f"User: {msg['user'][:200]}")
                if msg.get('assistant'):
                    context_parts.append(f"Assistant: {msg['assistant'][:200]}")

        # Add learned preferences
        preferences = cached_data.get('preferences', {})
        if preferences:
            context_parts.append(f"\n[User Preferences]")
            for pref, importance in list(preferences.items())[:5]:
                context_parts.append(f"- {pref} (importance: {importance:.1f})")

        # Add current topics
        topics = cached_data.get('topics', [])
        if topics:
            context_parts.append(f"\n[Current Topics]: {', '.join(topics[:10])}")

        # Add entities
        entities = cached_data.get('entities', {})
        if entities:
            context_parts.append(f"\n[Mentioned Entities]: {', '.join(list(entities.keys())[:10])}")

        return "\n".join(context_parts)

    async def save_conversation_turn(self, conversation_id: str, user_query: str,
                                    response: str, metadata: Dict = None) -> None:
        """Save a conversation turn to the database and cache"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            # Extract metadata
            model_used = metadata.get('model_used', 'unknown')
            processing_time = metadata.get('processing_time', 0)
            intent = metadata.get('intent', 'general')
            confidence = metadata.get('confidence', 0.5)
            requires_clarification = metadata.get('requires_clarification', False)
            user_id = metadata.get('user_id', 'unknown')

            # Save to conversations table
            cur.execute("""
                INSERT INTO conversations
                (conversation_id, user_query, response, model_used, processing_time,
                 timestamp, intent, confidence, requires_clarification, user_id, metadata)
                VALUES (%s, %s, %s, %s, %s, NOW(), %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                conversation_id,
                user_query,
                response,
                model_used,
                processing_time,
                intent,
                confidence,
                requires_clarification,
                user_id,
                Json(metadata) if metadata else None
            ))

            conversation_row_id = cur.fetchone()[0]
            conn.commit()

            # Update cache
            if conversation_id not in self.context_cache:
                self.context_cache[conversation_id] = {
                    "messages": [],
                    "topics": [],
                    "entities": {},
                    "preferences": {},
                    "last_accessed": datetime.now()
                }

            # Add to cached messages
            self.context_cache[conversation_id]['messages'].append({
                "id": conversation_row_id,
                "user": user_query,
                "assistant": response,
                "timestamp": datetime.now().isoformat(),
                "model": model_used,
                "intent": intent,
                "confidence": confidence
            })

            # Keep only recent messages in cache
            if len(self.context_cache[conversation_id]['messages']) > self.max_context_messages * 2:
                self.context_cache[conversation_id]['messages'] = \
                    self.context_cache[conversation_id]['messages'][-self.max_context_messages:]

            # Update last message time
            self.context_cache[conversation_id]['last_message_time'] = datetime.now()

            cur.close()
            conn.close()

            logger.info(f"Saved conversation turn for {conversation_id}")

        except Exception as e:
            logger.error(f"Error saving conversation turn: {e}")

    async def get_similar_conversations(self, query: str, limit: int = 5) -> List[Dict]:
        """Find similar past conversations"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Simple similarity search (could be enhanced with embeddings)
            cur.execute("""
                SELECT DISTINCT conversation_id, user_query, response, timestamp
                FROM conversations
                WHERE user_query ILIKE %s
                   OR response ILIKE %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (f'%{query[:50]}%', f'%{query[:50]}%', limit))

            similar = cur.fetchall()
            cur.close()
            conn.close()

            return similar

        except Exception as e:
            logger.error(f"Error finding similar conversations: {e}")
            return []

    def clear_old_cache(self) -> None:
        """Clear old cached conversations"""
        now = datetime.now()
        to_remove = []

        for conv_id, data in self.context_cache.items():
            last_accessed = data.get('last_accessed', now)
            if (now - last_accessed).total_seconds() > 3600:  # 1 hour
                to_remove.append(conv_id)

        for conv_id in to_remove:
            del self.context_cache[conv_id]

        if to_remove:
            logger.info(f"Cleared {len(to_remove)} old conversations from cache")


# Global instance
conversation_context = ConversationContextManager()


async def inject_context(request_data: Dict) -> Dict:
    """Inject conversation context into a request"""
    conversation_id = request_data.get('conversation_id', 'default')
    query = request_data.get('query', '')

    # Get context
    context = await conversation_context.get_conversation_context(conversation_id, query)

    # Inject context into request
    if context['current_context']:
        # Prepend context to query
        request_data['original_query'] = query
        request_data['query'] = f"{context['current_context']}\n\nCurrent message: {query}"
        request_data['conversation_context'] = context

    return request_data


async def save_turn(conversation_id: str, query: str, response: str, metadata: Dict = None):
    """Save a conversation turn"""
    await conversation_context.save_conversation_turn(
        conversation_id, query, response, metadata
    )