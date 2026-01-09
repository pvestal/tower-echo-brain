#!/usr/bin/env python3
"""
User Context Manager for Echo Brain
Manages individual user memory, preferences, styles, and access
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import asyncio
import aiofiles
import hashlib

logger = logging.getLogger(__name__)

class UserContext:
    """Individual user context with memory, preferences, and styles"""

    def __init__(self, username: str):
        self.username = username
        self.user_id = self._generate_user_id(username)
        self.created_at = datetime.now()
        self.last_interaction = datetime.now()

        # User-specific memory
        self.conversation_history = []
        self.knowledge_graph = {}
        self.learned_patterns = []
        self.personal_facts = {}

        # User preferences
        self.preferences = {
            "response_style": "balanced",  # verbose, concise, balanced, technical
            "personality": "professional",  # friendly, professional, casual, formal
            "temperature": 0.7,
            "max_tokens": 2048,
            "preferred_model": "llama3.1:8b",
            "notification_channels": ["telegram"],
            "timezone": "America/New_York",
            "language": "en",
            "dark_mode": True
        }

        # User styles and patterns
        self.communication_style = {
            "greeting_preference": "informal",  # formal, informal, none
            "emoji_usage": False,
            "technical_level": "advanced",  # beginner, intermediate, advanced, expert
            "verbosity": "normal",  # minimal, normal, detailed
            "humor_level": "occasional"  # none, occasional, frequent
        }

        # Access controls (individual for each user)
        self.permissions = {
            "execute_code": False,
            "system_commands": False,
            "image_generation": True,
            "llm_access": True,
            "file_access": False,
            "network_access": False,
            "database_access": False,
            "service_control": False
        }

        # User-specific metrics
        self.usage_metrics = {
            "total_interactions": 0,
            "total_tokens_used": 0,
            "images_generated": 0,
            "tasks_completed": 0,
            "learning_entries": 0
        }

    def _generate_user_id(self, username: str) -> str:
        """Generate unique user ID"""
        return hashlib.sha256(username.encode()).hexdigest()[:16]

    def update_interaction(self):
        """Update last interaction timestamp"""
        self.last_interaction = datetime.now()
        self.usage_metrics["total_interactions"] += 1

    def add_to_memory(self, key: str, value: Any):
        """Add information to user's personal memory"""
        self.personal_facts[key] = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "confidence": 1.0
        }
        self.usage_metrics["learning_entries"] += 1

    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of user context"""
        return {
            "username": self.username,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_interaction": self.last_interaction.isoformat(),
            "preferences": self.preferences,
            "communication_style": self.communication_style,
            "permissions": self.permissions,
            "usage_metrics": self.usage_metrics,
            "memory_size": {
                "conversations": len(self.conversation_history),
                "facts": len(self.personal_facts),
                "patterns": len(self.learned_patterns)
            }
        }


class UserContextManager:
    """Manages all user contexts with persistence"""

    def __init__(self, data_dir: str = "/opt/tower-echo-brain/data/user_contexts"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.contexts: Dict[str, UserContext] = {}
        self.creator_username = "patrick"

        logger.info(f"📁 User context manager initialized with data dir: {self.data_dir}")

    async def get_or_create_context(self, username: str) -> UserContext:
        """Get existing context or create new one for user"""
        if username in self.contexts:
            context = self.contexts[username]
            context.update_interaction()
            return context

        # Try to load from disk
        context = await self._load_context(username)
        if context:
            self.contexts[username] = context
            context.update_interaction()
            return context

        # Create new context
        context = UserContext(username)

        # Special permissions for creator
        if username == self.creator_username:
            context.permissions = {
                "execute_code": True,
                "system_commands": True,
                "image_generation": True,
                "llm_access": True,
                "file_access": True,
                "network_access": True,
                "database_access": True,
                "service_control": True
            }
            context.preferences["response_style"] = "technical"
            context.communication_style["technical_level"] = "expert"
            logger.info(f"👑 Creator context initialized with full permissions for {username}")

        self.contexts[username] = context
        await self._save_context(context)
        logger.info(f"✨ New user context created for {username}")

        return context

    async def update_preference(self, username: str, key: str, value: Any) -> bool:
        """Update user preference"""
        context = await self.get_or_create_context(username)

        if key in context.preferences:
            context.preferences[key] = value
            await self._save_context(context)
            logger.info(f"Updated preference {key}={value} for {username}")
            return True

        logger.warning(f"Unknown preference key: {key}")
        return False

    async def update_style(self, username: str, key: str, value: Any) -> bool:
        """Update user communication style"""
        context = await self.get_or_create_context(username)

        if key in context.communication_style:
            context.communication_style[key] = value
            await self._save_context(context)
            logger.info(f"Updated style {key}={value} for {username}")
            return True

        logger.warning(f"Unknown style key: {key}")
        return False

    async def add_conversation(self, username: str, role: str, content: str) -> None:
        """Add to user's conversation history"""
        context = await self.get_or_create_context(username)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content
        }

        context.conversation_history.append(entry)

        # Keep only last 100 messages per user
        if len(context.conversation_history) > 100:
            context.conversation_history = context.conversation_history[-100:]

        await self._save_context(context)

    async def learn_about_user(self, username: str, fact_key: str, fact_value: Any) -> None:
        """Learn and remember something about the user"""
        context = await self.get_or_create_context(username)
        context.add_to_memory(fact_key, fact_value)
        await self._save_context(context)
        logger.info(f"📝 Learned about {username}: {fact_key} = {fact_value}")

    async def get_user_memory(self, username: str) -> Dict[str, Any]:
        """Get all memories about a user"""
        context = await self.get_or_create_context(username)
        return context.personal_facts

    async def check_permission(self, username: str, permission: str) -> bool:
        """Check if user has specific permission"""
        context = await self.get_or_create_context(username)
        return context.permissions.get(permission, False)

    async def _save_context(self, context: UserContext) -> None:
        """Save context to disk"""
        file_path = self.data_dir / f"{context.username}.json"

        data = {
            "username": context.username,
            "user_id": context.user_id,
            "created_at": context.created_at.isoformat(),
            "last_interaction": context.last_interaction.isoformat(),
            "conversation_history": context.conversation_history[-50:],  # Save last 50
            "knowledge_graph": context.knowledge_graph,
            "learned_patterns": context.learned_patterns,
            "personal_facts": context.personal_facts,
            "preferences": context.preferences,
            "communication_style": context.communication_style,
            "permissions": context.permissions,
            "usage_metrics": context.usage_metrics
        }

        async with aiofiles.open(file_path, 'w') as f:
            await f.write(json.dumps(data, indent=2))

    async def _load_context(self, username: str) -> Optional[UserContext]:
        """Load context from disk"""
        file_path = self.data_dir / f"{username}.json"

        if not file_path.exists():
            return None

        try:
            async with aiofiles.open(file_path, 'r') as f:
                data = json.loads(await f.read())

            context = UserContext(username)
            context.user_id = data.get("user_id", context.user_id)
            context.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
            context.last_interaction = datetime.fromisoformat(data.get("last_interaction", datetime.now().isoformat()))
            context.conversation_history = data.get("conversation_history", [])
            context.knowledge_graph = data.get("knowledge_graph", {})
            context.learned_patterns = data.get("learned_patterns", [])
            context.personal_facts = data.get("personal_facts", {})
            context.preferences.update(data.get("preferences", {}))
            context.communication_style.update(data.get("communication_style", {}))
            context.permissions.update(data.get("permissions", {}))
            context.usage_metrics.update(data.get("usage_metrics", {}))

            logger.info(f"📂 Loaded existing context for {username}")
            return context

        except Exception as e:
            logger.error(f"Failed to load context for {username}: {e}")
            return None

    async def get_all_users(self) -> List[Dict[str, Any]]:
        """Get summary of all users"""
        users = []

        # Load all context files
        for file_path in self.data_dir.glob("*.json"):
            username = file_path.stem
            context = await self.get_or_create_context(username)
            users.append(context.get_context_summary())

        return users

    async def cleanup_old_conversations(self, days: int = 30) -> None:
        """Clean up old conversations older than specified days"""
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)

        for username, context in self.contexts.items():
            original_count = len(context.conversation_history)
            context.conversation_history = [
                entry for entry in context.conversation_history
                if datetime.fromisoformat(entry["timestamp"]).timestamp() > cutoff
            ]

            removed = original_count - len(context.conversation_history)
            if removed > 0:
                await self._save_context(context)
                logger.info(f"🗑️ Cleaned up {removed} old conversations for {username}")

# Singleton instance
user_context_manager = UserContextManager()

async def get_user_context_manager() -> UserContextManager:
    """Get user context manager instance"""
    return user_context_manager