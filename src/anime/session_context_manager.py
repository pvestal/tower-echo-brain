#!/usr/bin/env python3
"""
Cross-Platform Session Context Manager
Ensures project context persists across Telegram and browser interfaces.

This system:
- Manages session state across different platforms (Telegram, Browser, API)
- Implements session inheritance from previous projects and characters
- Provides context continuity for ongoing creative work
- Handles session expiration and cleanup
- Enables seamless transitions between interfaces
"""

import asyncio
import json
import logging
import os
import psycopg2
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import sys
import os
sys.path.append('/opt/tower-echo-brain/src')

from db.database import database

logger = logging.getLogger(__name__)

class Platform(Enum):
    """Supported platforms for anime generation"""
    ECHO_BRAIN = "echo_brain"
    TELEGRAM = "telegram"
    BROWSER = "browser"
    API = "api"
    WEBHOOK = "webhook"

class SessionState(Enum):
    """Session lifecycle states"""
    ACTIVE = "active"
    IDLE = "idle"
    SUSPENDED = "suspended"
    EXPIRED = "expired"
    MIGRATED = "migrated"

@dataclass
class SessionContext:
    """Complete session context data"""
    session_id: str
    user_id: str
    platform: Platform
    state: SessionState
    created_at: datetime
    last_activity: datetime
    expires_at: datetime

    # Project context
    current_project: Optional[str] = None
    project_binding_id: Optional[int] = None
    active_character: Optional[str] = None

    # Generation context
    pending_generations: List[Dict] = None
    last_generation_id: Optional[str] = None
    generation_history: List[Dict] = None

    # Conversation context
    conversation_thread: List[Dict] = None
    context_memory: Dict[str, Any] = None

    # Platform-specific data
    telegram_chat_id: Optional[str] = None
    browser_session_token: Optional[str] = None
    api_client_id: Optional[str] = None

    # Inheritance and migration
    inherited_from: Optional[str] = None
    migration_target: Optional[str] = None

    def __post_init__(self):
        if self.pending_generations is None:
            self.pending_generations = []
        if self.generation_history is None:
            self.generation_history = []
        if self.conversation_thread is None:
            self.conversation_thread = []
        if self.context_memory is None:
            self.context_memory = {}

class SessionContextManager:
    """Manages cross-platform session contexts for anime generation"""

    def __init__(self):
        self.db_config = {
            "database": os.environ.get("DB_NAME", "echo_brain"),
            "user": os.environ.get("DB_USER", "patrick")
        }

        # Session cache for active sessions
        self.active_sessions: Dict[str, SessionContext] = {}
        self.platform_sessions: Dict[Platform, Set[str]] = {
            platform: set() for platform in Platform
        }

        # Session configuration
        self.session_timeout = timedelta(hours=24)  # Sessions expire after 24 hours
        self.idle_timeout = timedelta(hours=2)      # Mark idle after 2 hours
        self.max_sessions_per_user = 5             # Limit concurrent sessions

        # Database initialization will be called explicitly
        self._db_initialized = False

    async def initialize_database(self):
        """Initialize session management tables"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Add missing columns to existing session table
            cursor.execute("""
                ALTER TABLE anime_echo_sessions
                ADD COLUMN IF NOT EXISTS session_state VARCHAR(20) DEFAULT 'active'
            """)

            cursor.execute("""
                ALTER TABLE anime_echo_sessions
                ADD COLUMN IF NOT EXISTS current_project VARCHAR(200)
            """)

            cursor.execute("""
                ALTER TABLE anime_echo_sessions
                ADD COLUMN IF NOT EXISTS last_generation_id VARCHAR(200)
            """)

            cursor.execute("""
                ALTER TABLE anime_echo_sessions
                ADD COLUMN IF NOT EXISTS generation_history JSONB DEFAULT '[]'
            """)

            cursor.execute("""
                ALTER TABLE anime_echo_sessions
                ADD COLUMN IF NOT EXISTS conversation_thread JSONB DEFAULT '[]'
            """)

            cursor.execute("""
                ALTER TABLE anime_echo_sessions
                ADD COLUMN IF NOT EXISTS context_memory JSONB DEFAULT '{}'
            """)

            cursor.execute("""
                ALTER TABLE anime_echo_sessions
                ADD COLUMN IF NOT EXISTS telegram_chat_id VARCHAR(100)
            """)

            cursor.execute("""
                ALTER TABLE anime_echo_sessions
                ADD COLUMN IF NOT EXISTS browser_session_token VARCHAR(200)
            """)

            cursor.execute("""
                ALTER TABLE anime_echo_sessions
                ADD COLUMN IF NOT EXISTS api_client_id VARCHAR(100)
            """)

            cursor.execute("""
                ALTER TABLE anime_echo_sessions
                ADD COLUMN IF NOT EXISTS inherited_from VARCHAR(100)
            """)

            cursor.execute("""
                ALTER TABLE anime_echo_sessions
                ADD COLUMN IF NOT EXISTS migration_target VARCHAR(100)
            """)

            # Session transitions table for tracking migrations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS anime_session_transitions (
                    id SERIAL PRIMARY KEY,
                    from_session_id VARCHAR(100) NOT NULL,
                    to_session_id VARCHAR(100) NOT NULL,
                    from_platform VARCHAR(50) NOT NULL,
                    to_platform VARCHAR(50) NOT NULL,
                    user_id VARCHAR(100) DEFAULT 'patrick',
                    transition_type VARCHAR(50) NOT NULL, -- migration, inheritance, fork
                    context_transferred JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Session analytics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS anime_session_analytics (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(100) NOT NULL,
                    platform VARCHAR(50) NOT NULL,
                    user_id VARCHAR(100) DEFAULT 'patrick',
                    activity_type VARCHAR(50) NOT NULL, -- generation, feedback, migration, idle
                    activity_data JSONB DEFAULT '{}',
                    duration_seconds INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_platform ON anime_echo_sessions(user_id, platform)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_state ON anime_echo_sessions(session_state)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_expires ON anime_echo_sessions(expires_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_transitions_sessions ON anime_session_transitions(from_session_id, to_session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analytics_session ON anime_session_analytics(session_id)")

            conn.commit()
            cursor.close()
            conn.close()
            logger.info("✅ Session context manager database initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize session database: {e}")

    async def create_session(self, user_id: str, platform: Platform,
                           inherit_from: Optional[str] = None,
                           platform_data: Optional[Dict] = None) -> SessionContext:
        """Create new session with optional inheritance"""
        try:
            # Initialize database if not already done
            if not self._db_initialized:
                await self.initialize_database()
                self._db_initialized = True
            session_id = f"{platform.value}_{user_id}_{uuid.uuid4().hex[:8]}"
            now = datetime.now()

            # Check session limits
            await self.cleanup_expired_sessions(user_id)
            await self.enforce_session_limits(user_id)

            # Create session context
            session = SessionContext(
                session_id=session_id,
                user_id=user_id,
                platform=platform,
                state=SessionState.ACTIVE,
                created_at=now,
                last_activity=now,
                expires_at=now + self.session_timeout,
                inherited_from=inherit_from
            )

            # Apply platform-specific data
            if platform_data:
                if platform == Platform.TELEGRAM:
                    session.telegram_chat_id = platform_data.get("chat_id")
                elif platform == Platform.BROWSER:
                    session.browser_session_token = platform_data.get("session_token")
                elif platform == Platform.API:
                    session.api_client_id = platform_data.get("client_id")

            # Inherit context from previous session if specified
            if inherit_from:
                await self.inherit_session_context(session, inherit_from)

            # Store in database
            await self.persist_session(session)

            # Add to active sessions cache
            self.active_sessions[session_id] = session
            self.platform_sessions[platform].add(session_id)

            logger.info(f"🆕 Created session: {session_id} ({platform.value})")

            # Record analytics
            await self.record_session_activity(
                session_id, "session_created",
                {"platform": platform.value, "inherited_from": inherit_from}
            )

            return session

        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise

    async def get_session(self, session_id: str, update_activity: bool = True) -> Optional[SessionContext]:
        """Get session by ID, optionally updating activity timestamp"""
        try:
            # Check cache first
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]

                # Update activity if requested
                if update_activity and session.state == SessionState.ACTIVE:
                    await self.update_session_activity(session_id)

                return session

            # Load from database
            session = await self.load_session_from_db(session_id)

            if session and session.state == SessionState.ACTIVE:
                # Add to cache
                self.active_sessions[session_id] = session
                self.platform_sessions[session.platform].add(session_id)

                # Update activity if requested
                if update_activity:
                    await self.update_session_activity(session_id)

                return session

            return None

        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None

    async def get_user_sessions(self, user_id: str, platform: Optional[Platform] = None) -> List[SessionContext]:
        """Get all active sessions for a user, optionally filtered by platform"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            if platform:
                cursor.execute("""
                    SELECT session_id FROM anime_echo_sessions
                    WHERE user_id = %s AND platform = %s AND session_state = 'active'
                    ORDER BY last_activity DESC
                """, (user_id, platform.value))
            else:
                cursor.execute("""
                    SELECT session_id FROM anime_echo_sessions
                    WHERE user_id = %s AND session_state = 'active'
                    ORDER BY last_activity DESC
                """, (user_id,))

            session_ids = [row[0] for row in cursor.fetchall()]
            cursor.close()
            conn.close()

            # Load sessions
            sessions = []
            for session_id in session_ids:
                session = await self.get_session(session_id, update_activity=False)
                if session:
                    sessions.append(session)

            return sessions

        except Exception as e:
            logger.error(f"Failed to get user sessions: {e}")
            return []

    async def migrate_session(self, from_session_id: str, to_platform: Platform,
                            platform_data: Optional[Dict] = None) -> Optional[SessionContext]:
        """Migrate session from one platform to another"""
        try:
            # Get source session
            source_session = await self.get_session(from_session_id, update_activity=False)
            if not source_session:
                logger.error(f"Source session not found: {from_session_id}")
                return None

            # Create new session on target platform
            new_session = await self.create_session(
                source_session.user_id,
                to_platform,
                inherit_from=from_session_id,
                platform_data=platform_data
            )

            # Transfer complete context
            new_session.current_project = source_session.current_project
            new_session.project_binding_id = source_session.project_binding_id
            new_session.active_character = source_session.active_character
            new_session.pending_generations = source_session.pending_generations.copy()
            new_session.generation_history = source_session.generation_history.copy()
            new_session.conversation_thread = source_session.conversation_thread.copy()
            new_session.context_memory = source_session.context_memory.copy()

            # Mark source session as migrated
            source_session.state = SessionState.MIGRATED
            source_session.migration_target = new_session.session_id

            # Persist changes
            await self.persist_session(new_session)
            await self.persist_session(source_session)

            # Record migration
            await self.record_session_transition(
                from_session_id, new_session.session_id,
                source_session.platform, to_platform, "migration"
            )

            logger.info(f"🔄 Migrated session: {from_session_id} → {new_session.session_id}")
            return new_session

        except Exception as e:
            logger.error(f"Failed to migrate session: {e}")
            return None

    async def fork_session(self, source_session_id: str, to_platform: Platform,
                         context_filter: Optional[List[str]] = None) -> Optional[SessionContext]:
        """Fork session with selective context transfer"""
        try:
            source_session = await self.get_session(source_session_id, update_activity=False)
            if not source_session:
                return None

            # Create forked session
            forked_session = await self.create_session(
                source_session.user_id,
                to_platform,
                inherit_from=source_session_id
            )

            # Transfer selected context
            if not context_filter or "project" in context_filter:
                forked_session.current_project = source_session.current_project
                forked_session.project_binding_id = source_session.project_binding_id

            if not context_filter or "character" in context_filter:
                forked_session.active_character = source_session.active_character

            if not context_filter or "memory" in context_filter:
                forked_session.context_memory = source_session.context_memory.copy()

            # Don't transfer pending generations or conversation thread for forks
            # These are platform-specific

            await self.persist_session(forked_session)

            # Record fork
            await self.record_session_transition(
                source_session_id, forked_session.session_id,
                source_session.platform, to_platform, "fork"
            )

            logger.info(f"🍴 Forked session: {source_session_id} → {forked_session.session_id}")
            return forked_session

        except Exception as e:
            logger.error(f"Failed to fork session: {e}")
            return None

    async def update_session_context(self, session_id: str, context_updates: Dict[str, Any]) -> bool:
        """Update session context data"""
        try:
            session = await self.get_session(session_id, update_activity=True)
            if not session:
                return False

            # Apply updates
            for key, value in context_updates.items():
                if key == "current_project":
                    session.current_project = value
                elif key == "active_character":
                    session.active_character = value
                elif key == "project_binding_id":
                    session.project_binding_id = value
                elif key == "add_generation":
                    session.pending_generations.append(value)
                elif key == "complete_generation":
                    # Move from pending to history
                    generation_id = value.get("generation_id")
                    session.pending_generations = [
                        g for g in session.pending_generations
                        if g.get("generation_id") != generation_id
                    ]
                    session.generation_history.append(value)
                    session.last_generation_id = generation_id
                elif key == "add_conversation":
                    session.conversation_thread.append(value)
                elif key == "update_memory":
                    session.context_memory.update(value)

            # Limit history sizes
            if len(session.generation_history) > 50:
                session.generation_history = session.generation_history[-50:]
            if len(session.conversation_thread) > 100:
                session.conversation_thread = session.conversation_thread[-100:]

            await self.persist_session(session)

            # Record activity
            await self.record_session_activity(
                session_id, "context_update", {"updates": list(context_updates.keys())}
            )

            return True

        except Exception as e:
            logger.error(f"Failed to update session context: {e}")
            return False

    async def get_session_continuity(self, user_id: str, target_platform: Platform) -> Optional[Dict[str, Any]]:
        """Get session continuity data for seamless platform transitions"""
        try:
            # Find most recent active session for user
            recent_sessions = await self.get_user_sessions(user_id)

            if not recent_sessions:
                return None

            # Prefer same platform, then most recent
            target_session = None
            for session in recent_sessions:
                if session.platform == target_platform:
                    target_session = session
                    break

            if not target_session:
                target_session = recent_sessions[0]  # Most recent

            # Build continuity data
            continuity = {
                "session_id": target_session.session_id,
                "platform": target_session.platform.value,
                "project_context": {
                    "current_project": target_session.current_project,
                    "active_character": target_session.active_character,
                    "last_generation": target_session.last_generation_id
                },
                "conversation_context": {
                    "recent_messages": target_session.conversation_thread[-5:],
                    "context_memory": target_session.context_memory
                },
                "suggestions": {
                    "continue_project": target_session.current_project is not None,
                    "use_character": target_session.active_character is not None,
                    "platform_migration_available": target_session.platform != target_platform
                }
            }

            return continuity

        except Exception as e:
            logger.error(f"Failed to get session continuity: {e}")
            return None

    async def cleanup_expired_sessions(self, user_id: Optional[str] = None):
        """Clean up expired sessions"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Mark expired sessions
            if user_id:
                cursor.execute("""
                    UPDATE anime_echo_sessions
                    SET session_state = 'expired'
                    WHERE user_id = %s AND expires_at < CURRENT_TIMESTAMP
                    AND session_state = 'active'
                """, (user_id,))
            else:
                cursor.execute("""
                    UPDATE anime_echo_sessions
                    SET session_state = 'expired'
                    WHERE expires_at < CURRENT_TIMESTAMP
                    AND session_state = 'active'
                """)

            expired_count = cursor.rowcount

            # Mark idle sessions
            idle_threshold = datetime.now() - self.idle_timeout
            cursor.execute("""
                UPDATE anime_echo_sessions
                SET session_state = 'idle'
                WHERE last_activity < %s
                AND session_state = 'active'
            """, (idle_threshold,))

            idle_count = cursor.rowcount

            conn.commit()
            cursor.close()
            conn.close()

            if expired_count > 0 or idle_count > 0:
                logger.info(f"🧹 Session cleanup: {expired_count} expired, {idle_count} idle")

            # Remove from cache
            expired_sessions = [
                sid for sid, session in self.active_sessions.items()
                if session.expires_at < datetime.now()
            ]

            for session_id in expired_sessions:
                session = self.active_sessions.pop(session_id, None)
                if session:
                    self.platform_sessions[session.platform].discard(session_id)

        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")

    async def persist_session(self, session: SessionContext):
        """Persist session to database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO anime_echo_sessions
                (session_id, user_id, platform, session_state, created_at, last_activity, expires_at,
                 current_project, project_binding_id, active_character, pending_generations,
                 last_generation_id, generation_history, conversation_thread, context_memory,
                 telegram_chat_id, browser_session_token, api_client_id, inherited_from, migration_target)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (session_id)
                DO UPDATE SET
                    session_state = EXCLUDED.session_state,
                    last_activity = EXCLUDED.last_activity,
                    expires_at = EXCLUDED.expires_at,
                    current_project = EXCLUDED.current_project,
                    project_binding_id = EXCLUDED.project_binding_id,
                    active_character = EXCLUDED.active_character,
                    pending_generations = EXCLUDED.pending_generations,
                    last_generation_id = EXCLUDED.last_generation_id,
                    generation_history = EXCLUDED.generation_history,
                    conversation_thread = EXCLUDED.conversation_thread,
                    context_memory = EXCLUDED.context_memory,
                    migration_target = EXCLUDED.migration_target
            """, (
                session.session_id, session.user_id, session.platform.value, session.state.value,
                session.created_at, session.last_activity, session.expires_at,
                session.current_project, session.project_binding_id, session.active_character,
                json.dumps(session.pending_generations), session.last_generation_id,
                json.dumps(session.generation_history), json.dumps(session.conversation_thread),
                json.dumps(session.context_memory), session.telegram_chat_id,
                session.browser_session_token, session.api_client_id,
                session.inherited_from, session.migration_target
            ))

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to persist session: {e}")

    async def load_session_from_db(self, session_id: str) -> Optional[SessionContext]:
        """Load session from database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT session_id, user_id, platform, session_state, created_at, last_activity, expires_at,
                       current_project, project_binding_id, active_character, pending_generations,
                       last_generation_id, generation_history, conversation_thread, context_memory,
                       telegram_chat_id, browser_session_token, api_client_id, inherited_from, migration_target
                FROM anime_echo_sessions
                WHERE session_id = %s
            """, (session_id,))

            row = cursor.fetchone()
            cursor.close()
            conn.close()

            if not row:
                return None

            # Parse row data
            (sid, user_id, platform, state, created_at, last_activity, expires_at,
             current_project, project_binding_id, active_character, pending_generations,
             last_generation_id, generation_history, conversation_thread, context_memory,
             telegram_chat_id, browser_session_token, api_client_id, inherited_from, migration_target) = row

            session = SessionContext(
                session_id=sid,
                user_id=user_id,
                platform=Platform(platform),
                state=SessionState(state),
                created_at=created_at,
                last_activity=last_activity,
                expires_at=expires_at,
                current_project=current_project,
                project_binding_id=project_binding_id,
                active_character=active_character,
                pending_generations=json.loads(pending_generations or "[]"),
                last_generation_id=last_generation_id,
                generation_history=json.loads(generation_history or "[]"),
                conversation_thread=json.loads(conversation_thread or "[]"),
                context_memory=json.loads(context_memory or "{}"),
                telegram_chat_id=telegram_chat_id,
                browser_session_token=browser_session_token,
                api_client_id=api_client_id,
                inherited_from=inherited_from,
                migration_target=migration_target
            )

            return session

        except Exception as e:
            logger.error(f"Failed to load session from database: {e}")
            return None

    async def inherit_session_context(self, new_session: SessionContext, inherit_from: str):
        """Inherit context from another session"""
        try:
            source_session = await self.load_session_from_db(inherit_from)
            if not source_session:
                logger.warning(f"Cannot inherit from non-existent session: {inherit_from}")
                return

            # Inherit project and character context
            new_session.current_project = source_session.current_project
            new_session.project_binding_id = source_session.project_binding_id
            new_session.active_character = source_session.active_character

            # Inherit relevant memory context
            new_session.context_memory = source_session.context_memory.copy()

            # Inherit recent conversation context (last 5 messages)
            if source_session.conversation_thread:
                new_session.conversation_thread = source_session.conversation_thread[-5:]

            logger.info(f"📥 Inherited context from session: {inherit_from}")

        except Exception as e:
            logger.error(f"Failed to inherit session context: {e}")

    async def update_session_activity(self, session_id: str):
        """Update session activity timestamp"""
        try:
            session = self.active_sessions.get(session_id)
            if session:
                session.last_activity = datetime.now()
                session.expires_at = datetime.now() + self.session_timeout

                # Update in database
                conn = psycopg2.connect(**self.db_config)
                cursor = conn.cursor()

                cursor.execute("""
                    UPDATE anime_echo_sessions
                    SET last_activity = %s, expires_at = %s
                    WHERE session_id = %s
                """, (session.last_activity, session.expires_at, session_id))

                conn.commit()
                cursor.close()
                conn.close()

        except Exception as e:
            logger.error(f"Failed to update session activity: {e}")

    async def enforce_session_limits(self, user_id: str):
        """Enforce maximum concurrent sessions per user"""
        try:
            user_sessions = await self.get_user_sessions(user_id)

            if len(user_sessions) >= self.max_sessions_per_user:
                # Mark oldest session as expired
                oldest_session = min(user_sessions, key=lambda s: s.last_activity)
                oldest_session.state = SessionState.EXPIRED

                await self.persist_session(oldest_session)

                # Remove from cache
                self.active_sessions.pop(oldest_session.session_id, None)
                for platform_set in self.platform_sessions.values():
                    platform_set.discard(oldest_session.session_id)

                logger.info(f"⚖️ Enforced session limit: expired {oldest_session.session_id}")

        except Exception as e:
            logger.error(f"Failed to enforce session limits: {e}")

    async def record_session_transition(self, from_session: str, to_session: str,
                                       from_platform: Platform, to_platform: Platform,
                                       transition_type: str):
        """Record session transition for analytics"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO anime_session_transitions
                (from_session_id, to_session_id, from_platform, to_platform, transition_type)
                VALUES (%s, %s, %s, %s, %s)
            """, (from_session, to_session, from_platform.value, to_platform.value, transition_type))

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to record session transition: {e}")

    async def record_session_activity(self, session_id: str, activity_type: str, activity_data: Dict):
        """Record session activity for analytics"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO anime_session_analytics
                (session_id, platform, activity_type, activity_data)
                VALUES (%s, %s, %s, %s)
            """, (session_id, "unknown", activity_type, json.dumps(activity_data)))

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to record session activity: {e}")

# Global session context manager
session_context_manager = SessionContextManager()

# Convenience functions
async def get_or_create_session(user_id: str, platform: str, platform_data: Optional[Dict] = None) -> SessionContext:
    """Get existing session or create new one"""
    platform_enum = Platform(platform)

    # Try to get existing session for platform
    user_sessions = await session_context_manager.get_user_sessions(user_id, platform_enum)

    if user_sessions:
        return user_sessions[0]  # Return most recent

    # Create new session
    return await session_context_manager.create_session(user_id, platform_enum, platform_data=platform_data)

async def migrate_to_platform(session_id: str, target_platform: str, platform_data: Optional[Dict] = None) -> Optional[SessionContext]:
    """Migrate session to different platform"""
    target_platform_enum = Platform(target_platform)
    return await session_context_manager.migrate_session(session_id, target_platform_enum, platform_data)

async def get_session_continuity_data(user_id: str, target_platform: str) -> Optional[Dict]:
    """Get session continuity data for platform transitions"""
    target_platform_enum = Platform(target_platform)
    return await session_context_manager.get_session_continuity(user_id, target_platform_enum)