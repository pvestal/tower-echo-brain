#!/usr/bin/env python3
"""
Echo-Telegram Bridge for Veteran Guardian Bot
Integrates Telegram bots with Echo Brain's intelligence scaling system
Provides crisis detection and Board of Directors decision making
"""

import asyncio
import aiohttp
import json
import logging
import os
import psycopg2
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from cryptography.fernet import Fernet
import jwt
import redis
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class CrisisLevel(Enum):
    """Crisis detection levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class MessageType(Enum):
    """Message type classification"""
    GENERAL = "general"
    CRISIS = "crisis"
    RESOURCE_REQUEST = "resource_request"
    CHECK_IN = "check_in"
    SUPPORT = "support"

@dataclass
class ConversationContext:
    """Context for ongoing conversations"""
    user_id: str
    conversation_id: str
    platform: str
    risk_level: CrisisLevel
    message_count: int
    last_activity: datetime
    session_data: Dict

class EchoTelegramBridge:
    """
    Bridge between Telegram bots and Echo Brain intelligence system
    Handles crisis detection, conversation persistence, and intelligent routing
    """

    def __init__(self):
        self.echo_base_url = "http://localhost:8309/api/echo"
        self.db_host = "localhost"
        self.db_user = "patrick"
        self.db_name = "tower_veteran"
        self.redis_client = redis.Redis(host='localhost', port=6379, db=1)

        # Encryption for sensitive data
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher = Fernet(self.encryption_key)

        # JWT for secure API calls
        self.jwt_secret = self._get_jwt_secret()

        # Active conversations tracking
        self.active_conversations: Dict[str, ConversationContext] = {}

        # Crisis detection thresholds
        self.crisis_keywords = [
            'suicide', 'kill myself', 'end it all', 'not worth living',
            'hopeless', 'no point', 'better off dead', 'hurt myself',
            'can\'t go on', 'give up', 'worthless', 'burden'
        ]

        # Initialize database connection
        self._init_database()

    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for sensitive data"""
        key_file = "/opt/tower-echo-brain/.encryption_key"
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)  # Secure permissions
            return key

    def _get_jwt_secret(self) -> str:
        """Get JWT secret for API authentication"""
        secret = os.getenv('JWT_SECRET', 'echo-bridge-secret-2025')
        return secret

    def _init_database(self):
        """Initialize database connection and ensure tables exist"""
        try:
            self.db_connection = psycopg2.connect(
                host=self.db_host,
                user=self.db_user,
                database=self.db_name,
                password="password"  # Should be in environment variable
            )
            self._ensure_tables_exist()
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def _ensure_tables_exist(self):
        """Ensure required tables exist for conversation persistence"""
        with self.db_connection.cursor() as cursor:
            # Enhanced conversations table for Echo integration
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS echo_conversations (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id VARCHAR(100) NOT NULL,
                    platform VARCHAR(50) NOT NULL,
                    conversation_id VARCHAR(200) NOT NULL,
                    echo_conversation_id VARCHAR(200),
                    risk_level VARCHAR(20) DEFAULT 'low',
                    session_data JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT true,
                    UNIQUE(user_id, platform, conversation_id)
                );
            """)

            # Enhanced messages table with Echo integration
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS echo_messages (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    conversation_id UUID REFERENCES echo_conversations(id),
                    message_type VARCHAR(50) NOT NULL,
                    content_encrypted TEXT NOT NULL,
                    echo_response_encrypted TEXT,
                    intelligence_level VARCHAR(20),
                    model_used VARCHAR(100),
                    crisis_level VARCHAR(20) DEFAULT 'low',
                    director_decisions JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_at TIMESTAMP
                );
            """)

            # Crisis alerts with Board of Directors integration
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS echo_crisis_alerts (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    conversation_id UUID REFERENCES echo_conversations(id),
                    message_id UUID REFERENCES echo_messages(id),
                    crisis_level VARCHAR(20) NOT NULL,
                    risk_factors JSONB NOT NULL,
                    board_decision JSONB,
                    human_notified BOOLEAN DEFAULT false,
                    resolved BOOLEAN DEFAULT false,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP
                );
            """)

            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_echo_conv_user_platform ON echo_conversations(user_id, platform);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_echo_conv_active ON echo_conversations(is_active, last_activity);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_echo_messages_conv ON echo_messages(conversation_id, created_at);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_echo_crisis_unresolved ON echo_crisis_alerts(resolved, created_at);")

            self.db_connection.commit()
            logger.info("Database tables verified/created")

    async def process_message(self, user_id: str, message: str, platform: str = "telegram") -> Tuple[str, Dict]:
        """
        Process incoming message through Echo Brain intelligence system
        Returns: (response, metadata)
        """
        try:
            # Get or create conversation context
            conversation_context = await self._get_conversation_context(user_id, platform)

            # Encrypt the message for storage
            encrypted_message = self.cipher.encrypt(message.encode()).decode()

            # Store message in database
            message_id = await self._store_message(
                conversation_context.conversation_id,
                MessageType.GENERAL.value,
                encrypted_message
            )

            # Perform crisis detection
            crisis_level = await self._detect_crisis_level(message, conversation_context)

            # Route to appropriate Echo Brain intelligence level
            intelligence_level = self._determine_intelligence_level(crisis_level, conversation_context)

            # Call Echo Brain API
            echo_response, model_used = await self._call_echo_brain(
                message,
                conversation_context,
                intelligence_level
            )

            # Encrypt and store response
            encrypted_response = self.cipher.encrypt(echo_response.encode()).decode()
            await self._update_message_response(
                message_id,
                encrypted_response,
                intelligence_level,
                model_used,
                crisis_level
            )

            # Handle crisis situations
            if crisis_level in [CrisisLevel.HIGH, CrisisLevel.CRITICAL]:
                await self._handle_crisis(
                    conversation_context.conversation_id,
                    message_id,
                    crisis_level,
                    message
                )

            # Update conversation context
            await self._update_conversation_activity(conversation_context.conversation_id)

            metadata = {
                "intelligence_level": intelligence_level,
                "model_used": model_used,
                "crisis_level": crisis_level.value,
                "message_id": str(message_id)
            }

            return echo_response, metadata

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "I'm having trouble processing your message right now. Please try again.", {}

    async def _get_conversation_context(self, user_id: str, platform: str) -> ConversationContext:
        """Get or create conversation context for user"""
        context_key = f"{user_id}:{platform}"

        if context_key in self.active_conversations:
            context = self.active_conversations[context_key]
            context.last_activity = datetime.now()
            context.message_count += 1
            return context

        # Check database for existing conversation
        with self.db_connection.cursor() as cursor:
            cursor.execute("""
                SELECT id, conversation_id, risk_level, session_data, last_activity
                FROM echo_conversations
                WHERE user_id = %s AND platform = %s AND is_active = true
                ORDER BY last_activity DESC
                LIMIT 1
            """, (user_id, platform))

            result = cursor.fetchone()

            if result:
                db_id, conv_id, risk_level, session_data, last_activity = result
                context = ConversationContext(
                    user_id=user_id,
                    conversation_id=conv_id,
                    platform=platform,
                    risk_level=CrisisLevel(risk_level),
                    message_count=1,
                    last_activity=datetime.now(),
                    session_data=session_data or {}
                )
            else:
                # Create new conversation
                conv_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO echo_conversations (user_id, platform, conversation_id, risk_level)
                    VALUES (%s, %s, %s, %s)
                """, (user_id, platform, conv_id, CrisisLevel.LOW.value))
                self.db_connection.commit()

                context = ConversationContext(
                    user_id=user_id,
                    conversation_id=conv_id,
                    platform=platform,
                    risk_level=CrisisLevel.LOW,
                    message_count=1,
                    last_activity=datetime.now(),
                    session_data={}
                )

        self.active_conversations[context_key] = context
        return context

    async def _store_message(self, conversation_id: str, message_type: str, encrypted_content: str) -> uuid.UUID:
        """Store message in database"""
        with self.db_connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO echo_messages (conversation_id, message_type, content_encrypted)
                VALUES ((SELECT id FROM echo_conversations WHERE conversation_id = %s), %s, %s)
                RETURNING id
            """, (conversation_id, message_type, encrypted_content))

            message_id = cursor.fetchone()[0]
            self.db_connection.commit()
            return message_id

    async def _detect_crisis_level(self, message: str, context: ConversationContext) -> CrisisLevel:
        """Detect crisis level in message"""
        message_lower = message.lower()

        # Critical keywords
        critical_count = sum(1 for keyword in ['suicide', 'kill myself', 'end it all']
                           if keyword in message_lower)
        if critical_count > 0:
            return CrisisLevel.CRITICAL

        # High-risk keywords
        high_risk_count = sum(1 for keyword in self.crisis_keywords
                            if keyword in message_lower)

        if high_risk_count >= 3:
            return CrisisLevel.HIGH
        elif high_risk_count >= 2:
            return CrisisLevel.MODERATE
        elif high_risk_count >= 1:
            return CrisisLevel.LOW

        return CrisisLevel.LOW

    def _determine_intelligence_level(self, crisis_level: CrisisLevel, context: ConversationContext) -> str:
        """Determine appropriate Echo Brain intelligence level"""
        if crisis_level == CrisisLevel.CRITICAL:
            return "genius"  # 70B model for critical situations
        elif crisis_level == CrisisLevel.HIGH:
            return "expert"  # 32B model for high-risk
        elif crisis_level == CrisisLevel.MODERATE:
            return "professional"  # 7B model for moderate risk
        elif context.message_count > 5:
            return "standard"  # 3B model for extended conversations
        else:
            return "quick"  # 1B model for initial responses

    async def _call_echo_brain(self, message: str, context: ConversationContext, intelligence_level: str) -> Tuple[str, str]:
        """Call Echo Brain API with appropriate intelligence level"""
        try:
            # Prepare context for Echo Brain
            echo_context = {
                "user_profile": {
                    "user_id": context.user_id,
                    "platform": context.platform,
                    "risk_level": context.risk_level.value,
                    "conversation_history": context.session_data.get("recent_topics", [])
                },
                "conversation_id": context.conversation_id,
                "crisis_detection": True,
                "veteran_support_mode": True
            }

            # Create JWT token for secure API call
            token_payload = {
                "user_id": context.user_id,
                "conversation_id": context.conversation_id,
                "platform": context.platform,
                "exp": datetime.utcnow() + timedelta(minutes=5)
            }
            jwt_token = jwt.encode(token_payload, self.jwt_secret, algorithm="HS256")

            # Call Echo Brain API
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {jwt_token}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "query": message,
                    "context": echo_context,
                    "intelligence_level": intelligence_level,
                    "user_id": context.user_id,
                    "conversation_id": context.conversation_id
                }

                async with session.post(
                    f"{self.echo_base_url}/query",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", ""), result.get("model_used", "unknown")
                    else:
                        logger.error(f"Echo Brain API error: {response.status}")
                        return "I'm having trouble accessing my intelligence systems. Please try again.", "fallback"

        except Exception as e:
            logger.error(f"Echo Brain API call failed: {e}")
            return "I'm experiencing technical difficulties. Please try again.", "error"

    async def _update_message_response(self, message_id: uuid.UUID, encrypted_response: str,
                                     intelligence_level: str, model_used: str, crisis_level: CrisisLevel):
        """Update message with Echo Brain response"""
        with self.db_connection.cursor() as cursor:
            cursor.execute("""
                UPDATE echo_messages
                SET echo_response_encrypted = %s,
                    intelligence_level = %s,
                    model_used = %s,
                    crisis_level = %s,
                    processed_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (encrypted_response, intelligence_level, model_used, crisis_level.value, message_id))
            self.db_connection.commit()

    async def _handle_crisis(self, conversation_id: str, message_id: uuid.UUID,
                           crisis_level: CrisisLevel, original_message: str):
        """Handle crisis situations with Board of Directors"""
        try:
            # Create crisis alert
            risk_factors = {
                "crisis_level": crisis_level.value,
                "message_content_indicators": self._extract_risk_indicators(original_message),
                "timestamp": datetime.now().isoformat()
            }

            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO echo_crisis_alerts (conversation_id, message_id, crisis_level, risk_factors)
                    VALUES ((SELECT id FROM echo_conversations WHERE conversation_id = %s), %s, %s, %s)
                    RETURNING id
                """, (conversation_id, message_id, crisis_level.value, json.dumps(risk_factors)))

                alert_id = cursor.fetchone()[0]
                self.db_connection.commit()

            # Call Board of Directors for decision making
            board_decision = await self._consult_board_of_directors(crisis_level, original_message)

            # Update alert with board decision
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    UPDATE echo_crisis_alerts
                    SET board_decision = %s
                    WHERE id = %s
                """, (json.dumps(board_decision), alert_id))
                self.db_connection.commit()

            # If critical, notify human oversight
            if crisis_level == CrisisLevel.CRITICAL:
                await self._notify_human_oversight(conversation_id, alert_id, risk_factors)

            logger.info(f"Crisis handled: {crisis_level.value} for conversation {conversation_id}")

        except Exception as e:
            logger.error(f"Crisis handling failed: {e}")

    async def _consult_board_of_directors(self, crisis_level: CrisisLevel, message: str) -> Dict:
        """Consult Board of Directors for crisis decision making"""
        try:
            # Call Board API (assuming it's available)
            async with aiohttp.ClientSession() as session:
                payload = {
                    "crisis_level": crisis_level.value,
                    "message_content": message,
                    "decision_type": "crisis_intervention",
                    "urgency": "high" if crisis_level in [CrisisLevel.HIGH, CrisisLevel.CRITICAL] else "medium"
                }

                async with session.post(
                    "http://localhost:8356/api/board/crisis-decision",
                    json=payload
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"decision": "escalate_to_human", "confidence": 0.5, "reason": "board_api_unavailable"}

        except Exception as e:
            logger.error(f"Board consultation failed: {e}")
            return {"decision": "escalate_to_human", "confidence": 0.0, "reason": f"error: {str(e)}"}

    def _extract_risk_indicators(self, message: str) -> List[str]:
        """Extract risk indicators from message"""
        indicators = []
        message_lower = message.lower()

        for keyword in self.crisis_keywords:
            if keyword in message_lower:
                indicators.append(keyword)

        return indicators

    async def _notify_human_oversight(self, conversation_id: str, alert_id: uuid.UUID, risk_factors: Dict):
        """Notify human oversight for critical situations"""
        try:
            # Send notification to Tower dashboard
            notification_payload = {
                "type": "veteran_crisis_alert",
                "severity": "critical",
                "conversation_id": conversation_id,
                "alert_id": str(alert_id),
                "risk_factors": risk_factors,
                "timestamp": datetime.now().isoformat()
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:8080/api/alerts/veteran-guardian",
                    json=notification_payload
                ) as response:
                    if response.status == 200:
                        logger.info(f"Human oversight notified for alert {alert_id}")
                    else:
                        logger.error(f"Failed to notify human oversight: {response.status}")

        except Exception as e:
            logger.error(f"Human oversight notification failed: {e}")

    async def _update_conversation_activity(self, conversation_id: str):
        """Update conversation last activity timestamp"""
        with self.db_connection.cursor() as cursor:
            cursor.execute("""
                UPDATE echo_conversations
                SET last_activity = CURRENT_TIMESTAMP
                WHERE conversation_id = %s
            """, (conversation_id,))
            self.db_connection.commit()

    async def get_conversation_history(self, user_id: str, platform: str, limit: int = 10) -> List[Dict]:
        """Get conversation history for user"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    SELECT m.id, m.message_type, m.content_encrypted, m.echo_response_encrypted,
                           m.intelligence_level, m.model_used, m.crisis_level, m.created_at
                    FROM echo_messages m
                    JOIN echo_conversations c ON m.conversation_id = c.id
                    WHERE c.user_id = %s AND c.platform = %s
                    ORDER BY m.created_at DESC
                    LIMIT %s
                """, (user_id, platform, limit))

                messages = []
                for row in cursor.fetchall():
                    message_id, msg_type, encrypted_content, encrypted_response, \
                    intel_level, model_used, crisis_level, created_at = row

                    # Decrypt content
                    content = self.cipher.decrypt(encrypted_content.encode()).decode()
                    response = None
                    if encrypted_response:
                        response = self.cipher.decrypt(encrypted_response.encode()).decode()

                    messages.append({
                        "id": str(message_id),
                        "type": msg_type,
                        "content": content,
                        "response": response,
                        "intelligence_level": intel_level,
                        "model_used": model_used,
                        "crisis_level": crisis_level,
                        "timestamp": created_at.isoformat()
                    })

                return messages

        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []

    async def cleanup_old_conversations(self, days: int = 30):
        """Clean up old inactive conversations"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            with self.db_connection.cursor() as cursor:
                # Mark old conversations as inactive
                cursor.execute("""
                    UPDATE echo_conversations
                    SET is_active = false
                    WHERE last_activity < %s AND is_active = true
                """, (cutoff_date,))

                rows_affected = cursor.rowcount
                self.db_connection.commit()

                logger.info(f"Cleaned up {rows_affected} old conversations")

                # Clean up active conversations cache
                current_time = datetime.now()
                expired_keys = [
                    key for key, context in self.active_conversations.items()
                    if (current_time - context.last_activity).days > 1
                ]

                for key in expired_keys:
                    del self.active_conversations[key]

                logger.info(f"Cleaned up {len(expired_keys)} cached conversations")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# Example usage and testing
if __name__ == "__main__":
    async def test_bridge():
        """Test the bridge functionality"""
        bridge = EchoTelegramBridge()

        # Test normal message
        response, metadata = await bridge.process_message(
            "test_user_123",
            "Hello, I'm feeling a bit down today but managing okay.",
            "telegram"
        )
        print(f"Normal message response: {response}")
        print(f"Metadata: {metadata}")

        # Test crisis message
        response, metadata = await bridge.process_message(
            "test_user_123",
            "I can't take this anymore, I just want it all to end.",
            "telegram"
        )
        print(f"Crisis message response: {response}")
        print(f"Metadata: {metadata}")

        # Get conversation history
        history = await bridge.get_conversation_history("test_user_123", "telegram")
        print(f"Conversation history: {len(history)} messages")

    # Run test
    asyncio.run(test_bridge())