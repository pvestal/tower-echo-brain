#!/usr/bin/env python3
"""
AI Assist Memory Consolidation System
Implements long-term memory storage and retrieval for continuous learning
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Types of memories Echo can store"""
    CONVERSATION = "conversation"
    LEARNING = "learning"
    PATTERN = "pattern"
    SKILL = "skill"
    PREFERENCE = "preference"
    CONTEXT = "context"
    ERROR = "error"
    SUCCESS = "success"

@dataclass
class Memory:
    """Individual memory unit"""
    id: str
    type: MemoryType
    content: Dict[str, Any]
    embedding: Optional[List[float]]
    timestamp: datetime
    importance: float  # 0-1 scale
    access_count: int
    last_accessed: datetime
    associations: List[str]  # Related memory IDs
    decay_rate: float  # How quickly memory fades
    reinforcement_count: int  # Times reinforced

class EchoMemoryConsolidation:
    """Manages Echo's long-term memory consolidation and retrieval"""

    def __init__(self, db_config: Optional[Dict] = None):
        self.db_config = db_config or {
            'dbname': 'echo_memory',
            'user': 'echo',
            'host': 'localhost',
            'port': 5432
        }

        # Initialize sentence transformer for embeddings
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Memory configuration
        self.config = {
            'consolidation_interval': 3600,  # Consolidate every hour
            'importance_threshold': 0.3,  # Min importance to store
            'max_short_term': 1000,  # Max short-term memories
            'embedding_dim': 384,  # Dimension of sentence embeddings
            'decay_base': 0.95,  # Base decay rate
            'reinforcement_boost': 0.1  # Importance boost per reinforcement
        }

        # Short-term memory buffer
        self.short_term_buffer: List[Memory] = []

        # Memory index for fast retrieval
        self.memory_index: Dict[str, Memory] = {}

        # Pattern detection
        self.patterns: Dict[str, Dict[str, Any]] = {}

        # Initialize database
        self.initialize_database()

    def initialize_database(self):
        """Create memory database tables if they don't exist"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            # Create memories table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id VARCHAR(64) PRIMARY KEY,
                    type VARCHAR(32) NOT NULL,
                    content JSONB NOT NULL,
                    embedding FLOAT8[],
                    timestamp TIMESTAMP NOT NULL,
                    importance FLOAT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP NOT NULL,
                    associations TEXT[],
                    decay_rate FLOAT DEFAULT 0.95,
                    reinforcement_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create patterns table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id VARCHAR(64) PRIMARY KEY,
                    pattern_type VARCHAR(32) NOT NULL,
                    pattern_data JSONB NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    last_seen TIMESTAMP NOT NULL,
                    confidence FLOAT DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create memory associations table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memory_associations (
                    memory_id VARCHAR(64) REFERENCES memories(id),
                    associated_id VARCHAR(64) REFERENCES memories(id),
                    strength FLOAT DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (memory_id, associated_id)
                )
            """)

            # Create indexes for performance
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC)")

            conn.commit()
            conn.close()
            logger.info("Memory database initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            # Create in-memory fallback
            self.use_in_memory = True

    async def store_memory(self, content: Dict[str, Any],
                          memory_type: MemoryType,
                          importance: float = 0.5) -> str:
        """Store a new memory"""
        # Generate unique ID
        memory_id = hashlib.sha256(
            f"{content}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        # Create embedding
        text = self._extract_text(content)
        embedding = self.encoder.encode(text).tolist()

        # Create memory object
        memory = Memory(
            id=memory_id,
            type=memory_type,
            content=content,
            embedding=embedding,
            timestamp=datetime.now(),
            importance=importance,
            access_count=0,
            last_accessed=datetime.now(),
            associations=[],
            decay_rate=self.config['decay_base'],
            reinforcement_count=0
        )

        # Add to short-term buffer
        self.short_term_buffer.append(memory)

        # Check if consolidation needed
        if len(self.short_term_buffer) >= self.config['max_short_term']:
            await self.consolidate_memories()

        # Detect patterns
        await self.detect_patterns(memory)

        return memory_id

    async def recall_memory(self, query: str,
                          top_k: int = 5,
                          memory_types: Optional[List[MemoryType]] = None) -> List[Memory]:
        """Recall relevant memories based on query"""
        # Create query embedding
        query_embedding = self.encoder.encode(query)

        # Search in database
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Build type filter
            type_filter = ""
            if memory_types:
                types = "','".join([mt.value for mt in memory_types])
                type_filter = f"AND type IN ('{types}')"

            # Retrieve memories with similarity search
            # Note: In production, use pgvector for efficient similarity search
            cur.execute(f"""
                SELECT *,
                       1 - (embedding <-> %s::vector) as similarity
                FROM memories
                WHERE importance > %s {type_filter}
                ORDER BY similarity DESC, importance DESC
                LIMIT %s
            """, (query_embedding.tolist(), self.config['importance_threshold'], top_k))

            memories = []
            for row in cur.fetchall():
                memory = self._row_to_memory(row)
                # Update access count
                self._update_access(memory.id)
                memories.append(memory)

            conn.close()
            return memories

        except Exception as e:
            logger.error(f"Memory recall failed: {e}")
            # Fallback to buffer search
            return self._search_buffer(query, top_k)

    async def consolidate_memories(self):
        """Consolidate short-term memories to long-term storage"""
        logger.info(f"Consolidating {len(self.short_term_buffer)} memories")

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            for memory in self.short_term_buffer:
                # Apply decay
                memory.importance *= memory.decay_rate

                # Skip if below threshold
                if memory.importance < self.config['importance_threshold']:
                    continue

                # Find associations
                associations = await self.find_associations(memory)
                memory.associations = associations

                # Store in database
                cur.execute("""
                    INSERT INTO memories
                    (id, type, content, embedding, timestamp, importance,
                     access_count, last_accessed, associations, decay_rate,
                     reinforcement_count)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        importance = EXCLUDED.importance,
                        access_count = memories.access_count + 1,
                        last_accessed = EXCLUDED.last_accessed,
                        reinforcement_count = memories.reinforcement_count + 1
                """, (
                    memory.id, memory.type.value, json.dumps(memory.content),
                    memory.embedding, memory.timestamp, memory.importance,
                    memory.access_count, memory.last_accessed,
                    memory.associations, memory.decay_rate,
                    memory.reinforcement_count
                ))

                # Store associations
                for assoc_id in associations:
                    cur.execute("""
                        INSERT INTO memory_associations (memory_id, associated_id, strength)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (memory_id, associated_id) DO UPDATE SET
                            strength = memory_associations.strength * 1.1
                    """, (memory.id, assoc_id, 0.5))

            conn.commit()
            conn.close()

            # Clear buffer
            self.short_term_buffer = []
            logger.info("Memory consolidation complete")

        except Exception as e:
            logger.error(f"Consolidation failed: {e}")

    async def find_associations(self, memory: Memory,
                               threshold: float = 0.7) -> List[str]:
        """Find memories associated with the given memory"""
        associations = []

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            # Find similar memories using embedding similarity
            cur.execute("""
                SELECT id, 1 - (embedding <-> %s::vector) as similarity
                FROM memories
                WHERE id != %s
                AND 1 - (embedding <-> %s::vector) > %s
                ORDER BY similarity DESC
                LIMIT 10
            """, (memory.embedding, memory.id, memory.embedding, threshold))

            for row in cur.fetchall():
                associations.append(row[0])

            conn.close()

        except Exception as e:
            logger.error(f"Association finding failed: {e}")

        return associations

    async def detect_patterns(self, memory: Memory):
        """Detect patterns in memories"""
        # Extract key features
        features = self._extract_features(memory.content)

        for feature_key, feature_value in features.items():
            pattern_id = hashlib.sha256(
                f"{feature_key}:{feature_value}".encode()
            ).hexdigest()[:16]

            if pattern_id in self.patterns:
                # Update existing pattern
                self.patterns[pattern_id]['frequency'] += 1
                self.patterns[pattern_id]['last_seen'] = datetime.now()
                self.patterns[pattern_id]['confidence'] = min(
                    0.95,
                    self.patterns[pattern_id]['confidence'] * 1.05
                )
            else:
                # Create new pattern
                self.patterns[pattern_id] = {
                    'type': feature_key,
                    'value': feature_value,
                    'frequency': 1,
                    'last_seen': datetime.now(),
                    'confidence': 0.5,
                    'memories': [memory.id]
                }

            # Store significant patterns
            if self.patterns[pattern_id]['frequency'] >= 3:
                await self._store_pattern(pattern_id, self.patterns[pattern_id])

    async def reinforce_memory(self, memory_id: str, boost: float = 0.1):
        """Reinforce a memory to prevent decay"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            cur.execute("""
                UPDATE memories
                SET importance = LEAST(1.0, importance + %s),
                    reinforcement_count = reinforcement_count + 1,
                    last_accessed = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (boost, memory_id))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Memory reinforcement failed: {e}")

    async def forget_memories(self, threshold: float = 0.1):
        """Remove memories below importance threshold"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            # Apply decay to all memories
            cur.execute("""
                UPDATE memories
                SET importance = importance * decay_rate
                WHERE last_accessed < CURRENT_TIMESTAMP - INTERVAL '7 days'
            """)

            # Delete unimportant memories
            cur.execute("""
                DELETE FROM memories
                WHERE importance < %s
                AND reinforcement_count < 3
            """, (threshold,))

            affected = cur.rowcount
            conn.commit()
            conn.close()

            logger.info(f"Forgotten {affected} memories below threshold")

        except Exception as e:
            logger.error(f"Memory forgetting failed: {e}")

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            stats = {}

            # Total memories
            cur.execute("SELECT COUNT(*) as total FROM memories")
            stats['total_memories'] = cur.fetchone()['total']

            # Memory types distribution
            cur.execute("""
                SELECT type, COUNT(*) as count
                FROM memories
                GROUP BY type
            """)
            stats['type_distribution'] = {row['type']: row['count']
                                         for row in cur.fetchall()}

            # Average importance
            cur.execute("SELECT AVG(importance) as avg_importance FROM memories")
            stats['average_importance'] = cur.fetchone()['avg_importance']

            # Most accessed memories
            cur.execute("""
                SELECT id, content->>'summary' as summary, access_count
                FROM memories
                ORDER BY access_count DESC
                LIMIT 5
            """)
            stats['most_accessed'] = cur.fetchall()

            # Pattern statistics
            cur.execute("SELECT COUNT(*) as total FROM patterns")
            stats['total_patterns'] = cur.fetchone()['total']

            conn.close()
            return stats

        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {}

    def _extract_text(self, content: Dict[str, Any]) -> str:
        """Extract text from content for embedding"""
        text_parts = []
        for key, value in content.items():
            if isinstance(value, str):
                text_parts.append(f"{key}: {value}")
            elif isinstance(value, (list, dict)):
                text_parts.append(f"{key}: {json.dumps(value)[:100]}")
        return " ".join(text_parts)

    def _extract_features(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key features for pattern detection"""
        features = {}

        # Extract common patterns
        if 'user' in content:
            features['user'] = content['user']
        if 'action' in content:
            features['action'] = content['action']
        if 'intent' in content:
            features['intent'] = content['intent']
        if 'emotion' in content:
            features['emotion'] = content['emotion']
        if 'topic' in content:
            features['topic'] = content['topic']

        return features

    async def _store_pattern(self, pattern_id: str, pattern_data: Dict[str, Any]):
        """Store detected pattern in database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO patterns (id, pattern_type, pattern_data, frequency,
                                    last_seen, confidence)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    frequency = EXCLUDED.frequency,
                    last_seen = EXCLUDED.last_seen,
                    confidence = EXCLUDED.confidence
            """, (
                pattern_id, pattern_data['type'], json.dumps(pattern_data),
                pattern_data['frequency'], pattern_data['last_seen'],
                pattern_data['confidence']
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Pattern storage failed: {e}")

    def _row_to_memory(self, row: Dict) -> Memory:
        """Convert database row to Memory object"""
        return Memory(
            id=row['id'],
            type=MemoryType(row['type']),
            content=row['content'],
            embedding=row['embedding'],
            timestamp=row['timestamp'],
            importance=row['importance'],
            access_count=row['access_count'],
            last_accessed=row['last_accessed'],
            associations=row['associations'] or [],
            decay_rate=row['decay_rate'],
            reinforcement_count=row['reinforcement_count']
        )

    def _update_access(self, memory_id: str):
        """Update memory access count"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            cur.execute("""
                UPDATE memories
                SET access_count = access_count + 1,
                    last_accessed = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (memory_id,))
            conn.commit()
            conn.close()
        except:
            pass

    def _search_buffer(self, query: str, top_k: int) -> List[Memory]:
        """Fallback search in memory buffer"""
        query_embedding = self.encoder.encode(query)

        results = []
        for memory in self.short_term_buffer:
            if memory.embedding:
                similarity = np.dot(query_embedding, memory.embedding)
                results.append((similarity, memory))

        results.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in results[:top_k]]


# Integration with AI Assist
async def integrate_with_echo():
    """Integrate memory consolidation with AI Assist"""
    memory_system = EchoMemoryConsolidation()

    # Example: Store conversation memory
    conversation_memory = {
        'user': 'Patrick',
        'message': 'Help me generate an anime character',
        'response': 'Created cyberpunk character Kai',
        'context': 'anime_generation',
        'success': True
    }

    memory_id = await memory_system.store_memory(
        conversation_memory,
        MemoryType.CONVERSATION,
        importance=0.8
    )

    logger.info(f"Stored memory: {memory_id}")

    # Recall related memories
    memories = await memory_system.recall_memory(
        "anime character generation",
        top_k=3
    )

    for memory in memories:
        logger.info(f"Recalled: {memory.content.get('summary', memory.id)}")

    # Get statistics
    stats = await memory_system.get_memory_stats()
    logger.info(f"Memory stats: {stats}")

if __name__ == "__main__":
    asyncio.run(integrate_with_echo())