"""
Persistent Memory System for Echo Brain
Maintains context and learned knowledge across restarts
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import asyncpg
from dataclasses import dataclass, asdict
import pickle
import base64

logger = logging.getLogger(__name__)

@dataclass
class MemoryEntry:
    """Single memory entry with metadata"""
    id: str
    category: str
    content: Any
    importance: float
    timestamp: datetime
    access_count: int
    last_accessed: datetime
    metadata: Dict[str, Any]

class PersistentMemorySystem:
    """Maintains Echo Brain's memory across restarts"""

    def __init__(
        self,
        db_config: Dict[str, str],
        memory_path: str = "/opt/tower-echo-brain/data/persistent_memory/"
    ):
        self.db_config = db_config
        self.memory_path = Path(memory_path)
        self.memory_path.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.memory_cache = {}

    async def connect(self):
        """Establish database connection"""
        try:
            self.conn = await asyncpg.connect(
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 5432),
                user=self.db_config.get('user', 'patrick'),
                password=self.db_config.get('password', 'RP78eIrW7cI2jYvL5akt1yurE'),
                database=self.db_config.get('database', 'echo_brain')
            )
            logger.info("Connected to persistent memory database")

            # Create table if not exists
            await self._create_memory_table()

        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    async def _create_memory_table(self):
        """Create memory table if it doesn't exist"""
        await self.conn.execute('''
            CREATE TABLE IF NOT EXISTS persistent_memories (
                id VARCHAR(255) PRIMARY KEY,
                category VARCHAR(100) NOT NULL,
                content TEXT NOT NULL,
                importance FLOAT DEFAULT 0.5,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create indexes for fast queries
        await self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_memory_category ON persistent_memories(category);
        ''')
        await self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_memory_importance ON persistent_memories(importance DESC);
        ''')
        await self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_memory_accessed ON persistent_memories(last_accessed DESC);
        ''')

    async def store_memory(
        self,
        memory_id: str,
        category: str,
        content: Any,
        importance: float = 0.5,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Store a memory entry"""
        try:
            # Serialize complex content
            if not isinstance(content, (str, int, float, bool)):
                content = base64.b64encode(pickle.dumps(content)).decode('utf-8')

            # Store in database
            await self.conn.execute('''
                INSERT INTO persistent_memories
                (id, category, content, importance, metadata, timestamp, last_accessed, access_count)
                VALUES ($1, $2, $3, $4, $5, $6, $6, 0)
                ON CONFLICT (id) DO UPDATE SET
                    content = $3,
                    importance = $4,
                    metadata = $5,
                    updated_at = $6,
                    access_count = persistent_memories.access_count + 1
            ''', memory_id, category, str(content), importance,
                json.dumps(metadata or {}), datetime.now())

            # Update cache
            self.memory_cache[memory_id] = {
                'category': category,
                'content': content,
                'importance': importance,
                'metadata': metadata,
                'timestamp': datetime.now()
            }

            # Also save to disk for backup
            memory_file = self.memory_path / f"{category}_{memory_id}.json"
            with open(memory_file, 'w') as f:
                json.dump({
                    'id': memory_id,
                    'category': category,
                    'content': str(content),
                    'importance': importance,
                    'metadata': metadata,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)

            logger.info(f"Stored memory: {memory_id} in category: {category}")
            return True

        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return False

    async def retrieve_memory(
        self,
        memory_id: str = None,
        category: str = None,
        min_importance: float = 0.0,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve memories based on criteria"""
        try:
            query = "SELECT * FROM persistent_memories WHERE 1=1"
            params = []
            param_count = 0

            if memory_id:
                param_count += 1
                query += f" AND id = ${param_count}"
                params.append(memory_id)

            if category:
                param_count += 1
                query += f" AND category = ${param_count}"
                params.append(category)

            if min_importance > 0:
                param_count += 1
                query += f" AND importance >= ${param_count}"
                params.append(min_importance)

            query += " ORDER BY importance DESC, last_accessed DESC"
            query += f" LIMIT {limit}"

            rows = await self.conn.fetch(query, *params)

            memories = []
            for row in rows:
                # Update access stats
                await self.conn.execute('''
                    UPDATE persistent_memories
                    SET access_count = access_count + 1, last_accessed = $1
                    WHERE id = $2
                ''', datetime.now(), row['id'])

                memory = dict(row)
                # Try to deserialize content if it's base64 encoded
                try:
                    memory['content'] = pickle.loads(base64.b64decode(memory['content']))
                except:
                    pass  # Keep as string if not serialized

                memories.append(memory)

            return memories

        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []

    async def consolidate_memories(self):
        """Consolidate and optimize memory storage"""
        try:
            # Remove old, unimportant memories
            cutoff_date = datetime.now() - timedelta(days=30)
            await self.conn.execute('''
                DELETE FROM persistent_memories
                WHERE importance < 0.3
                AND last_accessed < $1
                AND access_count < 5
            ''', cutoff_date)

            # Boost importance of frequently accessed memories
            await self.conn.execute('''
                UPDATE persistent_memories
                SET importance = LEAST(importance * 1.1, 1.0)
                WHERE access_count > 10
            ''')

            # Vacuum the database for optimization
            await self.conn.execute('VACUUM ANALYZE persistent_memories')

            logger.info("Memory consolidation completed")

        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")

    async def backup_to_disk(self):
        """Backup all memories to disk"""
        try:
            memories = await self.conn.fetch('SELECT * FROM persistent_memories')
            backup_file = self.memory_path / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            backup_data = []
            for memory in memories:
                backup_data.append({
                    'id': memory['id'],
                    'category': memory['category'],
                    'content': memory['content'],
                    'importance': memory['importance'],
                    'metadata': json.loads(memory['metadata']) if memory['metadata'] else {},
                    'timestamp': memory['timestamp'].isoformat(),
                    'access_count': memory['access_count'],
                    'last_accessed': memory['last_accessed'].isoformat()
                })

            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2)

            logger.info(f"Backed up {len(backup_data)} memories to {backup_file}")
            return backup_file

        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return None

    async def restore_from_backup(self, backup_file: Path):
        """Restore memories from backup"""
        try:
            with open(backup_file, 'r') as f:
                backup_data = json.load(f)

            for memory in backup_data:
                await self.store_memory(
                    memory['id'],
                    memory['category'],
                    memory['content'],
                    memory['importance'],
                    memory.get('metadata', {})
                )

            logger.info(f"Restored {len(backup_data)} memories from backup")
            return True

        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False

    async def get_startup_context(self) -> Dict[str, Any]:
        """Get essential context for system startup"""
        try:
            # Retrieve high-importance memories
            critical_memories = await self.retrieve_memory(
                min_importance=0.7,
                limit=50
            )

            # Get recent memories
            recent = await self.conn.fetch('''
                SELECT * FROM persistent_memories
                ORDER BY last_accessed DESC
                LIMIT 20
            ''')

            # Get system state memories
            system_state = await self.retrieve_memory(
                category='system_state',
                limit=10
            )

            # Get learned patterns
            patterns = await self.retrieve_memory(
                category='learned_patterns',
                limit=20
            )

            context = {
                'critical_memories': critical_memories,
                'recent_context': [dict(r) for r in recent],
                'system_state': system_state,
                'learned_patterns': patterns,
                'memory_stats': {
                    'total_memories': await self.conn.fetchval('SELECT COUNT(*) FROM persistent_memories'),
                    'categories': await self.conn.fetch('SELECT DISTINCT category FROM persistent_memories'),
                    'avg_importance': await self.conn.fetchval('SELECT AVG(importance) FROM persistent_memories')
                }
            }

            logger.info(f"Loaded startup context with {len(critical_memories)} critical memories")
            return context

        except Exception as e:
            logger.error(f"Failed to get startup context: {e}")
            return {}

    async def close(self):
        """Close database connection"""
        if self.conn:
            await self.conn.close()
            logger.info("Closed persistent memory connection")


async def test_persistent_memory():
    """Test persistent memory system"""

    print("=" * 60)
    print("PERSISTENT MEMORY SYSTEM TEST")
    print("=" * 60)

    # Initialize memory system
    memory = PersistentMemorySystem({
        'host': 'localhost',
        'user': 'patrick',
        'password': 'RP78eIrW7cI2jYvL5akt1yurE',
        'database': 'echo_brain'
    })

    await memory.connect()

    # Store various types of memories
    print("\nStoring memories...")

    # Store system configuration
    await memory.store_memory(
        "config_001",
        "system_state",
        {"model": "qwen2.5-coder:32b", "temperature": 0.7},
        importance=0.9,
        metadata={"type": "configuration"}
    )

    # Store learned pattern
    await memory.store_memory(
        "pattern_001",
        "learned_patterns",
        "User prefers concise responses with code examples",
        importance=0.8,
        metadata={"confidence": 0.95}
    )

    # Store task result
    await memory.store_memory(
        "task_001",
        "task_results",
        {"task": "ComfyUI integration", "status": "completed", "output": "/opt/tower-echo-brain/data/outputs/echo_brain_comfyui_00001_.png"},
        importance=0.6,
        metadata={"execution_time": 8.0}
    )

    # Store conversation context
    await memory.store_memory(
        "context_001",
        "conversation",
        "Working on achieving 100% autonomy for Echo Brain system",
        importance=0.7,
        metadata={"session": "2025-12-30"}
    )

    print("✅ Stored 4 different types of memories")

    # Retrieve memories
    print("\nRetrieving high-importance memories...")
    important = await memory.retrieve_memory(min_importance=0.7, limit=5)
    for mem in important:
        print(f"  - [{mem['category']}] {mem['id']}: importance={mem['importance']:.1f}")

    # Get startup context
    print("\nGetting startup context...")
    context = await memory.get_startup_context()
    print(f"  Critical memories: {len(context['critical_memories'])}")
    print(f"  Recent context: {len(context['recent_context'])}")
    print(f"  Total memories: {context['memory_stats']['total_memories']}")

    # Backup memories
    print("\nBacking up memories to disk...")
    backup_file = await memory.backup_to_disk()
    if backup_file:
        print(f"✅ Backup saved to: {backup_file}")

    # Test memory consolidation
    print("\nConsolidating memories...")
    await memory.consolidate_memories()
    print("✅ Memory consolidation complete")

    await memory.close()

    print("\n" + "=" * 60)
    return True


if __name__ == "__main__":
    asyncio.run(test_persistent_memory())