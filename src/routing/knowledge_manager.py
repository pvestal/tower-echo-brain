#!/usr/bin/env python3
"""
Knowledge Manager for AI Assist Board of Directors
Manages knowledge base integration, versioning, analytics, and reporting
"""

import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import psycopg2
import psycopg2.extras
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import redis
import sqlite3
from pathlib import Path
import hashlib
import pickle
import threading
from collections import defaultdict, deque
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class KnowledgeType(Enum):
    DECISION_PATTERN = "decision_pattern"
    DIRECTOR_BEHAVIOR = "director_behavior"
    USER_PREFERENCE = "user_preference"
    SYSTEM_METRIC = "system_metric"
    ERROR_PATTERN = "error_pattern"
    PERFORMANCE_DATA = "performance_data"
    SECURITY_EVENT = "security_event"
    LEARNING_INSIGHT = "learning_insight"

class VersionAction(Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    ROLLBACK = "rollback"

@dataclass
class KnowledgeItem:
    """Individual knowledge base item"""
    item_id: str
    knowledge_type: KnowledgeType
    title: str
    content: Dict[str, Any]
    tags: List[str]
    created_at: datetime
    updated_at: datetime
    version: int
    created_by: str
    confidence: float  # 0.0 to 1.0
    relevance_score: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = None

@dataclass
class KnowledgeVersion:
    """Knowledge base version record"""
    version_id: str
    version_number: int
    action: VersionAction
    description: str
    created_at: datetime
    created_by: str
    affected_items: List[str]
    snapshot_hash: str
    rollback_data: Optional[bytes] = None

@dataclass
class KnowledgeQuery:
    """Knowledge base query specification"""
    query_text: str
    knowledge_types: List[KnowledgeType] = None
    tags: List[str] = None
    created_after: datetime = None
    created_before: datetime = None
    min_confidence: float = 0.0
    max_results: int = 100
    include_metadata: bool = True

@dataclass
class KnowledgeSearchResult:
    """Search result from knowledge base"""
    items: List[KnowledgeItem]
    total_count: int
    query_time: float
    relevance_scores: List[float]
    suggestions: List[str] = None

@dataclass
class KnowledgeAnalytics:
    """Analytics data for knowledge base"""
    total_items: int
    items_by_type: Dict[str, int]
    items_by_tag: Dict[str, int]
    avg_confidence: float
    recent_activity: List[Dict[str, Any]]
    version_history_size: int
    cache_hit_rate: float
    query_performance: Dict[str, float]

class KnowledgeManager:
    """
    Comprehensive knowledge management system for AI Assist Board of Directors
    Handles knowledge storage, versioning, caching, and analytics
    """

    def __init__(self, db_config: Dict[str, str],
                 redis_config: Dict[str, Any] = None,
                 kb_file_path: str = None):
        """
        Initialize KnowledgeManager

        Args:
            db_config: PostgreSQL connection parameters
            redis_config: Redis connection parameters (optional)
            kb_file_path: Path to SQLite knowledge base file (optional)
        """
        # Override with connection pool config to ensure consistency
        from src.db.connection_pool import get_database_config
        self.db_config = get_database_config()
        self.redis_config = redis_config or {}
        self.kb_file_path = kb_file_path or "/tmp/echo_kb.sqlite"

        # Caching and performance
        self.redis_client = None
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.query_times = deque(maxlen=1000)

        # Knowledge processing
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.knowledge_vectors = {}
        self.similarity_cache = {}

        # Thread safety
        self.lock = threading.RLock()

        # Version control
        self.current_version = 1
        self.version_snapshots = {}

        self._initialize_storage()
        self._initialize_cache()
        self._load_existing_knowledge()

    def _initialize_storage(self):
        """Initialize storage systems"""
        try:
            # Initialize PostgreSQL using connection pool
            from src.db.connection_pool import SyncConnection
            with SyncConnection() as conn:
                conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
                cursor = conn.cursor()

                # Knowledge items table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_items (
                    item_id VARCHAR(255) PRIMARY KEY,
                    knowledge_type VARCHAR(50) NOT NULL,
                    title TEXT NOT NULL,
                    content JSONB NOT NULL,
                    tags TEXT[],
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    version INTEGER DEFAULT 1,
                    created_by VARCHAR(255),
                    confidence FLOAT DEFAULT 1.0,
                    relevance_score FLOAT DEFAULT 1.0,
                    metadata JSONB
                );
            """)

            # Knowledge versions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_versions (
                    version_id VARCHAR(255) PRIMARY KEY,
                    version_number INTEGER NOT NULL,
                    action VARCHAR(50) NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    created_by VARCHAR(255),
                    affected_items TEXT[],
                    snapshot_hash VARCHAR(255),
                    rollback_data BYTEA
                );
            """)

            # Knowledge analytics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_analytics (
                    analytics_id VARCHAR(255) PRIMARY KEY,
                    date DATE DEFAULT CURRENT_DATE,
                    total_items INTEGER,
                    items_by_type JSONB,
                    items_by_tag JSONB,
                    avg_confidence FLOAT,
                    query_count INTEGER,
                    cache_hit_rate FLOAT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

            # Knowledge relationships table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_relationships (
                    relationship_id VARCHAR(255) PRIMARY KEY,
                    source_item_id VARCHAR(255) REFERENCES knowledge_items(item_id),
                    target_item_id VARCHAR(255) REFERENCES knowledge_items(item_id),
                    relationship_type VARCHAR(50),
                    strength FLOAT DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT NOW(),
                    metadata JSONB
                );
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_items_type ON knowledge_items(knowledge_type);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_items_tags ON knowledge_items USING GIN(tags);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_items_created_at ON knowledge_items(created_at);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_versions_number ON knowledge_versions(version_number);")

            # Full-text search index
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_items_content_search
                ON knowledge_items USING GIN(to_tsvector('english', title || ' ' || content::text));
            """)

            conn.close()

            # Initialize SQLite for local knowledge base
            if self.kb_file_path:
                Path(self.kb_file_path).parent.mkdir(parents=True, exist_ok=True)
                sqlite_conn = sqlite3.connect(self.kb_file_path)
                cursor = sqlite_conn.cursor()

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS local_knowledge (
                        item_id TEXT PRIMARY KEY,
                        content TEXT,
                        vector_data BLOB,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                sqlite_conn.commit()
                sqlite_conn.close()

            logger.info("Knowledge manager storage initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize storage: {e}")
            raise

    def _initialize_cache(self):
        """Initialize Redis cache"""
        try:
            if self.redis_config:
                import redis
                self.redis_client = redis.Redis(**self.redis_config)
                self.redis_client.ping()
                logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis cache: {e}")
            self.redis_client = None

    def _load_existing_knowledge(self):
        """Load existing knowledge for processing"""
        try:
            # Get current version
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COALESCE(MAX(version_number), 0) FROM knowledge_versions
            """)
            self.current_version = cursor.fetchone()[0] + 1

            # Load recent items for vector processing
            cursor.execute("""
                SELECT item_id, title, content FROM knowledge_items
                ORDER BY updated_at DESC
                LIMIT 1000
            """)

            items = cursor.fetchall()
            conn.close()

            if items:
                texts = []
                item_ids = []

                for item_id, title, content in items:
                    text = f"{title} {json.dumps(content)}"
                    texts.append(text)
                    item_ids.append(item_id)

                # Build TF-IDF vectors
                if len(texts) > 1:
                    vectors = self.vectorizer.fit_transform(texts)
                    for i, item_id in enumerate(item_ids):
                        self.knowledge_vectors[item_id] = vectors[i]

            logger.info(f"Loaded {len(items)} knowledge items for processing")

        except Exception as e:
            logger.error(f"Failed to load existing knowledge: {e}")

    def add_knowledge_item(self, item: KnowledgeItem) -> bool:
        """
        Add new knowledge item

        Args:
            item: KnowledgeItem to add

        Returns:
            bool: True if successfully added
        """
        try:
            with self.lock:
                conn = psycopg2.connect(**self.db_config)
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO knowledge_items (
                        item_id, knowledge_type, title, content, tags,
                        created_at, updated_at, version, created_by,
                        confidence, relevance_score, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    item.item_id, item.knowledge_type.value, item.title,
                    json.dumps(item.content), item.tags, item.created_at,
                    item.updated_at, item.version, item.created_by,
                    item.confidence, item.relevance_score,
                    json.dumps(item.metadata) if item.metadata else None
                ))

                conn.commit()
                conn.close()

                # Update vectors and cache
                self._update_item_vector(item)
                self._cache_item(item)

                # Create version record
                self._create_version_record(VersionAction.CREATE, f"Added item: {item.title}",
                                          [item.item_id], "system")

                logger.info(f"Added knowledge item: {item.item_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to add knowledge item: {e}")
            return False

    def update_knowledge_item(self, item: KnowledgeItem) -> bool:
        """
        Update existing knowledge item

        Args:
            item: Updated KnowledgeItem

        Returns:
            bool: True if successfully updated
        """
        try:
            with self.lock:
                # Store rollback data first
                rollback_data = self._get_item_rollback_data(item.item_id)

                conn = psycopg2.connect(**self.db_config)
                cursor = conn.cursor()

                item.updated_at = datetime.utcnow()
                item.version += 1

                cursor.execute("""
                    UPDATE knowledge_items SET
                        title = %s, content = %s, tags = %s, updated_at = %s,
                        version = %s, confidence = %s, relevance_score = %s,
                        metadata = %s
                    WHERE item_id = %s
                """, (
                    item.title, json.dumps(item.content), item.tags,
                    item.updated_at, item.version, item.confidence,
                    item.relevance_score,
                    json.dumps(item.metadata) if item.metadata else None,
                    item.item_id
                ))

                if cursor.rowcount > 0:
                    conn.commit()
                    conn.close()

                    # Update vectors and cache
                    self._update_item_vector(item)
                    self._cache_item(item)

                    # Create version record with rollback data
                    self._create_version_record(VersionAction.UPDATE, f"Updated item: {item.title}",
                                              [item.item_id], "system", rollback_data)

                    logger.info(f"Updated knowledge item: {item.item_id}")
                    return True
                else:
                    conn.close()
                    return False

        except Exception as e:
            logger.error(f"Failed to update knowledge item: {e}")
            return False

    def delete_knowledge_item(self, item_id: str, deleted_by: str = "system") -> bool:
        """
        Delete knowledge item (soft delete with versioning)

        Args:
            item_id: ID of item to delete
            deleted_by: User who deleted the item

        Returns:
            bool: True if successfully deleted
        """
        try:
            with self.lock:
                # Store rollback data
                rollback_data = self._get_item_rollback_data(item_id)
                if not rollback_data:
                    return False

                conn = psycopg2.connect(**self.db_config)
                cursor = conn.cursor()

                cursor.execute("""
                    DELETE FROM knowledge_items WHERE item_id = %s
                """, (item_id,))

                if cursor.rowcount > 0:
                    conn.commit()
                    conn.close()

                    # Remove from cache and vectors
                    self._remove_from_cache(item_id)
                    self.knowledge_vectors.pop(item_id, None)

                    # Create version record
                    self._create_version_record(VersionAction.DELETE, f"Deleted item: {item_id}",
                                              [item_id], deleted_by, rollback_data)

                    logger.info(f"Deleted knowledge item: {item_id}")
                    return True
                else:
                    conn.close()
                    return False

        except Exception as e:
            logger.error(f"Failed to delete knowledge item: {e}")
            return False

    def search_knowledge(self, query: KnowledgeQuery) -> KnowledgeSearchResult:
        """
        Search knowledge base with advanced filtering and ranking

        Args:
            query: KnowledgeQuery specification

        Returns:
            KnowledgeSearchResult: Search results
        """
        start_time = datetime.utcnow()

        try:
            with self.lock:
                # Check cache first
                cache_key = self._generate_query_cache_key(query)
                cached_result = self._get_cached_search_result(cache_key)

                if cached_result:
                    self.cache_hit_count += 1
                    return cached_result

                self.cache_miss_count += 1

                # Build SQL query
                sql_conditions = []
                params = []

                # Text search using full-text search
                if query.query_text:
                    sql_conditions.append("to_tsvector('english', title || ' ' || content::text) @@ plainto_tsquery(%s)")
                    params.append(query.query_text)

                # Knowledge type filtering
                if query.knowledge_types:
                    type_values = [kt.value for kt in query.knowledge_types]
                    sql_conditions.append(f"knowledge_type = ANY(%s)")
                    params.append(type_values)

                # Tag filtering
                if query.tags:
                    sql_conditions.append("tags && %s")
                    params.append(query.tags)

                # Date filtering
                if query.created_after:
                    sql_conditions.append("created_at >= %s")
                    params.append(query.created_after)

                if query.created_before:
                    sql_conditions.append("created_at <= %s")
                    params.append(query.created_before)

                # Confidence filtering
                if query.min_confidence > 0:
                    sql_conditions.append("confidence >= %s")
                    params.append(query.min_confidence)

                # Build final query
                where_clause = " AND ".join(sql_conditions) if sql_conditions else "TRUE"

                sql = f"""
                    SELECT item_id, knowledge_type, title, content, tags,
                           created_at, updated_at, version, created_by,
                           confidence, relevance_score, metadata
                    FROM knowledge_items
                    WHERE {where_clause}
                    ORDER BY relevance_score DESC, created_at DESC
                    LIMIT %s
                """
                params.append(query.max_results)

                # Execute search
                conn = psycopg2.connect(**self.db_config)
                cursor = conn.cursor()
                cursor.execute(sql, params)

                rows = cursor.fetchall()

                # Get total count
                count_sql = f"""
                    SELECT COUNT(*) FROM knowledge_items WHERE {where_clause}
                """
                cursor.execute(count_sql, params[:-1])  # Exclude LIMIT parameter
                total_count = cursor.fetchone()[0]

                conn.close()

                # Convert to KnowledgeItem objects
                items = []
                for row in rows:
                    item = KnowledgeItem(
                        item_id=row[0],
                        knowledge_type=KnowledgeType(row[1]),
                        title=row[2],
                        content=row[3],
                        tags=row[4] or [],
                        created_at=row[5],
                        updated_at=row[6],
                        version=row[7],
                        created_by=row[8] or "",
                        confidence=row[9],
                        relevance_score=row[10],
                        metadata=row[11]
                    )
                    items.append(item)

                # Calculate relevance scores using vector similarity
                relevance_scores = self._calculate_relevance_scores(query.query_text, items)

                # Sort by relevance if we have scores
                if relevance_scores:
                    sorted_pairs = sorted(zip(items, relevance_scores),
                                        key=lambda x: x[1], reverse=True)
                    items, relevance_scores = zip(*sorted_pairs)
                    items = list(items)
                    relevance_scores = list(relevance_scores)
                else:
                    relevance_scores = [item.relevance_score for item in items]

                query_time = (datetime.utcnow() - start_time).total_seconds()
                self.query_times.append(query_time)

                # Generate suggestions
                suggestions = self._generate_search_suggestions(query.query_text, items)

                result = KnowledgeSearchResult(
                    items=items,
                    total_count=total_count,
                    query_time=query_time,
                    relevance_scores=relevance_scores,
                    suggestions=suggestions
                )

                # Cache result
                self._cache_search_result(cache_key, result)

                return result

        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return KnowledgeSearchResult(
                items=[],
                total_count=0,
                query_time=0.0,
                relevance_scores=[]
            )

    def get_knowledge_item(self, item_id: str) -> Optional[KnowledgeItem]:
        """Get specific knowledge item by ID"""
        try:
            # Check cache first
            cached_item = self._get_cached_item(item_id)
            if cached_item:
                self.cache_hit_count += 1
                return cached_item

            self.cache_miss_count += 1

            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT item_id, knowledge_type, title, content, tags,
                       created_at, updated_at, version, created_by,
                       confidence, relevance_score, metadata
                FROM knowledge_items
                WHERE item_id = %s
            """, (item_id,))

            row = cursor.fetchone()
            conn.close()

            if row:
                item = KnowledgeItem(
                    item_id=row[0],
                    knowledge_type=KnowledgeType(row[1]),
                    title=row[2],
                    content=row[3],
                    tags=row[4] or [],
                    created_at=row[5],
                    updated_at=row[6],
                    version=row[7],
                    created_by=row[8] or "",
                    confidence=row[9],
                    relevance_score=row[10],
                    metadata=row[11]
                )

                self._cache_item(item)
                return item

            return None

        except Exception as e:
            logger.error(f"Failed to get knowledge item {item_id}: {e}")
            return None

    def get_analytics(self, start_date: datetime = None,
                     end_date: datetime = None) -> KnowledgeAnalytics:
        """
        Get knowledge base analytics

        Args:
            start_date: Start date for analytics period
            end_date: End date for analytics period

        Returns:
            KnowledgeAnalytics: Analytics data
        """
        try:
            if not start_date:
                start_date = datetime.utcnow() - timedelta(days=30)
            if not end_date:
                end_date = datetime.utcnow()

            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Total items
            cursor.execute("SELECT COUNT(*) FROM knowledge_items")
            total_items = cursor.fetchone()['count']

            # Items by type
            cursor.execute("""
                SELECT knowledge_type, COUNT(*)
                FROM knowledge_items
                GROUP BY knowledge_type
            """)
            items_by_type = {row['knowledge_type']: row['count']
                           for row in cursor.fetchall()}

            # Items by tag
            cursor.execute("""
                SELECT unnest(tags) as tag, COUNT(*)
                FROM knowledge_items
                WHERE tags IS NOT NULL
                GROUP BY unnest(tags)
                ORDER BY COUNT(*) DESC
                LIMIT 20
            """)
            items_by_tag = {row['tag']: row['count']
                          for row in cursor.fetchall()}

            # Average confidence
            cursor.execute("SELECT AVG(confidence) FROM knowledge_items")
            avg_confidence = cursor.fetchone()['avg'] or 0.0

            # Recent activity
            cursor.execute("""
                SELECT item_id, knowledge_type, title, created_at, updated_at
                FROM knowledge_items
                WHERE created_at BETWEEN %s AND %s OR updated_at BETWEEN %s AND %s
                ORDER BY GREATEST(created_at, updated_at) DESC
                LIMIT 50
            """, (start_date, end_date, start_date, end_date))

            recent_activity = []
            for row in cursor.fetchall():
                recent_activity.append({
                    'item_id': row['item_id'],
                    'type': row['knowledge_type'],
                    'title': row['title'],
                    'created_at': row['created_at'].isoformat(),
                    'updated_at': row['updated_at'].isoformat()
                })

            # Version history size
            cursor.execute("SELECT COUNT(*) FROM knowledge_versions")
            version_history_size = cursor.fetchone()['count']

            conn.close()

            # Cache statistics
            total_cache_requests = self.cache_hit_count + self.cache_miss_count
            cache_hit_rate = (self.cache_hit_count / total_cache_requests
                            if total_cache_requests > 0 else 0.0)

            # Query performance
            query_performance = {}
            if self.query_times:
                query_performance = {
                    'avg_query_time': np.mean(self.query_times),
                    'min_query_time': np.min(self.query_times),
                    'max_query_time': np.max(self.query_times),
                    'total_queries': len(self.query_times)
                }

            return KnowledgeAnalytics(
                total_items=total_items,
                items_by_type=items_by_type,
                items_by_tag=items_by_tag,
                avg_confidence=float(avg_confidence),
                recent_activity=recent_activity,
                version_history_size=version_history_size,
                cache_hit_rate=cache_hit_rate,
                query_performance=query_performance
            )

        except Exception as e:
            logger.error(f"Failed to get analytics: {e}")
            return KnowledgeAnalytics(
                total_items=0,
                items_by_type={},
                items_by_tag={},
                avg_confidence=0.0,
                recent_activity=[],
                version_history_size=0,
                cache_hit_rate=0.0,
                query_performance={}
            )

    def create_knowledge_backup(self) -> str:
        """Create backup of entire knowledge base"""
        try:
            backup_id = str(uuid.uuid4())
            backup_data = {
                'backup_id': backup_id,
                'created_at': datetime.utcnow().isoformat(),
                'version': self.current_version,
                'items': [],
                'versions': []
            }

            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Backup all items
            cursor.execute("SELECT * FROM knowledge_items")
            for row in cursor.fetchall():
                backup_data['items'].append(row)

            # Backup version history
            cursor.execute("SELECT * FROM knowledge_versions")
            for row in cursor.fetchall():
                backup_data['versions'].append(row)

            conn.close()

            # Save backup
            backup_file = Path(f"/tmp/kb_backup_{backup_id}.json")
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, default=str, indent=2)

            logger.info(f"Created knowledge base backup: {backup_id}")
            return backup_id

        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return ""

    def restore_from_backup(self, backup_id: str) -> bool:
        """Restore knowledge base from backup"""
        try:
            backup_file = Path(f"/tmp/kb_backup_{backup_id}.json")
            if not backup_file.exists():
                return False

            with open(backup_file, 'r') as f:
                backup_data = json.load(f)

            # Create restore version record first
            self._create_version_record(
                VersionAction.ROLLBACK,
                f"Restored from backup: {backup_id}",
                [],
                "system"
            )

            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Clear current data
            cursor.execute("DELETE FROM knowledge_items")
            cursor.execute("DELETE FROM knowledge_versions")

            # Restore items
            for item_data in backup_data['items']:
                cursor.execute("""
                    INSERT INTO knowledge_items VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, item_data)

            # Restore versions
            for version_data in backup_data['versions']:
                cursor.execute("""
                    INSERT INTO knowledge_versions VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, version_data)

            conn.commit()
            conn.close()

            # Clear caches and reload
            self._clear_all_caches()
            self._load_existing_knowledge()

            logger.info(f"Restored knowledge base from backup: {backup_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return False

    def _calculate_relevance_scores(self, query_text: str,
                                  items: List[KnowledgeItem]) -> List[float]:
        """Calculate relevance scores using vector similarity"""
        if not query_text or not self.knowledge_vectors:
            return []

        try:
            # Transform query to vector
            query_vector = self.vectorizer.transform([query_text])

            scores = []
            for item in items:
                if item.item_id in self.knowledge_vectors:
                    item_vector = self.knowledge_vectors[item.item_id]
                    similarity = cosine_similarity(query_vector, item_vector)[0][0]
                    scores.append(float(similarity))
                else:
                    scores.append(item.relevance_score)

            return scores

        except Exception as e:
            logger.error(f"Failed to calculate relevance scores: {e}")
            return [item.relevance_score for item in items]

    def _generate_search_suggestions(self, query_text: str,
                                   items: List[KnowledgeItem]) -> List[str]:
        """Generate search suggestions based on results"""
        if not items:
            return []

        suggestions = set()

        # Extract common tags
        for item in items[:10]:  # Top 10 results
            if item.tags:
                suggestions.update(item.tags[:3])  # Top 3 tags per item

        # Extract common words from titles
        if query_text:
            query_words = query_text.lower().split()
            for item in items[:5]:
                title_words = item.title.lower().split()
                for word in title_words:
                    if len(word) > 3 and word not in query_words:
                        suggestions.add(word)

        return list(suggestions)[:10]

    def _update_item_vector(self, item: KnowledgeItem):
        """Update vector representation of item"""
        try:
            text = f"{item.title} {json.dumps(item.content)}"

            # Update vectorizer if needed
            if len(self.knowledge_vectors) > 0:
                # Fit new text with existing corpus
                all_texts = [text]
                vector = self.vectorizer.transform(all_texts)
                self.knowledge_vectors[item.item_id] = vector[0]
            else:
                # First item
                vector = self.vectorizer.fit_transform([text])
                self.knowledge_vectors[item.item_id] = vector[0]

        except Exception as e:
            logger.error(f"Failed to update item vector: {e}")

    def _create_version_record(self, action: VersionAction, description: str,
                             affected_items: List[str], created_by: str,
                             rollback_data: bytes = None):
        """Create version control record"""
        try:
            version_id = str(uuid.uuid4())

            # Create snapshot hash
            snapshot_data = f"{self.current_version}:{len(affected_items)}:{description}"
            snapshot_hash = hashlib.sha256(snapshot_data.encode()).hexdigest()

            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO knowledge_versions (
                    version_id, version_number, action, description,
                    created_by, affected_items, snapshot_hash, rollback_data
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                version_id, self.current_version, action.value, description,
                created_by, affected_items, snapshot_hash, rollback_data
            ))

            conn.commit()
            conn.close()

            self.current_version += 1
            logger.debug(f"Created version record: {version_id}")

        except Exception as e:
            logger.error(f"Failed to create version record: {e}")

    def _get_item_rollback_data(self, item_id: str) -> Optional[bytes]:
        """Get rollback data for item"""
        try:
            item = self.get_knowledge_item(item_id)
            if item:
                return pickle.dumps(asdict(item))
            return None
        except Exception as e:
            logger.error(f"Failed to get rollback data: {e}")
            return None

    def _generate_query_cache_key(self, query: KnowledgeQuery) -> str:
        """Generate cache key for query"""
        key_data = {
            'query_text': query.query_text,
            'knowledge_types': [kt.value for kt in query.knowledge_types] if query.knowledge_types else [],
            'tags': query.tags or [],
            'min_confidence': query.min_confidence,
            'max_results': query.max_results
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return f"kb_search:{hashlib.md5(key_str.encode()).hexdigest()}"

    def _cache_item(self, item: KnowledgeItem):
        """Cache knowledge item"""
        if self.redis_client:
            try:
                key = f"kb_item:{item.item_id}"
                data = pickle.dumps(item)
                self.redis_client.setex(key, 3600, data)  # 1 hour TTL
            except Exception as e:
                logger.warning(f"Failed to cache item: {e}")

    def _get_cached_item(self, item_id: str) -> Optional[KnowledgeItem]:
        """Get cached knowledge item"""
        if self.redis_client:
            try:
                key = f"kb_item:{item_id}"
                data = self.redis_client.get(key)
                if data:
                    return pickle.loads(data)
            except Exception as e:
                logger.warning(f"Failed to get cached item: {e}")
        return None

    def _cache_search_result(self, cache_key: str, result: KnowledgeSearchResult):
        """Cache search result"""
        if self.redis_client:
            try:
                data = pickle.dumps(result)
                self.redis_client.setex(cache_key, 300, data)  # 5 minute TTL
            except Exception as e:
                logger.warning(f"Failed to cache search result: {e}")

    def _get_cached_search_result(self, cache_key: str) -> Optional[KnowledgeSearchResult]:
        """Get cached search result"""
        if self.redis_client:
            try:
                data = self.redis_client.get(cache_key)
                if data:
                    return pickle.loads(data)
            except Exception as e:
                logger.warning(f"Failed to get cached search result: {e}")
        return None

    def _remove_from_cache(self, item_id: str):
        """Remove item from cache"""
        if self.redis_client:
            try:
                key = f"kb_item:{item_id}"
                self.redis_client.delete(key)
            except Exception as e:
                logger.warning(f"Failed to remove from cache: {e}")

    def _clear_all_caches(self):
        """Clear all caches"""
        if self.redis_client:
            try:
                # Clear knowledge base related keys
                keys = self.redis_client.keys("kb_*")
                if keys:
                    self.redis_client.delete(*keys)
            except Exception as e:
                logger.warning(f"Failed to clear caches: {e}")

        # Clear local caches
        self.knowledge_vectors.clear()
        self.similarity_cache.clear()

    def cleanup(self):
        """Cleanup resources"""
        try:
            self._clear_all_caches()

            if self.redis_client:
                self.redis_client.close()

            logger.info("Knowledge manager cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# Factory functions for common configurations

def create_knowledge_manager_with_redis(db_config: Dict[str, str],
                                       redis_host: str = "localhost",
                                       redis_port: int = 6379) -> KnowledgeManager:
    """Create knowledge manager with Redis caching"""
    redis_config = {
        'host': redis_host,
        'port': redis_port,
        'decode_responses': False
    }
    return KnowledgeManager(db_config, redis_config)

def create_simple_knowledge_manager(db_config: Dict[str, str]) -> KnowledgeManager:
    """Create knowledge manager without Redis"""
    return KnowledgeManager(db_config)