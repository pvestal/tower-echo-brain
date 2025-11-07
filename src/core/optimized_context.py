"""
OPTIMIZED OMNISCIENT CONTEXT SYSTEM FOR ECHO BRAIN
Ultra-fast search across 198K+ context items with intelligent caching
"""
import asyncio
import psycopg2
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import time
import pickle
import redis

logger = logging.getLogger(__name__)

class OptimizedOmniscientContext:
    """Ultra-fast omniscient context with intelligent caching and query optimization"""

    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.connection = None

        # Redis cache for ultra-fast responses
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=2, decode_responses=False)
            self.redis_client.ping()
            logger.info("🚀 Redis cache connected for omniscient context")
        except:
            self.redis_client = None
            logger.warning("🚀 Redis not available, using memory cache")

        # In-memory caches
        self.personal_knowledge_cache = {}
        self.frequent_queries_cache = {}
        self.conversation_cache = {}

        # Cache TTLs
        self.personal_knowledge_ttl = 3600  # 1 hour
        self.query_cache_ttl = 300  # 5 minutes
        self.conversation_cache_ttl = 600  # 10 minutes

        # Performance optimization settings
        self.max_context_items = 5
        self.max_search_time = 2.0  # Maximum 2 seconds for search
        self.use_materialized_views = True

        # Query optimization patterns
        self.common_patterns = {
            'personal': ['name', 'patrick', 'my', 'i am', 'i work'],
            'projects': ['anime', 'music', 'tower', 'work', 'project'],
            'technical': ['code', 'system', 'service', 'api', 'database'],
            'recent': ['latest', 'recent', 'current', 'now', 'today']
        }

    async def connect(self):
        """Establish database connection with connection pooling"""
        try:
            # Use connection with specific optimizations
            db_config_optimized = self.db_config.copy()
            db_config_optimized.update({
                'connect_timeout': 3,
                'application_name': 'echo_omniscient_context',
                'options': '-c statement_timeout=5000'  # 5 second query timeout
            })

            self.connection = psycopg2.connect(**db_config_optimized)
            self.connection.autocommit = True  # Faster for read-only queries

            # Load personal knowledge into cache immediately
            await self._load_personal_knowledge_cache()

            logger.info("🧠 Optimized omniscient context connected")
        except Exception as e:
            logger.error(f"Failed to connect to optimized context: {e}")
            raise

    async def search_context_optimized(self, query: str, conversation_id: str = None,
                                     max_results: int = 5) -> List[Dict[str, Any]]:
        """Ultra-fast optimized context search with intelligent caching"""
        start_time = time.time()

        try:
            # 1. Check cache first
            cache_key = self._generate_cache_key(query, max_results)
            cached_result = await self._get_from_cache(cache_key, 'query')

            if cached_result:
                logger.info(f"🚀 Cache hit for query: '{query[:30]}...' ({time.time() - start_time:.3f}s)")
                return cached_result

            # 2. Optimize query based on patterns
            optimized_query, search_strategy = self._optimize_query(query)

            # 3. Use appropriate search strategy
            if search_strategy == 'personal':
                results = await self._search_personal_context(optimized_query, max_results)
            elif search_strategy == 'recent':
                results = await self._search_recent_context(optimized_query, max_results)
            elif search_strategy == 'technical':
                results = await self._search_technical_context(optimized_query, max_results)
            else:
                results = await self._search_general_context(optimized_query, max_results)

            # 4. Cache successful results
            search_time = time.time() - start_time
            if search_time < self.max_search_time and results:
                await self._store_in_cache(cache_key, results, 'query')

            logger.info(f"🔍 Optimized search: '{query[:30]}...' → {len(results)} results ({search_time:.3f}s)")
            return results

        except Exception as e:
            logger.error(f"Optimized search failed: {e}")
            return []

    def _optimize_query(self, query: str) -> Tuple[str, str]:
        """Analyze query and determine optimal search strategy"""
        query_lower = query.lower()

        # Determine search strategy based on query patterns
        for strategy, patterns in self.common_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return query, strategy

        return query, 'general'

    async def _search_personal_context(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Fast search for personal information"""
        # First check personal knowledge cache
        personal_data = await self._get_personal_knowledge_cached()

        results = []

        # Add personal knowledge as context
        for knowledge_type, items in personal_data.items():
            for key, data in items.items():
                if any(term in query.lower() for term in [key, knowledge_type]):
                    results.append({
                        'id': f'personal_{knowledge_type}_{key}',
                        'source_type': 'personal_knowledge',
                        'source_category': knowledge_type,
                        'title': f"Personal: {key.replace('_', ' ').title()}",
                        'content_preview': str(data.get('value', '')),
                        'importance_score': 100,
                        'query_relevance': data.get('confidence', 0.9)
                    })

        # Add targeted database search for personal content
        if len(results) < max_results:
            db_results = await self._fast_db_search(
                query,
                filters="tags @> ARRAY['patrick'] OR tags @> ARRAY['personal']",
                limit=max_results - len(results)
            )
            results.extend(db_results)

        return results[:max_results]

    async def _search_recent_context(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search recent conversations and activities"""
        # Check conversation cache first
        if hasattr(self, 'conversation_cache') and self.conversation_cache:
            recent_conversations = list(self.conversation_cache.values())[:3]
            results = [conv for conv in recent_conversations if query.lower() in conv.get('content_preview', '').lower()]
            if results:
                return results[:max_results]

        # Fast recent content search
        return await self._fast_db_search(
            query,
            filters="last_accessed_at > NOW() - INTERVAL '24 hours'",
            order_by="last_accessed_at DESC",
            limit=max_results
        )

    async def _search_technical_context(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search technical content (code, configs, APIs)"""
        return await self._fast_db_search(
            query,
            filters="source_category IN ('source_code', 'configuration', 'tower_services', 'system_config')",
            limit=max_results
        )

    async def _search_general_context(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """General optimized search using materialized views when possible"""
        if self.use_materialized_views:
            return await self._search_materialized_view(query, max_results)
        else:
            return await self._fast_db_search(query, limit=max_results)

    async def _fast_db_search(self, query: str, filters: str = "",
                            order_by: str = "importance_score DESC",
                            limit: int = 5) -> List[Dict[str, Any]]:
        """Fast database search with optimized queries"""
        try:
            cursor = self.connection.cursor()

            # Use the optimized search function we created
            if not filters:
                cursor.execute("""
                    SELECT * FROM search_omniscient_context(%s, %s, 50)
                """, (query, limit))
            else:
                # Custom search with filters
                cursor.execute(f"""
                    SELECT id, source_type, source_category, title, content_preview,
                           tags, importance_score,
                           ts_rank(to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content_preview, '')),
                                  plainto_tsquery('english', %s)) as relevance_score
                    FROM echo_context_registry
                    WHERE (to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content_preview, ''))
                           @@ plainto_tsquery('english', %s)
                           OR tags && string_to_array(lower(%s), ' '))
                      AND {filters}
                    ORDER BY {order_by}
                    LIMIT %s
                """, (query, query, query, limit))

            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'source_type': row[1],
                    'source_category': row[2],
                    'title': row[3],
                    'content_preview': row[4],
                    'tags': row[5] if len(row) > 5 else [],
                    'importance_score': row[6] if len(row) > 6 else 50,
                    'query_relevance': float(row[7]) if len(row) > 7 else 0.5
                })

            return results

        except Exception as e:
            logger.error(f"Fast DB search failed: {e}")
            return []

    async def _search_materialized_view(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using pre-computed materialized view for top content"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT id, source_type, source_category, title, content_preview, tags, importance_score,
                       ts_rank(to_tsvector('english', search_keywords), plainto_tsquery('english', %s)) as relevance
                FROM echo_top_context
                WHERE to_tsvector('english', search_keywords) @@ plainto_tsquery('english', %s)
                ORDER BY relevance DESC, importance_score DESC
                LIMIT %s
            """, (query, query, max_results))

            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'source_type': row[1],
                    'source_category': row[2],
                    'title': row[3],
                    'content_preview': row[4],
                    'tags': row[5],
                    'importance_score': row[6],
                    'query_relevance': float(row[7])
                })

            return results

        except Exception as e:
            logger.error(f"Materialized view search failed: {e}")
            return await self._fast_db_search(query, limit=max_results)

    async def _get_personal_knowledge_cached(self) -> Dict[str, Any]:
        """Get personal knowledge with smart caching"""
        cache_key = 'personal_knowledge'

        # Check memory cache first
        if cache_key in self.personal_knowledge_cache:
            cache_time, data = self.personal_knowledge_cache[cache_key]
            if time.time() - cache_time < self.personal_knowledge_ttl:
                return data

        # Check Redis cache
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    return pickle.loads(cached_data)
            except:
                pass

        # Load from database and cache
        return await self._load_personal_knowledge_cache()

    async def _load_personal_knowledge_cache(self) -> Dict[str, Any]:
        """Load and cache personal knowledge"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT knowledge_type, knowledge_key, knowledge_value, confidence
                FROM echo_personal_knowledge
                WHERE confidence >= 0.7
                ORDER BY confidence DESC
            """)

            knowledge = {}
            for row in cursor.fetchall():
                knowledge_type, key, value, confidence = row
                if knowledge_type not in knowledge:
                    knowledge[knowledge_type] = {}

                # Handle JSON values
                if isinstance(value, dict):
                    knowledge[knowledge_type][key] = {
                        'value': value,
                        'confidence': confidence
                    }
                else:
                    knowledge[knowledge_type][key] = {
                        'value': value,
                        'confidence': confidence
                    }

            # Cache in memory and Redis
            self.personal_knowledge_cache['personal_knowledge'] = (time.time(), knowledge)

            if self.redis_client:
                try:
                    self.redis_client.setex(
                        'personal_knowledge',
                        self.personal_knowledge_ttl,
                        pickle.dumps(knowledge)
                    )
                except:
                    pass

            return knowledge

        except Exception as e:
            logger.error(f"Failed to load personal knowledge: {e}")
            return {}

    def _generate_cache_key(self, query: str, max_results: int) -> str:
        """Generate cache key for query"""
        return hashlib.md5(f"{query}:{max_results}".encode()).hexdigest()

    async def _get_from_cache(self, cache_key: str, cache_type: str) -> Optional[List[Dict]]:
        """Get data from appropriate cache"""
        # Memory cache first
        memory_cache = getattr(self, f'{cache_type}_cache', {})
        if cache_key in memory_cache:
            cache_time, data = memory_cache[cache_key]
            if time.time() - cache_time < getattr(self, f'{cache_type}_cache_ttl', 300):
                return data

        # Redis cache
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(f"{cache_type}:{cache_key}")
                if cached_data:
                    return pickle.loads(cached_data)
            except:
                pass

        return None

    async def _store_in_cache(self, cache_key: str, data: Any, cache_type: str):
        """Store data in appropriate caches"""
        # Memory cache
        cache = getattr(self, f'{cache_type}_cache', {})
        cache[cache_key] = (time.time(), data)

        # Limit memory cache size
        if len(cache) > 100:
            # Remove oldest entries
            sorted_items = sorted(cache.items(), key=lambda x: x[1][0])
            for key, _ in sorted_items[:50]:
                del cache[key]

        # Redis cache
        if self.redis_client:
            try:
                ttl = getattr(self, f'{cache_type}_cache_ttl', 300)
                self.redis_client.setex(
                    f"{cache_type}:{cache_key}",
                    ttl,
                    pickle.dumps(data)
                )
            except:
                pass

    async def build_context_summary_optimized(self, query: str, conversation_id: str = None) -> str:
        """Build optimized context summary with minimal overhead"""
        start_time = time.time()

        # Get relevant context with optimization
        context_items = await self.search_context_optimized(query, conversation_id, max_results=3)

        if not context_items:
            return "📋 CONTEXT: No specific context found."

        # Get cached personal knowledge
        personal_knowledge = await self._get_personal_knowledge_cached()

        # Build minimal context summary
        context_summary = "📋 CONTEXT:\n"

        # Essential personal info only
        if personal_knowledge.get('personal_info'):
            name = personal_knowledge['personal_info'].get('name', {}).get('value', '')
            if name:
                context_summary += f"• Name: {name.strip('\"')}\n"

        if personal_knowledge.get('work_projects'):
            projects = []
            for key, data in personal_knowledge['work_projects'].items():
                if isinstance(data.get('value'), dict):
                    desc = data['value'].get('description', '')
                else:
                    desc = str(data.get('value', ''))
                if desc and 'anime' in desc.lower():
                    projects.append("anime production")
                elif desc and 'music' in desc.lower():
                    projects.append("music projects")
            if projects:
                context_summary += f"• Projects: {', '.join(set(projects))}\n"

        # Top relevant files/context (max 2)
        if context_items:
            context_summary += "📁 RELEVANT:\n"
            for item in context_items[:2]:
                context_summary += f"• {item['title']}: {item['content_preview'][:100]}...\n"

        build_time = time.time() - start_time
        logger.info(f"⚡ Context summary built in {build_time:.3f}s")

        return context_summary

# Global optimized instance
optimized_context = None

def get_optimized_omniscient_context() -> OptimizedOmniscientContext:
    """Get the global optimized omniscient context manager instance"""
    global optimized_context

    if optimized_context is None:
        db_config = {
            'host': '192.168.50.135',
            'user': 'patrick',
            'password': 'tower_echo_brain_secret_key_2025',
            'database': 'echo_brain'
        }
        optimized_context = OptimizedOmniscientContext(db_config)

    return optimized_context