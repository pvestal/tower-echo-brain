#!/usr/bin/env python3
"""
Semantic Knowledge Retrieval Manager
Queries the knowledge_items table using vector similarity search
"""

import psycopg2
import psycopg2.pool
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class KnowledgeRetrieval:
    """
    Retrieves knowledge items using semantic search
    Requires: sentence-transformers library
    """
    
    def __init__(self):
        self._db_config = {
            'host': 'localhost',
            'database': 'echo_brain',
            'user': 'patrick',
            'password': '***REMOVED***'
        }
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load embedding model (lazy loading)"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-mpnet-base-v2')
            logger.info("✅ Sentence-Transformers model loaded for knowledge retrieval")
        except ImportError:
            logger.warning("⚠️  sentence-transformers not installed - semantic search disabled")
            self.model = None
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            self.model = None
            
    def search_knowledge(
        self,
        query: str,
        limit: int = 5,
        similarity_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Semantic search across knowledge_items
        
        Args:
            query: Search query text
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of knowledge items with metadata
        """
        if not self.model:
            logger.warning("Semantic search unavailable - using fallback text search")
            return self._fallback_text_search(query, limit)
            
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query])[0]
            embedding_str = '[' + ','.join(map(str, query_embedding.tolist())) + ']'
            
            # Connect to database
            conn = psycopg2.connect(**self._db_config)
            cursor = conn.cursor()
            
            try:
                # Vector similarity search with explicit cast
                cursor.execute("""
                    SELECT 
                        item_id,
                        title,
                        knowledge_type,
                        content->>'original_text' as text_content,
                        metadata,
                        tags,
                        created_at,
                        1 - (embedding <=> %s::vector) as similarity
                    FROM knowledge_items
                    WHERE embedding IS NOT NULL
                      AND (1 - (embedding <=> %s::vector)) >= %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (embedding_str, embedding_str, similarity_threshold, embedding_str, limit))
                
                results = cursor.fetchall()
                
                # Format results
                knowledge = []
                for row in results:
                    knowledge.append({
                        'item_id': row[0],
                        'title': row[1],
                        'type': row[2],
                        'content': row[3][:500] if row[3] else '',  # Limit for context
                        'similarity': round(row[7], 3),
                        'metadata': row[4],
                        'tags': row[5],
                        'created_at': row[6].isoformat() if row[6] else None
                    })
                    
                logger.info(f"🔍 Found {len(knowledge)} knowledge items for query: {query[:50]}...")
                return knowledge
                
            finally:
                cursor.close()
                conn.close()
                    
        except Exception as e:
            logger.error(f"❌ Knowledge search failed: {e}")
            import traceback
            traceback.print_exc()
            return []
            
    def _fallback_text_search(
        self,
        query: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Fallback to PostgreSQL full-text search if embeddings unavailable
        """
        try:
            conn = psycopg2.connect(**self._db_config)
            cursor = conn.cursor()
            
            try:
                # Simple ILIKE search as fallback
                cursor.execute("""
                    SELECT 
                        item_id,
                        title,
                        knowledge_type,
                        content->>'original_text' as text_content,
                        metadata,
                        tags,
                        created_at
                    FROM knowledge_items
                    WHERE title ILIKE %s OR (content->>'original_text') ILIKE %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (f'%{query}%', f'%{query}%', limit))
                
                results = cursor.fetchall()
                
                knowledge = []
                for row in results:
                    knowledge.append({
                        'item_id': row[0],
                        'title': row[1],
                        'type': row[2],
                        'content': row[3][:500] if row[3] else '',
                        'similarity': 0.5,  # Default for text search
                        'metadata': row[4],
                        'tags': row[5],
                        'created_at': row[6].isoformat() if row[6] else None
                    })
                    
                logger.info(f"📝 Fallback text search found {len(knowledge)} items")
                return knowledge
                
            finally:
                cursor.close()
                conn.close()
                    
        except Exception as e:
            logger.error(f"❌ Fallback search failed: {e}")
            return []
            
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        try:
            conn = psycopg2.connect(**self._db_config)
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_items,
                        COUNT(embedding) as items_with_embeddings,
                        COUNT(DISTINCT knowledge_type) as knowledge_types,
                        pg_size_pretty(pg_total_relation_size('knowledge_items')) as table_size
                    FROM knowledge_items
                """)
                
                stats = cursor.fetchone()
                
                cursor.execute("""
                    SELECT knowledge_type, COUNT(*) as count
                    FROM knowledge_items
                    GROUP BY knowledge_type
                    ORDER BY count DESC
                    LIMIT 10
                """)
                
                types = cursor.fetchall()
                
                return {
                    'total_items': stats[0],
                    'items_with_embeddings': stats[1],
                    'knowledge_types': stats[2],
                    'table_size': stats[3],
                    'types_breakdown': [{
                        'type': row[0],
                        'count': row[1]
                    } for row in types]
                }
                
            finally:
                cursor.close()
                conn.close()
                    
        except Exception as e:
            logger.error(f"❌ Failed to get knowledge stats: {e}")
            return {}

# Global instance (optional - can be initialized per-request)
_knowledge_retrieval = None

def get_knowledge_retrieval() -> KnowledgeRetrieval:
    """Get or create knowledge retrieval instance"""
    global _knowledge_retrieval
    if _knowledge_retrieval is None:
        _knowledge_retrieval = KnowledgeRetrieval()
    return _knowledge_retrieval
