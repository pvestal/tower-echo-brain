"""
OMNISCIENT CONTEXT INTEGRATION FOR ECHO BRAIN
Provides unlimited context awareness for all Echo conversations
"""
import psycopg2
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class OmniscientContextManager:
    """Manages unlimited context awareness for Echo Brain"""

    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.connection = None
        self.max_context_items = 10  # Maximum context items to include per query
        self.relevance_threshold = 0.5  # Minimum relevance score

    def connect(self):
        """Establish database connection with timeout"""
        try:
            db_config_with_timeout = self.db_config.copy()
            db_config_with_timeout['connect_timeout'] = 5  # 5 second timeout
            self.connection = psycopg2.connect(**db_config_with_timeout)
            logger.info("🧠 Omniscient context manager connected")
        except Exception as e:
            logger.error(f"Failed to connect to omniscient context: {e}")
            raise

    def search_context(self, query: str, conversation_id: str = None,
                      max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search omniscient context for relevant information
        Returns context items ranked by relevance
        """
        if not self.connection:
            self.connect()

        try:
            cursor = self.connection.cursor()

            # Extract key terms from query for better matching
            key_terms = self._extract_key_terms(query)

            # Build comprehensive search query
            search_sql = """
                SELECT
                    id,
                    source_type,
                    source_category,
                    source_path,
                    title,
                    content_preview,
                    content_full,
                    metadata,
                    tags,
                    importance_score,
                    last_accessed_at,
                    ts_rank(search_vector, plainto_tsquery('english', %s)) as text_relevance
                FROM echo_context_registry
                WHERE
                    (
                        search_vector @@ plainto_tsquery('english', %s) OR
                        content_full ILIKE %s OR
                        tags && %s OR
                        source_path ILIKE %s
                    )
                    AND indexing_status = 'indexed'
                ORDER BY
                    importance_score DESC,
                    text_relevance DESC,
                    last_accessed_at DESC
                LIMIT %s;
            """

            # Prepare search parameters
            wildcard_query = f"%{query}%"
            tag_terms = key_terms + ['patrick', 'anime', 'music', 'tower', 'echo', 'claude']
            path_pattern = f"%{'/'.join(key_terms[:2]) if len(key_terms) >= 2 else key_terms[0] if key_terms else query}%"

            cursor.execute(search_sql, (
                query,           # Full-text search
                query,           # Full-text search (repeat)
                wildcard_query,  # Content LIKE search
                tag_terms,       # Tag array search
                path_pattern,    # Path pattern search
                max_results
            ))

            results = []
            for row in cursor.fetchall():
                result = {
                    'id': row[0],
                    'source_type': row[1],
                    'source_category': row[2],
                    'source_path': row[3],
                    'title': row[4],
                    'content_preview': row[5],
                    'content_full': row[6],
                    'metadata': row[7],
                    'tags': row[8],
                    'importance_score': row[9],
                    'last_accessed_at': row[10],
                    'text_relevance': float(row[11]) if row[11] else 0.0,
                    'query_relevance': self._calculate_query_relevance(query, row)
                }
                results.append(result)

            # Update access tracking
            self._track_context_usage(query, conversation_id, results)

            logger.info(f"🔍 Found {len(results)} omniscient context items for: '{query[:50]}...'")
            return results

        except Exception as e:
            logger.error(f"Omniscient context search failed: {e}")
            return []

    def get_personal_knowledge(self, knowledge_type: str = None) -> Dict[str, Any]:
        """Retrieve Patrick's personal knowledge from context"""
        if not self.connection:
            self.connect()

        try:
            cursor = self.connection.cursor()

            if knowledge_type:
                cursor.execute("""
                    SELECT knowledge_key, knowledge_value, confidence
                    FROM echo_personal_knowledge
                    WHERE knowledge_type = %s
                    ORDER BY confidence DESC
                """, (knowledge_type,))
            else:
                cursor.execute("""
                    SELECT knowledge_type, knowledge_key, knowledge_value, confidence
                    FROM echo_personal_knowledge
                    ORDER BY knowledge_type, confidence DESC
                """)

            knowledge = {}
            for row in cursor.fetchall():
                if knowledge_type:
                    knowledge[row[0]] = {
                        'value': row[1],
                        'confidence': row[2]
                    }
                else:
                    if row[0] not in knowledge:
                        knowledge[row[0]] = {}
                    knowledge[row[0]][row[1]] = {
                        'value': row[2],
                        'confidence': row[3]
                    }

            return knowledge

        except Exception as e:
            logger.error(f"Failed to retrieve personal knowledge: {e}")
            return {}

    def build_context_summary(self, query: str, conversation_id: str = None) -> str:
        """Build comprehensive context summary for Echo to use"""

        # Get relevant context items
        context_items = self.search_context(query, conversation_id)

        if not context_items:
            return "No specific context found for this query."

        # Get personal knowledge
        personal_knowledge = self.get_personal_knowledge()

        # Build context summary
        context_summary = "🧠 OMNISCIENT CONTEXT AWARENESS:\n\n"

        # Personal information section
        if personal_knowledge:
            context_summary += "📋 PERSONAL KNOWLEDGE:\n"
            for knowledge_type, items in personal_knowledge.items():
                if knowledge_type == 'personal_info':
                    for key, data in items.items():
                        if isinstance(data['value'], str) and data['value'].startswith('"'):
                            # Handle JSON string values
                            value = data['value'].strip('"')
                        else:
                            value = data['value']
                        context_summary += f"  • {key.title()}: {value} (confidence: {data['confidence']:.0%})\n"
                elif knowledge_type == 'work_projects':
                    for key, data in items.items():
                        if isinstance(data['value'], dict):
                            desc = data['value'].get('description', str(data['value']))
                        else:
                            desc = str(data['value'])
                        context_summary += f"  • {key.replace('_', ' ').title()}: {desc}\n"
            context_summary += "\n"

        # Relevant files and context
        if context_items:
            context_summary += "📁 RELEVANT CONTEXT:\n"

            # Group by category for better organization
            categories = {}
            for item in context_items[:10]:  # Top 10 most relevant
                category = item['source_category']
                if category not in categories:
                    categories[category] = []
                categories[category].append(item)

            for category, items in categories.items():
                context_summary += f"\n{category.replace('_', ' ').title()}:\n"
                for item in items[:3]:  # Top 3 per category
                    context_summary += f"  • {item['title']}: {item['content_preview'][:200]}...\n"
                    context_summary += f"    Source: {item['source_path']}\n"
                    context_summary += f"    Relevance: {item['query_relevance']:.0%}\n"

        # Add conversation history context
        conversation_context = self._get_conversation_context(conversation_id)
        if conversation_context:
            context_summary += f"\n💬 RECENT CONVERSATION:\n{conversation_context}\n"

        return context_summary

    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query for enhanced searching"""
        # Remove common words and extract meaningful terms
        stop_words = {'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'a', 'an', 'and', 'or', 'but', 'what', 'how',
                     'where', 'when', 'why', 'do', 'did', 'can', 'could', 'should', 'would'}

        # Extract words and filter
        words = re.findall(r'\b\w+\b', query.lower())
        key_terms = [word for word in words if len(word) > 2 and word not in stop_words]

        return key_terms[:10]  # Return top 10 terms

    def _calculate_query_relevance(self, query: str, context_row: Tuple) -> float:
        """Calculate how relevant a context item is to the query"""
        query_lower = query.lower()

        # Extract data from row
        title = (context_row[4] or '').lower()
        content = (context_row[5] or '').lower()  # content_preview
        tags = context_row[8] or []
        importance = context_row[9] or 0.5

        relevance = 0.0

        # Title match (high weight)
        title_matches = sum(1 for term in query_lower.split() if term in title)
        relevance += title_matches * 0.3

        # Content match (medium weight)
        content_matches = sum(1 for term in query_lower.split() if term in content)
        relevance += content_matches * 0.2

        # Tag match (medium weight)
        tag_matches = sum(1 for term in query_lower.split() if term in [tag.lower() for tag in tags])
        relevance += tag_matches * 0.25

        # Importance score boost
        relevance += (importance / 100.0) * 0.25

        return min(relevance, 1.0)

    def _get_conversation_context(self, conversation_id: str) -> str:
        """Get recent conversation context"""
        if not conversation_id:
            return ""

        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT content_preview
                FROM echo_context_registry
                WHERE source_type = 'conversation'
                  AND metadata->>'conversation_id' = %s
                ORDER BY source_created_at DESC
                LIMIT 3
            """, (conversation_id,))

            contexts = [row[0] for row in cursor.fetchall()]
            return "\n".join(contexts) if contexts else ""

        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}")
            return ""

    def _track_context_usage(self, query: str, conversation_id: str, results: List[Dict]):
        """Track context usage for analytics"""
        try:
            cursor = self.connection.cursor()

            for result in results[:5]:  # Track top 5 results
                cursor.execute("""
                    INSERT INTO echo_context_usage (
                        conversation_id, registry_id, query_text,
                        relevance_score, usage_type, used_at
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    conversation_id,
                    result['id'],
                    query[:500],  # Truncate long queries
                    result['query_relevance'],
                    'omniscient_search',
                    datetime.now()
                ))

            self.connection.commit()

        except Exception as e:
            logger.error(f"Failed to track context usage: {e}")

    def update_context_from_conversation(self, query: str, response: str,
                                       conversation_id: str, metadata: Dict = None):
        """Update omniscient context with new conversation data"""
        try:
            cursor = self.connection.cursor()

            # Check if this conversation is already in context
            cursor.execute("""
                SELECT id FROM echo_context_registry
                WHERE source_path = %s
            """, (f'realtime_conversation:{conversation_id}:latest',))

            content_full = f"QUERY: {query}\n\nRESPONSE: {response}"
            content_hash = hash(content_full)

            if cursor.fetchone():
                # Update existing conversation
                cursor.execute("""
                    UPDATE echo_context_registry SET
                        content_preview = %s,
                        content_full = %s,
                        content_hash = %s,
                        metadata = %s,
                        source_modified_at = NOW(),
                        last_accessed_at = NOW(),
                        access_frequency = access_frequency + 1
                    WHERE source_path = %s
                """, (
                    content_full[:500],
                    content_full,
                    str(content_hash),
                    json.dumps(metadata or {}),
                    f'realtime_conversation:{conversation_id}:latest'
                ))
            else:
                # Insert new conversation
                cursor.execute("""
                    INSERT INTO echo_context_registry (
                        source_type, source_category, source_path,
                        title, content_preview, content_full,
                        content_hash, metadata, tags, importance_score,
                        indexed_at, last_accessed_at, owner_id
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    'conversation',
                    'realtime',
                    f'realtime_conversation:{conversation_id}:latest',
                    f'Latest conversation in {conversation_id}',
                    content_full[:500],
                    content_full,
                    str(content_hash),
                    json.dumps(metadata or {}),
                    ['conversation', 'realtime', 'latest'],
                    95,  # High importance for current conversations
                    datetime.now(),
                    datetime.now(),
                    'patrick'
                ))

            self.connection.commit()
            logger.debug(f"Updated omniscient context with conversation: {conversation_id}")

        except Exception as e:
            logger.error(f"Failed to update conversation context: {e}")

# Global instance for Echo Brain integration
omniscient_context = None

def get_omniscient_context() -> OmniscientContextManager:
    """Get the global omniscient context manager instance"""
    global omniscient_context

    if omniscient_context is None:
        db_config = {
            'host': '192.168.50.135',
            'user': 'patrick',
            'password': 'tower_echo_brain_secret_key_2025',
            'database': 'echo_brain'
        }
        omniscient_context = OmniscientContextManager(db_config)
        omniscient_context.connect()

    return omniscient_context