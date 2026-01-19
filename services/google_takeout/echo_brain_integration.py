#!/usr/bin/env python3
"""
Echo Brain Integration for Omniscient Personal Data
Bridges omniscient data collection with Echo Brain's knowledge graph and learning systems
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np

# Echo Brain imports
import sys
sys.path.append('/opt/tower-echo-brain/src')

from core.knowledge_graph import KnowledgeGraph
from memory.vector_store import VectorStore
from learning.continuous_learner import ContinuousLearner
from reasoning.inference_engine import InferenceEngine

# Database imports
import asyncpg

logger = logging.getLogger(__name__)

class EchoBrainOmniscientIntegration:
    """
    Integration layer between omniscient data collection and Echo Brain intelligence
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_url = "postgresql://patrick:tower_echo_brain_secret_key_2025@localhost/echo_brain"

        # Echo Brain components
        self.knowledge_graph = None
        self.vector_store = None
        self.continuous_learner = None
        self.inference_engine = None

        # Processing configuration
        self.batch_size = config.get('processing', {}).get('batch_size', 100)
        self.embedding_model = config.get('embedding', {}).get('model', 'text-embedding-3-small')

        logger.info("🧠 Echo Brain Omniscient Integration initialized")

    async def initialize(self):
        """Initialize Echo Brain components"""
        try:
            # Initialize knowledge graph
            self.knowledge_graph = KnowledgeGraph(self.config)
            await self.knowledge_graph.initialize()

            # Initialize vector store
            self.vector_store = VectorStore(self.config)
            await self.vector_store.initialize()

            # Initialize continuous learner
            self.continuous_learner = ContinuousLearner(self.config)
            await self.continuous_learner.initialize()

            # Initialize inference engine
            self.inference_engine = InferenceEngine(self.config)
            await self.inference_engine.initialize()

            logger.info("✅ Echo Brain components initialized")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize Echo Brain components: {e}")
            return False

    async def process_omniscient_data_stream(self):
        """
        Continuously process omniscient data and integrate with Echo Brain
        """
        logger.info("🔄 Starting omniscient data stream processing")

        while True:
            try:
                # Get unprocessed data from omniscient database
                unprocessed_items = await self._get_unprocessed_items()

                if unprocessed_items:
                    logger.info(f"📥 Processing {len(unprocessed_items)} new omniscient items")

                    # Process items in batches
                    for i in range(0, len(unprocessed_items), self.batch_size):
                        batch = unprocessed_items[i:i + self.batch_size]
                        await self._process_data_batch(batch)

                    # Update knowledge graph relationships
                    await self._update_knowledge_relationships()

                    # Generate new insights
                    await self._generate_personal_insights()

                # Wait before next processing cycle
                await asyncio.sleep(30)  # Process every 30 seconds

            except Exception as e:
                logger.error(f"❌ Error in omniscient data stream processing: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _get_unprocessed_items(self) -> List[Dict[str, Any]]:
        """Get unprocessed omniscient data items"""
        conn = await asyncpg.connect(self.db_url)

        try:
            # Get items without embeddings or knowledge graph integration
            rows = await conn.fetch("""
                SELECT id, source, item_type, content, metadata, timestamp,
                       importance_score, privacy_level
                FROM omniscient_data
                WHERE embedding IS NULL
                   OR id NOT IN (
                       SELECT DISTINCT entity_1_id
                       FROM knowledge_relationships
                       WHERE entity_1_type = 'omniscient_data'
                   )
                ORDER BY importance_score DESC, timestamp DESC
                LIMIT $1
            """, self.batch_size * 2)

            return [dict(row) for row in rows]

        finally:
            await conn.close()

    async def _process_data_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of omniscient data items"""

        for item in batch:
            try:
                # Generate embeddings
                embedding = await self._generate_embedding(item)

                # Store in vector database
                await self._store_vector_embedding(item, embedding)

                # Update knowledge graph
                await self._integrate_knowledge_graph(item)

                # Trigger continuous learning
                await self._trigger_learning(item)

                # Update database with embedding
                await self._update_item_embedding(item['id'], embedding)

            except Exception as e:
                logger.error(f"❌ Failed to process item {item['id']}: {e}")

    async def _generate_embedding(self, item: Dict[str, Any]) -> List[float]:
        """Generate vector embedding for omniscient data item"""

        # Create comprehensive text representation
        text_content = self._create_text_representation(item)

        # Generate embedding using Echo Brain's embedding service
        try:
            embedding = await self.vector_store.generate_embedding(text_content)
            return embedding

        except Exception as e:
            logger.error(f"❌ Failed to generate embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 1536

    def _create_text_representation(self, item: Dict[str, Any]) -> str:
        """Create comprehensive text representation of omniscient data item"""

        parts = []

        # Basic metadata
        parts.append(f"Source: {item['source']}")
        parts.append(f"Type: {item['item_type']}")
        parts.append(f"Importance: {item['importance_score']}")
        parts.append(f"Privacy: {item['privacy_level']}")

        # Timestamp context
        timestamp = item['timestamp']
        if timestamp:
            parts.append(f"Date: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            parts.append(f"Day of week: {timestamp.strftime('%A')}")
            parts.append(f"Month: {timestamp.strftime('%B')}")

        # Content-specific text extraction
        content = item['content']

        if item['item_type'] == 'email':
            if 'subject' in content:
                parts.append(f"Subject: {content['subject']}")
            if 'sender' in content:
                parts.append(f"From: {content['sender']}")
            if 'body_text' in content:
                parts.append(f"Content: {content['body_text'][:500]}...")  # Limit length
            if 'keywords' in content:
                parts.append(f"Keywords: {', '.join(content['keywords'])}")

        elif item['item_type'] == 'image':
            if 'faces_detected' in content:
                face_names = [face.get('name', 'Unknown') for face in content['faces_detected']]
                parts.append(f"People: {', '.join(set(face_names))}")
            if 'objects_detected' in content:
                parts.append(f"Objects: {', '.join(content['objects_detected'])}")
            if 'location' in content and content['location']:
                parts.append(f"Location: {content['location']}")

        elif item['item_type'] == 'webpage_visit':
            if 'url' in content:
                parts.append(f"URL: {content['url']}")
            if 'title' in content:
                parts.append(f"Title: {content['title']}")
            if 'domain' in content:
                parts.append(f"Domain: {content['domain']}")
            if 'category' in content:
                parts.append(f"Category: {content['category']}")

        elif item['item_type'] == 'calendar_event':
            if 'title' in content:
                parts.append(f"Event: {content['title']}")
            if 'description' in content:
                parts.append(f"Description: {content['description']}")
            if 'location' in content:
                parts.append(f"Location: {content['location']}")
            if 'attendees' in content:
                parts.append(f"Attendees: {', '.join(content['attendees'])}")

        # Metadata context
        metadata = item.get('metadata', {})
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float)):
                    parts.append(f"{key}: {value}")

        return ' | '.join(parts)

    async def _store_vector_embedding(self, item: Dict[str, Any], embedding: List[float]):
        """Store vector embedding in Echo Brain's vector database"""

        try:
            # Create vector store entry
            vector_entry = {
                'id': f"omniscient_{item['id']}",
                'source': item['source'],
                'type': item['item_type'],
                'content_hash': item.get('content_hash'),
                'timestamp': item['timestamp'],
                'importance_score': item['importance_score'],
                'privacy_level': item['privacy_level'],
                'embedding': embedding,
                'metadata': {
                    'omniscient_id': item['id'],
                    'original_content': item['content'],
                    'processing_metadata': item.get('metadata', {})
                }
            }

            await self.vector_store.upsert(vector_entry)

        except Exception as e:
            logger.error(f"❌ Failed to store vector embedding: {e}")

    async def _integrate_knowledge_graph(self, item: Dict[str, Any]):
        """Integrate omniscient data item into Echo Brain's knowledge graph"""

        try:
            # Create entity for the omniscient data item
            entity_id = await self.knowledge_graph.create_entity(
                entity_type='omniscient_data',
                properties={
                    'omniscient_id': item['id'],
                    'source': item['source'],
                    'item_type': item['item_type'],
                    'timestamp': item['timestamp'].isoformat(),
                    'importance_score': item['importance_score'],
                    'privacy_level': item['privacy_level']
                }
            )

            # Extract and create relationships based on content
            await self._extract_knowledge_relationships(item, entity_id)

        except Exception as e:
            logger.error(f"❌ Failed to integrate with knowledge graph: {e}")

    async def _extract_knowledge_relationships(self, item: Dict[str, Any], entity_id: int):
        """Extract and create knowledge graph relationships"""

        content = item['content']
        item_type = item['item_type']

        if item_type == 'email':
            # Create relationships with people (sender, recipients)
            if 'sender' in content:
                person_id = await self._ensure_person_entity(content['sender'])
                await self.knowledge_graph.create_relationship(
                    entity_id, person_id, 'received_from',
                    strength=item['importance_score']
                )

            if 'recipients' in content:
                for recipient in content['recipients']:
                    person_id = await self._ensure_person_entity(recipient)
                    await self.knowledge_graph.create_relationship(
                        entity_id, person_id, 'sent_to',
                        strength=item['importance_score']
                    )

            # Extract topic relationships from keywords
            if 'keywords' in content:
                for keyword in content['keywords'][:10]:  # Limit to top 10
                    topic_id = await self._ensure_topic_entity(keyword)
                    await self.knowledge_graph.create_relationship(
                        entity_id, topic_id, 'discusses',
                        strength=min(item['importance_score'], 0.8)
                    )

        elif item_type == 'image':
            # Create relationships with detected people
            if 'faces_detected' in content:
                for face in content['faces_detected']:
                    if face.get('name') and face.get('name') != 'Unknown':
                        person_id = await self._ensure_person_entity(face['name'])
                        await self.knowledge_graph.create_relationship(
                            entity_id, person_id, 'contains_person',
                            strength=face.get('confidence', 0.5)
                        )

            # Location relationships
            if 'location' in content and content['location']:
                location_id = await self._ensure_location_entity(content['location'])
                await self.knowledge_graph.create_relationship(
                    entity_id, location_id, 'taken_at',
                    strength=0.9
                )

        elif item_type == 'webpage_visit':
            # Domain and category relationships
            if 'domain' in content:
                domain_id = await self._ensure_domain_entity(content['domain'])
                await self.knowledge_graph.create_relationship(
                    entity_id, domain_id, 'visited_domain',
                    strength=min(content.get('visit_count', 1) * 0.1, 1.0)
                )

            if 'category' in content:
                category_id = await self._ensure_category_entity(content['category'])
                await self.knowledge_graph.create_relationship(
                    entity_id, category_id, 'browsed_category',
                    strength=content.get('interest_score', 0.5)
                )

        elif item_type == 'calendar_event':
            # Attendee relationships
            if 'attendees' in content:
                for attendee in content['attendees']:
                    person_id = await self._ensure_person_entity(attendee)
                    await self.knowledge_graph.create_relationship(
                        entity_id, person_id, 'met_with',
                        strength=item['importance_score']
                    )

            # Location relationships
            if 'location' in content and content['location']:
                location_id = await self._ensure_location_entity(content['location'])
                await self.knowledge_graph.create_relationship(
                    entity_id, location_id, 'occurred_at',
                    strength=0.8
                )

    async def _ensure_person_entity(self, person_identifier: str) -> int:
        """Ensure person entity exists in knowledge graph"""

        # Clean and normalize person identifier
        clean_name = person_identifier.strip().lower()

        # Check if person already exists
        existing_id = await self.knowledge_graph.find_entity(
            entity_type='person',
            properties={'identifier': clean_name}
        )

        if existing_id:
            return existing_id

        # Create new person entity
        return await self.knowledge_graph.create_entity(
            entity_type='person',
            properties={
                'identifier': clean_name,
                'display_name': person_identifier,
                'created_from': 'omniscient_data'
            }
        )

    async def _ensure_topic_entity(self, topic: str) -> int:
        """Ensure topic entity exists in knowledge graph"""

        clean_topic = topic.strip().lower()

        existing_id = await self.knowledge_graph.find_entity(
            entity_type='topic',
            properties={'name': clean_topic}
        )

        if existing_id:
            return existing_id

        return await self.knowledge_graph.create_entity(
            entity_type='topic',
            properties={
                'name': clean_topic,
                'display_name': topic,
                'created_from': 'omniscient_data'
            }
        )

    async def _ensure_location_entity(self, location: Union[str, Dict]) -> int:
        """Ensure location entity exists in knowledge graph"""

        if isinstance(location, dict):
            location_name = location.get('name', str(location))
        else:
            location_name = str(location)

        clean_location = location_name.strip().lower()

        existing_id = await self.knowledge_graph.find_entity(
            entity_type='location',
            properties={'name': clean_location}
        )

        if existing_id:
            return existing_id

        return await self.knowledge_graph.create_entity(
            entity_type='location',
            properties={
                'name': clean_location,
                'display_name': location_name,
                'created_from': 'omniscient_data'
            }
        )

    async def _ensure_domain_entity(self, domain: str) -> int:
        """Ensure domain entity exists in knowledge graph"""

        clean_domain = domain.strip().lower()

        existing_id = await self.knowledge_graph.find_entity(
            entity_type='domain',
            properties={'name': clean_domain}
        )

        if existing_id:
            return existing_id

        return await self.knowledge_graph.create_entity(
            entity_type='domain',
            properties={
                'name': clean_domain,
                'created_from': 'omniscient_data'
            }
        )

    async def _ensure_category_entity(self, category: str) -> int:
        """Ensure category entity exists in knowledge graph"""

        clean_category = category.strip().lower()

        existing_id = await self.knowledge_graph.find_entity(
            entity_type='category',
            properties={'name': clean_category}
        )

        if existing_id:
            return existing_id

        return await self.knowledge_graph.create_entity(
            entity_type='category',
            properties={
                'name': clean_category,
                'created_from': 'omniscient_data'
            }
        )

    async def _trigger_learning(self, item: Dict[str, Any]):
        """Trigger continuous learning from omniscient data item"""

        try:
            # Create learning example from omniscient data
            learning_example = {
                'source': f"omniscient_{item['source']}",
                'content': self._create_text_representation(item),
                'metadata': {
                    'importance_score': item['importance_score'],
                    'privacy_level': item['privacy_level'],
                    'timestamp': item['timestamp'].isoformat(),
                    'item_type': item['item_type']
                }
            }

            # Submit to continuous learner
            await self.continuous_learner.learn_from_example(learning_example)

        except Exception as e:
            logger.error(f"❌ Failed to trigger learning: {e}")

    async def _update_item_embedding(self, item_id: int, embedding: List[float]):
        """Update omniscient data item with generated embedding"""

        conn = await asyncpg.connect(self.db_url)

        try:
            await conn.execute("""
                UPDATE omniscient_data
                SET embedding = $1, last_accessed = CURRENT_TIMESTAMP
                WHERE id = $2
            """, embedding, item_id)

        finally:
            await conn.close()

    async def _update_knowledge_relationships(self):
        """Update and strengthen knowledge graph relationships based on patterns"""

        try:
            # Find co-occurrence patterns
            await self._strengthen_cooccurrence_relationships()

            # Update relationship strengths based on frequency
            await self._update_relationship_strengths()

            # Discover new implicit relationships
            await self._discover_implicit_relationships()

        except Exception as e:
            logger.error(f"❌ Failed to update knowledge relationships: {e}")

    async def _strengthen_cooccurrence_relationships(self):
        """Strengthen relationships based on co-occurrence patterns"""

        conn = await asyncpg.connect(self.db_url)

        try:
            # Find entities that frequently appear together
            cooccurrences = await conn.fetch("""
                SELECT
                    r1.entity_2_id as entity_1,
                    r2.entity_2_id as entity_2,
                    COUNT(*) as cooccurrence_count,
                    AVG(r1.strength + r2.strength) / 2 as avg_strength
                FROM knowledge_relationships r1
                JOIN knowledge_relationships r2 ON r1.entity_1_id = r2.entity_1_id
                WHERE r1.entity_1_type = 'omniscient_data'
                  AND r2.entity_1_type = 'omniscient_data'
                  AND r1.entity_2_id != r2.entity_2_id
                  AND r1.entity_2_type = r2.entity_2_type
                GROUP BY r1.entity_2_id, r2.entity_2_id
                HAVING COUNT(*) >= 3  -- Minimum cooccurrence threshold
                ORDER BY cooccurrence_count DESC, avg_strength DESC
                LIMIT 1000
            """)

            # Create or strengthen relationships between cooccurring entities
            for row in cooccurrences:
                relationship_strength = min(row['avg_strength'] * 0.7, 0.9)

                await self.knowledge_graph.create_or_update_relationship(
                    row['entity_1'], row['entity_2'], 'frequently_with',
                    strength=relationship_strength,
                    metadata={'cooccurrence_count': row['cooccurrence_count']}
                )

        finally:
            await conn.close()

    async def _update_relationship_strengths(self):
        """Update relationship strengths based on frequency and recency"""

        # Implementation for updating relationship strengths
        # This would analyze temporal patterns and interaction frequencies
        pass

    async def _discover_implicit_relationships(self):
        """Discover new implicit relationships through graph analysis"""

        # Implementation for discovering transitive and implicit relationships
        # This would use graph algorithms to find hidden connections
        pass

    async def _generate_personal_insights(self):
        """Generate personal insights from omniscient data patterns"""

        try:
            # Analyze communication patterns
            communication_insights = await self._analyze_communication_patterns()

            # Analyze behavior patterns
            behavior_insights = await self._analyze_behavior_patterns()

            # Analyze social patterns
            social_insights = await self._analyze_social_patterns()

            # Analyze temporal patterns
            temporal_insights = await self._analyze_temporal_patterns()

            # Store insights in knowledge base
            all_insights = {
                'communication': communication_insights,
                'behavior': behavior_insights,
                'social': social_insights,
                'temporal': temporal_insights,
                'generated_at': datetime.now().isoformat()
            }

            await self._store_personal_insights(all_insights)

        except Exception as e:
            logger.error(f"❌ Failed to generate personal insights: {e}")

    async def _analyze_communication_patterns(self) -> Dict[str, Any]:
        """Analyze communication patterns from email and calendar data"""

        conn = await asyncpg.connect(self.db_url)

        try:
            # Email frequency analysis
            email_stats = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total_emails,
                    COUNT(DISTINCT sender) as unique_senders,
                    AVG(importance_score) as avg_importance,
                    COUNT(CASE WHEN sentiment_score > 0.3 THEN 1 END) as positive_emails,
                    COUNT(CASE WHEN sentiment_score < -0.3 THEN 1 END) as negative_emails
                FROM email_intelligence
                WHERE timestamp > CURRENT_DATE - INTERVAL '30 days'
            """)

            # Top communicators
            top_communicators = await conn.fetch("""
                SELECT sender, COUNT(*) as email_count, AVG(importance_score) as avg_importance
                FROM email_intelligence
                WHERE timestamp > CURRENT_DATE - INTERVAL '30 days'
                GROUP BY sender
                ORDER BY email_count DESC
                LIMIT 10
            """)

            return {
                'email_statistics': dict(email_stats) if email_stats else {},
                'top_communicators': [dict(row) for row in top_communicators],
                'analysis_period': '30_days'
            }

        finally:
            await conn.close()

    async def _analyze_behavior_patterns(self) -> Dict[str, Any]:
        """Analyze behavioral patterns from browsing and app usage"""

        conn = await asyncpg.connect(self.db_url)

        try:
            # Browsing patterns
            browsing_stats = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total_visits,
                    COUNT(DISTINCT domain) as unique_domains,
                    AVG(interest_score) as avg_interest,
                    COUNT(DISTINCT category) as categories_visited
                FROM browsing_intelligence
                WHERE visit_time > CURRENT_DATE - INTERVAL '7 days'
            """)

            # Top categories
            top_categories = await conn.fetch("""
                SELECT category, COUNT(*) as visit_count, AVG(interest_score) as avg_interest
                FROM browsing_intelligence
                WHERE visit_time > CURRENT_DATE - INTERVAL '7 days'
                GROUP BY category
                ORDER BY visit_count DESC
                LIMIT 10
            """)

            return {
                'browsing_statistics': dict(browsing_stats) if browsing_stats else {},
                'top_categories': [dict(row) for row in top_categories],
                'analysis_period': '7_days'
            }

        finally:
            await conn.close()

    async def _analyze_social_patterns(self) -> Dict[str, Any]:
        """Analyze social interaction patterns"""

        # Implementation for social pattern analysis
        return {
            'social_network_size': 0,
            'interaction_frequency': 0,
            'relationship_strengths': []
        }

    async def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns in all data"""

        # Implementation for temporal pattern analysis
        return {
            'active_hours': [],
            'weekly_patterns': {},
            'seasonal_trends': {}
        }

    async def _store_personal_insights(self, insights: Dict[str, Any]):
        """Store generated personal insights"""

        conn = await asyncpg.connect(self.db_url)

        try:
            await conn.execute("""
                INSERT INTO omniscient_data
                (source, item_type, content_hash, content, metadata, timestamp, importance_score, privacy_level)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, 'echo_brain', 'personal_insights',
                str(hash(json.dumps(insights, sort_keys=True))),
                insights, {'generated_by': 'echo_brain_integration'},
                datetime.now(), 1.0, 'private')

        finally:
            await conn.close()

    async def query_omniscient_knowledge(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Query omniscient knowledge using semantic search"""

        try:
            # Generate query embedding
            query_embedding = await self.vector_store.generate_embedding(query)

            # Semantic search in vector store
            similar_items = await self.vector_store.similarity_search(
                query_embedding, limit=limit, threshold=0.7
            )

            # Enhance results with knowledge graph context
            enhanced_results = []
            for item in similar_items:
                if 'omniscient_id' in item.get('metadata', {}):
                    omniscient_id = item['metadata']['omniscient_id']

                    # Get knowledge graph relationships
                    relationships = await self.knowledge_graph.get_entity_relationships(
                        omniscient_id, 'omniscient_data'
                    )

                    enhanced_item = {
                        **item,
                        'knowledge_relationships': relationships
                    }
                    enhanced_results.append(enhanced_item)

            return enhanced_results

        except Exception as e:
            logger.error(f"❌ Failed to query omniscient knowledge: {e}")
            return []

    async def get_personal_context(self, context_type: str = 'recent') -> Dict[str, Any]:
        """Get personal context for Echo Brain conversations"""

        try:
            if context_type == 'recent':
                return await self._get_recent_context()
            elif context_type == 'important':
                return await self._get_important_context()
            elif context_type == 'social':
                return await self._get_social_context()
            else:
                return await self._get_general_context()

        except Exception as e:
            logger.error(f"❌ Failed to get personal context: {e}")
            return {}

    async def _get_recent_context(self) -> Dict[str, Any]:
        """Get recent personal activity context"""

        conn = await asyncpg.connect(self.db_url)

        try:
            recent_items = await conn.fetch("""
                SELECT source, item_type, content, timestamp, importance_score
                FROM omniscient_data
                WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '24 hours'
                  AND importance_score > 0.6
                ORDER BY timestamp DESC
                LIMIT 20
            """)

            return {
                'context_type': 'recent_activity',
                'time_window': '24_hours',
                'items': [dict(row) for row in recent_items],
                'summary': f"Recent activity includes {len(recent_items)} significant items"
            }

        finally:
            await conn.close()

if __name__ == "__main__":
    # Example usage
    import yaml

    # Load configuration
    config_path = Path(__file__).parent / "config" / "settings.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    integration = EchoBrainOmniscientIntegration(config)

    async def main():
        await integration.initialize()
        await integration.process_omniscient_data_stream()

    asyncio.run(main())