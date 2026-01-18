#!/usr/bin/env python3
"""
Knowledge Graph Builder
Aggregates Takeout insights into a comprehensive knowledge graph
Updates Echo's persona based on discovered patterns
"""

import asyncio
import asyncpg
import json
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    def __init__(self):
        self.db_url = "postgresql://patrick:RP78eIrW7cI2jYvL5akt1yurE@localhost/echo_brain"

    async def build_relationship_graph(self, conn) -> Dict[str, Any]:
        """Build relationships between people, locations, and events"""
        logger.info("🔗 Building relationship graph...")

        # Get all people and their co-occurrences
        people_insights = await conn.fetch("""
            SELECT entity_name, entity_value, context
            FROM takeout_insights
            WHERE insight_type = 'person'
            AND entity_name IS NOT NULL
            ORDER BY extracted_at DESC
            LIMIT 10000
        """)

        # Build co-occurrence matrix (people who appear together)
        co_occurrences = defaultdict(lambda: defaultdict(int))

        for insight in people_insights:
            context = insight['context']
            entity_name = insight['entity_name']

            # Find other people in same context
            related = await conn.fetch("""
                SELECT DISTINCT entity_name
                FROM takeout_insights
                WHERE insight_type = 'person'
                AND context LIKE $1
                AND entity_name != $2
                LIMIT 10
            """, f"%{context[:50]}%", entity_name)

            for related_person in related:
                co_occurrences[entity_name][related_person['entity_name']] += 1

        logger.info(f"✅ Found {len(co_occurrences)} people with relationships")

        # Get location visit patterns
        location_visits = await conn.fetch("""
            SELECT l.name, l.latitude, l.longitude, l.visit_count, l.first_visited, l.last_visited
            FROM takeout_locations l
            ORDER BY l.visit_count DESC
            LIMIT 100
        """)

        # Identify significant locations (visited multiple times)
        significant_locations = [
            loc for loc in location_visits
            if loc['visit_count'] > 5 or
            (loc['last_visited'] and loc['first_visited'] and
             (loc['last_visited'] - loc['first_visited']).days > 30)
        ]

        logger.info(f"✅ Identified {len(significant_locations)} significant locations")

        return {
            'people_relationships': dict(co_occurrences),
            'significant_locations': [dict(loc) for loc in significant_locations],
            'relationship_count': sum(len(rels) for rels in co_occurrences.values())
        }

    async def identify_timeline_events(self, conn) -> List[Dict[str, Any]]:
        """Identify significant events from timeline"""
        logger.info("📅 Identifying timeline events...")

        # Group photos by date and location
        photo_clusters = await conn.fetch("""
            SELECT
                DATE(to_timestamp(CAST(entity_value->>'timestamp' AS BIGINT))) as event_date,
                entity_value->>'location' as location,
                COUNT(*) as photo_count,
                json_agg(entity_name) as people_involved
            FROM takeout_insights
            WHERE insight_type = 'location'
            AND entity_value->>'timestamp' IS NOT NULL
            GROUP BY event_date, location
            HAVING COUNT(*) > 5
            ORDER BY event_date DESC
            LIMIT 100
        """)

        events = []
        for cluster in photo_clusters:
            event_type = 'trip' if cluster['photo_count'] > 20 else 'outing'

            # Try to infer event type from location
            location = cluster['location'] or ''
            if any(word in location.lower() for word in ['beach', 'park', 'resort', 'hotel']):
                event_type = 'vacation'
            elif any(word in location.lower() for word in ['restaurant', 'cafe', 'bar']):
                event_type = 'social'

            events.append({
                'event_date': cluster['event_date'],
                'location': location,
                'photo_count': cluster['photo_count'],
                'event_type': event_type,
                'description': f"{event_type.title()} at {location or 'unknown location'}"
            })

            # Insert into events table
            await conn.execute("""
                INSERT INTO takeout_events (event_date, event_name, event_type, description, location, photo_count)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT DO NOTHING
            """, cluster['event_date'], f"{event_type.title()} - {location or 'Unknown'}",
                event_type, f"{cluster['photo_count']} photos taken",
                location, cluster['photo_count'])

        logger.info(f"✅ Identified {len(events)} significant events")
        return events

    async def aggregate_topic_patterns(self, conn) -> Dict[str, Any]:
        """Aggregate topic mentions and categorize interests"""
        logger.info("🏷️ Aggregating topic patterns...")

        # Get all topics from insights
        topics = await conn.fetch("""
            SELECT
                entity_name,
                entity_value,
                COUNT(*) as mention_count,
                MIN(extracted_at) as first_mention,
                MAX(extracted_at) as last_mention
            FROM takeout_insights
            WHERE insight_type = 'topic'
            AND entity_name IS NOT NULL
            GROUP BY entity_name, entity_value
            ORDER BY mention_count DESC
            LIMIT 100
        """)

        # Categorize topics
        categories = {
            'technical': ['api', 'code', 'python', 'javascript', 'database', 'server', 'docker', 'git'],
            'ai_ml': ['ai', 'ml', 'llm', 'echo', 'tower', 'model', 'neural'],
            'creative': ['anime', 'music', 'art', 'video', 'photo', 'design'],
            'personal': ['family', 'home', 'travel', 'food', 'health'],
            'finance': ['money', 'bank', 'crypto', 'trading', 'investment']
        }

        categorized_topics = defaultdict(list)

        for topic in topics:
            topic_name = topic['entity_name'].lower()
            category = 'other'

            for cat, keywords in categories.items():
                if any(keyword in topic_name for keyword in keywords):
                    category = cat
                    break

            categorized_topics[category].append({
                'name': topic['entity_name'],
                'mentions': topic['mention_count'],
                'first_mention': topic['first_mention'],
                'last_mention': topic['last_mention']
            })

            # Update topics table
            await conn.execute("""
                INSERT INTO takeout_topics (topic_name, category, mention_count, first_mention, last_mention)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (topic_name) DO UPDATE
                SET mention_count = takeout_topics.mention_count + $3,
                    last_mention = GREATEST(takeout_topics.last_mention, $5)
            """, topic['entity_name'], category, topic['mention_count'],
                topic['first_mention'].date(), topic['last_mention'].date())

        logger.info(f"✅ Categorized {sum(len(v) for v in categorized_topics.values())} topics")
        return dict(categorized_topics)

    async def update_persona_from_graph(self, conn, graph_data: Dict[str, Any]):
        """Update Echo's persona based on knowledge graph insights"""
        logger.info("🧠 Updating Echo's persona from knowledge graph...")

        # Get current persona
        current_persona = await conn.fetchrow('SELECT * FROM echo_persona ORDER BY last_updated DESC LIMIT 1')
        current_traits = current_persona['traits']
        new_traits = current_traits.copy() if isinstance(current_traits, dict) else dict(current_traits)

        # Analyze relationship patterns
        relationship_count = graph_data.get('relationship_count', 0)
        if relationship_count > 50:
            # High social connectivity
            new_traits['social_awareness'] = min(1.0, new_traits.get('social_awareness', 0.5) + 0.15)

        # Analyze location patterns
        significant_locations = len(graph_data.get('significant_locations', []))
        if significant_locations > 10:
            # Well-traveled
            new_traits['world_knowledge'] = min(1.0, new_traits.get('world_knowledge', 0.5) + 0.1)

        # Analyze communication patterns
        comm_patterns = await conn.fetchval('SELECT COUNT(*) FROM takeout_communication_patterns')
        if comm_patterns > 100:
            new_traits['communication_style_understanding'] = min(1.0, new_traits.get('communication_style_understanding', 0.5) + 0.1)

        # Update persona
        new_score = current_persona['performance_score'] + 0.3  # Major boost from graph analysis

        await conn.execute("""
            UPDATE echo_persona
            SET traits = $1,
                performance_score = $2,
                generation_count = generation_count + 1,
                last_updated = CURRENT_TIMESTAMP
            WHERE id = $3
        """, json.dumps(new_traits), new_score, current_persona['id'])

        # Record training history
        trait_updates = {
            'social_awareness': new_traits.get('social_awareness', 0) - current_traits.get('social_awareness', 0),
            'world_knowledge': new_traits.get('world_knowledge', 0) - current_traits.get('world_knowledge', 0),
            'communication_style_understanding': new_traits.get('communication_style_understanding', 0) - current_traits.get('communication_style_understanding', 0)
        }

        await conn.execute("""
            INSERT INTO persona_training_history
            (interaction_id, trait_updates, performance_delta, feedback_type, context)
            VALUES (NULL, $1, $2, $3, $4)
        """, json.dumps(trait_updates), 0.3, 'knowledge_graph_analysis',
            f"Analyzed {relationship_count} relationships, {significant_locations} locations")

        logger.info(f"✅ Persona updated! Performance score: {current_persona['performance_score']:.2f} → {new_score:.2f}")

        return new_traits

    async def run(self):
        """Build complete knowledge graph"""
        logger.info("🚀 Starting Knowledge Graph Builder")

        conn = await asyncpg.connect(self.db_url)

        try:
            # Build relationship graph
            graph_data = await self.build_relationship_graph(conn)

            # Identify timeline events
            events = await self.identify_timeline_events(conn)
            graph_data['events'] = events

            # Aggregate topic patterns
            topics = await self.aggregate_topic_patterns(conn)
            graph_data['topics'] = topics

            # Update Echo's persona
            new_traits = await self.update_persona_from_graph(conn, graph_data)

            # Summary
            logger.info("\n=== KNOWLEDGE GRAPH SUMMARY ===")
            logger.info(f"Relationships: {graph_data['relationship_count']}")
            logger.info(f"Significant Locations: {len(graph_data['significant_locations'])}")
            logger.info(f"Timeline Events: {len(events)}")
            logger.info(f"Topic Categories: {len(topics)}")
            logger.info(f"\nPersona Traits Updated:")
            for trait, value in sorted(new_traits.items()):
                logger.info(f"  {trait}: {value}")

            return graph_data

        finally:
            await conn.close()


async def main():
    builder = KnowledgeGraphBuilder()
    await builder.run()


if __name__ == '__main__':
    asyncio.run(main())
