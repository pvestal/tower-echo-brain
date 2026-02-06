#!/usr/bin/env python3
"""
Knowledge Graph Builder - Builds relationships between facts
"""

import logging
import asyncpg
from datetime import datetime
from typing import List, Dict, Tuple
import asyncio
import os
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    """Builds a knowledge graph from extracted facts"""

    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL", "postgresql://patrick:RP78eIrW7cI2jYvL5akt1yurE@localhost/echo_brain")

    async def register_with_autonomous_core(self):
        """Register as an autonomous goal"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8309/api/autonomous/goals",
                    json={
                        "name": "knowledge_graph_build",
                        "description": "Build knowledge graph relationships",
                        "safety_level": "auto",
                        "schedule": "0 2 * * *",  # Daily at 2 AM
                        "enabled": True
                    }
                )
                logger.info(f"Registered with autonomous core: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to register: {e}")

    async def get_facts(self, conn) -> List[Dict]:
        """Get all facts from database"""
        rows = await conn.fetch(
            """SELECT id, subject, predicate, object, confidence, source
               FROM facts
               ORDER BY id"""
        )
        return [dict(row) for row in rows]

    async def find_same_subject_edges(self, facts: List[Dict]) -> List[Tuple]:
        """Find facts with the same subject"""
        edges = []
        subject_map = {}

        # Group facts by subject
        for fact in facts:
            subject = fact['subject'].lower().strip()
            if subject not in subject_map:
                subject_map[subject] = []
            subject_map[subject].append(fact)

        # Create edges between facts with same subject
        for subject, fact_list in subject_map.items():
            if len(fact_list) > 1:
                for i in range(len(fact_list)):
                    for j in range(i + 1, len(fact_list)):
                        edges.append((
                            fact_list[i]['id'],
                            fact_list[j]['id'],
                            'same_subject',
                            0.9
                        ))

        logger.info(f"Found {len(edges)} same-subject edges")
        return edges

    async def find_object_to_subject_edges(self, facts: List[Dict]) -> List[Tuple]:
        """Find facts where object of one matches subject of another"""
        edges = []

        # Create lookups
        subject_lookup = {fact['subject'].lower().strip(): fact for fact in facts}

        for fact in facts:
            object_text = fact['object'].lower().strip()

            # Check if this object is another fact's subject
            if object_text in subject_lookup:
                target_fact = subject_lookup[object_text]
                if target_fact['id'] != fact['id']:
                    edges.append((
                        fact['id'],
                        target_fact['id'],
                        'object_to_subject',
                        0.85
                    ))

        logger.info(f"Found {len(edges)} object-to-subject edges")
        return edges

    async def find_co_located_edges(self, facts: List[Dict]) -> List[Tuple]:
        """Find facts from the same source"""
        edges = []
        source_map = {}

        # Group facts by source
        for fact in facts:
            source = fact.get('source', 'unknown')
            if source not in source_map:
                source_map[source] = []
            source_map[source].append(fact)

        # Create edges between facts from same source
        for source, fact_list in source_map.items():
            if len(fact_list) > 1:
                # Limit edges per source to avoid explosion
                for i in range(min(len(fact_list), 20)):
                    for j in range(i + 1, min(len(fact_list), 20)):
                        edges.append((
                            fact_list[i]['id'],
                            fact_list[j]['id'],
                            'co_located',
                            0.7
                        ))

        logger.info(f"Found {len(edges)} co-located edges")
        return edges

    async def find_semantic_edges(self, facts: List[Dict]) -> List[Tuple]:
        """Find semantically related facts"""
        edges = []

        # Keywords that indicate relationships
        relationship_keywords = {
            'tower': ['server', 'service', 'port', 'system'],
            'patrick': ['drives', 'owns', 'likes', 'uses'],
            'echo brain': ['ai', 'assistant', 'system', 'fastapi'],
            'anime': ['production', 'comfyui', 'lora', 'tdd', 'cgs'],
            'database': ['postgresql', 'qdrant', 'table', 'schema'],
        }

        for i, fact1 in enumerate(facts):
            fact1_text = f"{fact1['subject']} {fact1['predicate']} {fact1['object']}".lower()

            for j, fact2 in enumerate(facts[i+1:], i+1):
                if j >= len(facts):
                    break

                fact2_text = f"{fact2['subject']} {fact2['predicate']} {fact2['object']}".lower()

                # Check for semantic relationships
                for topic, keywords in relationship_keywords.items():
                    if topic in fact1_text and any(kw in fact2_text for kw in keywords):
                        edges.append((
                            fact1['id'],
                            fact2['id'],
                            'semantic',
                            0.6
                        ))
                        break

        logger.info(f"Found {len(edges)} semantic edges")
        return edges[:500]  # Limit to prevent explosion

    async def store_edges(self, conn, edges: List[Tuple]):
        """Store graph edges in database"""
        stored = 0

        for source_id, target_id, rel_type, confidence in edges:
            try:
                await conn.execute(
                    """INSERT INTO graph_edges
                       (source_fact_id, target_fact_id, relationship_type, confidence)
                       VALUES ($1, $2, $3, $4)
                       ON CONFLICT DO NOTHING""",
                    source_id, target_id, rel_type, confidence
                )
                stored += 1
            except Exception as e:
                logger.error(f"Error storing edge: {e}")

        return stored

    async def prune_old_edges(self, conn):
        """Remove old edges before rebuilding"""
        deleted = await conn.fetchval(
            """DELETE FROM graph_edges
               WHERE created_at < NOW() - INTERVAL '7 days'
               RETURNING COUNT(*)"""
        )
        logger.info(f"Pruned {deleted} old edges")

    async def run_cycle(self):
        """Run one graph building cycle"""
        try:
            conn = await asyncpg.connect(self.db_url)

            # Get all facts
            facts = await self.get_facts(conn)
            logger.info(f"Building graph from {len(facts)} facts")

            if not facts:
                logger.info("No facts to build graph from")
                await conn.close()
                return

            # Prune old edges
            await self.prune_old_edges(conn)

            # Find different types of edges
            all_edges = []

            # Same subject relationships
            same_subject = await self.find_same_subject_edges(facts)
            all_edges.extend(same_subject)

            # Object-to-subject relationships
            obj_to_subj = await self.find_object_to_subject_edges(facts)
            all_edges.extend(obj_to_subj)

            # Co-location relationships
            co_located = await self.find_co_located_edges(facts)
            all_edges.extend(co_located)

            # Semantic relationships
            semantic = await self.find_semantic_edges(facts)
            all_edges.extend(semantic)

            # Store all edges
            if all_edges:
                logger.info(f"Storing {len(all_edges)} total edges")
                stored = await self.store_edges(conn, all_edges)
                logger.info(f"Successfully stored {stored} edges")

            # Get final stats
            stats = await conn.fetchrow(
                """SELECT COUNT(*) as total_edges,
                          COUNT(DISTINCT source_fact_id) as connected_facts
                   FROM graph_edges"""
            )
            logger.info(f"Graph stats: {stats['total_edges']} edges connecting {stats['connected_facts']} facts")

            await conn.close()

        except Exception as e:
            logger.error(f"Error in graph building cycle: {e}")

    async def start(self):
        """Start the builder"""
        await self.register_with_autonomous_core()

        while True:
            try:
                await self.run_cycle()
                await asyncio.sleep(86400)  # Wait 24 hours
            except KeyboardInterrupt:
                logger.info("Stopping knowledge graph builder")
                break
            except Exception as e:
                logger.error(f"Builder error: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error

if __name__ == "__main__":
    builder = KnowledgeGraphBuilder()
    asyncio.run(builder.start())