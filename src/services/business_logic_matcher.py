#!/usr/bin/env python3
"""
Business Logic Pattern Matcher for Echo Brain
Matches queries against Patrick's learned business logic patterns (pattern retrieval only)
"""

import psycopg2
import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class BusinessLogicPatternMatcher:
    """Matches queries against Patrick's learned business logic patterns"""

    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'echo_brain',
            'user': os.getenv('DB_USER', 'echo_brain_app'),
            'password': os.getenv("TOWER_DB_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE")
        }
        self._cached_patterns = None
        self._cache_timestamp = None

    def get_relevant_patterns(self, query: str) -> List[Dict]:
        """Get business logic patterns relevant to the query"""
        patterns = self._get_cached_patterns()
        relevant = []

        query_lower = query.lower()

        for pattern in patterns:
            # Check if pattern applies to this query
            if self._pattern_matches_query(pattern, query_lower):
                relevant.append(pattern)

        # Sort by confidence/weight
        relevant.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        return relevant[:3]  # Top 3 most relevant

    def _pattern_matches_query(self, pattern: Dict, query_lower: str) -> bool:
        """Check if a pattern is relevant to the query"""
        fact_type = pattern.get('fact_type', '')
        learned_fact = pattern.get('learned_fact', '').lower()
        metadata = pattern.get('metadata', {})

        # Technical stack preferences
        if fact_type == 'technical_stack_preferences':
            tech_keywords = ['database', 'db', 'mysql', 'postgresql', 'postgres', 'frontend', 'vue', 'react', 'backend', 'python', 'node']
            if any(keyword in query_lower for keyword in tech_keywords):
                return True

        # Quality standards
        if fact_type == 'quality_standards':
            quality_keywords = ['working', 'fixed', 'done', 'complete', 'ready', 'test', 'verify']
            if any(keyword in query_lower for keyword in quality_keywords):
                return True

        # Communication patterns
        if fact_type == 'communication_patterns':
            frustration_keywords = ['fucking', 'broken', 'lying', 'bullshit', 'wrong']
            if any(keyword in query_lower for keyword in frustration_keywords):
                return True

        # Project priorities
        if fact_type == 'project_priorities':
            project_keywords = ['anime', 'production', 'echo', 'brain', 'tower', 'service']
            if any(keyword in query_lower for keyword in project_keywords):
                return True

        # Naming standards
        if fact_type == 'naming_standards':
            naming_keywords = ['name', 'file', 'enhanced', 'improved', 'unified', 'complete']
            if any(keyword in query_lower for keyword in naming_keywords):
                return True

        return False

    def _get_cached_patterns(self) -> List[Dict]:
        """Get patterns from cache or database"""
        now = datetime.now()

        # Cache for 5 minutes
        if (self._cached_patterns and self._cache_timestamp and
            (now - self._cache_timestamp).seconds < 300):
            return self._cached_patterns

        # Fetch from database
        try:
            db = psycopg2.connect(**self.db_config)
            cursor = db.cursor()

            cursor.execute("""
                SELECT fact_type, learned_fact, confidence, metadata
                FROM learning_history
                WHERE metadata->>'source' = 'patrick_conversation_analysis'
                AND created_at > NOW() - INTERVAL '24 hours'
                ORDER BY confidence DESC
            """)

            patterns = []
            for row in cursor.fetchall():
                patterns.append({
                    'fact_type': row[0],
                    'learned_fact': row[1],
                    'confidence': row[2] or 0.5,
                    'metadata': row[3] or {}
                })

            self._cached_patterns = patterns
            self._cache_timestamp = now

            db.close()
            logger.info(f"Loaded {len(patterns)} business logic patterns")
            return patterns

        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
            return []

    def transform_patterns_for_application(self, patterns: List[Dict]) -> List[Dict]:
        """
        Transform database patterns into format expected by BusinessLogicApplicator.
        Maps fact_type -> type, learned_fact -> business_logic, etc.
        """
        transformed = []

        for pattern in patterns:
            fact_type = pattern.get('fact_type', '')
            learned_fact = pattern.get('learned_fact', '')
            confidence = pattern.get('confidence', 0.5)
            metadata = pattern.get('metadata', {})

            # Map fact types to applicator pattern types
            pattern_type = self._map_fact_type_to_pattern_type(fact_type)
            trigger = self._extract_trigger_from_fact(learned_fact, fact_type)

            transformed_pattern = {
                'type': pattern_type,
                'trigger': trigger,
                'business_logic': learned_fact,
                'confidence': confidence,
                'metadata': metadata,
                'original_fact_type': fact_type  # Keep original for debugging
            }

            transformed.append(transformed_pattern)

        return transformed

    def _map_fact_type_to_pattern_type(self, fact_type: str) -> str:
        """Map database fact types to applicator pattern types"""
        mapping = {
            'technical_stack_preferences': 'preference',
            'quality_standards': 'requirement',
            'communication_patterns': 'anti_pattern',
            'project_priorities': 'context',
            'naming_standards': 'anti_pattern'
        }
        return mapping.get(fact_type, 'context')

    def _extract_trigger_from_fact(self, learned_fact: str, fact_type: str) -> str:
        """Extract trigger keywords from learned fact based on type"""
        fact_lower = learned_fact.lower()

        if fact_type == 'technical_stack_preferences':
            if 'postgresql' in fact_lower:
                return 'database'
            elif 'vue.js' in fact_lower:
                return 'frontend'
        elif fact_type == 'quality_standards':
            if 'proof' in fact_lower:
                return 'working'
        elif fact_type == 'project_priorities':
            if 'anime' in fact_lower:
                return 'anime'
        elif fact_type == 'naming_standards':
            if 'promotional' in fact_lower:
                return 'enhanced'

        return ''  # Default empty trigger

    def get_pattern_stats(self) -> Dict:
        """Get statistics about loaded patterns"""
        patterns = self._get_cached_patterns()

        stats = {
            'total_patterns': len(patterns),
            'by_type': {},
            'high_confidence': 0
        }

        for pattern in patterns:
            fact_type = pattern.get('fact_type')
            confidence = pattern.get('confidence', 0)

            stats['by_type'][fact_type] = stats['by_type'].get(fact_type, 0) + 1
            if confidence > 0.8:
                stats['high_confidence'] += 1

        return stats