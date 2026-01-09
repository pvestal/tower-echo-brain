#!/usr/bin/env python3
"""
Persona-Driven Threshold Engine (Enhanced - October 2025)
Improved complexity scoring algorithm with proper task type detection
Designed in collaboration with deepseek-coder and qwen2.5-coder on Tower
"""

from src.core.complexity_analyzer import ComplexityAnalyzer
import asyncio
import asyncpg
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class PersonaThresholdEngine:
    """Bridges agenticPersona DB with dynamic escalation"""
    
    def __init__(self, db_url: str = "postgresql://patrick@localhost/echo_brain"):
        self.db_url = db_url
        self.pool = None
        self.persona_cache = {}
        self.threshold_cache = {}
        
    async def initialize(self):
        """Initialize database connection pool"""
        self.pool = await asyncpg.create_pool(self.db_url, min_size=2, max_size=10)
        await self.load_persona()
        await self.load_thresholds()
        
    async def load_persona(self) -> Dict:
        """Load current persona from DB (JSONB traits)"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM echo_persona ORDER BY last_updated DESC LIMIT 1")
            if row:
                # Extract traits from JSONB column
                traits = json.loads(row['traits']) if isinstance(row['traits'], str) else row['traits']
                self.persona_cache = {
                    'id': row['id'],
                    'performance_score': row['performance_score'],
                    **traits  # Unpack JSONB traits
                }
                logger.info(f"Loaded persona with {len(traits)} traits")
        return self.persona_cache
        
    async def load_thresholds(self) -> Dict:
        """Load complexity thresholds from DB"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM complexity_thresholds ORDER BY min_score")
            self.threshold_cache = {}
            for row in rows:
                self.threshold_cache[row['tier']] = {
                    'min_score': row['min_score'],
                    'max_score': row['max_score'],
                    'model_name': row['model_name'],
                    'timeout_seconds': row['timeout_seconds'],
                    'auto_escalate': row['auto_escalate']
                }
            logger.info(f"Loaded {len(self.threshold_cache)} threshold tiers")
        return self.threshold_cache
    
    def calculate_complexity_score(self, message: str) -> float:
        """
        Calculate complexity score
        REFACTORED: Now delegates to ComplexityAnalyzer (Oct 22, 2025)
        """
        result = ComplexityAnalyzer.analyze(message)
        return result.score

    
    async def select_tier(self, message: str, context: Dict = None) -> Tuple[str, Dict]:
        """Dynamically select tier based on message + persona"""
        
        # Calculate complexity score using improved algorithm
        complexity_score = self.calculate_complexity_score(message)
        
        # Persona factors (from JSONB traits)
        technical_precision = self.persona_cache.get('technical_precision', 0.5)
        risk_taking = self.persona_cache.get('risk_taking', 0.5)
        
        # Adjust score based on persona
        if technical_precision > 0.7:
            complexity_score *= 1.2
        if risk_taking < 0.3:
            complexity_score *= 0.9
        
        # Special case: explicit escalation request
        if 'think harder' in message.lower() or 'use 70b' in message.lower():
            tier = 'cloud'
        else:
            # Find matching tier by score
            tier = 'small'  # default
            for tier_name, config in self.threshold_cache.items():
                if config['min_score'] <= complexity_score < config['max_score']:
                    tier = tier_name
                    break
        
        config = self.threshold_cache.get(tier, self.threshold_cache.get('small', {}))
        
        # Extract score components for logging
        message_lower = message.lower()
        word_count = len(message.split())
        questions = message.count("?")
        gen_count = sum(1 for kw in ["generate", "create", "make", "render"] if kw in message_lower)
        media_count = sum(1 for kw in ["video", "anime", "image", "animation"] if kw in message_lower)
        
        logger.info(
            f"Selected tier '{tier}' (score={complexity_score:.1f}, "
            f"words={word_count}, questions={questions}, gen={gen_count}, media={media_count})"
        )
        
        return tier, config
        
    async def update_from_feedback(self, tier_used: str, success: bool, quality: float):
        """Update persona based on feedback"""
        async with self.pool.acquire() as conn:
            # Load current traits
            row = await conn.fetchrow("SELECT traits FROM echo_persona ORDER BY last_updated DESC LIMIT 1")
            if row:
                traits = json.loads(row['traits']) if isinstance(row['traits'], str) else row['traits']
                
                # Adjust learning_rate based on success
                if success and quality > 0.8:
                    traits['learning_rate'] = min(1.0, traits.get('learning_rate', 0.5) * 1.05)
                elif not success or quality < 0.5:
                    traits['technical_precision'] = min(1.0, traits.get('technical_precision', 0.5) + 0.1)
                
                # Update database
                await conn.execute(
                    "UPDATE echo_persona SET traits = $1, last_updated = $2 WHERE id = (SELECT id FROM echo_persona ORDER BY last_updated DESC LIMIT 1)",
                    json.dumps(traits), datetime.now()
                )
        await self.load_persona()
        
    async def close(self):
        if self.pool:
            await self.pool.close()
