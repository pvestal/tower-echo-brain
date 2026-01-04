#!/usr/bin/env python3
"""
Persona-Driven Threshold Engine
Reads agenticPersona DB tables and dynamically adjusts escalation thresholds
"""

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
        """Load current persona from DB"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM echo_persona ORDER BY updated_at DESC LIMIT 1")
            if rows:
                row = rows[0]
                self.persona_cache = dict(row)
                logger.info(f"Loaded persona with {len(self.persona_cache)} traits")
        return self.persona_cache
        
    async def load_thresholds(self) -> Dict:
        """Load complexity thresholds from DB"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM complexity_thresholds ORDER BY tier_level")
            self.threshold_cache = {row['tier_name']: dict(row) for row in rows}
            logger.info(f"Loaded {len(self.threshold_cache)} threshold tiers")
        return self.threshold_cache
        
    async def select_tier(self, message: str, context: Dict = None) -> Tuple[str, Dict]:
        """Dynamically select tier based on message + persona"""
        # Message analysis
        word_count = len(message.split())
        questions = message.count('?')
        code_markers = sum(1 for kw in ['def ', 'class ', 'import ', 'function', 'implement'] if kw in message.lower())
        
        # Persona factors
        technical_depth = self.persona_cache.get('technical_depth', 5)
        
        # Tier selection
        if 'think harder' in message.lower():
            tier = 'genius'
        elif code_markers >= 2 or 'implement' in message.lower():
            tier = 'expert'  
        elif questions >= 3 or word_count > 50:
            tier = 'expert' if technical_depth >= 7 else 'standard'
        elif word_count < 10:
            tier = 'quick'
        else:
            tier = 'standard'
            
        config = self.threshold_cache.get(tier, self.threshold_cache.get('standard', {}))
        logger.info(f"Selected tier '{tier}' (words={word_count}, questions={questions}, code={code_markers})")
        
        return tier, config
        
    async def update_from_feedback(self, tier_used: str, success: bool, quality: float):
        """Update persona based on feedback"""
        async with self.pool.acquire() as conn:
            if success and quality > 0.8:
                await conn.execute(
                    "UPDATE echo_persona SET learning_rate = learning_rate * 1.05, updated_at = $1 WHERE id = (SELECT id FROM echo_persona ORDER BY updated_at DESC LIMIT 1)",
                    datetime.now()
                )
            elif not success or quality < 0.5:
                await conn.execute(
                    "UPDATE echo_persona SET technical_depth = LEAST(10, technical_depth + 1), updated_at = $1 WHERE id = (SELECT id FROM echo_persona ORDER BY updated_at DESC LIMIT 1)",
                    datetime.now()
                )
        await self.load_persona()
        
    async def close(self):
        if self.pool:
            await self.pool.close()
