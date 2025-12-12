#!/usr/bin/env python3
"""
Agentic Persona Self-Training System for Echo Brain
Enables continuous learning and personality evolution
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncpg
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class PersonaTrainer:
    """Trains and evolves Echo's Agentic Persona based on interactions"""

    def __init__(self, db_url: str = "postgresql://patrick:***REMOVED***@localhost/echo_brain"):
        self.db_url = db_url
        self.pool = None
        self.current_persona = {}
        self.learning_rate = 0.01
        self.trait_weights = {
            'helpfulness': 1.0,
            'technical_accuracy': 1.5,
            'proactiveness': 2.0,  # Patrick wants more proactive behavior
            'creativity': 1.2,
            'efficiency': 1.3,
            'humor': 0.5,
            'formality': 0.7,
            'verbosity': -0.5,  # Negative weight for conciseness
            'autonomy': 2.5  # Highest weight for autonomous behavior
        }

    async def initialize(self):
        """Initialize database connection and load current persona"""
        self.pool = await asyncpg.create_pool(self.db_url, min_size=2, max_size=10)
        await self.ensure_tables()
        await self.load_current_persona()

    async def ensure_tables(self):
        """Ensure persona training tables exist"""
        async with self.pool.acquire() as conn:
            # Create training history table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS persona_training_history (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    interaction_id INTEGER,
                    trait_updates JSONB,
                    performance_delta FLOAT,
                    feedback_type TEXT,
                    context TEXT
                )
            """)

            # Create learning patterns table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS persona_learning_patterns (
                    id SERIAL PRIMARY KEY,
                    pattern_type TEXT,
                    pattern_data JSONB,
                    frequency INTEGER DEFAULT 1,
                    success_rate FLOAT,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    async def load_current_persona(self):
        """Load the current persona from database"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM echo_persona
                ORDER BY last_updated DESC
                LIMIT 1
            """)

            if row:
                self.current_persona = {
                    'id': row['id'],
                    'traits': json.loads(row['traits']) if isinstance(row['traits'], str) else row['traits'],
                    'performance_score': row['performance_score'],
                    'generation_count': row['generation_count']
                }
            else:
                # Initialize default persona
                self.current_persona = await self.initialize_default_persona()

            logger.info(f"Loaded persona with performance score: {self.current_persona.get('performance_score', 0)}")

    async def initialize_default_persona(self) -> Dict:
        """Create a default persona focused on Patrick's preferences"""
        default_traits = {
            'helpfulness': 0.9,
            'technical_accuracy': 0.95,
            'proactiveness': 0.3,  # Start low, learn to be more proactive
            'creativity': 0.7,
            'efficiency': 0.8,
            'humor': 0.1,  # Minimal humor as requested
            'formality': 0.3,  # Casual, direct communication
            'verbosity': 0.2,  # Concise responses
            'autonomy': 0.4,  # Start moderate, increase over time
            'code_quality_focus': 0.9,
            'self_improvement': 0.8,
            'pattern_recognition': 0.7
        }

        async with self.pool.acquire() as conn:
            result = await conn.fetchrow("""
                INSERT INTO echo_persona (traits, performance_score, generation_count)
                VALUES ($1, $2, $3)
                RETURNING id
            """, json.dumps(default_traits), 0.5, 0)

            return {
                'id': result['id'],
                'traits': default_traits,
                'performance_score': 0.5,
                'generation_count': 0
            }

    async def train_from_interaction(self, interaction: Dict) -> Dict:
        """Train persona based on an interaction"""
        # Extract feedback signals
        feedback = self.extract_feedback_signals(interaction)

        # Calculate trait adjustments
        trait_updates = self.calculate_trait_updates(feedback)

        # Apply updates
        updated_persona = await self.apply_trait_updates(trait_updates)

        # Log training
        await self.log_training(interaction, trait_updates, feedback)

        return updated_persona

    def extract_feedback_signals(self, interaction: Dict) -> Dict:
        """Extract training signals from interaction"""
        feedback = {
            'positive': [],
            'negative': [],
            'implicit': []
        }

        message = interaction.get('message', '').lower()
        response = interaction.get('response', '').lower()

        # Positive signals
        if any(word in message for word in ['good', 'great', 'perfect', 'thanks', 'yes']):
            feedback['positive'].append('user_satisfaction')

        if 'more proactive' in message or 'be proactive' in message:
            feedback['negative'].append('insufficient_proactiveness')

        if 'too long' in message or 'be concise' in message:
            feedback['negative'].append('excessive_verbosity')

        # Implicit signals from Patrick's patterns
        if interaction.get('user') == 'patrick':
            # Patrick prefers direct, technical responses
            if len(response) < 500:
                feedback['implicit'].append('concise_response')

            if 'refactor' in message or 'improve' in message:
                feedback['implicit'].append('code_improvement_request')

        return feedback

    def calculate_trait_updates(self, feedback: Dict) -> Dict:
        """Calculate how traits should be updated based on feedback"""
        updates = {}

        # Process positive feedback
        for signal in feedback.get('positive', []):
            if signal == 'user_satisfaction':
                # Reinforce current behavior slightly
                updates['helpfulness'] = 0.01
                updates['technical_accuracy'] = 0.01

        # Process negative feedback
        for signal in feedback.get('negative', []):
            if signal == 'insufficient_proactiveness':
                updates['proactiveness'] = 0.05  # Increase significantly
                updates['autonomy'] = 0.03

            if signal == 'excessive_verbosity':
                updates['verbosity'] = -0.03  # Decrease
                updates['efficiency'] = 0.02

        # Process implicit feedback
        for signal in feedback.get('implicit', []):
            if signal == 'concise_response':
                updates['verbosity'] = -0.01

            if signal == 'code_improvement_request':
                updates['code_quality_focus'] = 0.02
                updates['proactiveness'] = 0.02

        return updates

    async def apply_trait_updates(self, updates: Dict) -> Dict:
        """Apply trait updates to current persona"""
        if not updates:
            return self.current_persona

        # Apply updates with learning rate
        for trait, delta in updates.items():
            if trait in self.current_persona['traits']:
                old_value = self.current_persona['traits'][trait]
                new_value = old_value + (delta * self.learning_rate)
                # Clamp between 0 and 1
                new_value = max(0.0, min(1.0, new_value))
                self.current_persona['traits'][trait] = new_value

                if abs(delta) > 0.02:  # Log significant changes
                    logger.info(f"Trait '{trait}' updated: {old_value:.3f} → {new_value:.3f}")

        # Update performance score
        self.current_persona['performance_score'] = self.calculate_performance_score()

        # Save to database
        await self.save_persona()

        return self.current_persona

    def calculate_performance_score(self) -> float:
        """Calculate overall performance score based on weighted traits"""
        score = 0.0
        total_weight = 0.0

        for trait, value in self.current_persona['traits'].items():
            weight = self.trait_weights.get(trait, 1.0)
            score += value * abs(weight)
            if weight < 0:  # Invert for negative weights
                score = score - value * abs(weight) * 2
            total_weight += abs(weight)

        return score / total_weight if total_weight > 0 else 0.5

    async def save_persona(self):
        """Save updated persona to database"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE echo_persona
                SET traits = $1,
                    performance_score = $2,
                    generation_count = generation_count + 1,
                    last_updated = CURRENT_TIMESTAMP
                WHERE id = $3
            """, json.dumps(self.current_persona['traits']),
                self.current_persona['performance_score'],
                self.current_persona['id'])

    async def log_training(self, interaction: Dict, updates: Dict, feedback: Dict):
        """Log training history"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO persona_training_history
                (interaction_id, trait_updates, performance_delta, feedback_type, context)
                VALUES ($1, $2, $3, $4, $5)
            """, interaction.get('id'), json.dumps(updates),
                self.current_persona['performance_score'] - 0.5,
                json.dumps(feedback), interaction.get('context', ''))

    async def analyze_learning_patterns(self) -> Dict:
        """Analyze patterns in learning to identify areas for improvement"""
        async with self.pool.acquire() as conn:
            # Get recent training history
            rows = await conn.fetch("""
                SELECT trait_updates, feedback_type, performance_delta
                FROM persona_training_history
                WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '7 days'
                ORDER BY timestamp DESC
                LIMIT 100
            """)

            patterns = defaultdict(list)
            for row in rows:
                updates = json.loads(row['trait_updates']) if row['trait_updates'] else {}
                for trait, delta in updates.items():
                    patterns[trait].append(delta)

            # Calculate trends
            trends = {}
            for trait, deltas in patterns.items():
                if deltas:
                    trends[trait] = {
                        'average_delta': np.mean(deltas),
                        'trend': 'increasing' if np.mean(deltas) > 0 else 'decreasing',
                        'volatility': np.std(deltas)
                    }

            return trends

    async def autonomous_self_improvement(self):
        """Autonomous self-improvement loop"""
        logger.info("🧠 Starting autonomous persona training")

        while True:
            try:
                # Analyze recent interactions
                async with self.pool.acquire() as conn:
                    recent = await conn.fetch("""
                        SELECT * FROM echo_unified_interactions
                        WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '1 hour'
                        AND response IS NOT NULL
                        ORDER BY timestamp DESC
                        LIMIT 10
                    """)

                    for interaction in recent:
                        interaction_dict = dict(interaction)
                        await self.train_from_interaction(interaction_dict)

                # Analyze learning patterns
                trends = await self.analyze_learning_patterns()

                # Focus on underperforming traits
                for trait, trend_data in trends.items():
                    if trait in ['proactiveness', 'autonomy', 'code_quality_focus']:
                        if self.current_persona['traits'].get(trait, 0) < 0.7:
                            # Boost important traits that are underperforming
                            await self.apply_trait_updates({trait: 0.05})
                            logger.info(f"📈 Boosting {trait} (currently {self.current_persona['traits'][trait]:.2f})")

                # Log current state
                logger.info(f"Current performance score: {self.current_persona['performance_score']:.3f}")
                logger.info(f"Proactiveness: {self.current_persona['traits'].get('proactiveness', 0):.3f}")
                logger.info(f"Autonomy: {self.current_persona['traits'].get('autonomy', 0):.3f}")

                # Sleep 30 minutes between training cycles
                await asyncio.sleep(1800)

            except Exception as e:
                logger.error(f"Persona training error: {e}")
                await asyncio.sleep(300)

# Global instance
persona_trainer = PersonaTrainer()