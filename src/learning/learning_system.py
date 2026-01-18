# 🔴 DEPRECATED: Use unified_router.py instead
# This file is being phased out in favor of single source of truth
# Import from: from src.routing.unified_router import unified_router

#!/usr/bin/env python3
"""
Learning System for Echo Brain
Implements actual learning from photos, conversations, and user interactions
Patrick Vestal - October 28, 2025
"""

import asyncio
import json
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncpg
import numpy as np
from PIL import Image
import imagehash
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle

logger = logging.getLogger(__name__)

class LearningSystem:
    """Actually implements learning from Patrick's data"""

    def __init__(self):
        self.db_url = "postgresql://patrick:RP78eIrW7cI2jYvL5akt1yurE@localhost/echo_brain"
        self.photos_db = "/opt/tower-echo-brain/photos.db"
        self.models_dir = Path("/opt/tower-echo-brain/models")
        self.models_dir.mkdir(exist_ok=True)

        # Learning components
        self.visual_model = None
        self.preference_model = None
        self.conversation_model = None
        self.pool = None

        # Patrick's data sources
        self.google_photos_path = Path("/mnt/10TB2/Google_Takeout_2025/Takeout/Google Photos")
        self.google_drive_path = Path("/mnt/10TB2/Google_Takeout_2025/Takeout/Drive")

        # Learning state
        self.preferences = {
            'visual_style': {},
            'content_topics': {},
            'interaction_patterns': {},
            'anime_preferences': {},
            'music_preferences': {},
            'technical_interests': {}
        }

    async def initialize(self):
        """Initialize all learning systems"""
        logger.info("🧠 Initializing Learning System")

        # Database connection
        self.pool = await asyncpg.create_pool(self.db_url, min_size=2, max_size=10)

        # Ensure tables exist
        await self.ensure_tables()

        # Load or create models
        await self.load_or_create_models()

        # Load existing preferences
        await self.load_preferences()

        logger.info("✅ Learning system initialized")

    async def ensure_tables(self):
        """Create learning-related tables"""
        async with self.pool.acquire() as conn:
            # Preferences table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS echo_preferences (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT DEFAULT 'patrick',
                    category TEXT NOT NULL,
                    preference_data JSONB NOT NULL,
                    confidence FLOAT DEFAULT 0.5,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_count INTEGER DEFAULT 1
                )
            """)

            # Learning history
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS echo_learning_history (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    data_source TEXT,
                    items_processed INTEGER,
                    insights_gained JSONB,
                    model_updates JSONB
                )
            """)

            # Visual patterns
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS echo_visual_patterns (
                    id SERIAL PRIMARY KEY,
                    pattern_hash TEXT UNIQUE,
                    pattern_type TEXT,
                    features JSONB,
                    frequency INTEGER DEFAULT 1,
                    user_rating FLOAT,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    async def load_or_create_models(self):
        """Load existing models or create new ones"""
        # Visual preference model
        visual_model_path = self.models_dir / "visual_preferences.pkl"
        if visual_model_path.exists():
            with open(visual_model_path, 'rb') as f:
                self.visual_model = pickle.load(f)
            logger.info("Loaded existing visual model")
        else:
            self.visual_model = self.create_visual_model()
            logger.info("Created new visual model")

        # Conversation model
        conv_model_path = self.models_dir / "conversation_model.pkl"
        if conv_model_path.exists():
            with open(conv_model_path, 'rb') as f:
                self.conversation_model = pickle.load(f)
        else:
            self.conversation_model = self.create_conversation_model()

    def create_visual_model(self):
        """Create visual preference learning model"""
        return {
            'color_preferences': KMeans(n_clusters=5),
            'composition_patterns': [],
            'subject_preferences': {},
            'style_clusters': KMeans(n_clusters=8),
            'scaler': StandardScaler()
        }

    def create_conversation_model(self):
        """Create conversation pattern model"""
        return {
            'response_patterns': {},
            'topic_interests': {},
            'interaction_style': {},
            'feedback_signals': []
        }

    async def load_preferences(self):
        """Load existing preferences from database"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT category, preference_data, confidence
                FROM echo_preferences
                WHERE user_id = 'patrick'
                ORDER BY last_updated DESC
            """)

            for row in rows:
                category = row['category']
                if category in self.preferences:
                    self.preferences[category] = row['preference_data']

        logger.info(f"Loaded {len(rows)} preference categories")

    async def learn_from_photos(self, batch_size: int = 100) -> Dict:
        """Actually learn from Patrick's photos"""
        logger.info("📸 Starting photo learning")

        conn = sqlite3.connect(self.photos_db)
        cursor = conn.cursor()

        # Get unprocessed photos
        cursor.execute("""
            SELECT file_path, perceptual_hash, metadata
            FROM photos
            WHERE file_path NOT IN (
                SELECT item_id FROM learning_processed WHERE item_type = 'photo'
            )
            LIMIT ?
        """, (batch_size,))

        photos = cursor.fetchall()
        insights = {
            'total_processed': 0,
            'patterns_found': [],
            'color_preferences': [],
            'subject_types': {}
        }

        for photo_path, phash, metadata in photos:
            try:
                # Analyze photo
                photo_insights = await self.analyze_photo(photo_path)

                # Update preferences
                await self.update_visual_preferences(photo_insights)

                # Track patterns
                if photo_insights.get('dominant_colors'):
                    insights['color_preferences'].extend(photo_insights['dominant_colors'])

                if photo_insights.get('subjects'):
                    for subject in photo_insights['subjects']:
                        insights['subject_types'][subject] = insights['subject_types'].get(subject, 0) + 1

                insights['total_processed'] += 1

            except Exception as e:
                logger.error(f"Error processing photo {photo_path}: {e}")

        # Save learning progress
        await self.save_learning_history('photos', insights)

        conn.close()
        return insights

    async def analyze_photo(self, photo_path: str) -> Dict:
        """Analyze individual photo for patterns"""
        insights = {}

        try:
            if Path(photo_path).exists():
                img = Image.open(photo_path)

                # Get dominant colors
                img_small = img.resize((150, 150))
                colors = img_small.getcolors(maxcolors=1000000)
                if colors:
                    sorted_colors = sorted(colors, key=lambda x: x[0], reverse=True)
                    insights['dominant_colors'] = [c[1] for c in sorted_colors[:5]]

                # Get image hash for similarity
                insights['image_hash'] = str(imagehash.average_hash(img))

                # Basic composition analysis
                width, height = img.size
                insights['aspect_ratio'] = width / height if height > 0 else 1
                insights['resolution_class'] = 'high' if width > 2000 else 'medium' if width > 1000 else 'low'

                # Extract EXIF data if available
                exif = img.getexif() if hasattr(img, 'getexif') else {}
                if exif:
                    insights['camera_info'] = str(exif.get(272, 'Unknown'))  # Camera description

        except Exception as e:
            logger.error(f"Photo analysis error: {e}")

        return insights

    async def update_visual_preferences(self, insights: Dict):
        """Update visual preferences based on photo insights"""
        if not insights:
            return

        # Update color preferences
        if 'dominant_colors' in insights:
            colors = self.preferences.get('visual_style', {}).get('colors', [])
            colors.extend(insights['dominant_colors'])
            # Keep last 1000 colors for clustering
            colors = colors[-1000:]
            self.preferences['visual_style']['colors'] = colors

        # Update composition preferences
        if 'aspect_ratio' in insights:
            ratios = self.preferences.get('visual_style', {}).get('aspect_ratios', [])
            ratios.append(insights['aspect_ratio'])
            self.preferences['visual_style']['aspect_ratios'] = ratios[-500:]

    async def learn_from_conversations(self) -> Dict:
        """Learn from conversation history"""
        logger.info("💬 Learning from conversations")

        async with self.pool.acquire() as conn:
            # Get recent conversations
            rows = await conn.fetch("""
                SELECT query, response, model_used, processing_time, created_at
                FROM echo_unified_interactions
                WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '30 days'
                ORDER BY created_at DESC
                LIMIT 500
            """)

            insights = {
                'total_conversations': len(rows),
                'topics': {},
                'response_preferences': {},
                'model_preferences': {}
            }

            for row in rows:
                query = row['query'] or ''
                response = row['response'] or ''
                model = row['model_used']

                # Extract topics
                topics = self.extract_topics(query)
                for topic in topics:
                    insights['topics'][topic] = insights['topics'].get(topic, 0) + 1

                # Track model usage
                if model:
                    insights['model_preferences'][model] = insights['model_preferences'].get(model, 0) + 1

                # Analyze response patterns
                if len(response) < 500:
                    insights['response_preferences']['prefers_concise'] = \
                        insights['response_preferences'].get('prefers_concise', 0) + 1

            # Update preferences
            self.preferences['interaction_patterns'] = insights
            await self.save_preferences('interaction_patterns')

            return insights

    def extract_topics(self, text: str) -> List[str]:
        """Extract topics from text"""
        topics = []

        # Technical topics
        tech_keywords = ['code', 'python', 'javascript', 'api', 'database', 'echo', 'tower',
                        'anime', 'comfyui', 'gpu', 'ai', 'model', 'training']
        for keyword in tech_keywords:
            if keyword in text.lower():
                topics.append(keyword)

        return topics

    async def learn_from_behavior(self) -> Dict:
        """Learn from user behavior patterns"""
        logger.info("🔄 Learning from behavior patterns")

        async with self.pool.acquire() as conn:
            # Analyze task patterns
            tasks = await conn.fetch("""
                SELECT task_type, status, created_at
                FROM echo_tasks
                WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
            """)

            # Analyze time patterns
            hour_distribution = {}
            for task in tasks:
                hour = task['created_at'].hour
                hour_distribution[hour] = hour_distribution.get(hour, 0) + 1

            insights = {
                'active_hours': hour_distribution,
                'task_preferences': {},
                'automation_opportunities': []
            }

            # Find automation opportunities
            if hour_distribution:
                peak_hour = max(hour_distribution, key=hour_distribution.get)
                insights['automation_opportunities'].append({
                    'type': 'scheduled_tasks',
                    'suggestion': f'Schedule proactive tasks around {peak_hour}:00'
                })

            return insights

    async def generate_personalized_response(self, query: str) -> Dict:
        """Generate response using learned preferences"""
        personalization = {
            'style_modifiers': [],
            'content_focus': [],
            'preferred_models': []
        }

        # Apply learned preferences
        if self.preferences.get('interaction_patterns', {}).get('response_preferences', {}).get('prefers_concise', 0) > 10:
            personalization['style_modifiers'].append('concise')

        # Check topic preferences
        topics = self.extract_topics(query)
        topic_scores = self.preferences.get('interaction_patterns', {}).get('topics', {})
        for topic in topics:
            if topic in topic_scores and topic_scores[topic] > 5:
                personalization['content_focus'].append(topic)

        # Model selection based on history
        model_prefs = self.preferences.get('interaction_patterns', {}).get('model_preferences', {})
        if model_prefs:
            sorted_models = sorted(model_prefs.items(), key=lambda x: x[1], reverse=True)
            personalization['preferred_models'] = [m[0] for m in sorted_models[:3]]

        return personalization

    async def save_preferences(self, category: str):
        """Save preferences to database"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO echo_preferences (user_id, category, preference_data, confidence)
                VALUES ('patrick', $1, $2, $3)
                ON CONFLICT (user_id, category)
                DO UPDATE SET
                    preference_data = $2,
                    confidence = confidence * 0.9 + 0.1,
                    last_updated = CURRENT_TIMESTAMP,
                    update_count = echo_preferences.update_count + 1
            """, category, json.dumps(self.preferences.get(category, {})), 0.7)

    async def save_learning_history(self, source: str, insights: Dict):
        """Save learning history"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO echo_learning_history (data_source, items_processed, insights_gained)
                VALUES ($1, $2, $3)
            """, source, insights.get('total_processed', 0), json.dumps(insights))

    async def continuous_learning_loop(self):
        """Main learning loop"""
        logger.info("🔄 Starting continuous learning loop")

        while True:
            try:
                # Learn from photos (batch of 50)
                photo_insights = await self.learn_from_photos(50)
                logger.info(f"Learned from {photo_insights['total_processed']} photos")

                # Learn from conversations
                conv_insights = await self.learn_from_conversations()
                logger.info(f"Analyzed {conv_insights['total_conversations']} conversations")

                # Learn from behavior
                behavior_insights = await self.learn_from_behavior()

                # Save models
                await self.save_models()

                # Log progress
                logger.info(f"Learning cycle complete. Preferences updated.")
                logger.info(f"Visual patterns: {len(self.preferences.get('visual_style', {}).get('colors', []))} colors tracked")
                logger.info(f"Topics of interest: {list(conv_insights.get('topics', {}).keys())[:5]}")

                # Sleep 15 minutes
                await asyncio.sleep(900)

            except Exception as e:
                logger.error(f"Learning loop error: {e}")
                await asyncio.sleep(300)

    async def save_models(self):
        """Save trained models to disk"""
        try:
            # Save visual model
            with open(self.models_dir / "visual_preferences.pkl", 'wb') as f:
                pickle.dump(self.visual_model, f)

            # Save conversation model
            with open(self.models_dir / "conversation_model.pkl", 'wb') as f:
                pickle.dump(self.conversation_model, f)

            # Save preferences
            with open(self.models_dir / "preferences.json", 'w') as f:
                json.dump(self.preferences, f, indent=2, default=str)

            logger.info("Models saved successfully")

        except Exception as e:
            logger.error(f"Error saving models: {e}")

# Global instance
learning_system = LearningSystem()