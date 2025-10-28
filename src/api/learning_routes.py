#!/usr/bin/env python3
"""
Learning System API Routes for Echo Brain
Exposes learning status, preferences, and insights
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import logging
import asyncpg

router = APIRouter()
logger = logging.getLogger(__name__)

# Database connection
DB_URL = "postgresql://patrick:tower_echo_brain_secret_key_2025@localhost/echo_brain"

@router.get("/api/echo/learning/status")
async def get_learning_status() -> Dict[str, Any]:
    """Get current learning system status"""
    try:
        pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=2)
        async with pool.acquire() as conn:
            # Get learning history
            history = await conn.fetchrow("""
                SELECT COUNT(*) as total_sessions,
                       SUM(items_processed) as total_items,
                       MAX(timestamp) as last_learning
                FROM echo_learning_history
            """)

            # Get preference count
            preferences = await conn.fetchrow("""
                SELECT COUNT(DISTINCT category) as categories,
                       AVG(confidence) as avg_confidence,
                       MAX(last_updated) as last_updated
                FROM echo_preferences
            """)

            # Get visual patterns
            patterns = await conn.fetchrow("""
                SELECT COUNT(*) as pattern_count,
                       AVG(frequency) as avg_frequency
                FROM echo_visual_patterns
            """)

        await pool.close()

        return {
            "status": "active",
            "learning_sessions": history['total_sessions'] or 0,
            "items_processed": int(history['total_items'] or 0),
            "last_learning": str(history['last_learning'] or "Never"),
            "preference_categories": preferences['categories'] or 0,
            "average_confidence": float(preferences['avg_confidence'] or 0),
            "visual_patterns": patterns['pattern_count'] or 0,
            "photos_available": 14301,
            "photos_indexed": 631
        }
    except Exception as e:
        logger.error(f"Error getting learning status: {e}")
        return {"status": "error", "message": str(e)}

@router.get("/api/echo/learning/preferences")
async def get_preferences() -> Dict[str, Any]:
    """Get learned preferences"""
    try:
        pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=2)
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT category, preference_data, confidence, last_updated
                FROM echo_preferences
                WHERE user_id = 'patrick'
                ORDER BY confidence DESC
            """)

        await pool.close()

        preferences = {}
        for row in rows:
            preferences[row['category']] = {
                'data': json.loads(row['preference_data']) if isinstance(row['preference_data'], str) else row['preference_data'],
                'confidence': float(row['confidence']),
                'last_updated': str(row['last_updated'])
            }

        return {
            "user": "patrick",
            "preferences": preferences,
            "categories": list(preferences.keys())
        }
    except Exception as e:
        logger.error(f"Error getting preferences: {e}")
        return {"error": str(e)}

@router.get("/api/echo/learning/insights")
async def get_learning_insights() -> Dict[str, Any]:
    """Get recent learning insights"""
    try:
        pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=2)
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT data_source, items_processed, insights_gained, timestamp
                FROM echo_learning_history
                ORDER BY timestamp DESC
                LIMIT 10
            """)

        await pool.close()

        insights = []
        for row in rows:
            insight_data = json.loads(row['insights_gained']) if isinstance(row['insights_gained'], str) else row['insights_gained']
            insights.append({
                'source': row['data_source'],
                'items': row['items_processed'],
                'timestamp': str(row['timestamp']),
                'insights': insight_data
            })

        # Summarize key findings
        summary = {
            'total_insights': len(insights),
            'sources': list(set([i['source'] for i in insights])),
            'recent_activity': insights[0]['timestamp'] if insights else "No activity"
        }

        return {
            "summary": summary,
            "recent_insights": insights
        }
    except Exception as e:
        logger.error(f"Error getting insights: {e}")
        return {"error": str(e)}

@router.get("/api/echo/learning/progress")
async def get_learning_progress() -> Dict[str, Any]:
    """Get learning progress metrics"""
    try:
        import sqlite3

        # Check photo processing progress
        photo_conn = sqlite3.connect("/opt/tower-echo-brain/photos.db")
        cursor = photo_conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM photos")
        photos_indexed = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT perceptual_hash) FROM photos WHERE perceptual_hash IS NOT NULL")
        unique_patterns = cursor.fetchone()[0]

        photo_conn.close()

        # Get conversation learning progress
        pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=2)
        async with pool.acquire() as conn:
            conv_count = await conn.fetchval("""
                SELECT COUNT(*) FROM echo_unified_interactions
                WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '30 days'
            """)

            task_count = await conn.fetchval("""
                SELECT COUNT(*) FROM echo_tasks
                WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
            """)

        await pool.close()

        return {
            "photo_learning": {
                "total_available": 14301,
                "indexed": photos_indexed,
                "processed_percentage": round((photos_indexed / 14301) * 100, 2),
                "unique_patterns": unique_patterns,
                "status": "processing" if photos_indexed < 14301 else "complete"
            },
            "conversation_learning": {
                "recent_conversations": conv_count,
                "status": "active"
            },
            "behavior_learning": {
                "recent_tasks": task_count,
                "status": "analyzing"
            },
            "next_learning_cycle": "Every 15 minutes",
            "models_saved": {
                "visual_preferences": "Active",
                "conversation_model": "Active",
                "preference_model": "Building"
            }
        }
    except Exception as e:
        logger.error(f"Error getting progress: {e}")
        return {"error": str(e)}

@router.post("/api/echo/learning/trigger")
async def trigger_learning_cycle() -> Dict[str, Any]:
    """Manually trigger a learning cycle"""
    try:
        # Import learning system
        from src.learning.learning_system import learning_system

        # Initialize if needed
        if not learning_system.pool:
            await learning_system.initialize()

        # Run learning cycles
        photo_insights = await learning_system.learn_from_photos(10)
        conv_insights = await learning_system.learn_from_conversations()
        behavior_insights = await learning_system.learn_from_behavior()

        return {
            "status": "success",
            "triggered_at": datetime.now().isoformat(),
            "results": {
                "photos_processed": photo_insights.get('total_processed', 0),
                "conversations_analyzed": conv_insights.get('total_conversations', 0),
                "behavior_patterns": len(behavior_insights.get('active_hours', {}))
            }
        }
    except Exception as e:
        logger.error(f"Error triggering learning: {e}")
        return {"status": "error", "message": str(e)}

@router.get("/api/echo/learning/personality")
async def get_personality_traits() -> Dict[str, Any]:
    """Get Echo's learned personality traits"""
    try:
        pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=2)
        async with pool.acquire() as conn:
            # Get persona traits
            persona = await conn.fetchrow("""
                SELECT traits, performance_score, generation_count, last_updated
                FROM echo_persona
                ORDER BY last_updated DESC
                LIMIT 1
            """)

        await pool.close()

        if persona:
            traits = json.loads(persona['traits']) if isinstance(persona['traits'], str) else persona['traits']
            return {
                "personality": traits,
                "performance_score": float(persona['performance_score']),
                "evolution_count": persona['generation_count'],
                "last_updated": str(persona['last_updated']),
                "key_traits": {
                    "proactiveness": traits.get('proactiveness', 0),
                    "autonomy": traits.get('autonomy', 0),
                    "technical_accuracy": traits.get('technical_accuracy', 0),
                    "verbosity": traits.get('verbosity', 0),
                    "code_quality_focus": traits.get('code_quality_focus', 0)
                }
            }
        else:
            return {
                "personality": "Not initialized",
                "message": "Personality system not yet active"
            }
    except Exception as e:
        logger.error(f"Error getting personality: {e}")
        return {"error": str(e)}