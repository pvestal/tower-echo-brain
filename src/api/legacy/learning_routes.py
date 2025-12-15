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
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)

class MusicTrainingData(BaseModel):
    """Music training data model"""
    type: str = "music_training"
    source: str = "apple_music"
    tracks: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    training_instructions: Dict[str, Any]

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

@router.post("/api/echo/training/music")
async def train_on_music_data(training_data: MusicTrainingData) -> Dict[str, Any]:
    """
    Accept Apple Music data for training Echo on Patrick's music preferences
    """
    try:
        logger.info(f"🎵 Received music training data: {len(training_data.tracks)} tracks from {training_data.source}")

        pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=2)

        async with pool.acquire() as conn:
            # Create learning session
            session_id = await conn.fetchval("""
                INSERT INTO learning_sessions (
                    session_name, session_type, trigger_reason,
                    items_processed, status
                ) VALUES ($1, $2, $3, $4, $5)
                RETURNING id
            """,
            f"music_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "music_training",
            f"Apple Music training with {len(training_data.tracks)} tracks",
            len(training_data.tracks),
            "completed"
            )

            # Store music preferences
            for track in training_data.tracks:
                # Store track preferences
                track_key = f"music_track_{track.get('id', 'unknown')}"
                track_data = {
                    "track_id": track.get("id"),
                    "title": track.get("title", "Unknown"),
                    "artist": track.get("artist", "Unknown"),
                    "album": track.get("album", "Unknown"),
                    "genre": track.get("genre", []),
                    "duration": track.get("duration"),
                    "energy": track.get("energy"),
                    "valence": track.get("valence"),
                    "tempo": track.get("tempo"),
                    "user_rating": track.get("user_rating"),
                    "play_count": track.get("play_count"),
                    "training_source": training_data.source,
                    "training_timestamp": datetime.now().isoformat()
                }

                await conn.execute("""
                    INSERT INTO patrick_preferences (preference_key, preference_value, category, updated_at)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (preference_key) DO UPDATE SET
                        preference_value = $2,
                        updated_at = $4
                """, track_key, json.dumps(track_data), "music_tracks", datetime.now())

            # Store general music preferences from metadata and instructions
            if training_data.metadata:
                metadata_key = f"music_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                await conn.execute("""
                    INSERT INTO patrick_preferences (preference_key, preference_value, category, updated_at)
                    VALUES ($1, $2, $3, $4)
                """, metadata_key, json.dumps(training_data.metadata), "music_metadata", datetime.now())

            if training_data.training_instructions:
                instructions_key = f"music_instructions_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                await conn.execute("""
                    INSERT INTO patrick_preferences (preference_key, preference_value, category, updated_at)
                    VALUES ($1, $2, $3, $4)
                """, instructions_key, json.dumps(training_data.training_instructions), "music_instructions", datetime.now())

            # Update session completion
            await conn.execute("""
                UPDATE learning_sessions
                SET end_time = $1, items_accepted = $2, status = 'completed'
                WHERE id = $3
            """, datetime.now(), len(training_data.tracks), session_id)

        await pool.close()

        # Extract key insights
        genres = []
        artists = []
        total_tracks = len(training_data.tracks)

        for track in training_data.tracks:
            if track.get("genre"):
                genres.extend(track["genre"] if isinstance(track["genre"], list) else [track["genre"]])
            if track.get("artist"):
                artists.append(track["artist"])

        top_genres = list(set(genres))[:5] if genres else []
        top_artists = list(set(artists))[:5] if artists else []

        response = {
            "status": "success",
            "message": f"Successfully processed {total_tracks} tracks from {training_data.source}",
            "session_id": session_id,
            "training_summary": {
                "total_tracks": total_tracks,
                "source": training_data.source,
                "top_genres": top_genres,
                "top_artists": top_artists,
                "has_metadata": bool(training_data.metadata),
                "has_instructions": bool(training_data.training_instructions)
            },
            "timestamp": datetime.now().isoformat(),
            "learned_patterns": {
                "genre_diversity": len(top_genres),
                "artist_diversity": len(top_artists),
                "data_richness": "high" if training_data.metadata else "basic"
            }
        }

        logger.info(f"✅ Music training completed: Session {session_id}, {total_tracks} tracks processed")
        return response

    except Exception as e:
        logger.error(f"❌ Error in music training: {e}")
        raise HTTPException(status_code=500, detail=f"Music training failed: {str(e)}")

@router.get("/api/echo/training/music/status")
async def get_music_training_status() -> Dict[str, Any]:
    """Get status of music training sessions"""
    try:
        pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=2)

        async with pool.acquire() as conn:
            # Get music training sessions
            sessions = await conn.fetch("""
                SELECT id, session_name, start_time, end_time,
                       items_processed, items_accepted, status
                FROM learning_sessions
                WHERE session_type = 'music_training'
                ORDER BY start_time DESC
                LIMIT 10
            """)

            # Get music preference counts
            music_prefs = await conn.fetchrow("""
                SELECT
                    COUNT(*) FILTER (WHERE category = 'music_tracks') as track_count,
                    COUNT(*) FILTER (WHERE category = 'music_metadata') as metadata_count,
                    COUNT(*) FILTER (WHERE category = 'music_instructions') as instruction_count,
                    MAX(updated_at) FILTER (WHERE category LIKE 'music_%') as last_update
                FROM patrick_preferences
                WHERE category LIKE 'music_%'
            """)

        await pool.close()

        session_list = []
        for session in sessions:
            session_list.append({
                "session_id": session["id"],
                "name": session["session_name"],
                "start_time": str(session["start_time"]),
                "end_time": str(session["end_time"]) if session["end_time"] else None,
                "tracks_processed": session["items_processed"],
                "tracks_accepted": session["items_accepted"],
                "status": session["status"]
            })

        return {
            "status": "active",
            "total_sessions": len(session_list),
            "stored_preferences": {
                "tracks": music_prefs["track_count"] or 0,
                "metadata_records": music_prefs["metadata_count"] or 0,
                "instruction_records": music_prefs["instruction_count"] or 0,
                "last_update": str(music_prefs["last_update"]) if music_prefs["last_update"] else "Never"
            },
            "recent_sessions": session_list[:5],
            "endpoint_ready": True
        }

    except Exception as e:
        logger.error(f"Error getting music training status: {e}")
        return {"status": "error", "message": str(e)}