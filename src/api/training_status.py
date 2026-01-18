#!/usr/bin/env python3
"""Training status API endpoint for Tower Dashboard integration"""

from fastapi import APIRouter
import psycopg2
from datetime import datetime
import json

router = APIRouter()

def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        database="echo_brain",
        user="patrick",
        password="tower_echo_brain_secret_key_2025"
    )

@router.get("/api/echo/training/status")
async def get_training_status():
    """Get comprehensive training status for Tower Dashboard"""
    conn = get_db_connection()
    cur = conn.cursor()

    # Get model training status
    cur.execute("""
        SELECT model_name, training_phase, epochs_completed,
               accuracy, dataset_size, updated_at
        FROM model_training_status
        WHERE model_name = 'echo_brain'
    """)
    model_status = cur.fetchone()

    # Get training feedback count
    cur.execute("""
        SELECT COUNT(*) as total,
               COUNT(CASE WHEN was_useful = true THEN 1 END) as useful,
               COUNT(CASE WHEN was_useful = false THEN 1 END) as not_useful,
               MAX(created_at) as latest
        FROM training_feedback
    """)
    feedback_stats = cur.fetchone()

    # Get learned patterns count
    cur.execute("""
        SELECT COUNT(*) as total,
               COUNT(DISTINCT pattern_type) as unique_types,
               MAX(created_at) as latest
        FROM learned_patterns
    """)
    patterns_stats = cur.fetchone()

    # Get learning history stats
    cur.execute("""
        SELECT COUNT(*) as total,
               MAX(created_at) as latest
        FROM learning_history
    """)
    history_stats = cur.fetchone()

    cur.close()
    conn.close()

    response = {
        "status": "active" if model_status else "inactive",
        "model": {
            "name": model_status[0] if model_status else "echo_brain",
            "phase": model_status[1] if model_status else "not_started",
            "epochs": model_status[2] if model_status else 0,
            "accuracy": float(model_status[3]) if model_status and model_status[3] else 0,
            "dataset_size": model_status[4] if model_status else 0,
            "last_update": model_status[5].isoformat() if model_status else None
        },
        "training_data": {
            "feedback": {
                "total": feedback_stats[0],
                "useful": feedback_stats[1],
                "not_useful": feedback_stats[2],
                "latest": feedback_stats[3].isoformat() if feedback_stats[3] else None
            },
            "patterns": {
                "total": patterns_stats[0],
                "unique_types": patterns_stats[1],
                "latest": patterns_stats[2].isoformat() if patterns_stats[2] else None
            },
            "learning_history": {
                "total": history_stats[0],
                "latest": history_stats[1].isoformat() if history_stats[1] else None
            }
        },
        "datasets": {
            "internal": {
                "conversations": "12,155 Claude conversation files",
                "echo_conversations": "633+ stored conversations",
                "learning_history": f"{history_stats[0]} learning entries"
            },
            "external": {
                "ollama_models": "24+ models available for training",
                "vector_database": "635+ embeddings in Qdrant",
                "knowledge_base": "Tower KB articles"
            }
        },
        "recommendation": "Training is active and processing data. Consider importing more Claude conversations for better pattern learning."
    }

    return response