#!/usr/bin/env python3
"""
Neural Process Metrics API - Real cognitive process mapping
Maps Echo Brain's actual internal processes to meaningful brain region visualization
"""

import json
import logging
import asyncio
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException
from src.db.database import database
import psycopg2

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


async def get_active_task_stats() -> Dict:
    """Get real task queue statistics"""
    try:
        conn = psycopg2.connect(**database.db_config)
        cursor = conn.cursor()

        # Get task statistics by type and priority
        cursor.execute("""
            SELECT
                task_type,
                status,
                priority,
                COUNT(*) as count,
                AVG(EXTRACT(EPOCH FROM (NOW() - created_at))) as avg_age_seconds
            FROM echo_tasks
            WHERE created_at > NOW() - INTERVAL '1 hour'
            GROUP BY task_type, status, priority
            ORDER BY priority DESC, count DESC;
        """)

        task_stats = cursor.fetchall()

        # Get active background processes
        cursor.execute("""
            SELECT
                task_type,
                status,
                description,
                priority,
                created_at
            FROM echo_tasks
            WHERE status IN ('pending', 'in_progress')
            ORDER BY priority DESC, created_at DESC
            LIMIT 20;
        """)

        active_tasks = cursor.fetchall()

        cursor.close()
        conn.close()

        return {
            "task_stats": task_stats,
            "active_tasks": active_tasks,
            "total_active": len(active_tasks)
        }
    except Exception as e:
        logger.error(f"Failed to get task stats: {e}")
        return {"task_stats": [], "active_tasks": [], "total_active": 0}


async def get_model_decision_activity() -> Dict:
    """Get recent model selection and decision-making activity"""
    try:
        conn = psycopg2.connect(**database.db_config)
        cursor = conn.cursor()

        # Get recent model decisions
        cursor.execute("""
            SELECT
                model_tier,
                selected_model,
                complexity_score,
                board_consensus,
                created_at
            FROM model_decisions
            WHERE created_at > NOW() - INTERVAL '1 hour'
            ORDER BY created_at DESC
            LIMIT 10;
        """)

        decisions = cursor.fetchall()

        # Get decision frequency by complexity
        cursor.execute("""
            SELECT
                model_tier,
                COUNT(*) as decision_count,
                AVG(complexity_score) as avg_complexity
            FROM model_decisions
            WHERE created_at > NOW() - INTERVAL '1 hour'
            GROUP BY model_tier
            ORDER BY decision_count DESC;
        """)

        complexity_stats = cursor.fetchall()

        cursor.close()
        conn.close()

        return {
            "recent_decisions": decisions,
            "complexity_stats": complexity_stats,
            "decision_frequency": len(decisions)
        }
    except Exception as e:
        logger.error(f"Failed to get decision activity: {e}")
        return {"recent_decisions": [], "complexity_stats": [], "decision_frequency": 0}


async def get_conversation_memory_activity() -> Dict:
    """Get memory processing and conversation activity"""
    try:
        conn = psycopg2.connect(**database.db_config)
        cursor = conn.cursor()

        # Get recent conversation activity
        cursor.execute("""
            SELECT
                COUNT(*) as total_interactions,
                COUNT(DISTINCT conversation_id) as unique_conversations,
                AVG(LENGTH(query)) as avg_query_length,
                AVG(LENGTH(response)) as avg_response_length
            FROM echo_unified_interactions
            WHERE timestamp > NOW() - INTERVAL '1 hour';
        """)

        interaction_stats = cursor.fetchone()

        # Get memory retrieval patterns
        cursor.execute("""
            SELECT
                conversation_id,
                COUNT(*) as interaction_count,
                MAX(timestamp) as last_activity,
                AVG(LENGTH(query) + LENGTH(response)) as avg_conversation_size
            FROM echo_unified_interactions
            WHERE timestamp > NOW() - INTERVAL '6 hours'
            GROUP BY conversation_id
            ORDER BY interaction_count DESC
            LIMIT 5;
        """)

        memory_patterns = cursor.fetchall()

        cursor.close()
        conn.close()

        return {
            "interaction_stats": interaction_stats,
            "memory_patterns": memory_patterns,
            "active_conversations": len(memory_patterns)
        }
    except Exception as e:
        logger.error(f"Failed to get memory activity: {e}")
        return {"interaction_stats": None, "memory_patterns": [], "active_conversations": 0}


async def get_background_learning_activity() -> Dict:
    """Get autonomous learning and persona adjustment activity"""
    try:
        conn = psycopg2.connect(**database.db_config)
        cursor = conn.cursor()

        # Check for persona training history
        cursor.execute("""
            SELECT
                COUNT(*) as training_events,
                MAX(created_at) as last_training
            FROM persona_training_history
            WHERE created_at > NOW() - INTERVAL '6 hours';
        """)

        training_stats = cursor.fetchone()

        # Get autonomous behavior activity
        cursor.execute("""
            SELECT
                task_type,
                COUNT(*) as execution_count,
                MAX(created_at) as last_execution
            FROM echo_tasks
            WHERE task_type IN ('AUTONOMOUS_REPAIR', 'CODE_REFACTOR', 'SYSTEM_OPTIMIZATION')
                AND created_at > NOW() - INTERVAL '6 hours'
            GROUP BY task_type;
        """)

        autonomous_activity = cursor.fetchall()

        cursor.close()
        conn.close()

        return {
            "training_stats": training_stats,
            "autonomous_activity": autonomous_activity,
            "learning_active": training_stats[0] > 0 if training_stats else False
        }
    except Exception as e:
        logger.error(f"Failed to get learning activity: {e}")
        return {"training_stats": None, "autonomous_activity": [], "learning_active": False}


@router.get("/api/echo/neural/activity")
async def get_neural_activity():
    """
    Get real neural activity mapped to brain regions based on actual Echo Brain processes

    Maps real cognitive processes to brain regions:
    - Prefrontal Cortex: Decision making, model selection, task prioritization
    - Temporal Lobe: Memory processing, conversation context, knowledge retrieval
    - Frontal Lobe: Active execution, API processing, system operations
    - Limbic System: Learning, persona adjustments, autonomous behaviors
    """
    try:
        # Get system resource metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        # Get real cognitive process data
        task_data = await get_active_task_stats()
        decision_data = await get_model_decision_activity()
        memory_data = await get_conversation_memory_activity()
        learning_data = await get_background_learning_activity()

        # Calculate real neural activity for each region

        # PREFRONTAL CORTEX - Decision Making & Planning
        decision_activity = min(100.0, (decision_data['decision_frequency'] * 10) +
                               (len([t for t in task_data['active_tasks'] if t[3] == 'HIGH']) * 5))
        prefrontal_neurons = []
        for task in task_data['active_tasks'][:5]:  # Show top 5 decision tasks
            if task[3] == 'HIGH':  # High priority tasks
                prefrontal_neurons.append({
                    "id": f"decision_{task[0]}",
                    "type": "decision",
                    "activity": min(100, task[3] == 'HIGH' and 80 or 40),
                    "task": task[0].replace('_', ' ').title(),
                    "status": task[1]
                })

        # TEMPORAL LOBE - Memory & Context
        memory_activity = min(100.0,
                            (memory_data['active_conversations'] * 15) +
                            (memory.percent / 2))
        temporal_neurons = []
        for conv in memory_data['memory_patterns'][:5]:
            temporal_neurons.append({
                "id": f"memory_{conv[0]}",
                "type": "memory",
                "activity": min(100, conv[1] * 3),  # Based on interaction count
                "task": f"Context: {conv[0][:20]}...",
                "status": "active"
            })

        # FRONTAL LOBE - Active Execution
        execution_activity = min(100.0, cpu_percent +
                               (len([t for t in task_data['active_tasks'] if t[1] == 'in_progress']) * 10))
        frontal_neurons = []
        for task in task_data['active_tasks'][:5]:
            if task[1] == 'in_progress':
                frontal_neurons.append({
                    "id": f"exec_{task[0]}",
                    "type": "execution",
                    "activity": 85,
                    "task": task[0].replace('_', ' ').title(),
                    "status": task[1]
                })

        # LIMBIC SYSTEM - Learning & Adaptation
        learning_activity = min(100.0,
                              (learning_data['learning_active'] and 60 or 20) +
                              (len(learning_data['autonomous_activity']) * 10))
        limbic_neurons = []
        for activity in learning_data['autonomous_activity'][:3]:
            limbic_neurons.append({
                "id": f"learning_{activity[0]}",
                "type": "learning",
                "activity": min(100, activity[1] * 8),
                "task": activity[0].replace('_', ' ').title(),
                "status": "adaptive"
            })

        return {
            "brain_regions": {
                "prefrontal_cortex": {
                    "activity_percent": round(decision_activity, 1),
                    "neuron_count": len(prefrontal_neurons),
                    "load_level": "high" if decision_activity > 70 else "medium" if decision_activity > 30 else "low",
                    "neurons": prefrontal_neurons,
                    "primary_function": "Decision Making & Model Selection",
                    "current_focus": decision_data['recent_decisions'][0][1] if decision_data['recent_decisions'] else "Standby"
                },
                "temporal_lobe": {
                    "activity_percent": round(memory_activity, 1),
                    "neuron_count": len(temporal_neurons),
                    "load_level": "high" if memory_activity > 70 else "medium" if memory_activity > 30 else "low",
                    "neurons": temporal_neurons,
                    "primary_function": "Memory & Context Processing",
                    "current_focus": f"{memory_data['active_conversations']} Active Conversations"
                },
                "frontal_lobe": {
                    "activity_percent": round(execution_activity, 1),
                    "neuron_count": len(frontal_neurons),
                    "load_level": "high" if execution_activity > 70 else "medium" if execution_activity > 30 else "low",
                    "neurons": frontal_neurons,
                    "primary_function": "Task Execution & System Operations",
                    "current_focus": f"CPU: {cpu_percent:.1f}%"
                },
                "limbic_system": {
                    "activity_percent": round(learning_activity, 1),
                    "neuron_count": len(limbic_neurons),
                    "load_level": "high" if learning_activity > 70 else "medium" if learning_activity > 30 else "low",
                    "neurons": limbic_neurons,
                    "primary_function": "Learning & Autonomous Adaptation",
                    "current_focus": "Persona Training" if learning_data['learning_active'] else "Background Monitoring"
                }
            },
            "overall_cognitive_load": round((decision_activity + memory_activity + execution_activity + learning_activity) / 4, 1),
            "total_active_neurons": len(prefrontal_neurons) + len(temporal_neurons) + len(frontal_neurons) + len(limbic_neurons),
            "system_metrics": {
                "cpu_percent": round(cpu_percent, 1),
                "memory_percent": round(memory.percent, 1),
                "active_tasks": task_data['total_active'],
                "recent_decisions": len(decision_data['recent_decisions'])
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting neural activity: {e}")
        raise HTTPException(status_code=500, detail=f"Neural activity error: {str(e)}")


@router.get("/api/echo/neural/heatmap")
async def get_neural_heatmap():
    """
    Generate real-time neural heatmap data for brain visualization
    Returns heatmap coordinates based on actual processing activity
    """
    try:
        # Get neural activity data
        activity_data = await get_neural_activity()
        brain_regions = activity_data["brain_regions"]

        # Generate heatmap coordinates for each region
        heatmap_data = {
            "prefrontal_cortex": {
                "center": {"x": 50, "y": 20},  # Top center
                "intensity": brain_regions["prefrontal_cortex"]["activity_percent"] / 100,
                "radius": max(20, brain_regions["prefrontal_cortex"]["neuron_count"] * 2),
                "color": "rgba(47, 129, 247, {})".format(brain_regions["prefrontal_cortex"]["activity_percent"] / 100)
            },
            "temporal_lobe": {
                "center": {"x": 20, "y": 60},  # Left side
                "intensity": brain_regions["temporal_lobe"]["activity_percent"] / 100,
                "radius": max(20, brain_regions["temporal_lobe"]["neuron_count"] * 2),
                "color": "rgba(16, 185, 129, {})".format(brain_regions["temporal_lobe"]["activity_percent"] / 100)
            },
            "frontal_lobe": {
                "center": {"x": 80, "y": 40},  # Right side
                "intensity": brain_regions["frontal_lobe"]["activity_percent"] / 100,
                "radius": max(20, brain_regions["frontal_lobe"]["neuron_count"] * 2),
                "color": "rgba(245, 158, 11, {})".format(brain_regions["frontal_lobe"]["activity_percent"] / 100)
            },
            "limbic_system": {
                "center": {"x": 50, "y": 80},  # Bottom center
                "intensity": brain_regions["limbic_system"]["activity_percent"] / 100,
                "radius": max(20, brain_regions["limbic_system"]["neuron_count"] * 2),
                "color": "rgba(124, 58, 237, {})".format(brain_regions["limbic_system"]["activity_percent"] / 100)
            }
        }

        return {
            "heatmap_regions": heatmap_data,
            "overall_activity": activity_data["overall_cognitive_load"],
            "neural_pathways": [
                {
                    "from": "prefrontal_cortex",
                    "to": "frontal_lobe",
                    "strength": min(100, (brain_regions["prefrontal_cortex"]["activity_percent"] +
                                        brain_regions["frontal_lobe"]["activity_percent"]) / 2),
                    "type": "decision_execution"
                },
                {
                    "from": "temporal_lobe",
                    "to": "frontal_lobe",
                    "strength": min(100, (brain_regions["temporal_lobe"]["activity_percent"] +
                                        brain_regions["frontal_lobe"]["activity_percent"]) / 2),
                    "type": "memory_processing"
                },
                {
                    "from": "limbic_system",
                    "to": "prefrontal_cortex",
                    "strength": min(100, (brain_regions["limbic_system"]["activity_percent"] +
                                        brain_regions["prefrontal_cortex"]["activity_percent"]) / 2),
                    "type": "learning_feedback"
                }
            ],
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error generating neural heatmap: {e}")
        raise HTTPException(status_code=500, detail=f"Heatmap error: {str(e)}")