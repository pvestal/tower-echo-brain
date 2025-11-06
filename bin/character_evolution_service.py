#!/usr/bin/env python3
"""
Character Evolution Service for Tower Anime Production Suite
Provides FastAPI endpoints for character development tracking and timeline management
"""

import os
import logging
import json
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Models for API requests/responses
class CharacterEvolutionCreate(BaseModel):
    character_id: int
    scene_id: Optional[int] = None
    evolution_type: str = Field(..., pattern=r'^(personality_shift|skill_gain|relationship_change|trauma|growth)$')
    title: str = Field(..., max_length=255)
    description: str
    impact_level: int = Field(..., ge=1, le=10)
    previous_state: Optional[Dict[str, Any]] = {}
    new_state: Optional[Dict[str, Any]] = {}
    triggers: Optional[List[str]] = []

class CharacterStateCreate(BaseModel):
    character_id: int
    evolution_timeline_id: Optional[int] = None
    state_snapshot: Dict[str, Any]
    personality_traits: Optional[Dict[str, Any]] = {}
    skills: Optional[Dict[str, Any]] = {}
    emotional_state: Optional[Dict[str, Any]] = {}
    relationships_snapshot: Optional[Dict[str, Any]] = {}
    story_arc_position: Optional[str] = 'introduction'
    character_level: int = Field(default=1, ge=1)
    notes: Optional[str] = None

class RelationshipDynamicsCreate(BaseModel):
    relationship_id: int
    evolution_timeline_id: Optional[int] = None
    relationship_strength: int = Field(..., ge=-10, le=10)
    relationship_status: str = Field(default='developing', pattern=r'^(developing|stable|deteriorating|broken|restored)$')
    interaction_frequency: str = Field(default='occasional', pattern=r'^(constant|frequent|occasional|rare|none)$')
    emotional_intensity: int = Field(..., ge=1, le=10)
    conflict_level: int = Field(default=0, ge=0, le=10)
    trust_level: int = Field(default=5, ge=0, le=10)
    dependency_level: int = Field(default=0, ge=0, le=10)
    recent_interactions: Optional[Dict[str, Any]] = {}
    relationship_milestones: Optional[List[str]] = []
    notes: Optional[str] = None

class EmotionalImpactCreate(BaseModel):
    character_id: int
    scene_id: Optional[int] = None
    evolution_timeline_id: Optional[int] = None
    trigger_event: str
    trigger_character_id: Optional[int] = None
    emotional_response: Dict[str, Any]
    intensity_level: int = Field(..., ge=1, le=10)
    duration_category: str = Field(default='short_term', pattern=r'^(momentary|short_term|lasting|permanent)$')
    baseline_impact: Optional[Dict[str, Any]] = {}
    coping_mechanism: Optional[str] = None
    long_term_effects: Optional[Dict[str, Any]] = {}
    recovery_timeline: Optional[int] = None

# Response models
class CharacterEvolutionResponse(BaseModel):
    id: int
    character_id: int
    scene_id: Optional[int]
    evolution_type: str
    title: str
    description: str
    impact_level: int
    previous_state: Dict[str, Any]
    new_state: Dict[str, Any]
    triggers: List[str]
    timestamp: datetime
    created_at: datetime

class CharacterTimelineResponse(BaseModel):
    character_id: int
    character_name: str
    timeline: List[CharacterEvolutionResponse]
    current_state: Optional[Dict[str, Any]]
    total_evolution_points: int

app = FastAPI(title="Character Evolution Service", version="1.0.0")

class CharacterEvolutionService:
    def __init__(self):
        self.db_config = {
            "host": "localhost",
            "database": "echo_brain",
            "user": os.getenv("TOWER_USER", os.getenv("TOWER_USER", "patrick")), 
            "password": "admin123"
        }
        
    def get_db_connection(self):
        """Get database connection with proper error handling"""
        try:
            conn = psycopg2.connect(**self.db_config)
            conn.autocommit = False
            return conn
        except psycopg2.Error as e:
            logger.error(f"Database connection failed: {e}")
            raise HTTPException(status_code=500, detail="Database connection failed")
    
    def create_evolution_event(self, evolution_data: CharacterEvolutionCreate) -> int:
        """Create a new character evolution event"""
        conn = None
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()
            
            # Insert evolution event
            insert_query = '''
                INSERT INTO character_evolution_timeline 
                (character_id, scene_id, evolution_type, title, description, impact_level, 
                 previous_state, new_state, triggers)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            '''
            
            cur.execute(insert_query, (
                evolution_data.character_id,
                evolution_data.scene_id,
                evolution_data.evolution_type,
                evolution_data.title,
                evolution_data.description,
                evolution_data.impact_level,
                json.dumps(evolution_data.previous_state),
                json.dumps(evolution_data.new_state),
                evolution_data.triggers
            ))
            
            evolution_id = cur.fetchone()[0]
            conn.commit()
            
            logger.info(f"Created evolution event {evolution_id} for character {evolution_data.character_id}")
            return evolution_id
            
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to create evolution event: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create evolution event: {str(e)}")
        finally:
            if conn:
                conn.close()
    
    def create_state_snapshot(self, state_data: CharacterStateCreate) -> int:
        """Create a character state snapshot"""
        conn = None
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()
            
            insert_query = '''
                INSERT INTO character_state_history
                (character_id, evolution_timeline_id, state_snapshot, personality_traits,
                 skills, emotional_state, relationships_snapshot, story_arc_position,
                 character_level, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            '''
            
            cur.execute(insert_query, (
                state_data.character_id,
                state_data.evolution_timeline_id,
                json.dumps(state_data.state_snapshot),
                json.dumps(state_data.personality_traits),
                json.dumps(state_data.skills),
                json.dumps(state_data.emotional_state),
                json.dumps(state_data.relationships_snapshot),
                state_data.story_arc_position,
                state_data.character_level,
                state_data.notes
            ))
            
            state_id = cur.fetchone()[0]
            conn.commit()
            
            logger.info(f"Created state snapshot {state_id} for character {state_data.character_id}")
            return state_id
            
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to create state snapshot: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create state snapshot: {str(e)}")
        finally:
            if conn:
                conn.close()
    
    def get_character_timeline(self, character_id: int) -> CharacterTimelineResponse:
        """Get complete evolution timeline for a character"""
        conn = None
        try:
            conn = self.get_db_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get character info
            cur.execute("SELECT name FROM anime_characters WHERE id = %s", (character_id,))
            character_result = cur.fetchone()
            if not character_result:
                raise HTTPException(status_code=404, detail="Character not found")
            character_name = character_result['name']
            
            # Get evolution timeline
            timeline_query = '''
                SELECT id, character_id, scene_id, evolution_type, title, description,
                       impact_level, previous_state, new_state, triggers, timestamp, created_at
                FROM character_evolution_timeline
                WHERE character_id = %s
                ORDER BY timestamp ASC
            '''
            
            cur.execute(timeline_query, (character_id,))
            timeline_rows = cur.fetchall()
            
            timeline = []
            total_evolution_points = 0
            
            for row in timeline_rows:
                evolution = CharacterEvolutionResponse(
                    id=row['id'],
                    character_id=row['character_id'],
                    scene_id=row['scene_id'],
                    evolution_type=row['evolution_type'],
                    title=row['title'],
                    description=row['description'],
                    impact_level=row['impact_level'],
                    previous_state=row['previous_state'] or {},
                    new_state=row['new_state'] or {},
                    triggers=row['triggers'] or [],
                    timestamp=row['timestamp'],
                    created_at=row['created_at']
                )
                timeline.append(evolution)
                total_evolution_points += row['impact_level']
            
            # Get current state (most recent)
            cur.execute('''
                SELECT state_snapshot FROM character_state_history
                WHERE character_id = %s
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (character_id,))
            
            current_state_row = cur.fetchone()
            current_state = current_state_row['state_snapshot'] if current_state_row else None
            
            return CharacterTimelineResponse(
                character_id=character_id,
                character_name=character_name,
                timeline=timeline,
                current_state=current_state,
                total_evolution_points=total_evolution_points
            )
            
        except psycopg2.Error as e:
            logger.error(f"Failed to get character timeline: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get character timeline: {str(e)}")
        finally:
            if conn:
                conn.close()
    
    def track_emotional_impact(self, impact_data: EmotionalImpactCreate) -> int:
        """Track emotional impact of an event on a character"""
        conn = None
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()
            
            insert_query = '''
                INSERT INTO emotional_impact_tracking
                (character_id, scene_id, evolution_timeline_id, trigger_event,
                 trigger_character_id, emotional_response, intensity_level,
                 duration_category, baseline_impact, coping_mechanism,
                 long_term_effects, recovery_timeline)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            '''
            
            cur.execute(insert_query, (
                impact_data.character_id,
                impact_data.scene_id,
                impact_data.evolution_timeline_id,
                impact_data.trigger_event,
                impact_data.trigger_character_id,
                json.dumps(impact_data.emotional_response),
                impact_data.intensity_level,
                impact_data.duration_category,
                json.dumps(impact_data.baseline_impact),
                impact_data.coping_mechanism,
                json.dumps(impact_data.long_term_effects),
                impact_data.recovery_timeline
            ))
            
            impact_id = cur.fetchone()[0]
            conn.commit()
            
            logger.info(f"Tracked emotional impact {impact_id} for character {impact_data.character_id}")
            return impact_id
            
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to track emotional impact: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to track emotional impact: {str(e)}")
        finally:
            if conn:
                conn.close()

# Initialize service
evolution_service = CharacterEvolutionService()

# API Endpoints
@app.post("/api/characters/evolution/", response_model=dict)
async def create_character_evolution(evolution_data: CharacterEvolutionCreate):
    """Create a new character evolution event"""
    try:
        evolution_id = evolution_service.create_evolution_event(evolution_data)
        return {"success": True, "evolution_id": evolution_id, "message": "Evolution event created successfully"}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in create_character_evolution: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/characters/timeline/{character_id}", response_model=CharacterTimelineResponse)
async def get_character_timeline(character_id: int):
    """Get complete character evolution timeline"""
    return evolution_service.get_character_timeline(character_id)

@app.post("/api/characters/state/", response_model=dict)
async def create_character_state(state_data: CharacterStateCreate):
    """Create a character state snapshot"""
    try:
        state_id = evolution_service.create_state_snapshot(state_data)
        return {"success": True, "state_id": state_id, "message": "State snapshot created successfully"}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in create_character_state: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/characters/emotional-impact/", response_model=dict)
async def track_emotional_impact(impact_data: EmotionalImpactCreate):
    """Track emotional impact of an event"""
    try:
        impact_id = evolution_service.track_emotional_impact(impact_data)
        return {"success": True, "impact_id": impact_id, "message": "Emotional impact tracked successfully"}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in track_emotional_impact: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/characters/{character_id}/relationships", response_model=dict)
async def get_character_relationships(character_id: int):
    """Get current relationship dynamics for a character"""
    conn = None
    try:
        conn = evolution_service.get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        query = '''
            SELECT cr.id, cr.relationship_type, cr.description,
                   c1.name as character1_name, c2.name as character2_name,
                   crd.relationship_strength, crd.relationship_status,
                   crd.interaction_frequency, crd.emotional_intensity,
                   crd.conflict_level, crd.trust_level, crd.dependency_level,
                   crd.recent_interactions, crd.relationship_milestones,
                   crd.timestamp as last_updated
            FROM character_relationships cr
            JOIN anime_characters c1 ON cr.character1_id = c1.id
            JOIN anime_characters c2 ON cr.character2_id = c2.id
            LEFT JOIN character_relationship_dynamics crd ON cr.id = crd.relationship_id
            WHERE cr.character1_id = %s OR cr.character2_id = %s
            ORDER BY crd.timestamp DESC
        '''
        
        cur.execute(query, (character_id, character_id))
        relationships = cur.fetchall()
        
        return {"success": True, "relationships": [dict(row) for row in relationships]}
        
    except psycopg2.Error as e:
        logger.error(f"Failed to get character relationships: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get relationships: {str(e)}")
    finally:
        if conn:
            conn.close()

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "character_evolution", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Character Evolution Service...")
    uvicorn.run(app, host="127.0.0.1", port=8350, log_level="info")