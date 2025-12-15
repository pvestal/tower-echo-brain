#!/usr/bin/env python3
"""
Integration script to add character evolution endpoints to the main Enhanced Echo Service
"""

import re

def integrate_character_evolution():
    # Read the enhanced service file
    with open('/opt/tower-echo-brain/bin/echo_enhanced_service.py', 'r') as f:
        content = f.read()
    
    # Character evolution imports to add
    character_imports = '''
# Character Evolution imports
from bin.character_evolution_service import (
    CharacterEvolutionCreate, CharacterStateCreate, RelationshipDynamicsCreate,
    EmotionalImpactCreate, CharacterEvolutionResponse, CharacterTimelineResponse,
    CharacterEvolutionService
)
'''
    
    # Character evolution endpoints to add
    character_endpoints = '''

# Initialize Character Evolution Service
character_evolution_service = CharacterEvolutionService()

# Character Evolution Endpoints
@app.post("/api/characters/evolution/", response_model=dict)
async def create_character_evolution(evolution_data: CharacterEvolutionCreate):
    """Create a new character evolution event"""
    try:
        evolution_id = character_evolution_service.create_evolution_event(evolution_data)
        return {"success": True, "evolution_id": evolution_id, "message": "Evolution event created successfully"}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in create_character_evolution: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/characters/timeline/{character_id}", response_model=CharacterTimelineResponse)
async def get_character_timeline(character_id: int):
    """Get complete character evolution timeline"""
    return character_evolution_service.get_character_timeline(character_id)

@app.post("/api/characters/state/", response_model=dict)
async def create_character_state(state_data: CharacterStateCreate):
    """Create a character state snapshot"""
    try:
        state_id = character_evolution_service.create_state_snapshot(state_data)
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
        impact_id = character_evolution_service.track_emotional_impact(impact_data)
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
        conn = character_evolution_service.get_db_connection()
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

@app.post("/api/characters/relationships/dynamics", response_model=dict)
async def create_relationship_dynamics(dynamics_data: RelationshipDynamicsCreate):
    """Track relationship dynamics changes"""
    conn = None
    try:
        conn = character_evolution_service.get_db_connection()
        cur = conn.cursor()
        
        insert_query = '''
            INSERT INTO character_relationship_dynamics
            (relationship_id, evolution_timeline_id, relationship_strength,
             relationship_status, interaction_frequency, emotional_intensity,
             conflict_level, trust_level, dependency_level, recent_interactions,
             relationship_milestones, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        '''
        
        import json
        cur.execute(insert_query, (
            dynamics_data.relationship_id,
            dynamics_data.evolution_timeline_id,
            dynamics_data.relationship_strength,
            dynamics_data.relationship_status,
            dynamics_data.interaction_frequency,
            dynamics_data.emotional_intensity,
            dynamics_data.conflict_level,
            dynamics_data.trust_level,
            dynamics_data.dependency_level,
            json.dumps(dynamics_data.recent_interactions),
            dynamics_data.relationship_milestones,
            dynamics_data.notes
        ))
        
        dynamics_id = cur.fetchone()[0]
        conn.commit()
        
        return {"success": True, "dynamics_id": dynamics_id, "message": "Relationship dynamics tracked successfully"}
        
    except psycopg2.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"Failed to track relationship dynamics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to track dynamics: {str(e)}")
    finally:
        if conn:
            conn.close()
'''
    
    # Find the insert position for imports (after existing imports)
    import_pattern = r'(from pathlib import Path\s*\n)'
    if re.search(import_pattern, content):
        content = re.sub(import_pattern, r'\1' + character_imports + '\n', content)
    
    # Find the insert position for endpoints (before the main block)
    main_pattern = r'(if __name__ == "__main__":)'
    if re.search(main_pattern, content):
        content = re.sub(main_pattern, character_endpoints + '\n' + r'\1', content)
    
    # Write the integrated service
    with open('/opt/tower-echo-brain/bin/echo_enhanced_service_with_evolution.py', 'w') as f:
        f.write(content)
    
    print("Character evolution endpoints integrated successfully!")
    print("New service file: echo_enhanced_service_with_evolution.py")

if __name__ == "__main__":
    integrate_character_evolution()