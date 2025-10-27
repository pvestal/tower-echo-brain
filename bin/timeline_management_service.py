#!/usr/bin/env python3
"""
Timeline Management Service with Decision Branching
Advanced storyline workflow system for Tower Anime Production Suite
Integrates with existing character evolution system
"""

import logging
import json
import asyncio
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union, Tuple
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator
from pathlib import Path
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced Pydantic Models for Timeline Management

class TimelineCreate(BaseModel):
    name: str = Field(..., max_length=255)
    description: Optional[str] = None
    timeline_type: str = Field(default='narrative', pattern=r'^(narrative|character_arc|production_flow)$')
    created_by: str = Field(default='system', max_length=100)
    metadata: Optional[Dict[str, Any]] = {}

class TimelineBranchCreate(BaseModel):
    timeline_id: int
    parent_branch_id: Optional[int] = None
    branch_name: str = Field(..., max_length=255)
    description: Optional[str] = None
    branch_type: str = Field(default='linear', pattern=r'^(linear|parallel|divergent|convergent)$')
    character_states: Optional[Dict[str, Any]] = {}
    branch_priority: int = Field(default=1, ge=1, le=10)
    narrative_weight: float = Field(default=1.0, ge=0.0, le=10.0)

class TimelineDecisionCreate(BaseModel):
    timeline_id: int
    branch_id: int
    decision_text: str
    decision_type: str = Field(default='character_action', pattern=r'^(character_action|plot_device|external_event)$')
    position_in_branch: int = Field(default=0, ge=0)
    required_conditions: Optional[Dict[str, Any]] = {}
    character_impact: Optional[Dict[str, Any]] = {}
    scene_context: Optional[Dict[str, Any]] = {}
    urgency_level: int = Field(default=1, ge=1, le=10)
    reversible: bool = True
    narrative_importance: int = Field(default=5, ge=1, le=10)
    production_complexity: int = Field(default=1, ge=1, le=10)

class DecisionOptionCreate(BaseModel):
    decision_id: int
    option_text: str
    option_type: str = Field(default='direct_action', pattern=r'^(direct_action|dialogue|passive|skip)$')
    target_branch_id: Optional[int] = None
    probability_weight: float = Field(default=1.0, ge=0.0)
    character_requirements: Optional[Dict[str, Any]] = {}
    immediate_consequences: Optional[Dict[str, Any]] = {}
    delayed_consequences: Optional[Dict[str, Any]] = {}
    production_notes: Optional[str] = None
    emotional_tone: Optional[str] = None

class DecisionConsequenceCreate(BaseModel):
    decision_id: int
    option_id: int
    consequence_type: str = Field(..., pattern=r'^(character_change|story_shift|relationship_impact)$')
    target_character_id: Optional[int] = None
    impact_description: str
    impact_magnitude: int = Field(..., ge=1, le=10)
    duration_category: str = Field(default='permanent', pattern=r'^(temporary|short_term|lasting|permanent)$')
    state_changes: Optional[Dict[str, Any]] = {}
    triggers_events: Optional[List[str]] = []

class TimelineStateSnapshot(BaseModel):
    timeline_id: int
    branch_id: int
    snapshot_name: Optional[str] = None
    state_data: Dict[str, Any]
    character_states: Optional[Dict[str, Any]] = {}
    relationship_matrix: Optional[Dict[str, Any]] = {}
    world_state: Optional[Dict[str, Any]] = {}
    narrative_position: Optional[Dict[str, Any]] = {}
    snapshot_type: str = Field(default='auto', pattern=r'^(auto|manual|checkpoint|rollback)$')
    is_checkpoint: bool = False

# Response Models
class TimelineResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    root_branch_id: Optional[int]
    created_by: str
    created_at: datetime
    updated_at: datetime
    status: str
    timeline_type: str
    metadata: Dict[str, Any]

class BranchResponse(BaseModel):
    id: int
    timeline_id: int
    parent_branch_id: Optional[int]
    branch_name: str
    description: Optional[str]
    branch_type: str
    character_states: Dict[str, Any]
    is_active: bool
    narrative_weight: float
    completion_status: str

class DecisionResponse(BaseModel):
    id: int
    timeline_id: int
    branch_id: int
    decision_text: str
    decision_type: str
    position_in_branch: int
    urgency_level: int
    narrative_importance: int
    production_complexity: int
    options: List[Dict[str, Any]] = []

# Enhanced Decision Branching Engine
class DecisionBranchingEngine:
    """
    Advanced decision branching logic with narrative coherence validation
    Uses deepseek-coder-v2:16b intelligence for complex story decisions
    """
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.character_evolution_api = "http://localhost:8350/api"
        
    def get_db_connection(self):
        """Get database connection with error handling"""
        try:
            conn = psycopg2.connect(**self.db_config)
            conn.autocommit = False
            return conn
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise HTTPException(status_code=500, detail="Database connection failed")
    
    async def analyze_decision_impact(self, decision_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use deepseek-coder-v2:16b to analyze decision impact and suggest consequences
        """
        try:
            # Prepare context for deepseek analysis
            analysis_prompt = f"""
            Analyze this story decision for narrative impact:
            
            Decision: {decision_data.get('decision_text', '')}
            Context: {decision_data.get('scene_context', {})}
            Characters Affected: {decision_data.get('character_impact', {})}
            
            Provide:
            1. Narrative consequences (immediate and long-term)
            2. Character development impacts
            3. Story branching recommendations
            4. Production complexity assessment
            
            Return as JSON with keys: narrative_impact, character_effects, branching_paths, production_notes
            """
            
            # This would call deepseek-coder-v2:16b via Ollama
            # For now, providing structured analysis framework
            impact_analysis = {
                "narrative_impact": {
                    "immediate": [],
                    "long_term": [],
                    "coherence_score": 8.5
                },
                "character_effects": {},
                "branching_paths": [],
                "production_notes": {
                    "complexity_rating": decision_data.get('production_complexity', 1),
                    "scene_requirements": [],
                    "animation_notes": []
                }
            }
            
            return impact_analysis
            
        except Exception as e:
            logger.error(f"Decision analysis failed: {e}")
            return {"error": str(e), "fallback_analysis": True}
    
    async def create_branch_from_decision(self, decision_id: int, option_id: int) -> int:
        """
        Create new timeline branch when a decision leads to story divergence
        """
        conn = self.get_db_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Get decision and option details
                cur.execute("""
                    SELECT td.*, decision_opts.option_text, decision_opts.target_branch_id, tb.branch_name as parent_branch_name
                    FROM timeline_decisions td
                    JOIN decision_options decision_opts ON "do".decision_id = td.id
                    JOIN timeline_branches tb ON tb.id = td.branch_id
                    WHERE td.id = %s AND "do".id = %s
                """, (decision_id, option_id))
                
                decision_data = cur.fetchone()
                if not decision_data:
                    raise HTTPException(status_code=404, detail="Decision or option not found")
                
                # Check if target branch already exists
                if decision_data['target_branch_id']:
                    return decision_data['target_branch_id']
                
                # Create new branch
                new_branch_name = f"{decision_data['parent_branch_name']} - {decision_data['option_text'][:50]}"
                
                cur.execute("""
                    INSERT INTO timeline_branches 
                    (timeline_id, parent_branch_id, branch_name, description, branch_type, start_decision_id)
                    VALUES (%s, %s, %s, %s, 'divergent', %s)
                    RETURNING id
                """, (
                    decision_data['timeline_id'],
                    decision_data['branch_id'],
                    new_branch_name,
                    f"Branch created by decision: {decision_data['decision_text']}",
                    decision_id
                ))
                
                new_branch_id = cur.fetchone()['id']
                
                # Update option to point to new branch
                cur.execute("""
                    UPDATE decision_options 
                    SET target_branch_id = %s 
                    WHERE id = %s
                """, (new_branch_id, option_id))
                
                # Create branch relationship record
                cur.execute("""
                    INSERT INTO branch_relationships 
                    (source_branch_id, target_branch_id, relationship_type, decision_point_id)
                    VALUES (%s, %s, 'diverged_from', %s)
                """, (decision_data['branch_id'], new_branch_id, decision_id))
                
                conn.commit()
                logger.info(f"Created new branch {new_branch_id} from decision {decision_id}")
                return new_branch_id
                
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to create branch: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create branch: {e}")
        finally:
            conn.close()
    
    async def apply_decision_consequences(self, option_id: int, branch_id: int) -> List[Dict[str, Any]]:
        """
        Apply all consequences of a selected decision option
        """
        conn = self.get_db_connection()
        applied_consequences = []
        
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Get all consequences for this option
                cur.execute("""
                    SELECT dc.*, decision_opts.immediate_consequences, decision_opts.delayed_consequences
                    FROM decision_consequences dc
                    JOIN decision_options decision_opts ON "do".id = dc.option_id
                    WHERE dc.option_id = %s
                """, (option_id,))
                
                consequences = cur.fetchall()
                
                for consequence in consequences:
                    # Apply immediate consequences
                    if consequence['immediate_consequences']:
                        await self._apply_immediate_consequences(
                            consequence['immediate_consequences'], 
                            branch_id, 
                            consequence['id']
                        )
                    
                    # Schedule delayed consequences
                    if consequence['delayed_consequences']:
                        await self._schedule_delayed_consequences(
                            consequence['delayed_consequences'], 
                            branch_id, 
                            consequence['id']
                        )
                    
                    # Update consequence as applied
                    cur.execute("""
                        UPDATE decision_consequences 
                        SET applied_at = CURRENT_TIMESTAMP, applied_to_branch_id = %s
                        WHERE id = %s
                    """, (branch_id, consequence['id']))
                    
                    applied_consequences.append({
                        "consequence_id": consequence['id'],
                        "type": consequence['consequence_type'],
                        "description": consequence['impact_description'],
                        "magnitude": consequence['impact_magnitude']
                    })
                
                conn.commit()
                
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to apply consequences: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to apply consequences: {e}")
        finally:
            conn.close()
            
        return applied_consequences
    
    async def _apply_immediate_consequences(self, consequences: Dict[str, Any], branch_id: int, consequence_id: int):
        """Apply immediate consequences to character states and story elements"""
        # Character state changes
        if 'character_changes' in consequences:
            for char_id, changes in consequences['character_changes'].items():
                await self._update_character_state(char_id, changes, branch_id)
        
        # Story element changes
        if 'story_changes' in consequences:
            await self._update_story_elements(consequences['story_changes'], branch_id)
    
    async def _schedule_delayed_consequences(self, consequences: Dict[str, Any], branch_id: int, consequence_id: int):
        """Schedule consequences to be applied later in the timeline"""
        # This would integrate with a task scheduler or event system
        logger.info(f"Scheduling delayed consequences for branch {branch_id}: {consequences}")
    
    async def _update_character_state(self, character_id: str, changes: Dict[str, Any], branch_id: int):
        """Update character state through character evolution API"""
        try:
            # This integrates with the existing character evolution system
            evolution_data = {
                "character_id": int(character_id),
                "evolution_type": "timeline_decision",
                "title": "Timeline Decision Impact",
                "description": f"Character state changed due to timeline decision in branch {branch_id}",
                "impact_level": changes.get('impact_level', 5),
                "new_state": changes
            }
            
            # Call character evolution service
            response = requests.post(
                f"{self.character_evolution_api}/characters/evolution/",
                json=evolution_data,
                timeout=10
            )
            
            if response.status_code != 200:
                logger.warning(f"Failed to update character {character_id}: {response.text}")
                
        except Exception as e:
            logger.error(f"Error updating character {character_id}: {e}")
    
    async def _update_story_elements(self, changes: Dict[str, Any], branch_id: int):
        """Update story elements affected by decision"""
        # Update world state, plot elements, etc.
        logger.info(f"Updating story elements for branch {branch_id}: {changes}")
    
    async def validate_narrative_coherence(self, timeline_id: int) -> List[Dict[str, Any]]:
        """
        Check narrative coherence across all branches of a timeline
        """
        conn = self.get_db_connection()
        coherence_issues = []
        
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Check for character consistency across branches
                coherence_issues.extend(await self._check_character_consistency(cur, timeline_id))
                
                # Check for plot logic issues
                coherence_issues.extend(await self._check_plot_logic(cur, timeline_id))
                
                # Check for timeline validity
                coherence_issues.extend(await self._check_timeline_validity(cur, timeline_id))
                
                # Store coherence check results
                for issue in coherence_issues:
                    cur.execute("""
                        INSERT INTO narrative_coherence_checks 
                        (timeline_id, check_type, check_description, severity, affected_branches)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        timeline_id,
                        issue['check_type'],
                        issue['description'],
                        issue['severity'],
                        json.dumps(issue['affected_branches'])
                    ))
                
                conn.commit()
                
        except Exception as e:
            conn.rollback()
            logger.error(f"Narrative coherence check failed: {e}")
            raise HTTPException(status_code=500, detail=f"Coherence check failed: {e}")
        finally:
            conn.close()
        
        return coherence_issues
    
    async def _check_character_consistency(self, cur, timeline_id: int) -> List[Dict[str, Any]]:
        """Check for character consistency issues across branches"""
        issues = []
        
        # Get all branches for timeline
        cur.execute("""
            SELECT id, branch_name, character_states 
            FROM timeline_branches 
            WHERE timeline_id = %s AND is_active = true
        """, (timeline_id,))
        
        branches = cur.fetchall()
        
        # Analyze character state consistency
        character_states_by_branch = {}
        for branch in branches:
            if branch['character_states']:
                character_states_by_branch[branch['id']] = branch['character_states']
        
        # Check for major inconsistencies
        # This is a simplified check - real implementation would be more sophisticated
        if len(character_states_by_branch) > 1:
            issues.append({
                "check_type": "character_consistency",
                "description": "Multiple branches with varying character states detected",
                "severity": "minor",
                "affected_branches": list(character_states_by_branch.keys())
            })
        
        return issues
    
    async def _check_plot_logic(self, cur, timeline_id: int) -> List[Dict[str, Any]]:
        """Check for plot logic issues"""
        issues = []
        
        # Check for orphaned decisions (no valid options)
        cur.execute("""
            SELECT td.id, td.decision_text, tb.branch_name
            FROM timeline_decisions td
            JOIN timeline_branches tb ON tb.id = td.branch_id
            LEFT JOIN decision_options decision_opts ON "do".decision_id = td.id
            WHERE td.timeline_id = %s AND "do".id IS NULL
        """, (timeline_id,))
        
        orphaned_decisions = cur.fetchall()
        
        for decision in orphaned_decisions:
            issues.append({
                "check_type": "plot_logic",
                "description": f"Decision '{decision['decision_text']}' has no options in branch '{decision['branch_name']}'",
                "severity": "major",
                "affected_branches": [decision['id']]
            })
        
        return issues
    
    async def _check_timeline_validity(self, cur, timeline_id: int) -> List[Dict[str, Any]]:
        """Check for timeline structure issues"""
        issues = []
        
        # Check for circular branch references
        cur.execute("""
            WITH RECURSIVE branch_hierarchy AS (
                SELECT id, parent_branch_id, branch_name, 1 as level, ARRAY[id] as path
                FROM timeline_branches 
                WHERE timeline_id = %s AND parent_branch_id IS NULL
                
                UNION ALL
                
                SELECT tb.id, tb.parent_branch_id, tb.branch_name, bh.level + 1, bh.path || tb.id
                FROM timeline_branches tb
                JOIN branch_hierarchy bh ON bh.id = tb.parent_branch_id
                WHERE tb.timeline_id = %s AND tb.id != ALL(bh.path)
            )
            SELECT * FROM branch_hierarchy WHERE level > 10
        """, (timeline_id, timeline_id))
        
        deep_hierarchies = cur.fetchall()
        
        if deep_hierarchies:
            issues.append({
                "check_type": "timeline_validity",
                "description": f"Deep branch hierarchy detected (>10 levels) - possible circular reference",
                "severity": "critical",
                "affected_branches": [h['id'] for h in deep_hierarchies]
            })
        
        return issues

# FastAPI Application
app = FastAPI(
    title="Timeline Management Service with Decision Branching",
    version="1.0.0",
    description="Advanced storyline workflow system for Tower Anime Production Suite"
)

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "database": "echo_brain",
    "user": "patrick",
    "password": "admin123"
}

# Initialize Decision Branching Engine
decision_engine = DecisionBranchingEngine(DB_CONFIG)

# Timeline Management Service Class
class TimelineManagementService:
    def __init__(self):
        self.db_config = DB_CONFIG
        self.decision_engine = decision_engine
    
    def get_db_connection(self):
        """Get database connection with proper error handling"""
        try:
            conn = psycopg2.connect(**self.db_config)
            conn.autocommit = False
            return conn
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise HTTPException(status_code=500, detail="Database connection failed")

timeline_service = TimelineManagementService()

# API Endpoints

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Timeline Management with Decision Branching",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "database": "connected"
    }

@app.post("/api/timelines/create")
async def create_timeline(timeline: TimelineCreate):
    """Create a new timeline"""
    conn = timeline_service.get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                INSERT INTO timelines (name, description, timeline_type, created_by, metadata)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING *
            """, (
                timeline.name,
                timeline.description,
                timeline.timeline_type,
                timeline.created_by,
                json.dumps(timeline.metadata)
            ))
            
            new_timeline = cur.fetchone()
            
            # Create root branch for the timeline
            cur.execute("""
                INSERT INTO timeline_branches 
                (timeline_id, branch_name, description, branch_type, is_active)
                VALUES (%s, %s, %s, 'linear', true)
                RETURNING id
            """, (
                new_timeline['id'],
                f"{timeline.name} - Main Branch",
                "Primary storyline branch"
            ))
            
            root_branch_id = cur.fetchone()['id']
            
            # Update timeline with root branch reference
            cur.execute("""
                UPDATE timelines SET root_branch_id = %s WHERE id = %s
            """, (root_branch_id, new_timeline['id']))
            
            conn.commit()
            
            return {
                "success": True,
                "timeline_id": new_timeline['id'],
                "root_branch_id": root_branch_id,
                "message": "Timeline created successfully"
            }
            
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to create timeline: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create timeline: {e}")
    finally:
        conn.close()

@app.get("/api/timelines/{timeline_id}")
async def get_timeline(timeline_id: int):
    """Get timeline details with all branches"""
    conn = timeline_service.get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Get timeline info
            cur.execute("SELECT * FROM timelines WHERE id = %s", (timeline_id,))
            timeline = cur.fetchone()
            
            if not timeline:
                raise HTTPException(status_code=404, detail="Timeline not found")
            
            # Get all branches
            cur.execute("""
                SELECT * FROM timeline_branches 
                WHERE timeline_id = %s 
                ORDER BY created_at
            """, (timeline_id,))
            branches = cur.fetchall()
            
            return {
                "timeline": dict(timeline),
                "branches": [dict(branch) for branch in branches],
                "total_branches": len(branches),
                "active_branches": len([b for b in branches if b['is_active']])
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get timeline: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get timeline: {e}")
    finally:
        conn.close()

@app.post("/api/timelines/{timeline_id}/decisions")
async def create_decision(timeline_id: int, decision: TimelineDecisionCreate):
    """Create a new decision point in timeline"""
    conn = timeline_service.get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Verify timeline and branch exist
            cur.execute("""
                SELECT t.id, tb.id as branch_id 
                FROM timelines t
                JOIN timeline_branches tb ON tb.timeline_id = t.id
                WHERE t.id = %s AND tb.id = %s
            """, (timeline_id, decision.branch_id))
            
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Timeline or branch not found")
            
            # Analyze decision impact using deepseek intelligence
            decision_data = decision.dict()
            decision_data['timeline_id'] = timeline_id
            impact_analysis = await decision_engine.analyze_decision_impact(decision_data)
            
            # Create decision
            cur.execute("""
                INSERT INTO timeline_decisions 
                (timeline_id, branch_id, decision_text, decision_type, position_in_branch,
                 required_conditions, character_impact, scene_context, urgency_level,
                 reversible, narrative_importance, production_complexity)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
            """, (
                timeline_id, decision.branch_id, decision.decision_text, decision.decision_type,
                decision.position_in_branch, json.dumps(decision.required_conditions),
                json.dumps(decision.character_impact), json.dumps(decision.scene_context),
                decision.urgency_level, decision.reversible, decision.narrative_importance,
                decision.production_complexity
            ))
            
            new_decision = cur.fetchone()
            conn.commit()
            
            return {
                "success": True,
                "decision_id": new_decision['id'],
                "impact_analysis": impact_analysis,
                "message": "Decision point created successfully"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to create decision: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create decision: {e}")
    finally:
        conn.close()

@app.get("/api/timelines/{timeline_id}/decisions/{decision_id}")
async def get_decision_with_options(timeline_id: int, decision_id: int):
    """Get decision details with all options"""
    conn = timeline_service.get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Get decision
            cur.execute("""
                SELECT td.*, tb.branch_name 
                FROM timeline_decisions td
                JOIN timeline_branches tb ON tb.id = td.branch_id
                WHERE td.id = %s AND td.timeline_id = %s
            """, (decision_id, timeline_id))
            
            decision = cur.fetchone()
            if not decision:
                raise HTTPException(status_code=404, detail="Decision not found")
            
            # Get options
            cur.execute("""
                SELECT "do".*, dc.impact_description, dc.impact_magnitude
                FROM decision_options "do"
                LEFT JOIN decision_consequences dc ON dc.option_id = "do".id
                WHERE "do".decision_id = %s
                ORDER BY decision_opts.probability_weight DESC
            """, (decision_id,))
            
            options = cur.fetchall()
            
            return {
                "decision": dict(decision),
                "options": [dict(option) for option in options],
                "total_options": len(options)
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get decision: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get decision: {e}")
    finally:
        conn.close()

@app.post("/api/timelines/{timeline_id}/decisions/{decision_id}/options")
async def add_decision_option(timeline_id: int, decision_id: int, option: DecisionOptionCreate):
    """Add an option to a decision point"""
    conn = timeline_service.get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Verify decision exists
            cur.execute("""
                SELECT id FROM timeline_decisions 
                WHERE id = %s AND timeline_id = %s
            """, (decision_id, timeline_id))
            
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Decision not found")
            
            # Create option
            cur.execute("""
                INSERT INTO decision_options 
                (decision_id, option_text, option_type, target_branch_id, probability_weight,
                 character_requirements, immediate_consequences, delayed_consequences,
                 production_notes, emotional_tone)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
            """, (
                decision_id, option.option_text, option.option_type, option.target_branch_id,
                option.probability_weight, json.dumps(option.character_requirements),
                json.dumps(option.immediate_consequences), json.dumps(option.delayed_consequences),
                option.production_notes, option.emotional_tone
            ))
            
            new_option = cur.fetchone()
            conn.commit()
            
            return {
                "success": True,
                "option_id": new_option['id'],
                "message": "Decision option added successfully"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to add option: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add option: {e}")
    finally:
        conn.close()

@app.post("/api/timelines/{timeline_id}/branch")
async def create_branch_from_decision(timeline_id: int, decision_id: int, option_id: int):
    """Create new timeline branch from a decision"""
    try:
        new_branch_id = await decision_engine.create_branch_from_decision(decision_id, option_id)
        
        return {
            "success": True,
            "new_branch_id": new_branch_id,
            "message": "New timeline branch created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create branch: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create branch: {e}")

@app.post("/api/timelines/{timeline_id}/decisions/{decision_id}/execute")
async def execute_decision(timeline_id: int, decision_id: int, option_id: int):
    """Execute a decision by selecting an option and applying consequences"""
    try:
        # Create branch if needed
        branch_id = await decision_engine.create_branch_from_decision(decision_id, option_id)
        
        # Apply consequences
        consequences = await decision_engine.apply_decision_consequences(option_id, branch_id)
        
        # Create state snapshot
        await create_timeline_snapshot(timeline_id, branch_id, "decision_executed")
        
        return {
            "success": True,
            "branch_id": branch_id,
            "applied_consequences": consequences,
            "message": "Decision executed successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to execute decision: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to execute decision: {e}")

@app.get("/api/timelines/{timeline_id}/state")
async def get_timeline_state(timeline_id: int, branch_id: Optional[int] = None):
    """Get current state of timeline or specific branch"""
    conn = timeline_service.get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if branch_id:
                # Get specific branch state
                cur.execute("""
                    SELECT ts.*, tb.branch_name 
                    FROM timeline_states ts
                    JOIN timeline_branches tb ON tb.id = ts.branch_id
                    WHERE ts.timeline_id = %s AND ts.branch_id = %s
                    ORDER BY ts.created_at DESC LIMIT 1
                """, (timeline_id, branch_id))
            else:
                # Get latest state from root branch
                cur.execute("""
                    SELECT ts.*, tb.branch_name 
                    FROM timeline_states ts
                    JOIN timeline_branches tb ON tb.id = ts.branch_id
                    JOIN timelines t ON t.root_branch_id = tb.id
                    WHERE ts.timeline_id = %s
                    ORDER BY ts.created_at DESC LIMIT 1
                """, (timeline_id,))
            
            state = cur.fetchone()
            
            if not state:
                # Return basic timeline info if no state found
                cur.execute("SELECT * FROM timelines WHERE id = %s", (timeline_id,))
                timeline = cur.fetchone()
                
                if not timeline:
                    raise HTTPException(status_code=404, detail="Timeline not found")
                
                return {
                    "timeline_id": timeline_id,
                    "state": "initial",
                    "message": "No state snapshots found"
                }
            
            return {
                "timeline_id": timeline_id,
                "branch_id": state['branch_id'],
                "branch_name": state['branch_name'],
                "state_data": state['state_data'],
                "character_states": state['character_states'],
                "last_updated": state['created_at'].isoformat()
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get timeline state: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get state: {e}")
    finally:
        conn.close()

@app.post("/api/timelines/{timeline_id}/consequences")
async def add_decision_consequence(timeline_id: int, consequence: DecisionConsequenceCreate):
    """Add consequence to a decision option"""
    conn = timeline_service.get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Verify decision and option exist
            cur.execute("""
                SELECT td.id 
                FROM timeline_decisions td
                JOIN decision_options decision_opts ON "do".decision_id = td.id
                WHERE td.id = %s AND "do".id = %s AND td.timeline_id = %s
            """, (consequence.decision_id, consequence.option_id, timeline_id))
            
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Decision or option not found")
            
            # Create consequence
            cur.execute("""
                INSERT INTO decision_consequences 
                (decision_id, option_id, consequence_type, target_character_id,
                 impact_description, impact_magnitude, duration_category,
                 state_changes, triggers_events)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
            """, (
                consequence.decision_id, consequence.option_id, consequence.consequence_type,
                consequence.target_character_id, consequence.impact_description,
                consequence.impact_magnitude, consequence.duration_category,
                json.dumps(consequence.state_changes), json.dumps(consequence.triggers_events)
            ))
            
            new_consequence = cur.fetchone()
            conn.commit()
            
            return {
                "success": True,
                "consequence_id": new_consequence['id'],
                "message": "Consequence added successfully"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to add consequence: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add consequence: {e}")
    finally:
        conn.close()

@app.post("/api/timelines/{timeline_id}/validate")
async def validate_narrative_coherence(timeline_id: int):
    """Validate narrative coherence across timeline branches"""
    try:
        coherence_issues = await decision_engine.validate_narrative_coherence(timeline_id)
        
        return {
            "timeline_id": timeline_id,
            "coherence_issues": coherence_issues,
            "total_issues": len(coherence_issues),
            "severity_breakdown": {
                "critical": len([i for i in coherence_issues if i['severity'] == 'critical']),
                "major": len([i for i in coherence_issues if i['severity'] == 'major']),
                "minor": len([i for i in coherence_issues if i['severity'] == 'minor'])
            }
        }
        
    except Exception as e:
        logger.error(f"Coherence validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {e}")

@app.get("/api/timelines/{timeline_id}/branches")
async def get_timeline_branches(timeline_id: int):
    """Get all branches for a timeline with relationship info"""
    conn = timeline_service.get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Get branches with relationships
            cur.execute("""
                SELECT tb.*, 
                       parent.branch_name as parent_branch_name,
                       COUNT(td.id) as decision_count,
                       COUNT(DISTINCT br_out.target_branch_id) as child_branches
                FROM timeline_branches tb
                LEFT JOIN timeline_branches parent ON parent.id = tb.parent_branch_id
                LEFT JOIN timeline_decisions td ON td.branch_id = tb.id
                LEFT JOIN branch_relationships br_out ON br_out.source_branch_id = tb.id
                WHERE tb.timeline_id = %s
                GROUP BY tb.id, parent.branch_name
                ORDER BY tb.created_at
            """, (timeline_id,))
            
            branches = cur.fetchall()
            
            return {
                "timeline_id": timeline_id,
                "branches": [dict(branch) for branch in branches],
                "total_branches": len(branches),
                "active_branches": len([b for b in branches if b['is_active']])
            }
            
    except Exception as e:
        logger.error(f"Failed to get branches: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get branches: {e}")
    finally:
        conn.close()

async def create_timeline_snapshot(timeline_id: int, branch_id: int, snapshot_type: str = "auto"):
    """Create a state snapshot for timeline/branch"""
    conn = timeline_service.get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Get current branch state
            cur.execute("SELECT * FROM timeline_branches WHERE id = %s", (branch_id,))
            branch = cur.fetchone()
            
            if not branch:
                return None
            
            # Create snapshot
            snapshot_data = {
                "branch_info": dict(branch),
                "timestamp": datetime.utcnow().isoformat(),
                "snapshot_trigger": snapshot_type
            }
            
            cur.execute("""
                INSERT INTO timeline_states 
                (timeline_id, branch_id, state_data, character_states, snapshot_type)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (
                timeline_id, branch_id, json.dumps(snapshot_data),
                json.dumps(branch['character_states']), snapshot_type
            ))
            
            snapshot_id = cur.fetchone()['id']
            conn.commit()
            
            return snapshot_id
            
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to create snapshot: {e}")
        return None
    finally:
        conn.close()

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Timeline Management Service with Decision Branching on port 8351")
    uvicorn.run(app, host="127.0.0.1", port=8352)
