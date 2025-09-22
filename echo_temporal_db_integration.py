#!/usr/bin/env python3
"""
Database integration for Temporal Reasoning and Self-Awareness modules
"""

import asyncio
import asyncpg
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import hashlib

class TemporalDatabaseManager:
    """
    Manages database storage for temporal events and reasoning
    """
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.pool = None
        
    async def initialize(self):
        """Initialize database connection and tables"""
        self.pool = await asyncpg.create_pool(
            host=self.db_config.get('host', 'localhost'),
            database=self.db_config.get('database', 'echo_brain'),
            user=self.db_config.get('user', 'echo'),
            password=self.db_config.get('password', 'echo_password'),
            min_size=1,
            max_size=10
        )
        
        # Create tables if they don't exist
        await self._create_tables()
    
    async def _create_tables(self):
        """Create necessary tables for temporal reasoning"""
        async with self.pool.acquire() as conn:
            # Temporal events table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS temporal_events (
                    id VARCHAR(255) PRIMARY KEY,
                    timeline_id VARCHAR(255) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    description TEXT,
                    event_type VARCHAR(50),
                    causes TEXT[], -- Array of event IDs
                    effects TEXT[], -- Array of event IDs
                    probability FLOAT DEFAULT 1.0,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    validated BOOLEAN DEFAULT FALSE,
                    INDEX idx_timeline (timeline_id),
                    INDEX idx_timestamp (timestamp)
                )
            ''')
            
            # Paradoxes table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS temporal_paradoxes (
                    id SERIAL PRIMARY KEY,
                    event_id VARCHAR(255),
                    timeline_id VARCHAR(255),
                    paradox_type VARCHAR(100),
                    description TEXT,
                    detected_at TIMESTAMP DEFAULT NOW(),
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution TEXT,
                    INDEX idx_event (event_id),
                    INDEX idx_timeline (timeline_id)
                )
            ''')
            
            # Causal chains table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS causal_chains (
                    id SERIAL PRIMARY KEY,
                    start_event_id VARCHAR(255),
                    end_event_id VARCHAR(255),
                    path_events TEXT[], -- Array of event IDs in sequence
                    path_length INT,
                    verified_at TIMESTAMP DEFAULT NOW(),
                    confidence_score FLOAT,
                    INDEX idx_start_end (start_event_id, end_event_id)
                )
            ''')
            
            # Timeline consistency scores
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS timeline_consistency (
                    id SERIAL PRIMARY KEY,
                    timeline_id VARCHAR(255) UNIQUE,
                    consistency_score FLOAT,
                    total_events INT,
                    valid_events INT,
                    last_checked TIMESTAMP DEFAULT NOW(),
                    metadata JSONB
                )
            ''')
            
            # Self-awareness reports
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS self_awareness_reports (
                    id SERIAL PRIMARY KEY,
                    report_type VARCHAR(100),
                    capabilities JSONB,
                    endpoints JSONB,
                    services JSONB,
                    temporal_capable BOOLEAN,
                    resources JSONB,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    query_context TEXT,
                    response_generated TEXT
                )
            ''')
            
            # Capability evolution tracking
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS capability_evolution (
                    id SERIAL PRIMARY KEY,
                    capability_name VARCHAR(255),
                    status VARCHAR(50), -- enabled, disabled, degraded, enhanced
                    version VARCHAR(50),
                    added_at TIMESTAMP,
                    modified_at TIMESTAMP DEFAULT NOW(),
                    description TEXT,
                    dependencies JSONB,
                    performance_metrics JSONB
                )
            ''')
    
    async def save_temporal_event(self, event: Dict) -> bool:
        """Save a temporal event to database"""
        async with self.pool.acquire() as conn:
            try:
                await conn.execute('''
                    INSERT INTO temporal_events 
                    (id, timeline_id, timestamp, description, event_type, causes, effects, probability, metadata, validated)
                    VALUES (, , , , , , , , , 0)
                    ON CONFLICT (id) DO UPDATE SET
                        validated = EXCLUDED.validated,
                        metadata = EXCLUDED.metadata
                ''',
                    event['id'],
                    event.get('timeline_id', 'main'),
                    datetime.fromisoformat(event['timestamp']),
                    event.get('description', ''),
                    event.get('event_type', 'present'),
                    event.get('causes', []),
                    event.get('effects', []),
                    event.get('probability', 1.0),
                    json.dumps(event.get('metadata', {})),
                    event.get('validated', False)
                )
                return True
            except Exception as e:
                print(f"Error saving temporal event: {e}")
                return False
    
    async def save_paradox(self, paradox: Dict) -> int:
        """Save detected paradox to database"""
        async with self.pool.acquire() as conn:
            result = await conn.fetchval('''
                INSERT INTO temporal_paradoxes 
                (event_id, timeline_id, paradox_type, description)
                VALUES (, , , )
                RETURNING id
            ''',
                paradox.get('event_id'),
                paradox.get('timeline_id', 'main'),
                paradox.get('type'),
                paradox.get('message', '')
            )
            return result
    
    async def save_causal_chain(self, chain: Dict) -> int:
        """Save verified causal chain"""
        async with self.pool.acquire() as conn:
            result = await conn.fetchval('''
                INSERT INTO causal_chains 
                (start_event_id, end_event_id, path_events, path_length, confidence_score)
                VALUES (, , , , )
                RETURNING id
            ''',
                chain['start_event_id'],
                chain['end_event_id'],
                chain.get('path', []),
                len(chain.get('path', [])),
                chain.get('confidence_score', 1.0)
            )
            return result
    
    async def update_timeline_consistency(self, timeline_id: str, score: float, total: int, valid: int):
        """Update timeline consistency score"""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO timeline_consistency 
                (timeline_id, consistency_score, total_events, valid_events)
                VALUES (, , , )
                ON CONFLICT (timeline_id) DO UPDATE SET
                    consistency_score = EXCLUDED.consistency_score,
                    total_events = EXCLUDED.total_events,
                    valid_events = EXCLUDED.valid_events,
                    last_checked = NOW()
            ''',
                timeline_id, score, total, valid
            )
    
    async def save_self_awareness_report(self, report: Dict) -> int:
        """Save self-awareness report"""
        async with self.pool.acquire() as conn:
            result = await conn.fetchval('''
                INSERT INTO self_awareness_reports 
                (report_type, capabilities, endpoints, services, temporal_capable, resources, query_context, response_generated)
                VALUES (, , , , , , , )
                RETURNING id
            ''',
                report.get('report_type', 'self_identification'),
                json.dumps(report.get('capabilities', {})),
                json.dumps(report.get('endpoints', {})),
                json.dumps(report.get('services', {})),
                report.get('temporal_capable', False),
                json.dumps(report.get('resources', {})),
                report.get('query_context', ''),
                report.get('response', '')
            )
            return result
    
    async def track_capability_evolution(self, capability: str, status: str, version: str = '1.0.0'):
        """Track capability changes over time"""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO capability_evolution 
                (capability_name, status, version, added_at)
                VALUES (, , , NOW())
                ON CONFLICT (capability_name) DO UPDATE SET
                    status = EXCLUDED.status,
                    version = EXCLUDED.version,
                    modified_at = NOW()
            ''',
                capability, status, version
            )
    
    async def get_temporal_events(self, timeline_id: str = 'main', limit: int = 100) -> List[Dict]:
        """Retrieve temporal events from database"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT * FROM temporal_events 
                WHERE timeline_id =  
                ORDER BY timestamp DESC 
                LIMIT 
            ''', timeline_id, limit)
            
            return [dict(row) for row in rows]
    
    async def get_unresolved_paradoxes(self) -> List[Dict]:
        """Get all unresolved paradoxes"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT * FROM temporal_paradoxes 
                WHERE resolved = FALSE 
                ORDER BY detected_at DESC
            ''')
            
            return [dict(row) for row in rows]
    
    async def get_capability_history(self, capability: str = None) -> List[Dict]:
        """Get capability evolution history"""
        async with self.pool.acquire() as conn:
            if capability:
                rows = await conn.fetch('''
                    SELECT * FROM capability_evolution 
                    WHERE capability_name =  
                    ORDER BY modified_at DESC
                ''', capability)
            else:
                rows = await conn.fetch('''
                    SELECT * FROM capability_evolution 
                    ORDER BY modified_at DESC
                ''')
            
            return [dict(row) for row in rows]
    
    async def get_latest_self_report(self) -> Optional[Dict]:
        """Get most recent self-awareness report"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow('''
                SELECT * FROM self_awareness_reports 
                ORDER BY timestamp DESC 
                LIMIT 1
            ''')
            
            return dict(row) if row else None
    
    async def analyze_temporal_patterns(self, timeline_id: str = 'main') -> Dict:
        """Analyze patterns in temporal events"""
        async with self.pool.acquire() as conn:
            # Get event statistics
            stats = await conn.fetchrow('''
                SELECT 
                    COUNT(*) as total_events,
                    COUNT(DISTINCT event_type) as event_types,
                    AVG(probability) as avg_probability,
                    COUNT(CASE WHEN validated = TRUE THEN 1 END) as validated_events
                FROM temporal_events 
                WHERE timeline_id = 
            ''', timeline_id)
            
            # Get paradox statistics
            paradoxes = await conn.fetchrow('''
                SELECT 
                    COUNT(*) as total_paradoxes,
                    COUNT(CASE WHEN resolved = FALSE THEN 1 END) as unresolved,
                    COUNT(DISTINCT paradox_type) as paradox_types
                FROM temporal_paradoxes 
                WHERE timeline_id = 
            ''', timeline_id)
            
            # Get causal chain statistics
            chains = await conn.fetchrow('''
                SELECT 
                    COUNT(*) as total_chains,
                    AVG(path_length) as avg_chain_length,
                    MAX(path_length) as max_chain_length,
                    AVG(confidence_score) as avg_confidence
                FROM causal_chains
            ''')
            
            return {
                'timeline_id': timeline_id,
                'event_statistics': dict(stats) if stats else {},
                'paradox_statistics': dict(paradoxes) if paradoxes else {},
                'causal_statistics': dict(chains) if chains else {},
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    async def cleanup_old_data(self, days: int = 30):
        """Clean up old temporal data"""
        async with self.pool.acquire() as conn:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Archive old events
            deleted_events = await conn.fetchval('''
                DELETE FROM temporal_events 
                WHERE created_at <  AND timeline_id != 'main'
                RETURNING COUNT(*)
            ''', cutoff_date)
            
            # Archive old reports
            deleted_reports = await conn.fetchval('''
                DELETE FROM self_awareness_reports 
                WHERE timestamp < 
                RETURNING COUNT(*)
            ''', cutoff_date)
            
            return {
                'deleted_events': deleted_events,
                'deleted_reports': deleted_reports,
                'cleanup_date': datetime.now().isoformat()
            }
    
    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()

# Integration helper
async def setup_temporal_database():
    """Setup database for temporal reasoning"""
    db_config = {
        'host': 'localhost',
        'database': 'echo_brain',
        'user': 'echo',
        'password': 'echo_password'
    }
    
    db_manager = TemporalDatabaseManager(db_config)
    await db_manager.initialize()
    
    # Track new capabilities
    await db_manager.track_capability_evolution('temporal_reasoning', 'enabled', '1.0.0')
    await db_manager.track_capability_evolution('self_awareness', 'enabled', '1.0.0')
    await db_manager.track_capability_evolution('paradox_detection', 'enabled', '1.0.0')
    await db_manager.track_capability_evolution('causal_verification', 'enabled', '1.0.0')
    
    return db_manager

# Export
__all__ = ['TemporalDatabaseManager', 'setup_temporal_database']
