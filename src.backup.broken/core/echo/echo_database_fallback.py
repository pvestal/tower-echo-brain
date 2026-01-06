#!/usr/bin/env python3
"""
Echo Database Fallback Manager
Provides SQLite fallback when PostgreSQL is unavailable
"""

import sqlite3
import psycopg2
import psycopg2.extras
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class EchoDatabaseManager:
    """Manages database operations with PostgreSQL primary and SQLite fallback"""
    
    def __init__(self, pg_config: Dict[str, str], sqlite_path: str = "/opt/tower-echo-brain/echo_fallback.db"):
        self.pg_config = pg_config
        self.sqlite_path = sqlite_path
        self.using_fallback = False
        
        # Initialize SQLite fallback
        self._init_sqlite_fallback()
        
        # Test PostgreSQL connection
        self._test_postgresql()
    
    def _test_postgresql(self):
        """Test PostgreSQL connection and set fallback flag"""
        try:
            conn = psycopg2.connect(**self.pg_config)
            conn.close()
            self.using_fallback = False
            logger.info("PostgreSQL connection successful")
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed: {e}")
            logger.info("Using SQLite fallback for Echo conversations")
            self.using_fallback = True
    
    def _init_sqlite_fallback(self):
        """Initialize SQLite database with Echo tables"""
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            
            # Create conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS echo_conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message TEXT NOT NULL,
                    response TEXT NOT NULL,
                    project_id TEXT,
                    context_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create self_analysis table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS echo_self_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_type TEXT NOT NULL,
                    analysis_depth TEXT NOT NULL,
                    trigger_context TEXT,
                    findings TEXT,
                    recommendations TEXT,
                    confidence_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create learning_insights table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS echo_learning_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    insight_type TEXT NOT NULL,
                    pattern_data TEXT,
                    learning_weight REAL DEFAULT 1.0,
                    application_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("SQLite fallback database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize SQLite fallback: {e}")
    
    def store_conversation(self, message: str, response: str, project_id: str = None, context: Dict = None) -> bool:
        """Store conversation in available database"""
        try:
            if not self.using_fallback:
                return self._store_conversation_postgres(message, response, project_id, context)
            else:
                return self._store_conversation_sqlite(message, response, project_id, context)
        except Exception as e:
            logger.error(f"Failed to store conversation: {e}")
            # Try fallback if primary fails
            if not self.using_fallback:
                logger.info("Attempting SQLite fallback for conversation storage")
                return self._store_conversation_sqlite(message, response, project_id, context)
            return False
    
    def _store_conversation_postgres(self, message: str, response: str, project_id: str = None, context: Dict = None) -> bool:
        """Store conversation in PostgreSQL"""
        try:
            conn = psycopg2.connect(**self.pg_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO echo_conversations (message, response, project_id, context_data, created_at)
                VALUES (%s, %s, %s, %s, %s)
            """, (message, response, project_id, json.dumps(context) if context else None, datetime.now()))
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"PostgreSQL conversation storage failed: {e}")
            self.using_fallback = True
            raise
    
    def _store_conversation_sqlite(self, message: str, response: str, project_id: str = None, context: Dict = None) -> bool:
        """Store conversation in SQLite fallback"""
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO echo_conversations (message, response, project_id, context_data, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (message, response, project_id, json.dumps(context) if context else None, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"SQLite conversation storage failed: {e}")
            return False
    
    def get_conversation_history(self, project_id: str = None, limit: int = 10) -> List[Dict]:
        """Get conversation history from available database"""
        try:
            if not self.using_fallback:
                return self._get_conversation_history_postgres(project_id, limit)
            else:
                return self._get_conversation_history_sqlite(project_id, limit)
        except Exception:
            # Try fallback if primary fails
            if not self.using_fallback:
                return self._get_conversation_history_sqlite(project_id, limit)
            return []
    
    def _get_conversation_history_postgres(self, project_id: str = None, limit: int = 10) -> List[Dict]:
        """Get conversation history from PostgreSQL"""
        try:
            conn = psycopg2.connect(**self.pg_config)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            if project_id:
                cursor.execute("""
                    SELECT message, response, created_at, context_data
                    FROM echo_conversations 
                    WHERE project_id = %s 
                    ORDER BY created_at DESC 
                    LIMIT %s
                """, (project_id, limit))
            else:
                cursor.execute("""
                    SELECT message, response, created_at, context_data
                    FROM echo_conversations 
                    WHERE project_id IS NULL 
                    ORDER BY created_at DESC 
                    LIMIT %s
                """, (limit,))
            
            conversations = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return [dict(conv) for conv in conversations]
            
        except Exception as e:
            logger.error(f"PostgreSQL history retrieval failed: {e}")
            self.using_fallback = True
            raise
    
    def _get_conversation_history_sqlite(self, project_id: str = None, limit: int = 10) -> List[Dict]:
        """Get conversation history from SQLite fallback"""
        try:
            conn = sqlite3.connect(self.sqlite_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if project_id:
                cursor.execute("""
                    SELECT message, response, created_at, context_data
                    FROM echo_conversations 
                    WHERE project_id = ? 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (project_id, limit))
            else:
                cursor.execute("""
                    SELECT message, response, created_at, context_data
                    FROM echo_conversations 
                    WHERE project_id IS NULL 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (limit,))
            
            conversations = cursor.fetchall()
            conn.close()
            
            return [dict(conv) for conv in conversations]
            
        except Exception as e:
            logger.error(f"SQLite history retrieval failed: {e}")
            return []
    
    def store_self_analysis(self, analysis_type: str, analysis_depth: str, 
                          trigger_context: Dict, findings: Dict, 
                          recommendations: List[str], confidence_score: float) -> bool:
        """Store self-analysis results"""
        try:
            if not self.using_fallback:
                return self._store_self_analysis_postgres(analysis_type, analysis_depth, trigger_context, findings, recommendations, confidence_score)
            else:
                return self._store_self_analysis_sqlite(analysis_type, analysis_depth, trigger_context, findings, recommendations, confidence_score)
        except Exception:
            if not self.using_fallback:
                return self._store_self_analysis_sqlite(analysis_type, analysis_depth, trigger_context, findings, recommendations, confidence_score)
            return False
    
    def _store_self_analysis_postgres(self, analysis_type: str, analysis_depth: str, 
                                    trigger_context: Dict, findings: Dict, 
                                    recommendations: List[str], confidence_score: float) -> bool:
        """Store self-analysis in PostgreSQL"""
        try:
            conn = psycopg2.connect(**self.pg_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO echo_self_analysis 
                (analysis_type, analysis_depth, trigger_context, findings, recommendations, confidence_score, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (analysis_type, analysis_depth, json.dumps(trigger_context), 
                  json.dumps(findings), json.dumps(recommendations), confidence_score, datetime.now()))
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"PostgreSQL self-analysis storage failed: {e}")
            self.using_fallback = True
            raise
    
    def _store_self_analysis_sqlite(self, analysis_type: str, analysis_depth: str, 
                                  trigger_context: Dict, findings: Dict, 
                                  recommendations: List[str], confidence_score: float) -> bool:
        """Store self-analysis in SQLite fallback"""
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO echo_self_analysis 
                (analysis_type, analysis_depth, trigger_context, findings, recommendations, confidence_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (analysis_type, analysis_depth, json.dumps(trigger_context), 
                  json.dumps(findings), json.dumps(recommendations), confidence_score, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"SQLite self-analysis storage failed: {e}")
            return False
    
    def get_database_status(self) -> Dict[str, Any]:
        """Get current database status"""
        return {
            "primary_database": "PostgreSQL",
            "fallback_database": "SQLite",
            "using_fallback": self.using_fallback,
            "postgresql_config": {
                "host": self.pg_config.get("host"),
                "database": self.pg_config.get("database"),
                "user": self.pg_config.get("user")
            },
            "sqlite_path": self.sqlite_path,
            "last_test": datetime.now().isoformat()
        }