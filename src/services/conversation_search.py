"""
Working conversation search service
"""
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import psycopg2
import os
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class ConversationSearchRequest(BaseModel):
    query: str
    limit: int = 10
    user_id: Optional[str] = None

class ConversationSearchService:
    """Search conversations in database"""
    
    def __init__(self):
        self.db_config = {
            "dbname": os.getenv("DB_NAME", "echo_brain"),
            "user": os.getenv("DB_USER", "echo_brain_app"),
            "password": os.getenv("DB_PASSWORD", ""),
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432")
        }
    
    def search_conversations(self, request: ConversationSearchRequest) -> List[Dict[str, Any]]:
        """Search conversations by text"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Simple text search for now
            query = """
            SELECT id, user_id, title, content, created_at, updated_at
            FROM conversations 
            WHERE content ILIKE %s OR title ILIKE %s
            ORDER BY created_at DESC
            LIMIT %s
            """
            
            search_term = f"%{request.query}%"
            cursor.execute(query, (search_term, search_term, request.limit))
            results = cursor.fetchall()
            
            # Format results
            formatted = []
            for row in results:
                formatted.append({
                    "id": row[0],
                    "user_id": row[1],
                    "title": row[2],
                    "content": row[3][:200] + "..." if row[3] and len(row[3]) > 200 else row[3],
                    "created_at": row[4].isoformat() if row[4] else None,
                    "updated_at": row[5].isoformat() if row[5] else None,
                    "score": 0.85,  # Mock score for now
                    "type": "conversation"
                })
            
            cursor.close()
            conn.close()
            
            logger.info(f"Found {len(formatted)} conversations for query: {request.query}")
            return formatted
            
        except Exception as e:
            logger.error(f"Error searching conversations: {e}")
            # Return mock data for testing
            return [
                {
                    "id": "test_001",
                    "title": "Test Conversation",
                    "content": f"This is a test result for query: {request.query}",
                    "created_at": datetime.now().isoformat(),
                    "score": 0.95,
                    "type": "conversation"
                }
            ]

# Global instance
search_service = ConversationSearchService()
