#!/usr/bin/env python3
"""
Anime Memory Integration for Echo Brain
Allows Echo to access stored anime character and preference data
"""

import psycopg2
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class AnimeMemoryIntegration:
    """Integration layer for accessing anime-specific memory"""
    
    def __init__(self):
        self.db_config = {
            "database": "echo_brain",
            "user": "patrick"
        }
    
    def get_character_info(self, character_name: str) -> Optional[Dict]:
        """Get character information from anime memory"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT character_name, canonical_description, visual_consistency_score,
                       generation_count, successful_generations, reference_images,
                       style_elements, last_generated
                FROM anime_character_memory 
                WHERE LOWER(character_name) = LOWER(%s)
            """, (character_name,))
            
            result = cursor.fetchone()
            if result:
                return {
                    "name": result[0],
                    "description": result[1],
                    "consistency_score": result[2],
                    "generation_count": result[3],
                    "successful_generations": result[4],
                    "reference_images": result[5],
                    "style_elements": result[6],
                    "last_generated": result[7]
                }
            
            conn.close()
            return None
            
        except Exception as e:
            logger.error(f"Error accessing character memory: {e}")
            return None
    
    def get_user_preferences(self, user_id: str = "patrick") -> List[Dict]:
        """Get user's anime generation preferences"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT preference_type, preference_value, confidence_score, evidence_count
                FROM anime_creative_preferences 
                WHERE user_id = %s
                ORDER BY confidence_score DESC
            """, (user_id,))
            
            results = cursor.fetchall()
            preferences = []
            for row in results:
                preferences.append({
                    "type": row[0],
                    "value": row[1],
                    "confidence": row[2],
                    "evidence_count": row[3]
                })
            
            conn.close()
            return preferences
            
        except Exception as e:
            logger.error(f"Error accessing preferences: {e}")
            return []

# Global instance
anime_memory = AnimeMemoryIntegration()

def get_anime_context(query: str) -> Dict:
    """Extract anime context from query and return relevant memory"""
    print(f"🎭 ANIME CONTEXT DEBUG: Query received: {query[:100]}...")
    context = {"characters": [], "preferences": []}

    # Check for character mentions (flexible search)
    character_search = [
        (["kai", "kai nakamura"], "kai nakamura"),
        (["aria", "aria chen"], "aria chen"),
        (["hiroshi", "hiroshi yamamoto"], "hiroshi yamamoto")
    ]

    for search_terms, full_name in character_search:
        if any(term in query.lower() for term in search_terms):
            print(f"🎭 FOUND CHARACTER MENTION: {full_name} (matched: {[term for term in search_terms if term in query.lower()]})")
            char_info = anime_memory.get_character_info(full_name)
            if char_info:
                context["characters"].append(char_info)
                print(f"🎭 LOADED CHARACTER: {char_info['name']}")

    # Get user preferences if anime-related query
    if any(keyword in query.lower() for keyword in ["anime", "character", "style", "generation", "kai", "aria", "hiroshi"]):
        context["preferences"] = anime_memory.get_user_preferences()

    return context
