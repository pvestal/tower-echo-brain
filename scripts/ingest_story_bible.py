#!/usr/bin/env python3
"""
Story Bible Ingestion Script
Embeds anime production data from PostgreSQL into Qdrant story_bible collection
"""

import psycopg2
import requests
import json
import uuid
from typing import List, Dict, Any
import logging
from datetime import datetime

# Configuration
POSTGRES_CONFIG = {
    'host': 'localhost',
    'user': 'patrick',
    'password': 'RP78eIrW7cI2jYvL5akt1yurE',
    'database': 'anime_production'
}

QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
COLLECTION_NAME = "story_bible"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_embedding(text: str) -> List[float]:
    """Get embedding from Ollama using nomic-embed-text"""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": text
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        logger.error(f"Failed to get embedding: {e}")
        raise

def create_collection():
    """Create or recreate the story_bible collection"""
    try:
        # Delete if exists
        requests.delete(f"{QDRANT_URL}/collections/{COLLECTION_NAME}")

        # Create collection with 768 dimensions (nomic-embed-text)
        response = requests.put(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}",
            json={
                "vectors": {
                    "size": 768,
                    "distance": "Cosine"
                }
            }
        )
        response.raise_for_status()
        logger.info(f"Created collection: {COLLECTION_NAME}")
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        raise

def format_character_document(char: Dict) -> str:
    """Format character data into a comprehensive document"""
    doc = f"CHARACTER: {char['name']}\n\n"

    if char['description']:
        doc += f"Description: {char['description']}\n\n"

    if char['age']:
        doc += f"Age: {char['age']}\n\n"

    if char['role']:
        doc += f"Role: {char['role']}\n\n"

    if char['character_role']:
        doc += f"Character Role: {char['character_role']}\n\n"

    if char['personality']:
        doc += f"Personality: {char['personality']}\n\n"

    if char['background']:
        doc += f"Background: {char['background']}\n\n"

    # Parse JSON fields
    if char['appearance_data']:
        try:
            appearance = json.loads(char['appearance_data']) if isinstance(char['appearance_data'], str) else char['appearance_data']
            doc += "Appearance:\n"
            for key, value in appearance.items():
                doc += f"  {key}: {value}\n"
            doc += "\n"
        except:
            pass

    if char['traits']:
        try:
            traits = json.loads(char['traits']) if isinstance(char['traits'], str) else char['traits']
            doc += "Traits:\n"
            for key, value in traits.items():
                doc += f"  {key}: {value}\n"
            doc += "\n"
        except:
            pass

    if char['relationships']:
        try:
            relationships = json.loads(char['relationships']) if isinstance(char['relationships'], str) else char['relationships']
            doc += "Relationships:\n"
            for key, value in relationships.items():
                doc += f"  {key}: {value}\n"
            doc += "\n"
        except:
            pass

    if char['design_prompt']:
        doc += f"Design Prompt: {char['design_prompt']}\n\n"

    return doc.strip()

def format_episode_document(episode: Dict) -> str:
    """Format episode data into a comprehensive document"""
    doc = f"EPISODE {episode['episode_number']}: {episode['title']}\n\n"

    if episode['description']:
        doc += f"Description: {episode['description']}\n\n"

    if episode['synopsis']:
        doc += f"Synopsis: {episode['synopsis']}\n\n"

    if episode['status']:
        doc += f"Status: {episode['status']}\n\n"

    if episode['production_status']:
        doc += f"Production Status: {episode['production_status']}\n\n"

    if episode['duration']:
        doc += f"Duration: {episode['duration']} seconds\n\n"

    if episode['tone_profile']:
        try:
            tone = json.loads(episode['tone_profile']) if isinstance(episode['tone_profile'], str) else episode['tone_profile']
            if tone:
                doc += "Tone Profile:\n"
                for key, value in tone.items():
                    doc += f"  {key}: {value}\n"
                doc += "\n"
        except:
            pass

    return doc.strip()

def format_scene_document(scene: Dict) -> str:
    """Format scene data into a comprehensive document"""
    doc = f"SCENE {scene['scene_number']}"
    if scene['title']:
        doc += f": {scene['title']}"
    doc += "\n\n"

    if scene['description']:
        doc += f"Description: {scene['description']}\n\n"

    if scene['narrative_text']:
        doc += f"Narrative: {scene['narrative_text']}\n\n"

    if scene['visual_description']:
        doc += f"Visual Description: {scene['visual_description']}\n\n"

    if scene['setting_description']:
        doc += f"Setting: {scene['setting_description']}\n\n"


    if scene['characters_present']:
        try:
            chars = scene['characters_present'] if isinstance(scene['characters_present'], list) else json.loads(scene['characters_present'])
            if chars:
                doc += f"Characters Present: {', '.join(chars)}\n\n"
        except:
            pass

    if scene['emotional_tone']:
        doc += f"Emotional Tone: {scene['emotional_tone']}\n\n"

    if scene['audio_mood']:
        doc += f"Audio Mood: {scene['audio_mood']}\n\n"

    if scene['dialogue']:
        try:
            dialogue = json.loads(scene['dialogue']) if isinstance(scene['dialogue'], str) else scene['dialogue']
            if dialogue:
                doc += "Dialogue:\n"
                for key, value in dialogue.items():
                    doc += f"  {key}: {value}\n"
                doc += "\n"
        except:
            pass

    if scene['narration']:
        doc += f"Narration: {scene['narration']}\n\n"

    if scene['continuity_notes']:
        doc += f"Continuity Notes: {scene['continuity_notes']}\n\n"

    if scene['camera_directions']:
        doc += f"Camera Directions: {scene['camera_directions']}\n\n"

    return doc.strip()

def ingest_data():
    """Main ingestion function"""
    logger.info("Starting story bible ingestion...")

    # Create collection
    create_collection()

    # Connect to PostgreSQL
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    cursor = conn.cursor()

    points = []
    point_id = 0

    try:
        # Ingest Characters
        logger.info("Processing characters...")
        cursor.execute("""
            SELECT id, project_id, name, description, age, role, character_role,
                   personality, background, appearance_data, traits, relationships, design_prompt
            FROM characters
            WHERE name IS NOT NULL
        """)

        for row in cursor.fetchall():
            char = dict(zip([desc[0] for desc in cursor.description], row))
            doc_text = format_character_document(char)

            if len(doc_text.strip()) > 20:  # Only meaningful documents
                embedding = get_embedding(doc_text)

                points.append({
                    "id": point_id,
                    "vector": embedding,
                    "payload": {
                        "content": doc_text,
                        "type": "character",
                        "name": char['name'],
                        "project_id": char['project_id'],
                        "source_id": char['id'],
                        "ingested_at": datetime.now().isoformat()
                    }
                })
                point_id += 1

        logger.info(f"Processed {len([p for p in points if p['payload']['type'] == 'character'])} characters")

        # Ingest Episodes
        logger.info("Processing episodes...")
        cursor.execute("""
            SELECT id, project_id, episode_number, title, description, synopsis,
                   status, production_status, duration, tone_profile
            FROM episodes
            WHERE title IS NOT NULL
        """)

        for row in cursor.fetchall():
            episode = dict(zip([desc[0] for desc in cursor.description], row))
            doc_text = format_episode_document(episode)

            if len(doc_text.strip()) > 20:
                embedding = get_embedding(doc_text)

                points.append({
                    "id": point_id,
                    "vector": embedding,
                    "payload": {
                        "content": doc_text,
                        "type": "episode",
                        "title": episode['title'],
                        "episode_number": episode['episode_number'],
                        "project_id": episode['project_id'],
                        "source_id": str(episode['id']),
                        "ingested_at": datetime.now().isoformat()
                    }
                })
                point_id += 1

        logger.info(f"Processed {len([p for p in points if p['payload']['type'] == 'episode'])} episodes")

        # Ingest Scenes
        logger.info("Processing scenes...")
        cursor.execute("""
            SELECT id, project_id, episode_id, scene_number, title, description,
                   narrative_text, visual_description, setting_description,
                   characters_present, emotional_tone, audio_mood, dialogue, narration,
                   continuity_notes, camera_directions
            FROM scenes
            WHERE description IS NOT NULL OR narrative_text IS NOT NULL
        """)

        for row in cursor.fetchall():
            scene = dict(zip([desc[0] for desc in cursor.description], row))
            doc_text = format_scene_document(scene)

            if len(doc_text.strip()) > 20:
                embedding = get_embedding(doc_text)

                points.append({
                    "id": point_id,
                    "vector": embedding,
                    "payload": {
                        "content": doc_text,
                        "type": "scene",
                        "title": scene['title'] if scene['title'] else f"Scene {scene['scene_number']}",
                        "scene_number": scene['scene_number'],
                        "project_id": scene['project_id'],
                        "episode_id": str(scene['episode_id']) if scene['episode_id'] else None,
                        "source_id": str(scene['id']),
                        "ingested_at": datetime.now().isoformat()
                    }
                })
                point_id += 1

        logger.info(f"Processed {len([p for p in points if p['payload']['type'] == 'scene'])} scenes")

        # Batch upload to Qdrant
        if points:
            logger.info(f"Uploading {len(points)} points to Qdrant...")

            # Upload in batches of 100
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]

                response = requests.put(
                    f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points",
                    json={"points": batch}
                )
                response.raise_for_status()
                logger.info(f"Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")

            logger.info(f"Successfully ingested {len(points)} documents into story_bible collection")
        else:
            logger.warning("No documents found to ingest")

    finally:
        cursor.close()
        conn.close()

def verify_collection():
    """Verify the collection was created successfully"""
    try:
        response = requests.get(f"{QDRANT_URL}/collections/{COLLECTION_NAME}")
        response.raise_for_status()
        info = response.json()

        logger.info(f"Collection info:")
        logger.info(f"  Points count: {info['result']['points_count']}")
        logger.info(f"  Vector size: {info['result']['config']['params']['vectors']['size']}")

        # Test search
        test_response = requests.post(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search",
            json={
                "vector": [0.1] * 768,  # Dummy vector
                "limit": 3,
                "with_payload": True
            }
        )
        test_response.raise_for_status()
        results = test_response.json()["result"]

        logger.info(f"Test search returned {len(results)} results")
        for i, result in enumerate(results[:2]):
            logger.info(f"  Result {i+1}: {result['payload']['type']} - {result['payload'].get('name', result['payload'].get('title', 'Unknown'))}")

    except Exception as e:
        logger.error(f"Failed to verify collection: {e}")

if __name__ == "__main__":
    try:
        ingest_data()
        verify_collection()
        logger.info("Story bible ingestion completed successfully!")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise