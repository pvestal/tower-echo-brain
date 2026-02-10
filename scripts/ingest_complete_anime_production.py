#!/usr/bin/env python3
"""
Complete Anime Production Data Ingestion for Story Bible
Embeds ALL anime production tables into story_bible collection
"""

import psycopg2
import requests
import json
import logging
from typing import List, Dict, Any, Optional
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

# Tables already ingested - skip these
SKIP_TABLES = {'characters', 'episodes', 'scenes'}

# High priority tables (process first)
PRIORITY_TABLES = {
    'projects', 'generation_profiles', 'lora_definitions', 'ai_models',
    'production_jobs', 'generation_history', 'scene_generations',
    'storylines', 'story_arcs', 'workflows'
}

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

def get_all_tables(cursor) -> List[Dict[str, Any]]:
    """Get all anime production tables with metadata"""
    cursor.execute("""
        SELECT t.table_name,
               COUNT(c.column_name) as column_count,
               array_agg(c.column_name ORDER BY c.ordinal_position) as columns,
               array_agg(c.data_type ORDER BY c.ordinal_position) as types
        FROM information_schema.tables t
        LEFT JOIN information_schema.columns c ON t.table_name = c.table_name
        WHERE t.table_schema = 'public'
        AND t.table_type = 'BASE TABLE'
        AND t.table_name NOT IN %s
        GROUP BY t.table_name
        ORDER BY
            CASE WHEN t.table_name = ANY(%s) THEN 0 ELSE 1 END,
            t.table_name
    """, (tuple(SKIP_TABLES), list(PRIORITY_TABLES)))

    tables = []
    for row in cursor.fetchall():
        tables.append({
            'name': row[0],
            'column_count': row[1],
            'columns': row[2] if row[2] else [],
            'types': row[3] if row[3] else []
        })

    return tables

def get_sample_data(cursor, table_name: str, limit: int = 5) -> List[Dict]:
    """Get sample data from a table"""
    try:
        cursor.execute(f"SELECT * FROM {table_name} LIMIT %s", (limit,))
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        logger.warning(f"Failed to sample data from {table_name}: {e}")
        return []

def format_table_document(table_info: Dict, sample_data: List[Dict]) -> str:
    """Format table information into a comprehensive document"""
    table_name = table_info['name']
    doc = f"ANIME PRODUCTION TABLE: {table_name}\n\n"

    # Determine table category and purpose
    category = determine_table_category(table_name)
    doc += f"Category: {category}\n"
    doc += f"Type: Database table for anime production pipeline\n\n"

    # Table structure
    doc += f"Columns ({table_info['column_count']}):\n"
    for i, (col, dtype) in enumerate(zip(table_info['columns'], table_info['types'])):
        doc += f"  - {col}: {dtype}\n"
    doc += "\n"

    # Sample data (if available)
    if sample_data:
        doc += f"Sample Data ({len(sample_data)} records):\n"
        for i, record in enumerate(sample_data[:3]):  # Show max 3 samples
            doc += f"  Record {i+1}:\n"
            for key, value in record.items():
                # Truncate long values
                if value is None:
                    str_value = "NULL"
                elif isinstance(value, (dict, list)):
                    str_value = json.dumps(value)[:100] + "..." if len(json.dumps(value)) > 100 else json.dumps(value)
                else:
                    str_value = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                doc += f"    {key}: {str_value}\n"
            doc += "\n"
    else:
        doc += "Sample Data: No data available\n\n"

    # Purpose and usage context
    purpose = get_table_purpose(table_name)
    doc += f"Purpose: {purpose}\n\n"

    # Related tables (inferred from name patterns)
    related_tables = infer_related_tables(table_name)
    if related_tables:
        doc += f"Related Tables: {', '.join(related_tables)}\n\n"

    doc += f"Row Count: {len(sample_data)} (sampled)\n"
    doc += f"Database: anime_production\n"

    return doc.strip()

def determine_table_category(table_name: str) -> str:
    """Determine the category of a table based on its name"""
    name_lower = table_name.lower()

    if any(x in name_lower for x in ['character', 'persona']):
        return "Character Management"
    elif any(x in name_lower for x in ['episode', 'scene', 'shot']):
        return "Content Structure"
    elif any(x in name_lower for x in ['generation', 'ai_model', 'workflow']):
        return "AI Generation"
    elif any(x in name_lower for x in ['lora', 'training']):
        return "Model Training"
    elif any(x in name_lower for x in ['production', 'queue', 'job']):
        return "Production Pipeline"
    elif any(x in name_lower for x in ['story', 'narrative', 'arc']):
        return "Story Development"
    elif any(x in name_lower for x in ['project', 'config', 'setting']):
        return "Project Management"
    elif any(x in name_lower for x in ['quality', 'approval', 'rating']):
        return "Quality Control"
    elif any(x in name_lower for x in ['timeline', 'branch', 'version']):
        return "Version Control"
    elif any(x in name_lower for x in ['asset', 'image', 'video']):
        return "Asset Management"
    else:
        return "System Configuration"

def get_table_purpose(table_name: str) -> str:
    """Get detailed purpose description for a table"""
    purposes = {
        'projects': 'Root project definitions with metadata and settings',
        'ai_models': 'Available AI models for generation with capabilities',
        'generation_profiles': 'Predefined generation settings and parameters',
        'generation_history': 'Complete log of all generation attempts and results',
        'lora_definitions': 'LoRA model definitions with training parameters',
        'production_jobs': 'Production pipeline job queue and status tracking',
        'scene_generations': 'Generated scene content and metadata',
        'storylines': 'Story structure and narrative flow definitions',
        'story_arcs': 'Character and plot arc development tracking',
        'workflows': 'ComfyUI workflow definitions and configurations',
        'character_training_jobs': 'Character-specific LoRA training job tracking',
        'production_queue': 'Active production job queue management',
        'quality_gates': 'Quality control checkpoints and requirements',
        'content_warnings': 'Content rating and warning categorization',
        'echo_brain_suggestions': 'AI-generated story and production suggestions'
    }

    return purposes.get(table_name, f"Anime production data table for {table_name.replace('_', ' ')}")

def infer_related_tables(table_name: str) -> List[str]:
    """Infer related tables based on naming patterns"""
    related = []
    name_parts = table_name.split('_')

    # Common relationships
    relationships = {
        'character': ['characters', 'character_training_jobs', 'character_embeddings'],
        'episode': ['episodes', 'episode_scenes', 'episode_timelines'],
        'scene': ['scenes', 'scene_assets', 'scene_generations'],
        'generation': ['generation_profiles', 'generation_history', 'generation_jobs'],
        'lora': ['lora_definitions', 'lora_training_jobs', 'lora_models'],
        'production': ['production_jobs', 'production_queue', 'production_profiles'],
        'story': ['storylines', 'story_arcs', 'story_changelog'],
        'project': ['projects', 'project_characters', 'project_configs']
    }

    for part in name_parts:
        if part in relationships:
            related.extend(relationships[part])

    # Remove self and duplicates
    related = list(set(related) - {table_name})
    return related[:5]  # Limit to 5 related tables

def ingest_all_tables():
    """Main ingestion function for all anime production tables"""
    logger.info("Starting complete anime production data ingestion...")

    # Get current collection size
    try:
        response = requests.get(f"{QDRANT_URL}/collections/{COLLECTION_NAME}")
        response.raise_for_status()
        current_count = response.json()["result"]["points_count"]
        logger.info(f"Current story_bible collection has {current_count} points")
    except Exception as e:
        logger.error(f"Failed to get collection size: {e}")
        return

    # Connect to PostgreSQL
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    cursor = conn.cursor()

    points = []
    point_id = current_count  # Start from current count

    try:
        # Get all tables
        tables = get_all_tables(cursor)
        logger.info(f"Found {len(tables)} tables to ingest")

        # Process each table
        for i, table_info in enumerate(tables):
            table_name = table_info['name']
            logger.info(f"Processing table {i+1}/{len(tables)}: {table_name}")

            try:
                # Get sample data
                sample_data = get_sample_data(cursor, table_name, limit=5)

                # Skip empty tables with no structure info
                if not sample_data and table_info['column_count'] == 0:
                    logger.warning(f"Skipping empty table: {table_name}")
                    continue

                # Format document
                doc_text = format_table_document(table_info, sample_data)

                # Skip very short documents
                if len(doc_text.strip()) < 100:
                    logger.warning(f"Skipping {table_name}: document too short")
                    continue

                # Get embedding
                embedding = get_embedding(doc_text)

                # Create point
                points.append({
                    "id": point_id,
                    "vector": embedding,
                    "payload": {
                        "content": doc_text,
                        "type": "database_table",
                        "table_name": table_name,
                        "category": determine_table_category(table_name),
                        "column_count": table_info['column_count'],
                        "sample_records": len(sample_data),
                        "database": "anime_production",
                        "ingested_at": datetime.now().isoformat()
                    }
                })
                point_id += 1

                # Batch upload every 25 tables to avoid memory issues
                if len(points) >= 25:
                    logger.info(f"Uploading batch of {len(points)} tables...")
                    upload_batch(points)
                    points = []

            except Exception as e:
                logger.error(f"Failed to process table {table_name}: {e}")
                continue

        # Upload remaining points
        if points:
            logger.info(f"Uploading final batch of {len(points)} tables...")
            upload_batch(points)

        logger.info("Complete anime production data ingestion finished!")

    finally:
        cursor.close()
        conn.close()

def upload_batch(points: List[Dict]):
    """Upload a batch of points to Qdrant"""
    try:
        response = requests.put(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points",
            json={"points": points}
        )
        response.raise_for_status()
        logger.info(f"Successfully uploaded {len(points)} points")
    except Exception as e:
        logger.error(f"Failed to upload batch: {e}")
        raise

def verify_complete_ingestion():
    """Verify the complete ingestion was successful"""
    try:
        # Check final collection size
        response = requests.get(f"{QDRANT_URL}/collections/{COLLECTION_NAME}")
        response.raise_for_status()
        info = response.json()["result"]

        logger.info(f"Final collection stats:")
        logger.info(f"  Total points: {info['points_count']}")

        # Check by type
        for doc_type in ["character", "episode", "scene", "workflow", "database_table"]:
            type_response = requests.post(
                f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search",
                json={
                    "vector": [0.1] * 768,
                    "limit": 100,
                    "with_payload": True,
                    "filter": {
                        "must": [{"key": "type", "match": {"value": doc_type}}]
                    }
                }
            )
            if type_response.status_code == 200:
                count = len(type_response.json()["result"])
                logger.info(f"  {doc_type}: {count} documents")

        # Sample some database tables
        table_response = requests.post(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search",
            json={
                "vector": [0.1] * 768,
                "limit": 5,
                "with_payload": True,
                "filter": {
                    "must": [{"key": "type", "match": {"value": "database_table"}}]
                }
            }
        )

        if table_response.status_code == 200:
            tables = table_response.json()["result"]
            logger.info(f"Sample database tables ingested:")
            for table in tables:
                payload = table["payload"]
                logger.info(f"  - {payload['table_name']} ({payload['category']}) - {payload['column_count']} cols")

    except Exception as e:
        logger.error(f"Failed to verify ingestion: {e}")

if __name__ == "__main__":
    try:
        ingest_all_tables()
        verify_complete_ingestion()
        logger.info("✅ Complete anime production ingestion successful!")
    except Exception as e:
        logger.error(f"❌ Complete ingestion failed: {e}")
        raise