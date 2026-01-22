#!/usr/bin/env python3
"""
Migrate knowledge_items to support 1024-dim embeddings
Store embeddings in Qdrant only, keep metadata in PostgreSQL
"""

import json
import time
import logging
import requests
import psycopg2
from pathlib import Path
from datetime import datetime
import uuid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
OLLAMA_URL = "http://localhost:11434/api/embeddings"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "echo_memory"
EMBEDDING_MODEL = "mxbai-embed-large:latest"

DB_CONFIG = {
    'dbname': 'echo_brain',
    'user': 'patrick',
    'password': 'RP78eIrW7cI2jYvL5akt1yurE',
    'host': 'localhost'
}

def migrate_photos_and_videos():
    """Process photos and videos without embeddings"""
    logger.info("Starting migration for photos and videos...")

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Get items without embeddings
    cursor.execute("""
        SELECT item_id, knowledge_type, title, content::text, metadata::text
        FROM knowledge_items
        WHERE embedding IS NULL
        AND knowledge_type IN ('personal_photo', 'personal_video')
        ORDER BY created_at DESC
        LIMIT 100
    """)

    items = cursor.fetchall()
    logger.info(f"Processing {len(items)} items...")

    batch_points = []
    processed = 0

    for item in items:
        item_id, item_type, title, content_str, metadata_str = item

        # Parse JSON
        content = json.loads(content_str) if content_str else {}
        metadata = json.loads(metadata_str) if metadata_str else {}

        # Create text for embedding
        text_parts = []
        if title:
            text_parts.append(f"Title: {title}")
        text_parts.append(f"Type: {item_type}")

        if isinstance(content, dict):
            if content.get('file_path'):
                text_parts.append(f"Path: {content['file_path']}")
            if content.get('description'):
                text_parts.append(f"Description: {content['description']}")

        if isinstance(metadata, dict):
            if metadata.get('date_taken'):
                text_parts.append(f"Date: {metadata['date_taken']}")

        text = " | ".join(text_parts)

        # Generate embedding
        try:
            response = requests.post(
                OLLAMA_URL,
                json={"model": EMBEDDING_MODEL, "prompt": text},
                timeout=30
            )

            if response.status_code == 200:
                embedding = response.json().get('embedding')
                if embedding and len(embedding) == 1024:
                    # Create point for Qdrant
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, item_id))
                    batch_points.append({
                        "id": point_id,
                        "vector": embedding,
                        "payload": {
                            "item_id": item_id,
                            "type": item_type,
                            "title": title or "",
                            "content": content,
                            "metadata": metadata,
                            "source": "knowledge_items",
                            "timestamp": datetime.now().isoformat()
                        }
                    })
                    processed += 1

                    # Store batch every 50 items
                    if len(batch_points) >= 50:
                        store_batch(batch_points)
                        logger.info(f"Stored batch of {len(batch_points)} items. Total: {processed}")
                        batch_points = []

        except Exception as e:
            logger.error(f"Error processing {item_id}: {e}")
            continue

    # Store remaining items
    if batch_points:
        store_batch(batch_points)
        logger.info(f"Stored final batch of {len(batch_points)} items")

    cursor.close()
    conn.close()

    logger.info(f"✅ Migration complete! Processed {processed} items")
    return processed

def store_batch(points):
    """Store batch of points in Qdrant"""
    try:
        response = requests.put(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points",
            json={"points": points},
            params={"wait": "true"},
            timeout=60
        )
        if response.status_code != 200:
            logger.error(f"Failed to store batch: {response.status_code}")
    except Exception as e:
        logger.error(f"Error storing batch: {e}")

def check_progress():
    """Check migration progress"""
    try:
        # Check Qdrant
        response = requests.get(f"{QDRANT_URL}/collections/{COLLECTION_NAME}")
        if response.status_code == 200:
            data = response.json()
            point_count = data['result']['points_count']
            logger.info(f"Qdrant {COLLECTION_NAME}: {point_count} vectors")

        # Check PostgreSQL
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT knowledge_type,
                   COUNT(*) as total,
                   COUNT(embedding) as with_768_embeddings
            FROM knowledge_items
            WHERE knowledge_type IN ('personal_photo', 'personal_video')
            GROUP BY knowledge_type
        """)

        logger.info("Database status:")
        for row in cursor.fetchall():
            logger.info(f"  {row[0]}: {row[1]} total, {row[2]} with old 768-dim embeddings")

        cursor.close()
        conn.close()

    except Exception as e:
        logger.error(f"Error checking progress: {e}")

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("KNOWLEDGE ITEMS TO QDRANT MIGRATION")
    logger.info("="*60)

    check_progress()

    total_processed = 0
    while True:
        processed = migrate_photos_and_videos()
        total_processed += processed

        if processed < 100:
            logger.info(f"✅ All items processed! Total: {total_processed}")
            break
        else:
            logger.info(f"Continuing... Total so far: {total_processed}")
            time.sleep(2)

    logger.info("\nFinal status:")
    check_progress()