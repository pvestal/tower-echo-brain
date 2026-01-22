#!/usr/bin/env python3
"""
Fix and regenerate embeddings with 1024 dimensions
Includes comprehensive error handling and monitoring
"""

import json
import time
import sys
import logging
import requests
import psycopg2
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/tower-echo-brain/logs/regeneration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
OLLAMA_URL = "http://localhost:11434/api/embeddings"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "echo_memory"
EMBEDDING_MODEL = "mxbai-embed-large:latest"  # 1024 dimensions
BATCH_SIZE = 100
MAX_WORKERS = 4
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5

# Database config
DB_CONFIG = {
    'dbname': 'echo_brain',
    'user': 'patrick',
    'password': 'RP78eIrW7cI2jYvL5akt1yurE',
    'host': 'localhost'
}

class EmbeddingRegenerator:
    def __init__(self):
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }
        self.running = True
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown gracefully"""
        logger.info("🛑 Shutdown signal received, finishing current batch...")
        self.running = False

    def verify_collection(self) -> bool:
        """Verify collection exists with correct dimensions"""
        try:
            response = requests.get(f"{QDRANT_URL}/collections/{COLLECTION_NAME}")
            if response.status_code != 200:
                logger.error(f"Collection {COLLECTION_NAME} doesn't exist")
                return False

            data = response.json()
            vector_size = data['result']['config']['params']['vectors']['size']
            if vector_size != 1024:
                logger.error(f"Collection has wrong dimensions: {vector_size} (expected 1024)")
                return False

            logger.info(f"✅ Collection {COLLECTION_NAME} verified (1024 dimensions)")
            return True
        except Exception as e:
            logger.error(f"Failed to verify collection: {e}")
            return False

    def get_items_without_embeddings(self) -> List[Dict]:
        """Get items from knowledge_items table without embeddings"""
        items = []
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # Get items without embeddings (photos and videos)
            cursor.execute("""
                SELECT item_id, knowledge_type, title, content::text, metadata::text
                FROM knowledge_items
                WHERE embedding IS NULL
                AND knowledge_type IN ('personal_photo', 'personal_video')
                LIMIT 1000
            """)

            rows = cursor.fetchall()
            for row in rows:
                items.append({
                    'item_id': row[0],
                    'type': row[1],
                    'title': row[2],
                    'content': json.loads(row[3]) if row[3] else {},
                    'metadata': json.loads(row[4]) if row[4] else {}
                })

            cursor.close()
            conn.close()
            logger.info(f"Found {len(items)} items without embeddings")
            return items
        except Exception as e:
            logger.error(f"Database error: {e}")
            return []

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate 1024-dim embedding using Ollama"""
        for attempt in range(RETRY_ATTEMPTS):
            try:
                response = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": EMBEDDING_MODEL,
                        "prompt": text
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    embedding = response.json().get('embedding')
                    if embedding and len(embedding) == 1024:
                        return embedding
                    else:
                        logger.warning(f"Invalid embedding dimensions: {len(embedding) if embedding else 0}")
                else:
                    logger.warning(f"Ollama returned {response.status_code}: {response.text}")

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout generating embedding (attempt {attempt + 1}/{RETRY_ATTEMPTS})")
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")

            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)

        return None

    def create_text_representation(self, item: Dict) -> str:
        """Create text representation for embedding"""
        parts = []

        # Add title
        if item.get('title'):
            parts.append(f"Title: {item['title']}")

        # Add type
        parts.append(f"Type: {item.get('type', 'unknown')}")

        # Add content fields
        content = item.get('content', {})
        if isinstance(content, dict):
            if content.get('file_path'):
                parts.append(f"Path: {content['file_path']}")
            if content.get('description'):
                parts.append(f"Description: {content['description']}")
            if content.get('tags'):
                parts.append(f"Tags: {', '.join(content['tags'])}")

        # Add metadata
        metadata = item.get('metadata', {})
        if isinstance(metadata, dict):
            if metadata.get('date_taken'):
                parts.append(f"Date: {metadata['date_taken']}")
            if metadata.get('location'):
                parts.append(f"Location: {metadata['location']}")

        return " | ".join(parts)

    def store_in_qdrant(self, items_with_embeddings: List[Dict]) -> bool:
        """Store embeddings in Qdrant"""
        try:
            points = []
            for item in items_with_embeddings:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, item['item_id']))
                points.append({
                    "id": point_id,
                    "vector": item['embedding'],
                    "payload": {
                        "item_id": item['item_id'],
                        "type": item['type'],
                        "title": item['title'],
                        "text": item['text'],
                        "source": "knowledge_items",
                        "timestamp": datetime.now().isoformat()
                    }
                })

            response = requests.put(
                f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points",
                json={"points": points},
                params={"wait": "true"}
            )

            if response.status_code == 200:
                logger.info(f"✅ Stored {len(points)} embeddings in Qdrant")
                return True
            else:
                logger.error(f"Failed to store in Qdrant: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error storing in Qdrant: {e}")
            return False

    def update_database(self, items_with_embeddings: List[Dict]) -> bool:
        """Update PostgreSQL with embeddings"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            for item in items_with_embeddings:
                # Convert embedding to PostgreSQL vector format
                embedding_str = '[' + ','.join(map(str, item['embedding'])) + ']'
                cursor.execute("""
                    UPDATE knowledge_items
                    SET embedding = %s::vector(1024),
                        updated_at = NOW()
                    WHERE item_id = %s
                """, (embedding_str, item['item_id']))

            conn.commit()
            cursor.close()
            conn.close()
            logger.info(f"✅ Updated {len(items_with_embeddings)} items in database")
            return True

        except Exception as e:
            logger.error(f"Database update error: {e}")
            return False

    def process_batch(self, items: List[Dict]) -> Dict[str, int]:
        """Process a batch of items"""
        batch_stats = {'successful': 0, 'failed': 0, 'skipped': 0}
        items_with_embeddings = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_item = {}

            for item in items:
                text = self.create_text_representation(item)
                if not text or len(text) < 10:
                    batch_stats['skipped'] += 1
                    continue

                future = executor.submit(self.generate_embedding, text)
                future_to_item[future] = (item, text)

            for future in as_completed(future_to_item):
                if not self.running:
                    break

                item, text = future_to_item[future]
                try:
                    embedding = future.result()
                    if embedding:
                        items_with_embeddings.append({
                            'item_id': item['item_id'],
                            'type': item['type'],
                            'title': item['title'],
                            'text': text,
                            'embedding': embedding
                        })
                        batch_stats['successful'] += 1
                    else:
                        batch_stats['failed'] += 1
                        self.stats['errors'].append(f"Failed to generate embedding for {item['item_id']}")
                except Exception as e:
                    batch_stats['failed'] += 1
                    self.stats['errors'].append(f"Error processing {item['item_id']}: {e}")

        # Store batch
        if items_with_embeddings:
            if self.store_in_qdrant(items_with_embeddings):
                self.update_database(items_with_embeddings)

        return batch_stats

    def run_regeneration(self):
        """Main regeneration process"""
        logger.info("🚀 Starting embedding regeneration...")

        if not self.verify_collection():
            logger.error("Collection verification failed")
            return

        while self.running:
            items = self.get_items_without_embeddings()
            if not items:
                logger.info("✅ No more items to process")
                break

            logger.info(f"Processing batch of {len(items)} items...")
            batch_stats = self.process_batch(items)

            # Update stats
            self.stats['successful'] += batch_stats['successful']
            self.stats['failed'] += batch_stats['failed']
            self.stats['skipped'] += batch_stats['skipped']
            self.stats['total_processed'] += len(items)

            # Log progress
            logger.info(f"Progress: {self.stats['successful']} successful, "
                       f"{self.stats['failed']} failed, {self.stats['skipped']} skipped")

            # Brief pause between batches
            time.sleep(1)

        self.print_final_report()

    def print_final_report(self):
        """Print final statistics"""
        logger.info("\n" + "="*60)
        logger.info("REGENERATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Total processed: {self.stats['total_processed']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Skipped: {self.stats['skipped']}")

        if self.stats['errors']:
            logger.info(f"\nFirst 10 errors:")
            for error in self.stats['errors'][:10]:
                logger.error(f"  - {error}")

class EmbeddingMonitor:
    """Monitor embedding generation progress"""

    def __init__(self):
        self.running = True
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        self.running = False

    def check_status(self):
        """Check current status of embeddings"""
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
                       COUNT(embedding) as with_embeddings
                FROM knowledge_items
                GROUP BY knowledge_type
                ORDER BY total DESC
            """)

            logger.info("\nDatabase status:")
            for row in cursor.fetchall():
                logger.info(f"  {row[0]}: {row[2]}/{row[1]} have embeddings")

            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Monitor error: {e}")

    def run_monitor(self):
        """Run continuous monitoring"""
        logger.info("📊 Starting embedding monitor...")
        while self.running:
            self.check_status()
            time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        monitor = EmbeddingMonitor()
        monitor.run_monitor()
    else:
        regenerator = EmbeddingRegenerator()
        regenerator.run_regeneration()