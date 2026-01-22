#!/usr/bin/env python3
"""
Consolidated Photo Pipeline for Echo Brain
Unifies all photo sources into PostgreSQL + Qdrant
No more SQLite bullshit - everything in one place
"""

import os
import hashlib
import json
import logging
import psycopg2
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DB_CONFIG = {
    'dbname': 'echo_brain',
    'user': 'patrick',
    'password': 'RP78eIrW7cI2jYvL5akt1yurE',
    'host': 'localhost'
}

OLLAMA_URL = "http://localhost:11434/api/embeddings"
QDRANT_URL = "http://localhost:6333"
COLLECTION = "echo_memory"
MODEL = "mxbai-embed-large:latest"

class ConsolidatedPhotoSystem:
    """Single source of truth for all photos"""

    def __init__(self):
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.ensure_schema()

    def ensure_schema(self):
        """Create unified photo tables in PostgreSQL"""
        cursor = self.conn.cursor()

        # Main photo media table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS photo_media (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                file_path TEXT UNIQUE NOT NULL,
                file_hash VARCHAR(64) NOT NULL,
                file_size BIGINT,
                mime_type VARCHAR(100),
                width INTEGER,
                height INTEGER,
                date_taken TIMESTAMP,
                source VARCHAR(50), -- 'takeout', 'cloud', 'local'
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_photo_hash ON photo_media(file_hash);
            CREATE INDEX IF NOT EXISTS idx_photo_source ON photo_media(source);
            CREATE INDEX IF NOT EXISTS idx_photo_date ON photo_media(date_taken);
        """)

        # Deduplication tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS photo_duplicates (
                id SERIAL PRIMARY KEY,
                original_id UUID REFERENCES photo_media(id),
                duplicate_path TEXT,
                duplicate_source VARCHAR(50),
                similarity_score FLOAT,
                detected_at TIMESTAMP DEFAULT NOW()
            );
        """)

        # Vector tracking (links to Qdrant)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS photo_vectors (
                photo_id UUID PRIMARY KEY REFERENCES photo_media(id),
                vector_id VARCHAR(100), -- Qdrant point ID
                collection VARCHAR(50),
                dimensions INTEGER,
                model VARCHAR(100),
                generated_at TIMESTAMP DEFAULT NOW()
            );
        """)

        self.conn.commit()
        logger.info("✅ Schema ready")

    def calculate_hash(self, filepath: str) -> str:
        """Calculate SHA256 hash for deduplication"""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def find_all_photos(self) -> Dict[str, List[Path]]:
        """Find all photos across the system"""
        locations = {
            'takeout': [
                Path('/mnt/10TB2/Google_Takeout_2025/Takeout/Google Photos'),
                Path('/mnt/10TB2/Google_Takeout_2025/AllExtracted/Takeout/Google Photos')
            ],
            'local': [
                Path('/home/patrick/Pictures'),
                Path('/home/patrick/Videos'),
                Path('/mnt/10TB2/staging/google_photos_incoming')
            ]
        }

        results = {'takeout': [], 'local': [], 'cloud': []}
        extensions = {'.jpg', '.jpeg', '.png', '.mp4', '.mov', '.avi', '.mkv'}

        for source, paths in locations.items():
            for base_path in paths:
                if not base_path.exists():
                    continue

                for ext in extensions:
                    results[source].extend(base_path.rglob(f'*{ext}'))
                    results[source].extend(base_path.rglob(f'*{ext.upper()}'))

        logger.info(f"Found: Takeout={len(results['takeout'])}, Local={len(results['local'])}")
        return results

    def import_photo(self, filepath: Path, source: str) -> Optional[uuid.UUID]:
        """Import single photo with deduplication"""
        cursor = self.conn.cursor()

        # Calculate hash
        try:
            file_hash = self.calculate_hash(str(filepath))
        except Exception as e:
            logger.error(f"Hash error for {filepath}: {e}")
            return None

        # Check for duplicate
        cursor.execute("""
            SELECT id, file_path FROM photo_media WHERE file_hash = %s
        """, (file_hash,))

        existing = cursor.fetchone()
        if existing:
            # Record duplicate
            cursor.execute("""
                INSERT INTO photo_duplicates (original_id, duplicate_path, duplicate_source)
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (existing[0], str(filepath), source))
            self.conn.commit()
            return None

        # Extract metadata
        metadata = {
            'original_name': filepath.name,
            'parent_dir': filepath.parent.name
        }

        # Get file info
        stat = filepath.stat()

        # Insert new photo
        cursor.execute("""
            INSERT INTO photo_media (
                file_path, file_hash, file_size, source, metadata
            ) VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (str(filepath), file_hash, stat.st_size, source, json.dumps(metadata)))

        photo_id = cursor.fetchone()[0]
        self.conn.commit()

        return photo_id

    def generate_embedding(self, photo_id: uuid.UUID) -> bool:
        """Generate and store 1024-dim embedding"""
        cursor = self.conn.cursor()

        # Get photo info
        cursor.execute("""
            SELECT file_path, metadata FROM photo_media WHERE id = %s
        """, (photo_id,))

        row = cursor.fetchone()
        if not row:
            return False

        filepath, metadata = row

        # Create text for embedding
        text_parts = [f"Photo: {Path(filepath).name}"]
        if metadata:
            if metadata.get('parent_dir'):
                text_parts.append(f"Folder: {metadata['parent_dir']}")

        text = " | ".join(text_parts)

        # Generate embedding
        try:
            response = requests.post(
                OLLAMA_URL,
                json={"model": MODEL, "prompt": text},
                timeout=30
            )

            if response.status_code != 200:
                return False

            embedding = response.json().get('embedding')
            if not embedding or len(embedding) != 1024:
                return False

        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return False

        # Store in Qdrant
        point_id = str(photo_id)

        try:
            response = requests.put(
                f"{QDRANT_URL}/collections/{COLLECTION}/points",
                json={
                    "points": [{
                        "id": point_id,
                        "vector": embedding,
                        "payload": {
                            "photo_id": str(photo_id),
                            "file_path": filepath,
                            "source": "photo_media",
                            "type": "photo"
                        }
                    }]
                },
                params={"wait": "true"}
            )

            if response.status_code != 200:
                return False

        except Exception as e:
            logger.error(f"Qdrant error: {e}")
            return False

        # Record vector creation
        cursor.execute("""
            INSERT INTO photo_vectors (photo_id, vector_id, collection, dimensions, model)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (photo_id) DO UPDATE SET
                vector_id = EXCLUDED.vector_id,
                generated_at = NOW()
        """, (photo_id, point_id, COLLECTION, 1024, MODEL))

        self.conn.commit()
        return True

    def migrate_from_sqlite(self):
        """Import existing SQLite data"""
        import sqlite3

        sqlite_dbs = [
            '/mnt/10TB2/Google_Takeout_2025/photos_comparison.db',
            '/opt/tower-auth/google_photos.db'
        ]

        for db_path in sqlite_dbs:
            if not Path(db_path).exists():
                continue

            logger.info(f"Migrating {db_path}")

            sqlite_conn = sqlite3.connect(db_path)
            sqlite_cursor = sqlite_conn.cursor()

            # Check tables
            tables = sqlite_cursor.execute("""
                SELECT name FROM sqlite_master WHERE type='table'
            """).fetchall()

            for table in tables:
                table_name = table[0]
                if 'photo' not in table_name.lower():
                    continue

                # Get data
                rows = sqlite_cursor.execute(f"SELECT * FROM {table_name}").fetchall()
                logger.info(f"  Found {len(rows)} rows in {table_name}")

                # Import based on table structure
                # (Implementation specific to each table)

            sqlite_conn.close()

    def run_full_pipeline(self):
        """Run complete consolidation"""
        logger.info("=== CONSOLIDATED PHOTO PIPELINE ===")

        # 1. Find all photos
        all_photos = self.find_all_photos()

        # 2. Import with deduplication
        imported = 0
        duplicates = 0

        for source, paths in all_photos.items():
            logger.info(f"Processing {source}: {len(paths)} files")

            for filepath in paths[:100]:  # Test batch
                photo_id = self.import_photo(filepath, source)

                if photo_id:
                    imported += 1
                    # Generate embedding
                    if self.generate_embedding(photo_id):
                        logger.info(f"✅ {filepath.name}")
                else:
                    duplicates += 1

        # 3. Report
        logger.info(f"""
        === COMPLETE ===
        Imported: {imported}
        Duplicates: {duplicates}
        """)

        # 4. Stats
        cursor = self.conn.cursor()
        cursor.execute("SELECT source, COUNT(*) FROM photo_media GROUP BY source")
        for row in cursor.fetchall():
            logger.info(f"  {row[0]}: {row[1]} photos")

if __name__ == "__main__":
    system = ConsolidatedPhotoSystem()
    system.migrate_from_sqlite()
    system.run_full_pipeline()