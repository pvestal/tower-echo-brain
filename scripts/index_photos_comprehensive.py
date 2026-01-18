#!/usr/bin/env python3
"""
Comprehensive Photo Indexing for Echo Brain
Index all 142,658+ photos into Qdrant for semantic search and retrieval.

Enables Echo Brain to answer questions like:
- "Show me photos from Christmas 2023"
- "Find sunset photos from last summer"
- "What photos did I take in New York?"
"""
import asyncio
import sqlite3
import sys
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4
import json

sys.path.insert(0, '/opt/tower-echo-brain')

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from src.services.embedding_service import create_embedding_service

# Configuration
PHOTOS_DB = "/mnt/10TB2/Google_Takeout_2025/photos_comparison.db"
QDRANT_URL = "http://192.168.50.135:6333"
BATCH_SIZE = 50

async def index_photos_for_echo_brain():
    """Index all photos into Echo Brain's semantic search system."""
    print(f"=== Photo Indexing for Echo Brain Started: {datetime.now().isoformat()} ===\n")

    # Initialize services
    print("🔧 Initializing embedding service...")
    embedding_service = await create_embedding_service()
    qdrant = QdrantClient(url=QDRANT_URL)

    # Create photos collection
    print("📦 Creating photos collection in Qdrant...")
    try:
        qdrant.delete_collection("photos")
        print("  ✅ Deleted existing photos collection")
    except:
        pass

    qdrant.create_collection(
        collection_name="photos",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    print("  ✅ Created new photos collection")

    # Connect to photos database
    conn = sqlite3.connect(PHOTOS_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get photo counts
    tower_count = cursor.execute("SELECT COUNT(*) FROM tower_photos").fetchone()[0]
    local_count = cursor.execute("SELECT COUNT(*) FROM local_photos").fetchone()[0]
    total_photos = tower_count + local_count

    print(f"📊 Found {total_photos:,} photos to index:")
    print(f"  • Tower photos: {tower_count:,}")
    print(f"  • Local photos: {local_count:,}\n")

    # Index Tower photos
    print("🏗️ Indexing Tower photos...")
    tower_photos = cursor.execute("""
        SELECT path, size, hash, date, rowid
        FROM tower_photos
        ORDER BY date DESC
    """).fetchall()

    await index_photo_batch(
        photos=tower_photos,
        embedding_service=embedding_service,
        qdrant=qdrant,
        photo_type="tower",
        batch_size=BATCH_SIZE
    )

    # Index Local photos
    print("\n🏠 Indexing Local photos...")
    local_photos = cursor.execute("""
        SELECT filepath, filename, size, hash, date_taken, id
        FROM local_photos
        ORDER BY date_taken DESC
    """).fetchall()

    await index_photo_batch(
        photos=local_photos,
        embedding_service=embedding_service,
        qdrant=qdrant,
        photo_type="local",
        batch_size=BATCH_SIZE
    )

    # Verification
    print("\n=== Verification ===")
    try:
        collection_info = qdrant.get_collection("photos")
        print(f"✅ Photos collection: {collection_info.points_count:,} photos indexed")

        # Test search
        test_query = "Christmas family photos December"
        test_embedding = await embedding_service.embed_single(test_query)
        results = qdrant.search(
            collection_name="photos",
            query_vector=test_embedding,
            limit=3
        )

        print(f"🔍 Test search for '{test_query}':")
        if results:
            print(f"  ✅ Found {len(results)} results")
            for i, result in enumerate(results, 1):
                photo_info = result.payload.get('filename', 'Unknown')[:50]
                print(f"    {i}. {photo_info} (score: {result.score:.3f})")
        else:
            print("  ⚠️ No results found")

    except Exception as e:
        print(f"❌ Verification error: {e}")

    conn.close()
    print(f"\n=== Photo Indexing Complete: {datetime.now().isoformat()} ===")

async def index_photo_batch(photos, embedding_service, qdrant, photo_type, batch_size):
    """Index a batch of photos with progress tracking."""
    total = len(photos)
    processed = 0

    for i in range(0, total, batch_size):
        batch = photos[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total - 1) // batch_size + 1

        print(f"  Processing {photo_type} batch {batch_num}/{total_batches} ({len(batch)} photos)...")

        points = []
        for photo in batch:
            try:
                # Extract photo metadata based on type
                if photo_type == "tower":
                    searchable_text = create_tower_photo_text(photo)
                    payload = create_tower_photo_payload(photo)
                else:  # local
                    searchable_text = create_local_photo_text(photo)
                    payload = create_local_photo_payload(photo)

                # Generate embedding
                embedding = await embedding_service.embed_single(searchable_text)

                # Create point
                point = PointStruct(
                    id=str(uuid4()),
                    vector=embedding,
                    payload=payload
                )
                points.append(point)

            except Exception as e:
                print(f"    ⚠️ Error processing photo {photo[0] if photo_type == 'tower' else photo[1]}: {e}")
                continue

        # Upload batch to Qdrant
        try:
            qdrant.upsert(collection_name="photos", points=points)
            processed += len(points)
            print(f"    ✅ Uploaded {len(points)} photos (Total: {processed:,}/{total:,})")
        except Exception as e:
            print(f"    ❌ Failed to upload batch: {e}")

def create_tower_photo_text(photo):
    """Create searchable text for tower photos."""
    path = photo['path'] or ''
    date = photo['date'] or ''
    size = photo['size'] or 0

    # Extract meaningful info from path
    path_parts = Path(path).parts if path else []
    filename = Path(path).name if path else ''

    # Extract date information
    date_info = extract_date_context(date)

    # Extract location/event info from path
    location_info = extract_location_from_path(path)

    # Create comprehensive searchable text
    text = f"""
Photo: {filename}
Date: {date} {date_info}
Location: {location_info}
Path: {' '.join(path_parts)}
Size: {format_file_size(size)}
Type: tower photo
Collection: {extract_collection_from_path(path)}
"""

    return text.strip()

def create_local_photo_text(photo):
    """Create searchable text for local photos."""
    filepath = photo['filepath'] or ''
    filename = photo['filename'] or ''
    date_taken = photo['date_taken'] or ''
    size = photo['size'] or 0

    # Extract meaningful info
    path_parts = Path(filepath).parts if filepath else []
    date_info = extract_date_context(date_taken)
    location_info = extract_location_from_path(filepath)

    # Create comprehensive searchable text
    text = f"""
Photo: {filename}
Date: {date_taken} {date_info}
Location: {location_info}
Path: {' '.join(path_parts)}
Size: {format_file_size(size)}
Type: local photo
Collection: {extract_collection_from_path(filepath)}
"""

    return text.strip()

def create_tower_photo_payload(photo):
    """Create Qdrant payload for tower photos."""
    return {
        'source': 'tower_photos',
        'path': photo['path'],
        'filename': Path(photo['path']).name if photo['path'] else '',
        'size': photo['size'],
        'hash': photo['hash'],
        'date': photo['date'],
        'date_context': extract_date_context(photo['date']),
        'location': extract_location_from_path(photo['path']),
        'collection': extract_collection_from_path(photo['path']),
        'file_extension': Path(photo['path']).suffix.lower() if photo['path'] else '',
        'searchable_text': create_tower_photo_text(photo)[:1000]
    }

def create_local_photo_payload(photo):
    """Create Qdrant payload for local photos."""
    return {
        'source': 'local_photos',
        'filepath': photo['filepath'],
        'filename': photo['filename'],
        'size': photo['size'],
        'hash': photo['hash'],
        'date_taken': photo['date_taken'],
        'date_context': extract_date_context(photo['date_taken']),
        'location': extract_location_from_path(photo['filepath']),
        'collection': extract_collection_from_path(photo['filepath']),
        'file_extension': Path(photo['filename']).suffix.lower() if photo['filename'] else '',
        'searchable_text': create_local_photo_text(photo)[:1000]
    }

def extract_date_context(date_str):
    """Extract semantic date context (season, holiday, etc.)."""
    if not date_str:
        return ""

    try:
        # Parse various date formats
        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y']:
            try:
                date_obj = datetime.strptime(date_str.split()[0], fmt)
                break
            except:
                continue
        else:
            return ""

        context = []

        # Season
        month = date_obj.month
        if month in [12, 1, 2]:
            context.append("winter")
        elif month in [3, 4, 5]:
            context.append("spring")
        elif month in [6, 7, 8]:
            context.append("summer")
        else:
            context.append("fall")

        # Holiday context
        if month == 12 and date_obj.day >= 20:
            context.append("Christmas holiday season")
        elif month == 10 and date_obj.day >= 25:
            context.append("Halloween")
        elif month == 7 and date_obj.day == 4:
            context.append("Fourth of July Independence Day")
        elif month == 11 and 22 <= date_obj.day <= 28:
            context.append("Thanksgiving")

        # Year context
        context.append(f"year {date_obj.year}")

        return " ".join(context)

    except Exception:
        return ""

def extract_location_from_path(path):
    """Extract location information from file path."""
    if not path:
        return ""

    path_lower = path.lower()
    locations = []

    # Common location patterns
    location_patterns = [
        r'(?i)(new\s*york|nyc|manhattan|brooklyn)',
        r'(?i)(los\s*angeles|la|hollywood)',
        r'(?i)(chicago|illinois)',
        r'(?i)(san\s*francisco|sf|bay\s*area)',
        r'(?i)(paris|france)',
        r'(?i)(london|england|uk)',
        r'(?i)(tokyo|japan)',
        r'(?i)(beach|ocean|lake|mountain|park)',
        r'(?i)(vacation|trip|travel)',
        r'(?i)(home|house|garden)',
        r'(?i)(wedding|party|event)',
        r'(?i)(christmas|holiday|birthday)'
    ]

    for pattern in location_patterns:
        matches = re.findall(pattern, path)
        locations.extend([m.lower().strip() for m in matches if isinstance(m, str)])

    return " ".join(set(locations)) if locations else "unknown location"

def extract_collection_from_path(path):
    """Extract collection/album name from path."""
    if not path:
        return ""

    # Get the parent directory name as collection
    parts = Path(path).parts
    if len(parts) > 1:
        return parts[-2].replace('_', ' ').replace('-', ' ')
    return ""

def format_file_size(size_bytes):
    """Format file size in human readable format."""
    if not size_bytes:
        return "unknown size"

    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"

if __name__ == "__main__":
    asyncio.run(index_photos_for_echo_brain())