#!/usr/bin/env python3
"""
Enhanced Echo Brain Ingestion - Personal Media & Google Data
Includes photos, videos, Google Drive, and all personal data
"""

import json
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import uuid
import httpx
import hashlib
from PIL import Image, ExifTags
import subprocess
import mimetypes

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/echo_media_ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "mxbai-embed-large"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "echo_memory"

# Track ingestion stats
stats = {
    'photos': 0,
    'videos': 0,
    'google_data': 0,
    'documents': 0,
    'total': 0,
    'errors': 0
}

def embed_text(text):
    """Create embedding using Ollama"""
    if not text or len(text) < 10:
        return None
    try:
        with httpx.Client(timeout=30) as client:
            text_to_embed = text[:2000] if len(text) > 2000 else text
            response = client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": text_to_embed}
            )
            if response.status_code == 200:
                return response.json()["embedding"]
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        stats['errors'] += 1
    return None

def upload_batch(points):
    """Upload points to Qdrant"""
    if not points:
        return True
    try:
        with httpx.Client(timeout=60) as client:
            response = client.put(
                f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points",
                json={"points": points}
            )
            if response.status_code == 200:
                logger.info(f"✅ Uploaded {len(points)} media vectors")
                return True
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        stats['errors'] += 1
    return False

def extract_photo_metadata(image_path):
    """Extract metadata from photos"""
    try:
        with Image.open(image_path) as img:
            # Basic info
            info = {
                'filename': image_path.name,
                'size': f"{img.width}x{img.height}",
                'format': img.format,
                'mode': img.mode
            }

            # EXIF data
            exifdata = img.getexif()
            if exifdata:
                for tag_id in exifdata:
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    data = exifdata.get(tag_id)
                    if isinstance(data, bytes):
                        data = data.decode('utf-8', errors='ignore')
                    info[str(tag)] = str(data)[:100]  # Limit length

            return info
    except Exception as e:
        logger.error(f"Error extracting metadata from {image_path}: {e}")
        return {'filename': image_path.name, 'error': str(e)}

def extract_video_metadata(video_path):
    """Extract metadata from videos"""
    try:
        # Use ffprobe to get video metadata
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            metadata = json.loads(result.stdout)

            # Extract key info
            info = {
                'filename': video_path.name,
                'duration': metadata.get('format', {}).get('duration', 'unknown'),
                'size': metadata.get('format', {}).get('size', 'unknown'),
                'format_name': metadata.get('format', {}).get('format_name', 'unknown'),
                'bit_rate': metadata.get('format', {}).get('bit_rate', 'unknown')
            }

            # Video stream info
            for stream in metadata.get('streams', []):
                if stream.get('codec_type') == 'video':
                    info['video_codec'] = stream.get('codec_name', 'unknown')
                    info['resolution'] = f"{stream.get('width', 0)}x{stream.get('height', 0)}"
                    info['fps'] = stream.get('r_frame_rate', 'unknown')
                    break

            return info
        else:
            return {'filename': video_path.name, 'error': 'ffprobe failed'}

    except Exception as e:
        logger.error(f"Error extracting video metadata from {video_path}: {e}")
        return {'filename': video_path.name, 'error': str(e)}

def ingest_photos():
    """Ingest photo metadata and content"""
    logger.info("📸 INGESTING PHOTOS...")
    points = []

    # Photo directories to check
    photo_dirs = [
        "/home/patrick/Pictures",
        "/home/patrick/Downloads",
        "/home/patrick/Documents",
        "/mnt/1TB-storage",  # Check for mounted storage
        "/mnt/10TB2"  # Your large storage
    ]

    photo_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

    for base_dir in photo_dirs:
        if not Path(base_dir).exists():
            continue

        logger.info(f"📂 Scanning {base_dir} for photos...")

        for ext in photo_extensions:
            for photo_path in Path(base_dir).rglob(f"*{ext}"):
                # Skip system/cache files
                if any(skip in str(photo_path) for skip in ['.cache', '.git', 'node_modules', '__pycache__']):
                    continue

                try:
                    metadata = extract_photo_metadata(photo_path)

                    # Create searchable content
                    content = f"Photo: {photo_path.name}\n"
                    content += f"Location: {photo_path.parent}\n"
                    content += f"Directory: {photo_path.parent.name}\n"

                    for key, value in metadata.items():
                        content += f"{key}: {value}\n"

                    # Add path context for organization
                    path_parts = str(photo_path).split('/')
                    if len(path_parts) > 3:
                        content += f"Context: {' > '.join(path_parts[-4:-1])}\n"

                    embedding = embed_text(content)
                    if embedding:
                        points.append({
                            "id": str(uuid.uuid4()),
                            "vector": embedding,
                            "payload": {
                                "content": content[:1000],
                                "source": str(photo_path),
                                "type": "photo",
                                "filename": photo_path.name,
                                "directory": str(photo_path.parent),
                                "metadata": metadata,
                                "timestamp": datetime.now().isoformat()
                            }
                        })
                        stats['photos'] += 1
                        stats['total'] += 1

                        if len(points) >= 50:
                            upload_batch(points)
                            points = []

                except Exception as e:
                    logger.error(f"Error processing photo {photo_path}: {e}")
                    stats['errors'] += 1

    upload_batch(points)
    logger.info(f"✅ Ingested {stats['photos']} photos")

def ingest_videos():
    """Ingest video metadata and content"""
    logger.info("🎬 INGESTING VIDEOS...")
    points = []

    # Video directories - including your linked Videos folder
    video_dirs = [
        "/home/patrick/Videos",  # This links to /mnt/10TB2/Videos
        "/mnt/10TB2/Videos",     # Direct access
        "/home/patrick/Downloads",
        "/mnt/1TB-storage"
    ]

    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v']

    for base_dir in video_dirs:
        if not Path(base_dir).exists():
            continue

        logger.info(f"🎥 Scanning {base_dir} for videos...")

        for ext in video_extensions:
            for video_path in Path(base_dir).rglob(f"*{ext}"):
                # Skip system files
                if any(skip in str(video_path) for skip in ['.cache', '.git', 'node_modules']):
                    continue

                try:
                    metadata = extract_video_metadata(video_path)

                    # Create searchable content
                    content = f"Video: {video_path.name}\n"
                    content += f"Location: {video_path.parent}\n"
                    content += f"Year Directory: {video_path.parent.name}\n"

                    for key, value in metadata.items():
                        content += f"{key}: {value}\n"

                    # Add year context from directory structure
                    path_parts = str(video_path).split('/')
                    for part in path_parts:
                        if part.isdigit() and len(part) == 4 and 1990 <= int(part) <= 2030:
                            content += f"Year: {part}\n"
                            break

                    embedding = embed_text(content)
                    if embedding:
                        points.append({
                            "id": str(uuid.uuid4()),
                            "vector": embedding,
                            "payload": {
                                "content": content[:1000],
                                "source": str(video_path),
                                "type": "video",
                                "filename": video_path.name,
                                "directory": str(video_path.parent),
                                "metadata": metadata,
                                "timestamp": datetime.now().isoformat()
                            }
                        })
                        stats['videos'] += 1
                        stats['total'] += 1

                        if len(points) >= 50:
                            upload_batch(points)
                            points = []

                except Exception as e:
                    logger.error(f"Error processing video {video_path}: {e}")
                    stats['errors'] += 1

    upload_batch(points)
    logger.info(f"✅ Ingested {stats['videos']} videos")

def ingest_google_data():
    """Ingest Google Drive, Photos, and cloud data"""
    logger.info("☁️ INGESTING GOOGLE DATA...")
    points = []

    # Look for Google data in various locations
    google_locations = [
        "/home/patrick/.config/google-drive-ocamlfuse",
        "/home/patrick/.config/rclone",
        "/home/patrick/Google Drive",
        "/home/patrick/GoogleDrive",
        "/home/patrick/Downloads/cloud-media",
        "/home/patrick/.local/share/gphotos-sync"
    ]

    for location in google_locations:
        if Path(location).exists():
            logger.info(f"Found Google data at {location}")

            # Process all files in Google directories
            for file_path in Path(location).rglob("*"):
                if file_path.is_file() and file_path.suffix in ['.json', '.db', '.sqlite', '.txt', '.log']:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()[:2000]

                        description = f"Google Data: {file_path.name}\n"
                        description += f"Location: {file_path}\n"
                        description += f"Type: Google sync/config\n"
                        description += f"Content:\n{content}"

                        embedding = embed_text(description)
                        if embedding:
                            points.append({
                                "id": str(uuid.uuid4()),
                                "vector": embedding,
                                "payload": {
                                    "content": description[:1000],
                                    "source": str(file_path),
                                    "type": "google_data",
                                    "filename": file_path.name,
                                    "timestamp": datetime.now().isoformat()
                                }
                            })
                            stats['google_data'] += 1
                            stats['total'] += 1

                            if len(points) >= 50:
                                upload_batch(points)
                                points = []

                    except Exception as e:
                        logger.error(f"Error processing Google file {file_path}: {e}")
                        stats['errors'] += 1

    upload_batch(points)
    logger.info(f"✅ Ingested {stats['google_data']} Google data files")

def ingest_personal_documents():
    """Ingest personal documents"""
    logger.info("📄 INGESTING PERSONAL DOCUMENTS...")
    points = []

    doc_dirs = [
        "/home/patrick/Documents",
        "/home/patrick/Desktop",
        "/home/patrick/Downloads"
    ]

    doc_extensions = ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt']

    for base_dir in doc_dirs:
        if not Path(base_dir).exists():
            continue

        for ext in doc_extensions:
            for doc_path in Path(base_dir).rglob(f"*{ext}"):
                if any(skip in str(doc_path) for skip in ['.cache', '.git', 'node_modules']):
                    continue

                try:
                    # For text files, read content directly
                    if ext in ['.txt', '.rtf']:
                        with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()[:1500]
                    else:
                        # For other formats, just use filename and metadata
                        stat = doc_path.stat()
                        content = f"Document: {doc_path.name}\n"
                        content += f"Size: {stat.st_size} bytes\n"
                        content += f"Modified: {datetime.fromtimestamp(stat.st_mtime)}\n"
                        content += f"Path: {doc_path.parent}\n"

                    description = f"Document: {doc_path.name}\n"
                    description += f"Type: {ext}\n"
                    description += f"Location: {doc_path}\n"
                    description += f"Content:\n{content}"

                    embedding = embed_text(description)
                    if embedding:
                        points.append({
                            "id": str(uuid.uuid4()),
                            "vector": embedding,
                            "payload": {
                                "content": description[:1000],
                                "source": str(doc_path),
                                "type": "document",
                                "filename": doc_path.name,
                                "extension": ext,
                                "timestamp": datetime.now().isoformat()
                            }
                        })
                        stats['documents'] += 1
                        stats['total'] += 1

                        if len(points) >= 50:
                            upload_batch(points)
                            points = []

                except Exception as e:
                    logger.error(f"Error processing document {doc_path}: {e}")
                    stats['errors'] += 1

    upload_batch(points)
    logger.info(f"✅ Ingested {stats['documents']} documents")

def main():
    logger.info("=" * 70)
    logger.info("📸 ECHO BRAIN: PERSONAL MEDIA & GOOGLE DATA INGESTION")
    logger.info("=" * 70)

    try:
        # Check if ffprobe is available for video metadata
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
        logger.info("✅ ffprobe available for video metadata")
    except:
        logger.warning("⚠️ ffprobe not available - video metadata will be limited")

    # Run all ingestion functions
    try:
        ingest_photos()
    except Exception as e:
        logger.error(f"Photo ingestion failed: {e}")

    try:
        ingest_videos()
    except Exception as e:
        logger.error(f"Video ingestion failed: {e}")

    try:
        ingest_google_data()
    except Exception as e:
        logger.error(f"Google data ingestion failed: {e}")

    try:
        ingest_personal_documents()
    except Exception as e:
        logger.error(f"Document ingestion failed: {e}")

    # Final stats
    logger.info("\n" + "=" * 70)
    logger.info("✅ PERSONAL MEDIA INGESTION COMPLETE!")
    logger.info(f"   • Photos: {stats['photos']:,}")
    logger.info(f"   • Videos: {stats['videos']:,}")
    logger.info(f"   • Google Data: {stats['google_data']:,}")
    logger.info(f"   • Documents: {stats['documents']:,}")
    logger.info(f"   • TOTAL: {stats['total']:,} new vectors")
    logger.info(f"   • Errors: {stats['errors']:,}")
    logger.info("=" * 70)

    # Save stats
    with open("/tmp/echo_media_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\n📊 Added {stats['total']} personal media vectors to Echo Brain")
    logger.info("🧠 Your photos, videos, and Google data are now searchable!")

if __name__ == "__main__":
    main()