#!/usr/bin/env python3
"""
Google Suite & Drive Integration for Echo Brain
Uses Google APIs to ingest Docs, Sheets, Gmail, Photos, etc.
"""

import json
import os
import logging
from pathlib import Path
from datetime import datetime
import uuid
import httpx

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/echo_google_ingestion.log'),
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
    'google_docs': 0,
    'google_sheets': 0,
    'google_photos': 0,
    'gmail': 0,
    'google_drive': 0,
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
                logger.info(f"✅ Uploaded {len(points)} Google vectors")
                return True
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        stats['errors'] += 1
    return False

def check_google_credentials():
    """Check for Google API credentials"""
    credential_paths = [
        "/home/patrick/.config/gcloud/credentials.db",
        "/home/patrick/.credentials/",
        "/home/patrick/.config/google/",
        "/home/patrick/google-credentials.json"
    ]

    found_creds = []
    for path in credential_paths:
        if Path(path).exists():
            found_creds.append(path)

    return found_creds

def ingest_google_takeout():
    """Ingest Google Takeout data if available"""
    logger.info("📦 CHECKING FOR GOOGLE TAKEOUT DATA...")
    points = []

    # Common Google Takeout locations
    takeout_dirs = [
        "/home/patrick/Downloads/Takeout",
        "/home/patrick/Google Takeout",
        "/home/patrick/Documents/Google Takeout",
        "/mnt/1TB-storage/Google Takeout",
        "/mnt/10TB2/Google Takeout"
    ]

    for takeout_dir in takeout_dirs:
        if not Path(takeout_dir).exists():
            continue

        logger.info(f"📁 Found Google Takeout at {takeout_dir}")

        # Process different Google services
        services = {
            'Drive': 'google_drive',
            'Photos': 'google_photos',
            'Gmail': 'gmail',
            'Docs': 'google_docs',
            'Sheets': 'google_sheets',
            'Keep': 'google_keep',
            'YouTube': 'youtube',
            'Search': 'google_search',
            'Maps': 'google_maps',
            'Calendar': 'google_calendar'
        }

        for service_name, service_type in services.items():
            service_path = Path(takeout_dir) / service_name
            if service_path.exists():
                logger.info(f"   📂 Processing {service_name}...")

                # Process all files in service directory
                for file_path in service_path.rglob("*"):
                    if file_path.is_file():
                        try:
                            # Handle different file types
                            content = ""

                            if file_path.suffix == '.json':
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                    content = json.dumps(data, indent=2)[:1500]

                            elif file_path.suffix in ['.txt', '.html']:
                                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read()[:1500]

                            elif file_path.suffix in ['.csv']:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read()[:1000]

                            if content:
                                description = f"Google {service_name}: {file_path.name}\n"
                                description += f"Service: {service_name}\n"
                                description += f"Path: {file_path}\n"
                                description += f"Content:\n{content}"

                                embedding = embed_text(description)
                                if embedding:
                                    points.append({
                                        "id": str(uuid.uuid4()),
                                        "vector": embedding,
                                        "payload": {
                                            "content": description[:1000],
                                            "source": str(file_path),
                                            "type": service_type,
                                            "service": service_name,
                                            "filename": file_path.name,
                                            "timestamp": datetime.now().isoformat()
                                        }
                                    })
                                    stats[service_type] = stats.get(service_type, 0) + 1
                                    stats['total'] += 1

                                    if len(points) >= 50:
                                        upload_batch(points)
                                        points = []

                        except Exception as e:
                            logger.error(f"Error processing {file_path}: {e}")
                            stats['errors'] += 1

    upload_batch(points)
    return len(takeout_dirs) > 0

def ingest_browser_data():
    """Ingest browser history and bookmarks"""
    logger.info("🌐 INGESTING BROWSER DATA...")
    points = []

    # Common browser data locations
    browser_paths = {
        'Chrome': "/home/patrick/.config/google-chrome/Default",
        'Chromium': "/home/patrick/.config/chromium/Default",
        'Firefox': "/home/patrick/.mozilla/firefox"
    }

    for browser, path in browser_paths.items():
        if Path(path).exists():
            logger.info(f"🔍 Found {browser} data")

            # Look for history and bookmarks files
            for db_file in Path(path).glob("*.db"):
                if db_file.name in ['History', 'Bookmarks', 'Web Data']:
                    try:
                        # Create metadata entry for browser data
                        stat = db_file.stat()
                        content = f"Browser: {browser}\n"
                        content += f"Database: {db_file.name}\n"
                        content += f"Size: {stat.st_size} bytes\n"
                        content += f"Modified: {datetime.fromtimestamp(stat.st_mtime)}\n"
                        content += f"Contains: browsing history, bookmarks, form data\n"

                        embedding = embed_text(content)
                        if embedding:
                            points.append({
                                "id": str(uuid.uuid4()),
                                "vector": embedding,
                                "payload": {
                                    "content": content,
                                    "source": str(db_file),
                                    "type": "browser_data",
                                    "browser": browser,
                                    "database": db_file.name,
                                    "timestamp": datetime.now().isoformat()
                                }
                            })
                            stats['google_drive'] += 1  # Use drive as catch-all
                            stats['total'] += 1

                    except Exception as e:
                        logger.error(f"Error processing browser db {db_file}: {e}")
                        stats['errors'] += 1

    upload_batch(points)

def ingest_cloud_sync_data():
    """Ingest cloud sync metadata"""
    logger.info("☁️ INGESTING CLOUD SYNC DATA...")
    points = []

    # Cloud sync locations
    sync_dirs = [
        "/home/patrick/.config/rclone",
        "/home/patrick/.config/google-drive-ocamlfuse",
        "/home/patrick/.local/share/gnome-online-accounts",
        "/home/patrick/.config/evolution",
        "/home/patrick/.thunderbird"
    ]

    for sync_dir in sync_dirs:
        if Path(sync_dir).exists():
            logger.info(f"📂 Processing cloud sync: {Path(sync_dir).name}")

            for config_file in Path(sync_dir).rglob("*"):
                if config_file.is_file() and config_file.suffix in ['.conf', '.json', '.db', '.cfg']:
                    try:
                        with open(config_file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()[:1000]

                        description = f"Cloud Sync Config: {config_file.name}\n"
                        description += f"Service: {Path(sync_dir).name}\n"
                        description += f"Path: {config_file}\n"
                        description += f"Content:\n{content}"

                        embedding = embed_text(description)
                        if embedding:
                            points.append({
                                "id": str(uuid.uuid4()),
                                "vector": embedding,
                                "payload": {
                                    "content": description[:1000],
                                    "source": str(config_file),
                                    "type": "cloud_sync",
                                    "service": Path(sync_dir).name,
                                    "timestamp": datetime.now().isoformat()
                                }
                            })
                            stats['google_drive'] += 1
                            stats['total'] += 1

                    except Exception as e:
                        logger.error(f"Error processing sync config {config_file}: {e}")
                        stats['errors'] += 1

    upload_batch(points)

def main():
    logger.info("=" * 70)
    logger.info("🔍 ECHO BRAIN: GOOGLE SUITE & CLOUD DATA INGESTION")
    logger.info("=" * 70)

    # Check for Google credentials
    creds = check_google_credentials()
    if creds:
        logger.info(f"✅ Found Google credentials at: {', '.join(creds)}")
    else:
        logger.warning("⚠️ No Google API credentials found - using available data only")

    # Run ingestion functions
    try:
        takeout_found = ingest_google_takeout()
        if not takeout_found:
            logger.info("ℹ️ No Google Takeout data found")
    except Exception as e:
        logger.error(f"Google Takeout ingestion failed: {e}")

    try:
        ingest_browser_data()
    except Exception as e:
        logger.error(f"Browser data ingestion failed: {e}")

    try:
        ingest_cloud_sync_data()
    except Exception as e:
        logger.error(f"Cloud sync ingestion failed: {e}")

    # Final stats
    logger.info("\n" + "=" * 70)
    logger.info("✅ GOOGLE SUITE INGESTION COMPLETE!")
    logger.info(f"   • Google Docs: {stats['google_docs']:,}")
    logger.info(f"   • Google Sheets: {stats['google_sheets']:,}")
    logger.info(f"   • Google Photos: {stats['google_photos']:,}")
    logger.info(f"   • Gmail: {stats['gmail']:,}")
    logger.info(f"   • Google Drive: {stats['google_drive']:,}")
    logger.info(f"   • TOTAL: {stats['total']:,} new vectors")
    logger.info(f"   • Errors: {stats['errors']:,}")
    logger.info("=" * 70)

    # Save stats
    with open("/tmp/echo_google_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    if stats['total'] > 0:
        logger.info(f"\n🎉 Added {stats['total']} Google/cloud vectors to Echo Brain!")
        logger.info("🔍 Your Google data is now searchable through Echo Brain")
    else:
        logger.info("\nℹ️ No Google data found to ingest")
        logger.info("💡 To add Google data:")
        logger.info("   • Download Google Takeout to ~/Downloads/Takeout")
        logger.info("   • Set up Google Drive sync")
        logger.info("   • Configure Google API credentials")

if __name__ == "__main__":
    main()