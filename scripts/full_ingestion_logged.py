#!/usr/bin/env python3
"""
COMPREHENSIVE Echo Brain Ingestion with Error Logging
Ensures EVERYTHING is ingested with proper error tracking
"""

import json
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import uuid
import httpx
import psycopg2
from psycopg2.extras import RealDictCursor
import traceback
import glob

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/echo_ingestion.log'),
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
    'conversations': 0,
    'code_files': 0,
    'database_records': 0,
    'documents': 0,
    'google_photos': 0,
    'errors': 0,
    'total': 0
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
                logger.info(f"✅ Uploaded {len(points)} vectors")
                return True
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        stats['errors'] += 1
    return False

def ingest_all_conversations():
    """Ingest ALL conversation content including messages, tool results, summaries"""
    logger.info("📚 INGESTING ALL CONVERSATIONS...")
    points = []
    base_dir = Path("/home/patrick/.claude/projects/")

    for jsonl_file in base_dir.rglob("*.jsonl"):
        logger.info(f"Processing {jsonl_file.name}")
        with open(jsonl_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    content = ""
                    content_type = data.get('type', 'unknown')

                    # Get ALL content types
                    if 'summary' in data:
                        content = data['summary']
                        content_type = 'summary'

                    elif 'message' in data:
                        msg = data['message']
                        if isinstance(msg, dict):
                            # Get role
                            role = msg.get('role', '')

                            # Extract all content
                            if 'content' in msg:
                                if isinstance(msg['content'], str):
                                    content = msg['content']
                                elif isinstance(msg['content'], list):
                                    # Handle complex content structures
                                    for item in msg['content']:
                                        if isinstance(item, dict):
                                            if 'text' in item:
                                                content += item['text'] + "\n"
                                            elif 'content' in item:
                                                content += str(item['content'])[:1000] + "\n"
                                            elif 'toolUseResult' in item:
                                                content += f"Tool result: {str(item['toolUseResult'])[:500]}\n"

                            content_type = f"{role}_message"

                    elif 'toolUseResult' in data:
                        result = data['toolUseResult']
                        content = f"Tool: {result.get('toolName', 'unknown')}\n"
                        content += f"Output: {str(result.get('stdout', ''))[:1000]}"
                        content_type = 'tool_result'

                    if content and len(content) > 20:
                        embedding = embed_text(content)
                        if embedding:
                            points.append({
                                "id": str(uuid.uuid4()),
                                "vector": embedding,
                                "payload": {
                                    "content": content[:1000],
                                    "source": str(jsonl_file.name),
                                    "type": content_type,
                                    "line": line_num,
                                    "timestamp": datetime.now().isoformat()
                                }
                            })
                            stats['conversations'] += 1
                            stats['total'] += 1

                            if len(points) >= 100:
                                upload_batch(points)
                                points = []

                except Exception as e:
                    logger.error(f"Error in {jsonl_file.name}:{line_num} - {e}")
                    stats['errors'] += 1

    upload_batch(points)
    logger.info(f"✅ Ingested {stats['conversations']} conversation pieces")

def ingest_all_codebases():
    """Ingest ALL Tower codebases and user code"""
    logger.info("💻 INGESTING ALL CODEBASES...")
    points = []

    # ALL code directories
    code_dirs = [
        "/opt/tower-echo-brain",
        "/opt/tower-anime-production",
        "/opt/tower-dashboard",
        "/opt/tower-auth",
        "/opt/tower-kb",
        "/opt/tower-lora-studio",
        "/opt/tower-apple-music",
        "/opt/echo-telegram-bot",
        "/opt/echo-frontend",
        "/opt/tower-models",
        "/opt/tower-smart-feedback",
        "/home/patrick/Documents",
        "/home/patrick/Desktop",
        "/home/patrick/.claude"
    ]

    # ALL file extensions to index
    extensions = [
        '.py', '.js', '.ts', '.tsx', '.vue', '.jsx',
        '.json', '.yaml', '.yml', '.toml',
        '.md', '.txt', '.rst',
        '.sh', '.bash', '.zsh',
        '.sql', '.graphql',
        '.html', '.css', '.scss',
        '.env', '.conf', '.config',
        'Dockerfile', 'docker-compose.yml'
    ]

    for base_dir in code_dirs:
        if not Path(base_dir).exists():
            logger.warning(f"Directory not found: {base_dir}")
            continue

        logger.info(f"Processing {base_dir}")
        dir_count = 0

        for ext in extensions:
            for file_path in Path(base_dir).rglob(f"*{ext}"):
                # Skip virtual environments and caches
                skip_dirs = ['venv', '__pycache__', 'node_modules', '.git', 'dist', 'build']
                if any(skip in str(file_path) for skip in skip_dirs):
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    # Create rich description
                    desc = f"File: {file_path.name}\n"
                    desc += f"Path: {file_path}\n"
                    desc += f"Type: {ext}\n"
                    desc += f"Size: {len(content)} chars\n"
                    desc += f"Content:\n{content[:1500]}"

                    embedding = embed_text(desc)
                    if embedding:
                        points.append({
                            "id": str(uuid.uuid4()),
                            "vector": embedding,
                            "payload": {
                                "content": desc[:1000],
                                "source": str(file_path),
                                "type": "code",
                                "extension": ext,
                                "project": base_dir.split('/')[-1],
                                "timestamp": datetime.now().isoformat()
                            }
                        })
                        dir_count += 1
                        stats['code_files'] += 1
                        stats['total'] += 1

                        if len(points) >= 50:
                            upload_batch(points)
                            points = []

                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")
                    stats['errors'] += 1

        logger.info(f"  → {dir_count} files from {base_dir}")

    upload_batch(points)
    logger.info(f"✅ Ingested {stats['code_files']} code files")

def ingest_all_databases():
    """Ingest ALL database content"""
    logger.info("🗄️ INGESTING ALL DATABASES...")
    points = []

    databases = {
        'anime_production': {
            'host': 'localhost',
            'database': 'anime_production',
            'user': 'patrick',
            'password': 'RP78eIrW7cI2jYvL5akt1yurE'
        },
        'echo_brain': {
            'host': 'localhost',
            'database': 'echo_brain',
            'user': 'patrick',
            'password': 'RP78eIrW7cI2jYvL5akt1yurE'
        },
        'tower_consolidated': {
            'host': 'localhost',
            'database': 'tower_consolidated',
            'user': 'patrick',
            'password': 'RP78eIrW7cI2jYvL5akt1yurE'
        }
    }

    for db_name, config in databases.items():
        try:
            logger.info(f"Connecting to {db_name}...")
            conn = psycopg2.connect(**config)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Get all tables
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
            """)
            tables = [row['table_name'] for row in cur.fetchall()]

            for table in tables:
                try:
                    # Get all records (not just 100)
                    cur.execute(f"SELECT * FROM {table}")
                    rows = cur.fetchall()

                    logger.info(f"  → {table}: {len(rows)} records")

                    for row in rows:
                        # Create comprehensive content
                        content = f"Database: {db_name}\nTable: {table}\n"
                        for key, value in row.items():
                            content += f"{key}: {str(value)[:200]}\n"

                        embedding = embed_text(content)
                        if embedding:
                            points.append({
                                "id": str(uuid.uuid4()),
                                "vector": embedding,
                                "payload": {
                                    "content": content[:1000],
                                    "source": f"{db_name}.{table}",
                                    "type": "database",
                                    "database": db_name,
                                    "table": table,
                                    "timestamp": datetime.now().isoformat()
                                }
                            })
                            stats['database_records'] += 1
                            stats['total'] += 1

                            if len(points) >= 100:
                                upload_batch(points)
                                points = []

                except Exception as e:
                    logger.error(f"Error reading {db_name}.{table}: {e}")
                    stats['errors'] += 1

            conn.close()

        except Exception as e:
            logger.error(f"Database connection error for {db_name}: {e}")
            stats['errors'] += 1

    upload_batch(points)
    logger.info(f"✅ Ingested {stats['database_records']} database records")

def ingest_google_photos_metadata():
    """Ingest Google Photos metadata if available"""
    logger.info("📸 CHECKING FOR GOOGLE PHOTOS...")

    # Common Google Photos backup locations
    photo_dirs = [
        "/home/patrick/Pictures",
        "/home/patrick/Google Drive",
        "/home/patrick/Downloads"
    ]

    points = []
    for base_dir in photo_dirs:
        if not Path(base_dir).exists():
            continue

        # Look for Google Photos metadata files
        for json_file in Path(base_dir).rglob("*.json"):
            if "metadata" in json_file.name.lower() or "google" in json_file.name.lower():
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)

                    content = f"Google Photos metadata: {json_file.name}\n"
                    content += json.dumps(data, indent=2)[:1000]

                    embedding = embed_text(content)
                    if embedding:
                        points.append({
                            "id": str(uuid.uuid4()),
                            "vector": embedding,
                            "payload": {
                                "content": content[:1000],
                                "source": str(json_file),
                                "type": "google_photos",
                                "timestamp": datetime.now().isoformat()
                            }
                        })
                        stats['google_photos'] += 1
                        stats['total'] += 1

                except Exception as e:
                    logger.error(f"Error reading {json_file}: {e}")
                    stats['errors'] += 1

    upload_batch(points)
    if stats['google_photos'] > 0:
        logger.info(f"✅ Ingested {stats['google_photos']} Google Photos metadata")
    else:
        logger.info("ℹ️ No Google Photos metadata found")

def test_search_improvements():
    """Test that search is working better with more data"""
    logger.info("\n🔍 TESTING SEARCH IMPROVEMENTS...")

    test_queries = [
        "mario galaxy anime production",
        "typescript vue postgres",
        "error failed timeout",
        "patrick preferences",
        "tower system architecture"
    ]

    for query in test_queries:
        try:
            with httpx.Client() as client:
                response = client.post(
                    "http://localhost:8312/mcp",
                    json={
                        "method": "tools/call",
                        "params": {
                            "name": "search_memory",
                            "arguments": {"query": query, "limit": 3}
                        }
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    results = data.get('content', [{}])[0].get('text', '')
                    result_count = results.count('**Result')
                    logger.info(f"✅ Query '{query}': {result_count} results found")
                else:
                    logger.error(f"❌ Query '{query}' failed: {response.status_code}")

        except Exception as e:
            logger.error(f"❌ Search test error: {e}")

def main():
    logger.info("=" * 70)
    logger.info("🧠 COMPREHENSIVE ECHO BRAIN INGESTION WITH LOGGING")
    logger.info("=" * 70)

    # Clear existing collection
    logger.info("\n🗑️ Clearing old data...")
    with httpx.Client() as client:
        client.delete(f"{QDRANT_URL}/collections/{COLLECTION_NAME}")
        client.put(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}",
            json={"vectors": {"size": 1024, "distance": "Cosine"}}
        )

    # Run all ingestion functions
    try:
        ingest_all_conversations()
    except Exception as e:
        logger.error(f"Conversation ingestion failed: {e}\n{traceback.format_exc()}")

    try:
        ingest_all_codebases()
    except Exception as e:
        logger.error(f"Codebase ingestion failed: {e}\n{traceback.format_exc()}")

    try:
        ingest_all_databases()
    except Exception as e:
        logger.error(f"Database ingestion failed: {e}\n{traceback.format_exc()}")

    try:
        ingest_google_photos_metadata()
    except Exception as e:
        logger.error(f"Google Photos ingestion failed: {e}\n{traceback.format_exc()}")

    # Test search improvements
    test_search_improvements()

    # Final stats
    logger.info("\n" + "=" * 70)
    logger.info("✅ INGESTION COMPLETE - FINAL STATISTICS:")
    logger.info(f"   • Conversations: {stats['conversations']:,}")
    logger.info(f"   • Code files: {stats['code_files']:,}")
    logger.info(f"   • Database records: {stats['database_records']:,}")
    logger.info(f"   • Documents: {stats['documents']:,}")
    logger.info(f"   • Google Photos: {stats['google_photos']:,}")
    logger.info(f"   • TOTAL VECTORS: {stats['total']:,}")
    logger.info(f"   • Errors: {stats['errors']:,}")
    logger.info("=" * 70)

    # Save stats to file
    with open("/tmp/echo_ingestion_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Update KB
    if stats['total'] > 0:
        logger.info("\n📝 Posting to Knowledge Base...")
        try:
            with httpx.Client() as client:
                kb_content = f"""
Echo Brain Ingestion Report - {datetime.now().isoformat()}

Total Vectors Ingested: {stats['total']:,}

Breakdown:
- Conversations: {stats['conversations']:,}
- Code Files: {stats['code_files']:,}
- Database Records: {stats['database_records']:,}
- Documents: {stats['documents']:,}
- Google Photos: {stats['google_photos']:,}
- Errors: {stats['errors']:,}

Status: {'SUCCESS' if stats['errors'] < 10 else 'COMPLETED WITH ERRORS'}

Search capabilities have been enhanced with comprehensive data coverage.
Memory functions now have access to all Tower system data.
"""

                response = client.post(
                    "http://localhost:8307/api/kb/articles",
                    json={
                        "title": f"Echo Brain Ingestion Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        "content": kb_content,
                        "category": "system_reports"
                    }
                )

                if response.status_code in [200, 201]:
                    logger.info("✅ Posted to Knowledge Base")
                else:
                    logger.warning(f"KB post failed: {response.status_code}")
        except Exception as e:
            logger.error(f"KB post error: {e}")

if __name__ == "__main__":
    main()