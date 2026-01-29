#!/usr/bin/env python3
"""
COMPREHENSIVE Echo Brain Ingestion
Ingests ALL data sources, not just conversation summaries
"""

import json
import os
from pathlib import Path
from datetime import datetime
import uuid
import httpx
import psycopg2
from psycopg2.extras import RealDictCursor

# Ollama for embeddings
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "mxbai-embed-large"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "echo_memory"

# Database config
DB_CONFIG = {
    'host': 'localhost',
    'database': 'anime_production',
    'user': 'patrick',
    'password': 'RP78eIrW7cI2jYvL5akt1yurE'
}

def embed_text(text):
    """Create embedding using Ollama"""
    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": text[:2000]}  # Limit to 2000 chars
            )
            if response.status_code == 200:
                return response.json()["embedding"]
    except:
        return None

def ingest_full_conversations():
    """Ingest FULL conversation content, not just summaries"""
    print("\n📚 INGESTING FULL CONVERSATIONS...")

    base_dir = Path("/home/patrick/.claude/projects/")
    points = []

    for jsonl_file in base_dir.rglob("*.jsonl"):
        with open(jsonl_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())

                    # Get ALL content, not just summaries
                    content = ""

                    # Extract summary
                    if data.get('type') == 'summary':
                        content = data.get('summary', '')
                        content_type = "summary"

                    # Extract actual messages
                    elif data.get('type') == 'user':
                        message = data.get('message', {})
                        if isinstance(message, dict):
                            role = message.get('role', '')
                            msg_content = message.get('content', '')
                            if isinstance(msg_content, list):
                                for item in msg_content:
                                    if isinstance(item, dict):
                                        if item.get('type') == 'text':
                                            content += item.get('text', '')
                                        elif item.get('type') == 'tool_result':
                                            content += str(item.get('content', ''))[:500]
                            elif isinstance(msg_content, str):
                                content = msg_content
                        content_type = "user_message"

                    # Extract assistant responses
                    elif data.get('type') == 'assistant':
                        message = data.get('message', {})
                        if isinstance(message, dict):
                            msg_content = message.get('content', '')
                            if isinstance(msg_content, list):
                                for item in msg_content:
                                    if isinstance(item, dict) and item.get('type') == 'text':
                                        content += item.get('text', '')
                            elif isinstance(msg_content, str):
                                content = msg_content
                        content_type = "assistant_response"

                    if content and len(content) > 10:  # Only meaningful content
                        embedding = embed_text(content)
                        if embedding:
                            points.append({
                                "id": str(uuid.uuid4()),
                                "vector": embedding,
                                "payload": {
                                    "content": content[:1000],  # Store first 1000 chars
                                    "source": str(jsonl_file),
                                    "type": content_type,
                                    "line_number": line_num,
                                    "timestamp": datetime.now().isoformat()
                                }
                            })

                except Exception as e:
                    continue

    print(f"   → Extracted {len(points)} conversation pieces")
    return points

def ingest_database_content():
    """Ingest anime production database content"""
    print("\n🗄️ INGESTING DATABASE CONTENT...")
    points = []

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Ingest characters
    cur.execute("SELECT * FROM characters")
    characters = cur.fetchall()
    for char in characters:
        content = f"Character: {char['name']}. {char.get('design_prompt', '')} {char.get('personality', '')}"
        embedding = embed_text(content)
        if embedding:
            points.append({
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "payload": {
                    "content": content,
                    "source": "anime_production.characters",
                    "type": "character",
                    "character_id": char['id'],
                    "timestamp": datetime.now().isoformat()
                }
            })

    print(f"   → Ingested {len(characters)} characters")

    # Ingest projects
    cur.execute("SELECT * FROM projects")
    projects = cur.fetchall()
    for proj in projects:
        content = f"Project: {proj['name']}. {proj.get('description', '')} Status: {proj.get('status', '')}"
        embedding = embed_text(content)
        if embedding:
            points.append({
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "payload": {
                    "content": content,
                    "source": "anime_production.projects",
                    "type": "project",
                    "project_id": proj['id'],
                    "timestamp": datetime.now().isoformat()
                }
            })

    print(f"   → Ingested {len(projects)} projects")

    # Ingest episodes
    cur.execute("SELECT * FROM episodes")
    episodes = cur.fetchall()
    for ep in episodes:
        content = f"Episode: {ep['title']}. {ep.get('description', '')}"
        embedding = embed_text(content)
        if embedding:
            points.append({
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "payload": {
                    "content": content,
                    "source": "anime_production.episodes",
                    "type": "episode",
                    "episode_id": str(ep['id']),
                    "timestamp": datetime.now().isoformat()
                }
            })

    print(f"   → Ingested {len(episodes)} episodes")

    conn.close()
    return points

def ingest_markdown_docs():
    """Ingest markdown documentation"""
    print("\n📝 INGESTING MARKDOWN DOCUMENTATION...")
    points = []

    # Key documentation directories
    doc_dirs = [
        "/home/patrick/.claude/",
        "/home/patrick/Documents/",
        "/opt/tower-echo-brain/docs/",
        "/opt/tower-anime-production/docs/"
    ]

    count = 0
    for base_dir in doc_dirs:
        base_path = Path(base_dir)
        if base_path.exists():
            for md_file in base_path.rglob("*.md"):
                try:
                    with open(md_file, 'r') as f:
                        content = f.read()[:2000]  # First 2000 chars

                    if content:
                        embedding = embed_text(content)
                        if embedding:
                            points.append({
                                "id": str(uuid.uuid4()),
                                "vector": embedding,
                                "payload": {
                                    "content": content[:1000],
                                    "source": str(md_file),
                                    "type": "documentation",
                                    "filename": md_file.name,
                                    "timestamp": datetime.now().isoformat()
                                }
                            })
                            count += 1
                except:
                    continue

    print(f"   → Ingested {count} markdown files")
    return points

def upload_to_qdrant(points, batch_size=100):
    """Upload points to Qdrant"""
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        try:
            with httpx.Client(timeout=30) as client:
                response = client.put(
                    f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points",
                    json={"points": batch}
                )
                if response.status_code == 200:
                    print(f"   ✅ Uploaded batch {i//batch_size + 1} ({len(batch)} points)")
        except Exception as e:
            print(f"   ❌ Error: {e}")

def main():
    print("=" * 60)
    print("🧠 COMPREHENSIVE ECHO BRAIN INGESTION")
    print("=" * 60)

    all_points = []

    # 1. Full conversations
    conv_points = ingest_full_conversations()
    all_points.extend(conv_points)

    # 2. Database content
    db_points = ingest_database_content()
    all_points.extend(db_points)

    # 3. Documentation
    doc_points = ingest_markdown_docs()
    all_points.extend(doc_points)

    print(f"\n📊 TOTAL POINTS TO UPLOAD: {len(all_points)}")

    if all_points:
        print("\n🚀 UPLOADING TO QDRANT...")
        upload_to_qdrant(all_points)

        print(f"\n✅ COMPREHENSIVE INGESTION COMPLETE!")
        print(f"   → {len(all_points)} total data points indexed")
        print(f"   → Conversations, database, documentation all included")

if __name__ == "__main__":
    main()