#!/usr/bin/env python3
"""Fast comprehensive ingestion with progress tracking"""

import json
import os
from pathlib import Path
from datetime import datetime
import uuid
import httpx
import psycopg2
from psycopg2.extras import RealDictCursor
import sys

OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "mxbai-embed-large"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "echo_memory"

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
    except:
        return None

def upload_batch(points):
    """Upload points to Qdrant"""
    if not points:
        return
    try:
        with httpx.Client(timeout=60) as client:
            response = client.put(
                f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points",
                json={"points": points}
            )
            if response.status_code == 200:
                print(f"   ✅ Uploaded {len(points)} vectors")
                return True
    except Exception as e:
        print(f"   ❌ Upload error: {e}")
    return False

def ingest_conversations():
    """Ingest conversation files"""
    print("\n📚 INGESTING CONVERSATIONS...")
    points = []
    base_dir = Path("/home/patrick/.claude/projects/")

    count = 0
    for jsonl_file in base_dir.rglob("*.jsonl"):
        with open(jsonl_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    content = ""

                    # Get summaries
                    if 'summary' in data:
                        content = data['summary']

                    # Get messages
                    elif 'message' in data:
                        msg = data['message']
                        if isinstance(msg, dict) and 'content' in msg:
                            if isinstance(msg['content'], str):
                                content = msg['content']
                            elif isinstance(msg['content'], list):
                                for item in msg['content']:
                                    if isinstance(item, dict) and 'text' in item:
                                        content += item['text'] + "\n"

                    if content and len(content) > 20:
                        embedding = embed_text(content)
                        if embedding:
                            points.append({
                                "id": str(uuid.uuid4()),
                                "vector": embedding,
                                "payload": {
                                    "content": content[:1000],
                                    "source": str(jsonl_file.name),
                                    "type": "conversation",
                                    "timestamp": datetime.now().isoformat()
                                }
                            })
                            count += 1

                            # Upload every 100 points
                            if len(points) >= 100:
                                upload_batch(points)
                                points = []
                                print(f"   → Processed {count} conversation pieces")

                except:
                    continue

    # Upload remaining
    upload_batch(points)
    print(f"   📊 Total conversations: {count}")
    return count

def ingest_databases():
    """Ingest database content"""
    print("\n🗄️ INGESTING DATABASES...")
    points = []
    count = 0

    DBS = {
        'anime_production': {
            'host': 'localhost',
            'database': 'anime_production',
            'user': 'patrick',
            'password': 'RP78eIrW7cI2jYvL5akt1yurE'
        }
    }

    for db_name, config in DBS.items():
        try:
            conn = psycopg2.connect(**config)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Ingest projects
            cur.execute("SELECT * FROM projects LIMIT 100")
            for row in cur.fetchall():
                content = f"Project: {row.get('name', '')}, {row.get('description', '')}"
                embedding = embed_text(content)
                if embedding:
                    points.append({
                        "id": str(uuid.uuid4()),
                        "vector": embedding,
                        "payload": {
                            "content": content[:1000],
                            "source": f"{db_name}.projects",
                            "type": "database",
                            "timestamp": datetime.now().isoformat()
                        }
                    })
                    count += 1

            # Ingest characters
            cur.execute("SELECT * FROM characters LIMIT 100")
            for row in cur.fetchall():
                content = f"Character: {row.get('name', '')}, {row.get('design_prompt', '')}"
                embedding = embed_text(content)
                if embedding:
                    points.append({
                        "id": str(uuid.uuid4()),
                        "vector": embedding,
                        "payload": {
                            "content": content[:1000],
                            "source": f"{db_name}.characters",
                            "type": "database",
                            "timestamp": datetime.now().isoformat()
                        }
                    })
                    count += 1

            # Upload batch
            if len(points) >= 50:
                upload_batch(points)
                points = []

            conn.close()
            print(f"   → {db_name}: {count} records")

        except Exception as e:
            print(f"   ❌ {db_name} error: {e}")

    upload_batch(points)
    print(f"   📊 Total database records: {count}")
    return count

def ingest_code_files():
    """Ingest Tower codebase files"""
    print("\n💻 INGESTING TOWER CODEBASES...")
    points = []
    count = 0

    # Process key Tower directories
    dirs = [
        "/opt/tower-echo-brain",
        "/opt/tower-anime-production",
        "/opt/tower-dashboard"
    ]

    for base_dir in dirs:
        if not Path(base_dir).exists():
            continue

        print(f"   📂 {base_dir}")
        dir_count = 0

        # Process Python files
        for py_file in Path(base_dir).rglob("*.py"):
            if 'venv' in str(py_file) or '__pycache__' in str(py_file):
                continue

            try:
                with open(py_file, 'r', errors='ignore') as f:
                    content = f.read()[:1500]

                desc = f"Python file: {py_file.name}\nPath: {py_file}\nContent:\n{content}"
                embedding = embed_text(desc)

                if embedding:
                    points.append({
                        "id": str(uuid.uuid4()),
                        "vector": embedding,
                        "payload": {
                            "content": desc[:1000],
                            "source": str(py_file),
                            "type": "code",
                            "language": "python",
                            "timestamp": datetime.now().isoformat()
                        }
                    })
                    dir_count += 1
                    count += 1

                    if len(points) >= 50:
                        upload_batch(points)
                        points = []

            except:
                continue

        print(f"      → {dir_count} files")

    upload_batch(points)
    print(f"   📊 Total code files: {count}")
    return count

def main():
    print("=" * 60)
    print("🚀 FAST ECHO BRAIN INGESTION")
    print("=" * 60)

    # Clear collection
    print("\n🗑️ Clearing old data...")
    with httpx.Client() as client:
        client.delete(f"{QDRANT_URL}/collections/{COLLECTION_NAME}")
        client.put(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}",
            json={"vectors": {"size": 1024, "distance": "Cosine"}}
        )

    # Run ingestion
    conv_count = ingest_conversations()
    db_count = ingest_databases()
    code_count = ingest_code_files()

    total = conv_count + db_count + code_count

    print("\n" + "=" * 60)
    print("✅ INGESTION COMPLETE!")
    print(f"   • Conversations: {conv_count}")
    print(f"   • Database records: {db_count}")
    print(f"   • Code files: {code_count}")
    print(f"   • TOTAL: {total} vectors")
    print("=" * 60)

    # Save status
    status = {
        "timestamp": datetime.now().isoformat(),
        "conversations": conv_count,
        "database": db_count,
        "code": code_count,
        "total": total,
        "status": "success"
    }

    with open("/tmp/echo_ingestion_status.json", "w") as f:
        json.dump(status, f, indent=2)

if __name__ == "__main__":
    main()