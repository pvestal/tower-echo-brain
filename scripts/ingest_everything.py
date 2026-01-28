#!/usr/bin/env python3
"""
INGEST EVERYTHING INTO ECHO BRAIN
No restrictions - Echo Brain needs access to EVERYTHING
"""

import json
import os
import ast
from pathlib import Path
from datetime import datetime
import uuid
import httpx
import psycopg2
from psycopg2.extras import RealDictCursor
import asyncio
import re

# Configuration
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "mxbai-embed-large"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "echo_memory"

# ALL Tower codebases - NO RESTRICTIONS
TOWER_CODEBASES = [
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

# Database configs
DBS = {
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

def embed_text(text):
    """Create embedding using Ollama"""
    if not text:
        return None
    try:
        with httpx.Client(timeout=30) as client:
            # Truncate to reasonable length
            text_to_embed = text[:2000] if len(text) > 2000 else text
            response = client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": text_to_embed}
            )
            if response.status_code == 200:
                return response.json()["embedding"]
    except:
        return None

def ingest_tower_codebases():
    """Ingest ALL Tower codebase files - Python, JS, TS, Vue, configs, EVERYTHING"""
    print("\n🏗️ INGESTING ALL TOWER CODEBASES...")
    points = []

    file_extensions = [
        '.py', '.js', '.ts', '.tsx', '.vue', '.jsx',
        '.json', '.yaml', '.yml', '.toml',
        '.md', '.txt', '.rst',
        '.sh', '.bash', '.zsh',
        '.sql', '.graphql',
        '.html', '.css', '.scss',
        '.env', '.conf', '.config',
        'Dockerfile', 'docker-compose.yml',
        'requirements.txt', 'package.json'
    ]

    total_files = 0
    for base_dir in TOWER_CODEBASES:
        if not Path(base_dir).exists():
            continue

        print(f"   📂 Processing {base_dir}...")
        dir_files = 0

        for ext in file_extensions:
            for file_path in Path(base_dir).rglob(f"*{ext}"):
                # Skip node_modules and venv
                if 'node_modules' in str(file_path) or 'venv' in str(file_path) or '__pycache__' in str(file_path):
                    continue

                try:
                    # Read file content
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    # Create descriptive text for embedding
                    desc = f"File: {file_path.name}\nPath: {file_path}\n"

                    # Extract key info based on file type
                    if file_path.suffix == '.py':
                        # Extract function/class definitions
                        functions = re.findall(r'def\s+(\w+)\s*\(', content)
                        classes = re.findall(r'class\s+(\w+)\s*[:\(]', content)
                        desc += f"Functions: {', '.join(functions[:10])}\n" if functions else ""
                        desc += f"Classes: {', '.join(classes[:10])}\n" if classes else ""

                    elif file_path.suffix in ['.js', '.ts', '.tsx']:
                        # Extract exports and functions
                        exports = re.findall(r'export\s+(?:default\s+)?(?:function|const|class)\s+(\w+)', content)
                        desc += f"Exports: {', '.join(exports[:10])}\n" if exports else ""

                    elif file_path.suffix == '.vue':
                        # Extract component name
                        comp_name = re.search(r'name:\s*[\'"](\w+)[\'"]', content)
                        if comp_name:
                            desc += f"Component: {comp_name.group(1)}\n"

                    # Add first 1500 chars of content
                    desc += f"\nContent:\n{content[:1500]}"

                    embedding = embed_text(desc)
                    if embedding:
                        points.append({
                            "id": str(uuid.uuid4()),
                            "vector": embedding,
                            "payload": {
                                "content": desc[:1000],
                                "source": str(file_path),
                                "type": "source_code",
                                "file_type": file_path.suffix,
                                "project": base_dir.split('/')[-1],
                                "timestamp": datetime.now().isoformat()
                            }
                        })
                        dir_files += 1
                        total_files += 1

                except Exception as e:
                    continue

        print(f"      → {dir_files} files indexed")

    print(f"   📊 Total codebase files: {total_files}")
    return points

def ingest_all_databases():
    """Ingest ALL database content"""
    print("\n🗄️ INGESTING ALL DATABASES...")
    points = []

    for db_name, config in DBS.items():
        print(f"   📀 Database: {db_name}")
        try:
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
                    cur.execute(f"SELECT * FROM {table} LIMIT 1000")
                    rows = cur.fetchall()

                    for row in rows:
                        # Create searchable content from row
                        content = f"Database: {db_name}, Table: {table}\n"
                        content += json.dumps(dict(row), default=str)[:1000]

                        embedding = embed_text(content)
                        if embedding:
                            points.append({
                                "id": str(uuid.uuid4()),
                                "vector": embedding,
                                "payload": {
                                    "content": content[:1000],
                                    "source": f"{db_name}.{table}",
                                    "type": "database_record",
                                    "database": db_name,
                                    "table": table,
                                    "timestamp": datetime.now().isoformat()
                                }
                            })

                    print(f"      → {table}: {len(rows)} records")

                except Exception as e:
                    continue

            conn.close()

        except Exception as e:
            print(f"      ❌ Error: {e}")

    return points

def ingest_full_conversations():
    """Ingest COMPLETE conversation content"""
    print("\n💬 INGESTING FULL CONVERSATIONS...")
    points = []

    base_dir = Path("/home/patrick/.claude/projects/")
    total_messages = 0

    for jsonl_file in base_dir.rglob("*.jsonl"):
        with open(jsonl_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())

                    # Extract ALL content types
                    content = ""
                    content_type = data.get('type', 'unknown')

                    # Get message content
                    if 'message' in data:
                        msg = data['message']
                        if isinstance(msg, dict):
                            # Extract text from various formats
                            if 'content' in msg:
                                if isinstance(msg['content'], str):
                                    content = msg['content']
                                elif isinstance(msg['content'], list):
                                    for item in msg['content']:
                                        if isinstance(item, dict):
                                            if 'text' in item:
                                                content += item['text'] + "\n"
                                            elif 'content' in item:
                                                content += str(item['content'])[:500] + "\n"

                    # Get summaries
                    if 'summary' in data:
                        content = data['summary']
                        content_type = 'summary'

                    # Get tool results
                    if 'toolUseResult' in data:
                        result = data['toolUseResult']
                        if isinstance(result, dict):
                            content = str(result.get('stdout', ''))[:500]
                            content_type = 'tool_result'

                    if content and len(content) > 20:
                        embedding = embed_text(content)
                        if embedding:
                            points.append({
                                "id": str(uuid.uuid4()),
                                "vector": embedding,
                                "payload": {
                                    "content": content[:1000],
                                    "source": str(jsonl_file),
                                    "type": f"conversation_{content_type}",
                                    "timestamp": data.get('timestamp', datetime.now().isoformat())
                                }
                            })
                            total_messages += 1

                except:
                    continue

    print(f"   📊 Total conversation pieces: {total_messages}")
    return points

def upload_to_qdrant(points, batch_size=100):
    """Upload points to Qdrant"""
    print(f"\n🚀 UPLOADING {len(points)} POINTS TO QDRANT...")

    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        try:
            with httpx.Client(timeout=60) as client:
                response = client.put(
                    f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points",
                    json={"points": batch}
                )
                if response.status_code == 200:
                    print(f"   ✅ Batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1} uploaded")
        except Exception as e:
            print(f"   ❌ Batch {i//batch_size + 1} error: {e}")

def main():
    print("=" * 70)
    print("🧠 ECHO BRAIN: INGESTING EVERYTHING")
    print("=" * 70)

    # Clear existing collection
    print("\n🗑️ Clearing old data...")
    with httpx.Client() as client:
        client.delete(f"{QDRANT_URL}/collections/{COLLECTION_NAME}")
        client.put(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}",
            json={"vectors": {"size": 1024, "distance": "Cosine", "on_disk": True}}
        )

    all_points = []

    # 1. Tower codebases
    code_points = ingest_tower_codebases()
    all_points.extend(code_points)

    # 2. All databases
    db_points = ingest_all_databases()
    all_points.extend(db_points)

    # 3. Full conversations
    conv_points = ingest_full_conversations()
    all_points.extend(conv_points)

    print(f"\n📊 TOTAL DATA POINTS: {len(all_points)}")

    # Upload everything
    if all_points:
        upload_to_qdrant(all_points)

    print("\n" + "=" * 70)
    print("✅ ECHO BRAIN NOW HAS ACCESS TO EVERYTHING:")
    print(f"   • {len(code_points)} source code files")
    print(f"   • {len(db_points)} database records")
    print(f"   • {len(conv_points)} conversation pieces")
    print(f"   • TOTAL: {len(all_points)} searchable data points")
    print("=" * 70)

    # Store result in KB
    result = {
        "timestamp": datetime.now().isoformat(),
        "total_points": len(all_points),
        "code_files": len(code_points),
        "database_records": len(db_points),
        "conversation_pieces": len(conv_points),
        "status": "complete"
    }

    with open("/tmp/echo_ingestion_result.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()