#!/usr/bin/env python3
"""
Real conversation ingestion for Echo Brain
Processes actual Claude conversation files and stores them in Qdrant
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime
import uuid
import httpx
from sentence_transformers import SentenceTransformer

# Use Ollama for embeddings to match MCP server
print("🔄 Using Ollama for embeddings...")
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "mxbai-embed-large"
print("✅ Ollama embedding setup complete")

def embed_text(text):
    """Create embedding using Ollama API to match MCP server"""
    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": text}
            )
            if response.status_code == 200:
                return response.json()["embedding"]
            else:
                print(f"❌ Ollama embedding error: {response.text}")
                return None
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return None

def ingest_conversation_file(file_path):
    """Process a single conversation file"""
    points = []

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())

                if data.get('type') == 'summary':
                    summary = data.get('summary', '')
                    leaf_uuid = data.get('leafUuid', '')

                    if summary:
                        # Create vector point
                        point_id = str(uuid.uuid4())
                        embedding = embed_text(summary)

                        if embedding:  # Only add if embedding succeeded
                            point = {
                                "id": point_id,
                                "vector": embedding,
                                "payload": {
                                    "content": summary,
                                    "source": str(file_path),
                                    "type": "conversation_summary",
                                    "leaf_uuid": leaf_uuid,
                                    "timestamp": datetime.now().isoformat(),
                                    "line_number": line_num
                                }
                            }
                            points.append(point)

            except json.JSONDecodeError as e:
                print(f"⚠️  JSON error in {file_path}:{line_num}: {e}")
                continue

    return points

def upload_to_qdrant(points, batch_size=100):
    """Upload points to Qdrant vector database"""
    qdrant_url = "http://localhost:6333"
    collection_name = "echo_memory"

    # Upload in batches
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]

        try:
            with httpx.Client(timeout=30) as client:
                response = client.put(
                    f"{qdrant_url}/collections/{collection_name}/points",
                    json={"points": batch}
                )

                if response.status_code == 200:
                    print(f"✅ Uploaded batch {i//batch_size + 1} ({len(batch)} points)")
                else:
                    print(f"❌ Upload failed for batch {i//batch_size + 1}: {response.text}")

        except Exception as e:
            print(f"❌ Error uploading batch: {e}")

def main():
    """Main ingestion process"""
    print("🧠 Real Echo Brain Conversation Ingestion")
    print("=" * 50)

    # Find ALL conversation files across all projects
    base_projects_dir = Path("/home/patrick/.claude/projects/")

    if not base_projects_dir.exists():
        print(f"❌ Projects directory not found: {base_projects_dir}")
        return

    # Discover all conversation files recursively
    conversation_files = list(base_projects_dir.rglob("*.jsonl"))

    # Group by project directory for reporting
    projects = {}
    for file_path in conversation_files:
        project_dir = file_path.parent.name
        if project_dir not in projects:
            projects[project_dir] = []
        projects[project_dir].append(file_path)

    print(f"📁 Found {len(conversation_files)} conversation files across {len(projects)} projects:")
    for project, files in projects.items():
        print(f"   → {project}: {len(files)} files")

    if not conversation_files:
        print("❌ No conversation files found")
        return

    # Process all files
    all_points = []

    for i, file_path in enumerate(conversation_files, 1):
        print(f"📖 Processing {file_path.name} ({i}/{len(conversation_files)})")
        points = ingest_conversation_file(file_path)
        all_points.extend(points)
        print(f"   → Found {len(points)} conversation summaries")

    print(f"\n📊 Total points to upload: {len(all_points)}")

    if all_points:
        print("🚀 Uploading to Qdrant...")
        upload_to_qdrant(all_points)

        # Log completion
        log_file = "/opt/tower-echo-brain/logs/real_ingestion.log"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        with open(log_file, "w") as f:
            f.write(f"Real ingestion completed at {datetime.now()}\n")
            f.write(f"Processed {len(conversation_files)} files\n")
            f.write(f"Uploaded {len(all_points)} conversation summaries\n")

        print(f"✅ Ingestion completed successfully!")
        print(f"   → {len(all_points)} conversation summaries indexed")
    else:
        print("⚠️  No data to upload")

if __name__ == "__main__":
    main()