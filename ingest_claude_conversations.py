#!/usr/bin/env python3
"""
Ingest NEW Claude Code conversations into Echo Brain
Skips already-ingested files to prevent duplicates
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
import requests
import sys

# Configuration
OLLAMA_URL = "http://localhost:11434/api/embeddings"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "echo_memory"
EMBEDDING_MODEL = "mxbai-embed-large:latest"  # 1024 dimensions

def get_existing_files():
    """Get list of already-ingested file names from Qdrant"""
    existing_files = set()
    offset = None

    while True:
        # Scroll through all claude_code entries
        payload = {
            "limit": 100,
            "filter": {"must": [{"key": "source", "match": {"value": "claude_code"}}]},
            "with_payload": ["file_name"]
        }
        if offset:
            payload["offset"] = offset

        try:
            response = requests.post(
                f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/scroll",
                json=payload,
                timeout=10
            )

            if response.status_code != 200:
                print(f"⚠️ Failed to query existing files: {response.status_code}")
                return existing_files

            data = response.json()
            points = data.get('result', {}).get('points', [])

            if not points:
                break

            for point in points:
                file_name = point.get('payload', {}).get('file_name')
                if file_name:
                    existing_files.add(file_name)

            # Check if there are more results
            next_page_offset = data.get('result', {}).get('next_page_offset')
            if not next_page_offset:
                break
            offset = next_page_offset

        except Exception as e:
            print(f"⚠️ Error querying existing files: {e}")
            break

    return existing_files

def extract_content(file_path):
    """Extract meaningful content from conversation file"""
    content_parts = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())

                    # Handle summaries
                    if data.get('type') == 'summary' and 'summary' in data:
                        summary = data['summary']
                        if isinstance(summary, str) and len(summary) > 50:
                            content_parts.append(summary[:1000])
                            if len(content_parts) >= 3:
                                break

                    # Handle user messages
                    elif data.get('type') == 'user' and data.get('message'):
                        msg = data['message']
                        content = msg.get('content')
                        if isinstance(content, str) and len(content) > 100:
                            if not content.startswith('<'):
                                content_parts.append(content[:500])
                                if len(content_parts) >= 3:
                                    break

                except:
                    continue
    except:
        return None

    if content_parts:
        return " ".join(content_parts)[:2000]
    return None

def ingest_file(file_path):
    """Ingest a single file"""
    content = extract_content(file_path)
    if not content:
        return False

    # Get embedding
    try:
        response = requests.post(OLLAMA_URL, json={
            'model': EMBEDDING_MODEL,
            'prompt': content
        }, timeout=10)

        if response.status_code != 200:
            return False

        embedding = response.json()['embedding']
        if len(embedding) != 1024:
            print(f"  ⚠️ Wrong dimension: {len(embedding)}")
            return False

    except:
        return False

    # Create point with proper UUID
    point = {
        'id': str(uuid.uuid4()),
        'vector': embedding,
        'payload': {
            'content': content[:1500],
            'source': 'claude_code',
            'file_name': file_path.name,
            'project': file_path.parent.name,
            'timestamp': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            'ingested_at': datetime.now().isoformat()
        }
    }

    # Add to Qdrant
    try:
        response = requests.put(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points",
            json={'points': [point]},
            timeout=5
        )
        return response.status_code == 200
    except:
        return False

def main():
    print("CLAUDE CODE CONVERSATION INGESTION (Incremental)")
    print("=" * 50)

    # Get already-ingested files
    print("📋 Checking for already-ingested conversations...")
    existing_files = get_existing_files()
    print(f"   Found {len(existing_files)} already-ingested files")

    # Find all conversation files
    claude_projects = Path.home() / '.claude' / 'projects'
    all_files = list(claude_projects.glob('**/*.jsonl'))
    print(f"📁 Found {len(all_files)} total conversation files")

    # Filter out already-ingested files
    new_files = [f for f in all_files if f.name not in existing_files]
    print(f"🆕 Found {len(new_files)} NEW conversation files to process")

    if not new_files:
        print("\n✅ No new conversations to ingest. Everything is up to date!")
        return

    # Get initial count
    response = requests.get(f"{QDRANT_URL}/collections/{COLLECTION_NAME}")
    initial_count = response.json()['result']['points_count']
    print(f"\n📊 Initial vectors: {initial_count:,}")
    print(f"🔧 Using embedding model: {EMBEDDING_MODEL}")
    print("-" * 50)

    # Process new files only
    successful = 0
    failed = 0

    for i, file_path in enumerate(new_files, 1):
        sys.stdout.write(f"\r[{i}/{len(new_files)}] Processing new files... Success: {successful}, Failed: {failed}")
        sys.stdout.flush()

        if ingest_file(file_path):
            successful += 1
        else:
            failed += 1

    print(f"\n" + "=" * 50)
    print(f"✅ COMPLETE!")
    print(f"  • New files processed: {len(new_files)}")
    print(f"  • Successfully ingested: {successful}")
    print(f"  • Failed: {failed}")
    print(f"  • Skipped (already ingested): {len(all_files) - len(new_files)}")

    # Get final count
    response = requests.get(f"{QDRANT_URL}/collections/{COLLECTION_NAME}")
    final_count = response.json()['result']['points_count']

    print(f"\n📊 RESULTS:")
    print(f"  • Initial vectors: {initial_count:,}")
    print(f"  • Final vectors: {final_count:,}")
    print(f"  • Vectors added: {final_count - initial_count:,}")

    if final_count > initial_count:
        print(f"\n✨ Successfully ingested {successful} new Claude Code conversations!")
        print("🧠 Your conversations are now up to date in Echo Brain.")
    elif successful == 0 and failed == 0:
        print("\n✅ All conversations already ingested. Nothing to do!")
    else:
        print("\n⚠️ Some files failed to ingest. Check the logs for errors.")

    # Log the run
    log_file = Path("/var/log/claude-conversation-ingest.log")
    try:
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()} - Processed {len(new_files)} new, {successful} success, {failed} failed, {len(all_files) - len(new_files)} skipped\n")
    except:
        pass  # Ignore logging errors

if __name__ == "__main__":
    main()