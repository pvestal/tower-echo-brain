#!/usr/bin/env python3
"""
Ingest ECHO_BRAIN_ARCHITECTURE.md into Qdrant echo_memory collection
"""
import hashlib
import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
import re

def generate_embedding(text, ollama_url="http://localhost:11434"):
    """Generate embedding using Ollama nomic-embed-text model"""
    req = urllib.request.Request(
        f"{ollama_url}/api/embed",
        data=json.dumps({
            "model": "nomic-embed-text",
            "input": text
        }).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )

    try:
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.load(resp)
        embedding = data.get('embeddings', [[]])[0]
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def check_duplicate(content_hash, qdrant_url="http://localhost:6333"):
    """Check if content hash already exists in Qdrant"""
    search_req = urllib.request.Request(
        f"{qdrant_url}/collections/echo_memory/points/scroll",
        data=json.dumps({
            "filter": {
                "must": [
                    {"key": "content_hash", "match": {"value": content_hash}}
                ]
            },
            "limit": 1
        }).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )

    try:
        resp = urllib.request.urlopen(search_req)
        data = json.load(resp)
        return len(data.get('result', {}).get('points', [])) > 0
    except:
        return False

def upsert_to_qdrant(vector, payload, qdrant_url="http://localhost:6333"):
    """Upsert vector to Qdrant collection"""
    import uuid

    point_id = str(uuid.uuid4())

    # Use PUT endpoint with correct structure
    req = urllib.request.Request(
        f"{qdrant_url}/collections/echo_memory/points",
        data=json.dumps({
            "points": [{
                "id": point_id,
                "vector": vector,
                "payload": payload
            }]
        }).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='PUT'  # Explicitly set method to PUT
    )

    try:
        resp = urllib.request.urlopen(req)
        return json.load(resp)
    except Exception as e:
        print(f"Error upserting to Qdrant: {e}")
        return None

def extract_section_title(content):
    """Extract first # header from content as section title"""
    lines = content.strip().split('\n')
    for line in lines:
        if line.startswith('#'):
            # Remove # symbols and strip
            title = line.lstrip('#').strip()
            return title
    return "Untitled Section"

def main():
    print("=" * 70)
    print("ECHO BRAIN ARCHITECTURE DOCUMENT INGESTION")
    print("=" * 70)

    start_time = time.time()

    # Read the architecture document
    doc_path = "/opt/tower-echo-brain/docs/ECHO_BRAIN_ARCHITECTURE.md"
    print(f"\nReading document: {doc_path}")

    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading document: {e}")
        return

    print(f"Document size: {len(content):,} characters")

    # Split by --- separators
    sections = content.split('\n---\n')
    print(f"Found {len(sections)} sections")

    # Process each section
    new_vectors = 0
    duplicates_skipped = 0

    for i, section in enumerate(sections, 1):
        section = section.strip()
        if not section:
            continue

        print(f"\n{'─' * 50}")
        print(f"Section {i}/{len(sections)}")

        # Extract section title
        title = extract_section_title(section)
        print(f"Title: {title}")

        # Calculate stats
        char_count = len(section)
        token_estimate = char_count // 4
        print(f"Size: {char_count:,} chars (~{token_estimate:,} tokens)")

        # Generate content hash
        content_hash = hashlib.sha256(section.encode('utf-8')).hexdigest()
        print(f"Hash: {content_hash[:16]}...")

        # Check for duplicate
        if check_duplicate(content_hash):
            print("Status: SKIPPED (duplicate)")
            duplicates_skipped += 1
            continue

        # Generate embedding
        print("Generating embedding...", end=" ")
        embedding = generate_embedding(section)

        if not embedding:
            print("FAILED")
            continue

        # Verify dimensions
        dims = len(embedding)
        if dims != 768:
            print(f"ERROR: Wrong dimensions ({dims} != 768)")
            continue

        print(f"OK ({dims}D)")

        # Create payload
        payload = {
            "source": "architecture_doc",
            "source_path": "docs/ECHO_BRAIN_ARCHITECTURE.md",
            "section_title": title,
            "content": section,
            "content_hash": content_hash,
            "content_type": "documentation",
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "priority": 100,  # Architecture docs have high priority
            "authoritative": True,  # This overrides conflicting sources
            "version": "2026-02-10"
        }

        # Upsert to Qdrant
        print("Inserting into Qdrant...", end=" ")
        result = upsert_to_qdrant(embedding, payload)

        if result:
            print("SUCCESS")
            new_vectors += 1
        else:
            print("FAILED")

    # Print summary
    elapsed = time.time() - start_time

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total sections processed: {len(sections)}")
    print(f"New vectors inserted: {new_vectors}")
    print(f"Duplicates skipped: {duplicates_skipped}")
    print(f"Total time elapsed: {elapsed:.2f} seconds")
    print("=" * 70)

if __name__ == "__main__":
    main()