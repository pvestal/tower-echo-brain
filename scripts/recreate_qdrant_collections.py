#!/usr/bin/env python3
"""
Recreate Qdrant collections with OpenAI embedding dimensions (1536D).
This will DELETE existing collections - the old 1024D vectors are orphaned anyway.
"""
import requests
import json

QDRANT_URL = "http://localhost:6333"
DIMENSION = 1536  # OpenAI text-embedding-3-small

COLLECTIONS = {
    "documents": {
        "description": "All ingested documents (Drive, local files, etc.)",
        "on_disk": True
    },
    "conversations": {
        "description": "Chat conversation history",
        "on_disk": True
    },
    "facts": {
        "description": "Extracted facts about Patrick",
        "on_disk": False  # Keep in memory for fast retrieval
    },
    "code": {
        "description": "Indexed codebase",
        "on_disk": True
    }
}

def recreate_collections():
    # List existing collections
    response = requests.get(f"{QDRANT_URL}/collections")
    existing = [c["name"] for c in response.json()["result"]["collections"]]
    print(f"Existing collections: {existing}")

    # Delete old collections
    old_collections = ["claude_conversations", "echo_memories", "kb_articles"]
    for name in old_collections:
        if name in existing:
            print(f"Deleting old collection: {name}")
            requests.delete(f"{QDRANT_URL}/collections/{name}")

    # Create new collections
    for name, config in COLLECTIONS.items():
        if name in existing:
            print(f"Deleting existing collection: {name}")
            requests.delete(f"{QDRANT_URL}/collections/{name}")

        print(f"Creating collection: {name} ({config['description']})")
        response = requests.put(
            f"{QDRANT_URL}/collections/{name}",
            json={
                "vectors": {
                    "size": DIMENSION,
                    "distance": "Cosine",
                    "on_disk": config["on_disk"]
                },
                "optimizers_config": {
                    "indexing_threshold": 10000
                }
            }
        )
        if response.status_code == 200:
            print(f"  ✅ Created successfully")
        else:
            print(f"  ❌ Error: {response.text}")

    # Verify
    print("\n=== New Collections ===")
    response = requests.get(f"{QDRANT_URL}/collections")
    for c in response.json()["result"]["collections"]:
        info = requests.get(f"{QDRANT_URL}/collections/{c['name']}").json()["result"]
        print(f"  {c['name']}: {info['config']['params']['vectors']['size']}D, {info['points_count']} points")

if __name__ == "__main__":
    recreate_collections()