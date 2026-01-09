#!/usr/bin/env python3
"""
Setup script to create the required Qdrant collection for Echo Brain vector memory system
"""

import json
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_echo_memories_collection():
    """Create the echo_memories collection in Qdrant"""

    qdrant_url = "http://localhost:6333"
    collection_name = "echo_memories"

    try:
        # Check if collection already exists
        logger.info(f"Checking if collection '{collection_name}' exists...")
        response = requests.get(f"{qdrant_url}/collections/{collection_name}")

        if response.status_code == 200:
            logger.info(f"✅ Collection '{collection_name}' already exists")
            return True

        elif response.status_code == 404:
            logger.info(f"Collection '{collection_name}' does not exist. Creating it...")

            # Create collection with 384 dimensions (typical for sentence-transformers)
            collection_config = {
                "vectors": {
                    "size": 384,  # Standard dimension for sentence-transformers/all-MiniLM-L6-v2
                    "distance": "Cosine"
                },
                "optimizers_config": {
                    "default_segment_number": 2,
                    "max_optimization_threads": 1
                },
                "replication_factor": 1
            }

            response = requests.put(
                f"{qdrant_url}/collections/{collection_name}",
                json=collection_config,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code in [200, 201]:
                logger.info(f"✅ Successfully created collection '{collection_name}'")
                return True
            else:
                logger.error(f"❌ Failed to create collection '{collection_name}': {response.status_code} - {response.text}")
                return False
        else:
            logger.error(f"❌ Unexpected response when checking collection: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        logger.error(f"❌ Error setting up Qdrant collection: {e}")
        return False

def verify_qdrant_service():
    """Verify Qdrant service is running and accessible"""

    qdrant_url = "http://localhost:6333"

    try:
        logger.info("Checking Qdrant service health...")
        response = requests.get(f"{qdrant_url}/")

        if response.status_code == 200:
            logger.info("✅ Qdrant service is running and accessible")
            return True
        else:
            logger.error(f"❌ Qdrant service returned unexpected status: {response.status_code}")
            return False

    except Exception as e:
        logger.error(f"❌ Cannot reach Qdrant service: {e}")
        return False

def main():
    """Main setup function"""

    logger.info("🚀 Starting Qdrant Vector Database Setup for Echo Brain")
    logger.info("=" * 60)

    # Step 1: Verify Qdrant service
    if not verify_qdrant_service():
        logger.error("❌ Qdrant service is not accessible. Please ensure it's running on port 6333")
        return False

    # Step 2: Create echo_memories collection
    if not create_echo_memories_collection():
        logger.error("❌ Failed to create echo_memories collection")
        return False

    # Step 3: Verify setup
    try:
        response = requests.get("http://localhost:6333/collections")
        collections = response.json()

        echo_memories_exists = any(
            col["name"] == "echo_memories"
            for col in collections.get("result", {}).get("collections", [])
        )

        if echo_memories_exists:
            logger.info("✅ Setup verification successful - echo_memories collection is ready")

            # Get collection info
            info_response = requests.get("http://localhost:6333/collections/echo_memories")
            if info_response.status_code == 200:
                info = info_response.json()
                vectors_config = info.get("result", {}).get("config", {}).get("params", {}).get("vectors", {})
                logger.info(f"📊 Collection info: {vectors_config.get('size', 'unknown')} dimensions, {vectors_config.get('distance', 'unknown')} distance")

            logger.info("🎉 Echo Brain vector memory system is ready!")
            return True
        else:
            logger.error("❌ Setup verification failed - collection not found")
            return False

    except Exception as e:
        logger.error(f"❌ Setup verification failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)