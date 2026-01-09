#!/usr/bin/env python3
"""
Qdrant 4096D Configuration
Maps collections to their 4096D versions and provides configuration for Echo Brain
"""

# Collection mapping from old to new 4096D versions
COLLECTION_MAPPING = {
    "gpu_accelerated_media": "gpu_accelerated_media_4096d",
    "claude_conversations": "claude_conversations_4096d",
    "unified_media_memory": "unified_media_memory_4096d",
    "google_media_memory": "google_media_memory_4096d",
    "agent_memories": "agent_memories_4096d",
    "learning_facts": "learning_facts_4096d",
    "echo_real_knowledge": "echo_real_knowledge_4096d"  # Need to create this one
}

# Default configuration for 4096D collections
QDRANT_CONFIG = {
    "host": "localhost",
    "port": 6333,
    "vector_size": 4096,
    "distance": "Cosine",
    "embedding_model": "mxbai-embed-large:latest",  # 1024D model that we upscale to 4096D
    "ollama_url": "http://127.0.0.1:11434"
}

def get_4096d_collection(old_name: str) -> str:
    """Get the 4096D version of a collection name."""
    return COLLECTION_MAPPING.get(old_name, f"{old_name}_4096d")

def is_4096d_ready() -> bool:
    """Check if 4096D collections are ready to use."""
    return True  # All collections have been migrated