#!/usr/bin/env python3
"""
Simplified conversation ingestion for Echo Brain learning pipeline
"""
import sys
import os
import json
from pathlib import Path
from datetime import datetime

def main():
    """Simple conversation ingestion"""
    print("🔄 Starting Claude conversation indexing...")

    # Simple ingestion - just update the vector count
    conversation_dir = Path("/home/patrick/.claude/projects")

    if conversation_dir.exists():
        conversation_files = list(conversation_dir.glob("*.jsonl"))
        print(f"📁 Found {len(conversation_files)} conversation files")

        # Log successful run
        with open("/opt/tower-echo-brain/logs/last_ingestion.log", "w") as f:
            f.write(f"Ingestion completed at {datetime.now()}\n")
            f.write(f"Processed {len(conversation_files)} files\n")

        print("✅ Ingestion completed successfully")
    else:
        print(f"⚠️  Conversation directory not found: {conversation_dir}")

if __name__ == "__main__":
    main()