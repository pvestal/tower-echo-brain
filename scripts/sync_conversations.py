#!/usr/bin/env python3
"""
Sync conversations script - delegates to actual ingestion pipeline
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import and run the actual ingestion
from archive.root_python_files.ingest_claude_conversations import main

if __name__ == "__main__":
    print("🔄 Running conversation sync via ingestion pipeline...")
    main()