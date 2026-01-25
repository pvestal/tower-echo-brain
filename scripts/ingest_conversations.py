#!/usr/bin/env python3
"""Ingest Claude conversations into Echo Brain"""
import sys
sys.path.insert(0, '/opt/tower-echo-brain')
from archive.root_python_files.ingest_claude_conversations import main
if __name__ == "__main__":
    main()
