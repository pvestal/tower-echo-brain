#!/usr/bin/env python3
"""
Clean up duplicate ingestion systems in Echo Brain
"""
import os
import sys
import shutil
from pathlib import Path

def find_duplicate_systems():
    """Find duplicate/overlapping systems"""
    src_dir = Path("/opt/tower-echo-brain/src")
    
    # Vector/memory systems
    vector_systems = []
    for pattern in ["*vector*.py", "*memory*.py", "*qdrant*.py"]:
        for f in src_dir.rglob(pattern):
            if f.is_file() and "__pycache__" not in str(f):
                vector_systems.append(str(f))
    
    print(f"Found {len(vector_systems)} vector/memory files")
    
    # Group by similar names
    from collections import defaultdict
    groups = defaultdict(list)
    
    for file in vector_systems:
        name = Path(file).name
        groups[name].append(file)
    
    # Show duplicates
    print("\n⚠️  DUPLICATE SYSTEMS FOUND:")
    for name, files in groups.items():
        if len(files) > 1:
            print(f"\n{name}:")
            for f in files:
                print(f"  - {f}")
    
    # Check which are imported
    print("\n🔍 CHECKING IMPORTS:")
    main_file = src_dir / "main.py"
    if main_file.exists():
        with open(main_file) as f:
            content = f.read()
            for name in groups:
                if name in content:
                    print(f"✅ {name} is imported in main.py")
                else:
                    print(f"❌ {name} NOT imported in main.py")

def suggest_cleanup():
    """Suggest which files to keep/remove"""
    print("\n🎯 SUGGESTED CLEANUP:")
    print("===================")
    print("KEEP:")
    print("  - /opt/tower-echo-brain/src/echo_vector_memory.py (main implementation)")
    print("  - /opt/tower-echo-brain/src/services/real_vector_search.py (search)")
    print("  - /opt/tower-echo-brain/src/routers/conversation_minimal_router.py (API)")
    
    print("\nCONSIDER REMOVING/RENAMING:")
    print("  - /opt/tower-echo-brain/src/qdrant_memory.py (duplicate of echo_vector_memory)")
    print("  - /opt/tower-echo-brain/src/services/vector_search.py (duplicate of real_vector_search)")
    print("  - /opt/tower-echo-brain/src/modules/memory/index_claude_memory.py (specialized, keep if needed)")
    
    print("\nDATABASE CONNECTIONS:")
    print("  All should use: os.getenv('DB_USER', 'echo_brain_service')")
    print("  Password from: os.getenv('DB_PASSWORD')")

if __name__ == "__main__":
    print("=== Echo Brain System Cleanup Analysis ===")
    find_duplicate_systems()
    suggest_cleanup()
