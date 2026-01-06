#!/usr/bin/env python3
"""
ACTUALLY reorganize Echo Brain - no more theater
Move files, delete junk, fix the mess
"""

import os
import shutil
from pathlib import Path
import re

base = Path("/opt/tower-echo-brain")

print("🔥 ACTUAL REORGANIZATION - NO BS")

# 1. DELETE ALL THE CRAP
print("\n1️⃣ Deleting backup files and junk...")
deleted = 0
for pattern in ["**/*.backup", "**/*.old", "**/*_backup.py", "**/*.pyc", "**/__pycache__"]:
    for f in base.glob(pattern):
        try:
            if f.is_file():
                f.unlink()
            else:
                shutil.rmtree(f)
            deleted += 1
        except:
            pass
print(f"   Deleted {deleted} files/dirs")

# 2. ORGANIZE ROOT FILES BY PATTERN
print("\n2️⃣ Moving root Python files to modules...")
moves = {
    "echo_": "src/core/echo/",
    "test_": "tests/",
    "anime_": "src/modules/generation/anime/",
    "agent_": "src/modules/agents/",
    "video_": "src/modules/generation/video/",
    "audio_": "src/modules/generation/audio/",
    "apple_": "src/integrations/apple/",
    "websocket": "src/infrastructure/websocket/",
    "database": "src/infrastructure/database/",
    "memory": "src/modules/memory/",
    "learning": "src/modules/learning/",
    "conversation": "src/modules/conversation/",
}

moved = 0
for py_file in base.glob("*.py"):
    filename = py_file.name.lower()

    # Find matching pattern
    for pattern, dest in moves.items():
        if pattern in filename:
            dest_path = base / dest
            dest_path.mkdir(parents=True, exist_ok=True)

            try:
                shutil.move(str(py_file), str(dest_path / py_file.name))
                moved += 1
                print(f"   Moved {py_file.name} -> {dest}")
                break
            except:
                pass

    # If no pattern matches, move to misc
    if py_file.exists():
        misc = base / "src/misc/"
        misc.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(py_file), str(misc / py_file.name))
            moved += 1
        except:
            pass

print(f"   Moved {moved} files")

# 3. CONSOLIDATE API FILES
print("\n3️⃣ Consolidating 35+ API files...")
api_dir = base / "src/api"
if api_dir.exists():
    # Keep only main files
    keep = ["echo.py", "__init__.py", "routes.py", "models.py", "dependencies.py"]
    consolidated = []

    for api_file in api_dir.glob("*.py"):
        if api_file.name not in keep and not api_file.name.startswith("_"):
            # Move to api/legacy/
            legacy = api_dir / "legacy"
            legacy.mkdir(exist_ok=True)
            try:
                shutil.move(str(api_file), str(legacy / api_file.name))
                consolidated.append(api_file.name)
            except:
                pass

    print(f"   Moved {len(consolidated)} API files to legacy/")

# 4. FIX BROKEN SYMLINKS
print("\n4️⃣ Removing broken symlinks...")
broken = 0
for path in base.rglob("*"):
    if path.is_symlink() and not path.exists():
        path.unlink()
        broken += 1
print(f"   Removed {broken} broken symlinks")

# 5. FINAL STATS
print("\n📊 FINAL RESULTS:")
remaining_root = len(list(base.glob("*.py")))
total_py = len(list(base.rglob("*.py")))

print(f"   Root Python files: {remaining_root} (was 215)")
print(f"   Total Python files: {total_py} (was 14,401)")
print(f"   Files moved: {moved}")
print(f"   Files deleted: {deleted}")

if remaining_root < 50:
    print("\n✅ SUCCESS - Root directory actually cleaned!")
else:
    print("\n⚠️  PARTIAL - Still work to do")

# Save what we did
report = {
    "deleted": deleted,
    "moved": moved,
    "remaining_root_files": remaining_root,
    "total_files": total_py,
    "broken_symlinks": broken
}

import json
with open(base / "actual_reorganization.json", 'w') as f:
    json.dump(report, f, indent=2)

print(f"\nReport saved to actual_reorganization.json")