#!/usr/bin/env python3
"""
Aggressively fix Echo Brain imports - just make it work
"""

import os
import shutil
from pathlib import Path

print("🔥 AGGRESSIVE FIX - Making Echo Brain work")

base = Path("/opt/tower-echo-brain")

# 1. Copy all critical files back to where they're expected
critical_copies = [
    ("src/db/models.py", "src/api/models_data.py"),  # Keep API models separate
    ("src/security/auth.py", "src/security/__init__.py"),  # Security init
]

for src, dst in critical_copies:
    src_path = base / src
    dst_path = base / dst
    if src_path.exists() and not dst_path.exists():
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        print(f"  Copied {src} -> {dst}")

# 2. Create missing __init__.py files
dirs_needing_init = [
    "src/security",
    "src/core",
    "src/api",
    "src/db",
    "src/services",
    "src/memory",
    "src/integrations",
    "src/utils",
    "src/modules",
]

for dir_path in dirs_needing_init:
    init_file = base / dir_path / "__init__.py"
    init_file.parent.mkdir(parents=True, exist_ok=True)
    if not init_file.exists():
        init_file.write_text("")
        print(f"  Created {dir_path}/__init__.py")

# 3. Create HTTPBearer if missing
security_init = base / "src/security/__init__.py"
if security_init.exists():
    content = security_init.read_text()
    if "HTTPBearer" not in content:
        # Add a simple HTTPBearer implementation
        bearer_code = '''
from fastapi import HTTPException, status
from fastapi.security import HTTPBearer as FastAPIHTTPBearer

class HTTPBearer(FastAPIHTTPBearer):
    """Simple HTTP Bearer authentication"""
    def __init__(self):
        super().__init__(auto_error=True)

# Export for compatibility
__all__ = ["HTTPBearer"]
'''
        security_init.write_text(content + bearer_code)
        print("  Added HTTPBearer to security/__init__.py")

# 4. Fix the most common broken imports by creating symlinks
symlinks = [
    ("src/models", "src/db/models.py"),
    ("src/database", "src/db/database.py"),
]

for link, target in symlinks:
    link_path = base / link
    target_path = base / target
    if target_path.exists() and not link_path.exists():
        link_path.parent.mkdir(parents=True, exist_ok=True)
        link_path.symlink_to(target_path.resolve())
        print(f"  Created symlink {link} -> {target}")

print("\n✅ Aggressive fixes applied - trying to start Echo Brain...")