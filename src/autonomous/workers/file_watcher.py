"""
File Watcher Worker - Echo Brain Phase 2c
Monitors key directories for new/changed files and triggers ingestion.
This is the first data connector — detects new ComfyUI outputs,
new workflows, new model files, etc. and feeds them into the pipeline.
"""
import asyncio
import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import asyncpg


DB_URL = os.getenv("DATABASE_URL", "postgresql://echo:echo_secure_password_123@localhost/echo_brain")

# Directories to watch and their categories
WATCH_DIRS = {
    # ComfyUI output — new generated images/videos
    "/mnt/1TB-storage/ComfyUI/output/": {
        "category": "anime:output",
        "extensions": {".png", ".jpg", ".jpeg", ".mp4", ".webm"},
        "description": "Generated anime content from ComfyUI",
    },
    # Workflows — new or modified ComfyUI workflows
    "/mnt/1TB-storage/workflows/": {
        "category": "anime:comfyui",
        "extensions": {".json", ".py"},
        "description": "ComfyUI workflow definitions",
    },
    "/mnt/1TB-storage/ComfyUI/user/default/workflows/": {
        "category": "anime:comfyui",
        "extensions": {".json"},
        "description": "ComfyUI saved workflows",
    },
    # New model files
    "/mnt/1TB-storage/models/loras/": {
        "category": "anime:safetensors",
        "extensions": {".safetensors"},
        "description": "New LoRA models",
    },
    "/mnt/1TB-storage/models/checkpoints/": {
        "category": "anime:safetensors",
        "extensions": {".safetensors"},
        "description": "New checkpoint models",
    },
    # Tower docs — any new documentation
    "/opt/tower-echo-brain/docs/": {
        "category": "tower:architecture",
        "extensions": {".md", ".txt"},
        "description": "Echo Brain documentation",
    },
}

# Only process files modified within this window (avoids re-scanning entire history)
LOOKBACK_HOURS = 2


class FileWatcher:
    """
    Lightweight file watcher that detects new/modified files in key directories.

    For image/video files: creates metadata entries (filename, size, timestamp)
    since we can't embed binary content. Vision model analysis comes in a later phase.

    For text/JSON files: triggers the domain_ingestor for full ingestion.
    """

    def __init__(self):
        self.stats = {"new_files": 0, "updated_files": 0, "skipped": 0, "errors": 0}

    async def run_cycle(self):
        """Scan watched directories for new/changed files."""
        print(f"[FileWatcher] Starting scan at {datetime.now().isoformat()}")
        self.stats = {k: 0 for k in self.stats}
        cutoff = datetime.now() - timedelta(hours=LOOKBACK_HOURS)

        conn = await asyncpg.connect(DB_URL)
        try:
            for dir_path, config in WATCH_DIRS.items():
                path = Path(dir_path)
                if not path.exists():
                    continue

                extensions = config["extensions"]
                category = config["category"]

                for filepath in path.rglob("*"):
                    if not filepath.is_file():
                        continue
                    if filepath.suffix.lower() not in extensions:
                        continue

                    # Only look at recently modified files
                    mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
                    if mtime < cutoff:
                        continue

                    # Check if we've already logged this version
                    content_hash = self._file_hash(filepath)
                    existing = await conn.fetchrow(
                        "SELECT id FROM domain_ingestion_log WHERE source_path = $1 AND content_hash = $2",
                        str(filepath), content_hash
                    )
                    if existing:
                        self.stats["skipped"] += 1
                        continue

                    # New or changed file detected
                    await self._handle_new_file(conn, filepath, category, content_hash, config)

            print(f"[FileWatcher] Scan complete: {json.dumps(self.stats)}")

        except Exception as e:
            print(f"[FileWatcher] ERROR: {e}")
            self.stats["errors"] += 1
        finally:
            await conn.close()

    async def _handle_new_file(self, conn, filepath: Path, category: str,
                                content_hash: str, config: dict):
        """Process a newly detected file."""
        is_binary = filepath.suffix.lower() in {".png", ".jpg", ".jpeg", ".mp4", ".webm", ".safetensors"}

        if is_binary:
            # For binary files: log metadata only (vision analysis is a future phase)
            size_mb = filepath.stat().st_size / (1024 * 1024)
            text = (
                f"New file detected: {filepath.name}\n"
                f"Type: {filepath.suffix.lower()}\n"
                f"Category: {category}\n"
                f"Path: {filepath}\n"
                f"Size: {size_mb:.1f} MB\n"
                f"Created: {datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()}"
            )

            # Store a lightweight record (no embedding for binary metadata)
            await conn.execute("""
                INSERT INTO domain_ingestion_log
                    (source_type, source_path, category, content_hash,
                     chunk_count, vector_ids, file_size_bytes, last_modified)
                VALUES ('file_watch', $1, $2, $3, 0, '{}', $4, $5)
                ON CONFLICT (source_path, content_hash) DO NOTHING
            """,
                str(filepath), category, content_hash,
                filepath.stat().st_size,
                datetime.fromtimestamp(filepath.stat().st_mtime),
            )

            # Create a notification for important new files
            if filepath.suffix.lower() == ".safetensors":
                await conn.execute("""
                    INSERT INTO notifications (title, body, priority, source, category)
                    VALUES ($1, $2, $3, $4, $5)
                """,
                    f"New model detected: {filepath.name}",
                    f"A new {filepath.suffix} file was found at {filepath}\nSize: {size_mb:.1f} MB",
                    "normal",
                    str(filepath),
                    category,
                )
                self.stats["new_files"] += 1

        else:
            # For text files: the domain_ingestor will pick it up on next cycle.
            # Just create a notification so we know something changed.
            await conn.execute("""
                INSERT INTO notifications (title, body, priority, source, category)
                VALUES ($1, $2, $3, $4, $5)
            """,
                f"File changed: {filepath.name}",
                f"Modified at {datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()}\nPath: {filepath}\nWill be ingested on next domain_ingestor cycle.",
                "low",
                str(filepath),
                category,
            )
            self.stats["new_files"] += 1

    def _file_hash(self, filepath: Path) -> str:
        """Quick hash using path + mtime + size (avoids reading large binaries)."""
        stat = filepath.stat()
        key = f"{filepath}:{stat.st_mtime}:{stat.st_size}"
        return hashlib.sha256(key.encode()).hexdigest()