#!/usr/bin/env python3
"""
Fast deduplication engine for Google Takeout files
"""

import asyncio
import hashlib
import asyncpg
from pathlib import Path
from datetime import datetime

class DeduplicationEngine:
    def __init__(self, db_url="postgresql://patrick:tower_echo_brain_secret_key_2025@localhost/echo_brain"):
        self.db_url = db_url
        self.old_takeout = Path("/mnt/10TB2/Google_Takeout_2025")
        self.new_takeout = Path("/opt/tower-echo-brain/data/takeout")

    def hash_file(self, filepath):
        """Fast SHA256 hash"""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    async def is_duplicate(self, filepath):
        """Check if file is duplicate of existing data"""
        file_hash = self.hash_file(filepath)

        # Check database first (fastest)
        conn = await asyncpg.connect(self.db_url)
        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM takeout_files_processed WHERE file_hash = $1)",
            file_hash
        )
        await conn.close()

        if exists:
            return True, "database"

        # Check old takeout (slower)
        for old_file in self.old_takeout.rglob("*"):
            if old_file.is_file() and old_file.stat().st_size == filepath.stat().st_size:
                old_hash = self.hash_file(old_file)
                if old_hash == file_hash:
                    return True, f"old_takeout:{old_file}"

        return False, None

    async def mark_processed(self, filepath, file_hash=None):
        """Mark file as processed in database"""
        if not file_hash:
            file_hash = self.hash_file(filepath)

        conn = await asyncpg.connect(self.db_url)
        await conn.execute("""
            INSERT INTO takeout_files_processed
            (file_path, file_hash, file_size_bytes, processed_at)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (file_hash) DO NOTHING
        """, str(filepath), file_hash, filepath.stat().st_size, datetime.now())
        await conn.close()

    async def scan_and_deduplicate(self, directory):
        """Scan directory and mark duplicates"""
        duplicates = []
        new_files = []

        for filepath in directory.rglob("*"):
            if filepath.is_file():
                is_dup, source = await self.is_duplicate(filepath)
                if is_dup:
                    duplicates.append((filepath, source))
                else:
                    new_files.append(filepath)
                    await self.mark_processed(filepath)

        return duplicates, new_files