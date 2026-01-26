#!/usr/bin/env python3
"""
Import ALL JSON Claude conversations into Echo Brain
Handles the 12,000+ JSON files in conversations directory
"""

import asyncio
import json
import os
from pathlib import Path
import hashlib
import asyncpg
import httpx
from datetime import datetime
import re
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class JSONConversationImporter:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'echo_brain',
            'user': 'patrick',
            'password': os.getenv("TOWER_DB_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE")
        }
        self.echo_api = "http://localhost:8309"
        self.conversations_dir = Path.home() / ".claude" / "conversations"

        self.imported = 0
        self.skipped = 0
        self.errors = 0
        self.total_processed = 0

    async def get_db_pool(self):
        """Create database connection pool"""
        return await asyncpg.create_pool(**self.db_config, max_size=10)

    def extract_json_content(self, json_data: Dict[Any, Any]) -> str:
        """Extract meaningful content from JSON conversation"""
        content_parts = []

        # Try different common structures
        if isinstance(json_data, dict):
            # Check for common fields
            for field in ['content', 'message', 'text', 'query', 'prompt', 'conversation']:
                if field in json_data:
                    content_parts.append(str(json_data[field])[:500])

            # Check for messages array
            if 'messages' in json_data and isinstance(json_data['messages'], list):
                for msg in json_data['messages'][:5]:  # First 5 messages
                    if isinstance(msg, dict):
                        content_parts.append(str(msg.get('content', msg.get('text', '')))[:200])
                    elif isinstance(msg, str):
                        content_parts.append(msg[:200])

            # Check for title/subject
            for field in ['title', 'subject', 'name']:
                if field in json_data:
                    content_parts.insert(0, f"Title: {json_data[field]}")

            # If no specific fields, just take first 500 chars of JSON
            if not content_parts:
                content_parts.append(json.dumps(json_data)[:500])

        elif isinstance(json_data, list) and json_data:
            # If it's a list, process first few items
            for item in json_data[:3]:
                if isinstance(item, dict):
                    content_parts.append(json.dumps(item)[:200])
                else:
                    content_parts.append(str(item)[:200])

        else:
            # Fallback: convert to string
            content_parts.append(str(json_data)[:500])

        return '\n'.join(content_parts) if content_parts else "Empty conversation"

    async def import_json_conversations(self, batch_size=50):
        """Import all JSON conversation files"""
        logger.info("💬 Starting JSON conversations import...")

        # Get all JSON files
        json_files = list(self.conversations_dir.glob("*.json"))
        total_files = len(json_files)
        logger.info(f"Found {total_files} JSON conversation files to import")

        # Sort by modification time (newest first)
        json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        pool = await self.get_db_pool()
        async with httpx.AsyncClient(timeout=30.0) as client:

            for batch_start in range(0, total_files, batch_size):
                batch_end = min(batch_start + batch_size, total_files)
                batch = json_files[batch_start:batch_end]

                logger.info(f"Processing batch {batch_start//batch_size + 1}: files {batch_start+1}-{batch_end}/{total_files}")

                for json_file in batch:
                    self.total_processed += 1

                    try:
                        # Read and parse JSON
                        with open(json_file, 'r', encoding='utf-8', errors='ignore') as f:
                            json_data = json.load(f)

                        # Extract meaningful content
                        content = self.extract_json_content(json_data)

                        # Extract date from filename
                        date_match = re.search(r'(\d{4}[-_]\d{2}[-_]\d{2})', json_file.name)
                        if not date_match:
                            # Try timestamp format
                            date_match = re.search(r'(\d{8})', json_file.name)
                            if date_match:
                                date_str = f"{date_match.group(1)[:4]}-{date_match.group(1)[4:6]}-{date_match.group(1)[6:8]}"
                            else:
                                date_str = datetime.fromtimestamp(json_file.stat().st_mtime).strftime('%Y-%m-%d')
                        else:
                            date_str = date_match.group(1).replace('_', '-')

                        # Generate hash for deduplication
                        content_hash = hashlib.md5(content.encode()).hexdigest()

                        # Check if already imported
                        async with pool.acquire() as conn:
                            exists = await conn.fetchval(
                                "SELECT EXISTS(SELECT 1 FROM conversations WHERE metadata->>'content_hash' = $1)",
                                content_hash
                            )

                            if exists:
                                self.skipped += 1
                                continue

                        # Prepare query for Echo
                        query = f"JSON_CONVERSATION from {date_str}: {json_file.stem}\n{content[:1000]}"

                        # Import to Echo
                        echo_response = await client.post(
                            f"{self.echo_api}/api/echo/query",
                            json={
                                "query": query,
                                "conversation_id": f"json_{json_file.stem}",
                                "metadata": {
                                    "source": "claude_json_conversations",
                                    "file": json_file.name,
                                    "date": date_str,
                                    "content_hash": content_hash,
                                    "file_type": "json",
                                    "imported_at": datetime.now().isoformat()
                                }
                            }
                        )

                        if echo_response.status_code == 200:
                            self.imported += 1
                        else:
                            self.errors += 1
                            logger.warning(f"Failed to import {json_file.name}: {echo_response.status_code}")

                    except json.JSONDecodeError:
                        self.errors += 1
                        logger.error(f"Invalid JSON in {json_file.name}")
                    except Exception as e:
                        self.errors += 1
                        logger.error(f"Error importing {json_file.name}: {e}")

                # Progress update every batch
                if self.total_processed % 100 == 0:
                    logger.info(f"Progress: {self.total_processed}/{total_files} processed")
                    logger.info(f"  ✅ Imported: {self.imported}, ⏭️ Skipped: {self.skipped}, ❌ Errors: {self.errors}")

                # Brief pause between batches to avoid overwhelming Echo
                if batch_start + batch_size < total_files:
                    await asyncio.sleep(0.5)

        await pool.close()

        logger.info(f"\n✅ JSON import complete!")
        logger.info(f"  • Processed: {self.total_processed} files")
        logger.info(f"  • Imported: {self.imported}")
        logger.info(f"  • Skipped (duplicates): {self.skipped}")
        logger.info(f"  • Errors: {self.errors}")

    async def update_echo_with_summary(self):
        """Store import summary in Echo"""
        async with httpx.AsyncClient() as client:
            summary = (
                f"JSON CONVERSATION IMPORT COMPLETE: Processed {self.total_processed} JSON files. "
                f"Successfully imported {self.imported} new conversations. "
                f"Skipped {self.skipped} duplicates. Encountered {self.errors} errors. "
                f"Echo Brain now has comprehensive knowledge from all Claude JSON conversations."
            )

            await client.post(
                f"{self.echo_api}/api/echo/query",
                json={
                    "query": summary,
                    "conversation_id": "json_import_complete",
                    "metadata": {
                        "type": "import_summary",
                        "processed": self.total_processed,
                        "imported": self.imported,
                        "skipped": self.skipped,
                        "errors": self.errors,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            )

async def main():
    """Run the JSON conversation import"""
    print("=" * 60)
    print("🚀 JSON CONVERSATION IMPORT TO ECHO BRAIN")
    print("=" * 60)
    print("This will import all JSON conversation files.")
    print("This may take 10-20 minutes for 12,000+ files...")
    print("-" * 60)

    # Confirm with user due to large volume
    response = input("Continue with import? (yes/no): ")
    if response.lower() != 'yes':
        print("Import cancelled.")
        return

    importer = JSONConversationImporter()
    await importer.import_json_conversations(batch_size=50)
    await importer.update_echo_with_summary()

    print("\n" + "=" * 60)
    print("✅ JSON IMPORT COMPLETE")
    print(f"Echo Brain absorbed {importer.imported} new JSON conversations")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())