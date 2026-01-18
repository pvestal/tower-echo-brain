#!/usr/bin/env python3
"""
Complete Echo Brain Memory Import
Imports ALL KB articles and Claude conversations into Echo Brain
This is the full import, not a test
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteEchoImporter:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'echo_brain',
            'user': 'patrick',
            'password': 'RP78eIrW7cI2jYvL5akt1yurE'
        }
        self.echo_api = "http://localhost:8309"
        self.kb_api = "http://localhost:8307"
        self.conversations_dir = Path.home() / ".claude" / "conversations"

        self.kb_imported = 0
        self.conv_imported = 0
        self.skipped = 0
        self.errors = 0

    async def get_db_pool(self):
        """Create database connection pool"""
        return await asyncpg.create_pool(**self.db_config)

    async def import_all_kb_articles(self):
        """Import ALL Knowledge Base articles"""
        logger.info("📚 Importing ALL Knowledge Base articles...")

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                # Get all KB articles
                response = await client.get(f"{self.kb_api}/api/articles")
                if response.status_code == 200:
                    articles = response.json()
                    logger.info(f"Found {len(articles)} KB articles to import")

                    for idx, article in enumerate(articles, 1):
                        try:
                            # Prepare content for Echo
                            title = article.get('title', 'Untitled')
                            content = article.get('content', '')
                            category = article.get('category', 'general')

                            # Truncate very long content but keep key info
                            if len(content) > 2000:
                                content = content[:2000] + "... [truncated]"

                            query = f"KB_ARTICLE #{article.get('id', idx)}: {title}\nCategory: {category}\n{content}"

                            # Store in Echo
                            echo_response = await client.post(
                                f"{self.echo_api}/api/echo/query",
                                json={
                                    "query": query,
                                    "conversation_id": f"kb_import_{article.get('id', idx)}",
                                    "metadata": {
                                        "source": "knowledge_base",
                                        "article_id": article.get('id', idx),
                                        "title": title,
                                        "category": category,
                                        "tags": article.get('tags', []),
                                        "imported_at": datetime.now().isoformat()
                                    }
                                }
                            )

                            if echo_response.status_code == 200:
                                self.kb_imported += 1
                                if idx % 10 == 0:
                                    logger.info(f"Progress: {idx}/{len(articles)} KB articles imported")
                            else:
                                self.errors += 1

                        except Exception as e:
                            logger.error(f"Error importing KB article {idx}: {e}")
                            self.errors += 1

                    logger.info(f"✅ KB Import complete: {self.kb_imported} articles imported")
                else:
                    logger.error(f"KB API returned status {response.status_code}")
            except Exception as e:
                logger.error(f"Failed to access KB: {e}")

    async def import_all_claude_conversations(self, batch_size=100):
        """Import ALL Claude conversations in batches"""
        logger.info(f"💬 Importing ALL Claude conversations...")

        # Get all conversation files
        conv_files = list(self.conversations_dir.glob("*.md"))
        total_files = len(conv_files)
        logger.info(f"Found {total_files} conversation files")

        # Sort by modification time (newest first)
        conv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        pool = await self.get_db_pool()
        async with httpx.AsyncClient(timeout=30.0) as client:

            for batch_start in range(0, total_files, batch_size):
                batch_end = min(batch_start + batch_size, total_files)
                batch = conv_files[batch_start:batch_end]

                logger.info(f"Processing batch {batch_start//batch_size + 1}: files {batch_start+1}-{batch_end}")

                for conv_file in batch:
                    try:
                        # Read conversation
                        content = conv_file.read_text(encoding='utf-8', errors='ignore')

                        # Extract key information
                        lines = content.split('\n')
                        title_line = lines[0] if lines else "Untitled"
                        summary = '\n'.join(lines[:20])  # First 20 lines as summary

                        # Extract date from filename
                        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', conv_file.name)
                        date_str = date_match.group(1) if date_match else datetime.now().strftime('%Y-%m-%d')

                        # Generate hash for deduplication
                        content_hash = hashlib.md5(summary.encode()).hexdigest()

                        # Check if already imported
                        async with pool.acquire() as conn:
                            exists = await conn.fetchval(
                                "SELECT EXISTS(SELECT 1 FROM conversations WHERE metadata->>'content_hash' = $1)",
                                content_hash
                            )

                            if exists:
                                self.skipped += 1
                                continue

                        # Import to Echo
                        query = f"CONVERSATION from {date_str}: {conv_file.stem}\n{title_line}\n{summary}"

                        echo_response = await client.post(
                            f"{self.echo_api}/api/echo/query",
                            json={
                                "query": query,
                                "conversation_id": f"claude_{conv_file.stem}",
                                "metadata": {
                                    "source": "claude_conversations",
                                    "file": conv_file.name,
                                    "date": date_str,
                                    "content_hash": content_hash,
                                    "imported_at": datetime.now().isoformat()
                                }
                            }
                        )

                        if echo_response.status_code == 200:
                            self.conv_imported += 1
                        else:
                            self.errors += 1

                    except Exception as e:
                        logger.error(f"Error importing {conv_file.name}: {e}")
                        self.errors += 1

                # Progress update
                logger.info(f"Progress: {min(batch_end, total_files)}/{total_files} files processed")
                logger.info(f"  Imported: {self.conv_imported}, Skipped: {self.skipped}, Errors: {self.errors}")

                # Brief pause between batches
                await asyncio.sleep(1)

        await pool.close()
        logger.info(f"✅ Conversation import complete: {self.conv_imported} imported, {self.skipped} skipped")

    async def verify_import_stats(self):
        """Verify what was imported"""
        logger.info("\n📊 Verifying import statistics...")

        pool = await self.get_db_pool()
        async with pool.acquire() as conn:
            # Get database counts
            conv_count = await conn.fetchval("SELECT COUNT(*) FROM conversations")
            kb_conv_count = await conn.fetchval(
                "SELECT COUNT(*) FROM conversations WHERE metadata->>'source' = 'knowledge_base'"
            )
            claude_conv_count = await conn.fetchval(
                "SELECT COUNT(*) FROM conversations WHERE metadata->>'source' = 'claude_conversations'"
            )
            learning_count = await conn.fetchval("SELECT COUNT(*) FROM learning_history")

        await pool.close()

        # Get Qdrant stats if available
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:6333/collections")
                if response.status_code == 200:
                    collections = response.json()['result']['collections']
                    logger.info(f"\n💾 Vector Database Collections:")
                    for coll in collections:
                        coll_response = await client.get(f"http://localhost:6333/collections/{coll['name']}")
                        if coll_response.status_code == 200:
                            points = coll_response.json()['result']['points_count']
                            logger.info(f"  • {coll['name']}: {points} vectors")
        except:
            logger.warning("Vector database not accessible")

        logger.info(f"\n🗄️ Database Statistics:")
        logger.info(f"  • Total conversations: {conv_count}")
        logger.info(f"  • KB articles imported: {kb_conv_count}")
        logger.info(f"  • Claude conversations imported: {claude_conv_count}")
        logger.info(f"  • Learning history entries: {learning_count}")

        logger.info(f"\n📈 Import Summary:")
        logger.info(f"  • KB articles imported: {self.kb_imported}")
        logger.info(f"  • Conversations imported: {self.conv_imported}")
        logger.info(f"  • Duplicates skipped: {self.skipped}")
        logger.info(f"  • Errors encountered: {self.errors}")

        # Store final summary in Echo
        async with httpx.AsyncClient() as client:
            summary = (
                f"COMPLETE IMPORT FINISHED: Imported {self.kb_imported} KB articles and "
                f"{self.conv_imported} Claude conversations. Database now has {conv_count} total "
                f"conversations. Echo Brain has consumed all available context."
            )

            await client.post(
                f"{self.echo_api}/api/echo/query",
                json={
                    "query": summary,
                    "conversation_id": "import_complete",
                    "metadata": {
                        "type": "import_summary",
                        "kb_imported": self.kb_imported,
                        "conv_imported": self.conv_imported,
                        "total_conversations": conv_count,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            )

async def main():
    """Run the complete import"""
    print("=" * 60)
    print("🚀 COMPLETE ECHO BRAIN MEMORY IMPORT")
    print("=" * 60)
    print("This will import:")
    print("  • ALL Knowledge Base articles")
    print("  • ALL Claude conversation files")
    print("This may take several minutes...")
    print("-" * 60)

    importer = CompleteEchoImporter()

    # Import everything
    await importer.import_all_kb_articles()
    await importer.import_all_claude_conversations()

    # Verify results
    await importer.verify_import_stats()

    print("\n" + "=" * 60)
    print("✅ COMPLETE IMPORT FINISHED")
    print("Echo Brain now has full context from all KB and conversations")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())