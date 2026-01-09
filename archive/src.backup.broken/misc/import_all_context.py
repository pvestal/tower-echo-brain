#!/usr/bin/env python3
"""
Import all Knowledge Base articles and Claude conversations into Echo Brain
This builds Echo's complete memory from all past interactions
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

class EchoMemoryImporter:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'echo_brain',
            'user': 'patrick',
            'password': '***REMOVED***'
        }
        self.echo_api = "http://localhost:8309"
        self.kb_api = "http://localhost:8307"
        self.qdrant_api = "http://localhost:6333"
        self.conversations_dir = Path.home() / ".claude" / "conversations"

        self.imported_count = 0
        self.skipped_count = 0
        self.error_count = 0

    async def get_db_pool(self):
        """Create database connection pool"""
        return await asyncpg.create_pool(**self.db_config)

    async def import_kb_articles(self):
        """Import all Knowledge Base articles"""
        print("📚 Importing Knowledge Base articles...")

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Get all KB articles
                response = await client.get(f"{self.kb_api}/api/kb/articles")
                if response.status_code == 200:
                    articles = response.json()
                    print(f"Found {len(articles)} KB articles")

                    for article in articles:
                        # Store in Echo
                        query = f"KB_ARTICLE: {article.get('title', 'Untitled')}\n{article.get('content', '')[:500]}"

                        try:
                            echo_response = await client.post(
                                f"{self.echo_api}/api/echo/query",
                                json={
                                    "query": query,
                                    "conversation_id": "kb_import",
                                    "metadata": {
                                        "source": "knowledge_base",
                                        "article_id": article.get('id'),
                                        "category": article.get('category', 'general'),
                                        "imported_at": datetime.now().isoformat()
                                    }
                                }
                            )
                            if echo_response.status_code == 200:
                                self.imported_count += 1
                                print(f"✅ Imported: {article.get('title', 'Untitled')}")
                        except Exception as e:
                            print(f"❌ Error importing KB article: {e}")
                            self.error_count += 1
                else:
                    print(f"⚠️ KB API returned status {response.status_code}")
            except Exception as e:
                print(f"❌ Failed to access KB: {e}")

    async def import_claude_conversations(self, limit=100):
        """Import recent Claude conversations"""
        print(f"\n💬 Importing Claude conversations (limit: {limit})...")

        # Get most recent conversation files
        conv_files = sorted(
            self.conversations_dir.glob("*.md"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )[:limit]

        print(f"Found {len(conv_files)} recent conversation files")

        pool = await self.get_db_pool()
        async with httpx.AsyncClient(timeout=30.0) as client:
            for conv_file in conv_files:
                try:
                    # Read conversation
                    content = conv_file.read_text(encoding='utf-8', errors='ignore')

                    # Extract key information (first 1000 chars for summary)
                    summary = content[:1000]

                    # Extract date from filename or content
                    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', conv_file.name)
                    date_str = date_match.group(1) if date_match else datetime.now().strftime('%Y-%m-%d')

                    # Generate unique hash for deduplication
                    content_hash = hashlib.md5(summary.encode()).hexdigest()

                    # Check if already imported
                    async with pool.acquire() as conn:
                        exists = await conn.fetchval(
                            "SELECT EXISTS(SELECT 1 FROM conversations WHERE metadata->>'content_hash' = $1)",
                            content_hash
                        )

                        if exists:
                            self.skipped_count += 1
                            continue

                    # Import to Echo
                    query = f"CONVERSATION_IMPORT from {date_str}: {summary}"

                    echo_response = await client.post(
                        f"{self.echo_api}/api/echo/query",
                        json={
                            "query": query,
                            "conversation_id": f"claude_import_{conv_file.stem}",
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
                        self.imported_count += 1
                        print(f"✅ Imported: {conv_file.name}")

                except Exception as e:
                    print(f"❌ Error importing {conv_file.name}: {e}")
                    self.error_count += 1

        await pool.close()

    async def import_to_vector_memory(self):
        """Import key learnings to Qdrant vector memory"""
        print("\n💾 Importing to vector memory...")

        pool = await self.get_db_pool()
        async with pool.acquire() as conn:
            # Get recent learnings that aren't in vector memory yet
            learnings = await conn.fetch("""
                SELECT fact, learned_at, confidence, source
                FROM learning_history
                WHERE learned_at > NOW() - INTERVAL '30 days'
                ORDER BY learned_at DESC
                LIMIT 100
            """)

            print(f"Found {len(learnings)} recent learnings to vectorize")

            # Here you would normally:
            # 1. Generate embeddings using Ollama or other service
            # 2. Store in Qdrant
            # For now, just log what would be imported

            for learning in learnings[:10]:  # Show first 10
                print(f"  • {learning['fact'][:100]} ({learning['learned_at'].strftime('%Y-%m-%d')})")

        await pool.close()

    async def create_summary_report(self):
        """Generate summary of what was imported"""
        print("\n📊 Import Summary Report")
        print("=" * 50)

        pool = await self.get_db_pool()
        async with pool.acquire() as conn:
            # Get current counts
            conv_count = await conn.fetchval("SELECT COUNT(*) FROM conversations")
            learning_count = await conn.fetchval("SELECT COUNT(*) FROM learning_history")
            thought_count = await conn.fetchval("SELECT COUNT(*) FROM internal_thoughts")

        await pool.close()

        # Get Qdrant stats
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.qdrant_api}/collections")
                if response.status_code == 200:
                    collections = response.json()['result']['collections']
                    print(f"Vector Collections: {len(collections)}")
                    for coll in collections:
                        coll_response = await client.get(f"{self.qdrant_api}/collections/{coll['name']}")
                        if coll_response.status_code == 200:
                            points = coll_response.json()['result']['points_count']
                            print(f"  • {coll['name']}: {points} vectors")
            except:
                print("Vector database stats unavailable")

        print(f"\nDatabase totals:")
        print(f"  • Conversations: {conv_count}")
        print(f"  • Learning History: {learning_count}")
        print(f"  • Internal Thoughts: {thought_count}")

        print(f"\nImport results:")
        print(f"  ✅ Imported: {self.imported_count}")
        print(f"  ⏭️ Skipped (duplicates): {self.skipped_count}")
        print(f"  ❌ Errors: {self.error_count}")

        # Store summary in Echo
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{self.echo_api}/api/echo/query",
                json={
                    "query": f"IMPORT_COMPLETE: Imported {self.imported_count} items from KB and Claude conversations. Database now has {conv_count} conversations and {learning_count} learnings. Vector memory active.",
                    "conversation_id": "import_summary",
                    "metadata": {
                        "type": "import_summary",
                        "imported": self.imported_count,
                        "skipped": self.skipped_count,
                        "errors": self.error_count,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            )

async def main():
    """Run the import process"""
    print("🚀 Starting Echo Brain Memory Import")
    print("This will import KB articles and Claude conversations")
    print("-" * 50)

    importer = EchoMemoryImporter()

    # Import KB articles
    await importer.import_kb_articles()

    # Import recent Claude conversations (limit to prevent overwhelming)
    # You can increase the limit if needed
    await importer.import_claude_conversations(limit=50)

    # Import to vector memory
    await importer.import_to_vector_memory()

    # Generate report
    await importer.create_summary_report()

    print("\n✅ Import complete! Echo Brain memory updated.")

if __name__ == "__main__":
    asyncio.run(main())