#!/usr/bin/env python3
"""
Conversation Watcher - Monitors for new conversations and ingests them
"""

import json
import logging
import hashlib
import httpx
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from uuid import uuid4
import asyncio
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationWatcher:
    """Watches for new conversations and ingests them into memory"""

    def __init__(self):
        # Database URL from environment (set by service with Vault)
        self.db_url = os.getenv("DATABASE_URL")
        if not self.db_url:
            logger.error("DATABASE_URL not set in environment")
            raise ValueError("DATABASE_URL environment variable required")
        self.qdrant_url = "http://localhost:6333"
        self.ollama_url = "http://localhost:11434"
        self.collection = "echo_memory"

        # Directories to watch
        self.watch_dirs = [
            Path("/opt/tower-echo-brain/conversations"),
            Path("/home/patrick/.claude/projects"),
        ]

        # Tracking file for processed conversations
        self.tracking_file = Path("/opt/tower-echo-brain/data/processed_conversations.json")
        self.tracking_file.parent.mkdir(exist_ok=True)
        self.processed_files = self.load_processed_files()
        self._seen_hashes: set = set()  # SHA-256 dedup within a cycle

    def load_processed_files(self) -> set:
        """Load list of already processed files"""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, 'r') as f:
                    return set(json.load(f))
            except:
                pass
        return set()

    def save_processed_files(self):
        """Save list of processed files"""
        with open(self.tracking_file, 'w') as f:
            json.dump(list(self.processed_files), f, indent=2)

    async def register_with_autonomous_core(self):
        """Register as an autonomous goal"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8309/api/autonomous/goals",
                    json={
                        "name": "conversation_watcher",
                        "description": "Monitor and ingest new conversations",
                        "safety_level": "notify",
                        "schedule": "*/10 * * * *",  # Every 10 minutes
                        "enabled": True
                    }
                )
                logger.info(f"Registered with autonomous core: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to register: {e}")

    async def find_new_conversations(self) -> List[Path]:
        """Find conversation files that haven't been processed"""
        new_files = []

        for watch_dir in self.watch_dirs:
            if not watch_dir.exists():
                continue

            # Look for conversation files
            for pattern in ["*.json", "*.jsonl", "*.txt", "*.md"]:
                for file_path in watch_dir.rglob(pattern):
                    # Skip if already processed
                    if str(file_path) in self.processed_files:
                        continue

                    # Skip if too old (older than 30 days)
                    if file_path.stat().st_mtime < (datetime.now().timestamp() - 30 * 86400):
                        continue

                    # Skip system files
                    if any(skip in str(file_path) for skip in [
                        "__pycache__", ".git", "node_modules", ".env", "venv"
                    ]):
                        continue

                    new_files.append(file_path)

        logger.info(f"Found {len(new_files)} new conversation files")
        return new_files

    def parse_conversation_content(self, file_path: Path) -> List[Dict]:
        """Parse conversation file into chunks"""
        chunks = []

        try:
            content = file_path.read_text(errors='ignore')

            # Handle different file types
            if file_path.suffix == '.jsonl':
                # Claude conversation format
                messages = []
                for line in content.split('\n'):
                    if line.strip():
                        try:
                            msg = json.loads(line)
                            if 'role' in msg and 'content' in msg:
                                messages.append(msg)
                        except:
                            pass

                # Chunk by message
                for msg in messages:
                    if isinstance(msg.get('content'), str):
                        chunks.append({
                            'text': f"[{msg['role']}]: {msg['content'][:2000]}",
                            'role': msg['role'],
                            'timestamp': msg.get('timestamp', '')
                        })

            elif file_path.suffix == '.json':
                # Try to parse as JSON
                try:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        # Extract text content
                        if 'messages' in data:
                            for msg in data['messages']:
                                chunks.append({
                                    'text': str(msg)[:2000],
                                    'timestamp': data.get('timestamp', '')
                                })
                        else:
                            chunks.append({
                                'text': json.dumps(data, indent=2)[:2000],
                                'timestamp': data.get('timestamp', '')
                            })
                except:
                    # Fallback to raw text
                    chunks.append({'text': content[:2000]})

            else:
                # Plain text or markdown - chunk by paragraphs
                paragraphs = content.split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        chunks.append({'text': para[:2000]})

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")

        return chunks

    async def ingest_conversation(self, file_path: Path):
        """Ingest a conversation file into Qdrant"""
        logger.info(f"Ingesting conversation: {file_path}")

        chunks = self.parse_conversation_content(file_path)
        if not chunks:
            return 0

        stored_count = 0
        points = []

        for chunk in chunks:
            try:
                # SHA-256 dedup: skip chunks we've already embedded
                chunk_hash = hashlib.sha256(chunk['text'].encode()).hexdigest()[:16]
                if chunk_hash in self._seen_hashes:
                    continue
                self._seen_hashes.add(chunk_hash)

                # Get embedding — Ollama /api/embed returns {"embeddings": [[...]]}
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.post(
                        f"{self.ollama_url}/api/embed",
                        json={"model": "nomic-embed-text", "input": chunk['text']}
                    )
                    resp_data = response.json()
                    # Handle both old ("embedding") and new ("embeddings") response formats
                    embedding = resp_data.get("embeddings", [None])[0] or resp_data.get("embedding")
                    if not embedding:
                        logger.warning(f"No embedding returned, keys: {list(resp_data.keys())}")
                        continue

                    points.append({
                        "id": str(uuid4()),
                        "vector": embedding,
                        "payload": {
                            "content": chunk['text'],
                            "type": "conversation",
                            "source": "conversation_watcher",
                            "file_path": str(file_path),
                            "role": chunk.get('role', 'unknown'),
                            "timestamp": chunk.get('timestamp', ''),
                            "content_hash": chunk_hash,
                            "ingested_at": datetime.now().isoformat()
                        }
                    })

                    # Batch store every 10 points
                    if len(points) >= 10:
                        await client.put(
                            f"{self.qdrant_url}/collections/{self.collection}/points",
                            json={"points": points}
                        )
                        stored_count += len(points)
                        points = []

            except Exception as e:
                logger.error(f"Error embedding chunk: {e}")

        # Store remaining points
        if points:
            try:
                async with httpx.AsyncClient() as client:
                    await client.put(
                        f"{self.qdrant_url}/collections/{self.collection}/points",
                        json={"points": points}
                    )
                    stored_count += len(points)
            except Exception as e:
                logger.error(f"Error storing points: {e}")

        return stored_count

    async def send_notification(self, message: str):
        """Send notification via autonomous core"""
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    "http://localhost:8309/api/autonomous/notifications",
                    json={
                        "title": "Conversation Watcher",
                        "message": message,
                        "level": "info"
                    }
                )
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")

    async def run_cycle(self):
        """Run one watch cycle"""
        try:
            # Find new conversations
            new_files = await self.find_new_conversations()

            if new_files:
                total_chunks = 0
                for file_path in new_files[:10]:  # Process max 10 files per cycle
                    chunks_stored = await self.ingest_conversation(file_path)
                    total_chunks += chunks_stored

                    # Mark as processed
                    self.processed_files.add(str(file_path))

                    logger.info(f"Ingested {chunks_stored} chunks from {file_path.name}")

                # Save tracking
                self.save_processed_files()

                # Send notification
                await self.send_notification(
                    f"Ingested {len(new_files)} new conversations with {total_chunks} chunks"
                )
            else:
                logger.info("No new conversations found")

        except Exception as e:
            logger.error(f"Error in watch cycle: {e}")

    async def start(self):
        """Start the watcher"""
        await self.register_with_autonomous_core()

        while True:
            try:
                await self.run_cycle()
                await asyncio.sleep(600)  # Wait 10 minutes
            except KeyboardInterrupt:
                logger.info("Stopping conversation watcher")
                break
            except Exception as e:
                logger.error(f"Watcher error: {e}")
                await asyncio.sleep(60)

if __name__ == "__main__":
    watcher = ConversationWatcher()
    asyncio.run(watcher.start())