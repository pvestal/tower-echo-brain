#!/usr/bin/env python3
"""
Ingest Claude conversation exports into Echo Brain.
Usage: python ingest_conversations.py /path/to/conversations.json
"""
import asyncio
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "/opt/tower-echo-brain")
from src.autonomous.workers.domain_ingestor import DomainIngestor
import asyncpg

DB_URL = "postgresql://echo:echo_secure_password_123@localhost/echo_brain"

async def ingest_file(filepath, ingestor, conn):
    data = json.loads(filepath.read_text())
    convos = data if isinstance(data, list) else [data]
    total = 0

    for convo in convos:
        title = convo.get("name", convo.get("title", "untitled"))
        messages = convo.get("chat_messages", convo.get("messages", []))
        if not messages:
            continue

        current = f"Claude Conversation: {title}\nDate: {convo.get('created_at', '?')}\n\n"
        chunk_n = 0

        for msg in messages:
            role = msg.get("sender", msg.get("role", "?"))
            content = msg.get("text", "")
            if not content and isinstance(msg.get("content"), list):
                content = " ".join(c.get("text", "") for c in msg["content"]
                                  if isinstance(c, dict) and c.get("type") == "text")
            entry = f"[{role}]: {content}\n\n"

            if len(current) + len(entry) > 6000:
                h = hashlib.sha256(current.encode()).hexdigest()
                sp = f"convo:{title}:chunk_{chunk_n}"
                if not await conn.fetchrow("SELECT id FROM domain_ingestion_log WHERE source_path=$1", sp):
                    vid = await ingestor._embed_and_store(
                        {"text": current, "metadata": {"title": title}}, "tower:conversations", sp)
                    if vid:
                        await conn.execute("""INSERT INTO domain_ingestion_log
                            (source_type,source_path,category,content_hash,chunk_count,vector_ids)
                            VALUES('conversation',$1,'tower:conversations',$2,1,$3)""", sp, h, [vid])
                        total += 1
                chunk_n += 1
                current = f"Claude Conversation: {title} (continued)\n\n{entry}"
            else:
                current += entry

        if current.strip():
            h = hashlib.sha256(current.encode()).hexdigest()
            sp = f"convo:{title}:chunk_{chunk_n}"
            if not await conn.fetchrow("SELECT id FROM domain_ingestion_log WHERE source_path=$1", sp):
                vid = await ingestor._embed_and_store(
                    {"text": current, "metadata": {"title": title}}, "tower:conversations", sp)
                if vid:
                    await conn.execute("""INSERT INTO domain_ingestion_log
                        (source_type,source_path,category,content_hash,chunk_count,vector_ids)
                        VALUES('conversation',$1,'tower:conversations',$2,1,$3)""", sp, h, [vid])
                    total += 1
    return total

async def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest_conversations.py /path/to/conversations.json")
        sys.exit(1)

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"Error: {target} does not exist")
        sys.exit(1)

    ingestor = DomainIngestor()
    conn = await asyncpg.connect(DB_URL)
    try:
        files = [target] if target.is_file() else list(target.glob("*.json"))
        print(f"Found {len(files)} file(s)")
        grand = 0
        for f in files:
            print(f"  {f.name}...", end=" ")
            n = await ingest_file(f, ingestor, conn)
            print(f"{n} vectors")
            grand += n
        print(f"\nTotal: {grand} vectors")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())