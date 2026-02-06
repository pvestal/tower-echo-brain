#!/usr/bin/env python3
"""Scan filesystem for safetensors and ingest metadata. Usage: python scan_models.py"""
import asyncio
import hashlib
import os
from pathlib import Path
from datetime import datetime
from uuid import uuid4
import asyncpg
import httpx

DIRS = ["/mnt/1TB-storage/models", "/mnt/1TB-storage/ComfyUI/models", "/opt/ComfyUI/models"]
OLLAMA = "http://localhost:11434"
QDRANT = "http://localhost:6333"
DB = "postgresql://echo:echo_secure_password_123@localhost/echo_brain"

async def main():
    conn = await asyncpg.connect(DB)
    total = 0
    try:
        for d in DIRS:
            base = Path(d)
            if not base.exists():
                continue
            print(f"Scanning {base}...")
            for f in base.rglob("*.safetensors"):
                rel = str(f.relative_to(base)).lower()
                mtype = ("lora" if "lora" in rel else "checkpoint" if "checkpoint" in rel
                         else "vae" if "vae" in rel else "controlnet" if "controlnet" in rel else "unknown")
                mb = f.stat().st_size / 1048576
                text = f"Model: {f.name}\nType: {mtype}\nPath: {f}\nSize: {mb:.0f}MB"
                h = hashlib.sha256(f"{f}:{f.stat().st_mtime}".encode()).hexdigest()
                sp = f"model:{f}"
                if await conn.fetchrow("SELECT id FROM domain_ingestion_log WHERE source_path=$1 AND content_hash=$2", sp, h):
                    continue
                async with httpx.AsyncClient(timeout=60) as c:
                    r = await c.post(f"{OLLAMA}/api/embeddings", json={"model": "mxbai-embed-large:latest", "prompt": text})
                    if r.status_code != 200:
                        print(f"  ❌ Failed to embed {f.name}")
                        continue
                    emb = r.json().get("embedding")
                    if not emb:
                        continue
                    pid = str(uuid4())
                    await c.put(f"{QDRANT}/collections/echo_memory/points",
                        json={"points": [{"id": pid, "vector": emb,
                              "payload": {"text": text, "category": "anime:safetensors", "source": sp,
                                          "model_type": mtype, "model_name": f.name, "size_mb": round(mb)}}]})
                await conn.execute("""INSERT INTO domain_ingestion_log
                    (source_type,source_path,category,content_hash,chunk_count,vector_ids,file_size_bytes)
                    VALUES('model_scan',$1,'anime:safetensors',$2,1,$3,$4)""", sp, h, [pid], f.stat().st_size)
                total += 1
                print(f"  ✅ {mtype}: {f.name} ({mb:.0f}MB)")
    finally:
        await conn.close()
    print(f"\nIngested {total} models")

if __name__ == "__main__":
    asyncio.run(main())