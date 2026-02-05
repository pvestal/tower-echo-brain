#!/usr/bin/env python3
"""Ingest PostgreSQL schemas into Qdrant"""
import httpx
import json
import re
from uuid import uuid4
import time

OLLAMA_URL = "http://localhost:11434"
QDRANT_URL = "http://localhost:6333"
COLLECTION = "echo_memory"

def get_embedding(text):
    resp = httpx.post(f"{OLLAMA_URL}/api/embeddings",
        json={"model": "mxbai-embed-large", "prompt": text}, timeout=30)
    return resp.json()["embedding"]

def parse_create_tables(sql_path):
    """Extract CREATE TABLE statements"""
    with open(sql_path) as f:
        content = f.read()

    # Split on CREATE TABLE
    tables = re.findall(
        r'(CREATE TABLE[^;]+;)',
        content, re.DOTALL | re.IGNORECASE
    )
    return tables

def main():
    print("INGESTING DATABASE SCHEMAS")
    print("=" * 70)

    tables = parse_create_tables("/tmp/tower_schemas.sql")
    print(f"Found {len(tables)} CREATE TABLE statements")

    points = []
    for i, table_sql in enumerate(tables):
        # Extract table name
        match = re.search(r'CREATE TABLE\s+(?:IF NOT EXISTS\s+)?(\S+)', table_sql, re.IGNORECASE)
        table_name = match.group(1) if match else f"unknown_table_{i}"
        print(f"  {i+1}/{len(tables)}: {table_name}")

        text = f"# Database table: {table_name}\n{table_sql[:2000]}"
        try:
            embedding = get_embedding(text)
            points.append({
                "id": str(uuid4()),
                "vector": embedding,
                "payload": {
                    "text": text,
                    "source": "database_schema",
                    "table_name": table_name,
                    "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                }
            })
        except Exception as e:
            print(f"  ERROR: {e}")

    if points:
        resp = httpx.put(f"{QDRANT_URL}/collections/{COLLECTION}/points",
            json={"points": points}, timeout=30)
        print(f"\nStored {len(points)} schema vectors")

    resp = httpx.get(f"{QDRANT_URL}/collections/{COLLECTION}").json()
    print(f"Total vectors: {resp['result']['points_count']}")

if __name__ == "__main__":
    main()