#!/usr/bin/env python3
"""
Background Fact Extraction Worker
Processes vectors and extracts structured facts using Ollama.
Runs continuously, can be stopped with Ctrl+C or kill.
"""

import psycopg2
import httpx
import json
import time
import sys
import os
from datetime import datetime

DB_CONFIG = {
    'host': 'localhost',
    'database': 'echo_brain',
    'user': 'patrick',
    'password': 'RP78eIrW7cI2jYvL5akt1yurE'
}

OLLAMA_URL = 'http://localhost:11434'
MODEL = 'gemma2:9b'  # Good balance of speed and quality
BATCH_SIZE = 10
SLEEP_BETWEEN_BATCHES = 2  # seconds

def get_unprocessed_vectors(conn, limit=10):
    """Get vectors that haven't had facts extracted."""
    cur = conn.cursor()
    cur.execute("""
        SELECT id, content_text, content_type
        FROM embeddings
        WHERE id NOT IN (SELECT DISTINCT source_embedding_id FROM facts WHERE source_embedding_id IS NOT NULL)
        AND content_text IS NOT NULL
        AND length(content_text) > 50
        LIMIT %s
    """, (limit,))
    return cur.fetchall()

def extract_facts_with_ollama(text: str) -> list:
    """Use Ollama to extract facts from text."""
    prompt = f"""Extract factual statements from this text. Return ONLY a JSON array of objects with "subject", "predicate", "object" fields.

Rules:
- Extract concrete facts, not opinions
- Subject should be a noun (person, place, thing, project)
- Predicate should be a verb or relationship
- Object should be the value or target
- Maximum 5 facts per text
- If no clear facts, return empty array []

Text:
{text[:2000]}

JSON array:"""

    try:
        response = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=60.0
        )
        result = response.json().get('response', '').strip()

        # Try to parse JSON from response
        # Handle common issues like markdown code blocks
        if '```json' in result:
            result = result.split('```json')[1].split('```')[0]
        elif '```' in result:
            result = result.split('```')[1].split('```')[0]

        facts = json.loads(result)
        if isinstance(facts, list):
            return facts
        return []
    except Exception as e:
        print(f"  Extraction error: {e}", file=sys.stderr)
        return []

def store_facts(conn, facts: list, source_embedding_id: str):
    """Store extracted facts in database."""
    cur = conn.cursor()
    stored = 0
    for fact in facts:
        if all(k in fact for k in ['subject', 'predicate', 'object']):
            try:
                cur.execute("""
                    INSERT INTO facts (subject, predicate, object, confidence, source_embedding_id)
                    VALUES (%s, %s, %s, 0.8, %s)
                    ON CONFLICT DO NOTHING
                """, (
                    str(fact['subject'])[:500],
                    str(fact['predicate'])[:500],
                    str(fact['object'])[:500],
                    source_embedding_id
                ))
                stored += 1
            except Exception as e:
                print(f"  Store error: {e}", file=sys.stderr)
    conn.commit()
    return stored

def main():
    print(f"=== Fact Extraction Worker Started ===")
    print(f"Model: {MODEL}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Press Ctrl+C to stop\n")

    conn = psycopg2.connect(**DB_CONFIG)

    total_processed = 0
    total_facts = 0
    start_time = datetime.now()

    try:
        while True:
            vectors = get_unprocessed_vectors(conn, BATCH_SIZE)

            if not vectors:
                print("No more unprocessed vectors. Sleeping 60s...")
                time.sleep(60)
                continue

            for vec_id, content, content_type in vectors:
                print(f"Processing {vec_id[:8]}... ({content_type})", end=" ")

                facts = extract_facts_with_ollama(content)

                if facts:
                    stored = store_facts(conn, facts, str(vec_id))
                    total_facts += stored
                    print(f"→ {stored} facts")
                else:
                    # Mark as processed even if no facts
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT INTO facts (subject, predicate, object, confidence, source_embedding_id)
                        VALUES ('_no_facts_', '_extracted_', '_empty_', 0.0, %s)
                        ON CONFLICT DO NOTHING
                    """, (str(vec_id),))
                    conn.commit()
                    print("→ no facts")

                total_processed += 1

            # Progress report
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            rate = total_processed / elapsed if elapsed > 0 else 0
            print(f"\n--- Progress: {total_processed} vectors, {total_facts} facts, {rate:.1f}/min ---\n")

            time.sleep(SLEEP_BETWEEN_BATCHES)

    except KeyboardInterrupt:
        print(f"\n\n=== Stopped ===")
        print(f"Processed: {total_processed} vectors")
        print(f"Extracted: {total_facts} facts")
        print(f"Runtime: {(datetime.now() - start_time).total_seconds() / 60:.1f} minutes")
    finally:
        conn.close()

if __name__ == "__main__":
    main()