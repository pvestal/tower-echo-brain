#!/usr/bin/env python3
"""Monitor ingestion progress"""

import httpx
import time
import psutil

def check_progress():
    # Check Qdrant vector count
    try:
        with httpx.Client() as client:
            response = client.get("http://localhost:6333/collections/echo_memory")
            if response.status_code == 200:
                data = response.json()
                vector_count = data.get('result', {}).get('points_count', 0)
                print(f"Vectors in Qdrant: {vector_count:,}")
    except:
        print("Qdrant not responding")

    # Check process memory
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        if 'ingest_everything' in proc.info['name'] or \
           (proc.info['name'] == 'python' and proc.info['pid'] == 228190):
            mem_mb = proc.info['memory_info'].rss / 1024 / 1024
            print(f"Ingestion process memory: {mem_mb:.1f} MB")
            break

if __name__ == "__main__":
    while True:
        check_progress()
        print("-" * 30)
        time.sleep(5)