#!/usr/bin/env python3
"""
Check the status of fact extraction from vectors
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

async def check_status():
    """Check and display fact extraction status"""
    from scripts.extract_facts_from_vectors import FactExtractor

    extractor = FactExtractor()
    conn = await extractor.connect_postgres()

    try:
        # Get extraction statistics
        stats = await conn.fetchrow("""
            SELECT
                COUNT(DISTINCT source_collection) as collections,
                COUNT(*) as total_processed,
                SUM(facts_extracted) as total_facts,
                COUNT(*) FILTER (WHERE status = 'completed') as successful,
                COUNT(*) FILTER (WHERE status = 'failed') as failed,
                MAX(processed_at) as last_processed,
                MIN(processed_at) as first_processed
            FROM fact_extraction_log
        """)

        # Get per-collection stats
        collection_stats = await conn.fetch("""
            SELECT
                source_collection,
                COUNT(*) as vectors_processed,
                SUM(facts_extracted) as facts,
                ROUND(AVG(facts_extracted), 1) as avg_facts_per_vector,
                COUNT(*) FILTER (WHERE status = 'failed') as failed
            FROM fact_extraction_log
            GROUP BY source_collection
            ORDER BY vectors_processed DESC
        """)

        # Get recent extraction rate
        recent_stats = await conn.fetchrow("""
            SELECT
                COUNT(*) as recent_count,
                SUM(facts_extracted) as recent_facts
            FROM fact_extraction_log
            WHERE processed_at > NOW() - INTERVAL '1 hour'
        """)

        # Check service status
        result = subprocess.run(
            ['systemctl', 'is-active', 'tower-fact-extraction'],
            capture_output=True,
            text=True
        )
        service_active = result.stdout.strip() == 'active'

        # Check Qdrant collection status
        from qdrant_client import QdrantClient
        qdrant = QdrantClient(host="localhost", port=6333)

        echo_memory_count = 0
        try:
            collection_info = qdrant.get_collection("echo_memory_768")
            echo_memory_count = collection_info.points_count
        except:
            pass

        # Display status
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║            FACT EXTRACTION STATUS REPORT                    ║
╚══════════════════════════════════════════════════════════════╝

Service Status:  {'🟢 RUNNING' if service_active else '🔴 STOPPED'}

Overall Statistics:
  ├─ Total Vectors Processed: {stats['total_processed']:,}
  ├─ Total Facts Extracted:   {stats['total_facts']:,}
  ├─ Successful Extractions:  {stats['successful']:,}
  ├─ Failed Extractions:      {stats['failed']:,}
  ├─ Collections Processed:   {stats['collections']}
  └─ Average Facts/Vector:    {stats['total_facts']/stats['total_processed']:.1f if stats['total_processed'] > 0 else 0}

Collection Breakdown:""")

        for coll in collection_stats:
            print(f"""  ├─ {coll['source_collection']}:
  │   ├─ Vectors: {coll['vectors_processed']:,}
  │   ├─ Facts: {coll['facts']:,}
  │   ├─ Avg/Vector: {coll['avg_facts_per_vector']}
  │   └─ Failed: {coll['failed']}""")

        print(f"""
Recent Activity (last hour):
  ├─ Vectors Processed: {recent_stats['recent_count']:,}
  └─ Facts Extracted:   {recent_stats['recent_facts']:,}

echo_memory_768 Collection:
  └─ Total Vectors: {echo_memory_count:,} (source for extraction)

Timeline:
  ├─ First Extraction: {stats['first_processed'] or 'Never'}
  └─ Last Extraction:  {stats['last_processed'] or 'Never'}

Commands:
  Start:   sudo systemctl start tower-fact-extraction
  Stop:    sudo systemctl stop tower-fact-extraction
  Logs:    sudo journalctl -u tower-fact-extraction -f
  Status:  python3 /opt/tower-echo-brain/scripts/check_fact_extraction_status.py

Note: Wait for reembedding to complete before starting fact extraction!
Current reembedding status: Check with
  python3 /opt/tower-echo-brain/scripts/check_reembedding_status.py
""")

    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(check_status())