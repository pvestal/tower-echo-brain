#!/usr/bin/env python3
"""
Check the status of the reembedding process
"""

import asyncio
import json
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

async def check_status():
    """Check and display reembedding status"""
    # Import here to avoid loading everything
    from scripts.reembed_to_768 import ReembeddingProcessor

    processor = ReembeddingProcessor()
    conn = await processor.connect_postgres()

    try:
        await processor.create_progress_table(conn)
        status = await processor.get_status(conn)

        # Check if service is running
        import subprocess
        result = subprocess.run(
            ['systemctl', 'is-active', 'tower-reembedding'],
            capture_output=True,
            text=True
        )
        service_active = result.stdout.strip() == 'active'

        # Get Qdrant collection status
        from qdrant_client import QdrantClient
        qdrant = QdrantClient(host="localhost", port=6333)
        try:
            collection_info = qdrant.get_collection("echo_memory_768")
            qdrant_count = collection_info.points_count
        except:
            qdrant_count = 0

        # Calculate progress percentage
        total = status['total']
        completed = status['completed']
        failed = status['failed']
        pending = status['pending']
        progress_pct = (completed / total * 100) if total > 0 else 0

        # Calculate ETA if processing
        eta_str = "N/A"
        rate_str = "N/A"
        if service_active and status['avg_processing_time_ms'] > 0:
            avg_seconds = status['avg_processing_time_ms'] / 1000
            remaining_seconds = pending * avg_seconds
            eta = datetime.now() + timedelta(seconds=remaining_seconds)
            eta_str = eta.strftime('%Y-%m-%d %H:%M:%S')
            if completed > 0 and status['last_processed']:
                # Calculate actual rate
                last_processed = datetime.fromisoformat(status['last_processed'])
                first_processed = datetime.fromisoformat(status['first_processed'])
                elapsed = (last_processed - first_processed).total_seconds()
                if elapsed > 0:
                    rate = completed / elapsed
                    rate_str = f"{rate:.1f} records/sec"

        # Display status
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║              REEMBEDDING STATUS REPORT                      ║
╚══════════════════════════════════════════════════════════════╝

Service Status:  {'🟢 RUNNING' if service_active else '🔴 STOPPED'}
Progress:        {completed:,} / {total:,} ({progress_pct:.1f}%)

█{'█' * int(progress_pct // 2)}{'░' * (50 - int(progress_pct // 2))}█ {progress_pct:.1f}%

Statistics:
  ├─ Completed:    {completed:,}
  ├─ Failed:       {failed:,}
  ├─ Pending:      {pending:,}
  └─ Qdrant Count: {qdrant_count:,}

Performance:
  ├─ Avg Time:     {status['avg_processing_time_ms']:.0f}ms per record
  ├─ Rate:         {rate_str}
  └─ ETA:          {eta_str}

Last Processed:  {status['last_processed'] or 'Never'}

Commands:
  Start:   sudo systemctl start tower-reembedding
  Stop:    sudo systemctl stop tower-reembedding
  Logs:    sudo journalctl -u tower-reembedding -f
  Status:  python3 /opt/tower-echo-brain/scripts/check_reembedding_status.py
""")

        # Return status for programmatic use
        return {
            'service_active': service_active,
            'progress_percent': progress_pct,
            'completed': completed,
            'total': total,
            'eta': eta_str
        }

    finally:
        await conn.close()

if __name__ == "__main__":
    status = asyncio.run(check_status())