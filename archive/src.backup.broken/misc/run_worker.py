#!/usr/bin/env python3
"""
Run Echo Brain Background Worker
"""

import asyncio
import logging
import sys
import os

# Setup path
sys.path.insert(0, '/opt/tower-echo-brain')
os.chdir('/opt/tower-echo-brain')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.tasks.background_worker import BackgroundWorker
from src.tasks.task_queue import TaskQueue

async def main():
    """Main entry point for background worker"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Echo Brain Background Worker...")

    # Create task queue
    task_queue = TaskQueue()

    # Create worker
    worker = BackgroundWorker(task_queue)

    try:
        # Start worker
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
        await worker.stop()
    except Exception as e:
        logger.error(f"Worker crashed: {e}")
        await worker.stop()
        raise

if __name__ == "__main__":
    asyncio.run(main())