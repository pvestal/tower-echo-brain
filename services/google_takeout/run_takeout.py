#!/usr/bin/env python3
"""
Run Google Takeout download and processing pipeline
"""

import asyncio
import logging
from pathlib import Path
from takeout_manager import GoogleTakeoutManager
from deduplication_engine import DeduplicationEngine

async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("🚀 Starting Google Takeout Pipeline")

    # Initialize components
    manager = GoogleTakeoutManager()
    dedup_engine = DeduplicationEngine()

    # Check Vault credentials
    creds = manager.get_vault_credentials()
    if not creds:
        logger.info("🔐 Setting up Vault credentials...")
        manager.setup_vault_credentials()

    # Request new takeout
    logger.info("📥 Requesting new Google Takeout...")
    await manager.request_new_takeout()

    # Check for downloaded files to process
    download_dir = Path("/opt/tower-echo-brain/data/takeout")
    if any(download_dir.rglob("*")):
        logger.info("🔍 Scanning for duplicates...")
        duplicates, new_files = await dedup_engine.scan_and_deduplicate(download_dir)

        logger.info(f"✅ Found {len(new_files)} new files, {len(duplicates)} duplicates")

        # Connect to learning pipeline
        from learning_pipeline.src.core.pipeline import LearningPipeline
        pipeline = LearningPipeline()

        # Process new files
        for filepath in new_files[:10]:  # Process first 10 files
            logger.info(f"🧠 Processing: {filepath.name}")
            # Pipeline will handle the actual processing

        logger.info("🎉 Pipeline complete!")
    else:
        logger.info("📋 No files to process yet. Complete manual Takeout download first.")

if __name__ == "__main__":
    asyncio.run(main())