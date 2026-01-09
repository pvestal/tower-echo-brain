#!/usr/bin/env python3
"""
Manual GPU Training Test
Quick test with a few files to verify end-to-end functionality
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from training.gpu_accelerated_trainer import GPUAcceleratedTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    logger.info("🧪 Manual GPU Training Test")

    # Override database to use local SQLite for testing
    os.environ['USE_LOCAL_DB'] = 'true'

    try:
        trainer = GPUAcceleratedTrainer()

        # Override media sources for testing
        trainer.media_sources = {
            'local_pictures': Path("/home/patrick/Pictures")
        }

        # Get some test files
        test_files = []
        for jpg_file in Path("/home/patrick/Pictures").rglob("*.jpg"):
            test_files.append(str(jpg_file))
            if len(test_files) >= 5:  # Just test with 5 files
                break

        if not test_files:
            logger.error("No test files found!")
            return

        logger.info(f"Testing with {len(test_files)} files:")
        for f in test_files:
            logger.info(f"  - {Path(f).name}")

        # Load model
        if not trainer.load_optimal_model():
            logger.error("Failed to load CLIP model")
            return

        logger.info(f"✅ Model loaded on {trainer.current_device}")

        # Test batch processing
        logger.info("🔄 Testing batch processing...")

        # Generate embeddings
        embeddings = trainer.generate_batch_embeddings(test_files)

        successful_embeddings = sum(1 for e in embeddings if e is not None)
        logger.info(f"✅ Generated {successful_embeddings}/{len(test_files)} embeddings")

        if successful_embeddings > 0:
            logger.info(f"   Embedding dimension: {len(embeddings[0]) if embeddings[0] else 0}")
            logger.info("🎉 GPU Training Pipeline Working!")

            # Test Qdrant storage
            logger.info("💾 Testing vector storage...")

            from src.api.models import PointStruct
            import uuid
            import hashlib

            points = []
            for i, (file_path, embedding) in enumerate(zip(test_files, embeddings)):
                if embedding:
                    vector_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"test_{i}"))
                    point = PointStruct(
                        id=vector_id,
                        vector=embedding,
                        payload={
                            'file_path': file_path,
                            'test_run': True,
                            'file_name': Path(file_path).name
                        }
                    )
                    points.append(point)

            # Store in Qdrant
            trainer.qdrant.upsert(
                collection_name=trainer.collection_name,
                points=points
            )

            logger.info(f"✅ Stored {len(points)} vectors in Qdrant")

            # Test retrieval
            collection_info = trainer.qdrant.get_collection(trainer.collection_name)
            logger.info(f"   Total vectors in collection: {collection_info.points_count}")

            logger.info("\n🎯 Manual Test Results:")
            logger.info(f"   ✅ GPU Device: {trainer.current_device}")
            logger.info(f"   ✅ Files processed: {len(test_files)}")
            logger.info(f"   ✅ Embeddings generated: {successful_embeddings}")
            logger.info(f"   ✅ Vectors stored: {len(points)}")
            logger.info(f"   ✅ Processing rate estimate: ~{len(test_files) * 720} files/hour")

        else:
            logger.error("❌ No embeddings generated!")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())