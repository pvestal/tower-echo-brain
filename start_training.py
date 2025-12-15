#!/usr/bin/env python3
"""
Start Echo Brain Training Pipeline
Launches background training processes for continuous learning
"""

import asyncio
import logging
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, '/opt/tower-echo-brain')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/tower-echo-brain/retraining/logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def start_learning_pipeline():
    """Initialize and start the learning pipeline"""
    logger.info("🚀 Starting Echo Brain Learning Pipeline")

    # Import learning components
    from src.core.echo.echo_learning_system import EchoLearningSystem
    from src.core.echo.echo_retraining_pipeline import RetrainingPipeline

    try:
        # Initialize learning system
        learning_system = EchoLearningSystem()
        logger.info("✅ Learning system initialized")

        # Get current learning status
        status = learning_system.get_learning_status()
        logger.info(f"📊 Learning Status: {json.dumps(status, indent=2)}")

        # Initialize retraining pipeline
        pipeline = RetrainingPipeline()
        await pipeline.initialize()
        logger.info("✅ Retraining pipeline initialized")

        # Start scheduled retraining job
        logger.info("🔄 Starting scheduled retraining...")
        job = await pipeline.trigger_retraining(
            model_name="echo_decision_engine",
            trigger_reason="Initial training activation",
            config={
                "model_type": "ensemble",
                "training_config": {
                    "epochs": 10,
                    "batch_size": 32,
                    "learning_rate": 0.001
                },
                "validation_split": 0.2
            }
        )
        logger.info(f"📋 Retraining job created: {job.job_id}")

        # Start continuous learning loop
        logger.info("♾️ Starting continuous learning loop...")
        while True:
            try:
                # Check for new learning data
                metrics = learning_system.learning_metrics
                logger.info(f"📈 Learning metrics: Patterns={len(learning_system.active_patterns)}, "
                          f"History={len(learning_system.decision_history)}")

                # Process any pending jobs
                pending_jobs = await pipeline.get_pending_jobs()
                if pending_jobs:
                    logger.info(f"📦 Processing {len(pending_jobs)} pending jobs")
                    for job in pending_jobs:
                        await pipeline.process_job(job)

                # Sleep for 5 minutes before next check
                await asyncio.sleep(300)

            except Exception as e:
                logger.error(f"❌ Error in learning loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

    except Exception as e:
        logger.error(f"❌ Failed to start learning pipeline: {e}")
        return False

    return True

async def start_model_training():
    """Start model training processes"""
    logger.info("🧠 Starting model training processes")

    try:
        # Import MLOps components
        from src.core.echo.echo_mlops_integration import MLOpsIntegration
        from src.core.echo.echo_drift_detector import DriftDetector

        # Initialize MLOps
        mlops = MLOpsIntegration()
        await mlops.initialize()
        logger.info("✅ MLOps integration initialized")

        # Initialize drift detection
        drift_detector = DriftDetector()
        logger.info("✅ Drift detector initialized")

        # Start monitoring loop
        logger.info("📊 Starting model monitoring...")
        while True:
            try:
                # Check for drift
                drift_status = await drift_detector.check_drift()
                if drift_status.get('drift_detected'):
                    logger.warning(f"⚠️ Drift detected: {drift_status}")
                    # Trigger retraining on drift

                # Check model performance
                performance = await mlops.get_model_performance()
                logger.info(f"📈 Model performance: {performance}")

                # Sleep for 10 minutes
                await asyncio.sleep(600)

            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                await asyncio.sleep(60)

    except Exception as e:
        logger.error(f"❌ Failed to start model training: {e}")
        return False

    return True

async def main():
    """Main entry point"""
    logger.info("="*60)
    logger.info("🧠 ECHO BRAIN TRAINING SYSTEM")
    logger.info("="*60)
    logger.info(f"Started at: {datetime.now().isoformat()}")

    # Create tasks for parallel execution
    tasks = [
        asyncio.create_task(start_learning_pipeline()),
        asyncio.create_task(start_model_training())
    ]

    # Run all tasks
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"✅ Training systems started: {results}")
    except Exception as e:
        logger.error(f"❌ Failed to start training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⏹️ Training stopped by user")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        sys.exit(1)