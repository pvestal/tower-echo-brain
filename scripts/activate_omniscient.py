#!/usr/bin/env python3
"""
Activation script for Echo Brain Omniscient capabilities
Integrates Wyze cameras and conversation training for comprehensive awareness.
"""

import asyncio
import sys
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add src to Python path
sys.path.insert(0, '/opt/tower-echo-brain/src')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from src.omniscient_pipeline import create_omniscient_pipeline, OmniscientPipeline
from training.conversation_extractor import ConversationExtractor, create_conversation_extractor
from integrations.wyze_camera import create_wyze_integration

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/opt/tower-echo-brain/logs/omniscient.log')
    ]
)

logger = logging.getLogger(__name__)

class OmniscientActivator:
    """Handles activation and management of omniscient capabilities"""

    def __init__(self):
        self.pipeline: OmniscientPipeline = None
        self.echo_brain = None

    async def initialize_echo_brain(self):
        """Initialize Echo Brain connection"""
        try:
            # Try to import and initialize Echo Brain
            try:
                from app_factory import create_app
                app = create_app()
                # Get Echo Brain instance from app context
                with app.app_context():
                    from startup import get_echo_brain_instance
                    self.echo_brain = get_echo_brain_instance()
                logger.info("✅ Connected to existing Echo Brain instance")
            except Exception as e:
                logger.warning(f"⚠️ Could not connect to Echo Brain instance: {e}")
                self.echo_brain = None

        except Exception as e:
            logger.error(f"❌ Error initializing Echo Brain: {e}")

    async def setup_dependencies(self):
        """Setup required dependencies for omniscient capabilities"""
        try:
            logger.info("🔄 Setting up dependencies...")

            # Install required Python packages
            required_packages = [
                "opencv-python",
                "face-recognition",
                "numpy",
                "scikit-learn",
                "pandas"
            ]

            import subprocess
            for package in required_packages:
                try:
                    __import__(package.replace('-', '_'))
                    logger.info(f"✅ {package} already installed")
                except ImportError:
                    logger.info(f"🔄 Installing {package}...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

            # Setup model directories
            model_dir = Path("/opt/tower-echo-brain/models")
            model_dir.mkdir(exist_ok=True)

            # Download YOLO models if not present
            await self._download_models()

            logger.info("✅ Dependencies setup complete")

        except Exception as e:
            logger.error(f"❌ Error setting up dependencies: {e}")
            raise

    async def _download_models(self):
        """Download required AI models"""
        try:
            import requests
            model_dir = Path("/opt/tower-echo-brain/models")

            models = {
                "yolov4.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
                "coco.names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
            }

            for filename, url in models.items():
                filepath = model_dir / filename
                if not filepath.exists():
                    logger.info(f"🔄 Downloading {filename}...")
                    response = requests.get(url)
                    response.raise_for_status()
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"✅ Downloaded {filename}")

            # Note about YOLO weights
            weights_path = model_dir / "yolov4.weights"
            if not weights_path.exists():
                logger.warning("⚠️ YOLOv4 weights not found. Download manually from:")
                logger.warning("   https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights")
                logger.warning(f"   Save to: {weights_path}")

        except Exception as e:
            logger.warning(f"⚠️ Could not download models: {e}")

    async def test_camera_connections(self):
        """Test camera connections before full activation"""
        try:
            logger.info("🔄 Testing camera connections...")

            # Create temporary camera integration for testing
            camera_integration = create_wyze_integration()

            # Test each camera from config
            from integrations.wyze_camera import load_camera_config
            cameras = load_camera_config()

            working_cameras = 0
            for camera in cameras:
                try:
                    # Simple ping test
                    import subprocess
                    result = subprocess.run(
                        ["ping", "-c", "1", camera.ip_address],
                        capture_output=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        logger.info(f"✅ Camera {camera.name} reachable at {camera.ip_address}")
                        working_cameras += 1
                    else:
                        logger.warning(f"⚠️ Camera {camera.name} not reachable at {camera.ip_address}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not test camera {camera.name}: {e}")

            logger.info(f"🎯 {working_cameras}/{len(cameras)} cameras reachable")

        except Exception as e:
            logger.error(f"❌ Error testing camera connections: {e}")

    async def test_conversation_extraction(self):
        """Test conversation data extraction"""
        try:
            logger.info("🔄 Testing conversation extraction...")

            db_config = {
                "host": "192.168.50.135",
                "user": "patrick",
                "password": os.getenv("TOWER_DB_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE"),
                "database": "echo_brain",
                "port": 5432
            }

            extractor = create_conversation_extractor(db_config)

            # Test extraction from last 24 hours
            start_date = datetime.now() - timedelta(hours=24)
            dataset = await extractor.extract_all_conversations(start_date)

            logger.info(f"✅ Extracted {dataset.total_conversations} conversations")
            logger.info(f"   Sources: {', '.join(dataset.sources)}")
            logger.info(f"   Quality distribution: {dataset.quality_distribution}")

        except Exception as e:
            logger.warning(f"⚠️ Conversation extraction test failed: {e}")

    async def activate_omniscient(self):
        """Activate full omniscient capabilities"""
        try:
            logger.info("🚀 Activating Echo Brain Omniscient capabilities...")

            # Create and initialize pipeline
            self.pipeline = await create_omniscient_pipeline(
                self.echo_brain,
                "/opt/tower-echo-brain/config"
            )

            # Add custom learning callbacks
            self.pipeline.add_learning_callback(self._on_learning_event)

            # Start the pipeline
            await self.pipeline.start()

            logger.info("🧠 Omniscient Pipeline fully activated")
            logger.info("🎯 Echo Brain now has comprehensive awareness capabilities:")
            logger.info("   ✅ Live camera monitoring with facial recognition")
            logger.info("   ✅ Real-time motion and object detection")
            logger.info("   ✅ Scene analysis and environmental awareness")
            logger.info("   ✅ Conversation data learning and training")
            logger.info("   ✅ Behavioral pattern detection")
            logger.info("   ✅ Continuous learning from all data sources")

            return True

        except Exception as e:
            logger.error(f"❌ Error activating omniscient capabilities: {e}")
            return False

    async def _on_learning_event(self, context):
        """Handle learning events from the omniscient pipeline"""
        try:
            insights = context.learning_insights
            if insights.get("improvement_suggestions"):
                logger.info(f"📈 Learning insight: {insights['improvement_suggestions'][0]}")

            # Log significant behavioral patterns
            patterns = context.behavioral_patterns.get("routines", [])
            for pattern in patterns:
                if pattern.get("confidence", 0) > 0.8:
                    logger.info(f"🔄 High confidence pattern detected: {pattern}")

        except Exception as e:
            logger.warning(f"⚠️ Error processing learning event: {e}")

    async def monitor_system(self):
        """Monitor system health and performance"""
        try:
            while self.pipeline and self.pipeline.running:
                # Get current awareness state
                awareness = self.pipeline.get_current_awareness()

                logger.info(f"🧠 System Status: {awareness.get('status', 'active')}")
                logger.info(f"   Active cameras: {awareness.get('active_cameras', 0)}")
                logger.info(f"   Recent detections: {awareness.get('recent_detections', 0)}")
                logger.info(f"   Behavior patterns: {awareness.get('behavior_patterns', 0)}")

                await asyncio.sleep(300)  # Check every 5 minutes

        except Exception as e:
            logger.error(f"❌ Error monitoring system: {e}")

    async def export_training_data(self, output_dir: str = "/opt/tower-echo-brain/data/training"):
        """Export training data for manual review and backup"""
        try:
            logger.info("📤 Exporting training data...")

            if not self.pipeline:
                logger.error("❌ Pipeline not initialized")
                return

            # Export conversation dataset
            extractor = self.pipeline.conversation_extractor
            if extractor:
                start_date = datetime.now() - timedelta(days=30)  # Last 30 days
                dataset = await extractor.extract_all_conversations(start_date)
                await extractor.export_dataset(dataset, f"{output_dir}/conversations")

            # Export camera detection data
            # This would export detection events and scene analysis
            # Implementation depends on storage format

            logger.info(f"✅ Training data exported to {output_dir}")

        except Exception as e:
            logger.error(f"❌ Error exporting training data: {e}")

async def main():
    """Main activation function"""
    activator = OmniscientActivator()

    try:
        print("🧠 Echo Brain Omniscient Capability Activation")
        print("=" * 50)

        # Initialize Echo Brain connection
        await activator.initialize_echo_brain()

        # Setup dependencies
        await activator.setup_dependencies()

        # Test systems
        await activator.test_camera_connections()
        await activator.test_conversation_extraction()

        # Activate omniscient capabilities
        success = await activator.activate_omniscient()

        if success:
            print("\n🎉 OMNISCIENT CAPABILITIES ACTIVATED SUCCESSFULLY!")
            print("\nEcho Brain now has:")
            print("🎥 Live camera monitoring and analysis")
            print("🧠 Continuous learning from conversations")
            print("🔍 Behavioral pattern detection")
            print("🌍 Environmental awareness")
            print("📊 Real-time training data integration")

            # Monitor system
            print("\n🔄 Starting system monitoring...")
            await activator.monitor_system()

        else:
            print("\n❌ ACTIVATION FAILED")
            print("Check logs for details: /opt/tower-echo-brain/logs/omniscient.log")

    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        if activator.pipeline:
            await activator.pipeline.stop()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.error(f"❌ Unexpected error in main: {e}")

if __name__ == "__main__":
    asyncio.run(main())