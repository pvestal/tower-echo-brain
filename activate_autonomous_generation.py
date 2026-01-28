#!/usr/bin/env python3
"""
Activate Echo Brain Autonomous Generation for Tokyo Debt Desire
This script starts the autonomous loop and adds the generation task
"""

import asyncio
import sys
import logging
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from capabilities.capability_registry import CapabilityRegistry, CapabilityType
from capabilities.autonomous_loop import AutonomousEventLoop, TaskPriority
from capabilities.image_generation import ImageGenerationCapability

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Main autonomous generation activation"""

    logger.info("🧠 Starting Echo Brain Autonomous Generation System")

    # Initialize capability registry
    registry = CapabilityRegistry()

    # Register image generation capability
    image_gen = ImageGenerationCapability()
    registry.register_capability(image_gen)

    logger.info("🎨 Registered image generation capability")

    # Initialize autonomous loop
    autonomous_loop = AutonomousEventLoop(registry)

    # Start the autonomous system
    await autonomous_loop.start()

    logger.info("🔄 Autonomous loop started")

    # Add Tokyo Debt Desire continuous generation task
    task_id = await autonomous_loop.add_task(
        name="Tokyo Debt Desire Autonomous Generation",
        description="Continuously generate photorealistic character variations for Tokyo Debt Desire project",
        priority=TaskPriority.HIGH,
        capability="image_generation",
        parameters={
            "task_type": "character_generation",
            "project": "Tokyo Debt Desire",
            "characters": ["Mei_Kobayashi", "Rina_Suzuki", "Yuki_Tanaka", "Takeshi_Sato"],
            "style": "photorealistic",
            "continuous": True
        }
    )

    logger.info(f"✅ Added Tokyo Debt Desire generation task: {task_id}")

    # Add periodic re-generation task
    await autonomous_loop.add_task(
        name="Quality Check and Regeneration",
        description="Check generated image quality and regenerate if needed",
        priority=TaskPriority.NORMAL,
        capability="image_generation",
        parameters={
            "task_type": "quality_check",
            "project": "Tokyo Debt Desire"
        }
    )

    logger.info("🔍 Added quality monitoring task")

    # Add style validation task
    await autonomous_loop.add_task(
        name="Style Validation",
        description="Validate photorealistic style and reject cartoon/anime images",
        priority=TaskPriority.NORMAL,
        capability="image_generation",
        parameters={
            "task_type": "style_validation",
            "project": "Tokyo Debt Desire"
        }
    )

    logger.info("🎯 Added style validation task")

    # Create request for continuous operation
    request_data = {
        "autonomous_goal": {
            "project": "Tokyo Debt Desire",
            "objective": "Generate continuous photorealistic character variations",
            "constraints": {
                "style": "ultra_realistic_photography",
                "reject_styles": ["cartoon", "anime", "illustration"],
                "model": "realisticVision",
                "quality_threshold": "high"
            },
            "schedule": {
                "frequency": "continuous",
                "batch_size": 4,
                "variations_per_character": 2
            }
        }
    }

    # Save autonomous goal to file for persistence
    goal_file = Path("/tmp/echo_brain_autonomous_goal.json")
    goal_file.write_text(json.dumps(request_data, indent=2))

    logger.info(f"💾 Saved autonomous goal to {goal_file}")

    # Show status
    status = autonomous_loop.get_status()
    logger.info("📊 System Status:")
    logger.info(f"  - Running: {status['is_running']}")
    logger.info(f"  - Pending tasks: {status['pending_tasks']}")
    logger.info(f"  - Running tasks: {status['running_tasks']}")

    # Keep running
    logger.info("🚀 Echo Brain Autonomous Generation is now ACTIVE")
    logger.info("📷 Generating photorealistic Tokyo Debt Desire characters...")
    logger.info("🔄 System will continuously generate and monitor image quality")

    try:
        # Run for a while to let it work
        await asyncio.sleep(300)  # Run for 5 minutes initially

        # Show final status
        final_status = autonomous_loop.get_status()
        logger.info("📈 Final Status:")
        logger.info(f"  - Completed tasks: {final_status['completed_tasks']}")
        logger.info(f"  - Recent completions: {len(final_status['tasks']['recent_completed'])}")

    except KeyboardInterrupt:
        logger.info("🛑 Stopping autonomous generation...")
        await autonomous_loop.stop()

if __name__ == "__main__":
    asyncio.run(main())