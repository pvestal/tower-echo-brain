#!/usr/bin/env python3
"""
Autonomous Video Generation Workflow for Echo Brain
Self-contained script that Echo can execute independently for anime video generation
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append('/opt/tower-echo-brain/src')

from tasks.task_queue import TaskQueue, Task, TaskType, TaskPriority, TaskStatus
from tasks.video_generation_tasks import (
    VideoGenerationExecutor,
    create_character_to_video_task,
    create_batch_generation_task,
    VideoTaskType
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/tower-echo-brain/logs/autonomous_video.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutonomousVideoOrchestrator:
    """Autonomous video generation orchestrator for Echo Brain"""

    def __init__(self):
        self.task_queue = TaskQueue()
        self.video_executor = VideoGenerationExecutor()
        self.running = False
        self.db_config = {
            'host': 'localhost',
            'database': 'echo_brain',
            'user': 'patrick',
            'password': os.getenv('DB_PASSWORD', 'tower_echo_brain_secret_key_2025')
        }

    async def initialize(self):
        """Initialize the autonomous workflow system"""
        try:
            await self.task_queue.initialize()
            logger.info("🤖 Autonomous Video Orchestrator initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            return False

    async def create_video_from_character(self,
                                       character_image_path: str,
                                       prompt: str = None,
                                       priority: TaskPriority = TaskPriority.HIGH) -> str:
        """Create autonomous video generation task"""
        try:
            # Validate input
            if not os.path.exists(character_image_path):
                raise FileNotFoundError(f"Character image not found: {character_image_path}")

            # Create task
            task = create_character_to_video_task(
                input_image_path=character_image_path,
                prompt=prompt,
                priority=priority
            )

            # Add to queue
            success = await self.task_queue.add_task(task)
            if success:
                logger.info(f"✅ Created autonomous video task: {task.id}")
                return task.id
            else:
                raise Exception("Failed to add task to queue")

        except Exception as e:
            logger.error(f"Failed to create video task: {e}")
            raise

    async def process_video_queue(self, max_tasks: int = 5):
        """Process video generation tasks from the queue"""
        try:
            processed = 0

            while processed < max_tasks:
                # Get next task
                task = await self.task_queue.get_next_task()
                if not task:
                    logger.info("No pending video tasks found")
                    break

                # Check if it's a video task
                if task.payload.get('video_task_type'):
                    logger.info(f"🎬 Processing video task: {task.name}")

                    # Update status to running
                    await self.task_queue.update_task_status(task.id, TaskStatus.RUNNING)

                    # Execute video generation
                    result = await self._execute_video_task(task)

                    # Update final status
                    if result.get('status') == 'completed':
                        await self.task_queue.update_task_status(
                            task.id, TaskStatus.COMPLETED, result
                        )
                        logger.info(f"✅ Video task completed: {task.id}")
                    else:
                        await self.task_queue.update_task_status(
                            task.id, TaskStatus.FAILED, error=result.get('error')
                        )
                        logger.error(f"❌ Video task failed: {task.id} - {result.get('error')}")

                    processed += 1
                else:
                    # Not a video task, skip
                    logger.debug(f"Skipping non-video task: {task.name}")
                    break

            logger.info(f"Processed {processed} video tasks")
            return processed

        except Exception as e:
            logger.error(f"Error processing video queue: {e}")
            return 0

    async def _execute_video_task(self, task: Task) -> dict:
        """Execute individual video generation task"""
        try:
            video_task_type = task.payload.get('video_task_type')

            if video_task_type == VideoTaskType.CHARACTER_TO_VIDEO.value:
                return await self.video_executor.execute_character_to_video(task)
            elif video_task_type == VideoTaskType.BATCH_GENERATION.value:
                return await self.video_executor.execute_batch_generation(task)
            else:
                return {
                    'status': 'failed',
                    'error': f"Unknown video task type: {video_task_type}"
                }

        except Exception as e:
            logger.error(f"Video task execution failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    async def run_continuous_processing(self, interval: int = 30):
        """Run continuous video processing loop"""
        logger.info(f"🔄 Starting continuous video processing (interval: {interval}s)")
        self.running = True

        try:
            while self.running:
                processed = await self.process_video_queue()

                if processed > 0:
                    logger.info(f"Processed {processed} video tasks in this cycle")

                # Wait before next cycle
                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping...")
        except Exception as e:
            logger.error(f"Error in continuous processing: {e}")
        finally:
            self.running = False
            logger.info("Continuous processing stopped")

    def stop_processing(self):
        """Stop continuous processing"""
        self.running = False

    async def get_queue_status(self) -> dict:
        """Get current queue status and statistics"""
        try:
            stats = await self.task_queue.get_task_stats()
            return {
                'status': 'running' if self.running else 'stopped',
                'queue_stats': stats,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get queue status: {e}")
            return {'error': str(e)}

    async def batch_process_directory(self, input_directory: str,
                                    pattern: str = "*.png") -> list:
        """Create batch tasks for all images in a directory"""
        try:
            input_path = Path(input_directory)
            if not input_path.exists():
                raise FileNotFoundError(f"Directory not found: {input_directory}")

            # Find matching images
            image_files = list(input_path.glob(pattern))
            if not image_files:
                logger.warning(f"No images found matching pattern: {pattern}")
                return []

            # Create batch task
            image_paths = [str(f) for f in image_files]
            task = create_batch_generation_task(
                input_images=image_paths,
                batch_name=f"Batch: {input_path.name}"
            )

            success = await self.task_queue.add_task(task)
            if success:
                logger.info(f"✅ Created batch task for {len(image_files)} images: {task.id}")
                return [task.id]
            else:
                raise Exception("Failed to add batch task to queue")

        except Exception as e:
            logger.error(f"Failed to create batch task: {e}")
            raise

async def main():
    """Main autonomous video generation workflow"""
    orchestrator = AutonomousVideoOrchestrator()

    if not await orchestrator.initialize():
        logger.error("Failed to initialize orchestrator")
        return

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "process":
            # Process current queue
            processed = await orchestrator.process_video_queue()
            print(f"Processed {processed} video tasks")

        elif command == "continuous":
            # Run continuous processing
            await orchestrator.run_continuous_processing()

        elif command == "create" and len(sys.argv) > 2:
            # Create new video task
            character_path = sys.argv[2]
            prompt = sys.argv[3] if len(sys.argv) > 3 else None

            task_id = await orchestrator.create_video_from_character(character_path, prompt)
            print(f"Created task: {task_id}")

        elif command == "batch" and len(sys.argv) > 2:
            # Create batch task
            directory = sys.argv[2]
            pattern = sys.argv[3] if len(sys.argv) > 3 else "*.png"

            task_ids = await orchestrator.batch_process_directory(directory, pattern)
            print(f"Created batch tasks: {task_ids}")

        elif command == "status":
            # Get status
            status = await orchestrator.get_queue_status()
            print(json.dumps(status, indent=2))

        elif command == "test":
            # Test with cyberpunk character
            cyberpunk_path = "/mnt/1TB-storage/ComfyUI/output/cyberpunk_goblin_slayer_crawl_00001_.png"
            if os.path.exists(cyberpunk_path):
                task_id = await orchestrator.create_video_from_character(
                    cyberpunk_path,
                    "cyberpunk goblin slayer, heavily armored knight, dynamic action sequence, anime style"
                )
                print(f"Created test task: {task_id}")

                # Process immediately
                processed = await orchestrator.process_video_queue(max_tasks=1)
                print(f"Processed: {processed} tasks")
            else:
                print(f"Test image not found: {cyberpunk_path}")

        else:
            print("Unknown command. Available commands:")
            print("  process - Process current queue")
            print("  continuous - Run continuous processing")
            print("  create <image_path> [prompt] - Create single video task")
            print("  batch <directory> [pattern] - Create batch task")
            print("  status - Get queue status")
            print("  test - Test with cyberpunk character")

    else:
        # Default: process current queue
        processed = await orchestrator.process_video_queue()
        print(f"Processed {processed} video tasks")

if __name__ == "__main__":
    asyncio.run(main())