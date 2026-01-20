"""
Example Usage of Echo Brain Autonomous Core

This file demonstrates how to use the autonomous system components
for creating goals, scheduling tasks, and running the main loop.
"""

import asyncio
import logging
from datetime import datetime, timedelta

# Import autonomous components
from . import (
    AutonomousCore, GoalManager, Scheduler, Executor,
    AutonomousState, TaskResult
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def example_basic_usage():
    """Demonstrates basic usage of autonomous components."""
    logger.info("=== Echo Brain Autonomous Core - Basic Usage Example ===")

    # Initialize components
    goal_manager = GoalManager()
    scheduler = Scheduler()
    executor = Executor()

    try:
        # 1. Create a goal
        logger.info("Creating a new goal...")
        goal_id = await goal_manager.create_goal(
            name="Learn about new conversation patterns",
            description="Analyze recent conversations to identify new patterns and insights",
            goal_type="analysis",
            priority=3,
            metadata={
                "data_source": "conversations",
                "analysis_type": "pattern_recognition"
            }
        )

        if goal_id:
            logger.info(f"Created goal with ID: {goal_id}")

            # 2. Schedule some tasks for the goal
            logger.info("Scheduling tasks...")

            task1_id = await scheduler.schedule_task(
                goal_id=goal_id,
                name="Analyze conversation sentiment",
                task_type="analysis",
                priority=2,
                metadata={
                    "analysis_type": "sentiment",
                    "data_range": "last_7_days"
                }
            )

            task2_id = await scheduler.schedule_task(
                goal_id=goal_id,
                name="Extract key topics from conversations",
                task_type="reasoning",
                priority=3,
                metadata={
                    "analysis_type": "topic_extraction",
                    "method": "semantic_analysis"
                }
            )

            task3_id = await scheduler.schedule_task(
                goal_id=goal_id,
                name="Generate summary report",
                task_type="coding",
                priority=4,
                scheduled_at=datetime.now() + timedelta(minutes=2),
                metadata={
                    "output_format": "markdown",
                    "include_charts": True
                }
            )

            logger.info(f"Scheduled tasks: {task1_id}, {task2_id}, {task3_id}")

            # 3. Execute tasks
            logger.info("Executing scheduled tasks...")

            for _ in range(3):  # Try to execute up to 3 tasks
                task = await scheduler.get_next_task()
                if task:
                    logger.info(f"Executing task {task.id}: {task.name}")
                    scheduler.record_task_execution()

                    result = await executor.execute(task.id)
                    logger.info(f"Task {task.id} result: success={result.success}")

                    if not result.success:
                        logger.error(f"Task {task.id} failed: {result.error}")
                else:
                    logger.info("No more tasks available")
                    break

            # 4. Check goal progress
            logger.info("Checking goal progress...")
            await goal_manager.update_goal_progress(goal_id)
            goals = await goal_manager.get_goals(goal_id=goal_id)
            if goals:
                goal = goals[0]
                logger.info(f"Goal '{goal['name']}' progress: {goal['progress_percent']:.1f}%")

        # 5. Get system status
        queue_status = await scheduler.get_queue_status()
        logger.info(f"Queue status: {queue_status}")

        # 6. Get agent capabilities
        capabilities = await executor.get_agent_capabilities()
        logger.info(f"Agent capabilities: {capabilities}")

    except Exception as e:
        logger.error(f"Error in basic usage example: {e}")
    finally:
        # Cleanup
        await goal_manager.cleanup() if hasattr(goal_manager, 'cleanup') else None
        await scheduler.cleanup()
        await executor.cleanup()


async def example_full_system():
    """Demonstrates the full autonomous system with AutonomousCore."""
    logger.info("=== Echo Brain Autonomous Core - Full System Example ===")

    # Configuration for the autonomous core
    config = {
        'cycle_interval': 10,  # Run cycle every 10 seconds
        'max_concurrent_tasks': 2,
        'max_tasks_per_minute': 5,
        'check_kill_switch': False  # Disable for demo
    }

    # Initialize the autonomous core
    core = AutonomousCore(config)

    try:
        # Start the autonomous system
        logger.info("Starting autonomous core...")
        success = await core.start()

        if not success:
            logger.error("Failed to start autonomous core")
            return

        # Create a test goal
        logger.info("Creating test goal...")
        goal_id = await core.create_goal(
            name="Test autonomous operations",
            description="A test goal to demonstrate autonomous task execution",
            goal_type="testing",
            priority=1,
            metadata={"demo": True, "auto_created": True}
        )

        if goal_id:
            # Schedule some test tasks
            await core.scheduler.schedule_task(
                goal_id=goal_id,
                name="Perform system health check",
                task_type="analysis",
                priority=1
            )

            await core.scheduler.schedule_task(
                goal_id=goal_id,
                name="Generate system status report",
                task_type="reasoning",
                priority=2
            )

        # Let it run for a bit
        logger.info("Running autonomous system for 30 seconds...")
        await asyncio.sleep(30)

        # Get system status
        status = await core.get_status()
        logger.info(f"System status: {status.state.value}")
        logger.info(f"Cycles completed: {status.cycles_completed}")
        logger.info(f"Tasks executed: {status.tasks_executed}")
        logger.info(f"Uptime: {status.uptime_seconds:.1f} seconds")

        # Get performance metrics
        metrics = core.get_performance_metrics()
        if metrics:
            logger.info(f"Performance metrics: {metrics}")

    except Exception as e:
        logger.error(f"Error in full system example: {e}")
    finally:
        # Stop the autonomous system
        logger.info("Stopping autonomous core...")
        await core.stop()


async def example_recurring_tasks():
    """Demonstrates recurring task scheduling."""
    logger.info("=== Recurring Tasks Example ===")

    goal_manager = GoalManager()
    scheduler = Scheduler()

    try:
        # Create a maintenance goal
        goal_id = await goal_manager.create_goal(
            name="System maintenance",
            description="Regular system maintenance tasks",
            goal_type="maintenance",
            priority=5
        )

        if goal_id:
            # Schedule recurring tasks
            logger.info("Scheduling recurring tasks...")

            # Health check every 30 minutes
            await scheduler.create_recurring_task(
                goal_id=goal_id,
                name="System health check",
                task_type="analysis",
                interval_minutes=30,
                priority=3
            )

            # Cleanup every 24 hours
            await scheduler.create_recurring_task(
                goal_id=goal_id,
                name="Database cleanup",
                task_type="maintenance",
                interval_minutes=24 * 60,  # 24 hours
                priority=7
            )

            # Get schedule for next 24 hours
            schedule = await scheduler.get_schedule(hours=24)
            logger.info(f"Scheduled tasks for next 24 hours: {len(schedule)}")

            for task in schedule:
                logger.info(f"  - {task.name} at {task.scheduled_at}")

    except Exception as e:
        logger.error(f"Error in recurring tasks example: {e}")
    finally:
        await goal_manager.cleanup() if hasattr(goal_manager, 'cleanup') else None
        await scheduler.cleanup()


async def main():
    """Run all examples."""
    try:
        await example_basic_usage()
        print("\n" + "="*50 + "\n")

        await example_recurring_tasks()
        print("\n" + "="*50 + "\n")

        # Uncomment to run full system example (runs for 30 seconds)
        # await example_full_system()

    except KeyboardInterrupt:
        logger.info("Examples interrupted by user")
    except Exception as e:
        logger.error(f"Error running examples: {e}")


if __name__ == "__main__":
    asyncio.run(main())