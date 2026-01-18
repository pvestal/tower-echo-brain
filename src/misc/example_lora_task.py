#!/usr/bin/env python3
"""
Example: Create LORA Training Tasks
Demonstrates how to use the LORA worker system
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, '/opt/tower-echo-brain')

from src.tasks.task_queue import TaskQueue, TaskPriority
from src.workers.lora_task_integration import (
    create_lora_generation_task,
    create_lora_tagging_task,
    create_lora_training_task,
    create_complete_lora_pipeline
)


async def example_single_task():
    """Example: Create a single LORA generation task"""
    print("=" * 60)
    print("Example 1: Creating a single LORA generation task")
    print("=" * 60)

    # Initialize task queue
    task_queue = TaskQueue()
    await task_queue.initialize()

    # Create generation task
    task = create_lora_generation_task(
        character_name="Elara",
        description="1girl, brown hair, blue eyes, fantasy armor, warrior",
        priority=TaskPriority.NORMAL
    )

    # Add to queue
    success = await task_queue.add_task(task)

    if success:
        print(f"✅ Task created successfully!")
        print(f"   Task ID: {task.id}")
        print(f"   Task Name: {task.name}")
        print(f"   Priority: {task.priority.name}")
        print(f"   Payload: {task.payload}")
    else:
        print(f"❌ Failed to create task")


async def example_complete_pipeline():
    """Example: Create complete LORA training pipeline"""
    print("\n" + "=" * 60)
    print("Example 2: Creating complete LORA training pipeline")
    print("=" * 60)

    # Initialize task queue
    task_queue = TaskQueue()
    await task_queue.initialize()

    # Create complete pipeline
    result = await create_complete_lora_pipeline(
        character_name="Elara",
        description="1girl, brown hair, blue eyes, fantasy armor, warrior",
        character_tags=["1girl", "brown hair", "blue eyes", "armor"],
        output_name="elara_warrior_lora",
        task_queue=task_queue
    )

    print(f"✅ LORA pipeline created successfully!")
    print(f"   Character: {result['character_name']}")
    print(f"   Output Name: {result['output_name']}")
    print(f"   Output Dir: {result['output_dir']}")
    print(f"\n   Task IDs:")
    print(f"   - Generation: {result['generation_task_id']}")
    print(f"   - Tagging: {result['tagging_task_id']}")
    print(f"   - Training: {result['training_task_id']}")


async def example_check_tasks():
    """Example: Check task status"""
    print("\n" + "=" * 60)
    print("Example 3: Checking task queue status")
    print("=" * 60)

    task_queue = TaskQueue()
    await task_queue.initialize()

    stats = await task_queue.get_task_stats()

    print(f"📊 Task Queue Statistics (Last 24h):")
    print(f"   Total Tasks: {stats.get('total_24h', 0)}")
    print(f"\n   By Status:")
    for status, count in stats.get('by_status', {}).items():
        print(f"     - {status}: {count}")
    print(f"\n   By Type:")
    for task_type, count in stats.get('by_type', {}).items():
        print(f"     - {task_type}: {count}")
    print(f"\n   By Priority:")
    for priority, count in stats.get('by_priority', {}).items():
        print(f"     - Priority {priority}: {count}")


async def example_manual_task_insertion():
    """Example: Manually insert task into database"""
    print("\n" + "=" * 60)
    print("Example 4: Manual task insertion via SQL")
    print("=" * 60)

    import psycopg2
    import json
    from datetime import datetime
    import uuid

    db_config = {
        "database": "echo_brain",
        "user": "patrick",
        "host": "localhost",
        "password": os.environ.get("DB_PASSWORD", "tower_echo_brain_secret_key_2025"),
        "port": 5432
    }

    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()

    task_id = str(uuid.uuid4())
    now = datetime.now()

    cursor.execute("""
        INSERT INTO echo_tasks (
            id, name, task_type, priority, status, payload,
            created_at, updated_at, retries, max_retries, timeout, creator
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (
        task_id,
        "Generate LORA images for Test Character",
        "user_request",  # task_type
        3,  # priority (NORMAL)
        "pending",  # status
        json.dumps({
            'character_name': 'TestCharacter',
            'description': '1girl, test character',
            'output_dir': '/mnt/1TB-storage/lora_datasets/TestCharacter'
        }),
        now,  # created_at
        now,  # updated_at
        0,  # retries
        3,  # max_retries
        3600,  # timeout
        'manual_example'  # creator
    ))

    conn.commit()
    cursor.close()
    conn.close()

    print(f"✅ Task inserted directly into database!")
    print(f"   Task ID: {task_id}")
    print(f"   Check status with: SELECT * FROM echo_tasks WHERE id = '{task_id}';")


async def main():
    """Run all examples"""
    print("\n🚀 LORA Worker System Examples\n")

    try:
        # Example 1: Single task
        await example_single_task()

        # Example 2: Complete pipeline
        await example_complete_pipeline()

        # Example 3: Check queue stats
        await example_check_tasks()

        # Example 4: Manual SQL insertion
        await example_manual_task_insertion()

        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Start the background worker to process tasks")
        print("2. Monitor task progress in echo_tasks table")
        print("3. Check logs in /opt/tower-echo-brain/logs/")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
