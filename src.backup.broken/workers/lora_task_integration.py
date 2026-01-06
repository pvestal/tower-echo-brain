#!/usr/bin/env python3
"""
LORA Task Integration for Echo Brain
Adds LORA task types and handlers to the Echo task queue system
"""

import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any

from src.tasks.task_queue import Task, TaskPriority, TaskStatus, TaskType

logger = logging.getLogger(__name__)

# Extend TaskType enum with LORA task types
# Note: These will be added to the main TaskType enum in task_queue.py
LORA_TASK_TYPES = {
    'LORA_GENERATION': 'lora_generation',
    'LORA_TAGGING': 'lora_tagging',
    'LORA_TRAINING': 'lora_training'
}


def create_lora_generation_task(
    character_name: str,
    description: str,
    output_dir: str = None,
    reference_image: str = None,
    priority: TaskPriority = TaskPriority.NORMAL
) -> Task:
    """
    Create a LORA image generation task

    Args:
        character_name: Name of the character
        description: Character description for prompts
        output_dir: Output directory (defaults to /mnt/1TB-storage/lora_datasets/{character_name})
        reference_image: Optional reference image path
        priority: Task priority

    Returns:
        Task object ready to be added to queue
    """
    task_id = str(uuid.uuid4())

    payload = {
        'character_name': character_name,
        'description': description,
        'output_dir': output_dir or f'/mnt/1TB-storage/lora_datasets/{character_name}',
        'reference_image': reference_image
    }

    return Task(
        id=task_id,
        name=f"Generate LORA training images for {character_name}",
        task_type=TaskType.USER_REQUEST,  # Will use LORA_GENERATION when enum is extended
        priority=priority,
        status=TaskStatus.PENDING,
        payload=payload,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        timeout=3600,  # 1 hour for 40 image generation
        max_retries=2,
        creator='lora_system'
    )


def create_lora_tagging_task(
    character_name: str,
    image_dir: str,
    character_tags: list = None,
    description: str = "",
    priority: TaskPriority = TaskPriority.NORMAL,
    depends_on: str = None
) -> Task:
    """
    Create a LORA image tagging task

    Args:
        character_name: Name of the character
        image_dir: Directory containing images to tag
        character_tags: Base character tags (e.g., ['1girl', 'brown hair', 'blue eyes'])
        description: Character description
        priority: Task priority
        depends_on: Task ID this depends on (e.g., generation task)

    Returns:
        Task object ready to be added to queue
    """
    task_id = str(uuid.uuid4())

    payload = {
        'character_name': character_name,
        'image_dir': image_dir,
        'character_tags': character_tags or [],
        'description': description
    }

    dependencies = [depends_on] if depends_on else []

    return Task(
        id=task_id,
        name=f"Tag LORA training images for {character_name}",
        task_type=TaskType.USER_REQUEST,  # Will use LORA_TAGGING when enum is extended
        priority=priority,
        status=TaskStatus.PENDING,
        payload=payload,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        timeout=300,  # 5 minutes for tagging
        max_retries=2,
        dependencies=dependencies,
        creator='lora_system'
    )


def create_lora_training_task(
    character_name: str,
    training_dir: str,
    output_name: str,
    base_model: str = '/mnt/1TB-storage/ComfyUI/models/checkpoints/dreamshaper_8.safetensors',
    network_dim: int = 32,
    network_alpha: int = 16,
    epochs: int = 15,
    learning_rate: float = 1e-4,
    priority: TaskPriority = TaskPriority.HIGH,
    depends_on: str = None
) -> Task:
    """
    Create a LORA training task

    Args:
        character_name: Name of the character
        training_dir: Directory with tagged images
        output_name: LORA output filename (without extension)
        base_model: Path to base checkpoint model
        network_dim: Network dimension (rank) for LORA
        network_alpha: Network alpha for LORA
        epochs: Number of training epochs
        learning_rate: Learning rate
        priority: Task priority (default HIGH for GPU-intensive task)
        depends_on: Task ID this depends on (e.g., tagging task)

    Returns:
        Task object ready to be added to queue
    """
    task_id = str(uuid.uuid4())

    payload = {
        'character_name': character_name,
        'training_dir': training_dir,
        'output_name': output_name,
        'base_model': base_model,
        'network_dim': network_dim,
        'network_alpha': network_alpha,
        'epochs': epochs,
        'learning_rate': learning_rate
    }

    dependencies = [depends_on] if depends_on else []

    return Task(
        id=task_id,
        name=f"Train LORA model for {character_name}",
        task_type=TaskType.USER_REQUEST,  # Will use LORA_TRAINING when enum is extended
        priority=priority,
        status=TaskStatus.PENDING,
        payload=payload,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        timeout=7200,  # 2 hours for training
        max_retries=1,  # Only retry once for expensive GPU task
        dependencies=dependencies,
        creator='lora_system'
    )


async def create_complete_lora_pipeline(
    character_name: str,
    description: str,
    character_tags: list,
    output_name: str = None,
    task_queue = None
) -> Dict[str, str]:
    """
    Create a complete LORA training pipeline with all 3 tasks

    Args:
        character_name: Name of the character
        description: Character description
        character_tags: Character tags (e.g., ['1girl', 'brown hair', 'blue eyes'])
        output_name: LORA output filename (defaults to character_name_lora)
        task_queue: TaskQueue instance to add tasks to

    Returns:
        Dict with task IDs for generation, tagging, and training
    """
    if not task_queue:
        raise ValueError("task_queue is required")

    output_name = output_name or f"{character_name.lower().replace(' ', '_')}_lora"
    output_dir = f'/mnt/1TB-storage/lora_datasets/{character_name}'

    # Task 1: Generate images
    generation_task = create_lora_generation_task(
        character_name=character_name,
        description=description,
        output_dir=output_dir,
        priority=TaskPriority.NORMAL
    )

    # Task 2: Tag images (depends on generation)
    tagging_task = create_lora_tagging_task(
        character_name=character_name,
        image_dir=f"{output_dir}/images",
        character_tags=character_tags,
        description=description,
        priority=TaskPriority.NORMAL,
        depends_on=generation_task.id
    )

    # Task 3: Train LORA (depends on tagging)
    training_task = create_lora_training_task(
        character_name=character_name,
        training_dir=output_dir,
        output_name=output_name,
        priority=TaskPriority.HIGH,
        depends_on=tagging_task.id
    )

    # Add tasks to queue
    await task_queue.add_task(generation_task)
    await task_queue.add_task(tagging_task)
    await task_queue.add_task(training_task)

    logger.info(f"✅ Created LORA training pipeline for {character_name}")
    logger.info(f"   Generation: {generation_task.id}")
    logger.info(f"   Tagging: {tagging_task.id}")
    logger.info(f"   Training: {training_task.id}")

    return {
        'character_name': character_name,
        'generation_task_id': generation_task.id,
        'tagging_task_id': tagging_task.id,
        'training_task_id': training_task.id,
        'output_name': output_name,
        'output_dir': output_dir
    }


def register_lora_handlers(background_worker):
    """
    Register LORA task handlers with the Echo background worker

    Args:
        background_worker: BackgroundWorker instance

    Usage:
        from src.workers.lora_task_integration import register_lora_handlers
        register_lora_handlers(worker)
    """
    from src.workers.lora_generation_worker import handle_lora_generation_task
    from src.workers.lora_tagging_worker import handle_lora_tagging_task
    from src.workers.lora_training_worker import handle_lora_training_task

    # Register handlers
    # Note: Once TaskType enum is extended, use proper enum values
    # For now, using USER_REQUEST type with custom payload detection
    background_worker.register_handler(TaskType.USER_REQUEST, handle_lora_generation_task)
    background_worker.register_handler(TaskType.USER_REQUEST, handle_lora_tagging_task)
    background_worker.register_handler(TaskType.USER_REQUEST, handle_lora_training_task)

    logger.info("✅ Registered LORA task handlers with background worker")
