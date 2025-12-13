# LORA Worker System for Echo Brain

Complete autonomous LORA training system integrated with Echo Brain's task queue.

## Overview

The LORA Worker System enables Echo Brain to autonomously generate training datasets, tag images, and train LORA models for character customization in ComfyUI.

## Architecture

### Components

1. **LORA Generation Worker** (`src/workers/lora_generation_worker.py`)
   - Generates 40 diverse training images via ComfyUI
   - Creates prompt variations (poses, expressions, angles, lighting)
   - Handles batch generation with error recovery
   - Output: Images + metadata JSON

2. **LORA Tagging Worker** (`src/workers/lora_tagging_worker.py`)
   - Creates .txt caption files for each image
   - Formats tags: `character_name, features, pose, expression, quality_tags`
   - Uses generation metadata for accurate tagging
   - Output: Tagged dataset ready for training

3. **LORA Training Worker** (`src/workers/lora_training_worker.py`)
   - Prepares kohya_ss dataset structure
   - Generates training configuration
   - Executes training via kohya_ss/accelerate
   - Output: Trained LORA model (.safetensors)

4. **Task Integration** (`src/workers/lora_task_integration.py`)
   - Task creation helpers
   - Pipeline orchestration
   - Dependency management
   - Worker registration

## Installation

### Files Location

```
/opt/tower-echo-brain/
├── src/
│   └── workers/
│       ├── __init__.py
│       ├── lora_generation_worker.py
│       ├── lora_tagging_worker.py
│       ├── lora_training_worker.py
│       └── lora_task_integration.py
└── example_lora_task.py
```

### Dependencies

Already installed in Echo Brain:
- asyncio
- aiohttp
- psycopg2
- redis

Additional (optional, for training):
- kohya_ss (install at `/opt/kohya_ss/`)
- accelerate
- NVIDIA GPU with CUDA

## Usage

### Method 1: Complete Pipeline (Recommended)

Creates all 3 tasks with automatic dependencies:

```python
from src.tasks.task_queue import TaskQueue
from src.workers.lora_task_integration import create_complete_lora_pipeline

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

# Returns task IDs
print(result['generation_task_id'])
print(result['tagging_task_id'])
print(result['training_task_id'])
```

### Method 2: Individual Tasks

Create tasks separately for more control:

```python
from src.workers.lora_task_integration import (
    create_lora_generation_task,
    create_lora_tagging_task,
    create_lora_training_task
)

# Task 1: Generate images
gen_task = create_lora_generation_task(
    character_name="Elara",
    description="1girl, brown hair, blue eyes, fantasy armor, warrior"
)
await task_queue.add_task(gen_task)

# Task 2: Tag images (after generation completes)
tag_task = create_lora_tagging_task(
    character_name="Elara",
    image_dir="/mnt/1TB-storage/lora_datasets/Elara/images",
    character_tags=["1girl", "brown hair", "blue eyes"],
    depends_on=gen_task.id  # Wait for generation
)
await task_queue.add_task(tag_task)

# Task 3: Train LORA (after tagging completes)
train_task = create_lora_training_task(
    character_name="Elara",
    training_dir="/mnt/1TB-storage/lora_datasets/Elara",
    output_name="elara_lora",
    depends_on=tag_task.id  # Wait for tagging
)
await task_queue.add_task(train_task)
```

### Method 3: Direct SQL Insertion

Insert tasks directly into the database:

```sql
INSERT INTO echo_tasks (
    name, task_type, priority, status, payload,
    created_at, updated_at, timeout, creator
) VALUES (
    'Generate LORA training images for MyCharacter',
    'user_request',
    3,  -- NORMAL priority
    'pending',
    '{"character_name": "MyCharacter", "description": "1girl, blonde hair", "output_dir": "/mnt/1TB-storage/lora_datasets/MyCharacter"}'::jsonb,
    NOW(),
    NOW(),
    3600,  -- 1 hour timeout
    'manual'
);
```

## Task Configuration

### Generation Task Payload

```json
{
    "character_name": "CharacterName",
    "description": "1girl, brown hair, blue eyes, armor",
    "output_dir": "/mnt/1TB-storage/lora_datasets/CharacterName",
    "reference_image": "/path/to/reference.png"  // optional
}
```

### Tagging Task Payload

```json
{
    "character_name": "CharacterName",
    "image_dir": "/mnt/1TB-storage/lora_datasets/CharacterName/images",
    "character_tags": ["1girl", "brown hair", "blue eyes"],
    "description": "Additional description"
}
```

### Training Task Payload

```json
{
    "character_name": "CharacterName",
    "training_dir": "/mnt/1TB-storage/lora_datasets/CharacterName",
    "output_name": "character_lora",
    "base_model": "/mnt/1TB-storage/ComfyUI/models/checkpoints/dreamshaper_8.safetensors",
    "network_dim": 32,
    "network_alpha": 16,
    "epochs": 15,
    "learning_rate": 0.0001
}
```

## Task Priorities

- **LORA_GENERATION**: `NORMAL` (Priority 3)
- **LORA_TAGGING**: `NORMAL` (Priority 3)
- **LORA_TRAINING**: `HIGH` (Priority 2) - GPU-intensive

## Task Timeouts

- **Generation**: 3600 seconds (1 hour for 40 images)
- **Tagging**: 300 seconds (5 minutes)
- **Training**: 7200 seconds (2 hours)

## Monitoring Tasks

### Check Task Status

```python
from src.tasks.task_queue import TaskQueue

task_queue = TaskQueue()
await task_queue.initialize()

# Get statistics
stats = await task_queue.get_task_stats()
print(stats)
```

### Query Database

```sql
-- Check all LORA tasks
SELECT id, name, status, priority, created_at, updated_at
FROM echo_tasks
WHERE name LIKE '%LORA%'
ORDER BY created_at DESC;

-- Check specific task
SELECT * FROM echo_tasks WHERE id = 'task-id-here';

-- Check task results
SELECT name, status, result, error
FROM echo_tasks
WHERE task_type = 'user_request'
AND status IN ('completed', 'failed');
```

## Output Structure

```
/mnt/1TB-storage/lora_datasets/CharacterName/
├── images/
│   ├── charactername_001.png
│   ├── charactername_001.txt
│   ├── charactername_002.png
│   ├── charactername_002.txt
│   └── ... (40 images + 40 txt files)
├── generation_metadata.json
└── tagging_results.json

/mnt/1TB-storage/ComfyUI/models/loras/
└── character_lora.safetensors  (trained LORA)
```

## Worker Registration

To enable task processing, register handlers with the background worker:

```python
from src.tasks.background_worker import BackgroundWorker
from src.workers.lora_task_integration import register_lora_handlers

# In your Echo startup code
worker = BackgroundWorker(task_queue)
register_lora_handlers(worker)
await worker.start()
```

## Testing

Run the example script:

```bash
cd /opt/tower-echo-brain
python3 example_lora_task.py
```

This will:
1. Create a single generation task
2. Create a complete 3-task pipeline
3. Show task queue statistics
4. Demonstrate manual task insertion

## Troubleshooting

### Issue: Tasks stuck in pending

**Solution**: Ensure background worker is running and handlers are registered.

```bash
# Check if background worker is active
systemctl --user status tower-echo-brain

# Check logs
tail -f /opt/tower-echo-brain/logs/echo.log
```

### Issue: ComfyUI connection failed

**Solution**: Verify ComfyUI is running on port 8188.

```bash
curl http://localhost:8188/system_stats
```

### Issue: Training fails (kohya_ss not found)

**Solution**: Training will simulate if kohya_ss isn't installed. To enable real training:

```bash
cd /opt
git clone https://github.com/bmaltais/kohya_ss.git
cd kohya_ss
./setup.sh
```

### Issue: GPU out of memory

**Solution**: Reduce network_dim or batch_size in training config.

## Integration with Echo Brain

The LORA worker system integrates seamlessly with Echo's existing infrastructure:

- **Task Queue**: Uses existing `echo_tasks` table
- **Background Worker**: Compatible with `BackgroundWorker` class
- **Redis**: Uses existing Redis instance for queue management
- **PostgreSQL**: Persists all task data and results
- **Logging**: Uses Echo's logging system

## Future Enhancements

- [ ] Extend `TaskType` enum with dedicated LORA task types
- [ ] Add progress callbacks for real-time status updates
- [ ] Implement automatic quality assessment
- [ ] Add character style transfer capabilities
- [ ] Support for multiple base models
- [ ] Automatic hyperparameter tuning
- [ ] Integration with Echo's learning system

## License

Part of Echo Brain - Tower AI System

## Authors

Created by Claude Code (Anthropic) for Patrick's Echo Brain system
Date: December 12, 2025
