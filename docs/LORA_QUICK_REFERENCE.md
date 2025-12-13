# LORA Worker System - Quick Reference

## Quick Start

### Create Complete LORA Pipeline (One Command)

```python
import asyncio
from src.tasks.task_queue import TaskQueue
from src.workers.lora_task_integration import create_complete_lora_pipeline

async def create_lora():
    task_queue = TaskQueue()
    await task_queue.initialize()

    result = await create_complete_lora_pipeline(
        character_name="YourCharacter",
        description="1girl, brown hair, blue eyes, fantasy armor",
        character_tags=["1girl", "brown hair", "blue eyes"],
        task_queue=task_queue
    )

    print(f"Pipeline created! Training Task ID: {result['training_task_id']}")

asyncio.run(create_lora())
```

## File Locations

```
/opt/tower-echo-brain/
├── src/workers/
│   ├── lora_generation_worker.py   # Generates 40 training images
│   ├── lora_tagging_worker.py      # Creates .txt tag files
│   ├── lora_training_worker.py     # Trains LORA model
│   └── lora_task_integration.py    # Task helpers
├── docs/
│   ├── LORA_WORKER_SYSTEM_README.md           # Full documentation
│   └── LORA_SYSTEM_IMPLEMENTATION_SUMMARY.md  # Implementation details
└── example_lora_task.py            # Working examples
```

## Check Task Status

### Database Query
```sql
SELECT id, name, status, priority, created_at
FROM echo_tasks
WHERE name LIKE '%LORA%'
ORDER BY created_at DESC;
```

### Python
```python
from src.tasks.task_queue import TaskQueue

task_queue = TaskQueue()
await task_queue.initialize()
stats = await task_queue.get_task_stats()
print(stats)
```

## Task Types & Timeouts

| Task Type | Timeout | Priority | Description |
|-----------|---------|----------|-------------|
| LORA_GENERATION | 1 hour | NORMAL | Generate 40 images |
| LORA_TAGGING | 5 min | NORMAL | Create tag files |
| LORA_TRAINING | 2 hours | HIGH | Train LORA model |

## Output Locations

- **Images**: `/mnt/1TB-storage/lora_datasets/{character_name}/images/`
- **LORA Model**: `/mnt/1TB-storage/ComfyUI/models/loras/{output_name}.safetensors`

## Common Commands

### Run Example
```bash
cd /opt/tower-echo-brain
python3 example_lora_task.py
```

### Check Background Worker
```bash
systemctl --user status tower-echo-brain
```

### View Logs
```bash
tail -f /opt/tower-echo-brain/logs/echo.log
```

## Troubleshooting

### Tasks Not Processing
1. Check background worker: `systemctl --user status tower-echo-brain`
2. Check Redis: `redis-cli ping`
3. Check database: `psql -U patrick -d echo_brain`

### ComfyUI Issues
```bash
curl http://localhost:8188/system_stats
```

### Check GPU
```bash
nvidia-smi
```

## Task Priorities

- **1 = URGENT**: System failures
- **2 = HIGH**: GPU tasks (training)
- **3 = NORMAL**: Image generation, tagging
- **4 = LOW**: Background tasks
- **5 = SCHEDULED**: Daily reports

## Dependencies

Tasks respect dependency chains:
1. Generation → Tagging → Training
2. Parent task must complete before child starts
3. Automatic dependency resolution

## Quick Examples

### Single Task
```python
from src.workers.lora_task_integration import create_lora_generation_task

task = create_lora_generation_task(
    character_name="Elara",
    description="1girl, warrior"
)
await task_queue.add_task(task)
```

### Check Progress
```sql
SELECT
    name,
    status,
    started_at,
    completed_at,
    result->'data'->>'total_generated' as images_generated
FROM echo_tasks
WHERE id = 'task-id-here';
```

## Configuration

Default hyperparameters (can be overridden):

```python
{
    'network_dim': 32,      # LORA rank
    'network_alpha': 16,    # LORA alpha
    'epochs': 15,           # Training epochs
    'learning_rate': 1e-4,  # Learning rate
    'resolution': 768,      # Image size
    'steps': 28,            # ComfyUI steps
    'cfg': 7.5              # CFG scale
}
```

## Success Indicators

- ✅ Task status: `pending` → `running` → `completed`
- ✅ Images directory: 40 .png + 40 .txt files
- ✅ LORA file: .safetensors in ComfyUI models folder
- ✅ No errors in `echo_tasks.error` column

## Support

- Full docs: `/opt/tower-echo-brain/docs/LORA_WORKER_SYSTEM_README.md`
- Examples: `/opt/tower-echo-brain/example_lora_task.py`
- Implementation: `/opt/tower-echo-brain/docs/LORA_SYSTEM_IMPLEMENTATION_SUMMARY.md`
