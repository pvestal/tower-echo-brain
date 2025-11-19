# Echo Brain Developer Guide - Working Features Only

## Overview
This guide covers **only the features that actually work** in the Echo Brain system as of November 2025. No aspirational features or broken components are included.

## Quick Start: What Actually Works

### 1. Echo Brain Health Monitoring ✅
```bash
# Check service status
curl http://localhost:8309/api/echo/health

# Get system metrics
curl http://localhost:8309/api/echo/system/metrics
```

**Response Example**:
```json
{
  "status": "healthy",
  "service": "echo-brain",
  "cpu_percent": 1.5,
  "memory_percent": 13.2,
  "vram_used_gb": 2.24,
  "nvidia_vram_total_gb": 12.0
}
```

### 2. Patrick's Content Generator ✅
```python
import asyncio
from src.patrick_content_generator import patrick_generator

async def generate_character():
    # Generate Yuki from Tokyo Debt Crisis
    result = await patrick_generator.generate_character_image(
        project="tokyo_debt_crisis",
        character="yuki"
    )
    return result

# Run it
result = asyncio.run(generate_character())
print(f"Generation ID: {result['prompt_id']}")
```

**Available Characters**:
- **Tokyo Debt Crisis**: riku, yuki, sakura
- **Goblin Slayer Neon**: kai_nakamura, ryuu, hiroshi

**Expected Performance**: 15-30 seconds per image

### 3. LoRA Dataset Creation ✅
```python
from src.lora_dataset_creator import lora_creator

# Create dataset for Yuki
result = lora_creator.create_character_dataset("yuki", "tokyo_debt_crisis")

if result.get("status") == "success":
    print(f"Dataset created at: {result['dataset_path']}")
    print(f"Number of images: {result['num_images']}")
```

**What it creates**:
- `/mnt/1TB-storage/lora_datasets/tokyo_debt_crisis_yuki/`
- Numbered images: `0001.png`, `0002.png`, etc.
- Caption files: `0001.txt`, `0002.txt`, etc.
- Training config: `training_config.json`

### 4. Dynamic Workflow Generation ✅
```python
from src.workflow_generator import workflow_generator

# Create custom workflow
workflow = workflow_generator.generate_workflow(
    workflow_type="image",
    prompt="patrick_yuki, yakuza daughter, dramatic lighting",
    project="tokyo_debt_crisis"
)

# Submit to ComfyUI
import aiohttp
import json

async def submit_workflow(workflow):
    async with aiohttp.ClientSession() as session:
        payload = {
            "prompt": workflow,
            "client_id": "developer_test"
        }
        async with session.post(
            "http://localhost:8188/prompt",
            json=payload
        ) as response:
            return await response.json()
```

## Working API Endpoints

### Echo Brain Core
```bash
# Health check
GET http://localhost:8309/api/echo/health

# System metrics
GET http://localhost:8309/api/echo/system/metrics

# Recent conversation status
GET http://localhost:8309/api/echo/status

# Database statistics
GET http://localhost:8309/api/echo/db/stats
```

### ComfyUI Direct
```bash
# Check queue
GET http://localhost:8188/api/queue

# Get history (recent generations)
GET http://localhost:8188/api/history?limit=5

# Submit workflow
POST http://localhost:8188/prompt
```

## Code Examples That Work

### Generate All Characters for a Project
```python
async def generate_project_characters(project_name):
    """Generate all characters for a specific project"""

    project_chars = {
        "tokyo_debt_crisis": ["riku", "yuki", "sakura"],
        "goblin_slayer_neon": ["kai_nakamura", "ryuu", "hiroshi"]
    }

    if project_name not in project_chars:
        return {"error": f"Unknown project: {project_name}"}

    results = {}
    for character in project_chars[project_name]:
        try:
            result = await patrick_generator.generate_character_image(
                project=project_name,
                character=character
            )
            results[character] = result

            # Wait between requests to avoid overwhelming ComfyUI
            await asyncio.sleep(2)

        except Exception as e:
            results[character] = {"error": str(e)}

    return results
```

### Batch Dataset Creation
```python
def create_all_datasets():
    """Create LoRA datasets for all characters with existing images"""

    results = lora_creator.create_all_datasets()

    # Print summary
    for character, result in results.items():
        if result.get("status") == "success":
            print(f"✅ {character}: {result['num_images']} images")
        else:
            print(f"❌ {character}: {result.get('error', 'Unknown error')}")

    return results
```

### Monitor ComfyUI Queue
```python
import aiohttp
import asyncio

async def monitor_generation(prompt_id):
    """Monitor a ComfyUI generation until completion"""

    while True:
        async with aiohttp.ClientSession() as session:
            # Check queue
            async with session.get("http://localhost:8188/api/queue") as response:
                queue_data = await response.json()

            # Check if our job is running
            running = queue_data.get("queue_running", [])
            pending = queue_data.get("queue_pending", [])

            # Look for our prompt_id
            found_running = any(item[1] == prompt_id for item in running)
            found_pending = any(item[1] == prompt_id for item in pending)

            if found_running:
                print(f"🔄 Generation {prompt_id} is running...")
            elif found_pending:
                print(f"⏳ Generation {prompt_id} is pending...")
            else:
                # Check history to see if completed
                async with session.get(f"http://localhost:8188/api/history/{prompt_id}") as hist_response:
                    if hist_response.status == 200:
                        history = await hist_response.json()
                        if prompt_id in history:
                            print(f"✅ Generation {prompt_id} completed!")
                            return history[prompt_id]

                print(f"❓ Generation {prompt_id} status unknown")
                break

        await asyncio.sleep(5)

    return None
```

## File Locations and Structure

### Source Code
```
/opt/tower-echo-brain/src/
├── patrick_content_generator.py    # Character image generation
├── lora_dataset_creator.py         # Dataset preparation
├── lora_trainer.py                 # Training framework (simulated)
├── workflow_generator.py           # Dynamic workflow creation
├── main.py                         # FastAPI app entry point
└── app_factory.py                  # Application factory
```

### Generated Content
```
/mnt/1TB-storage/ComfyUI/output/
├── patrick_yuki_*.png              # Yuki character images
├── patrick_kai_nakamura_*.png      # Kai character images
└── patrick_*.png                   # Other character images

/mnt/1TB-storage/lora_datasets/
├── tokyo_debt_crisis_yuki/
│   ├── 10_patrick_yuki/
│   │   ├── 0001.png
│   │   ├── 0001.txt
│   │   └── ...
│   └── training_config.json
└── goblin_slayer_neon_kai_nakamura/
    └── ...
```

## Performance Expectations

### Realistic Timing
- **Character Image Generation**: 15-30 seconds
- **Dataset Creation**: 1-5 seconds (file operations only)
- **Workflow Generation**: Instant (in-memory operations)
- **ComfyUI Queue**: Variable (depends on current load)

### Resource Usage
- **Echo Brain Memory**: ~150MB
- **ComfyUI VRAM**: 2-8GB (depends on model and resolution)
- **Storage**: ~1-2MB per generated image

## Troubleshooting Working Features

### Common Issues and Solutions

#### 1. Echo Brain Not Responding
```bash
# Check service status
sudo systemctl status tower-echo-brain

# Restart if needed
sudo systemctl restart tower-echo-brain

# Check logs
sudo journalctl -u tower-echo-brain -f
```

#### 2. ComfyUI Generation Fails
```bash
# Check ComfyUI logs
tail -f /mnt/1TB-storage/ComfyUI/comfyui.log

# Verify models exist
ls -la /mnt/1TB-storage/ComfyUI/models/checkpoints/counterfeit_v3.safetensors
```

#### 3. No Images Found for Dataset Creation
```python
# Check what images exist
from pathlib import Path
source_dir = Path("/mnt/1TB-storage/ComfyUI/output")
patrick_images = list(source_dir.glob("patrick_*.png"))
print(f"Found {len(patrick_images)} Patrick images")
for img in patrick_images[:5]:
    print(f"  {img.name}")
```

## What NOT to Use (Broken Features)

❌ **Anime Production API (Port 8328)**: Returns 404 errors
❌ **Real-time Progress Tracking**: Not implemented
❌ **Actual LoRA Training**: Only creates placeholder files
❌ **Job Status API**: Broken for real jobs
❌ **Video Generation**: Extremely slow (8+ minutes)

## Integration Points

### With Other Tower Services
```python
# Save results to Knowledge Base (if available)
import requests

def save_to_kb(title, content):
    try:
        response = requests.post(
            "http://localhost:8307/api/kb/articles",
            json={
                "title": title,
                "content": content,
                "category": "echo_brain_generations",
                "tags": ["echo", "generation", "patrick"]
            }
        )
        return response.status_code == 200
    except:
        return False
```

### With ComfyUI Models
```python
# List available checkpoints
import requests

def get_available_models():
    try:
        response = requests.get("http://localhost:8188/api/object_info")
        if response.status_code == 200:
            data = response.json()
            checkpoints = data.get("CheckpointLoaderSimple", {}).get("input", {}).get("ckpt_name", [])
            return checkpoints
        return []
    except:
        return []
```

## Best Practices

1. **Always check service health** before starting generation jobs
2. **Use realistic timeouts** (30+ seconds for image generation)
3. **Monitor GPU memory** to avoid OOM errors
4. **Create datasets incrementally** (don't try to process everything at once)
5. **Test workflows** on simple prompts before complex ones
6. **Save successful configurations** for reuse

---

**Last Updated**: November 19, 2025
**Status**: All examples tested and working on Tower system
**Version**: Echo Brain Production v1.0