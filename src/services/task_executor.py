"""Execute coding tasks using local Ollama models"""
import httpx
import os
import json
import logging
from typing import Dict, Any
import asyncio
import psycopg2
from datetime import datetime

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"

MODEL_MAP = {
    "code_review": "deepseek-coder-v2:16b",
    "refactor": "deepseek-coder-v2:16b",
    "reasoning": "deepseek-r1:8b",
    "general": "qwen2.5:14b",
    "coding": "deepseek-coder-v2:16b",
    "analysis": "qwen2.5:14b",
    # Anime tasks
    "scene_description": "gemma2:9b",      # Narration agent's model
    "comfyui_prompt": "gemma2:9b",         # Generate ComfyUI prompts
    "story_development": "deepseek-r1:8b", # Plot and narrative
    "character_description": "gemma2:9b"   # Character details for LoRA
}

async def execute_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a task using the appropriate local model."""
    task_type = task.get("task_type", "general")
    model = task.get("model") or MODEL_MAP.get(task_type, "qwen2.5:3b")
    
    # Read target file if specified
    file_content = ""
    if task.get("target_file"):
        try:
            with open(task["target_file"], 'r') as f:
                file_content = f.read()
        except Exception as e:
            logger.error(f"Failed to read {task['target_file']}: {e}")
    
    prompt = f"""Task: {task.get('description', task)}
    
Task Type: {task_type}
Target file: {task.get('target_file', 'N/A')}
Requirements: {json.dumps(task.get('requirements', []))}

Current file content:
```python
{file_content}
```

Provide the complete solution:"""

    logger.info(f"Executing {task_type} task with model {model}")
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False}
            )
            result = response.json()
            
        return {
            "model_used": model,
            "response": result.get("response"),
            "task_id": task.get("id"),
            "task_type": task_type,
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        return {
            "model_used": model,
            "error": str(e),
            "task_id": task.get("id"),
            "task_type": task_type,
            "status": "failed",
            "timestamp": datetime.now().isoformat()
        }

async def execute_anime_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Execute anime generation tasks."""
    task_type = task.get("task_type")

    if task_type == "generate_image":
        # Call anime production API (checking both possible ports)
        ports = [8328, 8305]  # Try both known ports
        for port in ports:
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    response = await client.post(
                        f"http://localhost:{port}/api/anime/generate",
                        json={
                            "prompt": task.get("prompt"),
                            "negative_prompt": task.get("negative_prompt", ""),
                            "project_id": task.get("project_id"),
                            "character_id": task.get("character_id"),
                            "profile_name": task.get("profile_name", "tokyo_debt_realism"),
                        }
                    )
                    if response.status_code == 200:
                        return response.json()
            except:
                continue
        return {"error": "Anime production API not available"}

    elif task_type == "develop_scene":
        # Use LLM to develop scene narrative, then generate
        model = MODEL_MAP.get("scene_description", "gemma2:9b")
        prompt = f"""Develop a detailed anime scene description.

Scene concept: {task.get("concept")}
Project style: {task.get("style", "photorealistic")}
Characters: {task.get("characters", [])}

Output a JSON with:
- scene_description: Detailed visual description
- camera_angle: Suggested camera position
- lighting: Lighting mood
- comfyui_prompt: Prompt for image generation
- negative_prompt: What to avoid
"""
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False}
            )
            return {
                "task_type": task_type,
                "model_used": model,
                "scene_data": response.json().get("response")
            }

    elif task_type == "batch_generate":
        # Generate multiple images for a scene
        results = []
        for shot in task.get("shots", []):
            result = await execute_anime_task({
                "task_type": "generate_image",
                **shot
            })
            results.append(result)
        return {"shots": results, "count": len(results)}

    return {"error": "Unknown anime task type"}

def store_result(result: Dict[str, Any]):
    """Store task result in PostgreSQL."""
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="echo_brain",
            user="patrick",
            password=os.getenv("TOWER_DB_PASSWORD", os.getenv("TOWER_DB_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE"))
        )
        cur = conn.cursor()
        
        # Create table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS task_results (
                id SERIAL PRIMARY KEY,
                task_id VARCHAR(100),
                task_type VARCHAR(50),
                model_used VARCHAR(100),
                status VARCHAR(20),
                response TEXT,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cur.execute("""
            INSERT INTO task_results (task_id, task_type, model_used, status, response, error)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            result.get("task_id"),
            result.get("task_type"),
            result.get("model_used"),
            result.get("status"),
            result.get("response"),
            result.get("error")
        ))
        
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Result stored for task {result.get('task_id')}")
    except Exception as e:
        logger.error(f"Failed to store result: {e}")
