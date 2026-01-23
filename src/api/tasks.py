"""Task execution API endpoints for Echo Brain"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
import logging
from src.services.task_executor import execute_task, store_result
import asyncio

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/echo/tasks", tags=["tasks"])

class TaskRequest(BaseModel):
    task_type: str
    description: str
    target_file: Optional[str] = None
    requirements: Optional[List[str]] = []
    model: Optional[str] = None

class TaskResponse(BaseModel):
    task_id: str
    status: str
    model_used: Optional[str] = None
    response: Optional[str] = None
    error: Optional[str] = None

# In-memory task store (should use Redis in production)
task_results = {}

@router.post("/execute", response_model=TaskResponse)
async def execute_task_sync(task: TaskRequest):
    """Execute a task synchronously and return result."""
    task_dict = task.dict()
    task_dict["id"] = str(uuid.uuid4())
    
    result = await execute_task(task_dict)
    
    # Store in memory and DB
    task_results[task_dict["id"]] = result
    store_result(result)
    
    return TaskResponse(
        task_id=task_dict["id"],
        status=result.get("status", "completed"),
        model_used=result.get("model_used"),
        response=result.get("response"),
        error=result.get("error")
    )

@router.post("/queue", response_model=Dict[str, str])
async def queue_task(task: TaskRequest, background_tasks: BackgroundTasks):
    """Queue a task for background processing."""
    task_dict = task.dict()
    task_dict["id"] = str(uuid.uuid4())
    
    # Add to background tasks
    background_tasks.add_task(process_task_background, task_dict)
    
    return {
        "task_id": task_dict["id"],
        "status": "queued",
        "message": "Task queued for background processing"
    }

async def process_task_background(task_dict: Dict[str, Any]):
    """Process task in background."""
    try:
        result = await execute_task(task_dict)
        task_results[task_dict["id"]] = result
        store_result(result)
        logger.info(f"Background task {task_dict['id']} completed")
    except Exception as e:
        logger.error(f"Background task {task_dict['id']} failed: {e}")
        task_results[task_dict["id"]] = {
            "status": "failed",
            "error": str(e)
        }

@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a queued task."""
    if task_id not in task_results:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task_results[task_id]

@router.get("/results")
async def list_results(limit: int = 10):
    """List recent task results."""
    # Return last N results from memory
    result_list = list(task_results.values())[-limit:]
    return {"results": result_list, "total": len(task_results)}

@router.post("/anime/generate")
async def generate_anime(task: Dict[str, Any]):
    """Generate anime content using local models + ComfyUI."""
    from src.services.task_executor import execute_anime_task
    return await execute_anime_task(task)

@router.post("/anime/scene")
async def develop_scene(request: Dict[str, Any]):
    """Develop a scene description and generate it."""
    from src.services.task_executor import execute_anime_task
    return await execute_anime_task({
        "task_type": "develop_scene",
        "concept": request.get("concept"),
        "style": request.get("style", "photorealistic"),
        "characters": request.get("characters", [])
    })
