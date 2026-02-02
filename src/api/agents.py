"""Agent API endpoints"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import subprocess
import tempfile
import os

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agents", tags=["agents"])

# Import agents
try:
    from ..agents.coding_agent import coding_agent
    from ..agents.reasoning_agent import reasoning_agent
    from ..agents.narration_agent import narration_agent
except ImportError:
    # Fallback for standalone testing
    from src.agents.coding_agent import coding_agent
    from src.agents.reasoning_agent import reasoning_agent
    from src.agents.narration_agent import narration_agent

class CodingTaskRequest(BaseModel):
    task: str
    language: str = "python"
    validate: bool = True
    files: Optional[List[str]] = None
    requirements: Optional[str] = None

class CodingTaskResponse(BaseModel):
    task: str
    response: str
    code: Optional[str]
    validation: Optional[Dict]
    model: str
    context_used: Dict

@router.post("/coding", response_model=CodingTaskResponse)
async def run_coding_task(request: CodingTaskRequest):
    """Run a coding task through the coding agent"""
    try:
        logger.info(f"Coding agent request: {request.task[:50]}...")
        result = await coding_agent.process(
            task=request.task,
            context={
                "language": request.language,
                "validate": request.validate,
                "files": request.files,
                "requirements": request.requirements
            }
        )
        return CodingTaskResponse(**result)
    except Exception as e:
        logger.error(f"Coding agent failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/coding/history")
async def get_coding_history():
    """Get coding agent task history"""
    return {
        "agent": "CodingAgent",
        "model": getattr(coding_agent, 'model_name', 'unknown'),
        "history_count": len(coding_agent.history) if hasattr(coding_agent, 'history') else 0,
        "history": getattr(coding_agent, 'history', [])[-20:]
    }

class ReasoningTaskRequest(BaseModel):
    task: str
    background: Optional[str] = None
    constraints: Optional[str] = None
    options: Optional[List[str]] = None
    examples: Optional[List[str]] = None

class ReasoningTaskResponse(BaseModel):
    task: str
    response: str
    analysis: Optional[str]
    reasoning: Optional[str]
    conclusion: Optional[str]
    model: str

class NarrationTaskRequest(BaseModel):
    scene: str
    character: Optional[str] = None
    project_id: Optional[int] = None
    mood: Optional[str] = None
    genre: Optional[str] = None
    previous_scene: Optional[str] = None

class NarrationTaskResponse(BaseModel):
    scene: str
    narration: str
    mood: Optional[str]
    visual_notes: Optional[str]
    comfyui_prompt: Optional[str]
    character: Optional[str]
    project_id: Optional[int]
    model: str

class ExecuteRequest(BaseModel):
    code: Optional[str] = None  # For code execution
    task: Optional[str] = None  # For agent task
    agent_type: Optional[str] = None  # Type of agent (coding, reasoning, narration)
    language: str = "python"
    timeout: int = 30
    parameters: Optional[Dict[str, Any]] = None  # Additional parameters

@router.post("/reasoning", response_model=ReasoningTaskResponse)
async def run_reasoning_task(request: ReasoningTaskRequest):
    """Run a reasoning task through the reasoning agent"""
    try:
        logger.info(f"Reasoning agent request: {request.task[:50]}...")
        result = await reasoning_agent.process(
            task=request.task,
            context={
                "background": request.background,
                "constraints": request.constraints,
                "options": request.options,
                "examples": request.examples
            }
        )
        return ReasoningTaskResponse(**result)
    except Exception as e:
        logger.error(f"Reasoning agent failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reasoning/history")
async def get_reasoning_history():
    """Get reasoning agent task history"""
    return {
        "agent": "ReasoningAgent",
        "model": getattr(reasoning_agent, 'model_name', 'unknown'),
        "history_count": len(reasoning_agent.history) if hasattr(reasoning_agent, 'history') else 0,
        "history": getattr(reasoning_agent, 'history', [])[-20:]
    }

@router.post("/narration", response_model=NarrationTaskResponse)
async def run_narration_task(request: NarrationTaskRequest):
    """Run a narration task through the narration agent"""
    try:
        logger.info(f"Narration agent request: {request.scene[:50]}...")
        result = await narration_agent.process(
            task=request.scene,
            context={
                "character": request.character,
                "project_id": request.project_id,
                "mood": request.mood,
                "genre": request.genre,
                "previous_scene": request.previous_scene
            }
        )
        return NarrationTaskResponse(**result)
    except Exception as e:
        logger.error(f"Narration agent failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/narration/history")
async def get_narration_history():
    """Get narration agent task history"""
    return {
        "agent": "NarrationAgent",
        "model": getattr(narration_agent, 'model_name', 'unknown'),
        "history_count": len(narration_agent.history) if hasattr(narration_agent, 'history') else 0,
        "history": getattr(narration_agent, 'history', [])[-20:]
    }

@router.post("/execute")
async def execute_code(request: ExecuteRequest):
    """Execute code in sandbox or agent task"""
    # If agent_type is specified, route to agent execution
    if hasattr(request, 'agent_type') and request.agent_type:
        from src.services.agent_execution_service import agent_execution_service

        # Handle agent task execution
        task = getattr(request, 'task', request.code)  # Support both 'task' and 'code' fields
        parameters = getattr(request, 'parameters', {})
        if hasattr(request, 'code') and not parameters.get('code'):
            parameters['code'] = request.code

        result = await agent_execution_service.execute_task(
            agent_type=request.agent_type,
            task=task,
            parameters=parameters
        )
        return result

    # Otherwise do direct code execution
    if request.language.lower() not in ["python", "py"]:
        raise HTTPException(status_code=400, detail="Only Python execution supported currently")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(request.code)
        temp_path = f.name
    
    try:
        result = subprocess.run(
            ["python3", temp_path],
            capture_output=True,
            text=True,
            timeout=min(request.timeout, 60)  # Max 60 seconds
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[:5000],
            "stderr": result.stderr[:2000],
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Execution timed out ({request.timeout}s limit)"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        os.unlink(temp_path)

@router.get("/status")
async def get_agents_status():
    """Get status of all agents"""
    return {
        "agents": [
            {
                "name": "CodingAgent",
                "model": getattr(coding_agent, 'model_name', 'unknown'),
                "status": "active",
                "tasks_processed": len(coding_agent.history) if hasattr(coding_agent, 'history') else 0
            },
            {
                "name": "ReasoningAgent",
                "model": getattr(reasoning_agent, 'model_name', 'unknown'),
                "status": "active",
                "tasks_processed": len(reasoning_agent.history) if hasattr(reasoning_agent, 'history') else 0
            },
            {
                "name": "NarrationAgent",
                "model": getattr(narration_agent, 'model_name', 'unknown'),
                "status": "active",
                "tasks_processed": len(narration_agent.history) if hasattr(narration_agent, 'history') else 0
            }
        ]
    }

@router.post("/narration/anime")
async def narrate_and_generate(request: Dict[str, Any]):
    """Develop narrative and generate anime image."""
    scene_concept = request.get("scene_concept", "")
    project_id = request.get("project_id")

    if not scene_concept:
        raise HTTPException(status_code=400, detail="scene_concept is required")

    result = await narration_agent.generate_anime_scene(scene_concept, project_id)
    return result
