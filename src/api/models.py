import asyncio
#!/usr/bin/env python3
"""
Model management API routes for Echo Brain
"""
import asyncio
import subprocess
import uuid
import logging
import time
import json
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

try:
    from ..db.models import QueryRequest
except ImportError:
    # Fallback for standalone testing
    from src.db.models import QueryRequest

logger = logging.getLogger(__name__)
router = APIRouter()

# Model management response models
class ModelInfo(BaseModel):
    name: str
    size: str
    modified: str
    id: Optional[str] = None
    digest: Optional[str] = None

class ModelListResponse(BaseModel):
    models: List[ModelInfo]
    total: int
    timestamp: str

class ModelManifest(BaseModel):
    name: str
    parameters: Dict[str, Any]
    template: Optional[str] = None
    system: Optional[str] = None
    details: Dict[str, Any]

@router.get("/api/models/list")
async def list_all_models():
    """List all available models (compatible endpoint)"""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            models = []

            # Skip header line
            for line in lines[1:]:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 3:
                        models.append(ModelInfo(
                            name=parts[0],
                            size=f"{parts[1]} {parts[2] if len(parts) > 2 else ''}".strip(),
                            modified=" ".join(parts[3:]) if len(parts) > 3 else "unknown"
                        ))

            return ModelListResponse(
                models=models,
                total=len(models),
                timestamp=datetime.now().isoformat()
            )
        else:
            raise HTTPException(status_code=500, detail=f"Ollama command failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Ollama command timed out")
    except Exception as e:
        logger.error(f"Model listing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@router.get("/api/models/manifests")
async def get_model_manifests():
    """Get model manifests and details"""
    try:
        # Get list of models first
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail="Failed to list models")

        lines = result.stdout.strip().split('\n')[1:]  # Skip header
        manifests = []

        for line in lines:
            if line.strip():
                parts = line.split()
                if len(parts) >= 1:
                    model_name = parts[0]
                    try:
                        # Get model info
                        info_result = subprocess.run(
                            ["ollama", "show", model_name],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )

                        if info_result.returncode == 0:
                            # Parse the model info - this is simplified
                            manifests.append(ModelManifest(
                                name=model_name,
                                parameters={
                                    "size": f"{parts[1]} {parts[2]}" if len(parts) >= 3 else "unknown",
                                    "modified": " ".join(parts[3:]) if len(parts) > 3 else "unknown"
                                },
                                details={
                                    "raw_info": info_result.stdout[:500] if info_result.stdout else ""
                                }
                            ))
                    except subprocess.TimeoutExpired:
                        # Skip models that timeout
                        continue
                    except Exception:
                        # Skip models that error
                        continue

        return {"manifests": manifests, "total": len(manifests)}

    except Exception as e:
        logger.error(f"Manifest retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get manifests: {str(e)}")

@router.get("/api/echo/models/list")
async def list_models():
    """List all available Ollama models"""
    try:

        # Direct ollama list command
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = []
            for line in lines:
                if line:
                    parts = line.split()
                    if len(parts) >= 4:
                        models.append({
                            "name": parts[0],
                            "size": f"{parts[2]} {parts[3]}",
                            "modified": " ".join(parts[4:]) if len(parts) > 4 else ""
                        })
            return models
        return {"error": "Failed to list models"}
    except Exception as e:
        logger.error(f"Model listing failed: {e}")
        return {"error": str(e)}

@router.post("/api/echo/models/pull/{model_name}")
async def pull_model(model_name: str, background_tasks: BackgroundTasks):
    """Pull a specific model"""
    logger.info(f"📥 Pulling model: {model_name}")

    try:
        async def pull_model_async(model: str):
            try:
                # Run ollama pull command
                result = subprocess.run(
                    ["ollama", "pull", model],
                    capture_output=True,
                    text=True,
                    timeout=1800  # 30 minute timeout for model pulls
                )

                if result.returncode == 0:
                    logger.info(f"✅ Model {model} pulled successfully")
                    return {"success": True, "stdout": result.stdout}
                else:
                    logger.error(f"❌ Failed to pull {model}: {result.stderr}")
                    return {"success": False, "stderr": result.stderr}
            except subprocess.TimeoutExpired:
                logger.error(f"❌ Model pull timed out for {model}")
                return {"success": False, "error": "Pull operation timed out"}
            except Exception as e:
                logger.error(f"❌ Error pulling {model}: {e}")
                return {"success": False, "error": str(e)}

        task_id = str(uuid.uuid4())
        background_tasks.add_task(pull_model_async, model_name)

        return {
            "success": True,
            "message": f"Model pull started for {model_name}",
            "task_id": task_id
        }
    except Exception as e:
        logger.error(f"Model pull failed: {e}")
        return {"success": False, "error": str(e)}

@router.delete("/api/echo/models/{model_name}")
async def remove_model(model_name: str):
    """Remove a specific model"""
    logger.info(f"🗑️ Removing model: {model_name}")

    try:
        result = subprocess.run(
            ["ollama", "rm", model_name],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            return {"success": True, "message": f"Model {model_name} removed successfully"}
        else:
            return {"success": False, "error": result.stderr or "Failed to remove model"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Remove operation timed out"}
    except Exception as e:
        logger.error(f"Model removal failed: {e}")
        return {"success": False, "error": str(e)}

@router.get("/api/echo/models/status/{request_id}")
async def get_model_operation_status(request_id: str):
    """Get status of a model operation"""
    try:
        # For now, return a simple status - in a real implementation
        # this would track background tasks
        return {
            "request_id": request_id,
            "status": "completed",
            "message": "Model operation status tracking not fully implemented"
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {"error": str(e)}

@router.post("/api/echo/code")
async def generate_code_only(request: QueryRequest):
    """Code generation endpoint - forces code output"""
    from src.api.echo import query_echo

    # Modify query to force code generation
    code_request = QueryRequest(
        query=f"Write only code, no explanations: {request.query}",
        conversation_id=request.conversation_id,
        user_id=request.user_id,
        intelligence_level="small"  # Use code model
    )
    # Call the regular query handler with modified request
    response = await query_echo(code_request)
    response.mode = "code_only"
    return response

# Additional model definitions
class FeedbackRequest(BaseModel):
    feedback: str
    rating: Optional[int] = None

class FeedbackResponse(BaseModel):
    success: bool
    message: str

class Filter(BaseModel):
    field: str
    value: str
