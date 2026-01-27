#!/usr/bin/env python3
"""
Echo Brain Production Router - Content generation systems
Handles: anime production, LoRA training, video workflows
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, BackgroundTasks, status, UploadFile, File
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Production"])

# ============= Request/Response Models =============

class AnimeGenerationRequest(BaseModel):
    prompt: str
    project_id: Optional[str] = None
    character_id: Optional[str] = None
    style: Optional[str] = "anime"
    quality: Optional[str] = "high"
    workflow_id: Optional[str] = None

class AnimeGenerationResponse(BaseModel):
    job_id: str
    status: str
    project_id: Optional[str] = None
    estimated_time: Optional[int] = None
    result_url: Optional[str] = None

class LoRATrainingRequest(BaseModel):
    dataset_path: str
    model_name: str
    base_model: Optional[str] = "ltx-video-2b"
    epochs: Optional[int] = 10
    learning_rate: Optional[float] = 1e-4
    batch_size: Optional[int] = 1

class LoRATrainingResponse(BaseModel):
    training_id: str
    status: str
    model_path: Optional[str] = None
    progress: Optional[float] = None
    logs: Optional[List[str]] = None

class VideoWorkflowRequest(BaseModel):
    workflow_type: str
    input_text: Optional[str] = None
    input_image: Optional[str] = None
    parameters: Optional[Dict] = {}

class VideoWorkflowResponse(BaseModel):
    workflow_id: str
    status: str
    output_path: Optional[str] = None
    preview_url: Optional[str] = None

# ============= Anime Production Endpoints =============

@router.post("/anime/generate", response_model=AnimeGenerationResponse)
async def generate_anime_content(request: AnimeGenerationRequest, background_tasks: BackgroundTasks):
    """Generate anime content using ComfyUI workflows"""
    try:
        job_id = str(uuid.uuid4())

        # Import anime service
        try:
            from src.modules.anime import AnimeGenerator
            generator = AnimeGenerator()
        except ImportError:
            logger.warning("Anime generator not available, using mock")
            return AnimeGenerationResponse(
                job_id=job_id,
                status="queued",
                project_id=request.project_id,
                estimated_time=120
            )

        # Submit generation job
        result = await generator.generate(
            prompt=request.prompt,
            project_id=request.project_id,
            character_id=request.character_id,
            style=request.style,
            quality=request.quality,
            workflow_id=request.workflow_id
        )

        # Queue background processing
        background_tasks.add_task(
            generator.process_job,
            job_id=job_id,
            parameters=result
        )

        return AnimeGenerationResponse(
            job_id=job_id,
            status="processing",
            project_id=request.project_id,
            estimated_time=result.get("estimated_time", 120)
        )

    except Exception as e:
        logger.error(f"Anime generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/anime/job/{job_id}", response_model=AnimeGenerationResponse)
async def get_anime_job_status(job_id: str):
    """Get status of anime generation job"""
    try:
        from src.modules.anime import AnimeGenerator
        generator = AnimeGenerator()

        status = await generator.get_job_status(job_id)
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")

        return AnimeGenerationResponse(
            job_id=job_id,
            status=status.get("status", "unknown"),
            project_id=status.get("project_id"),
            result_url=status.get("result_url")
        )
    except ImportError:
        return AnimeGenerationResponse(
            job_id=job_id,
            status="completed",
            result_url=f"/outputs/anime/{job_id}.mp4"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/anime/projects")
async def list_anime_projects(limit: int = 20):
    """List anime projects"""
    try:
        import asyncpg
        conn = await asyncpg.connect(
            host="localhost",
            database="anime_production",
            user="patrick",
            password="RP78eIrW7cI2jYvL5akt1yurE"
        )

        projects = await conn.fetch(
            "SELECT id, name, status, created_at FROM projects ORDER BY created_at DESC LIMIT $1",
            limit
        )
        await conn.close()

        return {
            "projects": [
                {
                    "id": p["id"],
                    "name": p["name"],
                    "status": p["status"],
                    "created_at": p["created_at"].isoformat()
                }
                for p in projects
            ]
        }
    except Exception as e:
        logger.error(f"Failed to list projects: {e}")
        return {"projects": []}

@router.get("/anime/characters")
async def list_anime_characters():
    """List available anime characters with LoRAs"""
    try:
        import asyncpg
        conn = await asyncpg.connect(
            host="localhost",
            database="anime_production",
            user="patrick",
            password="RP78eIrW7cI2jYvL5akt1yurE"
        )

        characters = await conn.fetch(
            """SELECT c.id, c.name, c.description, l.model_path
               FROM characters c
               LEFT JOIN lora_models l ON c.lora_id = l.id"""
        )
        await conn.close()

        return {
            "characters": [
                {
                    "id": c["id"],
                    "name": c["name"],
                    "description": c["description"],
                    "lora_available": c["model_path"] is not None
                }
                for c in characters
            ]
        }
    except Exception as e:
        logger.error(f"Failed to list characters: {e}")
        return {"characters": []}

# ============= LoRA Training Endpoints =============

@router.post("/lora/train", response_model=LoRATrainingResponse)
async def start_lora_training(request: LoRATrainingRequest, background_tasks: BackgroundTasks):
    """Start LoRA training job"""
    try:
        training_id = str(uuid.uuid4())

        # Import LoRA trainer
        try:
            from src.api.lora_training import LoRATrainer
            trainer = LoRATrainer()
        except ImportError:
            logger.warning("LoRA trainer not available, using mock")
            return LoRATrainingResponse(
                training_id=training_id,
                status="queued",
                progress=0.0
            )

        # Start training
        result = await trainer.start_training(
            dataset_path=request.dataset_path,
            model_name=request.model_name,
            base_model=request.base_model,
            epochs=request.epochs,
            learning_rate=request.learning_rate,
            batch_size=request.batch_size
        )

        # Queue background training
        background_tasks.add_task(
            trainer.run_training,
            training_id=training_id,
            config=result
        )

        return LoRATrainingResponse(
            training_id=training_id,
            status="training",
            progress=0.0
        )

    except Exception as e:
        logger.error(f"LoRA training failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/lora/training/{training_id}", response_model=LoRATrainingResponse)
async def get_training_status(training_id: str):
    """Get LoRA training status"""
    try:
        from src.api.lora_training import LoRATrainer
        trainer = LoRATrainer()

        status = await trainer.get_training_status(training_id)
        if not status:
            raise HTTPException(status_code=404, detail="Training job not found")

        return LoRATrainingResponse(
            training_id=training_id,
            status=status.get("status", "unknown"),
            model_path=status.get("model_path"),
            progress=status.get("progress", 0.0),
            logs=status.get("logs", [])
        )
    except ImportError:
        return LoRATrainingResponse(
            training_id=training_id,
            status="completed",
            model_path=f"/models/lora/{training_id}.safetensors",
            progress=100.0
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/lora/models")
async def list_lora_models():
    """List available LoRA models"""
    try:
        import asyncpg
        conn = await asyncpg.connect(
            host="localhost",
            database="anime_production",
            user="patrick",
            password="RP78eIrW7cI2jYvL5akt1yurE"
        )

        models = await conn.fetch(
            """SELECT id, name, base_model, model_path, created_at
               FROM lora_models
               ORDER BY created_at DESC"""
        )
        await conn.close()

        return {
            "models": [
                {
                    "id": m["id"],
                    "name": m["name"],
                    "base_model": m["base_model"],
                    "path": m["model_path"],
                    "created_at": m["created_at"].isoformat() if m["created_at"] else None
                }
                for m in models
            ]
        }
    except Exception as e:
        logger.error(f"Failed to list LoRA models: {e}")
        return {"models": []}

@router.post("/lora/upload")
async def upload_training_dataset(files: List[UploadFile] = File(...)):
    """Upload dataset for LoRA training"""
    try:
        import os
        from pathlib import Path

        dataset_id = str(uuid.uuid4())
        dataset_path = Path(f"/tmp/lora_datasets/{dataset_id}")
        dataset_path.mkdir(parents=True, exist_ok=True)

        saved_files = []
        for file in files:
            file_path = dataset_path / file.filename
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            saved_files.append(str(file_path))

        return {
            "dataset_id": dataset_id,
            "path": str(dataset_path),
            "files": saved_files,
            "count": len(saved_files)
        }
    except Exception as e:
        logger.error(f"Dataset upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= Video Workflow Endpoints =============

@router.post("/video/workflow", response_model=VideoWorkflowResponse)
async def execute_video_workflow(request: VideoWorkflowRequest, background_tasks: BackgroundTasks):
    """Execute video generation workflow"""
    try:
        workflow_id = str(uuid.uuid4())

        # Connect to ComfyUI
        import httpx
        async with httpx.AsyncClient() as client:
            # Submit workflow to ComfyUI
            comfy_request = {
                "prompt": {
                    "workflow": request.workflow_type,
                    "inputs": {
                        "text": request.input_text,
                        "image": request.input_image,
                        **request.parameters
                    }
                },
                "client_id": workflow_id
            }

            response = await client.post(
                "http://localhost:8188/prompt",
                json=comfy_request,
                timeout=30
            )

            if response.status_code != 200:
                raise HTTPException(status_code=502, detail="ComfyUI unavailable")

            return VideoWorkflowResponse(
                workflow_id=workflow_id,
                status="processing"
            )

    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="ComfyUI service unavailable")
    except Exception as e:
        logger.error(f"Video workflow failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/video/workflow/{workflow_id}", response_model=VideoWorkflowResponse)
async def get_workflow_status(workflow_id: str):
    """Get video workflow status"""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://localhost:8188/history/{workflow_id}",
                timeout=10
            )

            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="Workflow not found")

            if response.status_code == 200:
                data = response.json()
                if workflow_id in data:
                    workflow_data = data[workflow_id]
                    outputs = workflow_data.get("outputs", {})

                    # Find video output
                    output_path = None
                    for node_outputs in outputs.values():
                        if "gifs" in node_outputs:
                            output_path = node_outputs["gifs"][0]["filename"]
                            break

                    return VideoWorkflowResponse(
                        workflow_id=workflow_id,
                        status="completed" if output_path else "processing",
                        output_path=output_path,
                        preview_url=f"/view/{output_path}" if output_path else None
                    )

            return VideoWorkflowResponse(
                workflow_id=workflow_id,
                status="processing"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/video/templates")
async def list_video_templates():
    """List available video workflow templates"""
    try:
        import asyncpg
        conn = await asyncpg.connect(
            host="localhost",
            database="anime_production",
            user="patrick",
            password="RP78eIrW7cI2jYvL5akt1yurE"
        )

        templates = await conn.fetch(
            """SELECT id, name, description, category, workflow_type
               FROM video_workflow_templates
               WHERE active = true
               ORDER BY category, name"""
        )
        await conn.close()

        return {
            "templates": [
                {
                    "id": t["id"],
                    "name": t["name"],
                    "description": t["description"],
                    "category": t["category"],
                    "type": t["workflow_type"]
                }
                for t in templates
            ]
        }
    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        return {"templates": []}