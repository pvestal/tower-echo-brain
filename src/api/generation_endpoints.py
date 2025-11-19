#!/usr/bin/env python3
"""
Generation-specific API endpoints for Echo Brain
Handles LoRA training, workflow generation, and optimized image generation
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

# Import Patrick's modules
try:
    from src.patrick_content_generator import patrick_generator
    from src.patrick_preference_learner import preference_learner
    from src.lora_trainer import lora_trainer
    from src.workflow_generator import workflow_generator
    from src.lora_dataset_creator import lora_creator
except ImportError as e:
    logging.warning(f"Generation modules not available: {e}")
    patrick_generator = None

logger = logging.getLogger(__name__)
router = APIRouter()

class LoRATrainRequest(BaseModel):
    character: str
    project: str
    force_retrain: bool = False

class WorkflowGenerateRequest(BaseModel):
    workflow_type: str = "image"  # image, video, lora_enhanced
    prompt: str
    negative_prompt: str = ""
    character: Optional[str] = None
    project: Optional[str] = None
    lora_model: Optional[str] = None
    use_optimized: bool = True

class DatasetCreateRequest(BaseModel):
    character: str
    project: str

@router.post("/api/echo/lora/train")
async def train_lora(request: LoRATrainRequest, background_tasks: BackgroundTasks):
    """Start LoRA training for a character"""
    if not lora_trainer:
        raise HTTPException(status_code=503, detail="LoRA trainer not available")

    try:
        # Check if dataset exists
        from pathlib import Path
        dataset_dir = Path(f"/mnt/1TB-storage/lora_datasets/{request.project}_{request.character}")

        if not dataset_dir.exists():
            return {
                "error": "Dataset not found",
                "message": f"Please create a dataset for {request.character} first",
                "dataset_path": str(dataset_dir)
            }

        # Check if LoRA already exists and not forcing retrain
        lora_path = Path(f"/mnt/1TB-storage/ComfyUI/models/loras/patrick_{request.character}_lora.safetensors")

        if lora_path.exists() and not request.force_retrain:
            return {
                "status": "exists",
                "message": f"LoRA already trained for {request.character}",
                "lora_path": str(lora_path)
            }

        # Start training in background
        result = await lora_trainer.train_lora(request.character, request.project)

        return {
            "status": "training_started",
            "character": request.character,
            "project": request.project,
            "result": result
        }

    except Exception as e:
        logger.error(f"LoRA training error: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.get("/api/echo/lora/list")
async def list_loras():
    """List available LoRA models"""
    if not lora_trainer:
        raise HTTPException(status_code=503, detail="LoRA trainer not available")

    try:
        loras = lora_trainer.get_available_loras()
        return {
            "count": len(loras),
            "loras": loras
        }
    except Exception as e:
        logger.error(f"Failed to list LoRAs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/echo/workflow/generate")
async def generate_workflow(request: WorkflowGenerateRequest):
    """Generate a ComfyUI workflow"""
    if not workflow_generator:
        raise HTTPException(status_code=503, detail="Workflow generator not available")

    try:
        # Use optimized workflows if requested
        if request.use_optimized and request.workflow_type == "image":
            # Load optimized workflow
            from pathlib import Path
            optimized_path = Path("/mnt/1TB-storage/ComfyUI/workflows/patrick_characters/optimized_anime_image_fast.json")

            if optimized_path.exists():
                workflow = json.loads(optimized_path.read_text())

                # Update prompt in workflow
                for node_id, node in workflow.items():
                    if "Loader" in node.get("class_type", ""):
                        node["inputs"]["positive"] = request.prompt
                        if request.negative_prompt:
                            node["inputs"]["negative"] = request.negative_prompt
                        break

                return {
                    "workflow": workflow,
                    "optimized": True,
                    "expected_time": "4-6 seconds",
                    "type": request.workflow_type
                }

        # Generate standard workflow
        workflow = workflow_generator.generate_workflow(
            workflow_type=request.workflow_type,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            character=request.character,
            project=request.project,
            lora_model=request.lora_model
        )

        return {
            "workflow": workflow,
            "optimized": False,
            "expected_time": "5-8 seconds" if request.workflow_type == "image" else "unknown",
            "type": request.workflow_type
        }

    except Exception as e:
        logger.error(f"Workflow generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/echo/dataset/create")
async def create_dataset(request: DatasetCreateRequest):
    """Create a LoRA training dataset for a character"""
    if not lora_creator:
        raise HTTPException(status_code=503, detail="Dataset creator not available")

    try:
        result = lora_creator.create_character_dataset(request.character, request.project)

        # Update preferences
        if preference_learner and 'error' not in result:
            preference_learner.learn_from_generation(
                request.character,
                request.project,
                "dataset_creation",
                True
            )

        return result

    except Exception as e:
        logger.error(f"Dataset creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/echo/dataset/list")
async def list_datasets():
    """List available training datasets"""
    try:
        from pathlib import Path
        dataset_base = Path("/mnt/1TB-storage/lora_datasets")

        datasets = []
        for dataset_dir in dataset_base.glob("*"):
            if dataset_dir.is_dir():
                config_file = dataset_dir / "training_config.json"
                if config_file.exists():
                    config = json.loads(config_file.read_text())
                    datasets.append({
                        "character": config.get("character"),
                        "project": config.get("project"),
                        "num_images": config.get("num_images"),
                        "path": str(dataset_dir)
                    })

        return {
            "count": len(datasets),
            "datasets": datasets
        }

    except Exception as e:
        logger.error(f"Failed to list datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/echo/generate/optimized")
async def generate_optimized(character: str, project: str):
    """Generate character image using optimized workflow"""
    if not patrick_generator:
        raise HTTPException(status_code=503, detail="Generator not available")

    try:
        # Use patrick's optimized generator
        result = await patrick_generator.generate_character_image(project, character)

        # Learn from generation
        if preference_learner and 'error' not in result:
            preference_learner.learn_from_generation(character, project, "optimized", True)

        return {
            "status": result.get("status", "unknown"),
            "character": character,
            "project": project,
            "generation_time": "4-6 seconds (optimized)",
            "result": result
        }

    except Exception as e:
        logger.error(f"Optimized generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/echo/preferences")
async def get_preferences():
    """Get Patrick's learned preferences"""
    if not preference_learner:
        raise HTTPException(status_code=503, detail="Preference learner not available")

    try:
        preferences = preference_learner.preferences
        recommendation = preference_learner.get_recommendation()

        return {
            "preferences": preferences,
            "recommendation": recommendation,
            "total_generated": preferences.get("generation_stats", {}).get("total_generated", 0)
        }

    except Exception as e:
        logger.error(f"Failed to get preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Export router
__all__ = ['router']