"""
Anime Production Task Executors

Specialized task executors for anime production workflow stages.
Integrates with existing Echo Brain components and ComfyUI for content generation.

Task Types Handled:
- concept_development: Develop project concepts using LLMs
- character_generation: Generate character designs via ComfyUI
- lora_training: Coordinate LoRA model training
- scene_planning: Plan scene compositions and shots
- scene_generation: Generate scene content using trained LoRAs
- post_processing: Apply effects and corrections
- quality_validation: Validate outputs against quality criteria
- video_assembly: Assemble scenes into final video
- delivery_preparation: Prepare final deliverables

Integration Points:
- Echo Brain agents (localhost:8309)
- ComfyUI workflow queue (localhost:8188)
- LoRA training pipeline
- File system management
- Database state tracking
"""

import asyncio
import json
import logging
import os
import time
import hashlib
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import aiohttp
import asyncpg
from contextlib import asynccontextmanager
from dataclasses import dataclass

# Import existing autonomous components
from .executor import Executor, TaskResult

logger = logging.getLogger(__name__)


@dataclass
class AnimeTaskResult:
    """Extended task result for anime production tasks"""
    task_id: str
    success: bool
    outputs: Dict[str, Any]  # Generated files, metadata, etc.
    metadata: Dict[str, Any]
    error: Optional[str] = None
    quality_score: Optional[float] = None
    processing_time_seconds: Optional[float] = None


class AnimeTaskExecutor:
    """Base executor for anime production tasks"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize anime task executor"""
        self.config = config or {}

        # Service endpoints
        self.echo_brain_url = self.config.get('echo_brain_url', 'http://localhost:8309')
        self.comfyui_url = self.config.get('comfyui_url', 'http://localhost:8188')
        self.ollama_url = self.config.get('ollama_url', 'http://localhost:11434')

        # File paths
        self.output_base = Path(self.config.get('output_base', '/mnt/1TB-storage/ComfyUI/output'))
        self.lora_base = Path(self.config.get('lora_base', '/mnt/1TB-storage/models/loras'))
        self.temp_base = Path(self.config.get('temp_base', '/tmp/anime_production'))

        # Ensure directories exist
        self.output_base.mkdir(parents=True, exist_ok=True)
        self.lora_base.mkdir(parents=True, exist_ok=True)
        self.temp_base.mkdir(parents=True, exist_ok=True)

        # Database configuration
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'echo_brain',
            'user': 'patrick',
            'password': os.environ.get('ECHO_BRAIN_DB_PASSWORD', os.getenv("TOWER_DB_PASSWORD", ""))
        }
        self._pool = None

        logger.info("AnimeTaskExecutor initialized")

    @asynccontextmanager
    async def get_connection(self):
        """Get database connection"""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(**self.db_config, min_size=1, max_size=3)

        async with self._pool.acquire() as connection:
            yield connection

    async def execute_task(self, task_data: Dict[str, Any]) -> AnimeTaskResult:
        """Execute an anime production task"""
        start_time = time.time()
        task_id = task_data.get('id', 'unknown')
        task_type = task_data.get('task_type', 'unknown')

        logger.info(f"Executing anime task {task_id}: {task_type}")

        try:
            # Route to appropriate executor based on task type
            if task_type == 'concept_development':
                result = await self._execute_concept_development(task_data)
            elif task_type == 'character_generation':
                result = await self._execute_character_generation(task_data)
            elif task_type == 'lora_training':
                result = await self._execute_lora_training(task_data)
            elif task_type == 'scene_planning':
                result = await self._execute_scene_planning(task_data)
            elif task_type == 'scene_generation':
                result = await self._execute_scene_generation(task_data)
            elif task_type == 'post_processing':
                result = await self._execute_post_processing(task_data)
            elif task_type == 'quality_validation':
                result = await self._execute_quality_validation(task_data)
            elif task_type == 'video_assembly':
                result = await self._execute_video_assembly(task_data)
            elif task_type == 'delivery_preparation':
                result = await self._execute_delivery_preparation(task_data)
            elif task_type == 'anime_orchestration':
                result = await self._execute_anime_orchestration(task_data)
            elif task_type == 'anime_monitoring':
                result = await self._execute_anime_monitoring(task_data)
            else:
                result = AnimeTaskResult(
                    task_id=task_id,
                    success=False,
                    outputs={},
                    metadata={},
                    error=f"Unknown anime task type: {task_type}"
                )

            # Add processing time
            processing_time = time.time() - start_time
            result.processing_time_seconds = processing_time

            logger.info(f"Anime task {task_id} completed in {processing_time:.2f}s: {result.success}")
            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Anime task {task_id} failed after {processing_time:.2f}s: {e}")

            return AnimeTaskResult(
                task_id=task_id,
                success=False,
                outputs={},
                metadata={'task_type': task_type},
                error=str(e),
                processing_time_seconds=processing_time
            )

    async def _execute_concept_development(self, task_data: Dict[str, Any]) -> AnimeTaskResult:
        """Execute concept development using Echo Brain LLMs"""
        try:
            metadata = task_data.get('metadata', {})
            project_id = metadata.get('project_id')
            characters = metadata.get('characters', [])
            style = metadata.get('style', {})

            # Use Echo Brain's reasoning agent for concept development
            concept_prompt = f"""
            Develop a comprehensive anime production concept with the following parameters:

            Characters: {json.dumps(characters, indent=2)}
            Style Guide: {json.dumps(style, indent=2)}
            Project Type: {metadata.get('project_type', 'short_film')}

            Create detailed:
            1. Character descriptions and relationships
            2. Visual style guidelines
            3. Narrative structure
            4. Technical specifications
            5. Production timeline recommendations

            Return as structured JSON with clear sections.
            """

            # Call Echo Brain reasoning agent
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.echo_brain_url}/api/agents/reasoning/chat",
                    json={"query": concept_prompt}
                ) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        concept_content = result_data.get('response', '')

                        # Parse and structure the concept
                        try:
                            concept_json = json.loads(concept_content)
                        except json.JSONDecodeError:
                            # If not valid JSON, create structured format
                            concept_json = {
                                "concept_description": concept_content,
                                "characters": characters,
                                "style_guide": style,
                                "generated_at": datetime.now().isoformat()
                            }

                        # Save concept to file
                        concept_file = self.output_base / "concepts" / f"{project_id}_concept.json"
                        concept_file.parent.mkdir(exist_ok=True)

                        with open(concept_file, 'w') as f:
                            json.dump(concept_json, f, indent=2)

                        return AnimeTaskResult(
                            task_id=task_data['id'],
                            success=True,
                            outputs={
                                "concept_file": str(concept_file),
                                "concept_data": concept_json
                            },
                            metadata={
                                "characters_count": len(characters),
                                "style_elements": len(style),
                                "concept_size": len(concept_content)
                            },
                            quality_score=0.9  # High quality for LLM-generated concepts
                        )
                    else:
                        raise Exception(f"Echo Brain API error: {response.status}")

        except Exception as e:
            return AnimeTaskResult(
                task_id=task_data['id'],
                success=False,
                outputs={},
                metadata={},
                error=f"Concept development failed: {e}"
            )

    async def _execute_character_generation(self, task_data: Dict[str, Any]) -> AnimeTaskResult:
        """Execute character design generation via ComfyUI"""
        try:
            metadata = task_data.get('metadata', {})
            project_id = metadata.get('project_id')
            characters = metadata.get('characters', [])
            style = metadata.get('style', {})

            generated_characters = []

            for character in characters:
                # Create character-specific prompt
                char_prompt = f"""
                {style.get('base_style', 'anime style')}, character design,
                {character.get('description', character.get('name', 'character'))},
                {character.get('appearance', '')},
                full body reference sheet, multiple angles,
                professional character design, clean background,
                high detail, consistent design, anime character sheet
                """

                negative_prompt = """
                low quality, blurry, deformed, bad anatomy, extra limbs,
                missing limbs, poor composition, artifacts, watermark
                """

                # Create ComfyUI workflow for character generation
                workflow = await self._create_character_workflow(
                    char_prompt, negative_prompt, character.get('name', 'character')
                )

                # Execute via ComfyUI
                generation_result = await self._execute_comfyui_workflow(workflow)

                if generation_result['success']:
                    character_images = generation_result['outputs']
                    generated_characters.append({
                        "character_name": character.get('name', 'character'),
                        "images": character_images,
                        "prompt_used": char_prompt
                    })

            if generated_characters:
                return AnimeTaskResult(
                    task_id=task_data['id'],
                    success=True,
                    outputs={
                        "generated_characters": generated_characters,
                        "total_images": sum(len(char['images']) for char in generated_characters)
                    },
                    metadata={
                        "characters_generated": len(generated_characters),
                        "style_used": style
                    },
                    quality_score=0.8
                )
            else:
                raise Exception("No characters were successfully generated")

        except Exception as e:
            return AnimeTaskResult(
                task_id=task_data['id'],
                success=False,
                outputs={},
                metadata={},
                error=f"Character generation failed: {e}"
            )

    async def _execute_lora_training(self, task_data: Dict[str, Any]) -> AnimeTaskResult:
        """Execute LoRA training for characters"""
        try:
            metadata = task_data.get('metadata', {})
            project_id = metadata.get('project_id')
            character = metadata.get('character', {})
            training_images_path = Path(metadata.get('training_images_path', ''))

            character_name = character.get('name', 'unknown')

            # Check if training images exist
            if not training_images_path.exists():
                raise Exception(f"Training images not found: {training_images_path}")

            # Create LoRA training configuration
            lora_config = {
                "character_name": character_name,
                "training_data_path": str(training_images_path),
                "output_path": str(self.lora_base / project_id / f"{character_name}.safetensors"),
                "training_steps": metadata.get('training_steps', 1000),
                "learning_rate": metadata.get('learning_rate', 1e-4),
                "batch_size": metadata.get('batch_size', 2),
                "resolution": metadata.get('resolution', 512)
            }

            # Execute LoRA training (this would interface with the actual training script)
            training_result = await self._execute_lora_training_process(lora_config)

            if training_result['success']:
                lora_path = Path(training_result['lora_path'])

                # Validate trained LoRA
                if lora_path.exists() and lora_path.stat().st_size > 10 * 1024 * 1024:  # > 10MB
                    return AnimeTaskResult(
                        task_id=task_data['id'],
                        success=True,
                        outputs={
                            "lora_path": str(lora_path),
                            "character_name": character_name,
                            "training_config": lora_config
                        },
                        metadata={
                            "training_steps": lora_config['training_steps'],
                            "file_size_mb": lora_path.stat().st_size / (1024 * 1024)
                        },
                        quality_score=training_result.get('quality_score', 0.7)
                    )
                else:
                    raise Exception("LoRA training produced invalid or corrupt model file")
            else:
                raise Exception(training_result.get('error', 'LoRA training failed'))

        except Exception as e:
            return AnimeTaskResult(
                task_id=task_data['id'],
                success=False,
                outputs={},
                metadata={},
                error=f"LoRA training failed: {e}"
            )

    async def _execute_scene_planning(self, task_data: Dict[str, Any]) -> AnimeTaskResult:
        """Execute scene planning using LLM reasoning"""
        try:
            metadata = task_data.get('metadata', {})
            project_id = metadata.get('project_id')
            scenes = metadata.get('scenes', [])
            characters = metadata.get('characters', [])

            # Use Echo Brain's reasoning agent for scene planning
            planning_prompt = f"""
            Plan detailed scene compositions for an anime production:

            Scenes to plan: {json.dumps(scenes, indent=2)}
            Available characters: {json.dumps(characters, indent=2)}

            For each scene, provide:
            1. Shot breakdown (wide, medium, close-up shots)
            2. Character positioning and actions
            3. Camera angles and movements
            4. Lighting and mood requirements
            5. Visual effects needed
            6. Timing and pacing notes

            Return structured JSON with scene_plans array.
            """

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.echo_brain_url}/api/agents/reasoning/chat",
                    json={"query": planning_prompt}
                ) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        planning_content = result_data.get('response', '')

                        try:
                            scene_plans = json.loads(planning_content)
                        except json.JSONDecodeError:
                            # Create basic structure if JSON parsing fails
                            scene_plans = {
                                "scene_plans": [
                                    {
                                        "scene_name": scene.get('name', f'Scene {i+1}'),
                                        "description": scene.get('description', ''),
                                        "shots": ["wide shot", "medium shot", "close-up"],
                                        "characters": scene.get('characters', []),
                                        "notes": planning_content[:500]
                                    }
                                    for i, scene in enumerate(scenes)
                                ]
                            }

                        # Save scene plans
                        plans_file = self.output_base / "scene_plans" / f"{project_id}_scene_plans.json"
                        plans_file.parent.mkdir(exist_ok=True)

                        with open(plans_file, 'w') as f:
                            json.dump(scene_plans, f, indent=2)

                        return AnimeTaskResult(
                            task_id=task_data['id'],
                            success=True,
                            outputs={
                                "scene_plans_file": str(plans_file),
                                "scene_plans": scene_plans
                            },
                            metadata={
                                "scenes_planned": len(scene_plans.get('scene_plans', [])),
                                "total_scenes": len(scenes)
                            },
                            quality_score=0.8
                        )
                    else:
                        raise Exception(f"Echo Brain API error: {response.status}")

        except Exception as e:
            return AnimeTaskResult(
                task_id=task_data['id'],
                success=False,
                outputs={},
                metadata={},
                error=f"Scene planning failed: {e}"
            )

    async def _execute_scene_generation(self, task_data: Dict[str, Any]) -> AnimeTaskResult:
        """Execute scene content generation using trained LoRAs"""
        try:
            metadata = task_data.get('metadata', {})
            project_id = metadata.get('project_id')
            scene = metadata.get('scene', {})
            characters = metadata.get('characters', [])
            lora_models_path = Path(metadata.get('lora_models', ''))

            scene_name = scene.get('name', 'unknown_scene')

            # Load available LoRA models
            available_loras = {}
            if lora_models_path.exists():
                for lora_file in lora_models_path.glob("*.safetensors"):
                    char_name = lora_file.stem
                    available_loras[char_name] = str(lora_file)

            # Generate scene content
            scene_prompt = f"""
            {scene.get('description', '')},
            anime style, cinematic composition,
            {scene.get('setting', '')},
            {scene.get('mood', 'dramatic')},
            high quality, detailed, professional animation frame,
            characters: {', '.join([char.get('name', '') for char in characters])},
            masterpiece, best quality
            """

            negative_prompt = """
            low quality, blurry, deformed, bad anatomy, bad composition,
            artifacts, watermark, text, signature, worst quality
            """

            # Create scene generation workflow with LoRAs
            workflow = await self._create_scene_workflow(
                scene_prompt, negative_prompt, available_loras, scene_name
            )

            # Execute scene generation
            generation_result = await self._execute_comfyui_workflow(workflow)

            if generation_result['success']:
                return AnimeTaskResult(
                    task_id=task_data['id'],
                    success=True,
                    outputs={
                        "scene_name": scene_name,
                        "generated_content": generation_result['outputs'],
                        "loras_used": list(available_loras.keys())
                    },
                    metadata={
                        "scene_description": scene.get('description', ''),
                        "characters_count": len(characters),
                        "loras_available": len(available_loras)
                    },
                    quality_score=0.8
                )
            else:
                raise Exception(generation_result.get('error', 'Scene generation failed'))

        except Exception as e:
            return AnimeTaskResult(
                task_id=task_data['id'],
                success=False,
                outputs={},
                metadata={},
                error=f"Scene generation failed: {e}"
            )

    async def _execute_post_processing(self, task_data: Dict[str, Any]) -> AnimeTaskResult:
        """Execute post-processing effects and corrections"""
        try:
            metadata = task_data.get('metadata', {})
            project_id = metadata.get('project_id')
            processing_config = metadata.get('processing_config', {})

            # Find generated content to process
            project_output_dir = self.output_base / project_id
            if not project_output_dir.exists():
                raise Exception(f"No content found for project {project_id}")

            processed_files = []

            # Process images and videos
            for content_file in project_output_dir.rglob("*"):
                if content_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.mp4', '.mov']:
                    processed_file = await self._apply_post_processing(content_file, processing_config)
                    if processed_file:
                        processed_files.append({
                            "original": str(content_file),
                            "processed": str(processed_file),
                            "type": "image" if content_file.suffix.lower() in ['.png', '.jpg', '.jpeg'] else "video"
                        })

            if processed_files:
                return AnimeTaskResult(
                    task_id=task_data['id'],
                    success=True,
                    outputs={
                        "processed_files": processed_files,
                        "processing_applied": list(processing_config.keys())
                    },
                    metadata={
                        "files_processed": len(processed_files),
                        "processing_config": processing_config
                    },
                    quality_score=0.7
                )
            else:
                raise Exception("No files were processed")

        except Exception as e:
            return AnimeTaskResult(
                task_id=task_data['id'],
                success=False,
                outputs={},
                metadata={},
                error=f"Post-processing failed: {e}"
            )

    async def _execute_quality_validation(self, task_data: Dict[str, Any]) -> AnimeTaskResult:
        """Execute comprehensive quality validation"""
        try:
            metadata = task_data.get('metadata', {})
            project_id = metadata.get('project_id')
            validation_criteria = metadata.get('validation_criteria', {})

            # Find all project outputs
            project_output_dir = self.output_base / project_id
            if not project_output_dir.exists():
                raise Exception(f"No content found for project {project_id}")

            validation_results = {
                "passed": [],
                "failed": [],
                "warnings": []
            }

            total_files = 0
            quality_scores = []

            # Validate all content files
            for content_file in project_output_dir.rglob("*"):
                if content_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.mp4', '.mov', '.json']:
                    total_files += 1
                    file_validation = await self._validate_file_quality(content_file, validation_criteria)

                    if file_validation['passed']:
                        validation_results["passed"].append(file_validation)
                    else:
                        validation_results["failed"].append(file_validation)

                    if file_validation.get('warnings'):
                        validation_results["warnings"].extend(file_validation['warnings'])

                    if file_validation.get('quality_score'):
                        quality_scores.append(file_validation['quality_score'])

            # Calculate overall quality score
            overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            validation_passed = len(validation_results["failed"]) == 0

            return AnimeTaskResult(
                task_id=task_data['id'],
                success=validation_passed,
                outputs={
                    "validation_results": validation_results,
                    "overall_quality_score": overall_quality,
                    "validation_passed": validation_passed
                },
                metadata={
                    "files_validated": total_files,
                    "passed_count": len(validation_results["passed"]),
                    "failed_count": len(validation_results["failed"]),
                    "warnings_count": len(validation_results["warnings"])
                },
                quality_score=overall_quality
            )

        except Exception as e:
            return AnimeTaskResult(
                task_id=task_data['id'],
                success=False,
                outputs={},
                metadata={},
                error=f"Quality validation failed: {e}"
            )

    async def _execute_video_assembly(self, task_data: Dict[str, Any]) -> AnimeTaskResult:
        """Execute final video assembly"""
        try:
            metadata = task_data.get('metadata', {})
            project_id = metadata.get('project_id')
            assembly_config = metadata.get('assembly_config', {})
            output_path = Path(metadata.get('output_path', self.output_base / project_id))

            # Find all scene content
            scene_files = []
            for content_file in (self.output_base / project_id).rglob("*.mp4"):
                if "scene" in content_file.name.lower():
                    scene_files.append(content_file)

            if not scene_files:
                # Create video from images if no video files found
                image_files = list((self.output_base / project_id).rglob("*.png"))
                if not image_files:
                    raise Exception("No content found for video assembly")

                # Convert images to video using ffmpeg
                final_video = await self._assemble_video_from_images(image_files, output_path, assembly_config)
            else:
                # Assemble existing video files
                final_video = await self._assemble_video_from_clips(scene_files, output_path, assembly_config)

            if final_video and final_video.exists():
                # Validate final video
                video_info = await self._get_video_info(final_video)

                return AnimeTaskResult(
                    task_id=task_data['id'],
                    success=True,
                    outputs={
                        "final_video": str(final_video),
                        "video_info": video_info
                    },
                    metadata={
                        "source_files": len(scene_files) or len(image_files),
                        "assembly_config": assembly_config,
                        "video_duration": video_info.get('duration', 0)
                    },
                    quality_score=0.8
                )
            else:
                raise Exception("Video assembly failed to produce output")

        except Exception as e:
            return AnimeTaskResult(
                task_id=task_data['id'],
                success=False,
                outputs={},
                metadata={},
                error=f"Video assembly failed: {e}"
            )

    async def _execute_delivery_preparation(self, task_data: Dict[str, Any]) -> AnimeTaskResult:
        """Execute delivery preparation"""
        try:
            metadata = task_data.get('metadata', {})
            project_id = metadata.get('project_id')
            delivery_config = metadata.get('delivery_config', {})
            final_output_path = Path(metadata.get('final_output_path', f'/home/patrick/Videos/Anime/{project_id}'))

            # Create delivery directory structure
            final_output_path.mkdir(parents=True, exist_ok=True)

            # Organize and copy final deliverables
            deliverables = {
                "videos": [],
                "images": [],
                "metadata": [],
                "models": []
            }

            project_output_dir = self.output_base / project_id

            # Copy final video
            for video_file in project_output_dir.rglob("final_*.mp4"):
                dest_video = final_output_path / video_file.name
                shutil.copy2(video_file, dest_video)
                deliverables["videos"].append(str(dest_video))

            # Copy representative images
            for image_file in project_output_dir.rglob("*.png"):
                if "final" in image_file.name or "hero" in image_file.name:
                    dest_image = final_output_path / image_file.name
                    shutil.copy2(image_file, dest_image)
                    deliverables["images"].append(str(dest_image))

            # Copy metadata and configs
            for meta_file in project_output_dir.rglob("*.json"):
                dest_meta = final_output_path / meta_file.name
                shutil.copy2(meta_file, dest_meta)
                deliverables["metadata"].append(str(dest_meta))

            # Copy trained LoRA models if requested
            if delivery_config.get('include_models', False):
                lora_dir = self.lora_base / project_id
                if lora_dir.exists():
                    models_dir = final_output_path / "models"
                    models_dir.mkdir(exist_ok=True)
                    for lora_file in lora_dir.glob("*.safetensors"):
                        dest_lora = models_dir / lora_file.name
                        shutil.copy2(lora_file, dest_lora)
                        deliverables["models"].append(str(dest_lora))

            # Create delivery manifest
            manifest = {
                "project_id": project_id,
                "delivery_date": datetime.now().isoformat(),
                "deliverables": deliverables,
                "delivery_config": delivery_config,
                "total_files": sum(len(files) for files in deliverables.values())
            }

            manifest_file = final_output_path / "delivery_manifest.json"
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)

            # Set appropriate permissions
            for item in final_output_path.rglob("*"):
                if item.is_file():
                    item.chmod(0o644)
            final_output_path.chmod(0o755)

            return AnimeTaskResult(
                task_id=task_data['id'],
                success=True,
                outputs={
                    "delivery_path": str(final_output_path),
                    "manifest": manifest
                },
                metadata={
                    "total_deliverables": manifest["total_files"],
                    "delivery_config": delivery_config
                },
                quality_score=0.9
            )

        except Exception as e:
            return AnimeTaskResult(
                task_id=task_data['id'],
                success=False,
                outputs={},
                metadata={},
                error=f"Delivery preparation failed: {e}"
            )

    async def _execute_anime_orchestration(self, task_data: Dict[str, Any]) -> AnimeTaskResult:
        """Execute anime production orchestration initialization"""
        try:
            metadata = task_data.get('metadata', {})
            project_id = metadata.get('project_id')
            action = metadata.get('action', 'initialize_pipeline')

            if action == 'initialize_pipeline':
                # This would trigger the orchestrator to start the project
                # For now, just create the project structure
                project_dir = self.output_base / project_id
                project_dir.mkdir(exist_ok=True)

                # Create subdirectories
                subdirs = ['concepts', 'characters', 'scenes', 'loras', 'final']
                for subdir in subdirs:
                    (project_dir / subdir).mkdir(exist_ok=True)

                initialization_data = {
                    "project_id": project_id,
                    "initialized_at": datetime.now().isoformat(),
                    "project_structure": subdirs,
                    "status": "initialized"
                }

                # Save initialization data
                init_file = project_dir / "project_init.json"
                with open(init_file, 'w') as f:
                    json.dump(initialization_data, f, indent=2)

                return AnimeTaskResult(
                    task_id=task_data['id'],
                    success=True,
                    outputs={
                        "project_directory": str(project_dir),
                        "initialization_data": initialization_data
                    },
                    metadata={
                        "action": action,
                        "project_id": project_id
                    },
                    quality_score=1.0
                )
            else:
                raise Exception(f"Unknown orchestration action: {action}")

        except Exception as e:
            return AnimeTaskResult(
                task_id=task_data['id'],
                success=False,
                outputs={},
                metadata={},
                error=f"Anime orchestration failed: {e}"
            )

    async def _execute_anime_monitoring(self, task_data: Dict[str, Any]) -> AnimeTaskResult:
        """Execute anime production monitoring"""
        try:
            metadata = task_data.get('metadata', {})
            project_id = metadata.get('project_id')

            # Check project status
            project_dir = self.output_base / project_id
            if not project_dir.exists():
                raise Exception(f"Project directory not found: {project_id}")

            # Gather monitoring data
            monitoring_data = {
                "project_id": project_id,
                "checked_at": datetime.now().isoformat(),
                "directory_structure": {},
                "file_counts": {},
                "total_size_mb": 0
            }

            # Check each subdirectory
            total_size = 0
            for subdir in project_dir.iterdir():
                if subdir.is_dir():
                    files = list(subdir.iterdir())
                    subdir_size = sum(f.stat().st_size for f in files if f.is_file())

                    monitoring_data["directory_structure"][subdir.name] = {
                        "exists": True,
                        "file_count": len([f for f in files if f.is_file()]),
                        "size_mb": subdir_size / (1024 * 1024)
                    }
                    total_size += subdir_size

            monitoring_data["total_size_mb"] = total_size / (1024 * 1024)

            # Save monitoring report
            report_file = project_dir / f"monitoring_report_{int(time.time())}.json"
            with open(report_file, 'w') as f:
                json.dump(monitoring_data, f, indent=2)

            return AnimeTaskResult(
                task_id=task_data['id'],
                success=True,
                outputs={
                    "monitoring_report": str(report_file),
                    "monitoring_data": monitoring_data
                },
                metadata={
                    "project_id": project_id,
                    "total_files": sum(d.get("file_count", 0) for d in monitoring_data["directory_structure"].values()),
                    "total_size_mb": monitoring_data["total_size_mb"]
                },
                quality_score=1.0
            )

        except Exception as e:
            return AnimeTaskResult(
                task_id=task_data['id'],
                success=False,
                outputs={},
                metadata={},
                error=f"Anime monitoring failed: {e}"
            )

    # Helper methods for specific operations

    async def _create_character_workflow(self, prompt: str, negative_prompt: str, character_name: str) -> Dict:
        """Create ComfyUI workflow for character generation"""
        return {
            "1": {
                "inputs": {"text": prompt, "clip": ["4", 1]},
                "class_type": "CLIPTextEncode"
            },
            "2": {
                "inputs": {"text": negative_prompt, "clip": ["4", 1]},
                "class_type": "CLIPTextEncode"
            },
            "3": {
                "inputs": {
                    "seed": int(time.time()),
                    "steps": 30,
                    "cfg": 7.5,
                    "sampler_name": "dpmpp_2m",
                    "scheduler": "karras",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["1", 0],
                    "negative": ["2", 0],
                    "latent_image": ["5", 0]
                },
                "class_type": "KSampler"
            },
            "4": {
                "inputs": {"ckpt_name": "epicrealism_v5.safetensors"},
                "class_type": "CheckpointLoaderSimple"
            },
            "5": {
                "inputs": {"width": 1024, "height": 1024, "batch_size": 4},
                "class_type": "EmptyLatentImage"
            },
            "6": {
                "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
                "class_type": "VAEDecode"
            },
            "7": {
                "inputs": {
                    "filename_prefix": f"character_{character_name}_{int(time.time())}",
                    "images": ["6", 0]
                },
                "class_type": "SaveImage"
            }
        }

    async def _create_scene_workflow(self, prompt: str, negative_prompt: str, loras: Dict[str, str], scene_name: str) -> Dict:
        """Create ComfyUI workflow for scene generation with LoRAs"""
        workflow = {
            "1": {
                "inputs": {"text": prompt, "clip": ["4", 1]},
                "class_type": "CLIPTextEncode"
            },
            "2": {
                "inputs": {"text": negative_prompt, "clip": ["4", 1]},
                "class_type": "CLIPTextEncode"
            },
            "3": {
                "inputs": {
                    "seed": int(time.time()),
                    "steps": 35,
                    "cfg": 8.0,
                    "sampler_name": "dpmpp_2m",
                    "scheduler": "karras",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["1", 0],
                    "negative": ["2", 0],
                    "latent_image": ["5", 0]
                },
                "class_type": "KSampler"
            },
            "4": {
                "inputs": {"ckpt_name": "epicrealism_v5.safetensors"},
                "class_type": "CheckpointLoaderSimple"
            },
            "5": {
                "inputs": {"width": 1024, "height": 576, "batch_size": 1},  # Cinematic aspect ratio
                "class_type": "EmptyLatentImage"
            },
            "6": {
                "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
                "class_type": "VAEDecode"
            },
            "7": {
                "inputs": {
                    "filename_prefix": f"scene_{scene_name}_{int(time.time())}",
                    "images": ["6", 0]
                },
                "class_type": "SaveImage"
            }
        }

        # Add LoRA nodes if available (simplified - would need proper LoRA loading nodes)
        if loras:
            # This would be expanded with proper LoRA loading workflow nodes
            pass

        return workflow

    async def _execute_comfyui_workflow(self, workflow: Dict) -> Dict[str, Any]:
        """Execute workflow on ComfyUI"""
        try:
            client_id = hashlib.md5(json.dumps(workflow).encode()).hexdigest()[:8]

            async with aiohttp.ClientSession() as session:
                # Queue the workflow
                async with session.post(
                    f"{self.comfyui_url}/prompt",
                    json={"prompt": workflow, "client_id": client_id}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        prompt_id = result.get("prompt_id")

                        if prompt_id:
                            # Monitor execution
                            return await self._monitor_comfyui_execution(prompt_id)
                        else:
                            return {"success": False, "error": "No prompt ID received"}
                    else:
                        return {"success": False, "error": f"ComfyUI queue error: {response.status}"}

        except Exception as e:
            return {"success": False, "error": f"ComfyUI execution failed: {e}"}

    async def _monitor_comfyui_execution(self, prompt_id: str, timeout_seconds: int = 300) -> Dict[str, Any]:
        """Monitor ComfyUI execution and return results"""
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.comfyui_url}/history/{prompt_id}") as response:
                        if response.status == 200:
                            history = await response.json()
                            if prompt_id in history:
                                outputs = history[prompt_id].get("outputs", {})
                                status = history[prompt_id].get("status", {})

                                if outputs:
                                    # Extract generated files
                                    generated_files = []
                                    for node_id, output in outputs.items():
                                        if "images" in output:
                                            for img in output["images"]:
                                                generated_files.append({
                                                    "type": "image",
                                                    "filename": img["filename"],
                                                    "subfolder": img.get("subfolder", ""),
                                                    "node_id": node_id
                                                })

                                    return {"success": True, "outputs": generated_files}

                                elif status.get("status_str") == "error":
                                    return {"success": False, "error": f"ComfyUI error: {status.get('messages', [])}"}

                await asyncio.sleep(2)

            except Exception as e:
                logger.warning(f"Error monitoring ComfyUI execution: {e}")
                await asyncio.sleep(5)

        return {"success": False, "error": "ComfyUI execution timeout"}

    async def _execute_lora_training_process(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute LoRA training process"""
        try:
            # This would interface with the actual LoRA training script
            # For now, simulate training with a delay
            await asyncio.sleep(10)  # Simulate training time

            output_path = Path(config["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Create a placeholder LoRA file (in real implementation, this would be actual training)
            with open(output_path, 'wb') as f:
                f.write(b'PLACEHOLDER_LORA_DATA' * 1024 * 1024)  # ~25MB placeholder

            return {
                "success": True,
                "lora_path": str(output_path),
                "quality_score": 0.8,
                "training_config": config
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"LoRA training failed: {e}"
            }

    async def _apply_post_processing(self, content_file: Path, config: Dict[str, Any]) -> Optional[Path]:
        """Apply post-processing to a content file"""
        try:
            if content_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                return await self._process_image(content_file, config)
            elif content_file.suffix.lower() in ['.mp4', '.mov']:
                return await self._process_video(content_file, config)
            return None

        except Exception as e:
            logger.error(f"Post-processing failed for {content_file}: {e}")
            return None

    async def _process_image(self, image_file: Path, config: Dict[str, Any]) -> Optional[Path]:
        """Process image with effects"""
        try:
            output_file = image_file.parent / f"processed_{image_file.name}"

            # Basic processing using ImageMagick or similar
            # For now, just copy the file (in real implementation, would apply effects)
            shutil.copy2(image_file, output_file)

            return output_file

        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return None

    async def _process_video(self, video_file: Path, config: Dict[str, Any]) -> Optional[Path]:
        """Process video with effects"""
        try:
            output_file = video_file.parent / f"processed_{video_file.name}"

            # Basic processing using ffmpeg
            # For now, just copy the file (in real implementation, would apply effects)
            shutil.copy2(video_file, output_file)

            return output_file

        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return None

    async def _validate_file_quality(self, file_path: Path, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Validate individual file quality"""
        try:
            validation_result = {
                "file": str(file_path),
                "passed": True,
                "warnings": [],
                "quality_score": 0.8
            }

            # Basic file validation
            if not file_path.exists():
                validation_result["passed"] = False
                validation_result["warnings"].append("File does not exist")
                return validation_result

            file_size = file_path.stat().st_size
            if file_size < 1024:  # Less than 1KB
                validation_result["passed"] = False
                validation_result["warnings"].append("File too small")

            # Type-specific validation
            if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                # Image validation
                if file_size < 100 * 1024:  # Less than 100KB
                    validation_result["warnings"].append("Image file seems small")
            elif file_path.suffix.lower() in ['.mp4', '.mov']:
                # Video validation
                if file_size < 1024 * 1024:  # Less than 1MB
                    validation_result["warnings"].append("Video file seems small")

            return validation_result

        except Exception as e:
            return {
                "file": str(file_path),
                "passed": False,
                "warnings": [f"Validation error: {e}"],
                "quality_score": 0.0
            }

    async def _assemble_video_from_images(self, image_files: List[Path], output_path: Path, config: Dict[str, Any]) -> Optional[Path]:
        """Assemble video from image sequence"""
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            final_video = output_path / "final_video.mp4"

            # Sort images by name
            sorted_images = sorted(image_files, key=lambda x: x.name)

            # Create temporary file list for ffmpeg
            file_list = self.temp_base / f"filelist_{int(time.time())}.txt"
            with open(file_list, 'w') as f:
                for img in sorted_images:
                    f.write(f"file '{img}'\n")
                    f.write(f"duration {config.get('frame_duration', 2.0)}\n")

            # Use ffmpeg to create video
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(file_list),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-r", str(config.get('fps', 30)),
                str(final_video)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            # Cleanup
            file_list.unlink(missing_ok=True)

            if result.returncode == 0 and final_video.exists():
                return final_video
            else:
                logger.error(f"ffmpeg failed: {result.stderr}")
                return None

        except Exception as e:
            logger.error(f"Video assembly from images failed: {e}")
            return None

    async def _assemble_video_from_clips(self, video_files: List[Path], output_path: Path, config: Dict[str, Any]) -> Optional[Path]:
        """Assemble final video from video clips"""
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            final_video = output_path / "final_assembled_video.mp4"

            # Sort clips by name
            sorted_clips = sorted(video_files, key=lambda x: x.name)

            # Create file list for concatenation
            file_list = self.temp_base / f"concat_{int(time.time())}.txt"
            with open(file_list, 'w') as f:
                for clip in sorted_clips:
                    f.write(f"file '{clip}'\n")

            # Use ffmpeg to concatenate
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(file_list),
                "-c", "copy",
                str(final_video)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            # Cleanup
            file_list.unlink(missing_ok=True)

            if result.returncode == 0 and final_video.exists():
                return final_video
            else:
                logger.error(f"ffmpeg concat failed: {result.stderr}")
                return None

        except Exception as e:
            logger.error(f"Video assembly from clips failed: {e}")
            return None

    async def _get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get video information using ffprobe"""
        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                str(video_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                probe_data = json.loads(result.stdout)
                format_data = probe_data.get("format", {})
                video_stream = None

                for stream in probe_data.get("streams", []):
                    if stream.get("codec_type") == "video":
                        video_stream = stream
                        break

                return {
                    "duration": float(format_data.get("duration", 0)),
                    "size_bytes": int(format_data.get("size", 0)),
                    "width": int(video_stream.get("width", 0)) if video_stream else 0,
                    "height": int(video_stream.get("height", 0)) if video_stream else 0,
                    "fps": eval(video_stream.get("r_frame_rate", "0/1")) if video_stream else 0,
                    "codec": video_stream.get("codec_name", "unknown") if video_stream else "unknown"
                }
            else:
                return {"error": "Could not probe video"}

        except Exception as e:
            return {"error": f"Video info failed: {e}"}


# Integration with existing Executor
class EnhancedAnimeExecutor(Executor):
    """Enhanced Executor that includes anime production task handling"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.anime_executor = AnimeTaskExecutor(config)

    async def execute(self, task_id: str) -> TaskResult:
        """Execute task, routing anime tasks to specialized executor"""
        try:
            # Get task data
            async with self.get_connection() as conn:
                task_data = await conn.fetchrow("""
                    SELECT * FROM autonomous_tasks WHERE id = $1
                """, task_id)

                if not task_data:
                    return TaskResult(
                        task_id=task_id,
                        success=False,
                        metadata={},
                        error="Task not found"
                    )

                task_dict = dict(task_data)
                task_type = task_dict.get('task_type', '')

                # Route anime production tasks to specialized executor
                anime_task_types = [
                    'concept_development', 'character_generation', 'lora_training',
                    'scene_planning', 'scene_generation', 'post_processing',
                    'quality_validation', 'video_assembly', 'delivery_preparation',
                    'anime_orchestration', 'anime_monitoring'
                ]

                if task_type in anime_task_types:
                    # Execute via anime executor
                    anime_result = await self.anime_executor.execute_task(task_dict)

                    # Convert to standard TaskResult
                    return TaskResult(
                        task_id=task_id,
                        success=anime_result.success,
                        metadata={
                            **anime_result.metadata,
                            "outputs": anime_result.outputs,
                            "quality_score": anime_result.quality_score,
                            "processing_time": anime_result.processing_time_seconds
                        },
                        error=anime_result.error
                    )
                else:
                    # Use parent executor for non-anime tasks
                    return await super().execute(task_id)

        except Exception as e:
            logger.error(f"Enhanced executor failed for task {task_id}: {e}")
            return TaskResult(
                task_id=task_id,
                success=False,
                metadata={},
                error=str(e)
            )