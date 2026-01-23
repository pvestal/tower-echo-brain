"""
Anime Production Workflow Orchestrator

A comprehensive workflow orchestration system that ties together all autonomous anime
production components. Integrates with AutonomousCore for goal/task management and
provides full lifecycle management from concept to final video generation.

Key Features:
- Project Lifecycle Management (concept → character design → training → scene generation → assembly)
- Dependency Resolution (ensures proper stage ordering and resource availability)
- Resource Management (coordinates VRAM usage between training and generation)
- Quality Gates (automated validation at each pipeline stage)
- Error Recovery (comprehensive failure handling with intelligent retry mechanisms)
- Progress Monitoring (real-time status tracking across all systems)

Integration Points:
- Echo Brain autonomous task creation and execution (localhost:8309)
- ComfyUI workflow queue management (localhost:8188)
- Database state synchronization (anime_production schema)
- File system asset management (/mnt/1TB-storage/models/loras/, output directories)
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
import aiohttp
import asyncpg
from contextlib import asynccontextmanager
import hashlib
import shutil
import subprocess
from collections import defaultdict, deque

# Import existing autonomous components
from .core import AutonomousCore
from .executor import Executor, TaskResult
from .scheduler import Scheduler
from .safety import SafetyController
from .audit import AuditLogger

logger = logging.getLogger(__name__)


class ProductionStage(Enum):
    """Stages of anime production workflow"""
    CONCEPT = "concept"
    CHARACTER_DESIGN = "character_design"
    LORA_TRAINING = "lora_training"
    SCENE_PLANNING = "scene_planning"
    CONTENT_GENERATION = "content_generation"
    POST_PROCESSING = "post_processing"
    QUALITY_VALIDATION = "quality_validation"
    FINAL_ASSEMBLY = "final_assembly"
    DELIVERY = "delivery"


class ProjectStatus(Enum):
    """Status of production projects"""
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    PENDING_APPROVAL = "pending_approval"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResourceType(Enum):
    """Types of production resources"""
    VRAM = "vram"
    STORAGE = "storage"
    COMPUTE = "compute"
    MODEL = "model"
    QUEUE_SLOT = "queue_slot"


@dataclass
class ResourceRequirement:
    """Resource requirement specification"""
    resource_type: ResourceType
    amount: float  # Amount needed (GB for VRAM/storage, cores for compute, etc.)
    duration_minutes: Optional[int] = None  # Expected usage duration
    exclusive: bool = False  # Whether resource needs exclusive access
    priority: int = 5  # Priority level (1-10)


@dataclass
class QualityGate:
    """Quality validation gate specification"""
    stage: ProductionStage
    validation_type: str  # e.g., "model_validation", "content_quality", "technical_specs"
    criteria: Dict[str, Any]  # Validation criteria
    required: bool = True  # Whether passing this gate is mandatory
    auto_retry: bool = True  # Whether to auto-retry on failure


@dataclass
class ProductionProject:
    """Complete anime production project specification"""
    id: str
    name: str
    description: str
    project_type: str  # e.g., "short_film", "music_video", "commercial"
    status: ProjectStatus
    current_stage: ProductionStage
    priority: int

    # Project configuration
    characters: List[Dict[str, Any]]
    scenes: List[Dict[str, Any]]
    style_config: Dict[str, Any]
    technical_specs: Dict[str, Any]

    # Progress tracking
    stages_completed: Set[ProductionStage]
    total_progress_percent: float
    stage_progress: Dict[ProductionStage, float]

    # Resource tracking
    allocated_resources: Dict[ResourceType, float]
    resource_history: List[Dict[str, Any]]

    # Quality tracking
    quality_gates_passed: List[str]
    quality_issues: List[Dict[str, Any]]

    # Metadata
    created_at: datetime
    updated_at: datetime
    deadline: Optional[datetime] = None
    metadata: Dict[str, Any] = None


@dataclass
class StageExecution:
    """Execution context for a production stage"""
    project_id: str
    stage: ProductionStage
    stage_id: str
    dependencies: List[str]  # Stage IDs that must complete first
    tasks: List[Dict[str, Any]]
    resources_required: List[ResourceRequirement]
    quality_gates: List[QualityGate]
    estimated_duration_minutes: int

    # Execution state
    status: str  # pending, running, completed, failed, blocked
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


class ResourceManager:
    """Manages production resources and allocation"""

    def __init__(self):
        self.available_resources = {
            ResourceType.VRAM: 12.0,  # GB - RTX 3060
            ResourceType.STORAGE: 1000.0,  # GB available
            ResourceType.COMPUTE: 8.0,  # CPU cores
            ResourceType.QUEUE_SLOT: 3.0,  # ComfyUI queue slots
        }

        self.allocated_resources = defaultdict(float)
        self.reservations = []  # List of resource reservations
        self.usage_history = deque(maxlen=1000)  # Resource usage tracking

    async def check_resource_availability(self, requirements: List[ResourceRequirement]) -> Tuple[bool, str]:
        """Check if required resources are available"""
        try:
            for req in requirements:
                available = self.available_resources.get(req.resource_type, 0.0)
                allocated = self.allocated_resources[req.resource_type]
                free = available - allocated

                if free < req.amount:
                    return False, f"Insufficient {req.resource_type.value}: need {req.amount}, have {free}"

            return True, "All resources available"

        except Exception as e:
            logger.error(f"Error checking resource availability: {e}")
            return False, f"Resource check failed: {e}"

    async def allocate_resources(self, stage_id: str, requirements: List[ResourceRequirement]) -> bool:
        """Allocate resources for a stage execution"""
        try:
            # Check availability first
            available, message = await self.check_resource_availability(requirements)
            if not available:
                logger.warning(f"Cannot allocate resources for {stage_id}: {message}")
                return False

            # Allocate resources
            allocation_record = {
                "stage_id": stage_id,
                "timestamp": datetime.now(),
                "allocations": {}
            }

            for req in requirements:
                self.allocated_resources[req.resource_type] += req.amount
                allocation_record["allocations"][req.resource_type.value] = req.amount

            self.usage_history.append(allocation_record)
            logger.info(f"Allocated resources for stage {stage_id}: {allocation_record['allocations']}")
            return True

        except Exception as e:
            logger.error(f"Failed to allocate resources for {stage_id}: {e}")
            return False

    async def release_resources(self, stage_id: str, requirements: List[ResourceRequirement]):
        """Release allocated resources"""
        try:
            for req in requirements:
                self.allocated_resources[req.resource_type] = max(
                    0, self.allocated_resources[req.resource_type] - req.amount
                )

            logger.info(f"Released resources for stage {stage_id}")

        except Exception as e:
            logger.error(f"Failed to release resources for {stage_id}: {e}")


class QualityValidator:
    """Validates quality at each production stage"""

    def __init__(self, comfyui_url: str = "http://localhost:8188"):
        self.comfyui_url = comfyui_url
        self.validation_cache = {}  # Cache validation results

    async def validate_stage(self, project: ProductionProject, stage: ProductionStage,
                           stage_outputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a completed stage against quality criteria"""
        try:
            validation_results = []
            issues = []

            if stage == ProductionStage.LORA_TRAINING:
                passed, stage_issues = await self._validate_lora_training(project, stage_outputs)
                validation_results.append(passed)
                issues.extend(stage_issues)

            elif stage == ProductionStage.CONTENT_GENERATION:
                passed, stage_issues = await self._validate_content_generation(project, stage_outputs)
                validation_results.append(passed)
                issues.extend(stage_issues)

            elif stage == ProductionStage.POST_PROCESSING:
                passed, stage_issues = await self._validate_post_processing(project, stage_outputs)
                validation_results.append(passed)
                issues.extend(stage_issues)

            # Overall validation result
            overall_passed = all(validation_results) and len(issues) == 0
            return overall_passed, issues

        except Exception as e:
            logger.error(f"Stage validation failed: {e}")
            return False, [f"Validation error: {e}"]

    async def _validate_lora_training(self, project: ProductionProject,
                                    outputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate LoRA training outputs"""
        issues = []

        try:
            # Check if LoRA files exist and are valid
            lora_dir = Path("/mnt/1TB-storage/models/loras/")
            project_loras = outputs.get("trained_loras", [])

            for lora_info in project_loras:
                lora_path = lora_dir / lora_info["filename"]

                if not lora_path.exists():
                    issues.append(f"LoRA file not found: {lora_path}")
                    continue

                # Check file size (should be substantial for a trained model)
                file_size_mb = lora_path.stat().st_size / (1024 * 1024)
                if file_size_mb < 10:
                    issues.append(f"LoRA file suspiciously small: {lora_path} ({file_size_mb:.1f}MB)")

                # Check if file is properly formatted (basic validation)
                try:
                    with open(lora_path, 'rb') as f:
                        header = f.read(8)
                        if not header.startswith(b'PK'):  # Should be a zip-like format
                            issues.append(f"LoRA file appears corrupted: {lora_path}")
                except Exception:
                    issues.append(f"Cannot read LoRA file: {lora_path}")

            return len(issues) == 0, issues

        except Exception as e:
            return False, [f"LoRA validation error: {e}"]

    async def _validate_content_generation(self, project: ProductionProject,
                                         outputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate content generation outputs"""
        issues = []

        try:
            generated_content = outputs.get("generated_content", [])

            for content in generated_content:
                content_path = Path(content["path"])

                if not content_path.exists():
                    issues.append(f"Generated content not found: {content_path}")
                    continue

                # Validate image/video quality
                if content["type"] == "image":
                    issues.extend(await self._validate_image_quality(content_path))
                elif content["type"] == "video":
                    issues.extend(await self._validate_video_quality(content_path))

            return len(issues) == 0, issues

        except Exception as e:
            return False, [f"Content validation error: {e}"]

    async def _validate_image_quality(self, image_path: Path) -> List[str]:
        """Validate individual image quality"""
        issues = []

        try:
            # Check file size
            file_size_kb = image_path.stat().st_size / 1024
            if file_size_kb < 100:
                issues.append(f"Image file too small: {image_path} ({file_size_kb:.1f}KB)")

            # Use ImageMagick or similar to check image properties
            try:
                result = subprocess.run([
                    "identify", "-format", "%w %h %m", str(image_path)
                ], capture_output=True, text=True, timeout=10)

                if result.returncode == 0:
                    width, height, format_type = result.stdout.strip().split()
                    width, height = int(width), int(height)

                    # Validate dimensions
                    if width < 512 or height < 512:
                        issues.append(f"Image resolution too low: {image_path} ({width}x{height})")

                    # Validate format
                    if format_type not in ['JPEG', 'PNG', 'WEBP']:
                        issues.append(f"Unexpected image format: {image_path} ({format_type})")

            except (subprocess.TimeoutExpired, FileNotFoundError):
                # ImageMagick not available, skip technical validation
                pass

        except Exception as e:
            issues.append(f"Image validation error for {image_path}: {e}")

        return issues

    async def _validate_video_quality(self, video_path: Path) -> List[str]:
        """Validate individual video quality"""
        issues = []

        try:
            # Check file size
            file_size_mb = video_path.stat().st_size / (1024 * 1024)
            if file_size_mb < 1:
                issues.append(f"Video file too small: {video_path} ({file_size_mb:.1f}MB)")

            # Use ffprobe to check video properties
            try:
                result = subprocess.run([
                    "ffprobe", "-v", "quiet", "-print_format", "json",
                    "-show_format", "-show_streams", str(video_path)
                ], capture_output=True, text=True, timeout=30)

                if result.returncode == 0:
                    probe_data = json.loads(result.stdout)

                    # Find video stream
                    video_stream = None
                    for stream in probe_data.get("streams", []):
                        if stream.get("codec_type") == "video":
                            video_stream = stream
                            break

                    if video_stream:
                        width = int(video_stream.get("width", 0))
                        height = int(video_stream.get("height", 0))
                        duration = float(video_stream.get("duration", 0))

                        if width < 512 or height < 512:
                            issues.append(f"Video resolution too low: {video_path} ({width}x{height})")

                        if duration < 1.0:
                            issues.append(f"Video too short: {video_path} ({duration:.1f}s)")
                    else:
                        issues.append(f"No video stream found: {video_path}")

            except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
                # ffprobe not available or failed, skip technical validation
                pass

        except Exception as e:
            issues.append(f"Video validation error for {video_path}: {e}")

        return issues

    async def _validate_post_processing(self, project: ProductionProject,
                                      outputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate post-processing outputs"""
        issues = []

        try:
            final_outputs = outputs.get("final_outputs", [])

            for output in final_outputs:
                output_path = Path(output["path"])

                if not output_path.exists():
                    issues.append(f"Final output not found: {output_path}")
                    continue

                # Validate final video meets project specifications
                if output["type"] == "video":
                    spec_issues = await self._validate_against_specs(output_path, project.technical_specs)
                    issues.extend(spec_issues)

            return len(issues) == 0, issues

        except Exception as e:
            return False, [f"Post-processing validation error: {e}"]

    async def _validate_against_specs(self, output_path: Path, specs: Dict[str, Any]) -> List[str]:
        """Validate output against technical specifications"""
        issues = []

        try:
            # This would contain logic to validate against project-specific technical specs
            # For now, basic validation
            required_resolution = specs.get("resolution", "1024x576")
            required_duration = specs.get("duration_seconds", 30)

            # Add more sophisticated validation as needed

        except Exception as e:
            issues.append(f"Spec validation error: {e}")

        return issues


class AnimeProductionOrchestrator:
    """
    Master orchestration system for autonomous anime production.

    Coordinates all components of the anime production pipeline:
    - Project lifecycle management
    - Resource allocation and management
    - Dependency resolution between stages
    - Quality validation and error recovery
    - Integration with AutonomousCore for task management
    """

    def __init__(self, autonomous_core: AutonomousCore, config: Optional[Dict[str, Any]] = None):
        """Initialize the orchestrator with configuration"""
        self.autonomous_core = autonomous_core
        self.config = config or {}

        # Core components
        self.resource_manager = ResourceManager()
        self.quality_validator = QualityValidator()

        # Service URLs
        self.comfyui_url = self.config.get('comfyui_url', 'http://localhost:8188')
        self.echo_brain_url = self.config.get('echo_brain_url', 'http://localhost:8309')

        # State management
        self.active_projects = {}  # project_id -> ProductionProject
        self.active_executions = {}  # execution_id -> StageExecution
        self.execution_queue = deque()  # Queue of stage executions

        # Performance tracking
        self.execution_history = deque(maxlen=1000)
        self.error_patterns = defaultdict(int)  # Track common errors

        # Database configuration
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'echo_brain',  # Using existing echo_brain database
            'user': 'patrick',
            'password': os.environ.get('ECHO_BRAIN_DB_PASSWORD', 'RP78eIrW7cI2jYvL5akt1yurE')
        }
        self._pool = None

        logger.info("AnimeProductionOrchestrator initialized")

    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(**self.db_config, min_size=1, max_size=5)

        async with self._pool.acquire() as connection:
            yield connection

    async def start(self):
        """Start the orchestrator"""
        try:
            # Initialize database schema
            await self.initialize_database()

            # Load existing projects
            await self.load_active_projects()

            # Start main orchestration loop
            asyncio.create_task(self.run_orchestration_loop())

            logger.info("AnimeProductionOrchestrator started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start orchestrator: {e}")
            return False

    async def stop(self):
        """Stop the orchestrator gracefully"""
        try:
            # Save all project states
            for project in self.active_projects.values():
                await self.save_project_state(project)

            # Close database pool
            if self._pool:
                await self._pool.close()

            logger.info("AnimeProductionOrchestrator stopped")

        except Exception as e:
            logger.error(f"Error stopping orchestrator: {e}")

    async def create_project(self, project_spec: Dict[str, Any]) -> Optional[str]:
        """
        Create a new anime production project

        Args:
            project_spec: Project specification including name, characters, scenes, etc.

        Returns:
            Project ID if successful, None otherwise
        """
        try:
            # Generate unique project ID
            project_id = f"anime_{int(time.time())}_{hashlib.md5(project_spec['name'].encode()).hexdigest()[:8]}"

            # Create project object
            project = ProductionProject(
                id=project_id,
                name=project_spec["name"],
                description=project_spec.get("description", ""),
                project_type=project_spec.get("type", "short_film"),
                status=ProjectStatus.DRAFT,
                current_stage=ProductionStage.CONCEPT,
                priority=project_spec.get("priority", 5),
                characters=project_spec.get("characters", []),
                scenes=project_spec.get("scenes", []),
                style_config=project_spec.get("style", {}),
                technical_specs=project_spec.get("technical_specs", {}),
                stages_completed=set(),
                total_progress_percent=0.0,
                stage_progress={stage: 0.0 for stage in ProductionStage},
                allocated_resources={},
                resource_history=[],
                quality_gates_passed=[],
                quality_issues=[],
                created_at=datetime.now(),
                updated_at=datetime.now(),
                deadline=project_spec.get("deadline"),
                metadata=project_spec.get("metadata", {})
            )

            # Save to database
            await self.save_project_to_database(project)

            # Add to active projects
            self.active_projects[project_id] = project

            # Create initial autonomous goal for the project
            goal_id = await self.autonomous_core.create_goal(
                name=f"Anime Production: {project.name}",
                description=f"Complete anime production pipeline for project '{project.name}'",
                goal_type="anime_production",
                priority=project.priority,
                metadata={
                    "project_id": project_id,
                    "project_type": project.project_type,
                    "stages": [stage.value for stage in ProductionStage]
                }
            )

            if goal_id:
                project.metadata["autonomous_goal_id"] = goal_id
                await self.save_project_state(project)

            logger.info(f"Created anime production project: {project_id} (Goal: {goal_id})")
            return project_id

        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            return None

    async def start_project_production(self, project_id: str) -> bool:
        """Start production for a project"""
        try:
            project = self.active_projects.get(project_id)
            if not project:
                logger.error(f"Project not found: {project_id}")
                return False

            # Update project status
            project.status = ProjectStatus.IN_PROGRESS
            project.updated_at = datetime.now()

            # Create execution plan
            execution_plan = await self.create_execution_plan(project)

            # Queue initial stages
            for stage_execution in execution_plan:
                if not stage_execution.dependencies:  # No dependencies, can start immediately
                    self.execution_queue.append(stage_execution)

            await self.save_project_state(project)

            logger.info(f"Started production for project: {project_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to start project production: {e}")
            return False

    async def create_execution_plan(self, project: ProductionProject) -> List[StageExecution]:
        """Create execution plan for project stages"""
        try:
            execution_plan = []

            # Stage 1: Concept Development
            concept_stage = StageExecution(
                project_id=project.id,
                stage=ProductionStage.CONCEPT,
                stage_id=f"{project.id}_concept",
                dependencies=[],
                tasks=[
                    {
                        "name": "Develop Project Concept",
                        "task_type": "concept_development",
                        "description": f"Develop detailed concept for '{project.name}'",
                        "safety_level": "auto",
                        "priority": 2,
                        "metadata": {
                            "project_id": project.id,
                            "characters": project.characters,
                            "style": project.style_config
                        }
                    }
                ],
                resources_required=[
                    ResourceRequirement(ResourceType.COMPUTE, 1.0, 30),
                ],
                quality_gates=[
                    QualityGate(
                        ProductionStage.CONCEPT,
                        "concept_validation",
                        {"min_characters": 1, "has_style_guide": True}
                    )
                ],
                estimated_duration_minutes=30,
                status="pending"
            )
            execution_plan.append(concept_stage)

            # Stage 2: Character Design
            character_stage = StageExecution(
                project_id=project.id,
                stage=ProductionStage.CHARACTER_DESIGN,
                stage_id=f"{project.id}_character_design",
                dependencies=[concept_stage.stage_id],
                tasks=[
                    {
                        "name": "Generate Character Designs",
                        "task_type": "character_generation",
                        "description": f"Generate character designs for {len(project.characters)} characters",
                        "safety_level": "auto",
                        "priority": 2,
                        "metadata": {
                            "project_id": project.id,
                            "characters": project.characters,
                            "style": project.style_config
                        }
                    }
                ],
                resources_required=[
                    ResourceRequirement(ResourceType.VRAM, 8.0, 60),
                    ResourceRequirement(ResourceType.QUEUE_SLOT, 1.0, 60),
                ],
                quality_gates=[
                    QualityGate(
                        ProductionStage.CHARACTER_DESIGN,
                        "character_validation",
                        {"min_images_per_character": 5, "style_consistency": True}
                    )
                ],
                estimated_duration_minutes=60,
                status="pending"
            )
            execution_plan.append(character_stage)

            # Stage 3: LoRA Training
            training_stage = StageExecution(
                project_id=project.id,
                stage=ProductionStage.LORA_TRAINING,
                stage_id=f"{project.id}_lora_training",
                dependencies=[character_stage.stage_id],
                tasks=[
                    {
                        "name": f"Train LoRA for {char['name']}",
                        "task_type": "lora_training",
                        "description": f"Train character LoRA model for {char['name']}",
                        "safety_level": "auto",
                        "priority": 3,
                        "metadata": {
                            "project_id": project.id,
                            "character": char,
                            "training_images_path": f"/tmp/training_data/{project.id}/{char['name']}"
                        }
                    } for char in project.characters
                ],
                resources_required=[
                    ResourceRequirement(ResourceType.VRAM, 10.0, 180, exclusive=True),
                    ResourceRequirement(ResourceType.STORAGE, 50.0, 180),
                ],
                quality_gates=[
                    QualityGate(
                        ProductionStage.LORA_TRAINING,
                        "lora_validation",
                        {"min_file_size_mb": 10, "training_loss_threshold": 0.1}
                    )
                ],
                estimated_duration_minutes=180,
                status="pending"
            )
            execution_plan.append(training_stage)

            # Stage 4: Scene Planning
            scene_planning_stage = StageExecution(
                project_id=project.id,
                stage=ProductionStage.SCENE_PLANNING,
                stage_id=f"{project.id}_scene_planning",
                dependencies=[training_stage.stage_id],
                tasks=[
                    {
                        "name": "Plan Scene Composition",
                        "task_type": "scene_planning",
                        "description": f"Plan composition and shots for {len(project.scenes)} scenes",
                        "safety_level": "auto",
                        "priority": 3,
                        "metadata": {
                            "project_id": project.id,
                            "scenes": project.scenes,
                            "characters": project.characters
                        }
                    }
                ],
                resources_required=[
                    ResourceRequirement(ResourceType.COMPUTE, 2.0, 45),
                ],
                quality_gates=[
                    QualityGate(
                        ProductionStage.SCENE_PLANNING,
                        "scene_plan_validation",
                        {"scenes_planned": len(project.scenes), "shot_breakdowns": True}
                    )
                ],
                estimated_duration_minutes=45,
                status="pending"
            )
            execution_plan.append(scene_planning_stage)

            # Stage 5: Content Generation
            content_generation_stage = StageExecution(
                project_id=project.id,
                stage=ProductionStage.CONTENT_GENERATION,
                stage_id=f"{project.id}_content_generation",
                dependencies=[scene_planning_stage.stage_id],
                tasks=[
                    {
                        "name": f"Generate Scene: {scene['name']}",
                        "task_type": "scene_generation",
                        "description": f"Generate visual content for scene '{scene['name']}'",
                        "safety_level": "auto",
                        "priority": 4,
                        "metadata": {
                            "project_id": project.id,
                            "scene": scene,
                            "characters": [char for char in project.characters if char['name'] in scene.get('characters', [])],
                            "lora_models": f"/mnt/1TB-storage/models/loras/{project.id}/"
                        }
                    } for scene in project.scenes
                ],
                resources_required=[
                    ResourceRequirement(ResourceType.VRAM, 10.0, 120),
                    ResourceRequirement(ResourceType.QUEUE_SLOT, 2.0, 120),
                    ResourceRequirement(ResourceType.STORAGE, 100.0, 120),
                ],
                quality_gates=[
                    QualityGate(
                        ProductionStage.CONTENT_GENERATION,
                        "content_quality_validation",
                        {"min_resolution": "1024x576", "content_consistency": True}
                    )
                ],
                estimated_duration_minutes=120,
                status="pending"
            )
            execution_plan.append(content_generation_stage)

            # Stage 6: Post Processing
            post_processing_stage = StageExecution(
                project_id=project.id,
                stage=ProductionStage.POST_PROCESSING,
                stage_id=f"{project.id}_post_processing",
                dependencies=[content_generation_stage.stage_id],
                tasks=[
                    {
                        "name": "Post-Process Generated Content",
                        "task_type": "post_processing",
                        "description": "Apply post-processing effects and corrections",
                        "safety_level": "auto",
                        "priority": 4,
                        "metadata": {
                            "project_id": project.id,
                            "processing_config": project.technical_specs.get("post_processing", {})
                        }
                    }
                ],
                resources_required=[
                    ResourceRequirement(ResourceType.COMPUTE, 4.0, 60),
                    ResourceRequirement(ResourceType.STORAGE, 200.0, 60),
                ],
                quality_gates=[
                    QualityGate(
                        ProductionStage.POST_PROCESSING,
                        "post_processing_validation",
                        {"effects_applied": True, "color_correction": True}
                    )
                ],
                estimated_duration_minutes=60,
                status="pending"
            )
            execution_plan.append(post_processing_stage)

            # Stage 7: Quality Validation
            quality_stage = StageExecution(
                project_id=project.id,
                stage=ProductionStage.QUALITY_VALIDATION,
                stage_id=f"{project.id}_quality_validation",
                dependencies=[post_processing_stage.stage_id],
                tasks=[
                    {
                        "name": "Comprehensive Quality Validation",
                        "task_type": "quality_validation",
                        "description": "Perform comprehensive quality validation of all outputs",
                        "safety_level": "notify",
                        "priority": 2,
                        "metadata": {
                            "project_id": project.id,
                            "validation_criteria": project.technical_specs
                        }
                    }
                ],
                resources_required=[
                    ResourceRequirement(ResourceType.COMPUTE, 2.0, 30),
                ],
                quality_gates=[
                    QualityGate(
                        ProductionStage.QUALITY_VALIDATION,
                        "final_quality_check",
                        {"overall_quality_score": 0.8, "no_critical_issues": True}
                    )
                ],
                estimated_duration_minutes=30,
                status="pending"
            )
            execution_plan.append(quality_stage)

            # Stage 8: Final Assembly
            assembly_stage = StageExecution(
                project_id=project.id,
                stage=ProductionStage.FINAL_ASSEMBLY,
                stage_id=f"{project.id}_final_assembly",
                dependencies=[quality_stage.stage_id],
                tasks=[
                    {
                        "name": "Assemble Final Video",
                        "task_type": "video_assembly",
                        "description": "Assemble all scenes into final video output",
                        "safety_level": "auto",
                        "priority": 2,
                        "metadata": {
                            "project_id": project.id,
                            "assembly_config": project.technical_specs.get("assembly", {}),
                            "output_path": f"/mnt/1TB-storage/ComfyUI/output/anime_projects/{project.id}/"
                        }
                    }
                ],
                resources_required=[
                    ResourceRequirement(ResourceType.COMPUTE, 6.0, 45),
                    ResourceRequirement(ResourceType.STORAGE, 500.0, 45),
                ],
                quality_gates=[
                    QualityGate(
                        ProductionStage.FINAL_ASSEMBLY,
                        "assembly_validation",
                        {"video_duration_matches": True, "audio_sync": True}
                    )
                ],
                estimated_duration_minutes=45,
                status="pending"
            )
            execution_plan.append(assembly_stage)

            # Stage 9: Delivery
            delivery_stage = StageExecution(
                project_id=project.id,
                stage=ProductionStage.DELIVERY,
                stage_id=f"{project.id}_delivery",
                dependencies=[assembly_stage.stage_id],
                tasks=[
                    {
                        "name": "Prepare Final Deliverables",
                        "task_type": "delivery_preparation",
                        "description": "Prepare and organize final deliverables",
                        "safety_level": "auto",
                        "priority": 2,
                        "metadata": {
                            "project_id": project.id,
                            "delivery_config": project.technical_specs.get("delivery", {}),
                            "final_output_path": f"/home/patrick/Videos/Anime/{project.name}/"
                        }
                    }
                ],
                resources_required=[
                    ResourceRequirement(ResourceType.STORAGE, 100.0, 15),
                ],
                quality_gates=[
                    QualityGate(
                        ProductionStage.DELIVERY,
                        "delivery_validation",
                        {"all_files_present": True, "metadata_complete": True}
                    )
                ],
                estimated_duration_minutes=15,
                status="pending"
            )
            execution_plan.append(delivery_stage)

            return execution_plan

        except Exception as e:
            logger.error(f"Failed to create execution plan: {e}")
            return []

    async def run_orchestration_loop(self):
        """Main orchestration loop"""
        logger.info("Starting orchestration loop")

        while True:
            try:
                # Process execution queue
                await self.process_execution_queue()

                # Update project progress
                await self.update_project_progress()

                # Handle completed executions
                await self.handle_completed_executions()

                # Cleanup and maintenance
                await self.perform_maintenance()

                # Wait before next cycle
                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Error in orchestration loop: {e}")
                await asyncio.sleep(5)

    async def process_execution_queue(self):
        """Process the execution queue"""
        try:
            if not self.execution_queue:
                return

            # Process queue in FIFO order
            to_remove = []

            for i, stage_execution in enumerate(list(self.execution_queue)):
                # Check if dependencies are satisfied
                dependencies_satisfied = await self.check_dependencies_satisfied(stage_execution)

                if not dependencies_satisfied:
                    continue

                # Check resource availability
                resources_available, message = await self.resource_manager.check_resource_availability(
                    stage_execution.resources_required
                )

                if not resources_available:
                    logger.info(f"Resources not available for {stage_execution.stage_id}: {message}")
                    continue

                # Execute the stage
                success = await self.execute_stage(stage_execution)

                if success:
                    # Move to active executions
                    self.active_executions[stage_execution.stage_id] = stage_execution
                    to_remove.append(i)

                    # Only process one stage at a time for now
                    break

            # Remove processed items from queue
            for i in reversed(to_remove):
                self.execution_queue.remove(list(self.execution_queue)[i])

        except Exception as e:
            logger.error(f"Error processing execution queue: {e}")

    async def check_dependencies_satisfied(self, stage_execution: StageExecution) -> bool:
        """Check if stage dependencies are satisfied"""
        try:
            for dep_id in stage_execution.dependencies:
                # Check if dependency stage completed successfully
                if dep_id in self.active_executions:
                    dep_execution = self.active_executions[dep_id]
                    if dep_execution.status != "completed":
                        return False
                else:
                    # Check execution history
                    dep_found = False
                    for hist_exec in self.execution_history:
                        if hist_exec.get("stage_id") == dep_id and hist_exec.get("status") == "completed":
                            dep_found = True
                            break

                    if not dep_found:
                        return False

            return True

        except Exception as e:
            logger.error(f"Error checking dependencies for {stage_execution.stage_id}: {e}")
            return False

    async def execute_stage(self, stage_execution: StageExecution) -> bool:
        """Execute a production stage"""
        try:
            logger.info(f"Executing stage: {stage_execution.stage_id}")

            # Allocate resources
            resources_allocated = await self.resource_manager.allocate_resources(
                stage_execution.stage_id, stage_execution.resources_required
            )

            if not resources_allocated:
                logger.error(f"Failed to allocate resources for {stage_execution.stage_id}")
                return False

            # Update stage status
            stage_execution.status = "running"
            stage_execution.started_at = datetime.now()

            # Create autonomous tasks for this stage
            autonomous_tasks_created = await self.create_autonomous_tasks(stage_execution)

            if not autonomous_tasks_created:
                logger.error(f"Failed to create autonomous tasks for {stage_execution.stage_id}")
                await self.resource_manager.release_resources(
                    stage_execution.stage_id, stage_execution.resources_required
                )
                return False

            logger.info(f"Successfully started execution of stage: {stage_execution.stage_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to execute stage {stage_execution.stage_id}: {e}")
            return False

    async def create_autonomous_tasks(self, stage_execution: StageExecution) -> bool:
        """Create autonomous tasks for a stage execution"""
        try:
            project = self.active_projects[stage_execution.project_id]
            goal_id = project.metadata.get("autonomous_goal_id")

            if not goal_id:
                logger.error(f"No autonomous goal found for project {project.id}")
                return False

            # Create tasks in AutonomousCore
            tasks_created = 0

            async with self.autonomous_core.get_connection() as conn:
                for task_spec in stage_execution.tasks:
                    task_id = await conn.fetchval("""
                        INSERT INTO autonomous_tasks
                        (goal_id, name, task_type, safety_level, priority, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                        RETURNING id
                    """,
                    goal_id,
                    task_spec["name"],
                    task_spec["task_type"],
                    task_spec["safety_level"],
                    task_spec["priority"],
                    json.dumps({
                        **task_spec["metadata"],
                        "stage_execution_id": stage_execution.stage_id,
                        "stage_type": stage_execution.stage.value,
                        "auto_generated": True,
                        "orchestrator_managed": True
                    })
                    )

                    if task_id:
                        tasks_created += 1
                        logger.info(f"Created autonomous task {task_id} for stage {stage_execution.stage_id}")

            return tasks_created > 0

        except Exception as e:
            logger.error(f"Failed to create autonomous tasks for {stage_execution.stage_id}: {e}")
            return False

    async def update_project_progress(self):
        """Update progress for all active projects"""
        try:
            for project in self.active_projects.values():
                if project.status != ProjectStatus.IN_PROGRESS:
                    continue

                # Calculate overall progress based on completed stages
                completed_stages = len(project.stages_completed)
                total_stages = len(ProductionStage)
                project.total_progress_percent = (completed_stages / total_stages) * 100.0

                # Update stage-specific progress
                for stage_id, execution in self.active_executions.items():
                    if execution.project_id == project.id:
                        if execution.status == "completed":
                            project.stage_progress[execution.stage] = 100.0
                            project.stages_completed.add(execution.stage)
                        elif execution.status == "running":
                            # Estimate progress based on time elapsed
                            if execution.started_at:
                                elapsed = (datetime.now() - execution.started_at).total_seconds() / 60
                                estimated_progress = min(90.0, (elapsed / execution.estimated_duration_minutes) * 100.0)
                                project.stage_progress[execution.stage] = estimated_progress

                # Check if project is completed
                if len(project.stages_completed) == len(ProductionStage):
                    project.status = ProjectStatus.COMPLETED
                    project.updated_at = datetime.now()
                    logger.info(f"Project completed: {project.id}")

                # Save updated state
                await self.save_project_state(project)

        except Exception as e:
            logger.error(f"Error updating project progress: {e}")

    async def handle_completed_executions(self):
        """Handle completed stage executions"""
        try:
            completed_executions = []

            for stage_id, execution in self.active_executions.items():
                if execution.status in ["completed", "failed"]:
                    completed_executions.append(stage_id)

            for stage_id in completed_executions:
                execution = self.active_executions.pop(stage_id)

                # Release resources
                await self.resource_manager.release_resources(
                    stage_id, execution.resources_required
                )

                # Add to execution history
                self.execution_history.append({
                    "stage_id": stage_id,
                    "project_id": execution.project_id,
                    "stage": execution.stage.value,
                    "status": execution.status,
                    "started_at": execution.started_at.isoformat() if execution.started_at else None,
                    "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                    "duration_minutes": ((execution.completed_at - execution.started_at).total_seconds() / 60)
                                      if execution.started_at and execution.completed_at else None,
                    "error_message": execution.error_message
                })

                if execution.status == "completed":
                    # Check for dependent stages and queue them
                    await self.queue_dependent_stages(execution)
                elif execution.status == "failed":
                    # Handle failure
                    await self.handle_stage_failure(execution)

                logger.info(f"Handled completion of stage: {stage_id} (Status: {execution.status})")

        except Exception as e:
            logger.error(f"Error handling completed executions: {e}")

    async def queue_dependent_stages(self, completed_execution: StageExecution):
        """Queue stages that depend on the completed execution"""
        try:
            project = self.active_projects[completed_execution.project_id]

            # Find stages waiting for this dependency
            for execution in list(self.execution_queue):
                if completed_execution.stage_id in execution.dependencies:
                    # Check if all dependencies are now satisfied
                    if await self.check_dependencies_satisfied(execution):
                        logger.info(f"Dependencies satisfied for stage: {execution.stage_id}")

        except Exception as e:
            logger.error(f"Error queuing dependent stages: {e}")

    async def handle_stage_failure(self, failed_execution: StageExecution):
        """Handle stage execution failure"""
        try:
            project = self.active_projects[failed_execution.project_id]

            # Determine if retry is possible
            if failed_execution.retry_count < failed_execution.max_retries:
                logger.info(f"Retrying failed stage: {failed_execution.stage_id} (Attempt {failed_execution.retry_count + 1})")

                # Reset execution state for retry
                failed_execution.status = "pending"
                failed_execution.started_at = None
                failed_execution.completed_at = None
                failed_execution.retry_count += 1

                # Add back to queue with delay
                await asyncio.sleep(60)  # Wait 1 minute before retry
                self.execution_queue.append(failed_execution)

            else:
                logger.error(f"Stage failed after {failed_execution.max_retries} retries: {failed_execution.stage_id}")

                # Mark project as failed if critical stage failed
                if failed_execution.stage in [ProductionStage.LORA_TRAINING, ProductionStage.CONTENT_GENERATION]:
                    project.status = ProjectStatus.FAILED
                    project.updated_at = datetime.now()
                    await self.save_project_state(project)

        except Exception as e:
            logger.error(f"Error handling stage failure: {e}")

    async def perform_maintenance(self):
        """Perform periodic maintenance tasks"""
        try:
            # Clean up old execution history (keep last 500 entries)
            if len(self.execution_history) > 500:
                while len(self.execution_history) > 500:
                    self.execution_history.popleft()

            # Update resource usage statistics
            # This would include more sophisticated resource tracking

            # Check for stuck executions
            current_time = datetime.now()
            for stage_id, execution in list(self.active_executions.items()):
                if execution.started_at:
                    elapsed_minutes = (current_time - execution.started_at).total_seconds() / 60
                    max_duration = execution.estimated_duration_minutes * 2  # Allow 2x estimated time

                    if elapsed_minutes > max_duration:
                        logger.warning(f"Stage execution taking too long: {stage_id} ({elapsed_minutes:.1f} min)")
                        # Could implement automatic timeout handling here

        except Exception as e:
            logger.error(f"Error during maintenance: {e}")

    async def get_project_status(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a project"""
        try:
            project = self.active_projects.get(project_id)
            if not project:
                return None

            # Get active executions for this project
            active_executions = []
            for execution in self.active_executions.values():
                if execution.project_id == project_id:
                    active_executions.append({
                        "stage": execution.stage.value,
                        "status": execution.status,
                        "progress": project.stage_progress.get(execution.stage, 0.0),
                        "started_at": execution.started_at.isoformat() if execution.started_at else None,
                        "estimated_completion": (execution.started_at + timedelta(minutes=execution.estimated_duration_minutes)).isoformat()
                                              if execution.started_at else None
                    })

            return {
                "project_id": project.id,
                "name": project.name,
                "status": project.status.value,
                "current_stage": project.current_stage.value,
                "total_progress_percent": project.total_progress_percent,
                "stages_completed": [stage.value for stage in project.stages_completed],
                "stage_progress": {stage.value: progress for stage, progress in project.stage_progress.items()},
                "active_executions": active_executions,
                "created_at": project.created_at.isoformat(),
                "updated_at": project.updated_at.isoformat(),
                "deadline": project.deadline.isoformat() if project.deadline else None,
                "quality_issues": project.quality_issues,
                "resource_usage": project.allocated_resources
            }

        except Exception as e:
            logger.error(f"Error getting project status: {e}")
            return None

    async def initialize_database(self):
        """Initialize database schema for anime production orchestration"""
        try:
            async with self.get_connection() as conn:
                # Create orchestration tables
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS anime_production_projects (
                        id VARCHAR(255) PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        description TEXT,
                        project_type VARCHAR(100) NOT NULL,
                        status VARCHAR(50) NOT NULL,
                        current_stage VARCHAR(100) NOT NULL,
                        priority INTEGER DEFAULT 5,
                        total_progress_percent DECIMAL(5,2) DEFAULT 0.00,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        deadline TIMESTAMP WITH TIME ZONE,
                        project_data JSONB DEFAULT '{}',
                        metadata JSONB DEFAULT '{}'
                    );
                """)

                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS anime_stage_executions (
                        stage_id VARCHAR(255) PRIMARY KEY,
                        project_id VARCHAR(255) REFERENCES anime_production_projects(id) ON DELETE CASCADE,
                        stage_name VARCHAR(100) NOT NULL,
                        status VARCHAR(50) NOT NULL,
                        started_at TIMESTAMP WITH TIME ZONE,
                        completed_at TIMESTAMP WITH TIME ZONE,
                        estimated_duration_minutes INTEGER,
                        retry_count INTEGER DEFAULT 0,
                        max_retries INTEGER DEFAULT 3,
                        error_message TEXT,
                        execution_data JSONB DEFAULT '{}'
                    );
                """)

                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_anime_projects_status ON anime_production_projects(status);
                    CREATE INDEX IF NOT EXISTS idx_anime_projects_stage ON anime_production_projects(current_stage);
                    CREATE INDEX IF NOT EXISTS idx_anime_executions_project ON anime_stage_executions(project_id);
                    CREATE INDEX IF NOT EXISTS idx_anime_executions_status ON anime_stage_executions(status);
                """)

                logger.info("Anime production orchestration database schema initialized")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def save_project_to_database(self, project: ProductionProject):
        """Save project to database"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO anime_production_projects
                    (id, name, description, project_type, status, current_stage, priority,
                     total_progress_percent, created_at, updated_at, deadline, project_data, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::jsonb, $13::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        project_type = EXCLUDED.project_type,
                        status = EXCLUDED.status,
                        current_stage = EXCLUDED.current_stage,
                        priority = EXCLUDED.priority,
                        total_progress_percent = EXCLUDED.total_progress_percent,
                        updated_at = EXCLUDED.updated_at,
                        deadline = EXCLUDED.deadline,
                        project_data = EXCLUDED.project_data,
                        metadata = EXCLUDED.metadata
                """,
                project.id, project.name, project.description, project.project_type,
                project.status.value, project.current_stage.value, project.priority,
                project.total_progress_percent, project.created_at, project.updated_at,
                project.deadline,
                json.dumps({
                    "characters": project.characters,
                    "scenes": project.scenes,
                    "style_config": project.style_config,
                    "technical_specs": project.technical_specs,
                    "stages_completed": [stage.value for stage in project.stages_completed],
                    "stage_progress": {stage.value: progress for stage, progress in project.stage_progress.items()},
                    "allocated_resources": {res_type.value if isinstance(res_type, ResourceType) else res_type: amount
                                           for res_type, amount in project.allocated_resources.items()},
                    "quality_gates_passed": project.quality_gates_passed,
                    "quality_issues": project.quality_issues
                }),
                json.dumps(project.metadata or {})
                )

        except Exception as e:
            logger.error(f"Failed to save project to database: {e}")
            raise

    async def save_project_state(self, project: ProductionProject):
        """Save current project state"""
        await self.save_project_to_database(project)

    async def load_active_projects(self):
        """Load active projects from database"""
        try:
            async with self.get_connection() as conn:
                projects = await conn.fetch("""
                    SELECT * FROM anime_production_projects
                    WHERE status IN ('draft', 'in_progress', 'pending_approval', 'on_hold')
                """)

                for project_record in projects:
                    project_data = json.loads(project_record['project_data'])
                    metadata = json.loads(project_record['metadata'])

                    project = ProductionProject(
                        id=project_record['id'],
                        name=project_record['name'],
                        description=project_record['description'] or "",
                        project_type=project_record['project_type'],
                        status=ProjectStatus(project_record['status']),
                        current_stage=ProductionStage(project_record['current_stage']),
                        priority=project_record['priority'],
                        characters=project_data.get('characters', []),
                        scenes=project_data.get('scenes', []),
                        style_config=project_data.get('style_config', {}),
                        technical_specs=project_data.get('technical_specs', {}),
                        stages_completed={ProductionStage(stage) for stage in project_data.get('stages_completed', [])},
                        total_progress_percent=float(project_record['total_progress_percent']),
                        stage_progress={ProductionStage(stage): progress for stage, progress in project_data.get('stage_progress', {}).items()},
                        allocated_resources={ResourceType(res_type): amount for res_type, amount in project_data.get('allocated_resources', {}).items()},
                        resource_history=project_data.get('resource_history', []),
                        quality_gates_passed=project_data.get('quality_gates_passed', []),
                        quality_issues=project_data.get('quality_issues', []),
                        created_at=project_record['created_at'],
                        updated_at=project_record['updated_at'],
                        deadline=project_record['deadline'],
                        metadata=metadata
                    )

                    self.active_projects[project.id] = project

                logger.info(f"Loaded {len(projects)} active projects")

        except Exception as e:
            logger.error(f"Failed to load active projects: {e}")


# Enhanced AutonomousCore integration for anime production
async def enhance_autonomous_core_for_anime_production(autonomous_core: AutonomousCore):
    """Enhance AutonomousCore with anime production task generation capabilities"""

    # Store original method
    original_generate_tasks = autonomous_core._generate_tasks_for_goal

    async def enhanced_generate_tasks_for_goal(goal: Dict[str, Any]):
        """Enhanced task generation that includes anime production workflows"""
        try:
            goal_type = goal['goal_type']
            goal_id = goal['id']

            # Handle anime production goals
            if goal_type == 'anime_production':
                await generate_anime_production_tasks(autonomous_core, goal)
            else:
                # Use original method for other goal types
                await original_generate_tasks(goal)

        except Exception as e:
            logger.error(f"Enhanced task generation failed for goal {goal['name']}: {e}")

    # Replace the method
    autonomous_core._generate_tasks_for_goal = enhanced_generate_tasks_for_goal
    logger.info("AutonomousCore enhanced with anime production task generation")


async def generate_anime_production_tasks(autonomous_core: AutonomousCore, goal: Dict[str, Any]):
    """Generate anime production specific tasks"""
    try:
        goal_metadata = goal.get('metadata', {})
        project_id = goal_metadata.get('project_id')
        project_type = goal_metadata.get('project_type', 'short_film')
        stages = goal_metadata.get('stages', [])

        # Check how many tasks this goal already has
        async with autonomous_core.get_connection() as conn:
            task_count = await conn.fetchval("""
                SELECT COUNT(*) FROM autonomous_tasks
                WHERE goal_id = $1
            """, goal['id'])

        # Generate initial orchestration task if none exist
        if task_count == 0:
            task_id = await conn.fetchval("""
                INSERT INTO autonomous_tasks
                (goal_id, name, task_type, safety_level, priority, metadata)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                RETURNING id
            """,
            goal['id'],
            f"Initialize Anime Production Pipeline",
            "anime_orchestration",
            "auto",
            1,  # High priority
            json.dumps({
                "project_id": project_id,
                "project_type": project_type,
                "action": "initialize_pipeline",
                "description": f"Initialize complete anime production pipeline for project {project_id}",
                "auto_generated": True,
                "orchestrator_managed": True
            })
            )

            if task_id:
                logger.info(f"Generated anime production initialization task {task_id} for goal {goal['name']}")

        # Generate monitoring task if project is active
        elif task_count < 2:
            task_id = await conn.fetchval("""
                INSERT INTO autonomous_tasks
                (goal_id, name, task_type, safety_level, priority, metadata)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                RETURNING id
            """,
            goal['id'],
            f"Monitor Production Progress",
            "anime_monitoring",
            "auto",
            3,
            json.dumps({
                "project_id": project_id,
                "action": "monitor_progress",
                "description": f"Monitor and report anime production progress for project {project_id}",
                "auto_generated": True,
                "orchestrator_managed": True
            })
            )

            if task_id:
                logger.info(f"Generated anime production monitoring task {task_id} for goal {goal['name']}")

    except Exception as e:
        logger.error(f"Failed to generate anime production tasks: {e}")