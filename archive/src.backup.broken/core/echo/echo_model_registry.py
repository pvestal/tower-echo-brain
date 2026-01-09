#!/usr/bin/env python3
"""
Echo Model Registry - Version Control and Lifecycle Management
============================================================

Comprehensive model registry for AI Assist that handles:
- Model versioning and metadata tracking
- Lifecycle management (dev/staging/prod)
- Rollback capabilities
- Performance metrics tracking
- Model comparison and selection
- Automated deployment workflows

Architecture:
- SQLite backend for model metadata
- File system storage for model artifacts
- REST API for model operations
- Integration with existing Echo learning system
"""

import asyncio
import hashlib
import json
import logging
import pickle
import shutil
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import aiofiles
import numpy as np
import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

# Configuration
MODEL_REGISTRY_DB = "/opt/tower-echo-brain/data/model_registry.db"
MODEL_STORAGE_PATH = "/opt/tower-echo-brain/models/"
BACKUP_PATH = "/opt/tower-echo-brain/models/backups/"


class ModelStage(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class ModelType(Enum):
    DECISION_ENGINE = "decision_engine"
    LEARNING_MODEL = "learning_model"
    DIRECTOR_CONFIG = "director_config"
    CLASSIFIER = "classifier"
    REGRESSION = "regression"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"


@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    custom_metrics: Dict[str, float] = None


@dataclass
class ModelMetadata:
    model_id: str
    name: str
    version: str
    model_type: ModelType
    stage: ModelStage
    created_at: datetime
    created_by: str
    description: str
    tags: List[str]
    metrics: ModelMetrics
    hyperparameters: Dict[str, Any]
    training_data_hash: str
    model_hash: str
    parent_model_id: Optional[str] = None
    dependencies: List[str] = None
    deployment_config: Dict[str, Any] = None


class ModelRegistryAPI:
    def __init__(self):
        self.db_path = MODEL_REGISTRY_DB
        self.storage_path = Path(MODEL_STORAGE_PATH)
        self.backup_path = Path(BACKUP_PATH)
        self.logger = logging.getLogger(__name__)
        self._ensure_directories()
        self._init_database()

    def _ensure_directories(self):
        """Create necessary directories"""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.backup_path.mkdir(parents=True, exist_ok=True)

    def _init_database(self):
        """Initialize the model registry database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    created_by TEXT NOT NULL,
                    description TEXT,
                    tags TEXT,
                    metrics TEXT,
                    hyperparameters TEXT,
                    training_data_hash TEXT,
                    model_hash TEXT,
                    parent_model_id TEXT,
                    dependencies TEXT,
                    deployment_config TEXT,
                    is_active BOOLEAN DEFAULT 1
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_deployments (
                    deployment_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    deployed_at TIMESTAMP NOT NULL,
                    deployed_by TEXT NOT NULL,
                    status TEXT NOT NULL,
                    endpoint_url TEXT,
                    config TEXT,
                    FOREIGN KEY (model_id) REFERENCES models (model_id)
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_experiments (
                    experiment_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    experiment_name TEXT NOT NULL,
                    started_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    parameters TEXT,
                    results TEXT,
                    status TEXT,
                    FOREIGN KEY (model_id) REFERENCES models (model_id)
                )
            """
            )

            # Indexes for performance
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_models_stage ON models(stage)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_models_type ON models(model_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_models_created ON models(created_at)"
            )

    def _generate_model_id(self, name: str, version: str) -> str:
        """Generate unique model ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{name}_{version}_{timestamp}"

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    async def register_model(
        self,
        name: str,
        version: str,
        model_type: ModelType,
        model_artifact: Union[str, bytes, Any],
        description: str = "",
        tags: List[str] = None,
        metrics: ModelMetrics = None,
        hyperparameters: Dict[str, Any] = None,
        training_data_hash: str = "",
        parent_model_id: str = None,
        dependencies: List[str] = None,
        created_by: str = "system",
    ) -> str:
        """Register a new model in the registry"""

        model_id = self._generate_model_id(name, version)
        tags = tags or []
        hyperparameters = hyperparameters or {}
        dependencies = dependencies or []

        # Save model artifact
        model_path = self.storage_path / f"{model_id}.pkl"

        if isinstance(model_artifact, (str, bytes)):
            async with aiofiles.open(model_path, "wb") as f:
                await f.write(
                    model_artifact
                    if isinstance(model_artifact, bytes)
                    else model_artifact.encode()
                )
        else:
            # Serialize Python object
            with open(model_path, "wb") as f:
                pickle.dump(model_artifact, f)

        # Calculate model hash
        model_hash = self._calculate_file_hash(model_path)

        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version=version,
            model_type=model_type,
            stage=ModelStage.DEVELOPMENT,
            created_at=datetime.now(),
            created_by=created_by,
            description=description,
            tags=tags,
            metrics=metrics,
            hyperparameters=hyperparameters,
            training_data_hash=training_data_hash,
            model_hash=model_hash,
            parent_model_id=parent_model_id,
            dependencies=dependencies,
        )

        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO models (
                    model_id, name, version, model_type, stage, created_at,
                    created_by, description, tags, metrics, hyperparameters,
                    training_data_hash, model_hash, parent_model_id, dependencies
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    model_id,
                    name,
                    version,
                    model_type.value,
                    metadata.stage.value,
                    metadata.created_at,
                    created_by,
                    description,
                    json.dumps(tags),
                    json.dumps(asdict(metrics)) if metrics else None,
                    json.dumps(hyperparameters),
                    training_data_hash,
                    model_hash,
                    parent_model_id,
                    json.dumps(dependencies),
                ),
            )

        self.logger.info(f"Model registered: {model_id}")
        return model_id

    async def promote_model(
        self, model_id: str, target_stage: ModelStage, promoted_by: str = "system"
    ) -> bool:
        """Promote model to a different stage"""

        with sqlite3.connect(self.db_path) as conn:
            # Check if model exists
            cursor = conn.execute(
                "SELECT stage FROM models WHERE model_id = ? AND is_active = 1",
                (model_id,),
            )
            result = cursor.fetchone()

            if not result:
                raise ValueError(f"Model {model_id} not found")

            current_stage = result[0]

            # Validate promotion path
            valid_promotions = {
                ModelStage.DEVELOPMENT.value: [
                    ModelStage.STAGING.value,
                    ModelStage.ARCHIVED.value,
                ],
                ModelStage.STAGING.value: [
                    ModelStage.PRODUCTION.value,
                    ModelStage.DEVELOPMENT.value,
                    ModelStage.ARCHIVED.value,
                ],
                ModelStage.PRODUCTION.value: [
                    ModelStage.DEPRECATED.value,
                    ModelStage.ARCHIVED.value,
                ],
            }

            if target_stage.value not in valid_promotions.get(current_stage, []):
                raise ValueError(
                    f"Invalid promotion from {current_stage} to {target_stage.value}"
                )

            # If promoting to production, demote current production model
            if target_stage == ModelStage.PRODUCTION:
                cursor = conn.execute(
                    """
                    SELECT model_id FROM models 
                    WHERE stage = ? AND model_type = (
                        SELECT model_type FROM models WHERE model_id = ?
                    ) AND is_active = 1
                """,
                    (ModelStage.PRODUCTION.value, model_id),
                )

                current_prod = cursor.fetchone()
                if current_prod:
                    conn.execute(
                        """
                        UPDATE models SET stage = ? WHERE model_id = ?
                    """,
                        (ModelStage.DEPRECATED.value, current_prod[0]),
                    )

            # Update model stage
            conn.execute(
                """
                UPDATE models SET stage = ? WHERE model_id = ?
            """,
                (target_stage.value, model_id),
            )

        self.logger.info(f"Model {model_id} promoted to {target_stage.value}")
        return True

    async def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata"""

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM models WHERE model_id = ? AND is_active = 1
            """,
                (model_id,),
            )

            row = cursor.fetchone()
            if not row:
                return None

            # Parse JSON fields
            tags = json.loads(row["tags"]) if row["tags"] else []
            metrics_data = json.loads(
                row["metrics"]) if row["metrics"] else None
            metrics = ModelMetrics(**metrics_data) if metrics_data else None
            hyperparameters = (
                json.loads(row["hyperparameters"]
                           ) if row["hyperparameters"] else {}
            )
            dependencies = (
                json.loads(row["dependencies"]) if row["dependencies"] else []
            )

            return ModelMetadata(
                model_id=row["model_id"],
                name=row["name"],
                version=row["version"],
                model_type=ModelType(row["model_type"]),
                stage=ModelStage(row["stage"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                created_by=row["created_by"],
                description=row["description"] or "",
                tags=tags,
                metrics=metrics,
                hyperparameters=hyperparameters,
                training_data_hash=row["training_data_hash"] or "",
                model_hash=row["model_hash"] or "",
                parent_model_id=row["parent_model_id"],
                dependencies=dependencies,
            )

    async def load_model(self, model_id: str) -> Any:
        """Load model artifact from storage"""

        model_path = self.storage_path / f"{model_id}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_id}")

        with open(model_path, "rb") as f:
            return pickle.load(f)

    async def list_models(
        self,
        model_type: Optional[ModelType] = None,
        stage: Optional[ModelStage] = None,
        limit: int = 100,
    ) -> List[ModelMetadata]:
        """List models with optional filters"""

        query = "SELECT * FROM models WHERE is_active = 1"
        params = []

        if model_type:
            query += " AND model_type = ?"
            params.append(model_type.value)

        if stage:
            query += " AND stage = ?"
            params.append(stage.value)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        models = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)

            for row in cursor.fetchall():
                metadata = await self.get_model_metadata(row["model_id"])
                if metadata:
                    models.append(metadata)

        return models

    async def get_production_model(self, model_type: ModelType) -> Optional[str]:
        """Get current production model ID for a type"""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT model_id FROM models 
                WHERE model_type = ? AND stage = ? AND is_active = 1
                ORDER BY created_at DESC LIMIT 1
            """,
                (model_type.value, ModelStage.PRODUCTION.value),
            )

            result = cursor.fetchone()
            return result[0] if result else None

    async def rollback_model(self, model_type: ModelType, target_model_id: str) -> bool:
        """Rollback to a previous model version"""

        # Verify target model exists and is valid for rollback
        metadata = await self.get_model_metadata(target_model_id)
        if not metadata or metadata.model_type != model_type:
            raise ValueError(f"Invalid rollback target: {target_model_id}")

        # Create backup of current production model
        current_prod = await self.get_production_model(model_type)
        if current_prod:
            await self.promote_model(current_prod, ModelStage.DEPRECATED)

        # Promote target model to production
        await self.promote_model(target_model_id, ModelStage.PRODUCTION)

        self.logger.info(
            f"Rolled back {model_type.value} to {target_model_id}")
        return True

    async def archive_model(self, model_id: str) -> bool:
        """Archive a model (soft delete)"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE models SET is_active = 0 WHERE model_id = ?
            """,
                (model_id,),
            )

        # Move model file to backup
        model_path = self.storage_path / f"{model_id}.pkl"
        if model_path.exists():
            backup_path = self.backup_path / f"{model_id}.pkl"
            shutil.move(str(model_path), str(backup_path))

        self.logger.info(f"Model {model_id} archived")
        return True

    async def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple models"""

        comparison = {"models": {}, "summary": {}}

        for model_id in model_ids:
            metadata = await self.get_model_metadata(model_id)
            if metadata:
                comparison["models"][model_id] = {
                    "name": metadata.name,
                    "version": metadata.version,
                    "stage": metadata.stage.value,
                    "metrics": asdict(metadata.metrics) if metadata.metrics else {},
                    "created_at": metadata.created_at.isoformat(),
                }

        # Generate comparison summary
        if len(comparison["models"]) > 1:
            metrics_keys = set()
            for model_data in comparison["models"].values():
                metrics_keys.update(model_data["metrics"].keys())

            comparison["summary"]["best_by_metric"] = {}
            for metric in metrics_keys:
                best_model = max(
                    comparison["models"].items(),
                    key=lambda x: x[1]["metrics"].get(metric, 0),
                    default=(None, None),
                )
                if best_model[0]:
                    comparison["summary"]["best_by_metric"][metric] = best_model[0]

        return comparison


# FastAPI Integration
app = FastAPI(title="Echo Model Registry", version="1.0.0")
registry = ModelRegistryAPI()


class ModelRegistrationRequest(BaseModel):
    name: str
    version: str
    model_type: str
    description: str = ""
    tags: List[str] = []
    hyperparameters: Dict[str, Any] = {}
    training_data_hash: str = ""
    parent_model_id: Optional[str] = None
    dependencies: List[str] = []


@app.post("/models/register")
async def register_model_endpoint(request: ModelRegistrationRequest):
    """Register a new model"""
    try:
        model_type = ModelType(request.model_type)
        model_id = await registry.register_model(
            name=request.name,
            version=request.version,
            model_type=model_type,
            model_artifact=b"placeholder",  # In real implementation, accept file upload
            description=request.description,
            tags=request.tags,
            hyperparameters=request.hyperparameters,
            training_data_hash=request.training_data_hash,
            parent_model_id=request.parent_model_id,
            dependencies=request.dependencies,
        )
        return {"model_id": model_id, "status": "registered"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/models/{model_id}/promote")
async def promote_model_endpoint(model_id: str, target_stage: str):
    """Promote model to different stage"""
    try:
        stage = ModelStage(target_stage)
        success = await registry.promote_model(model_id, stage)
        return {"model_id": model_id, "new_stage": target_stage, "success": success}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/models/{model_id}")
async def get_model_endpoint(model_id: str):
    """Get model metadata"""
    metadata = await registry.get_model_metadata(model_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Model not found")

    return {
        "model_id": metadata.model_id,
        "name": metadata.name,
        "version": metadata.version,
        "model_type": metadata.model_type.value,
        "stage": metadata.stage.value,
        "created_at": metadata.created_at.isoformat(),
        "description": metadata.description,
        "tags": metadata.tags,
        "metrics": asdict(metadata.metrics) if metadata.metrics else None,
    }


@app.get("/models")
async def list_models_endpoint(
    model_type: Optional[str] = None, stage: Optional[str] = None, limit: int = 100
):
    """List models with filters"""
    try:
        model_type_enum = ModelType(model_type) if model_type else None
        stage_enum = ModelStage(stage) if stage else None

        models = await registry.list_models(model_type_enum, stage_enum, limit)

        return {
            "models": [
                {
                    "model_id": m.model_id,
                    "name": m.name,
                    "version": m.version,
                    "model_type": m.model_type.value,
                    "stage": m.stage.value,
                    "created_at": m.created_at.isoformat(),
                }
                for m in models
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/models/rollback/{model_type}")
async def rollback_model_endpoint(model_type: str, target_model_id: str):
    """Rollback to previous model"""
    try:
        model_type_enum = ModelType(model_type)
        success = await registry.rollback_model(model_type_enum, target_model_id)
        return {
            "model_type": model_type,
            "rollback_to": target_model_id,
            "success": success,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/models/compare")
async def compare_models_endpoint(model_ids: List[str]):
    """Compare multiple models"""
    try:
        comparison = await registry.compare_models(model_ids)
        return comparison
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=8340)
