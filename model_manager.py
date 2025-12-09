#!/usr/bin/env python3
"""
Ollama Model Management System for Echo Brain
Provides full CRUD operations for Ollama models with Board of Directors oversight
"""

import asyncio
import subprocess
import logging
import json
import psycopg2
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from fastapi import HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
import aiohttp
import os

from routing.service_registry import ServiceRegistry
from routing.request_logger import RequestLogger, TaskDecision, DecisionStatus
from routing.auth_middleware import get_current_user, require_permission

logger = logging.getLogger(__name__)

# Database connection
DB_CONFIG = {
    "host": "localhost",
    "database": "echo_brain",
    "user": os.getenv("TOWER_USER", os.getenv("TOWER_USER", "patrick"))
}

class ModelOperation(str, Enum):
    PULL = "pull"
    UPDATE = "update"
    REMOVE = "remove"
    LIST = "list"
    SHOW = "show"

class ModelStatus(str, Enum):
    AVAILABLE = "available"
    DOWNLOADING = "downloading"
    INSTALLED = "installed"
    FAILED = "failed"
    REMOVED = "removed"
    UPDATING = "updating"

# Pydantic Models
class ModelManagementRequest(BaseModel):
    operation: ModelOperation
    model_name: str
    tag: Optional[str] = "latest"
    force: bool = False
    reason: str = Field(..., description="Reason for the operation")
    user_id: str = Field(default="admin")

class ModelManagementResponse(BaseModel):
    request_id: str
    operation: ModelOperation
    model: str
    status: ModelStatus
    message: str
    board_decision_id: Optional[str] = None
    requires_approval: bool = False

class ModelInfo(BaseModel):
    name: str
    tag: str
    size: str
    modified: datetime
    status: ModelStatus
    parameters: Optional[int] = None
    specialization: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

class OllamaModelManager:
    """
    Manages Ollama models with Board of Directors oversight and database tracking
    """

    def __init__(self, board_registry: ServiceRegistry, request_logger: RequestLogger):
        self.board = board_registry
        self.tracker = request_logger
        self.ollama_api = "http://localhost:11434"
        self.db_conn = None
        self._init_database()

        # Define critical models that require board approval to modify
        self.protected_models = {
            "llama3.1:70b",      # Genius level
            "qwen2.5-coder:32b", # Expert level
            "deepseek-coder-v2:16b", # Specialized coding
            "mixtral:8x7b",      # Creative specialist
            "codellama:70b"      # Analysis specialist
        }

    def _init_database(self):
        """Initialize database connection and create tables if needed"""
        try:
            self.db_conn = psycopg2.connect(**DB_CONFIG)
            cursor = self.db_conn.cursor()

            # Create model tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ollama_models (
                    model_id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    tag VARCHAR(50) DEFAULT 'latest',
                    size_bytes BIGINT,
                    parameters BIGINT,
                    specialization VARCHAR(100),
                    status VARCHAR(50) NOT NULL,
                    installed_at TIMESTAMP,
                    last_used TIMESTAMP,
                    usage_count INTEGER DEFAULT 0,
                    performance_score FLOAT DEFAULT 0.0,
                    board_approved BOOLEAN DEFAULT FALSE,
                    UNIQUE(name, tag)
                );
            """)

            # Create model operations log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_operations (
                    operation_id SERIAL PRIMARY KEY,
                    request_id VARCHAR(100) UNIQUE,
                    model_name VARCHAR(255),
                    operation VARCHAR(50),
                    status VARCHAR(50),
                    user_id VARCHAR(100),
                    reason TEXT,
                    board_decision_id VARCHAR(100),
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    output TEXT,
                    error TEXT
                );
            """)

            self.db_conn.commit()
            logger.info("Database initialized for model tracking")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    async def request_model_operation(
        self,
        request: ModelManagementRequest,
        background_tasks: BackgroundTasks
    ) -> ModelManagementResponse:
        """
        Process a model management request with board oversight
        """
        request_id = f"model_{request.operation.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Check if operation requires board approval
        requires_approval = await self._requires_board_approval(request)

        if requires_approval:
            # Submit to Board of Directors
            board_decision_id = await self._submit_to_board(request, request_id)

            # Start background monitoring
            background_tasks.add_task(
                self._monitor_board_decision,
                request_id,
                board_decision_id,
                request
            )

            return ModelManagementResponse(
                request_id=request_id,
                operation=request.operation,
                model=f"{request.model_name}:{request.tag}",
                status=ModelStatus.DOWNLOADING if request.operation == ModelOperation.PULL else ModelStatus.UPDATING,
                message="Request submitted to Board of Directors for approval",
                board_decision_id=board_decision_id,
                requires_approval=True
            )
        else:
            # Execute immediately for non-critical operations
            result = await self._execute_operation(request, request_id)
            return result

    async def _requires_board_approval(self, request: ModelManagementRequest) -> bool:
        """
        Determine if an operation requires board approval
        """
        model_full = f"{request.model_name}:{request.tag}"

        # Removing protected models requires approval
        if request.operation == ModelOperation.REMOVE and model_full in self.protected_models:
            return True

        # Pulling large models (>20GB) requires approval
        if request.operation == ModelOperation.PULL:
            model_size = await self._estimate_model_size(request.model_name)
            if model_size > 20 * 1024 * 1024 * 1024:  # 20GB
                return True

        # Admin operations don't require approval
        if request.user_id == "admin":
            return False

        return False

    async def _submit_to_board(self, request: ModelManagementRequest, request_id: str) -> str:
        """
        Submit operation to Board of Directors for approval
        """
        task = TaskDecision(
            task_id=request_id,
            description=f"Model {request.operation.value}: {request.model_name}:{request.tag}",
            task_type="model_management",
            priority="high" if request.operation == ModelOperation.REMOVE else "normal",
            user_id=request.user_id,
            created_at=datetime.now(),
            status=DecisionStatus.PENDING,
            context={
                "operation": request.operation.value,
                "model": request.model_name,
                "tag": request.tag,
                "reason": request.reason,
                "force": request.force
            }
        )

        # Register with decision tracker
        self.tracker.start_tracking(task)

        # Submit to board for evaluation
        evaluations = await self.board.route_task({
            "task_type": "model_management",
            "operation": request.operation.value,
            "model": request.model_name,
            "context": task.context
        })

        # Build consensus
        consensus = self.board.build_consensus(evaluations)

        # Update decision
        self.tracker.update_decision(
            request_id,
            consensus["recommendation"],
            consensus["confidence"],
            evaluations
        )

        return request_id

    async def _monitor_board_decision(
        self,
        request_id: str,
        board_decision_id: str,
        request: ModelManagementRequest
    ):
        """
        Monitor board decision and execute if approved
        """
        max_wait = 300  # 5 minutes
        check_interval = 5
        elapsed = 0

        while elapsed < max_wait:
            decision = self.tracker.get_decision_status(board_decision_id)

            if decision and decision.status == DecisionStatus.APPROVED:
                # Execute the approved operation
                await self._execute_operation(request, request_id)
                break
            elif decision and decision.status == DecisionStatus.REJECTED:
                # Log rejection
                self._log_operation(
                    request_id,
                    request,
                    "rejected",
                    error="Board of Directors rejected the operation"
                )
                break

            await asyncio.sleep(check_interval)
            elapsed += check_interval

    async def _execute_operation(
        self,
        request: ModelManagementRequest,
        request_id: str
    ) -> ModelManagementResponse:
        """
        Execute the actual Ollama operation
        """
        try:
            if request.operation == ModelOperation.PULL:
                result = await self._pull_model(request.model_name, request.tag)
            elif request.operation == ModelOperation.UPDATE:
                result = await self._update_model(request.model_name, request.tag)
            elif request.operation == ModelOperation.REMOVE:
                result = await self._remove_model(request.model_name, request.tag)
            elif request.operation == ModelOperation.LIST:
                result = await self._list_models()
            elif request.operation == ModelOperation.SHOW:
                result = await self._show_model(request.model_name, request.tag)
            else:
                raise ValueError(f"Unknown operation: {request.operation}")

            # Log successful operation
            self._log_operation(request_id, request, "completed", output=json.dumps(result))

            # Update database
            self._update_model_database(request.model_name, request.tag, request.operation)

            return ModelManagementResponse(
                request_id=request_id,
                operation=request.operation,
                model=f"{request.model_name}:{request.tag}",
                status=ModelStatus.INSTALLED if request.operation == ModelOperation.PULL else ModelStatus.AVAILABLE,
                message=f"Operation {request.operation.value} completed successfully",
                board_decision_id=None,
                requires_approval=False
            )

        except Exception as e:
            logger.error(f"Operation failed: {e}")
            self._log_operation(request_id, request, "failed", error=str(e))

            return ModelManagementResponse(
                request_id=request_id,
                operation=request.operation,
                model=f"{request.model_name}:{request.tag}",
                status=ModelStatus.FAILED,
                message=f"Operation failed: {str(e)}",
                board_decision_id=None,
                requires_approval=False
            )

    async def _pull_model(self, model_name: str, tag: str) -> Dict:
        """
        Pull a model from Ollama registry
        """
        model_full = f"{model_name}:{tag}"

        # Use subprocess for long-running pull operation
        process = await asyncio.create_subprocess_exec(
            "ollama", "pull", model_full,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise Exception(f"Failed to pull model: {stderr.decode()}")

        return {"model": model_full, "status": "pulled", "output": stdout.decode()}

    async def _update_model(self, model_name: str, tag: str) -> Dict:
        """
        Update an existing model
        """
        # First pull the latest version
        result = await self._pull_model(model_name, tag)
        result["status"] = "updated"
        return result

    async def _remove_model(self, model_name: str, tag: str) -> Dict:
        """
        Remove a model from the system
        """
        model_full = f"{model_name}:{tag}"

        process = await asyncio.create_subprocess_exec(
            "ollama", "rm", model_full,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise Exception(f"Failed to remove model: {stderr.decode()}")

        return {"model": model_full, "status": "removed", "output": stdout.decode()}

    async def _list_models(self) -> List[Dict]:
        """
        List all installed models
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.ollama_api}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("models", [])
                else:
                    raise Exception(f"Failed to list models: HTTP {response.status}")

    async def _show_model(self, model_name: str, tag: str) -> Dict:
        """
        Show details about a specific model
        """
        model_full = f"{model_name}:{tag}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.ollama_api}/api/show",
                json={"name": model_full}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Failed to show model: HTTP {response.status}")

    async def _estimate_model_size(self, model_name: str) -> int:
        """
        Estimate model size based on name patterns
        """
        size_map = {
            "70b": 40 * 1024 * 1024 * 1024,  # 40GB
            "32b": 20 * 1024 * 1024 * 1024,  # 20GB
            "13b": 8 * 1024 * 1024 * 1024,   # 8GB
            "7b": 4 * 1024 * 1024 * 1024,    # 4GB
            "3b": 2 * 1024 * 1024 * 1024,    # 2GB
            "1b": 650 * 1024 * 1024,          # 650MB
        }

        for pattern, size in size_map.items():
            if pattern in model_name.lower():
                return size

        return 5 * 1024 * 1024 * 1024  # Default 5GB

    def _log_operation(
        self,
        request_id: str,
        request: ModelManagementRequest,
        status: str,
        output: Optional[str] = None,
        error: Optional[str] = None
    ):
        """
        Log operation to database
        """
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT INTO model_operations
                (request_id, model_name, operation, status, user_id, reason, output, error, completed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (request_id) DO UPDATE
                SET status = EXCLUDED.status,
                    output = EXCLUDED.output,
                    error = EXCLUDED.error,
                    completed_at = EXCLUDED.completed_at
            """, (
                request_id,
                f"{request.model_name}:{request.tag}",
                request.operation.value,
                status,
                request.user_id,
                request.reason,
                output,
                error,
                datetime.now() if status in ["completed", "failed", "rejected"] else None
            ))
            self.db_conn.commit()
        except Exception as e:
            logger.error(f"Failed to log operation: {e}")

    def _update_model_database(
        self,
        model_name: str,
        tag: str,
        operation: ModelOperation
    ):
        """
        Update model tracking in database
        """
        try:
            cursor = self.db_conn.cursor()

            if operation == ModelOperation.PULL:
                cursor.execute("""
                    INSERT INTO ollama_models (name, tag, status, installed_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (name, tag) DO UPDATE
                    SET status = EXCLUDED.status,
                        installed_at = EXCLUDED.installed_at
                """, (model_name, tag, ModelStatus.INSTALLED.value, datetime.now()))

            elif operation == ModelOperation.REMOVE:
                cursor.execute("""
                    UPDATE ollama_models
                    SET status = %s
                    WHERE name = %s AND tag = %s
                """, (ModelStatus.REMOVED.value, model_name, tag))

            self.db_conn.commit()
        except Exception as e:
            logger.error(f"Failed to update model database: {e}")

    async def get_installed_models(self) -> List[ModelInfo]:
        """
        Get list of currently installed models with metadata
        """
        models = await self._list_models()
        result = []

        for model in models:
            # Parse model information
            name_parts = model["name"].split(":")
            model_name = name_parts[0]
            tag = name_parts[1] if len(name_parts) > 1 else "latest"

            # Determine specialization
            specialization = None
            if "coder" in model_name or "code" in model_name:
                specialization = "coding"
            elif "creative" in model_name or "mixtral" in model_name:
                specialization = "creative"
            elif "llama" in model_name and "70b" in model_name:
                specialization = "genius"

            # Convert size to human-readable string
            size_bytes = model.get("size", 0)
            if isinstance(size_bytes, int):
                if size_bytes > 1024 * 1024 * 1024:
                    size_str = f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
                elif size_bytes > 1024 * 1024:
                    size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
                else:
                    size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                size_str = str(size_bytes)

            result.append(ModelInfo(
                name=model_name,
                tag=tag,
                size=size_str,
                modified=datetime.fromisoformat(model.get("modified", datetime.now().isoformat())),
                status=ModelStatus.INSTALLED,
                parameters=self._extract_parameters(model_name),
                specialization=specialization
            ))

        return result

    def _extract_parameters(self, model_name: str) -> Optional[int]:
        """
        Extract parameter count from model name
        """
        import re
        match = re.search(r'(\d+)b', model_name.lower())
        if match:
            return int(match.group(1))
        return None

# Create singleton instance
_manager_instance: Optional[OllamaModelManager] = None

def get_model_manager(
    board_registry: ServiceRegistry,
    request_logger: RequestLogger
) -> OllamaModelManager:
    """
    Get or create model manager instance
    """
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = OllamaModelManager(board_registry, request_logger)
    return _manager_instance