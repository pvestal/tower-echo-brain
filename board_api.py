#!/usr/bin/env python3
"""
Board API for AI Assist Board of Directors
FastAPI endpoints for transparent decision tracking and user interaction
"""

import logging
import json
import uuid
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import asyncio
from contextlib import asynccontextmanager

from routing.request_logger import (
    RequestLogger, TaskDecision, DecisionPoint, DirectorEvaluation,
    Evidence, EvidenceType, DecisionStatus
)
from routing.service_registry import ServiceRegistry
from routing.auth_middleware import (
    get_current_user, get_optional_user, require_permission,
    authenticate_websocket, auth_middleware
)

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class TaskSubmissionRequest(BaseModel):
    task_description: str
    user_id: str = Field(default="default")
    priority: str = Field(default="normal")  # low, normal, high, critical
    context: Optional[Dict[str, Any]] = {}
    expected_completion_time: Optional[str] = None

class TaskSubmissionResponse(BaseModel):
    task_id: str
    status: str
    message: str
    estimated_completion_time: Optional[datetime] = None

class DecisionDetailsRequest(BaseModel):
    include_evidence: bool = Field(default=True)
    include_timeline: bool = Field(default=True)
    include_director_details: bool = Field(default=True)

class DecisionDetailsResponse(BaseModel):
    task_id: str
    current_status: str
    consensus_score: float
    confidence_score: float
    director_count: int
    evidence_count: int
    timeline: List[Dict[str, Any]]
    director_evaluations: List[Dict[str, Any]]
    user_override_available: bool
    estimated_completion: Optional[datetime]

class UserFeedbackRequest(BaseModel):
    feedback_type: str  # approve, reject, modify, comment
    feedback_content: str
    override_recommendation: Optional[str] = None
    reasoning: str

class UserFeedbackResponse(BaseModel):
    feedback_id: str
    status: str
    message: str
    decision_updated: bool

class BoardStatusResponse(BaseModel):
    active_tasks: int
    completed_tasks_today: int
    average_consensus_score: float
    average_processing_time: float
    director_status: Dict[str, Dict[str, Any]]
    system_health: str

class DirectorInfo(BaseModel):
    director_id: str
    director_name: str
    specialization: str
    status: str
    current_load: int
    average_response_time: float
    approval_rate: float
    last_activity: Optional[datetime]

class DirectorsListResponse(BaseModel):
    directors: List[DirectorInfo]
    total_count: int
    active_count: int

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Failed to send WebSocket message: {e}")

    async def broadcast_json(self, data: dict):
        message = json.dumps(data)
        await self.broadcast(message)

class BoardAPI:
    """
    FastAPI application for AI Assist Board of Directors
    Provides real-time decision tracking and user interaction
    """

    def __init__(self, request_logger: RequestLogger, service_registry: ServiceRegistry):
        self.request_logger = request_logger
        self.service_registry = service_registry
        self.connection_manager = ConnectionManager()
        self.active_evaluations: Dict[str, asyncio.Task] = {}

        # Initialize FastAPI app
        self.app = FastAPI(
            title="AI Assist Board API",
            description="API for transparent AI decision tracking and user control",
            version="1.0.0"
        )

        # Add global exception handler
        @self.app.exception_handler(Exception)
        async def global_exception_handler(request, exc):
            """Global exception handler for unhandled errors"""
            request_id = str(uuid.uuid4())
            logger.error(f"Unhandled exception [request_id: {request_id}]: {exc}", exc_info=True)

            # Don't expose internal error details in production
            if isinstance(exc, HTTPException):
                return JSONResponse(
                    status_code=exc.status_code,
                    content={"detail": exc.detail, "request_id": request_id}
                )

            return JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal server error",
                    "request_id": request_id,
                    "type": "internal_error"
                }
            )

        self._setup_routes()

    def _setup_routes(self):
        """Setup all API routes"""

        @self.app.post("/api/board/task", response_model=TaskSubmissionResponse)
        async def submit_task(
            request: TaskSubmissionRequest,
            background_tasks: BackgroundTasks,
            user_info: Dict[str, Any] = Depends(require_permission("board.submit_task"))
        ) -> TaskSubmissionResponse:
            """Submit a task for board evaluation"""
            try:
                task_id = str(uuid.uuid4())

                # Start task tracking
                task_decision = self.request_logger.start_task_tracking(
                    task_id=task_id,
                    user_id=user_info.get('user_id', 'anonymous'),
                    original_request=request.task_description
                )

                # Start asynchronous board evaluation
                background_tasks.add_task(
                    self._evaluate_task_async,
                    task_id,
                    request.task_description,
                    request.context or {},
                    request.priority
                )

                # Broadcast task submission
                await self.connection_manager.broadcast_json({
                    "type": "task_submitted",
                    "task_id": task_id,
                    "user_id": request.user_id,
                    "description": request.task_description,
                    "timestamp": datetime.utcnow().isoformat()
                })

                return TaskSubmissionResponse(
                    task_id=task_id,
                    status="submitted",
                    message="Task submitted for board evaluation",
                    estimated_completion_time=datetime.utcnow() + timedelta(minutes=5)
                )

            except Exception as e:
                logger.error(f"Failed to submit task: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/board/decisions/{task_id}", response_model=DecisionDetailsResponse)
        async def get_decision_details(
            task_id: str,
            request: DecisionDetailsRequest = DecisionDetailsRequest()
        ) -> DecisionDetailsResponse:
            """Get detailed decision information for a task"""
            try:
                task_decision = self.request_logger.get_task_decision(task_id)
                if not task_decision:
                    raise HTTPException(status_code=404, detail="Task not found")

                # Format timeline
                timeline = []
                if request.include_timeline:
                    for point in task_decision.decision_points:
                        timeline.append({
                            "id": point.id,
                            "timestamp": point.timestamp.isoformat(),
                            "status": point.status.value,
                            "description": point.description,
                            "consensus_score": point.consensus_score,
                            "confidence": point.total_confidence,
                            "director_count": len(point.director_evaluations),
                            "user_feedback": point.user_feedback
                        })

                # Format director evaluations
                director_evaluations = []
                if request.include_director_details:
                    for point in task_decision.decision_points:
                        for evaluation in point.director_evaluations:
                            eval_dict = {
                                "director_id": evaluation.director_id,
                                "director_name": evaluation.director_name,
                                "recommendation": evaluation.recommendation,
                                "confidence": evaluation.confidence,
                                "risk_score": evaluation.risk_score,
                                "processing_time": evaluation.processing_time,
                                "timestamp": evaluation.timestamp.isoformat(),
                                "reasoning": evaluation.reasoning
                            }

                            if request.include_evidence:
                                eval_dict["evidence"] = [
                                    {
                                        "id": evidence.id,
                                        "type": evidence.type.value,
                                        "source": evidence.source,
                                        "weight": evidence.weight,
                                        "confidence": evidence.confidence,
                                        "reasoning": evidence.reasoning,
                                        "data_summary": self._summarize_evidence_data(evidence.data)
                                    }
                                    for evidence in evaluation.evidence
                                ]

                            director_evaluations.append(eval_dict)

                return DecisionDetailsResponse(
                    task_id=task_id,
                    current_status=task_decision.status.value,
                    consensus_score=task_decision.consensus_score,
                    confidence_score=task_decision.confidence_score,
                    director_count=len(task_decision.director_participation),
                    evidence_count=task_decision.evidence_count,
                    timeline=timeline,
                    director_evaluations=director_evaluations,
                    user_override_available=task_decision.status in [
                        DecisionStatus.COMPLETED,
                        DecisionStatus.IN_PROGRESS
                    ],
                    estimated_completion=task_decision.completion_timestamp
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get decision details: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/board/feedback/{task_id}", response_model=UserFeedbackResponse)
        async def submit_user_feedback(
            task_id: str,
            request: UserFeedbackRequest,
            user_info: Dict[str, Any] = Depends(require_permission("board.provide_feedback"))
        ) -> UserFeedbackResponse:
            """Submit user feedback or override for a task decision"""
            try:
                task_decision = self.request_logger.get_task_decision(task_id)
                if not task_decision:
                    raise HTTPException(status_code=404, detail="Task not found")

                feedback_id = str(uuid.uuid4())
                decision_updated = False

                # Handle override requests
                if request.feedback_type in ["approve", "reject", "modify"]:
                    if not request.override_recommendation:
                        raise HTTPException(
                            status_code=400,
                            detail="Override recommendation required for override feedback"
                        )

                    success = self.request_logger.add_user_override(
                        task_id=task_id,
                        user_id=user_info.get('user_id', 'anonymous'),
                        override_type=request.feedback_type,
                        original_rec=task_decision.final_recommendation,
                        new_rec=request.override_recommendation,
                        reasoning=request.reasoning
                    )

                    if success:
                        decision_updated = True

                        # Broadcast override
                        await self.connection_manager.broadcast_json({
                            "type": "user_override",
                            "task_id": task_id,
                            "override_type": request.feedback_type,
                            "timestamp": datetime.utcnow().isoformat()
                        })

                # Handle general feedback
                else:
                    # TODO: Store general feedback in database
                    logger.info(f"Received feedback for task {task_id}: {request.feedback_content}")

                return UserFeedbackResponse(
                    feedback_id=feedback_id,
                    status="accepted",
                    message="Feedback recorded successfully",
                    decision_updated=decision_updated
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to submit feedback: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/board/status", response_model=BoardStatusResponse)
        async def get_board_status() -> BoardStatusResponse:
            """Get current board status and metrics"""
            try:
                # Get active tasks count
                active_tasks = len(self.request_logger.active_tasks)

                # Get analytics for today
                today = datetime.utcnow().date()
                start_of_day = datetime.combine(today, datetime.min.time())
                end_of_day = datetime.combine(today, datetime.max.time())

                analytics = self.request_logger.get_board_analytics(start_of_day, end_of_day)

                # Get director status
                director_status = {}
                for director_id, director in self.service_registry.directors.items():
                    director_status[director_id] = {
                        "name": director.get_director_name(),
                        "specialization": director.get_specialization(),
                        "status": "active",  # TODO: Implement actual status tracking
                        "current_load": 0,  # TODO: Track current evaluations
                        "response_time": 0.0  # TODO: Calculate from recent evaluations
                    }

                return BoardStatusResponse(
                    active_tasks=active_tasks,
                    completed_tasks_today=analytics.get("overall_stats", {}).get("completed_tasks", 0),
                    average_consensus_score=analytics.get("overall_stats", {}).get("avg_consensus", 0.0),
                    average_processing_time=analytics.get("overall_stats", {}).get("avg_processing_time", 0.0),
                    director_status=director_status,
                    system_health="healthy"  # TODO: Implement health checks
                )

            except Exception as e:
                logger.error(f"Failed to get board status: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/board/directors", response_model=DirectorsListResponse)
        async def get_directors_list() -> DirectorsListResponse:
            """Get list of all directors with their details"""
            try:
                directors = []
                active_count = 0

                for director_id, director in self.service_registry.directors.items():
                    # TODO: Get actual metrics from database
                    director_info = DirectorInfo(
                        director_id=director_id,
                        director_name=director.get_director_name(),
                        specialization=director.get_specialization(),
                        status="active",
                        current_load=0,
                        average_response_time=0.0,
                        approval_rate=0.0,
                        last_activity=datetime.utcnow()
                    )
                    directors.append(director_info)
                    active_count += 1

                return DirectorsListResponse(
                    directors=directors,
                    total_count=len(directors),
                    active_count=active_count
                )

            except Exception as e:
                logger.error(f"Failed to get directors list: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.websocket("/api/board/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time board updates with authentication"""
            # Authenticate WebSocket connection
            user_info = await authenticate_websocket(websocket)

            if not user_info:
                await websocket.close(code=1008, reason="Authentication required")
                logger.warning("WebSocket connection rejected: authentication failed")
                return

            # Check connection limits
            if len(self.connection_manager.active_connections) >= int(os.environ.get("MAX_WEBSOCKET_CONNECTIONS", "50")):
                await websocket.close(code=1013, reason="Connection limit exceeded")
                logger.warning(f"WebSocket connection rejected: limit exceeded for user {user_info.get('user_id', 'unknown')}")
                return

            await self.connection_manager.connect(websocket)
            logger.info(f"Authenticated WebSocket connection for user: {user_info.get('user_id', 'unknown')}")

            try:
                # Send welcome message with user info
                await websocket.send_text(json.dumps({
                    "type": "welcome",
                    "user_id": user_info.get('user_id', 'unknown'),
                    "message": "Connected to AI Assist Board",
                    "permissions": user_info.get('permissions', [])
                }))

                while True:
                    # Keep connection alive and listen for client messages
                    data = await websocket.receive_text()

                    # Handle client messages (e.g., subscribe to specific task updates)
                    try:
                        message = json.loads(data)

                        if message.get("type") == "ping":
                            await websocket.send_text(json.dumps({"type": "pong"}))
                        elif message.get("type") == "subscribe":
                            # Handle subscription requests (could add permission checks here)
                            task_id = message.get("task_id")
                            if task_id:
                                await websocket.send_text(json.dumps({
                                    "type": "subscribed",
                                    "task_id": task_id
                                }))
                                logger.info(f"User {user_info.get('user_id')} subscribed to task {task_id}")

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON from WebSocket client {user_info.get('user_id', 'unknown')}: {data}")
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "Invalid JSON format"
                        }))

            except WebSocketDisconnect:
                self.connection_manager.disconnect(websocket)
                logger.info(f"WebSocket disconnected for user: {user_info.get('user_id', 'unknown')}")

        @self.app.get("/api/board/analytics")
        async def get_board_analytics(
            start_date: Optional[str] = None,
            end_date: Optional[str] = None
        ):
            """Get board analytics for specified date range"""
            try:
                if start_date:
                    start = datetime.fromisoformat(start_date)
                else:
                    start = datetime.utcnow() - timedelta(days=7)

                if end_date:
                    end = datetime.fromisoformat(end_date)
                else:
                    end = datetime.utcnow()

                analytics = self.request_logger.get_board_analytics(start, end)
                return analytics

            except Exception as e:
                logger.error(f"Failed to get analytics: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/board/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "active_connections": len(self.connection_manager.active_connections),
                "active_tasks": len(self.request_logger.active_tasks)
            }

    async def _evaluate_task_async(self, task_id: str, task_description: str,
                                 context: Dict[str, Any], priority: str):
        """Asynchronously evaluate task with all directors"""
        try:
            logger.info(f"Starting async evaluation for task {task_id}")

            # Create evaluation tasks for all directors
            evaluation_tasks = []
            for director_id, director in self.service_registry.directors.items():
                eval_task = asyncio.create_task(
                    self._evaluate_with_director(task_id, director_id, director,
                                               task_description, context, priority)
                )
                evaluation_tasks.append(eval_task)

            # Wait for all evaluations to complete
            evaluations = await asyncio.gather(*evaluation_tasks, return_exceptions=True)

            # Process results
            successful_evaluations = []
            for i, result in enumerate(evaluations):
                if isinstance(result, Exception):
                    director_id = list(self.service_registry.directors.keys())[i]
                    logger.error(f"Director {director_id} evaluation failed: {result}")
                else:
                    successful_evaluations.append(result)

            # Update task with all evaluations
            for evaluation in successful_evaluations:
                if evaluation:
                    self.request_logger.add_director_evaluation(task_id, evaluation)

            # Generate final recommendation
            final_recommendation = self._generate_final_recommendation(successful_evaluations)

            # Finalize task
            self.request_logger.finalize_task_decision(task_id, final_recommendation)

            # Broadcast completion
            await self.connection_manager.broadcast_json({
                "type": "task_completed",
                "task_id": task_id,
                "final_recommendation": final_recommendation,
                "director_count": len(successful_evaluations),
                "timestamp": datetime.utcnow().isoformat()
            })

            logger.info(f"Completed evaluation for task {task_id}: {final_recommendation}")

        except Exception as e:
            logger.error(f"Failed to evaluate task {task_id}: {e}")

            # Broadcast error
            await self.connection_manager.broadcast_json({
                "type": "task_error",
                "task_id": task_id,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })

    async def _evaluate_with_director(self, task_id: str, director_id: str,
                                    director, task_description: str,
                                    context: Dict[str, Any], priority: str) -> Optional[DirectorEvaluation]:
        """Evaluate task with a specific director"""
        try:
            start_time = datetime.utcnow()

            # Broadcast director started
            await self.connection_manager.broadcast_json({
                "type": "director_started",
                "task_id": task_id,
                "director_id": director_id,
                "director_name": director.get_director_name(),
                "timestamp": start_time.isoformat()
            })

            # Perform evaluation
            evaluation_result = await director.evaluate_async(
                task_description=task_description,
                context=context,
                priority=priority
            )

            processing_time = (datetime.utcnow() - start_time).total_seconds()

            # Create evaluation object
            evaluation = DirectorEvaluation(
                director_id=director_id,
                director_name=director.get_director_name(),
                recommendation=evaluation_result.get("recommendation", "approve"),
                confidence=evaluation_result.get("confidence", 0.5),
                reasoning=evaluation_result.get("reasoning", "No reasoning provided"),
                evidence=[],  # TODO: Extract evidence from evaluation_result
                processing_time=processing_time,
                risk_score=evaluation_result.get("risk_score", 0.0),
                timestamp=datetime.utcnow()
            )

            # Broadcast director completed
            await self.connection_manager.broadcast_json({
                "type": "director_completed",
                "task_id": task_id,
                "director_id": director_id,
                "director_name": director.get_director_name(),
                "recommendation": evaluation.recommendation,
                "confidence": evaluation.confidence,
                "timestamp": datetime.utcnow().isoformat()
            })

            return evaluation

        except Exception as e:
            logger.error(f"Director {director_id} evaluation failed: {e}")

            # Broadcast director error
            await self.connection_manager.broadcast_json({
                "type": "director_error",
                "task_id": task_id,
                "director_id": director_id,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })

            return None

    def _generate_final_recommendation(self, evaluations: List[DirectorEvaluation]) -> str:
        """Generate final recommendation from director evaluations"""
        if not evaluations:
            return "Unable to generate recommendation - no director evaluations available"

        # Count recommendations
        recommendations = {}
        total_confidence = 0.0

        for evaluation in evaluations:
            rec = evaluation.recommendation
            confidence = evaluation.confidence

            if rec not in recommendations:
                recommendations[rec] = {"count": 0, "total_confidence": 0.0}

            recommendations[rec]["count"] += 1
            recommendations[rec]["total_confidence"] += confidence
            total_confidence += confidence

        # Find recommendation with highest weighted score
        best_recommendation = None
        best_score = 0.0

        for rec, data in recommendations.items():
            # Score = (count / total_evaluations) * (avg_confidence)
            avg_confidence = data["total_confidence"] / data["count"]
            score = (data["count"] / len(evaluations)) * avg_confidence

            if score > best_score:
                best_score = score
                best_recommendation = rec

        return best_recommendation or "approve"

    def _summarize_evidence_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of evidence data for API response"""
        if not data:
            return {}

        # Return a summary that doesn't expose sensitive details
        return {
            "data_type": type(data).__name__,
            "keys": list(data.keys()) if isinstance(data, dict) else [],
            "size": len(str(data))
        }

    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance"""
        return self.app

# Factory function to create BoardAPI instance
def create_board_api(request_logger: RequestLogger,
                    service_registry: ServiceRegistry) -> BoardAPI:
    """
    Factory function to create BoardAPI instance

    Args:
        request_logger: RequestLogger instance
        service_registry: ServiceRegistry instance

    Returns:
        BoardAPI: Configured BoardAPI instance
    """
    return BoardAPI(request_logger, service_registry)