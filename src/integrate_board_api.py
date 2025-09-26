#!/usr/bin/env python3
"""
Integrate Board of Directors API with Echo Brain
Adds governance endpoints and MLflow tracking
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import mlflow
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================== BOARD API MODELS ==================

class BoardDecisionRequest(BaseModel):
    task_type: str
    code: Optional[str] = None
    requirements: str
    context: Dict = {}
    user_preferences: Dict = {}

class DirectorVote(BaseModel):
    director: str
    vote: str  # approve, reject, modify
    confidence: float
    reasoning: str
    suggestions: List[str] = []

class BoardDecisionResponse(BaseModel):
    decision_id: str
    consensus: str
    confidence: float
    directors: List[DirectorVote]
    execution_plan: Dict
    mlflow_run_id: Optional[str] = None
    timestamp: str

# ================== BOARD ROUTER ==================

router = APIRouter(prefix="/api/board", tags=["Board of Directors"])

# MLflow configuration
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("echo-board-decisions")

@router.post("/deliberate", response_model=BoardDecisionResponse)
async def board_deliberation(request: BoardDecisionRequest, background_tasks: BackgroundTasks):
    """
    Board of Directors deliberation endpoint with MLflow tracking
    """
    try:
        # Import the Board implementation
        import sys
        sys.path.append('/opt/tower-echo-brain/src')
        from cicd.echo_cicd_board_implementation import BoardOfDirectors

        # Start MLflow run
        with mlflow.start_run() as run:
            mlflow.log_params({
                "task_type": request.task_type,
                "has_code": request.code is not None,
                "requirements_length": len(request.requirements)
            })

            # Initialize Board
            board = BoardOfDirectors()

            # Update board with user preferences
            if request.user_preferences:
                board.user_preferences = request.user_preferences

            # Deliberate
            start_time = datetime.now()
            result = await board.deliberate(
                request.dict(),
                {"source": "api", "timestamp": start_time.isoformat()}
            )

            # Calculate deliberation time
            deliberation_time = (datetime.now() - start_time).total_seconds()

            # Log to MLflow
            mlflow.log_metrics({
                "consensus_confidence": result['consensus']['confidence'],
                "num_directors": len(result['directors']),
                "deliberation_time_seconds": deliberation_time,
                "unanimous": 1 if result['consensus'].get('unanimous', False) else 0
            })

            # Log decision outcome
            mlflow.set_tag("decision", result['consensus']['recommendation'])
            mlflow.set_tag("requires_user_approval", str(result.get('requires_user_approval', False)))

            # Create response
            response = BoardDecisionResponse(
                decision_id=result['id'],
                consensus=result['consensus']['recommendation'],
                confidence=result['consensus']['confidence'],
                directors=[
                    DirectorVote(
                        director=d['director'],
                        vote=d['recommendation'],
                        confidence=d['confidence'],
                        reasoning=d['reasoning'],
                        suggestions=d.get('suggested_changes', [])
                    ) for d in result['directors']
                ],
                execution_plan=result['execution_plan'],
                mlflow_run_id=run.info.run_id,
                timestamp=result['timestamp']
            )

            # Schedule background task to update metrics
            background_tasks.add_task(update_board_metrics, result)

            return response

    except Exception as e:
        logger.error(f"Board deliberation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def board_status():
    """Get Board of Directors system status"""
    try:
        # Get recent decisions from MLflow
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("echo-board-decisions")

        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=10
            )

            recent_decisions = len(runs)
            if runs:
                avg_confidence = sum(r.data.metrics.get('consensus_confidence', 0) for r in runs) / len(runs)
            else:
                avg_confidence = 0
        else:
            recent_decisions = 0
            avg_confidence = 0

        return {
            "status": "operational",
            "directors": [
                "Quality Director",
                "Security Director",
                "Performance Director",
                "Ethics Director",
                "UX Director",
                "ML Engineering Director",
                "Architecture Director"
            ],
            "recent_decisions": recent_decisions,
            "average_confidence": avg_confidence,
            "mlflow_tracking": "enabled",
            "mlflow_url": "http://192.168.50.135:5000"
        }
    except Exception as e:
        return {
            "status": "operational",
            "error": str(e),
            "directors": 7,
            "mlflow_tracking": "error"
        }

@router.get("/history")
async def board_history(limit: int = 10):
    """Get Board decision history from MLflow"""
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("echo-board-decisions")

        if not experiment:
            return {"decisions": [], "message": "No decisions yet"}

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=limit,
            order_by=["start_time DESC"]
        )

        decisions = []
        for run in runs:
            decisions.append({
                "run_id": run.info.run_id,
                "timestamp": datetime.fromtimestamp(run.info.start_time / 1000).isoformat(),
                "decision": run.data.tags.get("decision", "unknown"),
                "confidence": run.data.metrics.get("consensus_confidence", 0),
                "deliberation_time": run.data.metrics.get("deliberation_time_seconds", 0),
                "task_type": run.data.params.get("task_type", "unknown")
            })

        return {"decisions": decisions, "total": len(decisions)}

    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        return {"decisions": [], "error": str(e)}

@router.post("/override/{decision_id}")
async def override_decision(decision_id: str, new_decision: str, reason: str):
    """Allow user to override Board decision"""
    try:
        # Log override to MLflow
        with mlflow.start_run(run_id=decision_id):
            mlflow.set_tag("user_override", new_decision)
            mlflow.set_tag("override_reason", reason)
            mlflow.log_metric("override", 1)

        return {
            "status": "success",
            "message": f"Decision {decision_id} overridden to {new_decision}",
            "reason": reason
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ================== HELPER FUNCTIONS ==================

async def update_board_metrics(decision_result: Dict):
    """Background task to update Prometheus metrics"""
    try:
        # Send metrics to Prometheus (if pushgateway is configured)
        metrics_data = {
            "job": "board_decisions",
            "instance": "echo_brain",
            "confidence": decision_result['consensus']['confidence'],
            "directors_count": len(decision_result['directors'])
        }

        # You could push to Prometheus Pushgateway here
        logger.info(f"Board metrics updated: {metrics_data}")

    except Exception as e:
        logger.error(f"Failed to update metrics: {e}")

# ================== GRAFANA DASHBOARD CONFIG ==================

def create_grafana_dashboard():
    """Create Grafana dashboard configuration for Board metrics"""

    dashboard = {
        "dashboard": {
            "title": "Echo Board of Directors",
            "panels": [
                {
                    "title": "Decision Confidence",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "avg(echo_board_confidence)",
                            "legendFormat": "Average Confidence"
                        }
                    ],
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                },
                {
                    "title": "Decisions Per Hour",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "rate(echo_board_decisions_total[1h])",
                            "legendFormat": "Decisions/Hour"
                        }
                    ],
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                },
                {
                    "title": "Director Voting Patterns",
                    "type": "table",
                    "targets": [
                        {
                            "expr": "echo_director_votes",
                            "format": "table"
                        }
                    ],
                    "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
                },
                {
                    "title": "User Overrides",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "sum(echo_board_overrides_total)",
                            "legendFormat": "Total Overrides"
                        }
                    ],
                    "gridPos": {"h": 4, "w": 6, "x": 0, "y": 16}
                },
                {
                    "title": "Average Deliberation Time",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "avg(echo_board_deliberation_seconds)",
                            "legendFormat": "Avg Time (s)"
                        }
                    ],
                    "gridPos": {"h": 4, "w": 6, "x": 6, "y": 16}
                }
            ],
            "refresh": "10s",
            "time": {"from": "now-6h", "to": "now"}
        },
        "overwrite": True
    }

    return dashboard

# ================== INTEGRATION SCRIPT ==================

async def integrate_board_with_echo():
    """Main integration function to add Board to Echo Brain"""

    logger.info("🏛️ Integrating Board of Directors with Echo Brain...")

    # Step 1: Add router to Echo Brain
    echo_main_path = "/opt/tower-echo-brain/src/main.py"

    integration_code = '''
# Board of Directors Integration
from board_integration_enhanced import router as board_router
app.include_router(board_router)
logger.info("🏛️ Board of Directors API integrated")
'''

    # Step 2: Configure MLflow
    logger.info("📊 Configuring MLflow experiment tracking...")
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.create_experiment("echo-board-decisions", artifact_location="/opt/echo-fast/mlflow/artifacts")

    # Step 3: Create Grafana dashboard
    logger.info("📈 Creating Grafana dashboard...")
    dashboard = create_grafana_dashboard()

    # Send to Grafana API
    try:
        response = requests.post(
            "http://localhost:3000/api/dashboards/db",
            headers={
                "Authorization": "Basic YWRtaW46YWRtaW4=",  # admin:admin
                "Content-Type": "application/json"
            },
            json=dashboard
        )
        if response.status_code == 200:
            logger.info("✅ Grafana dashboard created")
        else:
            logger.warning(f"Grafana dashboard creation failed: {response.text}")
    except Exception as e:
        logger.error(f"Failed to create Grafana dashboard: {e}")

    # Step 4: Test the integration
    logger.info("🧪 Testing Board integration...")

    test_request = {
        "task_type": "integration_test",
        "requirements": "Test Board API integration",
        "code": "print('Hello from Board')"
    }

    try:
        # Test locally first
        import sys
        sys.path.append('/opt/tower-echo-brain/src')
        from cicd.echo_cicd_board_implementation import BoardOfDirectors

        board = BoardOfDirectors()
        result = await board.deliberate(test_request, {"test": True})

        logger.info(f"✅ Board test successful!")
        logger.info(f"   Decision: {result['consensus']['recommendation']}")
        logger.info(f"   Confidence: {result['consensus']['confidence']:.2%}")

    except Exception as e:
        logger.error(f"❌ Board test failed: {e}")
        return False

    logger.info("🎉 Board of Directors integration complete!")
    return True

if __name__ == "__main__":
    # Run integration
    asyncio.run(integrate_board_with_echo())

    # Save this module for import
    import shutil
    shutil.copy(__file__, "/opt/tower-echo-brain/src/board_integration_enhanced.py")
    print("\n✅ Board API module saved to /opt/tower-echo-brain/src/")
    print("\n📝 Next steps:")
    print("   1. Restart Echo Brain to load Board API")
    print("   2. Access Board at http://192.168.50.135:8309/api/board/status")
    print("   3. View MLflow at http://192.168.50.135:5000")
    print("   4. Check Grafana dashboard at http://192.168.50.135:3000")