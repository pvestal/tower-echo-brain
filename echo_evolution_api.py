#!/usr/bin/env python3
"""
Echo Brain Evolution System - Simplified Activation
Activates existing evolution components with minimal changes
"""
import asyncio
import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# Add Echo modules to path
sys.path.insert(0, '/opt/tower-echo-brain')

# Import existing modules
from echo_autonomous_evolution import EchoAutonomousEvolution, EvolutionTrigger
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Echo Evolution API", version="1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize evolution system
evolution_system = None

@app.on_event("startup")
async def startup():
    """Initialize evolution system on startup"""
    global evolution_system
    try:
        logger.info("Starting Echo Evolution System...")
        evolution_system = EchoAutonomousEvolution()

        # Start background tasks
        asyncio.create_task(scheduled_evolution())

        logger.info("✅ Echo Evolution System started successfully")
    except Exception as e:
        logger.error(f"Failed to start evolution: {e}")

async def scheduled_evolution():
    """Run scheduled evolution cycles"""
    while True:
        try:
            # Wait 24 hours between evolution cycles
            await asyncio.sleep(86400)

            if evolution_system:
                logger.info("Running scheduled evolution cycle...")
                result = await evolution_system.trigger_evolution(
                    EvolutionTrigger.SCHEDULED,
                    {"timestamp": datetime.now().isoformat()}
                )
                logger.info(f"Evolution result: {result}")
        except Exception as e:
            logger.error(f"Scheduled evolution error: {e}")
            await asyncio.sleep(3600)  # Wait 1 hour on error

@app.get("/api/echo/evolution/status")
async def get_status():
    """Get evolution system status"""
    if not evolution_system:
        return {"status": "initializing"}

    return {
        "status": "active",
        "timestamp": datetime.now().isoformat(),
        "config": evolution_system.evolution_config,
        "metrics": evolution_system.evolution_metrics,
        "capabilities": {
            "self_analysis": bool(evolution_system.self_analysis),
            "git_manager": bool(evolution_system.git_manager)
        }
    }

@app.post("/api/echo/evolution/trigger")
async def trigger_evolution(reason: str = "manual"):
    """Manually trigger evolution"""
    if not evolution_system:
        raise HTTPException(status_code=503, detail="Evolution system not ready")

    try:
        result = await evolution_system.trigger_evolution(
            EvolutionTrigger.MANUAL,
            {"reason": reason, "timestamp": datetime.now().isoformat()}
        )
        return {"status": "triggered", "result": result}
    except Exception as e:
        logger.error(f"Evolution trigger error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/echo/evolution/analyze")
async def analyze_self(depth: str = "functional"):
    """Trigger self-analysis"""
    if not evolution_system or not evolution_system.self_analysis:
        raise HTTPException(status_code=503, detail="Analysis system not ready")

    try:
        from echo_self_analysis import AnalysisDepth
        depth_enum = AnalysisDepth[depth.upper()]
        result = await evolution_system.self_analysis.analyze(
            depth_enum,
            {"timestamp": datetime.now().isoformat()}
        )
        return result
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/echo/evolution/git-status")
async def get_git_status():
    """Get git repository status"""
    if not evolution_system or not evolution_system.git_manager:
        return {"error": "Git manager not initialized"}

    try:
        status = evolution_system.git_manager.get_git_status()
        return status
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Echo Evolution"}

if __name__ == "__main__":
    # Run the service
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8311,
        log_level="info"
    )