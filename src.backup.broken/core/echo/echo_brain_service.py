
# ===== ORCHESTRATION LAYER =====
from src.core.echo.echo_orchestrator_real import EchoOrchestrator

orchestrator = EchoOrchestrator()

@app.post("/api/echo/orchestrate")
async def orchestrate_task(task: dict):
    """Orchestrate complex creative tasks"""
    result = await orchestrator.orchestrate_creative_task(task)
    return result

@app.post("/api/echo/generate/image")
async def generate_image(request: dict):
    """Generate image through orchestration"""
    task = {
        "type": "image",
        "prompt": request.get("prompt"),
        "style": request.get("style", "anime")
    }
    return await orchestrator.orchestrate_creative_task(task)

@app.post("/api/echo/generate/trailer")
async def generate_trailer(request: dict):
    """Generate complete trailer"""
    task = {
        "type": "trailer",
        **request
    }
    return await orchestrator.orchestrate_creative_task(task)
