import pytest
import asyncio
from httpx import AsyncClient
from src.main_refactored import app

@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_video_generation():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/api/video/generate", json={
            "prompt": "test prompt",
            "style": "anime",
            "duration": 5
        })
        assert response.status_code in [200, 202]
