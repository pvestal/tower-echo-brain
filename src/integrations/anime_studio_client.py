"""
Anime Studio API Client
========================
HTTP client for Echo Brain → Anime Studio (localhost:8401).
Localhost requests get admin fallback auth — no token needed.
"""
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

ANIME_STUDIO_BASE = "http://localhost:8401"


class AnimeStudioClient:
    """Async HTTP client for anime-studio API."""

    def __init__(self, base_url: str = ANIME_STUDIO_BASE, timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout

    async def _request(self, method: str, path: str, **kwargs) -> dict | list | None:
        """Make an authenticated request to anime-studio."""
        kwargs.setdefault("timeout", self.timeout)
        url = f"{self.base_url}{path}"
        try:
            async with httpx.AsyncClient() as client:
                resp = await getattr(client, method)(url, **kwargs)
                resp.raise_for_status()
                if resp.headers.get("content-type", "").startswith("application/json"):
                    return resp.json()
                return {"status": "ok", "status_code": resp.status_code}
        except httpx.ConnectError:
            logger.error("Anime Studio unreachable at %s", self.base_url)
            return None
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %d from %s: %s", e.response.status_code, path, e.response.text[:200])
            return None
        except Exception as e:
            logger.error("Request failed %s %s: %s", method, path, e)
            return None

    async def get(self, path: str, **kwargs):
        return await self._request("get", path, **kwargs)

    async def post(self, path: str, **kwargs):
        return await self._request("post", path, **kwargs)

    async def put(self, path: str, **kwargs):
        return await self._request("put", path, **kwargs)

    async def patch(self, path: str, **kwargs):
        return await self._request("patch", path, **kwargs)

    # ── Projects ──────────────────────────────────────────────

    async def list_projects(self) -> list[dict]:
        data = await self.get("/api/story/projects")
        if not data:
            return []
        return data.get("projects", data) if isinstance(data, dict) else data

    async def get_project(self, project_id: int) -> dict | None:
        return await self.get(f"/api/story/projects/{project_id}")

    async def create_project(self, name: str, content_rating: str = "R",
                             genre: str = "", premise: str = "") -> dict | None:
        return await self.post("/api/story/projects", json={
            "name": name, "content_rating": content_rating,
            "genre": genre, "premise": premise,
        })

    # ── Characters ────────────────────────────────────────────

    async def list_characters(self, project_id: int) -> list[dict]:
        data = await self.get(f"/api/story/projects/{project_id}/characters")
        if not data:
            return []
        return data.get("characters", data) if isinstance(data, dict) else data

    async def update_character(self, character_id: int, updates: dict) -> dict | None:
        return await self.patch(f"/api/story/characters/{character_id}", json=updates)

    # ── Orchestrator ──────────────────────────────────────────

    async def orchestrator_status(self) -> dict | None:
        return await self.get("/api/system/orchestrator/status")

    async def orchestrator_toggle(self, enabled: bool) -> dict | None:
        return await self.post("/api/system/orchestrator/toggle", json={"enabled": enabled})

    async def orchestrator_initialize(self, project_id: int, target: int = 100) -> dict | None:
        return await self.post("/api/system/orchestrator/initialize",
                               json={"project_id": project_id, "training_target": target})

    # ── Generation ────────────────────────────────────────────

    async def generate_image(self, character_slug: str, count: int = 1,
                             prompt_override: str | None = None) -> dict:
        results, errors = [], []
        for _ in range(min(count, 5)):
            body: dict = {"generation_type": "image"}
            if prompt_override:
                body["prompt_override"] = prompt_override
            data = await self.post(f"/api/visual/generate/{character_slug}", json=body)
            if data:
                results.append(data)
            else:
                errors.append(f"Failed for {character_slug}")
        return {"generated": len(results), "results": results, "errors": errors}

    # ── Approval ──────────────────────────────────────────────

    async def pending_images(self, project_id: int) -> list[dict]:
        data = await self.get("/api/training/approval/pending", params={"project_id": project_id})
        if not data:
            return []
        return data.get("images", data) if isinstance(data, dict) else data

    async def approve_image(self, image_id: str, approved: bool) -> dict | None:
        return await self.post("/api/training/approval/approve", json={
            "image_id": image_id, "approved": approved,
        })

    # ── Scenes & Shots ────────────────────────────────────────

    async def list_scenes(self, project_id: int) -> list[dict]:
        data = await self.get(f"/api/scenes", params={"project_id": project_id})
        if not data:
            return []
        return data.get("scenes", data) if isinstance(data, dict) else data

    async def pending_videos(self, project_id: int) -> list[dict]:
        data = await self.get(f"/api/scenes/pending-videos", params={"project_id": project_id})
        if not data:
            return []
        return data.get("videos", data) if isinstance(data, dict) else data

    # ── Training ──────────────────────────────────────────────

    async def training_status(self) -> dict | None:
        return await self.get("/api/training/status")

    async def replenish(self, target: int = 50, batch_size: int = 5) -> dict | None:
        return await self.post("/api/training/replenish", json={
            "target_per_character": target, "max_batch_size": batch_size,
        })

    # ── Pipeline ──────────────────────────────────────────────

    async def pipeline_status(self, project_id: int) -> dict | None:
        return await self.get(f"/api/system/orchestrator/pipeline/{project_id}")

    # ── Health ────────────────────────────────────────────────

    async def health(self) -> bool:
        data = await self.get("/api/system/health")
        return data is not None


# Global singleton
anime_studio = AnimeStudioClient()
