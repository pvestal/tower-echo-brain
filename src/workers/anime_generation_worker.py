#!/usr/bin/env python3
"""
Anime Generation Worker for Echo Brain
Autonomous worker for generating anime videos with proper workflows
"""

import logging
import asyncio
import httpx
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class AnimeGenerationWorker:
    """
    Autonomous worker for anime video generation
    Integrates with Tower Anime Production API
    """

    def __init__(self):
        self.anime_api_url = "http://localhost:8328"
        self.active_jobs = {}

    async def generate_anime_video(
        self,
        prompt: str,
        workflow: str = "anime_30sec_rife_workflow",
        character_name: Optional[str] = None,
        style: str = "anime"
    ) -> Dict[str, Any]:
        """
        Generate anime video using specified workflow

        Args:
            prompt: Text description of the scene
            workflow: Workflow to use (defaults to 30sec RIFE)
            character_name: Optional character name
            style: Visual style (anime, cyberpunk, etc)

        Returns:
            Dict with job_id and status
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Submit generation request
                response = await client.post(
                    f"{self.anime_api_url}/api/video/generate",
                    json={
                        "prompt": prompt,
                        "workflow": workflow,
                        "character_name": character_name,
                        "style": style,
                        "steps": 20,
                        "width": 512,
                        "height": 288
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    job_id = result["job_id"]

                    # Track job
                    self.active_jobs[job_id] = {
                        "status": "queued",
                        "started_at": datetime.now(),
                        "prompt": prompt,
                        "workflow": workflow
                    }

                    logger.info(f"🎬 Anime generation job {job_id} started with {workflow}")

                    # Start monitoring in background
                    asyncio.create_task(self._monitor_job(job_id))

                    return result
                else:
                    logger.error(f"Failed to start generation: {response.status_code}")
                    return {"error": f"API error: {response.status_code}"}

        except Exception as e:
            logger.error(f"Anime generation failed: {e}")
            return {"error": str(e)}

    async def _monitor_job(self, job_id: str):
        """Monitor job progress in background"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                for i in range(60):  # Monitor for up to 3 minutes
                    await asyncio.sleep(3)

                    response = await client.get(
                        f"{self.anime_api_url}/api/video/status/{job_id}"
                    )

                    if response.status_code == 200:
                        status = response.json()
                        self.active_jobs[job_id]["status"] = status["status"]
                        self.active_jobs[job_id]["progress"] = status["progress"]

                        if status["status"] == "completed":
                            logger.info(f"✅ Job {job_id} completed: {status['video_url']}")
                            self.active_jobs[job_id]["video_url"] = status["video_url"]
                            break
                        elif status["status"] == "failed":
                            logger.error(f"❌ Job {job_id} failed: {status.get('error_message')}")
                            break

        except Exception as e:
            logger.error(f"Error monitoring job {job_id}: {e}")

    async def generate_episode_segment(
        self,
        episode_num: int,
        segment_num: int,
        scene_description: str
    ) -> Dict[str, Any]:
        """
        Generate a specific episode segment

        Args:
            episode_num: Episode number
            segment_num: Segment number within episode
            scene_description: Description of the scene

        Returns:
            Generation result
        """
        prompt = f"Episode {episode_num}, Scene {segment_num}: {scene_description}"

        # Use high-quality workflow for episode content
        return await self.generate_anime_video(
            prompt=prompt,
            workflow="anime_30sec_rife_workflow",
            style="high_quality_anime"
        )

    async def batch_generate(self, prompts: list[str]) -> list[Dict[str, Any]]:
        """
        Generate multiple videos in parallel

        Args:
            prompts: List of prompt strings

        Returns:
            List of generation results
        """
        tasks = [
            self.generate_anime_video(prompt=prompt)
            for prompt in prompts
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log results
        successful = sum(1 for r in results if not isinstance(r, Exception) and "error" not in r)
        logger.info(f"📊 Batch generation: {successful}/{len(prompts)} successful")

        return results

    def get_active_jobs(self) -> Dict[str, Any]:
        """Get status of all active jobs"""
        return self.active_jobs.copy()

    async def health_check(self) -> Dict[str, Any]:
        """Check worker and API health"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.anime_api_url}/api/health")

                return {
                    "worker": "healthy",
                    "api": "healthy" if response.status_code == 200 else "unhealthy",
                    "active_jobs": len(self.active_jobs),
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {
                "worker": "healthy",
                "api": "error",
                "error": str(e),
                "active_jobs": len(self.active_jobs),
                "timestamp": datetime.now().isoformat()
            }

# Global instance for Echo Brain integration
anime_worker = AnimeGenerationWorker()