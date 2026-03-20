"""GPU Arbiter Client — lightweight client for Echo Brain to coordinate with anime-studio.

Before heavy Ollama work (gemma3 vision, reasoning), Echo Brain workers should
check the arbiter to see if the AMD GPU is available or busy with ComfyUI-ROCm
or anime-studio vision batches.

Usage:
    from services.gpu_arbiter_client import arbiter

    # Check if OK to use gemma3
    if await arbiter.can_use_heavy_model():
        # safe to call Ollama with gemma3
        ...
    else:
        # defer or use lighter model

    # Claim GPU for long reasoning task
    claim_id = await arbiter.claim("echo_reasoning", duration_s=120)
    try:
        # do heavy work
        ...
    finally:
        await arbiter.release(claim_id)
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

ARBITER_BASE = "http://localhost:8401/api/system/gpu/arbiter"
TIMEOUT = 5.0


class GpuArbiterClient:
    """Async client for the anime-studio GPU Arbiter."""

    async def get_status(self) -> dict:
        """Get full arbiter status."""
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.get(ARBITER_BASE)
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.warning(f"Arbiter status check failed: {e}")
            return {}

    async def is_vision_busy(self) -> bool:
        """Check if anime-studio is running a vision review batch."""
        status = await self.get_status()
        claims = status.get("claims", {})
        return any(c.get("type") == "vision_review" for c in claims.values())

    async def is_rocm_busy(self) -> bool:
        """Check if ComfyUI-ROCm is generating."""
        status = await self.get_status()
        return status.get("comfyui_rocm", {}).get("busy", False)

    async def is_heavy_model_loaded(self) -> bool:
        """Check if gemma3 is currently loaded in Ollama."""
        status = await self.get_status()
        return status.get("vision_model", {}).get("loaded", False)

    async def can_use_heavy_model(self) -> bool:
        """Check if it's safe to use a heavy Ollama model on GPU (gemma3 vision).

        Returns True if:
        - No vision review batch is running
        - ComfyUI-ROCm is not actively generating
        - No active COMFYUI_ROCM claim (video gen in progress)
        - OR arbiter is unreachable (fail-open)
        """
        try:
            status = await self.get_status()
            if not status:
                return True  # Fail open if arbiter unreachable

            # If any claim is active, don't load GPU models
            claims = status.get("claims", {})
            for c in claims.values():
                ctype = c.get("type", "")
                if ctype in ("vision_review", "comfyui_rocm"):
                    logger.info(f"Heavy model deferred: {ctype} claim active ({c.get('caller')})")
                    return False

            # If ComfyUI-ROCm is busy, don't contend for VRAM
            if status.get("comfyui_rocm", {}).get("busy", False):
                logger.info("Heavy model deferred: ComfyUI-ROCm generating")
                return False

            return True
        except Exception:
            return True  # Fail open

    async def claim(self, task_type: str = "echo_reasoning",
                    duration_s: int = 300) -> Optional[str]:
        """Claim AMD GPU for Echo Brain work. Returns claim_id or None."""
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.post(f"{ARBITER_BASE}/claim", json={
                    "type": task_type,
                    "caller": "echo_brain",
                    "duration_s": duration_s,
                })
                resp.raise_for_status()
                data = resp.json()
                if data.get("granted"):
                    return data.get("claim_id")
                logger.info(f"GPU claim denied: {data.get('reason')}")
                return None
        except Exception as e:
            logger.warning(f"GPU claim failed: {e}")
            return None

    async def release(self, claim_id: Optional[str]) -> bool:
        """Release a GPU claim."""
        if not claim_id:
            return True
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.post(f"{ARBITER_BASE}/release", json={
                    "claim_id": claim_id,
                })
                resp.raise_for_status()
                return resp.json().get("released", False)
        except Exception as e:
            logger.warning(f"GPU release failed: {e}")
            return False


# Module-level singleton
arbiter = GpuArbiterClient()
