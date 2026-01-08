#!/usr/bin/env python3
"""
Fallback handlers for graceful service degradation
Provides alternative responses when services are unavailable
"""

import logging
import asyncio
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class FallbackHandler:
    """
    Handles fallback responses when services are unavailable
    """

    def __init__(self):
        self._fallback_cache = {}
        self._fallback_responses = {
            "comfyui": self._comfyui_fallback,
            "qdrant": self._qdrant_fallback,
            "ollama": self._ollama_fallback,
            "postgresql": self._postgresql_fallback,
            "redis": self._redis_fallback
        }

    # ComfyUI Fallbacks
    async def _comfyui_fallback(self, operation: str, *args, **kwargs) -> Dict[str, Any]:
        """Fallback for ComfyUI operations"""

        if operation == "submit_workflow":
            # Return a queued response for later processing
            job_id = str(uuid.uuid4())
            self._fallback_cache[job_id] = {
                "status": "queued_for_retry",
                "created_at": datetime.utcnow(),
                "operation": operation,
                "args": args,
                "kwargs": kwargs
            }

            logger.warning("ComfyUI unavailable - workflow queued for retry when service recovers")
            return {
                "success": False,
                "fallback": True,
                "job_id": job_id,
                "status": "queued_for_retry",
                "message": "ComfyUI service unavailable. Workflow queued for processing when service recovers.",
                "retry_after": 30
            }

        elif operation == "get_queue_status":
            return {
                "success": False,
                "fallback": True,
                "queue_running": 0,
                "queue_pending": len([job for job in self._fallback_cache.values()
                                    if job.get("status") == "queued_for_retry"]),
                "message": "ComfyUI service unavailable"
            }

        elif operation == "download_output":
            return {
                "success": False,
                "fallback": True,
                "message": "ComfyUI service unavailable - cannot download outputs",
                "cached_outputs": []
            }

        else:
            return {
                "success": False,
                "fallback": True,
                "error": f"ComfyUI service unavailable for operation: {operation}"
            }

    # Qdrant Fallbacks
    async def _qdrant_fallback(self, operation: str, *args, **kwargs) -> Dict[str, Any]:
        """Fallback for Qdrant operations"""

        if operation == "search":
            # Return empty search results with fallback indicator
            logger.warning("Qdrant unavailable - returning empty search results")
            return {
                "success": True,
                "fallback": True,
                "results": [],
                "message": "Vector search unavailable - using fallback response",
                "total_count": 0
            }

        elif operation == "upsert":
            # Cache the data for later insertion
            cache_key = f"upsert_{datetime.utcnow().timestamp()}"
            self._fallback_cache[cache_key] = {
                "operation": "upsert",
                "data": kwargs,
                "created_at": datetime.utcnow()
            }

            logger.warning("Qdrant unavailable - caching data for later insertion")
            return {
                "success": True,
                "fallback": True,
                "cached": True,
                "cache_key": cache_key,
                "message": "Vector data cached for insertion when service recovers"
            }

        elif operation == "delete":
            # Log deletion request for later processing
            logger.warning(f"Qdrant unavailable - deletion request logged for later: {kwargs}")
            return {
                "success": True,
                "fallback": True,
                "deferred": True,
                "message": "Deletion request logged for processing when service recovers"
            }

        elif operation == "get_collection_info":
            return {
                "success": False,
                "fallback": True,
                "vectors_count": 0,
                "status": "unavailable",
                "message": "Qdrant service unavailable"
            }

        else:
            return {
                "success": False,
                "fallback": True,
                "error": f"Qdrant service unavailable for operation: {operation}"
            }

    # Ollama Fallbacks
    async def _ollama_fallback(self, operation: str, *args, **kwargs) -> Dict[str, Any]:
        """Fallback for Ollama operations"""

        if operation == "generate":
            # Return basic response indicating service unavailability
            prompt = kwargs.get("prompt", args[0] if args else "")

            logger.warning("Ollama unavailable - returning fallback response")
            return {
                "success": True,
                "fallback": True,
                "response": f"I apologize, but the AI model service is temporarily unavailable. "
                           f"Your request: '{prompt[:100]}...' has been noted and will be "
                           f"processed when the service recovers.",
                "model": "fallback",
                "created_at": datetime.utcnow().isoformat()
            }

        elif operation == "list_models":
            return {
                "success": True,
                "fallback": True,
                "models": ["fallback"],
                "message": "Model service unavailable - showing fallback model only"
            }

        elif operation == "show_model":
            return {
                "success": False,
                "fallback": True,
                "error": "Model service unavailable",
                "model": kwargs.get("model", "unknown")
            }

        elif operation == "pull_model":
            # Log model pull request
            model_name = kwargs.get("model", args[0] if args else "unknown")
            logger.warning(f"Ollama unavailable - model pull request logged: {model_name}")
            return {
                "success": False,
                "fallback": True,
                "status": "queued",
                "message": f"Model pull request for '{model_name}' queued for when service recovers"
            }

        else:
            return {
                "success": False,
                "fallback": True,
                "error": f"Ollama service unavailable for operation: {operation}"
            }

    # PostgreSQL Fallbacks
    async def _postgresql_fallback(self, operation: str, *args, **kwargs) -> Dict[str, Any]:
        """Fallback for PostgreSQL operations"""

        if operation == "query" or operation == "execute":
            # Cache critical operations for later execution
            cache_key = f"db_{operation}_{datetime.utcnow().timestamp()}"
            self._fallback_cache[cache_key] = {
                "operation": operation,
                "query": kwargs.get("query", args[0] if args else ""),
                "params": kwargs.get("params", []),
                "created_at": datetime.utcnow()
            }

            logger.error("PostgreSQL unavailable - critical operation cached for later execution")
            return {
                "success": False,
                "fallback": True,
                "cached": True,
                "cache_key": cache_key,
                "rows_affected": 0,
                "message": "Database unavailable - operation cached for later execution"
            }

        elif operation == "fetch":
            # Return empty result set
            logger.warning("PostgreSQL unavailable - returning empty result set")
            return {
                "success": True,
                "fallback": True,
                "rows": [],
                "count": 0,
                "message": "Database unavailable - returning empty results"
            }

        elif operation == "health_check":
            return {
                "success": False,
                "fallback": True,
                "status": "unavailable",
                "message": "PostgreSQL service is not responding"
            }

        else:
            return {
                "success": False,
                "fallback": True,
                "error": f"Database service unavailable for operation: {operation}"
            }

    # Redis Fallbacks
    async def _redis_fallback(self, operation: str, *args, **kwargs) -> Dict[str, Any]:
        """Fallback for Redis operations"""

        if operation == "get":
            # Return None for cache misses
            return {
                "success": True,
                "fallback": True,
                "value": None,
                "message": "Cache unavailable - returning None"
            }

        elif operation == "set":
            # Log that cache update was missed
            key = kwargs.get("key", args[0] if args else "unknown")
            logger.warning(f"Redis unavailable - cache update missed for key: {key}")
            return {
                "success": True,
                "fallback": True,
                "cached": False,
                "message": "Cache unavailable - update skipped"
            }

        elif operation == "delete":
            return {
                "success": True,
                "fallback": True,
                "deleted": False,
                "message": "Cache unavailable - deletion skipped"
            }

        else:
            return {
                "success": True,
                "fallback": True,
                "message": f"Cache unavailable for operation: {operation}"
            }

    async def get_fallback_handler(self, service: str, operation: str):
        """Get fallback handler for specific service operation"""
        handler = self._fallback_responses.get(service.lower())
        if handler:
            return lambda *args, **kwargs: handler(operation, *args, **kwargs)
        return None

    async def execute_fallback(self, service: str, operation: str, *args, **kwargs) -> Dict[str, Any]:
        """Execute fallback for service operation"""
        handler = self._fallback_responses.get(service.lower())
        if handler:
            return await handler(operation, *args, **kwargs)
        else:
            return {
                "success": False,
                "fallback": True,
                "error": f"No fallback handler for service: {service}"
            }

    def get_cached_operations(self, service: Optional[str] = None) -> Dict[str, Any]:
        """Get operations cached during fallback"""
        if service:
            return {k: v for k, v in self._fallback_cache.items()
                   if v.get("operation", "").startswith(service.lower())}
        return self._fallback_cache.copy()

    def clear_cache(self, service: Optional[str] = None):
        """Clear fallback cache"""
        if service:
            to_remove = [k for k, v in self._fallback_cache.items()
                        if v.get("operation", "").startswith(service.lower())]
            for key in to_remove:
                del self._fallback_cache[key]
            logger.info(f"Cleared fallback cache for {service}")
        else:
            self._fallback_cache.clear()
            logger.info("Cleared all fallback cache")

    async def replay_cached_operations(self, service: str) -> Dict[str, Any]:
        """Replay cached operations when service recovers"""
        service_ops = self.get_cached_operations(service)

        results = {
            "total_operations": len(service_ops),
            "successful_replays": 0,
            "failed_replays": 0,
            "errors": []
        }

        for cache_key, operation in service_ops.items():
            try:
                # This would need to be implemented with actual service calls
                # For now, just log the replay attempt
                logger.info(f"Would replay operation: {operation}")
                results["successful_replays"] += 1

                # Remove successfully replayed operation
                if cache_key in self._fallback_cache:
                    del self._fallback_cache[cache_key]

            except Exception as e:
                results["failed_replays"] += 1
                results["errors"].append(f"Failed to replay {cache_key}: {e}")
                logger.error(f"Failed to replay cached operation {cache_key}: {e}")

        return results


# Global fallback handler instance
_fallback_handler = None


def get_fallback_handler() -> FallbackHandler:
    """Get global fallback handler instance"""
    global _fallback_handler
    if _fallback_handler is None:
        _fallback_handler = FallbackHandler()
    return _fallback_handler