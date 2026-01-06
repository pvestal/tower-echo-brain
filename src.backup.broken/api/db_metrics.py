#!/usr/bin/env python3
"""
Database Metrics and Connection Pool Health API Endpoints
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any, Optional
import logging
import time

from ..db.pool_manager import get_pool, health_check, get_query_stats
from ..db.async_database import get_async_database
from ..db.query_optimizer import get_index_recommendations

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/db", tags=["Database Metrics"])

@router.get("/health")
async def get_database_health():
    """
    Get comprehensive database and connection pool health status
    """
    try:
        health = await health_check()
        return {
            "status": "success",
            "timestamp": time.time(),
            "pool_health": health
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/pool/status")
async def get_pool_status():
    """
    Get current connection pool status and metrics
    """
    try:
        pool = await get_pool()
        health = await pool.health_check()

        # Extract detailed metrics if available
        metrics = health.get("metrics", {})

        return {
            "status": "success",
            "pool_status": health["status"],
            "connection_metrics": {
                "total_connections": metrics.get("total_connections", 0),
                "active_connections": metrics.get("active_connections", 0),
                "idle_connections": metrics.get("idle_connections", 0),
                "pool_utilization": (
                    metrics.get("active_connections", 0) / max(1, metrics.get("total_connections", 1))
                ) * 100
            },
            "performance_metrics": {
                "total_queries": metrics.get("total_queries", 0),
                "queries_per_second": metrics.get("queries_per_second", 0.0),
                "avg_query_time": metrics.get("avg_query_time", 0.0),
                "slow_queries": metrics.get("slow_queries", 0),
                "failed_queries": metrics.get("failed_queries", 0),
                "pool_exhausted": metrics.get("pool_exhausted", 0),
                "cache_hit_ratio": metrics.get("cache_hit_ratio", 0.0)
            },
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Failed to get pool status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get pool status: {str(e)}")

@router.get("/queries/slow")
async def get_slow_queries(
    limit: int = 20,
    min_execution_time: float = 1.0
):
    """
    Get slowest queries for performance analysis
    """
    try:
        db = await get_async_database()
        slow_queries = await db.get_slow_queries(limit, min_execution_time)

        return {
            "status": "success",
            "slow_queries": slow_queries,
            "analysis": {
                "total_slow_queries": len(slow_queries),
                "threshold_time": min_execution_time,
                "worst_query_time": max([q.get("avg_execution_time", 0) for q in slow_queries], default=0)
            },
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Failed to get slow queries: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get slow queries: {str(e)}")

@router.get("/queries/stats")
async def get_query_statistics(limit: int = 20):
    """
    Get query performance statistics from the connection pool
    """
    try:
        stats = await get_query_stats(limit)

        return {
            "status": "success",
            "query_stats": stats,
            "summary": {
                "total_tracked_queries": len(stats),
                "queries_analyzed": limit
            },
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Failed to get query stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get query stats: {str(e)}")

@router.get("/optimization/recommendations")
async def get_optimization_recommendations():
    """
    Get database optimization and index recommendations
    """
    try:
        db = await get_async_database()

        # Get slow queries for analysis
        slow_queries = await db.get_slow_queries(50, 0.5)

        # Generate index recommendations
        index_recommendations = await get_index_recommendations(slow_queries)

        # Get query stats for additional analysis
        query_stats = await get_query_stats(50)

        # Analyze patterns
        total_queries = sum(stat.get('count', 0) for stat in query_stats)
        avg_execution_time = sum(stat.get('avg_time', 0) for stat in query_stats) / max(1, len(query_stats))

        recommendations = {
            "index_recommendations": index_recommendations,
            "performance_insights": {
                "queries_needing_optimization": len(slow_queries),
                "average_query_time": avg_execution_time,
                "total_tracked_queries": total_queries
            },
            "general_recommendations": []
        }

        # Add general recommendations based on analysis
        if len(slow_queries) > 10:
            recommendations["general_recommendations"].append({
                "priority": "high",
                "recommendation": "High number of slow queries detected - review query patterns and consider adding indexes",
                "impact": "performance"
            })

        if avg_execution_time > 0.5:
            recommendations["general_recommendations"].append({
                "priority": "medium",
                "recommendation": f"Average query time is {avg_execution_time:.3f}s - consider query optimization",
                "impact": "performance"
            })

        return {
            "status": "success",
            "recommendations": recommendations,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Failed to get optimization recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")

@router.get("/analytics/interactions")
async def get_interaction_analytics(
    user_id: Optional[str] = None,
    hours_back: int = 24
):
    """
    Get interaction analytics and performance insights
    """
    try:
        db = await get_async_database()
        analytics = await db.get_interaction_analytics(user_id, hours_back)

        return {
            "status": "success",
            "analytics": analytics,
            "filters": {
                "user_id": user_id,
                "hours_back": hours_back
            },
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Failed to get interaction analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

@router.post("/maintenance/optimize")
async def run_database_optimization():
    """
    Run database optimization and maintenance tasks
    """
    try:
        db = await get_async_database()

        start_time = time.time()

        # Run optimization
        await db.optimize_tables()

        # Get updated health
        health = await db.get_pool_health()

        optimization_time = time.time() - start_time

        return {
            "status": "success",
            "optimization_completed": True,
            "optimization_time": optimization_time,
            "post_optimization_health": health,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Database optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@router.post("/maintenance/cleanup")
async def cleanup_old_data(days_to_keep: int = 90):
    """
    Clean up old database data to maintain performance
    """
    try:
        if days_to_keep < 7:
            raise HTTPException(status_code=400, detail="Cannot clean data newer than 7 days")

        db = await get_async_database()
        await db.cleanup_old_data(days_to_keep)

        return {
            "status": "success",
            "cleanup_completed": True,
            "days_kept": days_to_keep,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Data cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@router.post("/cache/invalidate")
async def invalidate_query_cache(pattern: Optional[str] = None):
    """
    Invalidate query cache entries
    """
    try:
        pool = await get_pool()
        await pool.invalidate_cache(pattern)

        return {
            "status": "success",
            "cache_invalidated": True,
            "pattern": pattern or "all",
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Cache invalidation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cache invalidation failed: {str(e)}")

@router.get("/dashboard")
async def get_database_dashboard():
    """
    Get comprehensive database dashboard data
    """
    try:
        # Gather all dashboard data
        health = await health_check()
        pool = await get_pool()
        pool_health = await pool.health_check()
        db = await get_async_database()

        # Performance data
        slow_queries = await db.get_slow_queries(10, 0.5)
        analytics = await db.get_interaction_analytics(hours_back=24)
        query_stats = await get_query_stats(10)

        # Calculate dashboard metrics
        metrics = pool_health.get("metrics", {})

        dashboard = {
            "overall_status": health["status"],
            "pool_status": {
                "status": pool_health["status"],
                "total_connections": metrics.get("total_connections", 0),
                "active_connections": metrics.get("active_connections", 0),
                "idle_connections": metrics.get("idle_connections", 0),
                "utilization_percent": round(
                    (metrics.get("active_connections", 0) / max(1, metrics.get("total_connections", 1))) * 100,
                    2
                )
            },
            "performance": {
                "total_queries": metrics.get("total_queries", 0),
                "queries_per_second": round(metrics.get("queries_per_second", 0.0), 2),
                "avg_query_time": round(metrics.get("avg_query_time", 0.0), 3),
                "slow_queries_count": len(slow_queries),
                "failed_queries": metrics.get("failed_queries", 0),
                "cache_hit_ratio": round(metrics.get("cache_hit_ratio", 0.0) * 100, 1)
            },
            "interactions": {
                "total_24h": analytics.get("total_interactions", 0),
                "unique_conversations_24h": analytics.get("unique_conversations", 0),
                "avg_processing_time": round(analytics.get("avg_processing_time", 0.0), 3),
                "models_used": analytics.get("models_used", 0)
            },
            "top_slow_queries": slow_queries[:5],
            "timestamp": time.time()
        }

        return {
            "status": "success",
            "dashboard": dashboard
        }

    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard generation failed: {str(e)}")