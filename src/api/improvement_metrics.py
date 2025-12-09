#!/usr/bin/env python3
"""
API endpoints for Echo Brain improvement metrics monitoring.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import json
import psycopg2
from datetime import datetime, timedelta
from pathlib import Path
import httpx

router = APIRouter()

@router.get("/improvement/metrics")
async def get_improvement_metrics() -> Dict[str, Any]:
    """Get current improvement system metrics."""
    metrics = {
        "status": "initializing",
        "metrics": {},
        "knowledge_sources": {},
        "improvements": []
    }

    # Load metrics file
    metrics_file = Path("/opt/tower-echo-brain/improvement_metrics.json")
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics["metrics"] = json.load(f)
            metrics["status"] = "running" if metrics["metrics"].get("last_update") else "idle"

    # Check Claude indexing progress
    try:
        log_file = Path("/opt/tower-echo-brain/logs/claude_indexing.log")
        if log_file.exists():
            with open(log_file) as f:
                lines = f.readlines()
                for line in reversed(lines[-20:]):
                    if "Progress:" in line:
                        metrics["claude_indexing"] = line.strip()
                        break
    except:
        pass

    # Database stats
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="echo_brain",
            user="patrick",
            password="tower_echo_brain_secret_key_2025"
        )
        cursor = conn.cursor()

        # Claude conversations indexed
        cursor.execute("""
            SELECT COUNT(*) FROM echo_unified_interactions
            WHERE model_used = 'claude'
        """)
        metrics["knowledge_sources"]["claude_conversations"] = cursor.fetchone()[0]

        # Recent improvements
        cursor.execute("""
            SELECT query, COUNT(*) as frequency
            FROM echo_unified_interactions
            WHERE timestamp > NOW() - INTERVAL '1 day'
            GROUP BY query
            HAVING COUNT(*) > 2
            ORDER BY COUNT(*) DESC
            LIMIT 5
        """)
        patterns = cursor.fetchall()
        metrics["repeated_queries"] = [
            {"query": q, "frequency": f} for q, f in patterns
        ]

        # Error rate
        cursor.execute("""
            SELECT
                COUNT(CASE WHEN response LIKE '%error%' THEN 1 END)::float /
                NULLIF(COUNT(*), 0) as error_rate,
                AVG(processing_time) as avg_processing_time
            FROM echo_unified_interactions
            WHERE timestamp > NOW() - INTERVAL '1 hour'
        """)
        error_rate, avg_time = cursor.fetchone()
        metrics["performance"] = {
            "error_rate": error_rate or 0.0,
            "avg_response_time": avg_time or 0.0
        }

        conn.close()
    except Exception as e:
        metrics["error"] = str(e)

    # Qdrant stats
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:6333/collections")
            collections = resp.json()["result"]["collections"]

            for col in collections:
                if "claude" in col["name"] or "learning" in col["name"]:
                    col_info = await client.get(f"http://localhost:6333/collections/{col['name']}")
                    info = col_info.json()["result"]
                    metrics["knowledge_sources"][col["name"]] = info["points_count"]
    except:
        pass

    return metrics

@router.get("/improvement/status")
async def get_improvement_status() -> Dict[str, Any]:
    """Get improvement system status."""
    status = {
        "learning_active": False,
        "spatial_reasoning": False,
        "claude_memory": False,
        "continuous_improvement": False,
        "models_available": []
    }

    # Check if improvement service is running
    import subprocess
    result = subprocess.run(
        ["systemctl", "is-active", "echo-improvement"],
        capture_output=True,
        text=True
    )
    status["continuous_improvement"] = result.stdout.strip() == "active"

    # Check available models
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')[1:]  # Skip header
        for line in lines:
            if line:
                parts = line.split()
                if parts:
                    model_name = parts[0]
                    if any(x in model_name for x in ["qwen3-vl", "llava", "vision"]):
                        status["models_available"].append(model_name)
                        status["spatial_reasoning"] = True

    # Check Claude memory collection
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:6333/collections/claude_conversations")
            if resp.status_code == 200:
                info = resp.json()["result"]
                status["claude_memory"] = info["points_count"] > 0
    except:
        pass

    return status

@router.post("/improvement/trigger")
async def trigger_improvement_cycle() -> Dict[str, str]:
    """Manually trigger an improvement cycle."""
    # TODO: Implement manual trigger
    return {"status": "Improvement cycle triggered"}

@router.get("/improvement/knowledge-graph")
async def get_knowledge_graph() -> Dict[str, Any]:
    """Get current knowledge graph structure."""
    graph = {
        "nodes": [],
        "edges": [],
        "stats": {}
    }

    # This would connect to the spatial reasoning system
    try:
        from src.improvement.spatial_reasoning import SpatialCodeUnderstanding
        spatial = SpatialCodeUnderstanding()
        spatial.build_codebase_graph(["/opt/tower-echo-brain"])

        graph["stats"] = {
            "total_nodes": spatial.graph.number_of_nodes(),
            "total_edges": spatial.graph.number_of_edges(),
            "services": len(spatial.service_map)
        }

        # Get top connected nodes
        import networkx as nx
        centrality = nx.degree_centrality(spatial.graph)
        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:20]

        for node, score in top_nodes:
            graph["nodes"].append({
                "id": node,
                "centrality": score,
                "type": "file" if ".py" in str(node) else "module"
            })

    except Exception as e:
        graph["error"] = str(e)

    return graph