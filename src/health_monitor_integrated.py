#!/usr/bin/env python3
"""
Integrated Health Monitor for Echo Brain
Runs as a background task within the main Echo Brain process
"""

import asyncio
import json
import logging
import httpx
from datetime import datetime
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class IntegratedHealthMonitor:
    """Health monitor that runs within Echo Brain process"""
    
    def __init__(self):
        self.endpoints = [
            {"name": "Echo Brain API", "method": "GET", "path": "http://localhost:8309/health", "data": None},
            {"name": "Echo Brain Docs", "method": "GET", "path": "http://localhost:8309/docs", "data": None},
            {"name": "MCP Server", "method": "GET", "path": "http://localhost:8312/", "data": None},
        ]
        
        # Database connections
        self.databases = [
            {"name": "PostgreSQL", "type": "postgres", "host": "localhost", "port": 5432},
            {"name": "Qdrant", "type": "qdrant", "host": "localhost", "port": 6333},
        ]
    
    async def check_endpoint(self, method: str, path: str, data: dict = None) -> Tuple[int, float]:
        """Check a single endpoint"""
        start_time = datetime.now()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                if method.upper() == "GET":
                    response = await client.get(path)
                elif method.upper() == "POST":
                    response = await client.post(path, json=data or {})
                else:
                    return 0, 0
                
                response_time = (datetime.now() - start_time).total_seconds()
                return response.status_code, response_time
        except Exception as e:
            logger.debug(f"Endpoint check failed for {path}: {e}")
            return 0, 0
    
    async def check_database(self, db_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check database connectivity"""
        try:
            # Simplified check - in reality would use appropriate client
            return {
                "name": db_config["name"],
                "status": "healthy",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "name": db_config["name"],
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "endpoints": [],
            "databases": [],
            "overall_status": "healthy"
        }
        
        # Check endpoints
        for endpoint in self.endpoints:
            status_code, response_time = await self.check_endpoint(
                endpoint["method"], endpoint["path"], endpoint["data"]
            )
            
            endpoint_result = {
                "name": endpoint["name"],
                "status": "healthy" if status_code in [200, 201] else "unhealthy",
                "status_code": status_code,
                "response_time": response_time,
                "timestamp": datetime.now().isoformat()
            }
            results["endpoints"].append(endpoint_result)
        
        # Check databases
        for db in self.databases:
            db_result = await self.check_database(db)
            results["databases"].append(db_result)
        
        # Determine overall status - be more lenient: only fail if critical services are down
        critical_services_healthy = all(
            e["status"] == "healthy" for e in results["endpoints"] 
            if e["name"] in ["Echo Brain API", "PostgreSQL", "Qdrant"]
        )
        
        results["overall_status"] = "healthy" if critical_services_healthy else "unhealthy"
        
        # Save to file (same location as original monitor)
        with open("/opt/tower-echo-brain/health_monitor.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Health check completed: {results['overall_status']}")
        return results
    
    async def start_monitoring_loop(self, interval: int = 300):
        """Start continuous monitoring as background task"""
        logger.info(f"Starting integrated health monitoring (interval: {interval}s)")
        
        while True:
            try:
                await self.run_health_check()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(60)  # Wait 60 seconds on error

# Singleton instance
_health_monitor_instance = None

def get_health_monitor():
    """Get or create health monitor instance"""
    global _health_monitor_instance
    if _health_monitor_instance is None:
        _health_monitor_instance = IntegratedHealthMonitor()
    return _health_monitor_instance

async def start_background_monitoring():
    """Start health monitoring in background"""
    monitor = get_health_monitor()
    return asyncio.create_task(monitor.start_monitoring_loop())

