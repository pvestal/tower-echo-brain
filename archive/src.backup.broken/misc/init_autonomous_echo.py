#!/usr/bin/env python3
"""
Initialize Echo's Autonomous Task System with Seed Tasks
"""

import asyncio
import logging
import requests
import json
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def add_seed_tasks():
    """Add initial seed tasks to get Echo started"""
    
    base_url = "http://localhost:8309"
    
    # Wait for Echo to be ready
    logger.info("⏳ Waiting for Echo service to be ready...")
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{base_url}/api/echo/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Echo service is ready")
                break
        except:
            pass
        
        if attempt < max_attempts - 1:
            await asyncio.sleep(2)
        else:
            logger.error("❌ Echo service failed to start")
            return False
    
    # Check if task system is ready
    try:
        response = requests.get(f"{base_url}/api/tasks/status", timeout=10)
        if response.status_code == 200:
            logger.info("✅ Task system is ready")
        else:
            logger.warning("⚠️ Task system may not be fully ready")
    except Exception as e:
        logger.warning(f"⚠️ Task system status check failed: {e}")
    
    # Seed tasks to initialize Echo's autonomous behavior
    seed_tasks = [
        {
            "name": "Initial System Health Check",
            "task_type": "monitoring",
            "priority": 2,  # HIGH
            "payload": {
                "target": "all_services",
                "check_type": "service_health",
                "description": "Comprehensive health check of all Tower services"
            }
        },
        {
            "name": "System Resource Baseline",
            "task_type": "monitoring",
            "priority": 3,  # NORMAL
            "payload": {
                "target": "system",
                "check_type": "system_resources",
                "description": "Establish baseline system resource usage"
            }
        },
        {
            "name": "Zombie Process Cleanup",
            "task_type": "maintenance",
            "priority": 2,  # HIGH
            "payload": {
                "action": "cleanup",
                "target": "zombie_processes",
                "description": "Clean up zombie processes on Tower"
            }
        },
        {
            "name": "Initial Memory Optimization",
            "task_type": "optimization",
            "priority": 3,  # NORMAL
            "payload": {
                "system": "memory",
                "metric": "usage",
                "description": "Initial memory optimization and cleanup"
            }
        },
        {
            "name": "Disk Usage Analysis",
            "task_type": "monitoring",
            "priority": 3,  # NORMAL
            "payload": {
                "target": "disk",
                "check_type": "disk_usage",
                "description": "Analyze disk usage across all mount points"
            }
        },
        {
            "name": "Log Pattern Learning - Errors",
            "task_type": "learning",
            "priority": 4,  # LOW
            "payload": {
                "data_source": "system_logs",
                "pattern": "error",
                "description": "Learn from error patterns in system logs"
            }
        },
        {
            "name": "Service Performance Baseline",
            "task_type": "analysis",
            "priority": 3,  # NORMAL
            "payload": {
                "analysis_type": "performance_baseline",
                "targets": ["echo", "anime", "knowledge_base", "comfyui"],
                "description": "Establish performance baselines for all services"
            }
        }
    ]
    
    added_tasks = 0
    for task_data in seed_tasks:
        try:
            response = requests.post(
                f"{base_url}/api/tasks/add",
                json=task_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Added seed task: {task_data['name']} (ID: {result.get('task_id', 'unknown')})")
                added_tasks += 1
            else:
                logger.error(f"❌ Failed to add seed task '{task_data['name']}': {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Error adding seed task '{task_data['name']}': {e}")
    
    logger.info(f"🌱 Added {added_tasks}/{len(seed_tasks)} seed tasks")
    
    # Check brain state
    try:
        response = requests.get(f"{base_url}/api/tasks/brain-state", timeout=5)
        if response.status_code == 200:
            brain_state = response.json()
            logger.info(f"🧠 Echo brain state: {brain_state.get('brain_state')} - {brain_state.get('activity_detail', '')}")
        else:
            logger.warning("⚠️ Could not get brain state")
    except Exception as e:
        logger.warning(f"⚠️ Error getting brain state: {e}")
    
    return added_tasks > 0

if __name__ == "__main__":
    asyncio.run(add_seed_tasks())
