#!/usr/bin/env python3
"""
Seed learning goals for Phase 2 autonomy
"""

import httpx
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def seed_goals():
    """Register Phase 2 learning goals"""

    goals = [
        {
            "name": "continuous_fact_extraction",
            "description": "Extract facts from unprocessed vectors in echo_memory",
            "safety_level": "auto",
            "schedule": "*/30 * * * *",  # Every 30 minutes
            "enabled": True,
            "metadata": {
                "worker": "fact_extraction_worker",
                "batch_size": 50,
                "model": "gemma2:9b"
            }
        },
        {
            "name": "conversation_watcher",
            "description": "Monitor and ingest new conversations into memory",
            "safety_level": "notify",
            "schedule": "*/10 * * * *",  # Every 10 minutes
            "enabled": True,
            "metadata": {
                "worker": "conversation_watcher",
                "watch_dirs": [
                    "/opt/tower-echo-brain/conversations",
                    "/home/patrick/.claude/projects"
                ]
            }
        },
        {
            "name": "knowledge_graph_build",
            "description": "Build knowledge graph relationships between facts",
            "safety_level": "auto",
            "schedule": "0 2 * * *",  # Daily at 2 AM
            "enabled": True,
            "metadata": {
                "worker": "knowledge_graph_builder",
                "relationship_types": [
                    "same_subject",
                    "object_to_subject",
                    "co_located",
                    "semantic"
                ]
            }
        }
    ]

    async with httpx.AsyncClient() as client:
        for goal in goals:
            try:
                # Check if goal already exists
                check_response = await client.get(
                    f"http://localhost:8309/api/autonomous/goals",
                    params={"name": goal["name"]}
                )

                if check_response.status_code == 200:
                    existing = check_response.json()
                    if any(g["name"] == goal["name"] for g in existing):
                        logger.info(f"Goal '{goal['name']}' already exists, updating...")
                        # Update existing goal
                        update_response = await client.put(
                            f"http://localhost:8309/api/autonomous/goals/{goal['name']}",
                            json=goal
                        )
                        logger.info(f"Updated: {goal['name']} - Status: {update_response.status_code}")
                        continue

                # Create new goal
                response = await client.post(
                    "http://localhost:8309/api/autonomous/goals",
                    json=goal
                )

                if response.status_code in [200, 201]:
                    logger.info(f"✅ Registered: {goal['name']}")
                else:
                    logger.error(f"❌ Failed to register {goal['name']}: {response.status_code}")
                    logger.error(response.text)

            except Exception as e:
                logger.error(f"❌ Error registering {goal['name']}: {e}")

    # Verify all goals are registered
    try:
        verify_response = await client.get("http://localhost:8309/api/autonomous/goals")
        if verify_response.status_code == 200:
            registered = verify_response.json()
            logger.info(f"\nRegistered goals: {len(registered)}")
            for g in registered:
                if g['name'] in [goal['name'] for goal in goals]:
                    logger.info(f"  - {g['name']}: {g['safety_level']} ({g['schedule']})")
    except Exception as e:
        logger.error(f"Error verifying goals: {e}")

if __name__ == "__main__":
    asyncio.run(seed_goals())