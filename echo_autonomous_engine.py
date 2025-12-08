#!/usr/bin/env python3
"""
Echo Autonomous Engine - Self-directed learning and task execution
"""
from echo_allhands_integration import AllHandsIntegration
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import asyncpg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    DEFERRED = "deferred"


@dataclass
class AutonomousTask:
    id: str
    name: str
    description: str
    priority: TaskPriority
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


class EchoAutonomousEngine:
    def __init__(self):
        self.db_pool = None
        self.running = False
        self.current_tasks = []
        self.learning_goals = [
            "Improve response accuracy",
            "Optimize service integration",
            "Learn from user interactions",
            "Enhance temporal reasoning",
            "Develop new capabilities",
        ]
        self.stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "learning_cycles": 0,
            "improvements_made": 0,
            "patterns_discovered": 0,
        }

    async def connect_db(self):
        """Connect to PostgreSQL database"""
        try:
            self.db_pool = await asyncpg.create_pool(
                host="localhost",
                database="echo_brain",
                user="echo_user",
                password="echo_secure_2025",
                min_size=1,
                max_size=10,
            )

            # Create tables
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS autonomous_tasks (
                        id SERIAL PRIMARY KEY,
                        task_id TEXT UNIQUE,
                        name TEXT,
                        description TEXT,
                        priority INTEGER,
                        status TEXT,
                        created_at TIMESTAMP,
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        result JSONB,
                        error TEXT,
                        retry_count INTEGER DEFAULT 0
                    )
                """
                )

                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS learning_patterns (
                        id SERIAL PRIMARY KEY,
                        pattern_type TEXT,
                        pattern_data JSONB,
                        confidence FLOAT,
                        discovered_at TIMESTAMP,
                        applied_count INTEGER DEFAULT 0
                    )
                """
                )

                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS self_improvements (
                        id SERIAL PRIMARY KEY,
                        improvement_type TEXT,
                        description TEXT,
                        before_metrics JSONB,
                        after_metrics JSONB,
                        implemented_at TIMESTAMP
                    )
                """
                )

            logger.info("✅ Database connected and tables created")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False

    async def generate_autonomous_tasks(self):
        """Generate tasks based on current goals and system state"""
        tasks = []

        # Knowledge Base Learning
        tasks.append(
            AutonomousTask(
                id=f"kb_scan_{int(time.time())}",
                name="Knowledge Base Analysis",
                description="Scan KB for patterns and learning opportunities",
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.PENDING,
                created_at=datetime.now(),
            )
        )

        # Service Health Monitoring
        tasks.append(
            AutonomousTask(
                id=f"health_check_{int(time.time())}",
                name="Service Health Check",
                description="Monitor all Tower services and optimize integration",
                priority=TaskPriority.HIGH,
                status=TaskStatus.PENDING,
                created_at=datetime.now(),
            )
        )

        # Self Performance Analysis
        tasks.append(
            AutonomousTask(
                id=f"self_analysis_{int(time.time())}",
                name="Self Performance Analysis",
                description="Analyze own response patterns and identify improvements",
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.PENDING,
                created_at=datetime.now(),
            )
        )

        # Creative Content Generation
        tasks.append(
            AutonomousTask(
                id=f"creative_gen_{int(time.time())}",
                name="Creative Content Exploration",
                description="Generate creative content to expand capabilities",
                priority=TaskPriority.LOW,
                status=TaskStatus.PENDING,
                created_at=datetime.now(),
            )
        )

        return tasks

    async def execute_task(self, task: AutonomousTask):
        """Execute a single autonomous task"""
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()

        try:
            if "Knowledge Base" in task.name:
                result = await self.analyze_knowledge_base()
            elif "Health Check" in task.name:
                result = await self.check_service_health()
            elif "Performance Analysis" in task.name:
                result = await self.analyze_self_performance()
            elif "Creative" in task.name:
                result = await self.generate_creative_content()
            else:
                result = {"status": "unknown_task"}

            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            self.stats["tasks_completed"] += 1

            # Store in database
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO autonomous_tasks 
                        (task_id, name, description, priority, status, created_at, completed_at, result)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        ON CONFLICT (task_id) DO UPDATE SET
                        status = $5, completed_at = $7, result = $8
                    """,
                        task.id,
                        task.name,
                        task.description,
                        task.priority.value,
                        task.status.value,
                        task.created_at,
                        task.completed_at,
                        json.dumps(result),
                    )

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.retry_count += 1
            self.stats["tasks_failed"] += 1
            logger.error(f"Task {task.name} failed: {e}")

            if task.retry_count < task.max_retries:
                task.status = TaskStatus.PENDING  # Retry later

    async def analyze_knowledge_base(self):
        """Analyze KB articles for learning patterns"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8307/api/articles") as resp:
                    articles = await resp.json()

                patterns = {
                    "total_articles": len(articles),
                    "categories": {},
                    "recent_topics": [],
                    "learning_opportunities": [],
                }

                for article in articles[-10:]:  # Analyze recent articles
                    category = article.get("category", "uncategorized")
                    patterns["categories"][category] = (
                        patterns["categories"].get(category, 0) + 1
                    )
                    patterns["recent_topics"].append(article.get("title", ""))

                    # Look for learning patterns
                    if "error" in article.get("content", "").lower():
                        patterns["learning_opportunities"].append(
                            {
                                "type": "error_analysis",
                                "article_id": article.get("id"),
                                "title": article.get("title"),
                            }
                        )

                self.stats["patterns_discovered"] += len(
                    patterns["learning_opportunities"]
                )
                return patterns
        except Exception as e:
            return {"error": str(e)}

    async def check_service_health(self):
        """Monitor Tower services health"""
        services = {
            "echo_brain": "http://localhost:8309/api/echo/health",
            "knowledge_base": "http://localhost:8307/api/health",
            "comfyui": "http://localhost:8188/api/health",
            "anime_service": "http://localhost:8328/api/anime/health",
            "dashboard": "http://localhost:8080/api/health",
            "vault": "http://localhost:8200/v1/sys/health",
        }

        health_status = {}
        async with aiohttp.ClientSession() as session:
            for name, url in services.items():
                try:
                    async with session.get(
                        url, timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        health_status[name] = {
                            "status": "healthy" if resp.status == 200 else "degraded",
                            "response_time": resp.headers.get(
                                "X-Response-Time", "unknown"
                            ),
                        }
                except Exception as e:
                    health_status[name] = {
                        "status": "unhealthy", "error": str(e)}

        # Identify optimization opportunities
        optimizations = []
        for name, status in health_status.items():
            if status["status"] != "healthy":
                optimizations.append(f"Service {name} needs attention")

        if optimizations:
            self.stats["improvements_made"] += 1

        return {"services": health_status, "optimizations": optimizations}

    async def analyze_self_performance(self):
        """Analyze Echo's own performance metrics"""
        metrics = {
            "response_patterns": {},
            "error_rate": 0,
            "avg_response_time": 0,
            "capability_usage": {},
            "improvement_areas": [],
        }

        # Query recent interactions from database
        if self.db_pool:
            async with self.db_pool.acquire() as conn:
                # Get task completion rate
                completed = await conn.fetchval(
                    "SELECT COUNT(*) FROM autonomous_tasks WHERE status = 'completed'"
                )
                failed = await conn.fetchval(
                    "SELECT COUNT(*) FROM autonomous_tasks WHERE status = 'failed'"
                )

                if completed + failed > 0:
                    metrics["success_rate"] = completed / (completed + failed)
                    metrics["error_rate"] = failed / (completed + failed)

                # Identify areas for improvement
                if metrics["error_rate"] > 0.1:
                    metrics["improvement_areas"].append(
                        "Reduce error rate below 10%")

                self.stats["learning_cycles"] += 1

        return metrics

    async def generate_creative_content(self):
        """Generate creative content to expand capabilities"""
        creative_tasks = [
            "Write a haiku about Tower's services",
            "Generate a system optimization plan",
            "Create a new integration idea",
            "Design a learning experiment",
        ]

        import random

        task = random.choice(creative_tasks)

        # Use Echo's own capabilities
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:8309/api/echo/chat",
                    json={"message": task, "context": "autonomous_creativity"},
                ) as resp:
                    result = await resp.json()
                    return {
                        "creative_task": task,
                        "generated_content": result.get("response", ""),
                        "timestamp": datetime.now().isoformat(),
                    }
        except Exception as e:
            return {"error": str(e), "task": task}

    async def run_autonomous_loop(self):
        """Main autonomous operation loop"""
        self.running = True
        logger.info("🤖 Echo Autonomous Engine starting...")

        while self.running:
            try:
                # Generate new tasks
                new_tasks = await self.generate_autonomous_tasks()

                # Execute tasks based on priority
                for task in sorted(new_tasks, key=lambda t: t.priority.value):
                    if not self.running:
                        break

                    logger.info(f"📋 Executing: {task.name}")
                    await self.execute_task(task)

                    # Brief pause between tasks
                    await asyncio.sleep(2)

                # Log stats
                logger.info(f"📊 Stats: {self.stats}")

                # Save stats to file
                with open("/opt/tower-echo-brain/autonomous_stats.json", "w") as f:
                    json.dump(self.stats, f, indent=2)

                # Wait before next cycle (5 minutes)
                await asyncio.sleep(300)

            except Exception as e:
                logger.error(f"Autonomous loop error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

    async def start(self):
        """Start the autonomous engine"""
        if await self.connect_db():
            await self.run_autonomous_loop()
        else:
            logger.error("Failed to start: Database connection required")

    def stop(self):
        """Stop the autonomous engine"""
        self.running = False
        logger.info("🛑 Autonomous engine stopping...")


async def main():
    engine = EchoAutonomousEngine()
    try:
        await engine.start()
    except KeyboardInterrupt:
        engine.stop()
        if engine.db_pool:
            await engine.db_pool.close()


if __name__ == "__main__":
    asyncio.run(main())

# Add AllHands integration


class EchoAutonomousEngineWithBoard(EchoAutonomousEngine):
    """Enhanced autonomous engine with Board of Directors integration"""

    def __init__(self):
        super().__init__()
        self.allhands = AllHandsIntegration()
        self.board_approved_tasks = []

    async def execute_task(self, task: AutonomousTask):
        """Execute task with board approval"""
        # Submit to board for approval
        board_result = await self.allhands.submit_to_board(
            {
                "name": task.name,
                "description": task.description,
                "priority": task.priority.value,
                "impact": "medium" if task.priority.value <= 2 else "low",
            }
        )

        if board_result.get("status") == "approved":
            # Execute with board approval
            await super().execute_task(task)

            # Log board decision
            self.board_approved_tasks.append(
                {
                    "task_id": task.id,
                    "board_decision": board_result,
                    "executed_at": datetime.now().isoformat(),
                }
            )
        else:
            logger.info(
                f"Task {task.name} not approved by board: {board_result}")
            task.status = TaskStatus.DEFERRED

    async def run_allhands_meeting(self):
        """Run periodic all-hands meetings for strategic decisions"""
        topics = [
            "What are the highest priority improvements for Echo?",
            "How can we better serve Patrick's needs?",
            "What new capabilities should Echo develop?",
            "How can Tower services be optimized?",
        ]

        import random

        topic = random.choice(topics)

        meeting = await self.allhands.coordinate_allhands_meeting(
            topic=topic, participants=["AI Assist", "Mistral", "Claude"]
        )

        # Save meeting results
        with open("/opt/tower-echo-brain/allhands_meetings.json", "a") as f:
            json.dump(meeting, f)
            f.write("\n")

        return meeting
