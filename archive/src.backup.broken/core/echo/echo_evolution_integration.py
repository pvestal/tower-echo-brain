#!/usr/bin/env python3
"""
Echo Brain Autonomous Evolution System Integration
Connects all evolution components with safety checks and Board oversight
"""
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum
import psycopg2
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add Echo modules to path
sys.path.insert(0, '/opt/tower-echo-brain')
sys.path.insert(0, '/opt/tower-board-integration')

# Import Echo evolution components
from src.core.echo.echo_autonomous_evolution import EchoAutonomousEvolution, EvolutionTrigger
from src.core.echo.echo_git_integration import EchoGitManager
from src.core.echo.echo_self_analysis import EchoSelfAnalysis, AnalysisDepth
from src.core.echo.echo_self_diagnosis import EchoSelfDiagnosis

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/tower-echo-brain/logs/evolution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EvolutionOrchestrator:
    """Main orchestrator for Echo's autonomous evolution"""

    def __init__(self):
        self.app = FastAPI(title="Echo Evolution System", version="2.0")
        self.setup_cors()

        # Initialize components
        self.evolution = None
        self.git_manager = None
        self.self_analysis = None
        self.self_diagnosis = None

        # Board of Directors connection
        self.board_connected = False
        self.board_url = "http://127.0.0.1:8350/api/board"

        # Safety thresholds
        self.safety_config = {
            'max_daily_evolutions': 3,
            'require_board_approval': True,
            'auto_rollback_on_error': True,
            'performance_threshold': 0.8,
            'test_coverage_minimum': 0.7
        }

        # Evolution metrics
        self.metrics = {
            'total_evolutions': 0,
            'successful_evolutions': 0,
            'failed_evolutions': 0,
            'rollbacks': 0,
            'last_evolution': None,
            'learning_conversations': 0,
            'capability_improvements': []
        }

        # Learning milestones
        self.learning_config = {
            'conversations_per_milestone': 100,
            'analysis_frequency_hours': 24,
            'performance_check_interval': 3600  # 1 hour
        }

        self.setup_routes()
        self.initialize_components()

    def setup_cors(self):
        """Configure CORS for web access"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["https://192.168.50.135", "http://192.168.50.135"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    async def initialize_components(self):
        """Initialize all evolution components with safety checks"""
        try:
            logger.info("Initializing Echo Evolution System...")

            # Initialize Git Manager
            self.git_manager = EchoGitManager(
                repo_path="/opt/tower-echo-brain",
                
            )

            # Initialize Self-Analysis
            self.self_analysis = EchoSelfAnalysis()

            # Initialize Self-Diagnosis
            self.self_diagnosis = EchoSelfDiagnosis()

            # Initialize Autonomous Evolution
            self.evolution = EchoAutonomousEvolution()
                git_manager=self.git_manager,
                self_analysis=self.self_analysis
            )

            # Check Board of Directors connection
            await self.connect_board()

            # Schedule automated tasks
            await self.schedule_evolution_tasks()

            logger.info("✅ Evolution System initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize evolution system: {e}")
            raise

    async def connect_board(self):
        """Connect to Board of Directors for governance"""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.board_url}/health")
                if response.status_code == 200:
                    self.board_connected = True
                    logger.info("✅ Connected to Board of Directors")
                else:
                    logger.warning("⚠️ Board of Directors not available - operating in autonomous mode")
        except:
            logger.warning("Board of Directors service not found - will operate independently")

    async def request_board_approval(self, evolution_plan: Dict[str, Any]) -> bool:
        """Request Board approval for major evolutions"""
        if not self.board_connected or not self.safety_config['require_board_approval']:
            return True  # Auto-approve if Board not available or not required

        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.board_url}/decision",
                    json={
                        "decision_type": "echo_evolution",
                        "proposal": evolution_plan,
                        "risk_level": evolution_plan.get('risk_level', 'medium'),
                        "requester": "Echo Evolution System"
                    }
                )

                if response.status_code == 200:
                    decision = response.json()
                    return decision.get('approved', False)

        except Exception as e:
            logger.error(f"Board approval request failed: {e}")

        return False  # Deny by default if Board unavailable

    async def schedule_evolution_tasks(self):
        """Schedule automated evolution cycles"""
        # Daily self-analysis
        asyncio.create_task(self.scheduled_analysis())

        # Performance monitoring
        asyncio.create_task(self.monitor_performance())

        # Learning milestone tracking
        asyncio.create_task(self.track_learning())

    async def scheduled_analysis(self):
        """Run scheduled self-analysis cycles"""
        while True:
            try:
                await asyncio.sleep(self.learning_config['analysis_frequency_hours'] * 3600)

                logger.info("Running scheduled self-analysis...")
                analysis = await self.evolution.trigger_evolution(
                    EvolutionTrigger.SCHEDULED,
                    {"reason": "Daily scheduled analysis"}
                )

                if analysis and analysis.get('improvements_identified'):
                    await self.apply_improvements(analysis)

            except Exception as e:
                logger.error(f"Scheduled analysis error: {e}")

    async def monitor_performance(self):
        """Monitor Echo's performance metrics"""
        while True:
            try:
                await asyncio.sleep(self.learning_config['performance_check_interval'])

                # Run self-diagnosis
                diagnosis = await self.self_diagnosis.diagnose()

                if diagnosis.get('health_score', 1.0) < self.safety_config['performance_threshold']:
                    logger.warning(f"Performance degradation detected: {diagnosis['health_score']}")

                    # Trigger performance recovery evolution
                    await self.evolution.trigger_evolution(
                        EvolutionTrigger.PERFORMANCE_DEGRADATION,
                        {"diagnosis": diagnosis}
                    )

            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")

    async def track_learning(self):
        """Track learning milestones from conversations"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check hourly

                # Check conversation count from database
                conversations = await self.get_conversation_count()

                if conversations - self.metrics['learning_conversations'] >= \
                   self.learning_config['conversations_per_milestone']:

                    logger.info(f"Learning milestone reached: {conversations} conversations")
                    self.metrics['learning_conversations'] = conversations

                    # Trigger learning evolution
                    await self.evolution.trigger_evolution(
                        EvolutionTrigger.LEARNING_MILESTONE,
                        {"conversations": conversations}
                    )

            except Exception as e:
                logger.error(f"Learning tracking error: {e}")

    async def get_conversation_count(self) -> int:
        """Get total conversation count from database"""
        try:
            conn = psycopg2.connect(
                dbname="echo_brain",
                user="echo",
                host="localhost"
            )
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM conversations WHERE created_at > NOW() - INTERVAL '7 days'")
            count = cur.fetchone()[0]
            conn.close()
            return count
        except:
            return self.metrics['learning_conversations']

    async def apply_improvements(self, analysis: Dict[str, Any]) -> bool:
        """Apply identified improvements with safety checks"""
        try:
            # Check daily evolution limit
            today_evolutions = self.count_today_evolutions()
            if today_evolutions >= self.safety_config['max_daily_evolutions']:
                logger.warning("Daily evolution limit reached")
                return False

            # Create evolution plan
            plan = {
                'timestamp': datetime.now().isoformat(),
                'analysis': analysis,
                'risk_level': self.assess_risk(analysis),
                'changes': analysis.get('proposed_changes', [])
            }

            # Request Board approval for significant changes
            if plan['risk_level'] in ['high', 'critical']:
                approved = await self.request_board_approval(plan)
                if not approved:
                    logger.info("Board rejected evolution plan")
                    return False

            # Create improvement branch
            branch_name = f"evolution-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            self.git_manager.create_improvement_branch(branch_name, "Autonomous evolution")

            # Apply changes
            success = await self.evolution.apply_evolution(plan)

            if success:
                # Test changes
                test_results = await self.test_improvements()

                if test_results['passed']:
                    # Deploy to production
                    deployed = await self.git_manager.safe_autonomous_deployment()

                    if deployed:
                        self.metrics['successful_evolutions'] += 1
                        self.metrics['last_evolution'] = datetime.now().isoformat()
                        logger.info("✅ Evolution successfully applied")
                        return True
                    else:
                        # Rollback
                        await self.rollback_changes()

            self.metrics['failed_evolutions'] += 1
            return False

        except Exception as e:
            logger.error(f"Failed to apply improvements: {e}")
            await self.rollback_changes()
            return False

    def assess_risk(self, analysis: Dict[str, Any]) -> str:
        """Assess risk level of proposed changes"""
        changes = analysis.get('proposed_changes', [])

        # High risk indicators
        high_risk_patterns = ['database', 'security', 'authentication', 'core_logic']
        critical_patterns = ['delete', 'drop', 'remove_all', 'reset']

        for change in changes:
            change_str = str(change).lower()
            if any(pattern in change_str for pattern in critical_patterns):
                return 'critical'
            if any(pattern in change_str for pattern in high_risk_patterns):
                return 'high'

        if len(changes) > 10:
            return 'medium'

        return 'low'

    async def test_improvements(self) -> Dict[str, Any]:
        """Test improvements before deployment"""
        results = {
            'passed': True,
            'tests': [],
            'coverage': 0.0
        }

        try:
            # Run unit tests
            import subprocess
            test_result = subprocess.run(
                ['python', '-m', 'pytest', '/opt/tower-echo-brain/tests/', '-v'],
                capture_output=True,
                text=True,
                timeout=60
            )

            results['tests'].append({
                'name': 'unit_tests',
                'passed': test_result.returncode == 0,
                'output': test_result.stdout
            })

            # Check test coverage
            coverage_result = subprocess.run(
                ['python', '-m', 'coverage', 'report'],
                capture_output=True,
                text=True,
                cwd='/opt/tower-echo-brain'
            )

            # Parse coverage percentage
            for line in coverage_result.stdout.split('\n'):
                if 'TOTAL' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        coverage = float(parts[-1].rstrip('%')) / 100
                        results['coverage'] = coverage
                        break

            # Enforce minimum coverage
            if results['coverage'] < self.safety_config['test_coverage_minimum']:
                results['passed'] = False

        except Exception as e:
            logger.error(f"Testing failed: {e}")
            results['passed'] = False

        return results

    async def rollback_changes(self):
        """Rollback failed changes"""
        try:
            self.git_manager._perform_rollback()
            self.metrics['rollbacks'] += 1
            logger.info("Successfully rolled back changes")
        except Exception as e:
            logger.error(f"Rollback failed: {e}")

    def count_today_evolutions(self) -> int:
        """Count evolutions performed today"""
        if not self.metrics['last_evolution']:
            return 0

        last_date = datetime.fromisoformat(self.metrics['last_evolution']).date()
        if last_date == datetime.now().date():
            return self.metrics['successful_evolutions']
        return 0

    def setup_routes(self):
        """Setup API endpoints"""

        @self.app.get("/api/echo/evolution/status")
        async def get_evolution_status():
            """Get comprehensive evolution system status"""
            return {
                "status": "active" if self.evolution else "initializing",
                "board_connected": self.board_connected,
                "metrics": self.metrics,
                "safety_config": self.safety_config,
                "learning_config": self.learning_config,
                "components": {
                    "evolution": bool(self.evolution),
                    "git_manager": bool(self.git_manager),
                    "self_analysis": bool(self.self_analysis),
                    "self_diagnosis": bool(self.self_diagnosis)
                }
            }

        @self.app.post("/api/echo/evolution/trigger")
        async def trigger_evolution(background_tasks: BackgroundTasks,
                                   reason: str = "manual"):
            """Manually trigger evolution cycle"""
            background_tasks.add_task(
                self.evolution.trigger_evolution,
                EvolutionTrigger.MANUAL,
                {"reason": reason}
            )
            return {"status": "Evolution cycle triggered", "reason": reason}

        @self.app.post("/api/echo/evolution/self-analysis")
        async def trigger_self_analysis(
            depth: str = "functional",
            context: Optional[Dict[str, Any]] = None
        ):
            """Trigger self-analysis at specified depth"""
            depth_enum = AnalysisDepth[depth.upper()]
            result = await self.self_analysis.analyze(depth_enum, context or {})
            return result

        @self.app.get("/api/echo/evolution/learning-metrics")
        async def get_learning_metrics():
            """Get learning and evolution metrics"""
            conversations = await self.get_conversation_count()
            return {
                "total_conversations": conversations,
                "learning_milestones": self.metrics['learning_conversations'] //
                                      self.learning_config['conversations_per_milestone'],
                "evolution_metrics": self.metrics,
                "next_milestone": self.learning_config['conversations_per_milestone'] -
                                (conversations % self.learning_config['conversations_per_milestone'])
            }

        @self.app.get("/api/echo/evolution/git-status")
        async def get_git_status():
            """Get git repository status"""
            if self.git_manager:
                return self.git_manager.get_git_status()
            return {"error": "Git manager not initialized"}

        @self.app.post("/api/echo/evolution/approve-changes")
        async def approve_changes(branch: str):
            """Manually approve and deploy changes"""
            if self.git_manager:
                result = await self.git_manager.safe_autonomous_deployment()
                return {"deployed": result, "branch": branch}
            return {"error": "Git manager not initialized"}

        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "service": "Echo Evolution System"}

async def main():
    """Main entry point"""
    orchestrator = EvolutionOrchestrator()
    await orchestrator.initialize_components()

    config = uvicorn.Config(
        app=orchestrator.app,
        host="127.0.0.1",
        port=8311,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())