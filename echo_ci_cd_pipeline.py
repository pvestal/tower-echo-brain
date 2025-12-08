#!/usr/bin/env python3
"""
Echo CI/CD Pipeline Orchestrator
Main orchestrator that coordinates test generation, deployment, and monitoring
Integrates with Echo's board system and learning capabilities
"""

import asyncio
import hashlib
import json
import os
import sqlite3
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiofiles
import aiohttp
import git
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from echo_deployment_manager import (DeploymentConfig, DeploymentManager,
                                     EnvironmentType)
# Import our custom modules
from echo_test_generator import EchoTestGeneratorService, TestSuite


class PipelineStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    TESTING = "testing"
    DEPLOYING = "deploying"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineStage(Enum):
    CODE_ANALYSIS = "code_analysis"
    TEST_GENERATION = "test_generation"
    TEST_EXECUTION = "test_execution"
    SECURITY_SCAN = "security_scan"
    BUILD = "build"
    DEPLOYMENT = "deployment"
    HEALTH_CHECK = "health_check"
    NOTIFICATION = "notification"


@dataclass
class PipelineConfig:
    """Configuration for CI/CD pipeline"""

    pipeline_id: str
    app_name: str
    git_repo: str
    git_branch: str
    git_commit: str
    trigger_type: str  # manual, webhook, schedule
    target_environments: List[str]
    run_tests: bool = True
    deploy_on_success: bool = True
    notify_on_completion: bool = True
    board_decision_id: Optional[str] = None


@dataclass
class PipelineRun:
    """Represents a pipeline execution"""

    id: str
    config: PipelineConfig
    status: PipelineStatus
    current_stage: Optional[PipelineStage]
    start_time: datetime
    end_time: Optional[datetime] = None
    stages: Dict[PipelineStage, Dict[str, Any]] = None
    artifacts: Dict[str, str] = None
    logs: List[str] = None
    test_results: Optional[Dict[str, Any]] = None
    deployment_results: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.stages is None:
            self.stages = {}
        if self.artifacts is None:
            self.artifacts = {}
        if self.logs is None:
            self.logs = []


class GitManager:
    """Handles Git operations for the pipeline"""

    def __init__(self, base_repo_dir: str = "/opt/tower-echo-brain/repos"):
        self.base_repo_dir = Path(base_repo_dir)
        self.base_repo_dir.mkdir(exist_ok=True)

    async def clone_or_update_repo(
        self, repo_url: str, branch: str, commit: str
    ) -> Tuple[bool, str]:
        """Clone or update repository to specific commit"""
        try:
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            repo_path = self.base_repo_dir / repo_name

            if repo_path.exists():
                # Update existing repo
                repo = git.Repo(repo_path)
                repo.git.fetch("origin")
                repo.git.checkout(commit)
                return True, str(repo_path)
            else:
                # Clone new repo
                repo = git.Repo.clone_from(repo_url, repo_path)
                repo.git.checkout(commit)
                return True, str(repo_path)

        except Exception as e:
            return False, f"Git operation failed: {str(e)}"

    async def get_changed_files(
        self, repo_path: str, base_commit: str, target_commit: str
    ) -> List[str]:
        """Get list of changed files between commits"""
        try:
            repo = git.Repo(repo_path)
            diff = repo.git.diff(
                "--name-only", f"{base_commit}..{target_commit}")
            return diff.split("\n") if diff else []
        except Exception:
            return []

    async def get_commit_info(self, repo_path: str, commit: str) -> Dict[str, Any]:
        """Get commit information"""
        try:
            repo = git.Repo(repo_path)
            commit_obj = repo.commit(commit)
            return {
                "sha": commit_obj.hexsha,
                "author": str(commit_obj.author),
                "message": commit_obj.message,
                "timestamp": commit_obj.committed_datetime.isoformat(),
            }
        except Exception:
            return {}


class SecurityScanner:
    """Performs security scans on code"""

    def __init__(self):
        self.scanners = ["bandit", "safety", "semgrep"]

    async def scan_code(self, repo_path: str) -> Dict[str, Any]:
        """Run security scans on repository"""
        results = {
            "overall_score": 0,
            "vulnerabilities": [],
            "warnings": [],
            "scan_results": {},
        }

        try:
            # Run bandit for Python security issues
            bandit_result = await self._run_bandit(repo_path)
            results["scan_results"]["bandit"] = bandit_result

            # Run safety for dependency vulnerabilities
            safety_result = await self._run_safety(repo_path)
            results["scan_results"]["safety"] = safety_result

            # Calculate overall score
            results["overall_score"] = self._calculate_security_score(
                results["scan_results"]
            )

        except Exception as e:
            results["error"] = str(e)

        return results

    async def _run_bandit(self, repo_path: str) -> Dict[str, Any]:
        """Run bandit security scanner"""
        try:
            result = subprocess.run(
                ["bandit", "-r", repo_path, "-f", "json"],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.stdout:
                return json.loads(result.stdout)
            else:
                return {"results": [], "metrics": {}}

        except Exception as e:
            return {"error": str(e)}

    async def _run_safety(self, repo_path: str) -> Dict[str, Any]:
        """Run safety dependency scanner"""
        try:
            requirements_file = Path(repo_path) / "requirements.txt"
            if not requirements_file.exists():
                return {"results": [], "note": "No requirements.txt found"}

            result = subprocess.run(
                ["safety", "check", "-r", str(requirements_file), "--json"],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.stdout:
                return json.loads(result.stdout)
            else:
                return {"results": []}

        except Exception as e:
            return {"error": str(e)}

    def _calculate_security_score(self, scan_results: Dict[str, Any]) -> int:
        """Calculate overall security score (0-100)"""
        score = 100

        # Deduct points for bandit issues
        bandit_results = scan_results.get("bandit", {}).get("results", [])
        for issue in bandit_results:
            severity = issue.get("issue_severity", "LOW")
            if severity == "HIGH":
                score -= 20
            elif severity == "MEDIUM":
                score -= 10
            elif severity == "LOW":
                score -= 5

        # Deduct points for safety vulnerabilities
        safety_results = scan_results.get("safety", {}).get("results", [])
        score -= len(safety_results) * 15

        return max(0, score)


class PipelineExecutor:
    """Executes CI/CD pipeline stages"""

    def __init__(self):
        self.git_manager = GitManager()
        self.test_generator = EchoTestGeneratorService()
        self.deployment_manager = DeploymentManager()
        self.security_scanner = SecurityScanner()

    async def execute_pipeline(self, pipeline_run: PipelineRun) -> PipelineRun:
        """Execute complete CI/CD pipeline"""
        try:
            pipeline_run.status = PipelineStatus.RUNNING
            pipeline_run.logs.append(f"Starting pipeline {pipeline_run.id}")

            # Stage 1: Code Analysis
            await self._execute_stage(
                pipeline_run, PipelineStage.CODE_ANALYSIS, self._code_analysis_stage
            )

            # Stage 2: Test Generation
            if pipeline_run.config.run_tests:
                await self._execute_stage(
                    pipeline_run,
                    PipelineStage.TEST_GENERATION,
                    self._test_generation_stage,
                )
                await self._execute_stage(
                    pipeline_run,
                    PipelineStage.TEST_EXECUTION,
                    self._test_execution_stage,
                )

            # Stage 3: Security Scan
            await self._execute_stage(
                pipeline_run, PipelineStage.SECURITY_SCAN, self._security_scan_stage
            )

            # Stage 4: Build
            await self._execute_stage(
                pipeline_run, PipelineStage.BUILD, self._build_stage
            )

            # Stage 5: Deployment
            if pipeline_run.config.deploy_on_success:
                await self._execute_stage(
                    pipeline_run, PipelineStage.DEPLOYMENT, self._deployment_stage
                )
                await self._execute_stage(
                    pipeline_run, PipelineStage.HEALTH_CHECK, self._health_check_stage
                )

            # Stage 6: Notification
            if pipeline_run.config.notify_on_completion:
                await self._execute_stage(
                    pipeline_run, PipelineStage.NOTIFICATION, self._notification_stage
                )

            pipeline_run.status = PipelineStatus.SUCCESS
            pipeline_run.end_time = datetime.now()
            pipeline_run.logs.append(
                f"Pipeline {pipeline_run.id} completed successfully"
            )

        except Exception as e:
            pipeline_run.status = PipelineStatus.FAILED
            pipeline_run.logs.append(f"Pipeline failed: {str(e)}")
            pipeline_run.end_time = datetime.now()

        return pipeline_run

    async def _execute_stage(
        self, pipeline_run: PipelineRun, stage: PipelineStage, stage_func
    ):
        """Execute a pipeline stage"""
        pipeline_run.current_stage = stage
        pipeline_run.logs.append(f"Starting stage: {stage.value}")

        stage_start = datetime.now()
        try:
            result = await stage_func(pipeline_run)
            pipeline_run.stages[stage] = {
                "status": "success",
                "start_time": stage_start.isoformat(),
                "end_time": datetime.now().isoformat(),
                "result": result,
            }
            pipeline_run.logs.append(
                f"Stage {stage.value} completed successfully")

        except Exception as e:
            pipeline_run.stages[stage] = {
                "status": "failed",
                "start_time": stage_start.isoformat(),
                "end_time": datetime.now().isoformat(),
                "error": str(e),
            }
            pipeline_run.logs.append(f"Stage {stage.value} failed: {str(e)}")
            raise

    async def _code_analysis_stage(self, pipeline_run: PipelineRun) -> Dict[str, Any]:
        """Analyze code and prepare for pipeline"""
        config = pipeline_run.config

        # Clone/update repository
        success, repo_path = await self.git_manager.clone_or_update_repo(
            config.git_repo, config.git_branch, config.git_commit
        )

        if not success:
            raise Exception(f"Failed to clone repository: {repo_path}")

        pipeline_run.artifacts["repo_path"] = repo_path

        # Get commit information
        commit_info = await self.git_manager.get_commit_info(
            repo_path, config.git_commit
        )

        # Analyze changed files
        changed_files = []
        if config.trigger_type == "webhook":
            # In a real scenario, we'd get the base commit from the webhook
            # For now, we'll analyze all Python files
            for file_path in Path(repo_path).rglob("*.py"):
                if file_path.is_file():
                    changed_files.append(str(file_path.relative_to(repo_path)))

        return {
            "repo_path": repo_path,
            "commit_info": commit_info,
            "changed_files": changed_files,
            "total_files": len(changed_files),
        }

    async def _test_generation_stage(self, pipeline_run: PipelineRun) -> Dict[str, Any]:
        """Generate tests for changed code"""
        repo_path = pipeline_run.artifacts["repo_path"]
        changed_files = pipeline_run.stages[PipelineStage.CODE_ANALYSIS]["result"][
            "changed_files"
        ]

        test_results = []

        for file_path in changed_files:
            if file_path.endswith(".py") and not file_path.startswith("test_"):
                full_path = os.path.join(repo_path, file_path)
                if os.path.exists(full_path):
                    module_name = file_path.replace(
                        "/", ".").replace(".py", "")

                    # Generate tests
                    result = await self.test_generator.generate_and_run_tests(
                        full_path, module_name, pipeline_run.config.board_decision_id
                    )

                    test_results.append(
                        {"file": file_path, "module": module_name, "result": result}
                    )

        return {"generated_tests": len(test_results), "test_results": test_results}

    async def _test_execution_stage(self, pipeline_run: PipelineRun) -> Dict[str, Any]:
        """Execute generated tests"""
        test_results = pipeline_run.stages[PipelineStage.TEST_GENERATION]["result"][
            "test_results"
        ]

        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        coverage_sum = 0

        for test_result in test_results:
            result = test_result["result"]
            if result.get("success"):
                total_tests += result.get("test_suite",
                                          {}).get("test_count", 0)
                if result.get("results", {}).get("success"):
                    passed_tests += result.get("test_suite",
                                               {}).get("test_count", 0)
                else:
                    failed_tests += result.get("test_suite",
                                               {}).get("test_count", 0)

                coverage = (
                    result.get("results", {}).get(
                        "coverage", {}).get("percentage", 0)
                )
                coverage_sum += coverage

        avg_coverage = coverage_sum / len(test_results) if test_results else 0

        # Fail pipeline if test failure rate is too high
        if (
            total_tests > 0 and (failed_tests / total_tests) > 0.1
        ):  # 10% failure threshold
            raise Exception(
                f"Test failure rate too high: {failed_tests}/{total_tests} failed"
            )

        pipeline_run.test_results = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "average_coverage": avg_coverage,
        }

        return pipeline_run.test_results

    async def _security_scan_stage(self, pipeline_run: PipelineRun) -> Dict[str, Any]:
        """Run security scans"""
        repo_path = pipeline_run.artifacts["repo_path"]

        security_results = await self.security_scanner.scan_code(repo_path)

        # Fail pipeline if security score is too low
        if security_results.get("overall_score", 0) < 70:  # 70% minimum security score
            raise Exception(
                f"Security score too low: {security_results['overall_score']}/100"
            )

        return security_results

    async def _build_stage(self, pipeline_run: PipelineRun) -> Dict[str, Any]:
        """Build application artifacts"""
        repo_path = pipeline_run.artifacts["repo_path"]

        # For Python applications, we'll create a simple build
        build_dir = Path(f"/tmp/echo_builds/{pipeline_run.id}")
        build_dir.mkdir(parents=True, exist_ok=True)

        # Copy source code
        import shutil

        shutil.copytree(repo_path, build_dir / "src", dirs_exist_ok=True)

        # Create build metadata
        build_metadata = {
            "build_id": pipeline_run.id,
            "commit": pipeline_run.config.git_commit,
            "timestamp": datetime.now().isoformat(),
            "test_results": pipeline_run.test_results,
        }

        with open(build_dir / "build.json", "w") as f:
            json.dump(build_metadata, f, indent=2)

        pipeline_run.artifacts["build_path"] = str(build_dir)

        return {
            "build_path": str(build_dir),
            "artifacts_created": ["src/", "build.json"],
        }

    async def _deployment_stage(self, pipeline_run: PipelineRun) -> Dict[str, Any]:
        """Deploy application to target environments"""
        build_path = pipeline_run.artifacts["build_path"]
        config = pipeline_run.config

        deployment_results = []

        for env in config.target_environments:
            try:
                deployment_config = DeploymentConfig(
                    app_name=config.app_name,
                    version=f"{config.git_commit[:8]}-{int(datetime.now().timestamp())}",
                    git_commit=config.git_commit,
                    source_path=os.path.join(build_path, "src"),
                    target_env=EnvironmentType(env),
                    port=(
                        8309 if env == "production" else 8310
                    ),  # Different ports for different envs
                    health_check_url="http://localhost:{port}/api/health",
                    dependencies=[],
                    environment_vars={
                        "PIPELINE_ID": pipeline_run.id,
                        "GIT_COMMIT": config.git_commit,
                        "BUILD_TIMESTAMP": datetime.now().isoformat(),
                    },
                )

                deployment_id = await self.deployment_manager.deploy(deployment_config)

                deployment_results.append(
                    {
                        "environment": env,
                        "deployment_id": deployment_id,
                        "status": "initiated",
                    }
                )

            except Exception as e:
                deployment_results.append(
                    {"environment": env, "error": str(e), "status": "failed"}
                )

        pipeline_run.deployment_results = deployment_results
        return {"deployments": deployment_results}

    async def _health_check_stage(self, pipeline_run: PipelineRun) -> Dict[str, Any]:
        """Verify deployed applications are healthy"""
        if not pipeline_run.deployment_results:
            return {"status": "skipped", "reason": "no deployments"}

        health_results = []

        for deployment in pipeline_run.deployment_results:
            if deployment["status"] == "initiated":
                # Wait for deployment to complete
                await asyncio.sleep(30)

                # Check deployment status
                deployment_status = await self.deployment_manager.get_deployment_status(
                    deployment["deployment_id"]
                )

                health_results.append(
                    {
                        "environment": deployment["environment"],
                        "deployment_id": deployment["deployment_id"],
                        "health_status": (
                            deployment_status.get("health_status", False)
                            if deployment_status
                            else False
                        ),
                        "deployment_status": (
                            deployment_status.get("status", "unknown")
                            if deployment_status
                            else "unknown"
                        ),
                    }
                )

        # Check if any critical environments failed
        for result in health_results:
            if result["environment"] == "production" and not result["health_status"]:
                raise Exception(
                    f"Production deployment health check failed: {result['deployment_id']}"
                )

        return {"health_checks": health_results}

    async def _notification_stage(self, pipeline_run: PipelineRun) -> Dict[str, Any]:
        """Send notifications about pipeline completion"""
        try:
            # Send to Echo's notification system
            notification_data = {
                "type": "pipeline_completion",
                "pipeline_id": pipeline_run.id,
                "status": pipeline_run.status.value,
                "app_name": pipeline_run.config.app_name,
                "git_commit": pipeline_run.config.git_commit,
                "duration": (
                    (pipeline_run.end_time - pipeline_run.start_time).total_seconds()
                    if pipeline_run.end_time
                    else None
                ),
                "test_results": pipeline_run.test_results,
                "deployment_results": pipeline_run.deployment_results,
            }

            # Post to notification service (port 8350)
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:8350/api/notifications",
                    json={
                        "recipient": "patrick.vestal.digital@gmail.com",
                        "subject": f"CI/CD Pipeline {pipeline_run.status.value}: {pipeline_run.config.app_name}",
                        "body": json.dumps(notification_data, indent=2),
                        "type": "email",
                    },
                ) as resp:
                    if resp.status == 200:
                        return {"status": "sent", "method": "email"}
                    else:
                        return {"status": "failed", "error": f"HTTP {resp.status}"}

        except Exception as e:
            return {"status": "failed", "error": str(e)}


class EchoCICDPipeline:
    """Main CI/CD pipeline service"""

    def __init__(self):
        self.executor = PipelineExecutor()
        self.db_path = "/opt/tower-echo-brain/data/pipelines.db"
        self.running_pipelines = {}
        self._init_database()

    def _init_database(self):
        """Initialize pipeline database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pipelines (
                    id TEXT PRIMARY KEY,
                    config TEXT,
                    status TEXT,
                    current_stage TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    stages TEXT,
                    artifacts TEXT,
                    logs TEXT,
                    test_results TEXT,
                    deployment_results TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pipeline_id TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    timestamp TEXT,
                    FOREIGN KEY (pipeline_id) REFERENCES pipelines (id)
                )
            """
            )

    async def trigger_pipeline(self, config: PipelineConfig) -> str:
        """Trigger a new pipeline run"""
        pipeline_id = f"{config.app_name}-{config.git_commit[:8]}-{int(datetime.now().timestamp())}"
        config.pipeline_id = pipeline_id

        pipeline_run = PipelineRun(
            id=pipeline_id,
            config=config,
            status=PipelineStatus.PENDING,
            current_stage=None,
            start_time=datetime.now(),
        )

        # Save to database
        await self._save_pipeline(pipeline_run)

        # Execute pipeline asynchronously
        asyncio.create_task(self._execute_pipeline_async(pipeline_run))

        return pipeline_id

    async def _execute_pipeline_async(self, pipeline_run: PipelineRun):
        """Execute pipeline asynchronously"""
        try:
            self.running_pipelines[pipeline_run.id] = pipeline_run
            pipeline_run = await self.executor.execute_pipeline(pipeline_run)
            await self._save_pipeline(pipeline_run)

            # Update Echo's learning system with pipeline results
            await self._update_echo_learning(pipeline_run)

        except Exception as e:
            pipeline_run.status = PipelineStatus.FAILED
            pipeline_run.logs.append(f"Pipeline execution failed: {str(e)}")
            await self._save_pipeline(pipeline_run)
        finally:
            self.running_pipelines.pop(pipeline_run.id, None)

    async def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get pipeline status"""
        # Check running pipelines first
        if pipeline_id in self.running_pipelines:
            return self._pipeline_to_dict(self.running_pipelines[pipeline_id])

        # Check database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM pipelines WHERE id = ?", (pipeline_id,)
            )
            row = cursor.fetchone()

            if row:
                columns = [description[0]
                           for description in cursor.description]
                pipeline_data = dict(zip(columns, row))

                # Parse JSON fields
                for field in [
                    "config",
                    "stages",
                    "artifacts",
                    "logs",
                    "test_results",
                    "deployment_results",
                ]:
                    if pipeline_data[field]:
                        pipeline_data[field] = json.loads(pipeline_data[field])

                return pipeline_data

            return None

    async def cancel_pipeline(self, pipeline_id: str) -> bool:
        """Cancel a running pipeline"""
        if pipeline_id in self.running_pipelines:
            pipeline_run = self.running_pipelines[pipeline_id]
            pipeline_run.status = PipelineStatus.CANCELLED
            pipeline_run.end_time = datetime.now()
            pipeline_run.logs.append("Pipeline cancelled by user")

            await self._save_pipeline(pipeline_run)
            return True

        return False

    async def list_pipelines(
        self, app_name: str = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """List recent pipelines"""
        with sqlite3.connect(self.db_path) as conn:
            if app_name:
                cursor = conn.execute(
                    """
                    SELECT * FROM pipelines 
                    WHERE JSON_EXTRACT(config, '$.app_name') = ?
                    ORDER BY start_time DESC LIMIT ?
                """,
                    (app_name, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM pipelines ORDER BY start_time DESC LIMIT ?
                """,
                    (limit,),
                )

            columns = [description[0] for description in cursor.description]
            pipelines = []

            for row in cursor.fetchall():
                pipeline_data = dict(zip(columns, row))

                # Parse JSON fields
                for field in [
                    "config",
                    "stages",
                    "artifacts",
                    "logs",
                    "test_results",
                    "deployment_results",
                ]:
                    if pipeline_data[field]:
                        pipeline_data[field] = json.loads(pipeline_data[field])

                pipelines.append(pipeline_data)

            return pipelines

    async def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics and analytics"""
        with sqlite3.connect(self.db_path) as conn:
            # Success rate
            cursor = conn.execute(
                """
                SELECT 
                    COUNT(*) as total_pipelines,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_pipelines,
                    AVG(CASE WHEN end_time IS NOT NULL THEN 
                        JULIANDAY(end_time) - JULIANDAY(start_time) 
                    ELSE NULL END) * 24 * 60 as avg_duration_minutes
                FROM pipelines
                WHERE start_time > datetime('now', '-7 days')
            """
            )

            result = cursor.fetchone()

            return {
                "total_pipelines": result[0] or 0,
                "successful_pipelines": result[1] or 0,
                "success_rate": (result[1] / result[0] * 100) if result[0] > 0 else 0,
                "average_duration_minutes": result[2] or 0,
            }

    async def _save_pipeline(self, pipeline_run: PipelineRun):
        """Save pipeline to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pipelines 
                (id, config, status, current_stage, start_time, end_time, 
                 stages, artifacts, logs, test_results, deployment_results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    pipeline_run.id,
                    json.dumps(asdict(pipeline_run.config), default=str),
                    pipeline_run.status.value,
                    (
                        pipeline_run.current_stage.value
                        if pipeline_run.current_stage
                        else None
                    ),
                    pipeline_run.start_time.isoformat(),
                    (
                        pipeline_run.end_time.isoformat()
                        if pipeline_run.end_time
                        else None
                    ),
                    json.dumps(
                        {k.value: v for k, v in pipeline_run.stages.items()},
                        default=str,
                    ),
                    json.dumps(pipeline_run.artifacts),
                    json.dumps(pipeline_run.logs),
                    json.dumps(pipeline_run.test_results),
                    json.dumps(pipeline_run.deployment_results),
                ),
            )

    def _pipeline_to_dict(self, pipeline_run: PipelineRun) -> Dict[str, Any]:
        """Convert pipeline run to dictionary"""
        return {
            "id": pipeline_run.id,
            "config": asdict(pipeline_run.config),
            "status": pipeline_run.status.value,
            "current_stage": (
                pipeline_run.current_stage.value if pipeline_run.current_stage else None
            ),
            "start_time": pipeline_run.start_time.isoformat(),
            "end_time": (
                pipeline_run.end_time.isoformat() if pipeline_run.end_time else None
            ),
            "stages": {k.value: v for k, v in pipeline_run.stages.items()},
            "artifacts": pipeline_run.artifacts,
            "logs": pipeline_run.logs,
            "test_results": pipeline_run.test_results,
            "deployment_results": pipeline_run.deployment_results,
        }

    async def _update_echo_learning(self, pipeline_run: PipelineRun):
        """Update Echo's learning system with pipeline results"""
        try:
            learning_data = {
                "type": "pipeline_execution",
                "pipeline_id": pipeline_run.id,
                "success": pipeline_run.status == PipelineStatus.SUCCESS,
                "duration": (
                    (pipeline_run.end_time - pipeline_run.start_time).total_seconds()
                    if pipeline_run.end_time
                    else None
                ),
                "stages_completed": len(
                    [
                        s
                        for s in pipeline_run.stages.values()
                        if s.get("status") == "success"
                    ]
                ),
                "test_coverage": (
                    pipeline_run.test_results.get("average_coverage", 0)
                    if pipeline_run.test_results
                    else 0
                ),
                "security_score": pipeline_run.stages.get(
                    PipelineStage.SECURITY_SCAN, {}
                )
                .get("result", {})
                .get("overall_score", 0),
                "app_name": pipeline_run.config.app_name,
                "git_commit": pipeline_run.config.git_commit,
            }

            # Post to Echo's learning endpoint
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:8309/api/learning/pipeline-feedback",
                    json=learning_data,
                ) as resp:
                    if resp.status != 200:
                        print(
                            f"Failed to update Echo learning: HTTP {resp.status}")

        except Exception as e:
            print(f"Error updating Echo learning: {e}")


# REST API for CI/CD pipeline

app = FastAPI(title="Echo CI/CD Pipeline", version="1.0.0")
pipeline_service = EchoCICDPipeline()


class PipelineTriggerRequest(BaseModel):
    app_name: str
    git_repo: str
    git_branch: str = "main"
    git_commit: str
    trigger_type: str = "manual"
    target_environments: List[str] = ["staging"]
    run_tests: bool = True
    deploy_on_success: bool = True
    notify_on_completion: bool = True
    board_decision_id: Optional[str] = None


@app.post("/api/pipelines/trigger")
async def trigger_pipeline(request: PipelineTriggerRequest):
    """Trigger a CI/CD pipeline"""
    try:
        config = PipelineConfig(
            pipeline_id="",  # Will be set by the service
            app_name=request.app_name,
            git_repo=request.git_repo,
            git_branch=request.git_branch,
            git_commit=request.git_commit,
            trigger_type=request.trigger_type,
            target_environments=request.target_environments,
            run_tests=request.run_tests,
            deploy_on_success=request.deploy_on_success,
            notify_on_completion=request.notify_on_completion,
            board_decision_id=request.board_decision_id,
        )

        pipeline_id = await pipeline_service.trigger_pipeline(config)
        return {"pipeline_id": pipeline_id, "status": "triggered"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pipelines/{pipeline_id}")
async def get_pipeline_status(pipeline_id: str):
    """Get pipeline status"""
    status = await pipeline_service.get_pipeline_status(pipeline_id)
    if not status:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return status


@app.post("/api/pipelines/{pipeline_id}/cancel")
async def cancel_pipeline(pipeline_id: str):
    """Cancel a pipeline"""
    success = await pipeline_service.cancel_pipeline(pipeline_id)
    if not success:
        raise HTTPException(
            status_code=404, detail="Pipeline not found or not running")
    return {"status": "cancelled"}


@app.get("/api/pipelines")
async def list_pipelines(app_name: Optional[str] = None, limit: int = 50):
    """List pipelines"""
    return await pipeline_service.list_pipelines(app_name, limit)


@app.get("/api/pipelines/metrics")
async def get_pipeline_metrics():
    """Get pipeline metrics"""
    return await pipeline_service.get_pipeline_metrics()


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "echo_ci_cd_pipeline",
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8342)
