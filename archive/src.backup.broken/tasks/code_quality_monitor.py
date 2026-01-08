#!/usr/bin/env python3
"""Code quality monitoring module for Echo Brain"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.tasks.task_queue import Task, TaskType, TaskPriority
from src.tasks.code_refactor_executor import code_refactor_executor

logger = logging.getLogger(__name__)

class CodeQualityMonitor:
    """Monitor code quality metrics and trends"""

    def __init__(self, task_queue=None):
        self.monitoring_active = True
        self.task_queue = task_queue
        self.monitored_projects = [
            "/opt/tower-echo-brain/src",
            "/opt/tower-anime-production",
            "/opt/tower-auth",
            "/opt/tower-kb",
            "/opt/tower-apple-music",
            "/opt/tower-personal-media",
            "/opt/tower-crypto-trader",
            "/opt/tower-dashboard"
        ]
        self.quality_threshold = 7.0  # Minimum quality score
        self.last_analysis = {}
        self.quality_history = {}

    async def analyze_code_quality(self, project_path: str) -> Dict[str, Any]:
        """Analyze code quality for a project and create refactor tasks if needed"""
        logger.info(f"🔍 Analyzing code quality: {project_path}")

        try:
            # Check if project path exists
            path = Path(project_path)
            if not path.exists():
                logger.warning(f"Project path does not exist: {project_path}")
                return {'status': 'path_not_found', 'project': project_path}

            # Run code quality analysis
            results = await code_refactor_executor.analyze_code_quality(project_path)

            # Store analysis results
            self.last_analysis[project_path] = results
            self.quality_history.setdefault(project_path, []).append({
                'timestamp': datetime.now().isoformat(),
                'score': results.get('pylint_score'),
                'file_count': results.get('file_count'),
                'issue_count': len(results.get('issues', []))
            })

            # Check if refactoring is needed
            if self._needs_refactoring(results):
                await self._create_refactor_task(project_path, results)

            return {
                'status': 'analyzed',
                'project': project_path,
                'quality_score': results.get('pylint_score'),
                'needs_refactoring': self._needs_refactoring(results),
                'file_count': results.get('file_count'),
                'issue_count': len(results.get('issues', []))
            }

        except Exception as e:
            logger.error(f"Code quality analysis failed for {project_path}: {e}")
            return {
                'status': 'error',
                'project': project_path,
                'error': str(e)
            }

    def _needs_refactoring(self, analysis_results: Dict[str, Any]) -> bool:
        """Determine if a project needs refactoring based on analysis"""
        pylint_score = analysis_results.get('pylint_score')
        file_count = analysis_results.get('file_count', 0)

        # Skip refactoring for very small projects or if no score available
        if file_count < 5 or pylint_score is None:
            return False

        # Need refactoring if score is below threshold
        return pylint_score < self.quality_threshold

    async def _create_refactor_task(self, project_path: str, analysis_results: Dict[str, Any]):
        """Create a code refactoring task for the project"""
        if not self.task_queue:
            logger.warning("No task queue available for creating refactor tasks")
            return

        try:
            task = Task(
                name=f"Code Refactoring: {Path(project_path).name}",
                task_type=TaskType.CODE_REFACTOR,
                priority=TaskPriority.NORMAL,
                payload={
                    'action': 'auto_fix_code',
                    'project_path': project_path,
                    'quality_score': analysis_results.get('pylint_score'),
                    'issue_count': len(analysis_results.get('issues', [])),
                    'file_count': analysis_results.get('file_count')
                },
                timeout=900  # 15 minutes timeout for refactoring
            )

            await self.task_queue.add_task(task)
            logger.info(f"📝 Created refactoring task for {project_path} (score: {analysis_results.get('pylint_score')})")

        except Exception as e:
            logger.error(f"Failed to create refactor task for {project_path}: {e}")

    async def trigger_emergency_refactoring(self, file_path: Optional[str] = None):
        """Trigger emergency refactoring for critical issues"""
        if file_path:
            # Refactor specific file
            project_path = str(Path(file_path).parent)
            await self.analyze_code_quality(project_path)
        else:
            # Scan all monitored projects
            for project in self.monitored_projects:
                await self.analyze_code_quality(project)

    async def get_quality_metrics(self) -> Dict[str, Any]:
        """Get overall quality metrics"""
        metrics = {
            'monitoring_active': self.monitoring_active,
            'monitored_projects': len(self.monitored_projects),
            'quality_threshold': self.quality_threshold,
            'projects': {}
        }

        for project_path, analysis in self.last_analysis.items():
            metrics['projects'][project_path] = {
                'quality_score': analysis.get('pylint_score'),
                'file_count': analysis.get('file_count'),
                'issue_count': len(analysis.get('issues', [])),
                'needs_refactoring': self._needs_refactoring(analysis),
                'last_analyzed': analysis.get('timestamp')
            }

        return metrics

    def get_quality_stats(self) -> Dict[str, Any]:
        """Get quality statistics for behavior reporting"""
        total_projects = len(self.monitored_projects)
        analyzed_projects = len(self.last_analysis)

        # Calculate average quality score
        scores = [analysis.get('pylint_score') for analysis in self.last_analysis.values()
                 if analysis.get('pylint_score') is not None]
        avg_score = sum(scores) / len(scores) if scores else None

        # Count projects needing refactoring
        projects_needing_refactor = sum(1 for analysis in self.last_analysis.values()
                                      if self._needs_refactoring(analysis))

        return {
            'total_projects': total_projects,
            'analyzed_projects': analyzed_projects,
            'average_quality_score': round(avg_score, 2) if avg_score else None,
            'projects_needing_refactor': projects_needing_refactor,
            'quality_threshold': self.quality_threshold,
            'monitoring_active': self.monitoring_active,
            'last_scan': max([analysis.get('timestamp', '') for analysis in self.last_analysis.values()], default='never')
        }