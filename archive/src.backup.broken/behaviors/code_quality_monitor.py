#!/usr/bin/env python3
"""
Code quality monitoring behavior for Echo Brain
"""
import asyncio
import logging
import os
import subprocess
from datetime import datetime
from typing import Dict, List
from src.tasks.task_queue import TaskQueue, create_scheduled_task, TaskType, TaskPriority

logger = logging.getLogger(__name__)

class CodeQualityMonitor:
    """Monitors code quality and creates refactoring tasks"""

    def __init__(self, task_queue: TaskQueue):
        self.task_queue = task_queue
        self.monitored_projects = [
            '/opt/tower-echo-brain/src',
            '/opt/tower-anime-production',
            '/opt/tower-auth',
            '/opt/tower-kb'
        ]
        self.quality_threshold = 7.0  # Pylint score threshold

    async def analyze_code_quality(self):
        """Analyze code quality across all monitored projects"""
        try:
            total_files = 0
            files_needing_refactor = 0

            for project_path in self.monitored_projects:
                if not os.path.exists(project_path):
                    continue

                python_files = await self._find_python_files(project_path)
                total_files += len(python_files)

                for file_path in python_files:
                    quality_score = await self._analyze_file_quality(file_path)

                    if quality_score is not None and quality_score < self.quality_threshold:
                        await self._create_refactor_task(file_path, quality_score)
                        files_needing_refactor += 1

            logger.info(f"📊 Code quality analysis: {files_needing_refactor}/{total_files} files need refactoring")

            # Log analysis results
            await self._log_quality_analysis(total_files, files_needing_refactor)

        except Exception as e:
            logger.error(f"❌ Code quality analysis failed: {e}")

    async def _find_python_files(self, project_path: str) -> List[str]:
        """Find all Python files in a project"""
        try:
            result = subprocess.run(
                ['find', project_path, '-name', '*.py', '-type', 'f'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                return [f.strip() for f in result.stdout.split('\n') if f.strip()]
            else:
                logger.warning(f"Find command failed for {project_path}: {result.stderr}")
                return []

        except subprocess.TimeoutExpired:
            logger.warning(f"Find command timed out for {project_path}")
            return []
        except Exception as e:
            logger.error(f"Error finding Python files in {project_path}: {e}")
            return []

    async def _analyze_file_quality(self, file_path: str) -> float:
        """Analyze quality of a single Python file"""
        try:
            # Check if pylint is available
            pylint_result = subprocess.run(['which', 'pylint'], capture_output=True)
            if pylint_result.returncode != 0:
                return None  # Pylint not available

            # Run pylint analysis
            result = subprocess.run(
                ['pylint', '--score=y', '--reports=n', file_path],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Extract score from pylint output
            for line in result.stdout.split('\n'):
                if 'Your code has been rated at' in line:
                    # Parse score like "Your code has been rated at 8.33/10"
                    score_part = line.split('rated at')[1].split('/')[0].strip()
                    return float(score_part)

            return None

        except (subprocess.TimeoutExpired, ValueError, IndexError):
            return None
        except Exception as e:
            logger.debug(f"Pylint analysis failed for {file_path}: {e}")
            return None

    async def _create_refactor_task(self, file_path: str, quality_score: float):
        """Create a refactoring task for a low-quality file"""
        try:
            task = await create_scheduled_task(
                TaskType.CODE_REFACTOR,
                TaskPriority.NORMAL,
                f"Refactor {os.path.basename(file_path)} (quality: {quality_score:.1f}/10)",
                f"Code quality analysis identified {file_path} as needing refactoring. Current quality score: {quality_score:.1f}/10 (threshold: {self.quality_threshold}/10)",
                {
                    'file_path': file_path,
                    'current_quality': quality_score,
                    'threshold': self.quality_threshold,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'refactor_type': 'quality_improvement'
                }
            )

            await self.task_queue.add_task_object(task)
            logger.info(f"📝 Created refactor task for {file_path} (score: {quality_score:.1f})")

        except Exception as e:
            logger.error(f"Failed to create refactor task for {file_path}: {e}")

    async def _log_quality_analysis(self, total_files: int, files_needing_refactor: int):
        """Log code quality analysis results"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'total_files_analyzed': total_files,
                'files_needing_refactor': files_needing_refactor,
                'quality_threshold': self.quality_threshold,
                'projects_analyzed': len(self.monitored_projects)
            }

            # Write to analysis log
            log_file = '/opt/tower-echo-brain/logs/code_quality_analysis.log'
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            with open(log_file, 'a') as f:
                f.write(f"{log_entry}\n")

        except Exception as e:
            logger.error(f"Failed to log quality analysis: {e}")

    async def format_code_with_black(self, file_path: str) -> bool:
        """Format code using black formatter if available"""
        try:
            # Check if black is available
            black_result = subprocess.run(['which', 'black'], capture_output=True)
            if black_result.returncode != 0:
                return False

            # Run black formatter
            result = subprocess.run(
                ['black', '--quiet', file_path],
                capture_output=True,
                timeout=10
            )

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            logger.warning(f"Black formatting timed out for {file_path}")
            return False
        except Exception as e:
            logger.error(f"Black formatting failed for {file_path}: {e}")
            return False

    def get_quality_stats(self) -> Dict:
        """Get code quality monitoring statistics"""
        return {
            'monitored_projects': len(self.monitored_projects),
            'quality_threshold': self.quality_threshold,
            'projects': self.monitored_projects
        }