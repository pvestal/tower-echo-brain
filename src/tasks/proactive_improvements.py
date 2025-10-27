#!/usr/bin/env python3
"""
Proactive Code Improvement System
Enables Echo to actively search, improve, and commit code changes
"""

import asyncio
import logging
import subprocess
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import git

logger = logging.getLogger(__name__)

class ProactiveImprovementEngine:
    """Engine for proactive code improvements and automated fixes"""

    def __init__(self):
        self.improvement_log = Path("/opt/tower-echo-brain/logs/proactive_improvements.log")
        self.improvement_log.parent.mkdir(parents=True, exist_ok=True)
        self.monitored_patterns = [
            "TODO",
            "FIXME",
            "XXX",
            "HACK",
            "BUG",
            "OPTIMIZE",
            "REFACTOR",
            "deprecated",
            "noqa",  # pylint/flake8 ignores that should be fixed
        ]
        self.auto_commit_enabled = False  # Safety first

    async def search_for_improvements(self, project_path: str) -> List[Dict[str, Any]]:
        """Search codebase for patterns that need improvement"""
        improvements = []

        for pattern in self.monitored_patterns:
            try:
                # Use ripgrep for fast searching
                result = subprocess.run(
                    ['rg', '-n', '--json', pattern, project_path],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.stdout:
                    for line in result.stdout.strip().split('\n'):
                        try:
                            data = json.loads(line)
                            if data.get('type') == 'match':
                                match_data = data.get('data', {})
                                improvements.append({
                                    'file': match_data.get('path', {}).get('text'),
                                    'line': match_data.get('line_number'),
                                    'pattern': pattern,
                                    'text': match_data.get('lines', {}).get('text', '').strip(),
                                    'priority': self._calculate_priority(pattern)
                                })
                        except json.JSONDecodeError:
                            continue

            except subprocess.TimeoutExpired:
                logger.warning(f"Search timeout for pattern {pattern}")
            except FileNotFoundError:
                # Fallback to grep if ripgrep not available
                try:
                    result = subprocess.run(
                        ['grep', '-rn', pattern, project_path],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if result.stdout:
                        for line in result.stdout.strip().split('\n'):
                            parts = line.split(':', 2)
                            if len(parts) >= 3:
                                improvements.append({
                                    'file': parts[0],
                                    'line': parts[1],
                                    'pattern': pattern,
                                    'text': parts[2].strip(),
                                    'priority': self._calculate_priority(pattern)
                                })
                except Exception as e:
                    logger.error(f"Grep fallback failed: {e}")

        # Sort by priority
        improvements.sort(key=lambda x: x['priority'], reverse=True)
        return improvements

    def _calculate_priority(self, pattern: str) -> int:
        """Calculate priority based on pattern type"""
        priority_map = {
            'BUG': 10,
            'FIXME': 8,
            'TODO': 5,
            'XXX': 7,
            'HACK': 6,
            'OPTIMIZE': 4,
            'REFACTOR': 3,
            'deprecated': 9,
            'noqa': 2
        }
        return priority_map.get(pattern, 1)

    async def auto_fix_file(self, file_path: str) -> Dict[str, Any]:
        """Automatically fix common issues in a file"""
        results = {
            'file': file_path,
            'fixes_applied': [],
            'success': False
        }

        if not Path(file_path).exists():
            results['error'] = 'File not found'
            return results

        try:
            # Black formatting
            black_result = subprocess.run(
                ['/opt/tower-echo-brain/venv/bin/black', file_path, '--quiet'],
                capture_output=True,
                timeout=30
            )
            if black_result.returncode == 0:
                results['fixes_applied'].append('black formatting')

            # Autopep8 fixes
            autopep8_result = subprocess.run(
                ['/opt/tower-echo-brain/venv/bin/autopep8', '--in-place', '--aggressive', file_path],
                capture_output=True,
                timeout=30
            )
            if autopep8_result.returncode == 0:
                results['fixes_applied'].append('autopep8 fixes')

            # Ruff fixes (if available)
            ruff_result = subprocess.run(
                ['/opt/tower-echo-brain/venv/bin/ruff', 'check', '--fix', file_path],
                capture_output=True,
                timeout=30
            )
            if ruff_result.returncode == 0:
                results['fixes_applied'].append('ruff fixes')

            results['success'] = len(results['fixes_applied']) > 0

        except Exception as e:
            results['error'] = str(e)

        return results

    async def commit_improvements(self, project_path: str, files: List[str], message: str) -> bool:
        """Commit improvements using git (requires permission)"""
        if not self.auto_commit_enabled:
            logger.info(f"Would commit {len(files)} files with message: {message}")
            return False

        try:
            repo = git.Repo(project_path)

            # Add files
            for file_path in files:
                repo.index.add([file_path])

            # Commit
            repo.index.commit(f"[Echo Auto-Fix] {message}")

            logger.info(f"✅ Committed {len(files)} improved files")
            return True

        except Exception as e:
            logger.error(f"Git commit failed: {e}")
            return False

    async def proactive_improvement_loop(self, project_paths: List[str]):
        """Main loop for proactive improvements"""
        logger.info("🚀 Proactive improvement engine started")

        while True:
            try:
                for project_path in project_paths:
                    logger.info(f"🔍 Scanning {project_path} for improvements")

                    # Search for improvements
                    improvements = await self.search_for_improvements(project_path)

                    if improvements:
                        logger.info(f"📋 Found {len(improvements)} potential improvements")

                        # Group by file
                        files_to_fix = {}
                        for imp in improvements[:10]:  # Limit to top 10 per cycle
                            file_path = imp['file']
                            if file_path not in files_to_fix:
                                files_to_fix[file_path] = []
                            files_to_fix[file_path].append(imp)

                        # Fix files
                        fixed_files = []
                        for file_path in files_to_fix:
                            result = await self.auto_fix_file(file_path)
                            if result['success']:
                                fixed_files.append(file_path)
                                logger.info(f"✅ Fixed {file_path}: {result['fixes_applied']}")

                        # Commit if enabled
                        if fixed_files and self.auto_commit_enabled:
                            await self.commit_improvements(
                                project_path,
                                fixed_files,
                                f"Auto-fixed {len(fixed_files)} files with code improvements"
                            )

                # Sleep 30 minutes between scans
                await asyncio.sleep(1800)

            except Exception as e:
                logger.error(f"Proactive improvement error: {e}")
                await asyncio.sleep(300)  # Retry in 5 minutes

    def enable_auto_commit(self, enabled: bool = True):
        """Enable or disable automatic git commits"""
        self.auto_commit_enabled = enabled
        logger.info(f"Auto-commit {'enabled' if enabled else 'disabled'}")

# Global instance
proactive_engine = ProactiveImprovementEngine()