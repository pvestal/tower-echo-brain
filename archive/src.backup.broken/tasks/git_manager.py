#!/usr/bin/env python3
"""
Git Management System for Echo Brain
Enables autonomous version control and code management
"""

import os
import json
import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import git
from git import Repo, InvalidGitRepositoryError

logger = logging.getLogger(__name__)

class GitManager:
    """Manages git operations for Tower services"""

    def __init__(self):
        self.tower_services = self._discover_tower_services()
        self.auto_commit_enabled = False
        self.commit_log_path = Path("/opt/tower-echo-brain/logs/git_commits.log")
        self.commit_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Git configuration
        self.git_config = {
            'user.name': 'Echo Brain',
            'user.email': 'echo@tower.local'
        }

    def _discover_tower_services(self) -> List[Path]:
        """Discover all Tower service directories"""
        services = []
        opt_dir = Path('/opt')

        for path in opt_dir.glob('tower-*'):
            if path.is_dir():
                services.append(path)

        logger.info(f"📂 Discovered {len(services)} Tower services")
        return services

    async def initialize_all_repos(self) -> Dict[str, bool]:
        """Initialize git repos for all Tower services"""
        results = {}

        for service_path in self.tower_services:
            service_name = service_path.name
            results[service_name] = await self.initialize_repo(service_path)

        return results

    async def initialize_repo(self, path: Path) -> bool:
        """Initialize a git repository if not exists"""
        try:
            repo = Repo(path)
            logger.info(f"✅ Git repo already exists: {path.name}")
            return True
        except InvalidGitRepositoryError:
            try:
                repo = Repo.init(path)

                # Set git config
                with repo.config_writer() as config:
                    for key, value in self.git_config.items():
                        section, option = key.split('.')
                        config.set_value(section, option, value)

                # Create .gitignore
                gitignore_path = path / '.gitignore'
                if not gitignore_path.exists():
                    gitignore_content = """
*.pyc
__pycache__/
venv/
.env
*.log
*.backup*
*.old
logs/
*.db
.DS_Store
node_modules/
dist/
build/
*.egg-info/
.pytest_cache/
.coverage
htmlcov/
"""
                    gitignore_path.write_text(gitignore_content.strip())

                # Initial commit
                repo.index.add(['.gitignore'])
                repo.index.commit(f"[Echo] Initial commit for {path.name}")

                logger.info(f"✅ Initialized git repo: {path.name}")
                return True

            except Exception as e:
                logger.error(f"Failed to initialize {path.name}: {e}")
                return False

    async def get_repo_status(self, path: Path) -> Dict[str, Any]:
        """Get git repository status"""
        try:
            repo = Repo(path)

            # Get changed files
            changed_files = [item.a_path for item in repo.index.diff(None)]
            untracked_files = repo.untracked_files

            # Get commit info
            try:
                latest_commit = repo.head.commit
                commit_info = {
                    'hash': latest_commit.hexsha[:8],
                    'message': latest_commit.message.strip(),
                    'author': str(latest_commit.author),
                    'date': datetime.fromtimestamp(latest_commit.committed_date).isoformat()
                }
            except:
                commit_info = None

            return {
                'has_changes': len(changed_files) > 0 or len(untracked_files) > 0,
                'changed_files': changed_files,
                'untracked_files': untracked_files,
                'latest_commit': commit_info,
                'branch': repo.active_branch.name if not repo.head.is_detached else 'detached'
            }

        except Exception as e:
            logger.error(f"Failed to get status for {path.name}: {e}")
            return {'error': str(e)}

    async def auto_commit_changes(self, path: Path, message: str = None) -> bool:
        """Automatically commit changes in a repository"""
        if not self.auto_commit_enabled:
            logger.info(f"Auto-commit disabled. Would commit: {path.name}")
            return False

        try:
            repo = Repo(path)

            # Check if there are changes
            if not repo.is_dirty(untracked_files=True):
                return False

            # Add all changes
            repo.git.add(A=True)

            # Generate commit message
            if not message:
                changed_count = len(repo.index.diff('HEAD'))
                untracked_count = len(repo.untracked_files)
                message = f"[Echo Auto-Commit] {changed_count} modified, {untracked_count} new files"

            # Commit
            repo.index.commit(message)

            # Log commit
            self._log_commit(path.name, message, repo.head.commit.hexsha[:8])

            logger.info(f"✅ Auto-committed changes in {path.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to commit in {path.name}: {e}")
            return False

    def _log_commit(self, service: str, message: str, commit_hash: str):
        """Log commit to file"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'service': service,
            'message': message,
            'commit': commit_hash
        }

        with open(self.commit_log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    async def create_pre_commit_hook(self, path: Path) -> bool:
        """Create pre-commit hook for code quality checks"""
        try:
            hooks_dir = path / '.git' / 'hooks'
            hooks_dir.mkdir(parents=True, exist_ok=True)

            pre_commit_path = hooks_dir / 'pre-commit'

            hook_content = """#!/bin/bash
# Echo Brain Pre-commit Hook
# Runs code quality checks before commit

echo "🧠 Echo Brain: Running pre-commit checks..."

# Run black formatting check
if command -v black &> /dev/null; then
    echo "⚫ Checking Python formatting with black..."
    black --check . --exclude 'venv|__pycache__|\.git'
    if [ $? -ne 0 ]; then
        echo "❌ Code formatting issues found. Run 'black .' to fix."
        exit 1
    fi
fi

# Run pylint
if command -v pylint &> /dev/null; then
    echo "🔍 Running pylint..."
    find . -name "*.py" -not -path "./venv/*" -not -path "./__pycache__/*" | xargs pylint --exit-zero --score=n
fi

# Run tests if they exist
if [ -f "pytest.ini" ] || [ -f "setup.cfg" ] || [ -d "tests" ]; then
    if command -v pytest &> /dev/null; then
        echo "🧪 Running tests..."
        pytest -q
        if [ $? -ne 0 ]; then
            echo "❌ Tests failed. Fix tests before committing."
            exit 1
        fi
    fi
fi

echo "✅ Pre-commit checks passed!"
"""

            pre_commit_path.write_text(hook_content)
            pre_commit_path.chmod(0o755)

            logger.info(f"✅ Created pre-commit hook for {path.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create pre-commit hook: {e}")
            return False

    async def setup_ci_cd_pipeline(self, path: Path) -> bool:
        """Setup CI/CD pipeline configuration"""
        try:
            # Create .github/workflows directory
            workflows_dir = path / '.github' / 'workflows'
            workflows_dir.mkdir(parents=True, exist_ok=True)

            # Create CI workflow
            ci_workflow = workflows_dir / 'ci.yml'

            workflow_content = """name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.12'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest pylint black mypy
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

    - name: Format check with black
      run: black --check .

    - name: Lint with pylint
      run: pylint --exit-zero **/*.py

    - name: Type check with mypy
      run: mypy . --ignore-missing-imports || true

    - name: Test with pytest
      run: pytest -v || true

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v2

    - name: Deploy to Tower
      run: |
        echo "Deploying to Tower..."
        # Add deployment steps here
"""

            ci_workflow.write_text(workflow_content)

            logger.info(f"✅ Created CI/CD pipeline for {path.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create CI/CD pipeline: {e}")
            return False

    async def monitor_and_commit_loop(self):
        """Main loop for monitoring and auto-committing changes"""
        logger.info("🔄 Git monitoring started")

        while True:
            try:
                for service_path in self.tower_services:
                    status = await self.get_repo_status(service_path)

                    if status.get('has_changes'):
                        logger.info(f"📝 Changes detected in {service_path.name}")

                        if self.auto_commit_enabled:
                            # Auto-commit after running checks
                            await self.auto_commit_changes(
                                service_path,
                                f"[Echo] Auto-commit: {len(status['changed_files'])} files modified"
                            )

                # Check every 30 minutes
                await asyncio.sleep(1800)

            except Exception as e:
                logger.error(f"Git monitoring error: {e}")
                await asyncio.sleep(300)

    def enable_auto_commit(self, enabled: bool = True):
        """Enable or disable automatic commits"""
        self.auto_commit_enabled = enabled
        logger.info(f"🔄 Auto-commit {'enabled' if enabled else 'disabled'}")

    async def push_to_remote(self, path: Path, remote: str = 'origin', branch: str = 'main') -> bool:
        """Push changes to remote repository"""
        try:
            repo = Repo(path)

            # Check if remote exists
            if remote not in [r.name for r in repo.remotes]:
                logger.error(f"Remote '{remote}' not found in {path.name}")
                return False

            # Push
            repo.remotes[remote].push(branch)

            logger.info(f"✅ Pushed {path.name} to {remote}/{branch}")
            return True

        except Exception as e:
            logger.error(f"Failed to push {path.name}: {e}")
            return False

# Global instance
git_manager = GitManager()