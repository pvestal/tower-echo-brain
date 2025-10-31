#!/usr/bin/env python3
"""
Git Automation Utilities for Echo Brain
Autonomous git management for video generation projects and code changes
"""

import asyncio
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class GitAutomation:
    """Autonomous git operations for Echo Brain"""

    def __init__(self, repo_path: str = None):
        self.repo_path = repo_path or "/opt/tower-echo-brain"
        self.ensure_git_config()

    def ensure_git_config(self):
        """Ensure git is configured for autonomous operations"""
        try:
            # Check if global git config exists
            result = subprocess.run(['git', 'config', '--global', 'user.name'],
                                  capture_output=True, text=True)
            if result.returncode != 0:
                # Set up git config for Echo Brain
                subprocess.run(['git', 'config', '--global', 'user.name', 'Echo Brain'], check=True)
                subprocess.run(['git', 'config', '--global', 'user.email', 'echo@tower.local'], check=True)
                logger.info("✅ Git configuration set for Echo Brain")

        except subprocess.CalledProcessError as e:
            logger.warning(f"Git config setup failed: {e}")

    async def initialize_repo(self, repo_path: str = None) -> Dict[str, Any]:
        """Initialize git repository if not exists"""
        path = repo_path or self.repo_path

        try:
            if not os.path.exists(os.path.join(path, '.git')):
                result = subprocess.run(['git', 'init'], cwd=path,
                                      capture_output=True, text=True, check=True)

                # Create initial .gitignore
                gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt

# Logs
*.log
logs/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
*.backup*
temp/
tmp/
"""
                gitignore_path = os.path.join(path, '.gitignore')
                with open(gitignore_path, 'w') as f:
                    f.write(gitignore_content.strip())

                logger.info(f"✅ Initialized git repository at {path}")
                return {'success': True, 'action': 'initialized', 'path': path}
            else:
                logger.info(f"Git repository already exists at {path}")
                return {'success': True, 'action': 'exists', 'path': path}

        except Exception as e:
            logger.error(f"Failed to initialize git repository: {e}")
            return {'success': False, 'error': str(e)}

    async def commit_changes(self,
                           message: str,
                           repo_path: str = None,
                           add_all: bool = True) -> Dict[str, Any]:
        """Commit changes to git repository"""
        path = repo_path or self.repo_path

        try:
            # Check if repo exists
            if not os.path.exists(os.path.join(path, '.git')):
                logger.warning(f"No git repository at {path}, initializing...")
                await self.initialize_repo(path)

            # Add files
            if add_all:
                subprocess.run(['git', 'add', '.'], cwd=path, check=True)

            # Check if there are changes to commit
            result = subprocess.run(['git', 'status', '--porcelain'],
                                  cwd=path, capture_output=True, text=True)

            if not result.stdout.strip():
                logger.info("No changes to commit")
                return {'success': True, 'action': 'no_changes', 'message': 'No changes to commit'}

            # Commit with Echo Brain signature
            commit_message = f"{message}\n\n🤖 Committed by Echo Brain Autonomous System\nTimestamp: {datetime.now().isoformat()}"

            commit_result = subprocess.run([
                'git', 'commit', '-m', commit_message
            ], cwd=path, capture_output=True, text=True, check=True)

            logger.info(f"✅ Committed changes: {message}")
            return {
                'success': True,
                'action': 'committed',
                'message': message,
                'commit_hash': self._get_latest_commit_hash(path),
                'files_changed': self._count_changed_files(path)
            }

        except subprocess.CalledProcessError as e:
            logger.error(f"Git commit failed: {e}")
            return {'success': False, 'error': str(e), 'stderr': e.stderr}
        except Exception as e:
            logger.error(f"Git commit error: {e}")
            return {'success': False, 'error': str(e)}

    async def create_project_branch(self,
                                  branch_name: str,
                                  repo_path: str = None) -> Dict[str, Any]:
        """Create and switch to a new project branch"""
        path = repo_path or self.repo_path

        try:
            # Check if branch already exists
            result = subprocess.run(['git', 'branch', '--list', branch_name],
                                  cwd=path, capture_output=True, text=True)

            if branch_name in result.stdout:
                # Switch to existing branch
                subprocess.run(['git', 'checkout', branch_name], cwd=path, check=True)
                logger.info(f"Switched to existing branch: {branch_name}")
                return {'success': True, 'action': 'switched', 'branch': branch_name}
            else:
                # Create and switch to new branch
                subprocess.run(['git', 'checkout', '-b', branch_name], cwd=path, check=True)
                logger.info(f"Created and switched to new branch: {branch_name}")
                return {'success': True, 'action': 'created', 'branch': branch_name}

        except subprocess.CalledProcessError as e:
            logger.error(f"Branch operation failed: {e}")
            return {'success': False, 'error': str(e)}

    async def auto_commit_video_outputs(self,
                                      video_files: List[str],
                                      task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Automatically commit successful video generation outputs"""
        try:
            # Create video outputs directory in repo if not exists
            video_repo_dir = os.path.join(self.repo_path, 'generated_videos')
            os.makedirs(video_repo_dir, exist_ok=True)

            # Copy video files to repo (or create references)
            video_refs = []
            for video_file in video_files:
                if os.path.exists(video_file):
                    # Create a reference file instead of copying large video
                    ref_filename = f"{Path(video_file).stem}_ref.json"
                    ref_path = os.path.join(video_repo_dir, ref_filename)

                    ref_data = {
                        'original_path': video_file,
                        'file_size': os.path.getsize(video_file),
                        'created_at': datetime.now().isoformat(),
                        'task_info': task_info
                    }

                    with open(ref_path, 'w') as f:
                        import json
                        json.dump(ref_data, f, indent=2)

                    video_refs.append(ref_path)

            # Commit the references
            if video_refs:
                commit_message = f"Generated video outputs: {len(video_files)} files"
                result = await self.commit_changes(commit_message)

                if result['success']:
                    logger.info(f"✅ Committed video output references: {len(video_refs)} files")

                return result
            else:
                return {'success': True, 'action': 'no_files', 'message': 'No video files to commit'}

        except Exception as e:
            logger.error(f"Auto-commit video outputs failed: {e}")
            return {'success': False, 'error': str(e)}

    async def get_repo_status(self, repo_path: str = None) -> Dict[str, Any]:
        """Get current repository status"""
        path = repo_path or self.repo_path

        try:
            status = {}

            # Check if git repo exists
            if not os.path.exists(os.path.join(path, '.git')):
                return {'exists': False, 'path': path}

            # Get current branch
            branch_result = subprocess.run(['git', 'branch', '--show-current'],
                                         cwd=path, capture_output=True, text=True)
            status['current_branch'] = branch_result.stdout.strip()

            # Get status
            status_result = subprocess.run(['git', 'status', '--porcelain'],
                                         cwd=path, capture_output=True, text=True)
            status['has_changes'] = bool(status_result.stdout.strip())
            status['changed_files'] = len(status_result.stdout.strip().split('\n')) if status['has_changes'] else 0

            # Get latest commit
            commit_result = subprocess.run(['git', 'log', '-1', '--format=%H %s'],
                                         cwd=path, capture_output=True, text=True)
            if commit_result.returncode == 0:
                commit_info = commit_result.stdout.strip().split(' ', 1)
                status['latest_commit'] = {
                    'hash': commit_info[0][:8],
                    'message': commit_info[1] if len(commit_info) > 1 else ''
                }

            # Get commit count
            count_result = subprocess.run(['git', 'rev-list', '--count', 'HEAD'],
                                        cwd=path, capture_output=True, text=True)
            if count_result.returncode == 0:
                status['total_commits'] = int(count_result.stdout.strip())

            status['exists'] = True
            status['path'] = path

            return status

        except Exception as e:
            logger.error(f"Failed to get repo status: {e}")
            return {'exists': False, 'error': str(e), 'path': path}

    def _get_latest_commit_hash(self, repo_path: str) -> str:
        """Get the latest commit hash"""
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                  cwd=repo_path, capture_output=True, text=True)
            return result.stdout.strip()[:8]
        except:
            return "unknown"

    def _count_changed_files(self, repo_path: str) -> int:
        """Count files in the last commit"""
        try:
            result = subprocess.run(['git', 'diff', '--name-only', 'HEAD~1', 'HEAD'],
                                  cwd=repo_path, capture_output=True, text=True)
            return len([line for line in result.stdout.strip().split('\n') if line])
        except:
            return 0

class VideoProjectGitManager:
    """Specialized git manager for video generation projects"""

    def __init__(self):
        self.git = GitAutomation()
        self.projects_base = "/mnt/1TB-storage/ComfyUI/output/projects"
        os.makedirs(self.projects_base, exist_ok=True)

    async def create_video_project(self, project_name: str, description: str = "") -> Dict[str, Any]:
        """Create a new video generation project with git tracking"""
        try:
            # Create project directory
            project_path = os.path.join(self.projects_base, project_name)
            os.makedirs(project_path, exist_ok=True)

            # Initialize git repository
            init_result = await self.git.initialize_repo(project_path)
            if not init_result['success']:
                return init_result

            # Create project structure
            project_structure = {
                'README.md': f"""# {project_name}

{description}

## Project Structure
- `inputs/` - Source images and assets
- `outputs/` - Generated videos and results
- `workflows/` - ComfyUI workflow files
- `scripts/` - Generation scripts and configurations

## Generated by Echo Brain Autonomous System
Created: {datetime.now().isoformat()}
""",
                'project_config.json': {
                    'name': project_name,
                    'description': description,
                    'created_at': datetime.now().isoformat(),
                    'created_by': 'echo_brain_autonomous',
                    'type': 'video_generation'
                }
            }

            # Create directories
            for subdir in ['inputs', 'outputs', 'workflows', 'scripts']:
                os.makedirs(os.path.join(project_path, subdir), exist_ok=True)

            # Write files
            for filename, content in project_structure.items():
                file_path = os.path.join(project_path, filename)
                if isinstance(content, dict):
                    import json
                    with open(file_path, 'w') as f:
                        json.dump(content, f, indent=2)
                else:
                    with open(file_path, 'w') as f:
                        f.write(content)

            # Initial commit
            commit_result = await self.git.commit_changes(
                f"Initial commit for video project: {project_name}",
                repo_path=project_path
            )

            logger.info(f"✅ Created video project: {project_name}")
            return {
                'success': True,
                'project_name': project_name,
                'project_path': project_path,
                'git_init': init_result,
                'initial_commit': commit_result
            }

        except Exception as e:
            logger.error(f"Failed to create video project: {e}")
            return {'success': False, 'error': str(e)}

    async def track_generation_session(self,
                                     project_name: str,
                                     session_info: Dict[str, Any]) -> Dict[str, Any]:
        """Track a video generation session in git"""
        try:
            project_path = os.path.join(self.projects_base, project_name)

            if not os.path.exists(project_path):
                # Create project if it doesn't exist
                await self.create_video_project(project_name, "Auto-created for generation session")

            # Create session log
            session_file = os.path.join(project_path, 'scripts', f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

            with open(session_file, 'w') as f:
                import json
                json.dump(session_info, f, indent=2)

            # Commit session
            commit_result = await self.git.commit_changes(
                f"Video generation session: {session_info.get('task_name', 'unnamed')}",
                repo_path=project_path
            )

            return {
                'success': True,
                'session_file': session_file,
                'commit_result': commit_result
            }

        except Exception as e:
            logger.error(f"Failed to track generation session: {e}")
            return {'success': False, 'error': str(e)}