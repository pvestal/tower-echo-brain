#!/usr/bin/env python3
"""
Git Operations API for Echo Brain Autonomous System
Provides git automation endpoints for repository management
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

# Import git automation modules
from src.execution.git_operations import GitOperationsManager, GitHubOperations
from src.tasks.git_manager import git_manager
from src.tasks.github_integration import github_integration

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize git components
git_ops = GitOperationsManager()
gh_ops = GitHubOperations()

# Request/Response models
class GitStatusRequest(BaseModel):
    repository_path: Optional[str] = "/opt/tower-echo-brain"

class GitCommitRequest(BaseModel):
    repository_path: Optional[str] = "/opt/tower-echo-brain"
    files: Optional[List[str]] = None
    message: Optional[str] = None
    category: str = "update"

class GitBranchRequest(BaseModel):
    feature_name: str
    repository_path: Optional[str] = "/opt/tower-echo-brain"

class GitPullRequestRequest(BaseModel):
    title: Optional[str] = None
    body: Optional[str] = None
    base: str = "main"
    draft: bool = False
    repository_path: Optional[str] = "/opt/tower-echo-brain"

class TowerSyncRequest(BaseModel):
    enable_auto_commit: bool = False
    services: Optional[List[str]] = None

@router.get("/git/status")
async def get_git_status(repo_path: str = "/opt/tower-echo-brain"):
    """Get git repository status"""
    try:
        git_ops_local = GitOperationsManager(Path(repo_path))
        status = await git_ops_local.get_status()

        return {
            "success": True,
            "repository": repo_path,
            "status": {
                "branch": status.branch,
                "modified_files": status.modified_files,
                "untracked_files": status.untracked_files,
                "staged_files": status.staged_files,
                "ahead": status.ahead,
                "behind": status.behind,
                "has_conflicts": status.has_conflicts,
                "is_clean": status.is_clean
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting git status: {e}")
        raise HTTPException(status_code=500, detail=f"Git status error: {str(e)}")

@router.post("/git/commit")
async def smart_commit(request: GitCommitRequest):
    """Create a smart commit with auto-generated message"""
    try:
        git_ops_local = GitOperationsManager(Path(request.repository_path))

        success, result = await git_ops_local.smart_commit(
            files=request.files,
            message=request.message,
            category=request.category
        )

        return {
            "success": success,
            "repository": request.repository_path,
            "commit_hash": result if success else None,
            "error": result if not success else None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error creating commit: {e}")
        raise HTTPException(status_code=500, detail=f"Git commit error: {str(e)}")

@router.post("/git/branch")
async def create_feature_branch(request: GitBranchRequest):
    """Create a new feature branch"""
    try:
        git_ops_local = GitOperationsManager(Path(request.repository_path))

        success, branch_name = await git_ops_local.create_feature_branch(
            request.feature_name
        )

        return {
            "success": success,
            "repository": request.repository_path,
            "branch_name": branch_name if success else None,
            "error": branch_name if not success else None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error creating branch: {e}")
        raise HTTPException(status_code=500, detail=f"Git branch error: {str(e)}")

@router.post("/git/pr")
async def create_pull_request(request: GitPullRequestRequest):
    """Create a GitHub pull request"""
    try:
        git_ops_local = GitOperationsManager(Path(request.repository_path))

        success, pr_info = await git_ops_local.create_pull_request(
            title=request.title,
            body=request.body,
            base=request.base,
            draft=request.draft
        )

        return {
            "success": success,
            "repository": request.repository_path,
            "pr_info": {
                "number": pr_info.number,
                "title": pr_info.title,
                "url": pr_info.url,
                "branch": pr_info.branch,
                "state": pr_info.state
            } if pr_info else None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error creating PR: {e}")
        raise HTTPException(status_code=500, detail=f"Git PR error: {str(e)}")

@router.get("/git/tower/status")
async def get_tower_ecosystem_status():
    """Get git status for all Tower services"""
    try:
        services = git_manager._discover_tower_services()
        status_results = {}

        for service_path in services:
            try:
                status = await git_manager.get_repo_status(service_path)
                status_results[service_path.name] = {
                    "has_changes": status.get("has_changes", False),
                    "changed_files": len(status.get("changed_files", [])),
                    "untracked_files": len(status.get("untracked_files", [])),
                    "latest_commit": status.get("latest_commit", {}).get("hash", "None"),
                    "branch": status.get("branch", "unknown"),
                    "error": status.get("error")
                }
            except Exception as e:
                status_results[service_path.name] = {
                    "error": str(e),
                    "has_changes": False
                }

        return {
            "success": True,
            "total_services": len(services),
            "services": status_results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting Tower status: {e}")
        raise HTTPException(status_code=500, detail=f"Tower status error: {str(e)}")

@router.post("/git/tower/sync")
async def sync_tower_ecosystem(request: TowerSyncRequest, background_tasks: BackgroundTasks):
    """Sync all Tower repositories with intelligent commits"""
    try:
        # Enable/disable auto commit
        git_manager.enable_auto_commit(request.enable_auto_commit)

        services = git_manager._discover_tower_services()
        sync_results = {}

        # Filter services if specific ones requested
        if request.services:
            services = [s for s in services if s.name in request.services]

        for service_path in services:
            try:
                # Initialize repo if needed
                init_success = await git_manager.initialize_repo(service_path)

                # Get status
                status = await git_manager.get_repo_status(service_path)

                commit_result = None
                if status.get("has_changes") and request.enable_auto_commit:
                    # Auto-commit changes
                    commit_result = await git_manager.auto_commit_changes(
                        service_path,
                        f"[Echo Auto-Sync] {len(status.get('changed_files', []))} files updated"
                    )

                sync_results[service_path.name] = {
                    "initialized": init_success,
                    "has_changes": status.get("has_changes", False),
                    "committed": commit_result,
                    "status": "success"
                }

            except Exception as e:
                logger.error(f"Error syncing {service_path.name}: {e}")
                sync_results[service_path.name] = {
                    "status": "error",
                    "error": str(e)
                }

        return {
            "success": True,
            "auto_commit_enabled": request.enable_auto_commit,
            "services_synced": len(sync_results),
            "results": sync_results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error syncing Tower ecosystem: {e}")
        raise HTTPException(status_code=500, detail=f"Tower sync error: {str(e)}")

@router.get("/git/github/status")
async def get_github_status():
    """Get GitHub integration status and open PRs"""
    try:
        auth_ok = await github_integration.check_auth()
        open_prs = await github_integration.get_open_prs()
        current_branch = github_integration.get_current_branch()

        return {
            "success": True,
            "github_auth": auth_ok,
            "current_branch": current_branch,
            "open_prs": len(open_prs),
            "pr_details": open_prs,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting GitHub status: {e}")
        raise HTTPException(status_code=500, detail=f"GitHub status error: {str(e)}")

@router.post("/git/automation/enable")
async def enable_git_automation(background_tasks: BackgroundTasks):
    """Enable autonomous git monitoring and commits"""
    try:
        git_manager.enable_auto_commit(True)

        # Start monitoring loop in background
        background_tasks.add_task(git_manager.monitor_and_commit_loop)

        return {
            "success": True,
            "status": "Git automation enabled",
            "monitoring": "Started background monitoring",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error enabling git automation: {e}")
        raise HTTPException(status_code=500, detail=f"Git automation error: {str(e)}")

@router.post("/git/automation/disable")
async def disable_git_automation():
    """Disable autonomous git commits"""
    try:
        git_manager.enable_auto_commit(False)

        return {
            "success": True,
            "status": "Git automation disabled",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error disabling git automation: {e}")
        raise HTTPException(status_code=500, detail=f"Git automation error: {str(e)}")

@router.get("/git/health")
async def git_health_check():
    """Health check for git automation system"""
    try:
        # Test git operations
        status = await git_ops.get_status()

        # Test GitHub integration
        gh_auth = await github_integration.check_auth()

        # Test Tower discovery
        services = git_manager._discover_tower_services()

        return {
            "success": True,
            "git_operations": "healthy",
            "github_auth": "authenticated" if gh_auth else "not_authenticated",
            "tower_services": len(services),
            "auto_commit_enabled": git_manager.auto_commit_enabled,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Git health check error: {e}")
        raise HTTPException(status_code=500, detail=f"Git health error: {str(e)}")

# Autonomous workflow endpoints
@router.post("/git/autonomous/quality-pr")
async def create_autonomous_quality_pr():
    """Create automated quality improvement PR"""
    try:
        # This would integrate with code quality analysis
        # For now, return placeholder
        return {
            "success": True,
            "status": "Quality analysis not yet implemented",
            "message": "Would analyze code quality and create PR if improvements found",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error creating quality PR: {e}")
        raise HTTPException(status_code=500, detail=f"Quality PR error: {str(e)}")

@router.get("/git/logs")
async def get_git_automation_logs():
    """Get git automation activity logs"""
    try:
        log_path = git_manager.commit_log_path

        if not log_path.exists():
            return {
                "success": True,
                "logs": [],
                "message": "No logs found"
            }

        # Read last 50 log entries
        logs = []
        with open(log_path, 'r') as f:
            lines = f.readlines()
            for line in lines[-50:]:
                try:
                    logs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        return {
            "success": True,
            "logs": logs,
            "total_entries": len(logs),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error reading git logs: {e}")
        raise HTTPException(status_code=500, detail=f"Git logs error: {str(e)}")