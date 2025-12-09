#!/usr/bin/env python3
"""
Echo GitHub Integration
Handles GitHub webhooks and integrates with GitHub API
Provides seamless CI/CD pipeline triggers from GitHub events
"""

import os
import json
import hmac
import hashlib
import asyncio
import aiohttp
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import base64
from urllib.parse import urlparse

# Import our pipeline service
from echo_ci_cd_pipeline import EchoCICDPipeline, PipelineConfig

@dataclass
class GitHubRepository:
    """GitHub repository information"""
    owner: str
    name: str
    full_name: str
    clone_url: str
    default_branch: str
    webhook_secret: Optional[str] = None

@dataclass
class WebhookEvent:
    """GitHub webhook event data"""
    event_type: str
    repository: GitHubRepository
    ref: str
    before_commit: str
    after_commit: str
    commits: List[Dict[str, Any]]
    pusher: Dict[str, Any]
    timestamp: datetime
    raw_payload: Dict[str, Any]

class GitHubAPIClient:
    """GitHub API client for repository operations"""
    
    def __init__(self, token: str = None):
        self.token = token or os.getenv('GITHUB_TOKEN')
        self.base_url = 'https://api.github.com'
        self.headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'EchoCI/1.0'
        }
        
        if self.token:
            self.headers['Authorization'] = f'token {self.token}'
    
    async def get_repository(self, owner: str, repo: str) -> Optional[Dict[str, Any]]:
        """Get repository information"""
        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(f'{self.base_url}/repos/{owner}/{repo}') as resp:
                    if resp.status == 200:
                        return await resp.json()
                    return None
        except Exception as e:
            print(f"Error fetching repository {owner}/{repo}: {e}")
            return None
    
    async def get_commit(self, owner: str, repo: str, commit_sha: str) -> Optional[Dict[str, Any]]:
        """Get commit information"""
        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(f'{self.base_url}/repos/{owner}/{repo}/commits/{commit_sha}') as resp:
                    if resp.status == 200:
                        return await resp.json()
                    return None
        except Exception as e:
            print(f"Error fetching commit {commit_sha}: {e}")
            return None
    
    async def create_status(self, owner: str, repo: str, commit_sha: str, 
                          state: str, description: str, context: str = "echo-ci",
                          target_url: str = None) -> bool:
        """Create commit status"""
        try:
            payload = {
                'state': state,  # pending, success, error, failure
                'description': description,
                'context': context
            }
            
            if target_url:
                payload['target_url'] = target_url
            
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.post(
                    f'{self.base_url}/repos/{owner}/{repo}/statuses/{commit_sha}',
                    json=payload
                ) as resp:
                    return resp.status == 201
        except Exception as e:
            print(f"Error creating status for {commit_sha}: {e}")
            return False
    
    async def create_comment(self, owner: str, repo: str, commit_sha: str, body: str) -> bool:
        """Create commit comment"""
        try:
            payload = {'body': body}
            
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.post(
                    f'{self.base_url}/repos/{owner}/{repo}/commits/{commit_sha}/comments',
                    json=payload
                ) as resp:
                    return resp.status == 201
        except Exception as e:
            print(f"Error creating comment for {commit_sha}: {e}")
            return False
    
    async def get_pull_request(self, owner: str, repo: str, pr_number: int) -> Optional[Dict[str, Any]]:
        """Get pull request information"""
        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(f'{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}') as resp:
                    if resp.status == 200:
                        return await resp.json()
                    return None
        except Exception as e:
            print(f"Error fetching PR {pr_number}: {e}")
            return None
    
    async def create_pr_comment(self, owner: str, repo: str, pr_number: int, body: str) -> bool:
        """Create pull request comment"""
        try:
            payload = {'body': body}
            
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.post(
                    f'{self.base_url}/repos/{owner}/{repo}/issues/{pr_number}/comments',
                    json=payload
                ) as resp:
                    return resp.status == 201
        except Exception as e:
            print(f"Error creating PR comment for {pr_number}: {e}")
            return False

class WebhookValidator:
    """Validates GitHub webhook signatures"""
    
    @staticmethod
    def validate_signature(payload: bytes, signature: str, secret: str) -> bool:
        """Validate GitHub webhook signature"""
        if not signature or not secret:
            return False
        
        try:
            # Remove 'sha256=' prefix if present
            if signature.startswith('sha256='):
                signature = signature[7:]
            
            # Calculate expected signature
            expected_signature = hmac.new(
                secret.encode('utf-8'),
                payload,
                hashlib.sha256
            ).hexdigest()
            
            # Compare signatures securely
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception:
            return False

class WebhookProcessor:
    """Processes GitHub webhook events"""
    
    def __init__(self, pipeline_service: EchoCICDPipeline, github_client: GitHubAPIClient):
        self.pipeline_service = pipeline_service
        self.github_client = github_client
        self.supported_events = ['push', 'pull_request', 'release']
    
    async def process_webhook(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming webhook event"""
        try:
            if event_type not in self.supported_events:
                return {'status': 'ignored', 'reason': f'Event type {event_type} not supported'}
            
            if event_type == 'push':
                return await self._process_push_event(payload)
            elif event_type == 'pull_request':
                return await self._process_pull_request_event(payload)
            elif event_type == 'release':
                return await self._process_release_event(payload)
            
            return {'status': 'error', 'reason': 'Unknown event type'}
            
        except Exception as e:
            return {'status': 'error', 'reason': str(e)}
    
    async def _process_push_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process push event"""
        repository = payload['repository']
        ref = payload['ref']
        commits = payload['commits']
        
        # Only process pushes to main/master branches
        if not ref.endswith('/main') and not ref.endswith('/master'):
            return {'status': 'ignored', 'reason': f'Push to non-main branch: {ref}'}
        
        # Skip if no commits
        if not commits:
            return {'status': 'ignored', 'reason': 'No commits in push'}
        
        # Get the latest commit
        head_commit = payload['head_commit']
        commit_sha = head_commit['id']
        
        # Create pending status
        await self.github_client.create_status(
            repository['owner']['login'],
            repository['name'],
            commit_sha,
            'pending',
            'Echo CI/CD pipeline started',
            'echo-ci'
        )
        
        # Trigger pipeline
        config = PipelineConfig(
            pipeline_id="",
            app_name=repository['name'],
            git_repo=repository['clone_url'],
            git_branch=ref.split('/')[-1],
            git_commit=commit_sha,
            trigger_type='webhook',
            target_environments=['staging'],  # Push to staging first
            run_tests=True,
            deploy_on_success=True,
            notify_on_completion=True
        )
        
        pipeline_id = await self.pipeline_service.trigger_pipeline(config)
        
        # Monitor pipeline and update status
        asyncio.create_task(self._monitor_pipeline_status(
            pipeline_id, repository['owner']['login'], repository['name'], commit_sha
        ))
        
        return {
            'status': 'processed',
            'pipeline_id': pipeline_id,
            'commit': commit_sha,
            'repository': repository['full_name']
        }
    
    async def _process_pull_request_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process pull request event"""
        action = payload['action']
        pull_request = payload['pull_request']
        repository = payload['repository']
        
        # Only process opened and synchronize (new commits) actions
        if action not in ['opened', 'synchronize']:
            return {'status': 'ignored', 'reason': f'PR action {action} not processed'}
        
        commit_sha = pull_request['head']['sha']
        branch = pull_request['head']['ref']
        
        # Create pending status
        await self.github_client.create_status(
            repository['owner']['login'],
            repository['name'],
            commit_sha,
            'pending',
            'Echo CI/CD pipeline started for PR',
            'echo-ci/pr'
        )
        
        # Trigger pipeline for PR (test only, no deployment)
        config = PipelineConfig(
            pipeline_id="",
            app_name=repository['name'],
            git_repo=repository['clone_url'],
            git_branch=branch,
            git_commit=commit_sha,
            trigger_type='webhook',
            target_environments=[],  # No deployment for PRs
            run_tests=True,
            deploy_on_success=False,
            notify_on_completion=False
        )
        
        pipeline_id = await self.pipeline_service.trigger_pipeline(config)
        
        # Monitor pipeline and update PR
        asyncio.create_task(self._monitor_pr_pipeline_status(
            pipeline_id, repository['owner']['login'], repository['name'], 
            commit_sha, pull_request['number']
        ))
        
        return {
            'status': 'processed',
            'pipeline_id': pipeline_id,
            'pr_number': pull_request['number'],
            'commit': commit_sha
        }
    
    async def _process_release_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process release event"""
        action = payload['action']
        release = payload['release']
        repository = payload['repository']
        
        if action != 'published':
            return {'status': 'ignored', 'reason': f'Release action {action} not processed'}
        
        tag_name = release['tag_name']
        target_commitish = release['target_commitish']
        
        # Trigger production deployment
        config = PipelineConfig(
            pipeline_id="",
            app_name=repository['name'],
            git_repo=repository['clone_url'],
            git_branch=target_commitish,
            git_commit=target_commitish,  # Will be resolved to actual commit
            trigger_type='webhook',
            target_environments=['production'],
            run_tests=True,
            deploy_on_success=True,
            notify_on_completion=True
        )
        
        pipeline_id = await self.pipeline_service.trigger_pipeline(config)
        
        return {
            'status': 'processed',
            'pipeline_id': pipeline_id,
            'release': tag_name,
            'target': target_commitish
        }
    
    async def _monitor_pipeline_status(self, pipeline_id: str, owner: str, repo: str, commit_sha: str):
        """Monitor pipeline and update GitHub status"""
        while True:
            try:
                status = await self.pipeline_service.get_pipeline_status(pipeline_id)
                if not status:
                    break
                
                pipeline_status = status['status']
                
                if pipeline_status == 'success':
                    await self.github_client.create_status(
                        owner, repo, commit_sha, 'success',
                        'Echo CI/CD pipeline completed successfully', 'echo-ci'
                    )
                    break
                elif pipeline_status == 'failed':
                    await self.github_client.create_status(
                        owner, repo, commit_sha, 'failure',
                        'Echo CI/CD pipeline failed', 'echo-ci'
                    )
                    
                    # Add detailed comment
                    await self._create_failure_comment(owner, repo, commit_sha, status)
                    break
                elif pipeline_status in ['cancelled']:
                    await self.github_client.create_status(
                        owner, repo, commit_sha, 'error',
                        'Echo CI/CD pipeline was cancelled', 'echo-ci'
                    )
                    break
                
                # Wait before checking again
                await asyncio.sleep(30)
                
            except Exception as e:
                print(f"Error monitoring pipeline {pipeline_id}: {e}")
                break
    
    async def _monitor_pr_pipeline_status(self, pipeline_id: str, owner: str, repo: str, 
                                        commit_sha: str, pr_number: int):
        """Monitor pipeline for PR and update status"""
        while True:
            try:
                status = await self.pipeline_service.get_pipeline_status(pipeline_id)
                if not status:
                    break
                
                pipeline_status = status['status']
                
                if pipeline_status == 'success':
                    await self.github_client.create_status(
                        owner, repo, commit_sha, 'success',
                        'Echo CI/CD tests passed', 'echo-ci/pr'
                    )
                    
                    # Add success comment to PR
                    await self._create_pr_success_comment(owner, repo, pr_number, status)
                    break
                    
                elif pipeline_status == 'failed':
                    await self.github_client.create_status(
                        owner, repo, commit_sha, 'failure',
                        'Echo CI/CD tests failed', 'echo-ci/pr'
                    )
                    
                    # Add detailed failure comment to PR
                    await self._create_pr_failure_comment(owner, repo, pr_number, status)
                    break
                
                # Wait before checking again
                await asyncio.sleep(30)
                
            except Exception as e:
                print(f"Error monitoring PR pipeline {pipeline_id}: {e}")
                break
    
    async def _create_failure_comment(self, owner: str, repo: str, commit_sha: str, status: Dict[str, Any]):
        """Create detailed failure comment"""
        test_results = status.get('test_results', {})
        stages = status.get('stages', {})
        
        comment = "## 🚫 Echo CI/CD Pipeline Failed\n\n"
        comment += f"**Pipeline ID:** \n"
        comment += f"**Commit:** \n\n"
        
        if test_results:
            comment += "### Test Results\n"
            comment += f"- Total Tests: {test_results.get('total_tests', 0)}\n"
            comment += f"- Passed: {test_results.get('passed_tests', 0)}\n"
            comment += f"- Failed: {test_results.get('failed_tests', 0)}\n"
            comment += f"- Coverage: {test_results.get('average_coverage', 0):.1f}%\n\n"
        
        # Add failed stages
        failed_stages = [stage for stage, info in stages.items() if info.get('status') == 'failed']
        if failed_stages:
            comment += "### Failed Stages\n"
            for stage in failed_stages:
                stage_info = stages[stage]
                comment += f"- **{stage}**: {stage_info.get('error', 'Unknown error')}\n"
        
        await self.github_client.create_comment(owner, repo, commit_sha, comment)
    
    async def _create_pr_success_comment(self, owner: str, repo: str, pr_number: int, status: Dict[str, Any]):
        """Create PR success comment"""
        test_results = status.get('test_results', {})
        
        comment = "## ✅ Echo CI/CD Tests Passed\n\n"
        comment += f"**Pipeline ID:** \n\n"
        
        if test_results:
            comment += "### Test Results\n"
            comment += f"- Total Tests: {test_results.get('total_tests', 0)}\n"
            comment += f"- All tests passed! 🎉\n"
            comment += f"- Coverage: {test_results.get('average_coverage', 0):.1f}%\n\n"
        
        comment += "This PR is ready for review and merge."
        
        await self.github_client.create_pr_comment(owner, repo, pr_number, comment)
    
    async def _create_pr_failure_comment(self, owner: str, repo: str, pr_number: int, status: Dict[str, Any]):
        """Create PR failure comment"""
        test_results = status.get('test_results', {})
        stages = status.get('stages', {})
        
        comment = "## ❌ Echo CI/CD Tests Failed\n\n"
        comment += f"**Pipeline ID:** \n\n"
        
        if test_results:
            comment += "### Test Results\n"
            comment += f"- Total Tests: {test_results.get('total_tests', 0)}\n"
            comment += f"- Passed: {test_results.get('passed_tests', 0)}\n"
            comment += f"- Failed: {test_results.get('failed_tests', 0)}\n"
            comment += f"- Coverage: {test_results.get('average_coverage', 0):.1f}%\n\n"
        
        # Add failed stages
        failed_stages = [stage for stage, info in stages.items() if info.get('status') == 'failed']
        if failed_stages:
            comment += "### Issues Found\n"
            for stage in failed_stages:
                stage_info = stages[stage]
                comment += f"- **{stage}**: {stage_info.get('error', 'Unknown error')}\n"
        
        comment += "\nPlease fix the issues and push new commits to retry the tests."
        
        await self.github_client.create_pr_comment(owner, repo, pr_number, comment)

class GitHubIntegrationService:
    """Main GitHub integration service"""
    
    def __init__(self):
        self.pipeline_service = EchoCICDPipeline()
        self.github_client = GitHubAPIClient()
        self.webhook_processor = WebhookProcessor(self.pipeline_service, self.github_client)
        self.validator = WebhookValidator()
        self.db_path = '/opt/tower-echo-brain/data/github_integrations.db'
        self._init_database()
    
    def _init_database(self):
        """Initialize GitHub integration database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS repositories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner TEXT,
                    name TEXT,
                    full_name TEXT UNIQUE,
                    clone_url TEXT,
                    default_branch TEXT,
                    webhook_secret TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS webhook_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT,
                    repository_full_name TEXT,
                    commit_sha TEXT,
                    pipeline_id TEXT,
                    status TEXT,
                    payload TEXT,
                    processed_at TEXT
                )
            ''')
    
    async def handle_webhook(self, event_type: str, payload: Dict[str, Any], 
                           signature: str = None) -> Dict[str, Any]:
        """Handle incoming GitHub webhook"""
        try:
            # Validate webhook signature if secret is configured
            repository_full_name = payload.get('repository', {}).get('full_name')
            if repository_full_name:
                repo_secret = await self._get_webhook_secret(repository_full_name)
                if repo_secret and signature:
                    payload_bytes = json.dumps(payload, separators=(',', ':')).encode('utf-8')
                    if not self.validator.validate_signature(payload_bytes, signature, repo_secret):
                        return {'status': 'error', 'reason': 'Invalid webhook signature'}
            
            # Process webhook
            result = await self.webhook_processor.process_webhook(event_type, payload)
            
            # Save webhook event
            await self._save_webhook_event(event_type, payload, result)
            
            return result
            
        except Exception as e:
            return {'status': 'error', 'reason': str(e)}
    
    async def register_repository(self, owner: str, name: str, webhook_secret: str = None) -> bool:
        """Register a repository for GitHub integration"""
        try:
            # Get repository info from GitHub
            repo_info = await self.github_client.get_repository(owner, name)
            if not repo_info:
                return False
            
            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO repositories 
                    (owner, name, full_name, clone_url, default_branch, webhook_secret, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    owner,
                    name,
                    repo_info['full_name'],
                    repo_info['clone_url'],
                    repo_info['default_branch'],
                    webhook_secret,
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
            
            return True
            
        except Exception as e:
            print(f"Error registering repository {owner}/{name}: {e}")
            return False
    
    async def list_repositories(self) -> List[Dict[str, Any]]:
        """List registered repositories"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT * FROM repositories ORDER BY created_at DESC')
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    async def get_webhook_events(self, repository_full_name: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get webhook events"""
        with sqlite3.connect(self.db_path) as conn:
            if repository_full_name:
                cursor = conn.execute('''
                    SELECT * FROM webhook_events 
                    WHERE repository_full_name = ?
                    ORDER BY processed_at DESC LIMIT ?
                ''', (repository_full_name, limit))
            else:
                cursor = conn.execute('''
                    SELECT * FROM webhook_events 
                    ORDER BY processed_at DESC LIMIT ?
                ''', (limit,))
            
            columns = [description[0] for description in cursor.description]
            events = []
            
            for row in cursor.fetchall():
                event_data = dict(zip(columns, row))
                if event_data['payload']:
                    event_data['payload'] = json.loads(event_data['payload'])
                events.append(event_data)
            
            return events
    
    async def _get_webhook_secret(self, repository_full_name: str) -> Optional[str]:
        """Get webhook secret for repository"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT webhook_secret FROM repositories WHERE full_name = ?',
                (repository_full_name,)
            )
            row = cursor.fetchone()
            return row[0] if row else None
    
    async def _save_webhook_event(self, event_type: str, payload: Dict[str, Any], result: Dict[str, Any]):
        """Save webhook event to database"""
        repository_full_name = payload.get('repository', {}).get('full_name')
        commit_sha = None
        pipeline_id = result.get('pipeline_id')
        
        # Extract commit SHA based on event type
        if event_type == 'push':
            commit_sha = payload.get('head_commit', {}).get('id')
        elif event_type == 'pull_request':
            commit_sha = payload.get('pull_request', {}).get('head', {}).get('sha')
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO webhook_events 
                (event_type, repository_full_name, commit_sha, pipeline_id, status, payload, processed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                event_type,
                repository_full_name,
                commit_sha,
                pipeline_id,
                result.get('status'),
                json.dumps(payload),
                datetime.now().isoformat()
            ))

# REST API for GitHub integration
from fastapi import FastAPI, HTTPException, Request, Header
from pydantic import BaseModel

app = FastAPI(title="Echo GitHub Integration", version="1.0.0")
github_service = GitHubIntegrationService()

class RepositoryRegistration(BaseModel):
    owner: str
    name: str
    webhook_secret: Optional[str] = None

@app.post("/webhook")
async def github_webhook(request: Request, x_github_event: str = Header(None), 
                        x_hub_signature_256: str = Header(None)):
    """Handle GitHub webhook"""
    try:
        payload = await request.json()
        result = await github_service.handle_webhook(
            x_github_event, payload, x_hub_signature_256
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/repositories")
async def register_repository(repo: RepositoryRegistration):
    """Register a repository for GitHub integration"""
    success = await github_service.register_repository(
        repo.owner, repo.name, repo.webhook_secret
    )
    if not success:
        raise HTTPException(status_code=400, detail="Failed to register repository")
    return {"status": "registered", "repository": f"{repo.owner}/{repo.name}"}

@app.get("/api/repositories")
async def list_repositories():
    """List registered repositories"""
    return await github_service.list_repositories()

@app.get("/api/webhook-events")
async def get_webhook_events(repository: Optional[str] = None, limit: int = 100):
    """Get webhook events"""
    return await github_service.get_webhook_events(repository, limit)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "echo_github_integration", 
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8343)
