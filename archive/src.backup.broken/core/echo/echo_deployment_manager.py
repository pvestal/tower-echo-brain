#!/usr/bin/env python3
"""
Echo Deployment Manager
Handles blue/green deployments with automatic rollback capabilities
Integrates with Echo's CI/CD pipeline for seamless deployments
"""

import os
import json
import shutil
import asyncio
import sqlite3
import subprocess
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import docker
import psutil

class DeploymentStatus(Enum):
    PENDING = "pending"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    TESTING = "testing"
    ACTIVE = "active"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"

class EnvironmentType(Enum):
    BLUE = "blue"
    GREEN = "green"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class DeploymentConfig:
    """Configuration for a deployment"""
    app_name: str
    version: str
    git_commit: str
    source_path: str
    target_env: EnvironmentType
    port: int
    health_check_url: str
    dependencies: List[str]
    environment_vars: Dict[str, str]
    rollback_enabled: bool = True
    health_check_timeout: int = 60
    deployment_timeout: int = 300

@dataclass
class Deployment:
    """Represents a deployment instance"""
    id: str
    config: DeploymentConfig
    status: DeploymentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    logs: List[str] = None
    health_status: bool = False
    previous_deployment_id: Optional[str] = None
    
    def __post_init__(self):
        if self.logs is None:
            self.logs = []

class HealthChecker:
    """Performs health checks on deployed services"""
    
    def __init__(self):
        self.timeout = 30
        self.retry_count = 3
        self.retry_delay = 5
    
    async def check_health(self, url: str, expected_status: int = 200) -> Tuple[bool, str]:
        """Check if a service is healthy"""
        for attempt in range(self.retry_count):
            try:
                response = requests.get(url, timeout=self.timeout)
                if response.status_code == expected_status:
                    return True, f"Health check passed (status: {response.status_code})"
                else:
                    return False, f"Health check failed (status: {response.status_code})"
            except requests.exceptions.RequestException as e:
                if attempt == self.retry_count - 1:
                    return False, f"Health check failed after {self.retry_count} attempts: {str(e)}"
                await asyncio.sleep(self.retry_delay)
        
        return False, "Health check failed"
    
    async def check_service_health(self, service_name: str, port: int) -> Tuple[bool, str]:
        """Check if a systemd service is healthy"""
        try:
            # Check systemd service status
            result = subprocess.run([
                'systemctl', '--user', 'is-active', service_name
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                return False, f"Service {service_name} is not active"
            
            # Check port availability
            for proc in psutil.net_connections():
                if proc.laddr.port == port and proc.status == 'LISTEN':
                    return True, f"Service {service_name} is healthy and listening on port {port}"
            
            return False, f"Service {service_name} is active but not listening on port {port}"
            
        except Exception as e:
            return False, f"Error checking service health: {str(e)}"

class ServiceManager:
    """Manages systemd services for deployments"""
    
    def __init__(self):
        self.service_dir = Path("/home/{os.getenv("TOWER_USER", "patrick")}/.config/systemd/user")
        self.service_dir.mkdir(parents=True, exist_ok=True)
    
    def create_service(self, deployment: Deployment) -> str:
        """Create systemd service file for deployment"""
        config = deployment.config
        service_name = f"{config.app_name}-{config.target_env.value}.service"
        service_path = self.service_dir / service_name
        
        # Environment variables
        env_vars = "\n".join([f"Environment={k}={v}" for k, v in config.environment_vars.items()])
        
        service_content = f"""[Unit]
Description={config.app_name} ({config.target_env.value}) - Version {config.version}
After=network.target

[Service]
Type=simple
User=patrick
WorkingDirectory={config.source_path}
ExecStart=/opt/tower-echo-brain/venv/bin/python -m src.main
Restart=always
RestartSec=10
{env_vars}
Environment=PORT={config.port}
Environment=ENV={config.target_env.value}

[Install]
WantedBy=default.target
"""
        
        with open(service_path, 'w') as f:
            f.write(service_content)
        
        return service_name
    
    async def start_service(self, service_name: str) -> Tuple[bool, str]:
        """Start a systemd service"""
        try:
            # Reload systemd daemon
            subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
            
            # Start service
            result = subprocess.run([
                'systemctl', '--user', 'start', service_name
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                return True, f"Service {service_name} started successfully"
            else:
                return False, f"Failed to start service {service_name}: {result.stderr}"
                
        except Exception as e:
            return False, f"Error starting service: {str(e)}"
    
    async def stop_service(self, service_name: str) -> Tuple[bool, str]:
        """Stop a systemd service"""
        try:
            result = subprocess.run([
                'systemctl', '--user', 'stop', service_name
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                return True, f"Service {service_name} stopped successfully"
            else:
                return False, f"Failed to stop service {service_name}: {result.stderr}"
                
        except Exception as e:
            return False, f"Error stopping service: {str(e)}"
    
    def remove_service(self, service_name: str) -> bool:
        """Remove systemd service file"""
        try:
            service_path = self.service_dir / service_name
            if service_path.exists():
                service_path.unlink()
                subprocess.run(['systemctl', '--user', 'daemon-reload'])
                return True
            return True
        except Exception:
            return False

class DeploymentExecutor:
    """Executes deployments with blue/green strategy"""
    
    def __init__(self):
        self.health_checker = HealthChecker()
        self.service_manager = ServiceManager()
        self.base_deployment_dir = Path("/opt/tower-echo-brain/deployments")
        self.base_deployment_dir.mkdir(exist_ok=True)
    
    async def execute_deployment(self, deployment: Deployment) -> Deployment:
        """Execute a blue/green deployment"""
        try:
            deployment.status = DeploymentStatus.DEPLOYING
            deployment.logs.append(f"Starting deployment {deployment.id}")
            
            # Step 1: Prepare deployment directory
            await self._prepare_deployment_environment(deployment)
            
            # Step 2: Deploy to inactive environment
            await self._deploy_to_environment(deployment)
            
            # Step 3: Health check
            deployment.status = DeploymentStatus.TESTING
            await self._perform_health_checks(deployment)
            
            # Step 4: Switch traffic (activate)
            await self._activate_deployment(deployment)
            
            deployment.status = DeploymentStatus.ACTIVE
            deployment.end_time = datetime.now()
            deployment.logs.append(f"Deployment {deployment.id} completed successfully")
            
            return deployment
            
        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            deployment.logs.append(f"Deployment failed: {str(e)}")
            
            if deployment.config.rollback_enabled:
                await self._rollback_deployment(deployment)
            
            return deployment
    
    async def _prepare_deployment_environment(self, deployment: Deployment):
        """Prepare the deployment environment"""
        config = deployment.config
        deployment_dir = self.base_deployment_dir / deployment.id
        
        deployment.logs.append("Preparing deployment environment")
        
        # Create deployment directory
        deployment_dir.mkdir(exist_ok=True)
        
        # Copy source code
        if os.path.exists(config.source_path):
            shutil.copytree(
                config.source_path,
                deployment_dir / "app",
                dirs_exist_ok=True
            )
        else:
            raise Exception(f"Source path {config.source_path} does not exist")
        
        # Update configuration for this deployment
        await self._update_deployment_config(deployment, deployment_dir)
    
    async def _deploy_to_environment(self, deployment: Deployment):
        """Deploy to the target environment"""
        deployment.logs.append(f"Deploying to {deployment.config.target_env.value} environment")
        
        # Create and start service
        service_name = self.service_manager.create_service(deployment)
        deployment.logs.append(f"Created service: {service_name}")
        
        success, message = await self.service_manager.start_service(service_name)
        deployment.logs.append(f"Service start result: {message}")
        
        if not success:
            raise Exception(f"Failed to start service: {message}")
        
        # Wait for service to initialize
        await asyncio.sleep(10)
    
    async def _perform_health_checks(self, deployment: Deployment):
        """Perform comprehensive health checks"""
        config = deployment.config
        deployment.logs.append("Performing health checks")
        
        # Wait for service startup
        await asyncio.sleep(5)
        
        # Check service health
        service_name = f"{config.app_name}-{config.target_env.value}.service"
        health_ok, health_msg = await self.health_checker.check_service_health(
            service_name, config.port
        )
        deployment.logs.append(f"Service health: {health_msg}")
        
        if not health_ok:
            raise Exception(f"Service health check failed: {health_msg}")
        
        # Check HTTP endpoint if available
        if config.health_check_url:
            url = config.health_check_url.replace('{port}', str(config.port))
            http_ok, http_msg = await self.health_checker.check_health(url)
            deployment.logs.append(f"HTTP health: {http_msg}")
            
            if not http_ok:
                raise Exception(f"HTTP health check failed: {http_msg}")
        
        deployment.health_status = True
    
    async def _activate_deployment(self, deployment: Deployment):
        """Activate the deployment (switch traffic)"""
        deployment.logs.append("Activating deployment")
        
        # In a real blue/green setup, this would update load balancer
        # For our setup, we'll update nginx configuration
        await self._update_nginx_config(deployment)
        
        # Stop previous deployment if it exists
        if deployment.previous_deployment_id:
            await self._deactivate_previous_deployment(deployment)
    
    async def _update_nginx_config(self, deployment: Deployment):
        """Update nginx configuration for new deployment"""
        config = deployment.config
        
        # Update nginx upstream for this service
        nginx_config = f"""
upstream {config.app_name}_backend {{
    server 127.0.0.1:{config.port};
}}
"""
        
        nginx_dir = Path("/etc/nginx/sites-available")
        if nginx_dir.exists():
            config_file = nginx_dir / f"{config.app_name}.conf"
            with open(config_file, 'w') as f:
                f.write(nginx_config)
            
            # Reload nginx
            subprocess.run(['sudo', 'nginx', '-s', 'reload'])
            deployment.logs.append("Updated nginx configuration")
    
    async def _deactivate_previous_deployment(self, deployment: Deployment):
        """Deactivate previous deployment"""
        # This would stop the previous version's service
        prev_service = f"{deployment.config.app_name}-previous.service"
        await self.service_manager.stop_service(prev_service)
        deployment.logs.append("Deactivated previous deployment")
    
    async def _rollback_deployment(self, deployment: Deployment):
        """Rollback failed deployment"""
        deployment.status = DeploymentStatus.ROLLING_BACK
        deployment.logs.append("Starting rollback procedure")
        
        try:
            # Stop current service
            service_name = f"{deployment.config.app_name}-{deployment.config.target_env.value}.service"
            await self.service_manager.stop_service(service_name)
            
            # Reactivate previous deployment if available
            if deployment.previous_deployment_id:
                await self._reactivate_previous_deployment(deployment)
            
            deployment.status = DeploymentStatus.ROLLED_BACK
            deployment.logs.append("Rollback completed successfully")
            
        except Exception as e:
            deployment.logs.append(f"Rollback failed: {str(e)}")
    
    async def _reactivate_previous_deployment(self, deployment: Deployment):
        """Reactivate previous deployment"""
        # Restart previous service
        prev_service = f"{deployment.config.app_name}-previous.service"
        success, message = await self.service_manager.start_service(prev_service)
        deployment.logs.append(f"Previous deployment reactivation: {message}")
    
    async def _update_deployment_config(self, deployment: Deployment, deployment_dir: Path):
        """Update deployment configuration files"""
        config = deployment.config
        
        # Update source path to point to deployment directory
        config.source_path = str(deployment_dir / "app")
        
        # Create deployment metadata
        metadata = {
            'deployment_id': deployment.id,
            'version': config.version,
            'git_commit': config.git_commit,
            'timestamp': deployment.start_time.isoformat(),
            'environment': config.target_env.value
        }
        
        with open(deployment_dir / "deployment.json", 'w') as f:
            json.dump(metadata, f, indent=2)

class DeploymentManager:
    """Main deployment manager orchestrating all deployment operations"""
    
    def __init__(self):
        self.executor = DeploymentExecutor()
        self.db_path = '/opt/tower-echo-brain/data/deployments.db'
        self.active_deployments = {}
        self._init_database()
    
    def _init_database(self):
        """Initialize deployment database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS deployments (
                    id TEXT PRIMARY KEY,
                    app_name TEXT,
                    version TEXT,
                    git_commit TEXT,
                    target_env TEXT,
                    status TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    logs TEXT,
                    health_status BOOLEAN,
                    previous_deployment_id TEXT,
                    config TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS deployment_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    deployment_id TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    timestamp TEXT,
                    FOREIGN KEY (deployment_id) REFERENCES deployments (id)
                )
            ''')
    
    async def deploy(self, config: DeploymentConfig) -> str:
        """Start a new deployment"""
        deployment_id = f"{config.app_name}-{config.version}-{int(time.time())}"
        
        # Get previous deployment for rollback
        previous_deployment = await self._get_active_deployment(config.app_name, config.target_env)
        
        deployment = Deployment(
            id=deployment_id,
            config=config,
            status=DeploymentStatus.PENDING,
            start_time=datetime.now(),
            previous_deployment_id=previous_deployment.id if previous_deployment else None
        )
        
        # Save to database
        await self._save_deployment(deployment)
        
        # Execute deployment asynchronously
        asyncio.create_task(self._execute_deployment_async(deployment))
        
        return deployment_id
    
    async def _execute_deployment_async(self, deployment: Deployment):
        """Execute deployment asynchronously"""
        try:
            deployment = await self.executor.execute_deployment(deployment)
            await self._save_deployment(deployment)
            
            if deployment.status == DeploymentStatus.ACTIVE:
                self.active_deployments[f"{deployment.config.app_name}-{deployment.config.target_env.value}"] = deployment
                
        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            deployment.logs.append(f"Deployment execution failed: {str(e)}")
            await self._save_deployment(deployment)
    
    async def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get deployment status"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT * FROM deployments WHERE id = ?', (deployment_id,)
            )
            row = cursor.fetchone()
            
            if row:
                columns = [description[0] for description in cursor.description]
                deployment_data = dict(zip(columns, row))
                deployment_data['logs'] = json.loads(deployment_data['logs'])
                return deployment_data
            
            return None
    
    async def rollback_deployment(self, deployment_id: str) -> bool:
        """Manually trigger rollback for a deployment"""
        deployment_data = await self.get_deployment_status(deployment_id)
        if not deployment_data:
            return False
        
        deployment = self._dict_to_deployment(deployment_data)
        
        if deployment.config.rollback_enabled:
            await self.executor._rollback_deployment(deployment)
            await self._save_deployment(deployment)
            return True
        
        return False
    
    async def list_deployments(self, app_name: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent deployments"""
        with sqlite3.connect(self.db_path) as conn:
            if app_name:
                cursor = conn.execute('''
                    SELECT * FROM deployments WHERE app_name = ? 
                    ORDER BY start_time DESC LIMIT ?
                ''', (app_name, limit))
            else:
                cursor = conn.execute('''
                    SELECT * FROM deployments ORDER BY start_time DESC LIMIT ?
                ''', (limit,))
            
            columns = [description[0] for description in cursor.description]
            deployments = []
            
            for row in cursor.fetchall():
                deployment_data = dict(zip(columns, row))
                deployment_data['logs'] = json.loads(deployment_data['logs'])
                deployments.append(deployment_data)
            
            return deployments
    
    async def get_deployment_metrics(self, deployment_id: str) -> List[Dict[str, Any]]:
        """Get metrics for a deployment"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT metric_name, metric_value, timestamp 
                FROM deployment_metrics WHERE deployment_id = ?
                ORDER BY timestamp DESC
            ''', (deployment_id,))
            
            return [
                {
                    'metric_name': row[0],
                    'metric_value': row[1],
                    'timestamp': row[2]
                }
                for row in cursor.fetchall()
            ]
    
    async def _save_deployment(self, deployment: Deployment):
        """Save deployment to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO deployments 
                (id, app_name, version, git_commit, target_env, status, start_time, 
                 end_time, logs, health_status, previous_deployment_id, config)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                deployment.id,
                deployment.config.app_name,
                deployment.config.version,
                deployment.config.git_commit,
                deployment.config.target_env.value,
                deployment.status.value,
                deployment.start_time.isoformat(),
                deployment.end_time.isoformat() if deployment.end_time else None,
                json.dumps(deployment.logs),
                deployment.health_status,
                deployment.previous_deployment_id,
                json.dumps(asdict(deployment.config), default=str)
            ))
    
    async def _get_active_deployment(self, app_name: str, target_env: EnvironmentType) -> Optional[Deployment]:
        """Get currently active deployment for app and environment"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT * FROM deployments 
                WHERE app_name = ? AND target_env = ? AND status = ?
                ORDER BY start_time DESC LIMIT 1
            ''', (app_name, target_env.value, DeploymentStatus.ACTIVE.value))
            
            row = cursor.fetchone()
            if row:
                columns = [description[0] for description in cursor.description]
                deployment_data = dict(zip(columns, row))
                return self._dict_to_deployment(deployment_data)
            
            return None
    
    def _dict_to_deployment(self, data: Dict[str, Any]) -> Deployment:
        """Convert dictionary to Deployment object"""
        config_data = json.loads(data['config'])
        config = DeploymentConfig(**config_data)
        
        return Deployment(
            id=data['id'],
            config=config,
            status=DeploymentStatus(data['status']),
            start_time=datetime.fromisoformat(data['start_time']),
            end_time=datetime.fromisoformat(data['end_time']) if data['end_time'] else None,
            logs=json.loads(data['logs']),
            health_status=data['health_status'],
            previous_deployment_id=data['previous_deployment_id']
        )

# REST API for deployment management
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

app = FastAPI(title="Echo Deployment Manager", version="1.0.0")
deployment_manager = DeploymentManager()

class DeployRequest(BaseModel):
    app_name: str
    version: str
    git_commit: str
    source_path: str
    target_env: str
    port: int
    health_check_url: Optional[str] = None
    dependencies: List[str] = []
    environment_vars: Dict[str, str] = {}

@app.post("/api/deploy")
async def deploy_application(request: DeployRequest):
    """Deploy an application"""
    try:
        config = DeploymentConfig(
            app_name=request.app_name,
            version=request.version,
            git_commit=request.git_commit,
            source_path=request.source_path,
            target_env=EnvironmentType(request.target_env),
            port=request.port,
            health_check_url=request.health_check_url or f"http://localhost:{request.port}/api/health",
            dependencies=request.dependencies,
            environment_vars=request.environment_vars
        )
        
        deployment_id = await deployment_manager.deploy(config)
        return {"deployment_id": deployment_id, "status": "initiated"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/deployments/{deployment_id}")
async def get_deployment_status(deployment_id: str):
    """Get deployment status"""
    status = await deployment_manager.get_deployment_status(deployment_id)
    if not status:
        raise HTTPException(status_code=404, detail="Deployment not found")
    return status

@app.post("/api/deployments/{deployment_id}/rollback")
async def rollback_deployment(deployment_id: str):
    """Rollback a deployment"""
    success = await deployment_manager.rollback_deployment(deployment_id)
    if not success:
        raise HTTPException(status_code=400, detail="Rollback failed or not allowed")
    return {"status": "rollback_initiated"}

@app.get("/api/deployments")
async def list_deployments(app_name: Optional[str] = None, limit: int = 50):
    """List deployments"""
    return await deployment_manager.list_deployments(app_name, limit)

@app.get("/api/deployments/{deployment_id}/metrics")
async def get_deployment_metrics(deployment_id: str):
    """Get deployment metrics"""
    return await deployment_manager.get_deployment_metrics(deployment_id)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "echo_deployment_manager", 
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8341)
