#!/usr/bin/env python3
"""
Agent Coordination API Routes for Echo Brain
Provides real-time monitoring and control of multi-agent improvement implementation
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/coordination", tags=["Agent Coordination"])

# Agent status tracking
class AgentStatus(BaseModel):
    id: str
    name: str
    status: str  # pending, active, completed, error, paused
    progress: float
    tasks: List[str]
    completed_tasks: int
    total_tasks: int
    efficiency: float
    errors: int
    last_update: datetime
    metadata: Optional[Dict[str, Any]] = None

class CoordinationCommand(BaseModel):
    command: str  # start, pause, resume, stop, reset
    agent_ids: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = None

class CoordinationAlert(BaseModel):
    level: str  # info, warning, error, critical, success
    title: str
    message: str
    timestamp: datetime
    agent_id: Optional[str] = None

# Global state management
agents_state: Dict[str, AgentStatus] = {}
active_websockets: List[WebSocket] = []
coordination_log: List[Dict] = []
system_alerts: List[CoordinationAlert] = []

# Define the 6 improvement agents
IMPROVEMENT_AGENTS = {
    'architecture_agent': {
        'name': 'Architecture Refactoring Agent',
        'icon': '🏗️',
        'tasks': [
            'Analyze current modular architecture',
            'Identify refactoring opportunities',
            'Design improved component structure',
            'Implement dependency injection improvements',
            'Validate architectural changes'
        ]
    },
    'performance_agent': {
        'name': 'Performance Optimization Agent',
        'icon': '⚡',
        'tasks': [
            'Profile current performance bottlenecks',
            'Optimize async operations',
            'Implement caching strategies',
            'Database query optimization',
            'Memory usage optimization'
        ]
    },
    'security_agent': {
        'name': 'Security Enhancement Agent',
        'icon': '🔒',
        'tasks': [
            'Security audit of current implementation',
            'Implement additional input validation',
            'Enhance authentication mechanisms',
            'Add security logging and monitoring',
            'Vulnerability testing and remediation'
        ]
    },
    'testing_agent': {
        'name': 'Testing Framework Agent',
        'icon': '🧪',
        'tasks': [
            'Expand unit test coverage',
            'Implement integration tests',
            'Performance testing framework',
            'Automated regression testing',
            'Load testing implementation'
        ]
    },
    'documentation_agent': {
        'name': 'Documentation Agent',
        'icon': '📚',
        'tasks': [
            'Update API documentation',
            'Create architectural diagrams',
            'Write implementation guides',
            'Update deployment procedures',
            'Create troubleshooting guides'
        ]
    },
    'monitoring_agent': {
        'name': 'Monitoring & Analytics Agent',
        'icon': '📊',
        'tasks': [
            'Implement advanced metrics collection',
            'Create performance dashboards',
            'Setup alerting systems',
            'Add business intelligence tracking',
            'Health check automation'
        ]
    }
}

def initialize_agents():
    """Initialize all improvement agents with default state"""
    global agents_state

    for agent_id, agent_config in IMPROVEMENT_AGENTS.items():
        agents_state[agent_id] = AgentStatus(
            id=agent_id,
            name=agent_config['name'],
            status='pending',
            progress=0.0,
            tasks=agent_config['tasks'],
            completed_tasks=0,
            total_tasks=len(agent_config['tasks']),
            efficiency=0.0,
            errors=0,
            last_update=datetime.now(),
            metadata={'icon': agent_config['icon']}
        )

    logger.info(f"🤖 Initialized {len(agents_state)} improvement agents")

async def broadcast_to_websockets(message: Dict):
    """Broadcast message to all connected WebSocket clients"""
    if active_websockets:
        disconnected = []
        for ws in active_websockets:
            try:
                await ws.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"WebSocket send failed: {e}")
                disconnected.append(ws)

        # Remove disconnected websockets
        for ws in disconnected:
            active_websockets.remove(ws)

def add_coordination_log(source: str, message: str, level: str = "info"):
    """Add entry to coordination log"""
    global coordination_log

    entry = {
        'timestamp': datetime.now().isoformat(),
        'source': source,
        'message': message,
        'level': level
    }

    coordination_log.append(entry)

    # Keep only last 1000 entries
    if len(coordination_log) > 1000:
        coordination_log = coordination_log[-1000:]

    # Broadcast to websockets
    asyncio.create_task(broadcast_to_websockets({
        'type': 'communication',
        'source': source,
        'message': message,
        'level': level,
        'timestamp': entry['timestamp']
    }))

def add_system_alert(level: str, title: str, message: str, agent_id: Optional[str] = None):
    """Add system alert"""
    global system_alerts

    alert = CoordinationAlert(
        level=level,
        title=title,
        message=message,
        timestamp=datetime.now(),
        agent_id=agent_id
    )

    system_alerts.append(alert)

    # Keep only last 100 alerts
    if len(system_alerts) > 100:
        system_alerts = system_alerts[-100:]

    # Broadcast to websockets
    asyncio.create_task(broadcast_to_websockets({
        'type': 'alert',
        'level': level,
        'title': title,
        'message': message,
        'agent_id': agent_id,
        'timestamp': alert.timestamp.isoformat()
    }))

@router.get("/status")
async def get_coordination_status():
    """Get overall coordination system status"""
    return {
        'system_status': 'active',
        'total_agents': len(agents_state),
        'active_agents': len([a for a in agents_state.values() if a.status == 'active']),
        'completed_agents': len([a for a in agents_state.values() if a.status == 'completed']),
        'overall_progress': sum(a.progress for a in agents_state.values()) / len(agents_state) if agents_state else 0,
        'websocket_connections': len(active_websockets),
        'total_alerts': len(system_alerts),
        'last_update': datetime.now().isoformat()
    }

@router.get("/agents")
async def get_all_agents():
    """Get status of all improvement agents"""
    return {
        'agents': [agent.dict() for agent in agents_state.values()],
        'timestamp': datetime.now().isoformat()
    }

@router.get("/agents/{agent_id}")
async def get_agent_status(agent_id: str):
    """Get detailed status of specific agent"""
    if agent_id not in agents_state:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    agent = agents_state[agent_id]

    # Add recent activity from logs
    recent_activity = [
        entry for entry in coordination_log[-50:]
        if agent.name in entry.get('source', '') or agent_id in entry.get('message', '')
    ]

    return {
        'agent': agent.dict(),
        'recent_activity': recent_activity,
        'timestamp': datetime.now().isoformat()
    }

@router.post("/command")
async def execute_coordination_command(command: CoordinationCommand):
    """Execute coordination command (start, pause, resume, stop, reset)"""
    try:
        target_agents = command.agent_ids or list(agents_state.keys())

        if command.command == 'start':
            for agent_id in target_agents:
                if agent_id in agents_state:
                    agents_state[agent_id].status = 'active'
                    agents_state[agent_id].last_update = datetime.now()

                    # Broadcast status update
                    await broadcast_to_websockets({
                        'type': 'agent_status',
                        'agent_id': agent_id,
                        'status': 'active',
                        'progress': agents_state[agent_id].progress
                    })

            add_system_alert('info', 'Implementation Started', f'Activated {len(target_agents)} agents')
            add_coordination_log('Coordinator', f'Started {len(target_agents)} agents: {", ".join(target_agents)}')

        elif command.command == 'pause':
            for agent_id in target_agents:
                if agent_id in agents_state and agents_state[agent_id].status == 'active':
                    agents_state[agent_id].status = 'paused'
                    agents_state[agent_id].last_update = datetime.now()

                    await broadcast_to_websockets({
                        'type': 'agent_status',
                        'agent_id': agent_id,
                        'status': 'paused',
                        'progress': agents_state[agent_id].progress
                    })

            add_system_alert('warning', 'Agents Paused', f'Paused {len(target_agents)} agents')
            add_coordination_log('Coordinator', f'Paused agents: {", ".join(target_agents)}')

        elif command.command == 'resume':
            for agent_id in target_agents:
                if agent_id in agents_state and agents_state[agent_id].status == 'paused':
                    agents_state[agent_id].status = 'active'
                    agents_state[agent_id].last_update = datetime.now()

                    await broadcast_to_websockets({
                        'type': 'agent_status',
                        'agent_id': agent_id,
                        'status': 'active',
                        'progress': agents_state[agent_id].progress
                    })

            add_system_alert('success', 'Agents Resumed', f'Resumed {len(target_agents)} agents')
            add_coordination_log('Coordinator', f'Resumed agents: {", ".join(target_agents)}')

        elif command.command == 'stop':
            for agent_id in target_agents:
                if agent_id in agents_state:
                    agents_state[agent_id].status = 'error'
                    agents_state[agent_id].last_update = datetime.now()

                    await broadcast_to_websockets({
                        'type': 'agent_status',
                        'agent_id': agent_id,
                        'status': 'error',
                        'progress': agents_state[agent_id].progress
                    })

            add_system_alert('critical', 'Emergency Stop', f'Stopped {len(target_agents)} agents')
            add_coordination_log('Coordinator', f'EMERGENCY STOP - Halted agents: {", ".join(target_agents)}')

        elif command.command == 'reset':
            for agent_id in target_agents:
                if agent_id in agents_state:
                    agents_state[agent_id].status = 'pending'
                    agents_state[agent_id].progress = 0.0
                    agents_state[agent_id].completed_tasks = 0
                    agents_state[agent_id].efficiency = 0.0
                    agents_state[agent_id].errors = 0
                    agents_state[agent_id].last_update = datetime.now()

                    await broadcast_to_websockets({
                        'type': 'agent_status',
                        'agent_id': agent_id,
                        'status': 'pending',
                        'progress': 0.0
                    })

            add_system_alert('info', 'Agents Reset', f'Reset {len(target_agents)} agents to initial state')
            add_coordination_log('Coordinator', f'Reset agents: {", ".join(target_agents)}')

        else:
            raise HTTPException(status_code=400, detail=f"Unknown command: {command.command}")

        return {
            'success': True,
            'command': command.command,
            'affected_agents': target_agents,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Command execution failed: {e}")
        add_system_alert('error', 'Command Failed', f'Failed to execute {command.command}: {str(e)}')
        raise HTTPException(status_code=500, detail=f"Command execution failed: {str(e)}")

@router.get("/logs")
async def get_coordination_logs(limit: int = 100):
    """Get recent coordination logs"""
    return {
        'logs': coordination_log[-limit:],
        'total_entries': len(coordination_log),
        'timestamp': datetime.now().isoformat()
    }

@router.get("/alerts")
async def get_system_alerts(limit: int = 50):
    """Get recent system alerts"""
    recent_alerts = system_alerts[-limit:]
    return {
        'alerts': [alert.dict() for alert in recent_alerts],
        'total_alerts': len(system_alerts),
        'timestamp': datetime.now().isoformat()
    }

@router.get("/metrics")
async def get_coordination_metrics():
    """Get coordination system metrics"""
    if not agents_state:
        return {'error': 'No agents initialized'}

    total_tasks = sum(agent.total_tasks for agent in agents_state.values())
    completed_tasks = sum(agent.completed_tasks for agent in agents_state.values())
    total_errors = sum(agent.errors for agent in agents_state.values())

    avg_efficiency = sum(agent.efficiency for agent in agents_state.values()) / len(agents_state)
    overall_progress = sum(agent.progress for agent in agents_state.values()) / len(agents_state)

    status_counts = {}
    for agent in agents_state.values():
        status_counts[agent.status] = status_counts.get(agent.status, 0) + 1

    return {
        'overall_progress': round(overall_progress, 2),
        'total_tasks': total_tasks,
        'completed_tasks': completed_tasks,
        'remaining_tasks': total_tasks - completed_tasks,
        'total_errors': total_errors,
        'average_efficiency': round(avg_efficiency, 2),
        'status_distribution': status_counts,
        'active_websockets': len(active_websockets),
        'log_entries': len(coordination_log),
        'system_alerts': len(system_alerts),
        'timestamp': datetime.now().isoformat()
    }

@router.websocket("/ws")
async def coordination_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time coordination updates"""
    await websocket.accept()
    active_websockets.append(websocket)

    try:
        # Send initial state
        await websocket.send_text(json.dumps({
            'type': 'initial_state',
            'agents': [agent.dict() for agent in agents_state.values()],
            'alerts': [alert.dict() for alert in system_alerts[-10:]],
            'logs': coordination_log[-20:],
            'timestamp': datetime.now().isoformat()
        }))

        add_coordination_log('WebSocket', 'Client connected to coordination dashboard')

        # Keep connection alive
        while True:
            try:
                # Send periodic heartbeat
                await websocket.send_text(json.dumps({
                    'type': 'heartbeat',
                    'timestamp': datetime.now().isoformat()
                }))
                await asyncio.sleep(30)
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket heartbeat failed: {e}")
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in active_websockets:
            active_websockets.remove(websocket)
        add_coordination_log('WebSocket', 'Client disconnected from coordination dashboard')

@router.post("/simulate/progress")
async def simulate_agent_progress(agent_id: Optional[str] = None):
    """Simulate agent progress for testing (development only)"""
    target_agents = [agent_id] if agent_id and agent_id in agents_state else list(agents_state.keys())

    for aid in target_agents:
        if aid in agents_state and agents_state[aid].status == 'active':
            agent = agents_state[aid]

            # Simulate progress increment
            progress_increment = min(10.0, 100.0 - agent.progress)
            agent.progress = min(100.0, agent.progress + progress_increment)

            # Update completed tasks based on progress
            expected_completed = int((agent.progress / 100.0) * agent.total_tasks)
            if expected_completed > agent.completed_tasks:
                agent.completed_tasks = expected_completed

                # Broadcast task completion
                await broadcast_to_websockets({
                    'type': 'agent_task_complete',
                    'agent_id': aid,
                    'task_index': agent.completed_tasks - 1
                })

                add_coordination_log(agent.name, f'Completed task: {agent.tasks[agent.completed_tasks - 1]}')

            # Update efficiency
            agent.efficiency = (agent.completed_tasks / agent.total_tasks) * 100
            agent.last_update = datetime.now()

            # Check if completed
            if agent.progress >= 100.0:
                agent.status = 'completed'
                add_system_alert('success', 'Agent Completed', f'{agent.name} has completed all tasks')
                add_coordination_log(agent.name, 'All tasks completed successfully')

            # Broadcast status update
            await broadcast_to_websockets({
                'type': 'agent_status',
                'agent_id': aid,
                'status': agent.status,
                'progress': agent.progress
            })

    return {
        'success': True,
        'simulated_agents': target_agents,
        'timestamp': datetime.now().isoformat()
    }

# Background task to simulate realistic agent behavior
async def agent_simulation_background():
    """Background task to simulate realistic agent progress"""
    while True:
        try:
            await asyncio.sleep(5)  # Run every 5 seconds

            for agent_id, agent in agents_state.items():
                if agent.status == 'active' and agent.progress < 100:
                    # Random progress increment (0.5% to 3%)
                    import random
                    increment = random.uniform(0.5, 3.0)
                    agent.progress = min(100.0, agent.progress + increment)

                    # Update tasks and efficiency
                    expected_completed = int((agent.progress / 100.0) * agent.total_tasks)
                    if expected_completed > agent.completed_tasks:
                        agent.completed_tasks = expected_completed

                        await broadcast_to_websockets({
                            'type': 'agent_task_complete',
                            'agent_id': agent_id,
                            'task_index': agent.completed_tasks - 1
                        })

                    agent.efficiency = (agent.completed_tasks / agent.total_tasks) * 100
                    agent.last_update = datetime.now()

                    # Check completion
                    if agent.progress >= 100.0:
                        agent.status = 'completed'
                        add_system_alert('success', 'Agent Completed', f'{agent.name} finished all tasks')

                    # Broadcast update
                    await broadcast_to_websockets({
                        'type': 'agent_status',
                        'agent_id': agent_id,
                        'status': agent.status,
                        'progress': agent.progress
                    })

        except Exception as e:
            logger.error(f"Agent simulation error: {e}")

# Initialize agents on startup
def setup_coordination_system():
    """Setup coordination system"""
    initialize_agents()

    add_system_alert('info', 'System Initialized', 'Agent coordination system is ready')
    add_coordination_log('System', 'Coordination system initialized with 6 improvement agents')

    # Start background simulation
    asyncio.create_task(agent_simulation_background())

    # Start periodic metrics broadcasting
    from ..websocket.coordination_websocket import start_periodic_metrics_broadcast
    start_periodic_metrics_broadcast()

# Register routes
def register_coordination_routes(app):
    """Register coordination routes with main FastAPI app"""
    app.include_router(router)
    logger.info("🤖 Agent coordination routes registered")

    # Setup system on startup
    setup_coordination_system()