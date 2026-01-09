#!/usr/bin/env python3
"""
WebSocket handlers for Agent Coordination Dashboard
Provides real-time updates for multi-agent improvement implementation
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Set, Optional
from fastapi import WebSocket, WebSocketDisconnect
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

class CoordinationWebSocketManager:
    """Manages WebSocket connections for coordination dashboard"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_metadata: Dict[WebSocket, Dict] = {}

    async def connect(self, websocket: WebSocket, client_info: Optional[Dict] = None):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)

        # Store client metadata
        self.connection_metadata[websocket] = {
            'connected_at': datetime.now(),
            'client_info': client_info or {},
            'message_count': 0
        }

        logger.info(f"📡 New coordination WebSocket connection (total: {len(self.active_connections)})")

        # Send initial state
        await self.send_initial_state(websocket)

    async def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

        if websocket in self.connection_metadata:
            metadata = self.connection_metadata.pop(websocket)
            logger.info(f"📡 WebSocket disconnected after {metadata['message_count']} messages")

        logger.info(f"📡 WebSocket disconnected (remaining: {len(self.active_connections)})")

    async def send_initial_state(self, websocket: WebSocket):
        """Send initial dashboard state to new connection"""
        try:
            # Get current system state
            from src.api.legacy.coordination_routes import agents_state, system_alerts, coordination_log

            initial_data = {
                'type': 'initial_state',
                'agents': [agent.dict() for agent in agents_state.values()],
                'alerts': [alert.dict() for alert in system_alerts[-10:]],
                'logs': coordination_log[-20:],
                'system_status': {
                    'total_agents': len(agents_state),
                    'active_agents': len([a for a in agents_state.values() if a.status == 'active']),
                    'overall_progress': sum(a.progress for a in agents_state.values()) / len(agents_state) if agents_state else 0
                },
                'timestamp': datetime.now().isoformat()
            }

            await websocket.send_text(json.dumps(initial_data))

        except Exception as e:
            logger.error(f"Failed to send initial state: {e}")

    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return

        message['timestamp'] = datetime.now().isoformat()
        message_text = json.dumps(message)

        disconnected = []

        for websocket in self.active_connections:
            try:
                await websocket.send_text(message_text)

                # Update message count
                if websocket in self.connection_metadata:
                    self.connection_metadata[websocket]['message_count'] += 1

            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket: {e}")
                disconnected.append(websocket)

        # Clean up disconnected websockets
        for websocket in disconnected:
            await self.disconnect(websocket)

    async def send_to_specific(self, websocket: WebSocket, message: Dict):
        """Send message to specific WebSocket connection"""
        try:
            message['timestamp'] = datetime.now().isoformat()
            await websocket.send_text(json.dumps(message))

            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]['message_count'] += 1

        except Exception as e:
            logger.error(f"Failed to send message to specific WebSocket: {e}")
            await self.disconnect(websocket)

    async def broadcast_agent_status(self, agent_id: str, status: str, progress: float, metadata: Optional[Dict] = None):
        """Broadcast agent status update"""
        message = {
            'type': 'agent_status_update',
            'agent_id': agent_id,
            'status': status,
            'progress': progress,
            'metadata': metadata or {}
        }
        await self.broadcast(message)

    async def broadcast_system_alert(self, level: str, title: str, message: str, agent_id: Optional[str] = None):
        """Broadcast system alert"""
        alert_message = {
            'type': 'system_alert',
            'level': level,
            'title': title,
            'message': message,
            'agent_id': agent_id
        }
        await self.broadcast(alert_message)

    async def broadcast_communication(self, source: str, message: str, level: str = 'info'):
        """Broadcast inter-agent communication"""
        comm_message = {
            'type': 'communication',
            'source': source,
            'message': message,
            'level': level
        }
        await self.broadcast(comm_message)

    async def broadcast_metrics_update(self, metrics: Dict):
        """Broadcast system metrics update"""
        message = {
            'type': 'metrics_update',
            'metrics': metrics
        }
        await self.broadcast(message)

    async def broadcast_task_completion(self, agent_id: str, task_index: int, task_name: str):
        """Broadcast task completion"""
        message = {
            'type': 'task_completion',
            'agent_id': agent_id,
            'task_index': task_index,
            'task_name': task_name
        }
        await self.broadcast(message)

    def get_connection_stats(self) -> Dict:
        """Get WebSocket connection statistics"""
        total_messages = sum(
            metadata.get('message_count', 0)
            for metadata in self.connection_metadata.values()
        )

        return {
            'active_connections': len(self.active_connections),
            'total_messages_sent': total_messages,
            'oldest_connection': min(
                (metadata['connected_at'] for metadata in self.connection_metadata.values()),
                default=None
            ),
            'newest_connection': max(
                (metadata['connected_at'] for metadata in self.connection_metadata.values()),
                default=None
            )
        }

# Global WebSocket manager instance
coordination_ws_manager = CoordinationWebSocketManager()

class CoordinationEventBroadcaster:
    """Handles broadcasting of coordination events across the system"""

    def __init__(self, ws_manager: CoordinationWebSocketManager):
        self.ws_manager = ws_manager
        self.event_history: List[Dict] = []

    async def agent_started(self, agent_id: str, agent_name: str):
        """Broadcast agent started event"""
        await self.ws_manager.broadcast_agent_status(agent_id, 'active', 0.0)
        await self.ws_manager.broadcast_communication('Coordinator', f'{agent_name} started')
        await self.ws_manager.broadcast_system_alert('info', 'Agent Started', f'{agent_name} is now active')

        self._log_event('agent_started', agent_id, agent_name)

    async def agent_paused(self, agent_id: str, agent_name: str, progress: float):
        """Broadcast agent paused event"""
        await self.ws_manager.broadcast_agent_status(agent_id, 'paused', progress)
        await self.ws_manager.broadcast_communication('Coordinator', f'{agent_name} paused at {progress:.1f}%')
        await self.ws_manager.broadcast_system_alert('warning', 'Agent Paused', f'{agent_name} temporarily suspended')

        self._log_event('agent_paused', agent_id, agent_name, {'progress': progress})

    async def agent_resumed(self, agent_id: str, agent_name: str, progress: float):
        """Broadcast agent resumed event"""
        await self.ws_manager.broadcast_agent_status(agent_id, 'active', progress)
        await self.ws_manager.broadcast_communication('Coordinator', f'{agent_name} resumed from {progress:.1f}%')
        await self.ws_manager.broadcast_system_alert('success', 'Agent Resumed', f'{agent_name} is active again')

        self._log_event('agent_resumed', agent_id, agent_name, {'progress': progress})

    async def agent_completed(self, agent_id: str, agent_name: str):
        """Broadcast agent completion event"""
        await self.ws_manager.broadcast_agent_status(agent_id, 'completed', 100.0)
        await self.ws_manager.broadcast_communication(agent_name, 'All tasks completed successfully')
        await self.ws_manager.broadcast_system_alert('success', 'Agent Completed', f'{agent_name} finished all tasks')

        self._log_event('agent_completed', agent_id, agent_name)

    async def agent_error(self, agent_id: str, agent_name: str, error: str, progress: float):
        """Broadcast agent error event"""
        await self.ws_manager.broadcast_agent_status(agent_id, 'error', progress)
        await self.ws_manager.broadcast_communication(agent_name, f'Error: {error}', 'error')
        await self.ws_manager.broadcast_system_alert('error', 'Agent Error', f'{agent_name}: {error}')

        self._log_event('agent_error', agent_id, agent_name, {'error': error, 'progress': progress})

    async def task_completed(self, agent_id: str, agent_name: str, task_index: int, task_name: str):
        """Broadcast task completion event"""
        await self.ws_manager.broadcast_task_completion(agent_id, task_index, task_name)
        await self.ws_manager.broadcast_communication(agent_name, f'Completed: {task_name}')

        self._log_event('task_completed', agent_id, agent_name, {'task_index': task_index, 'task_name': task_name})

    async def progress_update(self, agent_id: str, agent_name: str, progress: float, details: Optional[str] = None):
        """Broadcast progress update"""
        await self.ws_manager.broadcast_agent_status(agent_id, 'active', progress)

        if details:
            await self.ws_manager.broadcast_communication(agent_name, f'Progress {progress:.1f}%: {details}')

        self._log_event('progress_update', agent_id, agent_name, {'progress': progress, 'details': details})

    async def system_metrics_update(self, cpu: float, memory: float, nvidia_vram: float, amd_vram: float):
        """Broadcast system metrics update"""
        metrics = {
            'cpu': cpu,
            'memory': memory,
            'nvidia_vram': nvidia_vram,
            'amd_vram': amd_vram
        }
        await self.ws_manager.broadcast_metrics_update(metrics)

    async def coordination_command(self, command: str, agent_ids: List[str], success: bool):
        """Broadcast coordination command execution"""
        status = 'success' if success else 'error'
        message = f'Command "{command}" executed for {len(agent_ids)} agents'

        await self.ws_manager.broadcast_communication('Coordinator', message)
        await self.ws_manager.broadcast_system_alert(
            'success' if success else 'error',
            f'Command {command.title()}',
            message
        )

        self._log_event('coordination_command', None, 'Coordinator', {
            'command': command,
            'agent_ids': agent_ids,
            'success': success
        })

    def _log_event(self, event_type: str, agent_id: Optional[str], agent_name: str, metadata: Optional[Dict] = None):
        """Log event to history"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'agent_id': agent_id,
            'agent_name': agent_name,
            'metadata': metadata or {}
        }

        self.event_history.append(event)

        # Keep only last 1000 events
        if len(self.event_history) > 1000:
            self.event_history = self.event_history[-1000:]

    def get_event_history(self, limit: int = 100) -> List[Dict]:
        """Get recent event history"""
        return self.event_history[-limit:]

# Global event broadcaster
coordination_broadcaster = CoordinationEventBroadcaster(coordination_ws_manager)

async def periodic_metrics_broadcast():
    """Periodically broadcast system metrics to all connections"""
    while True:
        try:
            await asyncio.sleep(10)  # Every 10 seconds

            if coordination_ws_manager.active_connections:
                # Get system metrics (this would integrate with actual system monitoring)
                try:
                    import psutil
                    import subprocess

                    # CPU and Memory
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    memory_percent = memory.percent

                    # GPU metrics (simplified - would need actual GPU monitoring)
                    nvidia_vram = 0.0
                    amd_vram = 0.0

                    try:
                        # Check NVIDIA GPU usage
                        nvidia_result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.memory', '--format=csv,noheader,nounits'],
                                                     capture_output=True, text=True, timeout=5)
                        if nvidia_result.returncode == 0:
                            nvidia_vram = float(nvidia_result.stdout.strip())
                    except:
                        pass

                    try:
                        # Check AMD GPU usage (rocm-smi if available)
                        amd_result = subprocess.run(['rocm-smi', '--showmemuse'],
                                                  capture_output=True, text=True, timeout=5)
                        if amd_result.returncode == 0:
                            # Parse AMD GPU memory usage
                            pass
                    except:
                        pass

                    await coordination_broadcaster.system_metrics_update(
                        cpu=cpu_percent,
                        memory=memory_percent,
                        nvidia_vram=nvidia_vram,
                        amd_vram=amd_vram
                    )

                except Exception as e:
                    logger.error(f"Failed to get system metrics: {e}")

        except Exception as e:
            logger.error(f"Periodic metrics broadcast error: {e}")

def start_periodic_metrics_broadcast():
    """Start background metrics broadcasting - call this after event loop is running"""
    try:
        asyncio.create_task(periodic_metrics_broadcast())
        logger.info("📊 Periodic metrics broadcasting started")
    except Exception as e:
        logger.error(f"Failed to start periodic metrics broadcast: {e}")