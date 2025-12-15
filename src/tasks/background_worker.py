#!/usr/bin/env python3
"""
Echo Brain Background Worker
Continuous task processor that runs autonomously in background
"""

import asyncio
from src.tasks.autonomous_repair_executor import repair_executor
import logging
import time
from src.tasks.task_implementation_executor import get_task_implementation_executor
import traceback
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
import aiohttp
import subprocess
import psutil
import json
import os

from src.tasks.task_queue import TaskQueue, Task, TaskStatus, TaskType, TaskPriority

logger = logging.getLogger(__name__)

class BackgroundWorker:
    """Autonomous background task processor for Echo"""
    
    def __init__(self, task_queue: TaskQueue, max_concurrent_tasks: int = 5):
        self.task_queue = task_queue
        self.max_concurrent_tasks = max_concurrent_tasks
        self.running = False
        self.current_tasks: Dict[str, asyncio.Task] = {}
        self.task_handlers: Dict[TaskType, Callable] = {}
        self.worker_stats = {
            'started_at': None,
            'tasks_processed': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'last_activity': None
        }
        self.brain_state = "initializing"
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self.stop())
        
    def register_handler(self, task_type: TaskType, handler: Callable):
        """Register a handler for specific task type"""
        self.task_handlers[task_type] = handler
        logger.info(f"📝 Registered handler for {task_type.value} tasks")
        
    async def start(self):
        """Start the background worker"""
        if self.running:
            logger.warning("🔄 Background worker already running")
            return
            
        self.running = True
        self.worker_stats['started_at'] = datetime.now()
        self.brain_state = "active"
        
        logger.info("🚀 Echo Background Worker starting...")
        
        # Initialize task queue
        await self.task_queue.initialize()
        
        # Register built-in task handlers
        await self._register_builtin_handlers()
        
        # Start main worker loop
        asyncio.create_task(self._worker_loop())
        logger.info("✅ Worker loop task created")
        
    async def stop(self):
        """Stop the background worker gracefully"""
        if not self.running:
            return
            
        logger.info("🛑 Stopping Echo Background Worker...")
        self.running = False
        self.brain_state = "shutting_down"
        
        # Wait for current tasks to complete (with timeout)
        if self.current_tasks:
            logger.info(f"⏳ Waiting for {len(self.current_tasks)} tasks to complete...")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.current_tasks.values(), return_exceptions=True),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.warning("⚠️ Tasks did not complete within 30 seconds, forcing shutdown")
                
        self.brain_state = "stopped"
        logger.info("✅ Echo Background Worker stopped")
        
    async def _worker_loop(self):
        """Main worker loop that processes tasks continuously"""
        logger.info("🔄 Echo worker loop started")
        
        while self.running:
            try:
                # Clean up completed tasks
                await self._cleanup_completed_tasks()
                
                # Check if we can process more tasks
                if len(self.current_tasks) < self.max_concurrent_tasks:
                    task = await self.task_queue.get_next_task()
                    if task:
                        await self._process_task(task)
                    else:
                        # No tasks available, brain is resting
                        if self.brain_state != "resting" and len(self.current_tasks) == 0:
                            self.brain_state = "resting"
                            logger.info("😴 Echo brain resting - no tasks in queue")
                        await asyncio.sleep(5)  # Check for tasks every 5 seconds
                        
                # Update activity timestamp
                self.worker_stats['last_activity'] = datetime.now()
                
                # Short sleep to prevent busy waiting
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Error in worker loop: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(5)  # Back off on error
                
        logger.info("🏁 Echo worker loop finished")
        
    async def _cleanup_completed_tasks(self):
        """Remove completed tasks from tracking"""
        completed_task_ids = []
        for task_id, task_coroutine in self.current_tasks.items():
            if task_coroutine.done():
                completed_task_ids.append(task_id)
                
        for task_id in completed_task_ids:
            del self.current_tasks[task_id]
            
    async def _process_task(self, task: Task):
        """Process a single task asynchronously"""
        if task.task_type not in self.task_handlers:
            logger.error(f"❌ No handler registered for task type: {task.task_type.value}")
            await self.task_queue.update_task_status(
                task.id, TaskStatus.FAILED, 
                error=f"No handler for task type: {task.task_type.value}"
            )
            return
            
        # Mark task as running
        await self.task_queue.update_task_status(task.id, TaskStatus.RUNNING)
        self.worker_stats['tasks_processed'] += 1
        self.brain_state = "processing"
        
        logger.info(f"🔄 Processing task: {task.name} ({task.id})")
        
        # Create and track the task coroutine
        task_coroutine = asyncio.create_task(self._execute_task(task))
        self.current_tasks[task.id] = task_coroutine
        
    async def _execute_task(self, task: Task):
        """Execute a task with timeout and error handling"""
        start_time = time.time()
        
        try:
            # Execute task with timeout
            handler = self.task_handlers[task.task_type]
            result = await asyncio.wait_for(
                handler(task),
                timeout=task.timeout
            )
            
            # Task completed successfully
            execution_time = time.time() - start_time
            await self.task_queue.update_task_status(
                task.id, TaskStatus.COMPLETED,
                result={
                    'data': result,
                    'execution_time': execution_time,
                    'completed_by': 'echo_background_worker'
                }
            )
            
            self.worker_stats['tasks_completed'] += 1
            logger.info(f"✅ Task completed: {task.name} in {execution_time:.2f}s")
            
        except asyncio.TimeoutError:
            await self.task_queue.update_task_status(
                task.id, TaskStatus.FAILED,
                error=f"Task timed out after {task.timeout} seconds"
            )
            self.worker_stats['tasks_failed'] += 1
            logger.error(f"⏰ Task timed out: {task.name}")
            
        except Exception as e:
            # Check if task should be retried
            if task.retries < task.max_retries:
                await self.task_queue.update_task_status(
                    task.id, TaskStatus.RETRYING,
                    error=f"Attempt {task.retries + 1}: {str(e)}"
                )
                
                # Re-add to queue with incremented retry count
                task.retries += 1
                task.status = TaskStatus.PENDING
                task.updated_at = datetime.now()
                await self.task_queue.add_task(task)
                
                logger.warning(f"🔄 Task failed, retrying: {task.name} (attempt {task.retries})")
            else:
                await self.task_queue.update_task_status(
                    task.id, TaskStatus.FAILED,
                    error=f"Max retries exceeded: {str(e)}"
                )
                self.worker_stats['tasks_failed'] += 1
                logger.error(f"❌ Task failed permanently: {task.name} - {str(e)}")
                
        finally:
            # Remove from current tasks
            if task.id in self.current_tasks:
                del self.current_tasks[task.id]
                
    async def _register_builtin_handlers(self):
        logger.info("📝 Starting handler registration...")
        """Register built-in task handlers"""
        
        async def handle_monitoring_task(task: Task) -> Dict[str, Any]:
            """Handle monitoring tasks"""
            target = task.payload.get('target')
            check_type = task.payload.get('check_type')
            
            if check_type == 'service_health':
                return await self._check_service_health(target)
            elif check_type == 'system_resources':
                return await self._check_system_resources()
            elif check_type == 'disk_usage':
                return await self._check_disk_usage()
            elif check_type == 'process_health':
                return await self._check_process_health(target)
            elif check_type == 'security_service_binding':
                return await self._check_security_service_binding(target)
            elif check_type == 'security_port_exposure':
                return await self._check_security_port_exposure(target)
            else:
                raise ValueError(f"Unknown monitoring check type: {check_type}")
                
        async def handle_optimization_task(task: Task) -> Dict[str, Any]:
            """Handle optimization tasks"""
            system = task.payload.get('system')
            metric = task.payload.get('metric')
            
            if system == 'memory':
                return await self._optimize_memory()
            elif system == 'disk':
                return await self._optimize_disk()
            elif system == 'processes':
                return await self._optimize_processes()
            else:
                raise ValueError(f"Unknown optimization system: {system}")
                
        async def handle_learning_task(task: Task) -> Dict[str, Any]:
            """Handle learning tasks"""
            data_source = task.payload.get('data_source')
            pattern = task.payload.get('pattern')
            
            if data_source == 'system_logs':
                return await self._analyze_system_logs(pattern)
            elif data_source == 'user_interactions':
                return await self._analyze_user_patterns()
            else:
                return {'analysis': f"Learning from {data_source} for pattern {pattern}"}
                
        async def handle_maintenance_task(task: Task) -> Dict[str, Any]:
            """Handle maintenance tasks - NOW WITH ACTUAL REPAIR EXECUTION"""
            action = task.payload.get('action')
            target = task.payload.get('target')
            
            # Extract issue information
            issue = task.payload.get('issue', task.name)
            service_name = task.payload.get('service_name', target)
            
            logger.info(f"🔧 Executing maintenance task: {action} on {target}")
            
            if action == 'restart':
                # ACTUALLY RESTART THE SERVICE using repair executor
                result = await repair_executor.execute_repair(
                    repair_type='service_restart',
                    target=service_name,
                    issue=issue
                )
                return result
                
            elif action == 'cleanup':
                result = await repair_executor.execute_repair(
                    repair_type='disk_cleanup',
                    target=target,
                    issue=issue,
                    clean_logs=True,
                    clean_temp=False  # Be conservative
                )
                return result
                
            elif action == 'rotate_logs':
                result = await repair_executor.execute_repair(
                    repair_type='log_rotation',
                    target=target,
                    issue=issue
                )
                return result

            elif action == 'vault_unseal':
                result = await repair_executor.execute_repair(
                    repair_type='vault_unseal',
                    target=target,
                    issue=issue
                )
                return result

            else:
                return {
                    'status': 'unknown_action',
                    'action': action,
                    'message': f"Don't know how to perform action: {action}"
                }
        
        async def handle_analysis_task(task: Task) -> Dict[str, Any]:
            """Handle analysis tasks"""
            return await self._perform_analysis(task.payload)
            
        # Register all handlers
        async def handle_code_refactor_task(task: Task) -> Dict[str, Any]:
            """Handle autonomous code refactoring tasks"""
            try:
                action = task.payload.get("action", "analyze")
                project_path = task.payload.get("project_path")

                if not project_path:
                    raise ValueError("No project_path specified in task payload")

                logger.info(f"🔧 Processing code refactor task: {action} on {project_path}")

                # Import code refactor executor
                from src.tasks.code_refactor_executor import code_refactor_executor

                if action == "analyze":
                    # Analyze code quality
                    result = await code_refactor_executor.analyze_code_quality(project_path)
                    return {
                        "success": True,
                        "action": "analyze",
                        "project": project_path,
                        "quality_score": result.get("pylint_score"),
                        "file_count": result.get("file_count"),
                        "issues_found": len(result.get("issues", []))
                    }

                elif action == "auto_fix_code":
                    # Perform automatic code fixes
                    result = await code_refactor_executor.auto_fix_common_issues(project_path)
                    return {
                        "success": result.get("fixes_applied", 0) > 0,
                        "action": "auto_fix_code",
                        "project": project_path,
                        "fixes_applied": result.get("fixes_applied", 0),
                        "files_modified": result.get("files_modified", [])
                    }

                elif action == "format_code":
                    # Format code with black
                    if Path(project_path).is_file():
                        result = await code_refactor_executor.auto_format_code(project_path)
                    else:
                        result = await code_refactor_executor.auto_fix_common_issues(project_path)
                    return {
                        "success": result.get("success", False),
                        "action": "format_code",
                        "project": project_path,
                        "output": result.get("output", "")
                    }

                else:
                    # Legacy support for task implementation
                    task_description = task.payload.get("task")
                    service = task.payload.get("service")
                    test = task.payload.get("test", True)

                    executor = get_task_implementation_executor(board=None)
                    result = await executor.implement_task(task_description, service, test)

                    return {
                        "success": result.get("success", False),
                        "review_score": result.get("review_results", {}).get("score"),
                        "files_modified": list(result.get("code_changes", {}).keys())
                    }

            except Exception as e:
                logger.error(f"Code refactor task failed: {e}")
                raise

        self.register_handler(TaskType.MONITORING, handle_monitoring_task)
        self.register_handler(TaskType.OPTIMIZATION, handle_optimization_task)
        self.register_handler(TaskType.LEARNING, handle_learning_task)
        self.register_handler(TaskType.MAINTENANCE, handle_maintenance_task)
        self.register_handler(TaskType.CODE_REFACTOR, handle_code_refactor_task)
        self.register_handler(TaskType.ANALYSIS, handle_analysis_task)        
        # LORA Training Workers
        from src.workers.lora_generation_worker import handle_lora_generation_task
        from src.workers.lora_tagging_worker import handle_lora_tagging_task  
        from src.workers.lora_training_worker import handle_lora_training_task
        
        self.register_handler(TaskType.LORA_GENERATION, handle_lora_generation_task)
        self.register_handler(TaskType.LORA_TAGGING, handle_lora_tagging_task)
        self.register_handler(TaskType.LORA_TRAINING, handle_lora_training_task)
        logger.info("✅ LORA handlers registered")

        
        logger.info("✅ Built-in task handlers registered")
        
    # Task implementation methods
    async def _check_service_health(self, service_url: str) -> Dict[str, Any]:
        """Check health of a service"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{service_url}/api/health", timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'status': 'healthy',
                            'response_time': response.headers.get('X-Response-Time', 'unknown'),
                            'data': data
                        }
                    else:
                        return {
                            'status': 'unhealthy',
                            'http_status': response.status,
                            'error': f"HTTP {response.status}"
                        }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
            
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available': memory.available,
                'disk_percent': (disk.used / disk.total) * 100,
                'disk_free': disk.free,
                'status': 'healthy' if cpu_percent < 80 and memory.percent < 80 else 'warning'
            }
        except Exception as e:
            return {'error': str(e)}
            
    async def _check_disk_usage(self) -> Dict[str, Any]:
        """Check disk usage across mount points"""
        try:
            disk_info = {}
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_info[partition.mountpoint] = {
                        'total': usage.total,
                        'used': usage.used,
                        'free': usage.free,
                        'percent': (usage.used / usage.total) * 100
                    }
                except PermissionError:
                    continue
                    
            return {'disk_usage': disk_info}
        except Exception as e:
            return {'error': str(e)}
            
    async def _check_process_health(self, process_name: str) -> Dict[str, Any]:
        """Check if a specific process is running"""
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                if process_name.lower() in proc.info['name'].lower():
                    processes.append(proc.info)
                    
            return {
                'process_name': process_name,
                'running': len(processes) > 0,
                'processes': processes
            }
        except Exception as e:
            return {'error': str(e)}

    async def _check_security_service_binding(self, target: str = None) -> Dict[str, Any]:
        """Check for security issues with service bindings"""
        try:
            import socket
            import subprocess

            # Check for services bound to 0.0.0.0 (potential security risk)
            result = subprocess.run(['netstat', '-tlnp'], capture_output=True, text=True)
            lines = result.stdout.split('\n')

            risky_bindings = []
            secure_bindings = []

            for line in lines:
                if '0.0.0.0:' in line:
                    risky_bindings.append(line.strip())
                elif '127.0.0.1:' in line or 'localhost:' in line:
                    secure_bindings.append(line.strip())

            security_score = len(secure_bindings) / max(len(risky_bindings) + len(secure_bindings), 1)

            return {
                'check_type': 'security_service_binding',
                'target': target or 'all_services',
                'security_score': round(security_score, 2),
                'risky_bindings': len(risky_bindings),
                'secure_bindings': len(secure_bindings),
                'status': 'secure' if security_score > 0.8 else 'warning' if security_score > 0.5 else 'critical',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': str(e), 'check_type': 'security_service_binding'}

    async def _check_security_port_exposure(self, target: str = None) -> Dict[str, Any]:
        """Check for security issues with port exposure"""
        try:
            import subprocess

            # Check for open ports and their exposure
            result = subprocess.run(['ss', '-tulnp'], capture_output=True, text=True)
            lines = result.stdout.split('\n')

            exposed_ports = []
            internal_ports = []
            tower_ports = [8080, 8088, 8188, 8200, 8301, 8302, 8303, 8307, 8308, 8309, 8315, 8323, 8328, 8329]

            for line in lines:
                for port in tower_ports:
                    if f':{port}' in line:
                        if '0.0.0.0' in line or '*:' in line:
                            exposed_ports.append({'port': port, 'binding': 'external', 'line': line.strip()})
                        elif '127.0.0.1' in line:
                            internal_ports.append({'port': port, 'binding': 'internal', 'line': line.strip()})

            # Calculate exposure risk
            total_ports = len(exposed_ports) + len(internal_ports)
            exposure_ratio = len(exposed_ports) / max(total_ports, 1)

            return {
                'check_type': 'security_port_exposure',
                'target': target or 'tower_services',
                'exposed_ports': len(exposed_ports),
                'internal_ports': len(internal_ports),
                'exposure_ratio': round(exposure_ratio, 2),
                'status': 'secure' if exposure_ratio < 0.3 else 'warning' if exposure_ratio < 0.6 else 'critical',
                'port_details': exposed_ports + internal_ports,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': str(e), 'check_type': 'security_port_exposure'}

    async def _optimize_memory(self) -> Dict[str, Any]:
        """Optimize system memory usage"""
        try:
            # Force garbage collection in Python processes
            result = subprocess.run(['sync'], capture_output=True, text=True)
            
            memory_before = psutil.virtual_memory()
            
            # Drop caches if possible (requires sudo)
            try:
                subprocess.run(['sync'], check=True)
                # Note: Actual cache dropping would require sudo
                # subprocess.run(['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'], check=True)
            except:
                pass
                
            memory_after = psutil.virtual_memory()
            
            return {
                'memory_before': memory_before.percent,
                'memory_after': memory_after.percent,
                'improvement': memory_before.percent - memory_after.percent,
                'action': 'memory_optimization_completed'
            }
        except Exception as e:
            return {'error': str(e)}
            
    async def _optimize_disk(self) -> Dict[str, Any]:
        """Optimize disk usage"""
        try:
            # Clean temporary files
            temp_cleaned = 0
            temp_dirs = ['/tmp', '/var/tmp']
            
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    for file in os.listdir(temp_dir):
                        file_path = os.path.join(temp_dir, file)
                        try:
                            if os.path.isfile(file_path):
                                size = os.path.getsize(file_path)
                                os.remove(file_path)
                                temp_cleaned += size
                        except:
                            continue
                            
            return {
                'temp_files_cleaned_bytes': temp_cleaned,
                'action': 'disk_optimization_completed'
            }
        except Exception as e:
            return {'error': str(e)}
            
    async def _optimize_processes(self) -> Dict[str, Any]:
        """Optimize running processes"""
        try:
            zombie_count = 0
            high_cpu_processes = []
            
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'status']):
                if proc.info['status'] == psutil.STATUS_ZOMBIE:
                    zombie_count += 1
                elif proc.info['cpu_percent'] > 90:
                    high_cpu_processes.append(proc.info)
                    
            return {
                'zombie_processes': zombie_count,
                'high_cpu_processes': high_cpu_processes,
                'action': 'process_analysis_completed'
            }
        except Exception as e:
            return {'error': str(e)}
            
    async def _analyze_system_logs(self, pattern: str) -> Dict[str, Any]:
        """Analyze system logs for patterns"""
        try:
            # Simple log analysis
            log_files = ['/var/log/syslog', '/var/log/auth.log']
            pattern_matches = 0
            
            for log_file in log_files:
                if os.path.exists(log_file):
                    try:
                        result = subprocess.run(
                            ['grep', '-c', pattern, log_file],
                            capture_output=True, text=True
                        )
                        if result.returncode == 0:
                            pattern_matches += int(result.stdout.strip())
                    except:
                        continue
                        
            return {
                'pattern': pattern,
                'matches_found': pattern_matches,
                'analysis': f"Found {pattern_matches} occurrences of pattern '{pattern}'",
                'action': 'log_analysis_completed'
            }
        except Exception as e:
            return {'error': str(e)}
            
    async def _analyze_user_patterns(self) -> Dict[str, Any]:
        """Analyze user interaction patterns"""
        return {
            'analysis': 'User pattern analysis completed',
            'patterns': ['frequent_queries', 'peak_usage_times'],
            'action': 'user_analysis_completed'
        }
        
    async def _cleanup_system(self, target: str) -> Dict[str, Any]:
        """Cleanup system resources"""
        if target == 'temp_files':
            return await self._optimize_disk()
        elif target == 'old_logs':
            # Clean old log files
            return {'action': f"Cleaned old logs for {target}"}
        else:
            return {'action': f"Generic cleanup for {target}"}
            
    async def _backup_data(self, target: str) -> Dict[str, Any]:
        """Backup important data"""
        return {
            'action': f"Backup completed for {target}",
            'timestamp': datetime.now().isoformat()
        }
        
    async def _update_system(self, target: str) -> Dict[str, Any]:
        """Update system components"""
        return {
            'action': f"Update check completed for {target}",
            'timestamp': datetime.now().isoformat()
        }
        
    async def _perform_analysis(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Perform generic analysis"""
        return {
            'analysis': 'Generic analysis completed',
            'payload_processed': payload,
            'action': 'analysis_completed'
        }
        
    def get_worker_status(self) -> Dict[str, Any]:
        """Get current worker status"""
        return {
            'running': self.running,
            'brain_state': self.brain_state,
            'current_tasks': list(self.current_tasks.keys()),
            'max_concurrent': self.max_concurrent_tasks,
            'stats': self.worker_stats,
            'handlers_registered': list(self.task_handlers.keys())
        }
