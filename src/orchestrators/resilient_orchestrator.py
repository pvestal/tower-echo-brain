"""
Resilient Multimedia Orchestrator with Cloud Escalation
Handles local compute, monitors resources, and escalates to cloud when needed
"""

import asyncio
import aiohttp
import json
import time
import psutil
import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)

class ComputeLocation(Enum):
    LOCAL = "local"
    CLOUD_FIREBASE = "firebase"
    CLOUD_GCP = "gcp"
    HYBRID = "hybrid"

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 5
    HIGH = 10
    CRITICAL = 20

@dataclass
class ComputeMetrics:
    """Real-time compute metrics"""
    cpu_percent: float
    memory_percent: float
    gpu_memory_used: float = 0.0
    gpu_utilization: float = 0.0
    active_tasks: int = 0
    queue_depth: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_overloaded(self) -> bool:
        """Check if system is overloaded"""
        return (
            self.cpu_percent > 80 or
            self.memory_percent > 85 or
            self.gpu_utilization > 90 or
            self.active_tasks > 10
        )

    @property
    def load_score(self) -> float:
        """Calculate load score (0-100, higher = more loaded)"""
        return (
            self.cpu_percent * 0.3 +
            self.memory_percent * 0.3 +
            self.gpu_utilization * 0.3 +
            min(self.active_tasks * 5, 100) * 0.1
        )

class CircuitBreaker:
    """Circuit breaker pattern for service resilience"""

    def __init__(self, failure_threshold: int = 3, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def record_success(self):
        """Record successful call"""
        self.failures = 0
        self.state = "closed"

    def record_failure(self):
        """Record failed call"""
        self.failures += 1
        self.last_failure_time = time.time()

        if self.failures >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failures} failures")

    def is_open(self) -> bool:
        """Check if circuit is open"""
        if self.state == "closed":
            return False

        # Check if we should try half-open
        if self.state == "open" and self.last_failure_time:
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "half-open"
                logger.info("Circuit breaker entering half-open state")
                return False

        return self.state == "open"

    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)"""
        return self.state == "half-open"


class ResilientOrchestrator:
    """Resilient orchestrator with cloud escalation and intelligent routing"""

    def __init__(self, firebase_config: Optional[Dict] = None):
        # Service configuration
        self.services = {
            'comfyui': {'url': 'http://127.0.0.1:8188', 'timeout': 60},
            'voice': {'url': 'http://127.0.0.1:8312', 'timeout': 30},
            'music': {'url': 'http://127.0.0.1:8308', 'timeout': 90},
            'anime': {'url': 'http://127.0.0.1:8328', 'timeout': 60},
            'ollama': {'url': 'http://127.0.0.1:11434', 'timeout': 120}
        }

        # Circuit breakers for each service
        self.circuit_breakers = {
            service: CircuitBreaker() for service in self.services
        }

        # Task queue and metrics
        self.active_tasks = {}
        self.task_queue = asyncio.Queue()
        self.metrics_history = []

        # Cloud configuration
        self.firebase_config = firebase_config
        self.cloud_enabled = firebase_config is not None

        # Background monitor tasks (will be started when event loop is available)
        self._monitor_task = None
        self._queue_task = None

    async def start_background_tasks(self):
        """Start background monitoring tasks (call after event loop is running)"""
        if not self._monitor_task:
            self._monitor_task = asyncio.create_task(self._monitor_system())
            logger.info("Started system monitoring task")
        if not self._queue_task:
            self._queue_task = asyncio.create_task(self._process_queue())
            logger.info("Started queue processing task")

    async def _monitor_system(self):
        """Monitor system resources continuously"""
        while True:
            try:
                metrics = ComputeMetrics(
                    cpu_percent=psutil.cpu_percent(interval=1),
                    memory_percent=psutil.virtual_memory().percent,
                    active_tasks=len(self.active_tasks)
                )

                # Try to get GPU metrics if available
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                    metrics.gpu_utilization = gpu_util.gpu
                    metrics.gpu_memory_used = (mem_info.used / mem_info.total) * 100
                except:
                    pass  # No GPU or pynvml not available

                # Keep history (last 100 measurements)
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > 100:
                    self.metrics_history.pop(0)

                # Log if system is stressed
                if metrics.is_overloaded:
                    logger.warning(f"System overloaded: CPU={metrics.cpu_percent}%, "
                                 f"Memory={metrics.memory_percent}%, "
                                 f"Tasks={metrics.active_tasks}")

                await asyncio.sleep(5)  # Monitor every 5 seconds

            except Exception as e:
                logger.error(f"Error monitoring system: {e}")
                await asyncio.sleep(10)

    async def _process_queue(self):
        """Process queued tasks with priority"""
        while True:
            try:
                # Get task from queue
                task = await self.task_queue.get()

                # Check if we should process locally or escalate
                location = await self._determine_compute_location(task)

                if location == ComputeLocation.LOCAL:
                    asyncio.create_task(self._execute_task(task))
                else:
                    asyncio.create_task(self._escalate_to_cloud(task, location))

            except Exception as e:
                logger.error(f"Error processing queue: {e}")

    def enable_cloud_escalation(self):
        """Enable cloud escalation for testing"""
        self.cloud_enabled = True
        logger.info("Cloud escalation enabled")

    async def _determine_compute_location(self, task: Dict) -> ComputeLocation:
        """Intelligently determine where to run the task"""

        # Get current metrics
        if not self.metrics_history:
            return ComputeLocation.LOCAL

        current_metrics = self.metrics_history[-1]
        task_complexity = task.get('complexity', 'medium')

        # Decision logic
        if not self.cloud_enabled:
            return ComputeLocation.LOCAL

        # Critical tasks always get resources
        if task.get('priority') == TaskPriority.CRITICAL:
            if current_metrics.is_overloaded:
                return ComputeLocation.CLOUD_FIREBASE
            return ComputeLocation.LOCAL

        # Complex tasks on overloaded system go to cloud
        if task_complexity in ['high', 'extreme'] and current_metrics.load_score > 70:
            logger.info(f"Escalating complex task to cloud (load={current_metrics.load_score})")
            return ComputeLocation.CLOUD_FIREBASE

        # Burst handling - if queue is deep, use cloud for overflow
        if self.task_queue.qsize() > 5 and current_metrics.load_score > 60:
            logger.info(f"Escalating due to queue depth ({self.task_queue.qsize()} tasks)")
            return ComputeLocation.CLOUD_FIREBASE

        return ComputeLocation.LOCAL

    async def call_service_with_retry(self,
                                     service: str,
                                     endpoint: str,
                                     method: str = "POST",
                                     data: Dict = None,
                                     max_retries: int = 3) -> Dict[str, Any]:
        """Call service with exponential backoff retry"""

        # Check circuit breaker
        breaker = self.circuit_breakers.get(service)
        if breaker and breaker.is_open():
            logger.warning(f"Circuit breaker open for {service}")
            return {"success": False, "error": "Service unavailable (circuit open)"}

        # Get service config
        service_config = self.services.get(service)
        if not service_config:
            return {"success": False, "error": f"Unknown service: {service}"}

        base_url = service_config['url']
        timeout = service_config['timeout']

        # Retry with exponential backoff
        for attempt in range(max_retries):
            try:
                backoff = 2 ** attempt  # 1, 2, 4 seconds

                if attempt > 0:
                    logger.info(f"Retry {attempt}/{max_retries} for {service}{endpoint} "
                              f"after {backoff}s backoff")
                    await asyncio.sleep(backoff)

                # Make request with timeout
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method,
                        f"{base_url}{endpoint}",
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=timeout)
                    ) as response:
                        response_text = await response.text()

                        if response.status == 200:
                            # Success - reset circuit breaker
                            if breaker:
                                breaker.record_success()

                            try:
                                return json.loads(response_text)
                            except:
                                return {"success": True, "response": response_text}

                        # Non-200 status
                        logger.warning(f"{service} returned {response.status}: {response_text[:200]}")

                        # Don't retry on client errors (4xx)
                        if 400 <= response.status < 500:
                            if breaker:
                                breaker.record_failure()
                            return {
                                "success": False,
                                "error": f"Client error {response.status}: {response_text[:200]}"
                            }

            except asyncio.TimeoutError:
                logger.warning(f"Timeout calling {service}{endpoint} (attempt {attempt + 1})")
                continue

            except Exception as e:
                logger.error(f"Error calling {service}: {e}")
                continue

        # All retries failed
        if breaker:
            breaker.record_failure()

        return {
            "success": False,
            "error": f"Service call failed after {max_retries} retries"
        }

    async def generate_image(self, prompt: str, style: str = "anime",
                           priority: TaskPriority = TaskPriority.MEDIUM) -> Dict:
        """Generate image with resilience and cloud escalation"""

        task = {
            "id": hashlib.md5(f"{prompt}{time.time()}".encode()).hexdigest()[:8],
            "type": "image_generation",
            "service": "comfyui",
            "priority": priority,
            "complexity": "high" if len(prompt) > 100 else "medium",
            "payload": {
                "prompt": prompt,
                "style": style
            },
            "created_at": datetime.now()
        }

        # Add to active tasks
        self.active_tasks[task["id"]] = task

        try:
            # Check if we should escalate immediately
            location = await self._determine_compute_location(task)

            if location == ComputeLocation.CLOUD_FIREBASE:
                return await self._escalate_to_cloud(task, location)

            # Build ComfyUI workflow
            workflow = self._build_comfyui_workflow(prompt, style)

            # Call with retry
            result = await self.call_service_with_retry(
                "comfyui",
                "/prompt",
                data={"prompt": workflow}
            )

            if result.get("prompt_id"):
                result["compute_location"] = "local"
                result["success"] = True

            return result

        finally:
            # Remove from active tasks
            self.active_tasks.pop(task["id"], None)

    async def generate_voice(self, text: str, character: str = "echo_default",
                           priority: TaskPriority = TaskPriority.MEDIUM) -> Dict:
        """Generate voice with resilience"""

        task = {
            "id": hashlib.md5(f"{text}{time.time()}".encode()).hexdigest()[:8],
            "type": "voice_generation",
            "service": "voice",
            "priority": priority,
            "complexity": "low" if len(text) < 100 else "medium",
            "payload": {
                "text": text,
                "character": character
            }
        }

        self.active_tasks[task["id"]] = task

        try:
            # Voice is usually fast, keep it local unless system is very stressed
            current_load = self.metrics_history[-1].load_score if self.metrics_history else 0

            if current_load > 85 and self.cloud_enabled:
                return await self._escalate_to_cloud(task, ComputeLocation.CLOUD_FIREBASE)

            # Get auth token
            auth_result = await self.call_service_with_retry(
                "voice",
                "/api/auth/token",
                data={"username": "echo_brain", "password": "test"}
            )

            if not auth_result.get("access_token"):
                return {"success": False, "error": "Voice auth failed"}

            # Generate voice with auth
            headers = {"Authorization": f"Bearer {auth_result['access_token']}"}

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.services['voice']['url']}/api/tts",
                    json={
                        "text": text,
                        "voice": character,
                        "speed": 1.0
                    },
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        return {
                            "success": True,
                            "message": "Voice generated",
                            "character": character,
                            "compute_location": "local"
                        }

                    return {
                        "success": False,
                        "error": f"Voice generation failed: {response.status}"
                    }

        finally:
            self.active_tasks.pop(task["id"], None)

    async def orchestrate_complex_task(self, task_type: str, requirements: Dict,
                                      priority: TaskPriority = TaskPriority.MEDIUM) -> Dict:
        """Orchestrate complex multi-service tasks with intelligent routing"""

        orchestration_id = hashlib.md5(
            f"{task_type}{time.time()}".encode()
        ).hexdigest()[:12]

        logger.info(f"Starting orchestration {orchestration_id}: {task_type}")

        # Analyze task complexity
        subtasks = self._decompose_task(task_type, requirements)

        # Determine execution strategy
        if len(subtasks) > 3 and self.cloud_enabled:
            # Complex orchestration - consider hybrid approach
            strategy = "hybrid"
        else:
            strategy = "local"

        results = []

        # Execute subtasks with intelligent routing
        for subtask in subtasks:
            # Determine where to run this subtask
            if subtask.get("can_parallelize") and strategy == "hybrid":
                # Run in parallel on cloud
                result = await self._escalate_to_cloud(subtask, ComputeLocation.CLOUD_FIREBASE)
            else:
                # Run locally with retry
                result = await self._execute_subtask(subtask)

            results.append({
                "subtask": subtask["name"],
                "success": result.get("success", False),
                "result": result,
                "location": result.get("compute_location", "unknown")
            })

        # Aggregate results
        success_rate = sum(1 for r in results if r["success"]) / len(results)

        return {
            "orchestration_id": orchestration_id,
            "task_type": task_type,
            "strategy": strategy,
            "success": success_rate > 0.7,  # 70% success threshold
            "success_rate": success_rate,
            "subtasks": results,
            "execution_time": time.time(),
            "compute_metrics": self.get_current_metrics()
        }

    def _decompose_task(self, task_type: str, requirements: Dict) -> List[Dict]:
        """Decompose complex task into subtasks"""

        if task_type == "trailer":
            return [
                {
                    "name": "generate_title_card",
                    "service": "comfyui",
                    "can_parallelize": True,
                    "complexity": "high",
                    "payload": {
                        "prompt": requirements.get("title", "Epic Trailer")
                    }
                },
                {
                    "name": "generate_narration",
                    "service": "voice",
                    "can_parallelize": True,
                    "complexity": "low",
                    "payload": {
                        "text": requirements.get("narration", "Coming soon")
                    }
                },
                {
                    "name": "generate_music",
                    "service": "music",
                    "can_parallelize": True,
                    "complexity": "high",
                    "payload": {
                        "description": requirements.get("music_style", "epic cinematic")
                    }
                }
            ]

        # Default single task
        return [{
            "name": task_type,
            "service": "comfyui",
            "can_parallelize": False,
            "complexity": "medium",
            "payload": requirements
        }]

    async def _execute_subtask(self, subtask: Dict) -> Dict:
        """Execute a single subtask with appropriate service"""

        service = subtask.get("service")

        if service == "comfyui":
            return await self.generate_image(
                subtask["payload"].get("prompt", ""),
                priority=TaskPriority.MEDIUM
            )
        elif service == "voice":
            return await self.generate_voice(
                subtask["payload"].get("text", ""),
                priority=TaskPriority.MEDIUM
            )
        elif service == "music":
            # Music generation with fallback
            return await self._generate_music_with_fallback(
                subtask["payload"]
            )

        return {"success": False, "error": f"Unknown service: {service}"}

    async def _generate_music_with_fallback(self, payload: Dict) -> Dict:
        """Generate music with fallback to simpler generation"""

        # Try the full music generation first
        result = await self.call_service_with_retry(
            "music",
            "/api/generate/music",
            data={
                "style": "electronic",
                "emotion": "epic",
                "duration": payload.get("duration", 10),
                "bpm": 120,
                "key_signature": "C major"
            },
            max_retries=1  # Only try once to avoid long waits
        )

        if result.get("success"):
            return result

        # Fallback to simpler generation or pre-generated samples
        logger.info("Music generation failed, using fallback")

        return {
            "success": True,
            "message": "Using pre-generated music sample",
            "audio_file": "/opt/tower-music/samples/epic_trailer.mp3",
            "fallback": True,
            "compute_location": "local_cache"
        }

    async def _escalate_to_cloud(self, task: Dict, location: ComputeLocation) -> Dict:
        """Escalate task to cloud compute (Firebase/GCP)"""

        import os

        # Get Firebase Functions URL (emulator or production)
        firebase_url = os.getenv('FIREBASE_FUNCTIONS_URL', 'http://127.0.0.1:5001/tower-echo-brain/us-central1')

        logger.info(f"Escalating task {task.get('id', 'unknown')} to {location.value} via {firebase_url}")

        try:
            # Map task type to Firebase function
            function_map = {
                "image_generation": "generateImage",
                "voice_generation": "generateVoice",
                "music_generation": "generateMusic",
                "complex": "orchestrateTask"
            }

            function_name = function_map.get(task.get("type"), "orchestrateTask")
            url = f"{firebase_url}/{function_name}"

            # Prepare payload for Firebase Function
            payload = {
                "data": task.get("payload", {})
            }

            if function_name == "orchestrateTask":
                payload["data"]["task_type"] = task.get("type", "unknown")
                payload["data"]["subtasks"] = task.get("subtasks", [])
                payload["data"]["requirements"] = task.get("requirements", {})

            # Call Firebase Function
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=60) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Cloud escalation successful for task {task.get('id')}")
                        # Extract result from Firebase response format
                        actual_result = result.get("result", result)
                        actual_result["compute_location"] = "firebase"
                        return actual_result
                    else:
                        error_text = await response.text()
                        logger.error(f"Firebase function error ({response.status}): {error_text}")
                        # Fallback to local processing
                        return {
                            "success": False,
                            "error": f"Firebase returned {response.status}",
                            "compute_location": "local",
                            "fallback": True
                        }

        except aiohttp.ClientError as e:
            logger.warning(f"Firebase connection failed: {e}, falling back to local")
            return {
                "success": False,
                "error": "Firebase unavailable",
                "compute_location": "local",
                "fallback": True
            }
        except Exception as e:
            logger.error(f"Cloud escalation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "compute_location": "local",
                "fallback": True
            }

    def _build_comfyui_workflow(self, prompt: str, style: str) -> Dict:
        """Build ComfyUI workflow"""
        return {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "AOM3A1B.safetensors"}  # Use working 2GB model
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": f"{prompt}, {style} style, masterpiece, best quality",
                    "clip": ["1", 1]
                }
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "worst quality, low quality, blurry",
                    "clip": ["1", 1]
                }
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 512, "height": 512, "batch_size": 1}
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": int(time.time()),
                    "steps": 20,
                    "cfg": 7.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0]
                }
            },
            "6": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["5", 0],
                    "vae": ["1", 2]
                }
            },
            "7": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": f"ComfyUI",
                    "images": ["6", 0]
                }
            }
        }

    def get_current_metrics(self) -> Dict:
        """Get current system metrics"""
        if not self.metrics_history:
            return {"status": "no_data"}

        current = self.metrics_history[-1]

        return {
            "cpu_percent": current.cpu_percent,
            "memory_percent": current.memory_percent,
            "gpu_utilization": current.gpu_utilization,
            "active_tasks": len(self.active_tasks),
            "queue_depth": self.task_queue.qsize(),
            "load_score": current.load_score,
            "is_overloaded": current.is_overloaded,
            "circuit_breakers": {
                service: breaker.state
                for service, breaker in self.circuit_breakers.items()
            }
        }

    async def health_check(self) -> Dict:
        """Comprehensive health check"""
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "metrics": self.get_current_metrics(),
            "services": {}
        }

        # Check each service
        for service in self.services:
            breaker = self.circuit_breakers.get(service)

            if breaker and breaker.is_open():
                health["services"][service] = "circuit_open"
                health["status"] = "degraded"
            else:
                # Try health endpoint
                result = await self.call_service_with_retry(
                    service,
                    "/health",
                    method="GET",
                    max_retries=1
                )

                health["services"][service] = "healthy" if result.get("success") else "unhealthy"

                if not result.get("success"):
                    health["status"] = "degraded"

        return health

    async def orchestrate_multimedia(self, task_type: str, description: str, requirements: Dict) -> Dict:
        """Orchestrate multimedia task - wrapper for orchestrate_complex_task"""
        # Map to the existing orchestrate_complex_task method
        return await self.orchestrate_complex_task(
            task_type=task_type,
            requirements={
                **requirements,
                "description": description
            },
            priority=TaskPriority.MEDIUM
        )

    async def create_music(self, description: str, duration: int = 30) -> Dict:
        """Create music using the music generation service"""
        # Map to generate_music for compatibility
        emotion = "energetic"  # Default emotion

        # Simple emotion detection from description
        if any(word in description.lower() for word in ["sad", "melancholy", "slow"]):
            emotion = "melancholy"
        elif any(word in description.lower() for word in ["epic", "energetic", "upbeat"]):
            emotion = "energetic"
        elif any(word in description.lower() for word in ["peaceful", "calm", "relaxing"]):
            emotion = "peaceful"
        elif any(word in description.lower() for word in ["mysterious", "dark"]):
            emotion = "mysterious"

        return await self.generate_music(
            prompt=description,
            emotion=emotion,
            duration=duration
        )

    async def generate_music(self, prompt: str, emotion: str = "energetic",
                           priority: TaskPriority = TaskPriority.MEDIUM,
                           duration: int = 30) -> Dict:
        """Generate music with resilience"""
        # Map emotion strings to valid enums
        emotion_map = {
            "happy": "energetic",
            "sad": "melancholy",
            "calm": "peaceful",
            "energetic": "energetic",
            "peaceful": "peaceful",
            "melancholy": "melancholy",
            "mysterious": "mysterious"
        }

        emotion = emotion_map.get(emotion, "energetic")

        task = {
            "id": hashlib.md5(f"{prompt}{time.time()}".encode()).hexdigest()[:8],
            "type": "music_generation",
            "service": "music",
            "priority": priority,
            "complexity": "high",
            "payload": {
                "prompt": prompt,
                "emotion": emotion,
                "duration": duration
            }
        }

        # Add to queue for processing
        await self.task_queue.put(task)

        # Process with timeout and fallback
        try:
            result = await asyncio.wait_for(
                self._generate_music_with_fallback(task["payload"]),
                timeout=60
            )

            return {
                **result,
                "task_id": task["id"],
                "compute_location": "local"
            }
        except asyncio.TimeoutError:
            logger.warning(f"Music generation timeout for {prompt}")
            return {
                "success": False,
                "error": "Music generation timeout",
                "task_id": task["id"],
                "fallback": "Using cached music or simple generation"
            }

    async def get_service_status(self, service: str) -> Dict:
        """Get status of a specific service"""
        service_map = {
            "comfyui": "http://localhost:8188",
            "voice": "http://localhost:8312",
            "music": "http://localhost:8308",
            "anime": "http://localhost:8305",
            "kb": "http://localhost:8307"
        }

        if service not in service_map:
            return {
                "service": service,
                "success": False,
                "error": f"Unknown service: {service}",
                "status": "unknown"
            }

        url = service_map[service]
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/health", timeout=5) as response:
                    if response.status == 200:
                        return {
                            "service": service,
                            "success": True,
                            "status": "healthy",
                            "url": url,
                            "response_time": 0,
                            "circuit_state": self.circuit_breakers.get(service, CircuitBreaker()).state if service in self.circuit_breakers else "N/A"
                        }
                    else:
                        return {
                            "service": service,
                            "success": False,
                            "status": "unhealthy",
                            "url": url,
                            "http_status": response.status
                        }
        except Exception as e:
            return {
                "service": service,
                "success": False,
                "status": "error",
                "url": url,
                "error": str(e)
            }


# Global orchestrator instance
resilient_orchestrator = None

def initialize_orchestrator(firebase_config: Optional[Dict] = None):
    """Initialize the resilient orchestrator"""
    global resilient_orchestrator
    resilient_orchestrator = ResilientOrchestrator(firebase_config)
    logger.info("Resilient orchestrator initialized")
    return resilient_orchestrator