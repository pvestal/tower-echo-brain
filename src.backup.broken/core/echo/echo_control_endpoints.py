#!/usr/bin/env python3
"""
Echo Brain Control Endpoints for Vue3 Dashboard
Complete refactor with all missing endpoints
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from typing import Dict, Any, List
from datetime import datetime
import asyncio
import json
import aiohttp
import psutil
from dataclasses import dataclass, asdict

@dataclass
class EchoBrainState:
    """Echo Brain state for dashboard visualization"""
    current_thought: str = None
    thought_type: str = "idle"  # idle, logic, creative, analysis
    activity_level: int = 0
    recent_queries: int = 0
    response_time: float = 0.0
    neurons_active: int = 0
    left_hemisphere_active: bool = False
    right_hemisphere_active: bool = False
    connections_active: List[Dict] = None
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    last_update: str = None

class EchoControlAPI:
    """Main control API for Echo Brain dashboard"""

    def __init__(self):
        self.app = FastAPI(title="Echo Brain Control System")
        self.state = EchoBrainState()
        self.websocket_clients: List[WebSocket] = []
        self.setup_middleware()
        self.setup_routes()

        # Service endpoints to monitor
        self.service_endpoints = {
            "echo": ("http://127.0.0.1:8309", "Echo Brain"),
            "kb": ("http://127.0.0.1:8307", "Knowledge Base"),
            "comfyui": ("http://127.0.0.1:8188", "ComfyUI"),
            "anime": ("http://127.0.0.1:8328", "Anime Service"),
            "music": ("http://127.0.0.1:8315", "Music Service"),
            "voice": ("http://127.0.0.1:8312", "Voice Service"),
            "auth": ("http://127.0.0.1:8088", "Auth Service"),
            "vault": ("http://127.0.0.1:8200", "Vault")
        }

    def setup_middleware(self):
        """Configure CORS for dashboard access"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def setup_routes(self):
        """Setup all control endpoints"""

        @self.app.get("/api/echo/health")
        async def echo_health():
            """Main health endpoint with full state"""
            self.update_system_metrics()
            return JSONResponse({
                "status": "healthy",
                "service": "Echo Brain Enhanced",
                "timestamp": datetime.now().isoformat(),
                **asdict(self.state)
            })

        @self.app.get("/api/health")
        async def simple_health():
            """Simple health check"""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}

        @self.app.get("/api/echo/state")
        async def get_state():
            """Get current Echo state"""
            return asdict(self.state)

        @self.app.post("/api/echo/chat")
        async def chat(request: dict):
            """Process chat message with state updates"""
            message = request.get("message", "")

            # Update state based on message content
            await self.process_message(message)

            # Simulate processing
            start_time = datetime.now()
            await asyncio.sleep(0.5)
            response_time = (datetime.now() - start_time).total_seconds()

            self.state.response_time = response_time
            self.state.recent_queries += 1

            # Generate response
            response = {
                "response": f"Processed: {message}",
                "state": asdict(self.state),
                "timestamp": datetime.now().isoformat()
            }

            # Notify WebSocket clients
            await self.broadcast_state()

            return response

        @self.app.post("/api/echo/control")
        async def control(command: dict):
            """Control commands from dashboard"""
            cmd = command.get("command", "")
            params = command.get("params", {})

            if cmd == "activate_neuron":
                neuron_id = params.get("id", 0)
                await self.activate_neuron(neuron_id)
            elif cmd == "reset":
                await self.reset_state()
            elif cmd == "simulate_thought":
                thought_type = params.get("type", "analysis")
                await self.simulate_thought(thought_type)

            return {"success": True, "state": asdict(self.state)}

        @self.app.websocket("/ws/dashboard")
        async def dashboard_websocket(websocket: WebSocket):
            """WebSocket for real-time updates"""
            await websocket.accept()
            self.websocket_clients.append(websocket)

            try:
                # Send initial state
                await websocket.send_json({
                    "type": "initial",
                    "state": asdict(self.state)
                })

                # Keep connection alive
                while True:
                    data = await websocket.receive_text()
                    if data == "ping":
                        await websocket.send_text("pong")

            except WebSocketDisconnect:
                self.websocket_clients.remove(websocket)

        @self.app.get("/api/echo/services")
        async def check_services():
            """Check all service health statuses"""
            services = {}

            for name, (url, display_name) in self.service_endpoints.items():
                services[name] = await self.check_service_health(url, display_name)

            return services

        @self.app.get("/api/{service}/health")
        async def service_health(service: str):
            """Proxy health check for any service"""
            if service in self.service_endpoints:
                url, display_name = self.service_endpoints[service]
                return await self.check_service_health(url, display_name)
            return {"status": "unknown", "service": service}

        @self.app.get("/api/echo/metrics")
        async def get_metrics():
            """Get system metrics for dashboard"""
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "network_connections": len(psutil.net_connections()),
                "processes": len(psutil.pids()),
                "timestamp": datetime.now().isoformat()
            }

    async def process_message(self, message: str):
        """Process message and update brain state"""
        message_lower = message.lower()

        # Determine thought type
        if any(word in message_lower for word in ["analyze", "calculate", "debug", "logic"]):
            self.state.thought_type = "logic"
            self.state.left_hemisphere_active = True
            self.state.right_hemisphere_active = False
            self.state.neurons_active = 20
        elif any(word in message_lower for word in ["create", "generate", "imagine", "design"]):
            self.state.thought_type = "creative"
            self.state.left_hemisphere_active = False
            self.state.right_hemisphere_active = True
            self.state.neurons_active = 25
        else:
            self.state.thought_type = "analysis"
            self.state.left_hemisphere_active = True
            self.state.right_hemisphere_active = True
            self.state.neurons_active = 15

        self.state.current_thought = f"Processing: {message[:50]}..."
        self.state.activity_level = min(100, self.state.activity_level + 20)
        self.state.last_update = datetime.now().isoformat()

        # Schedule activity decrease
        asyncio.create_task(self.decrease_activity())

    async def decrease_activity(self):
        """Gradually decrease activity level"""
        await asyncio.sleep(3)
        self.state.activity_level = max(0, self.state.activity_level - 10)
        self.state.neurons_active = max(0, self.state.neurons_active - 5)

        if self.state.activity_level == 0:
            self.state.thought_type = "idle"
            self.state.current_thought = None
            self.state.left_hemisphere_active = False
            self.state.right_hemisphere_active = False

        await self.broadcast_state()

    async def activate_neuron(self, neuron_id: int):
        """Activate specific neuron"""
        self.state.neurons_active = min(40, self.state.neurons_active + 1)
        self.state.activity_level = min(100, self.state.activity_level + 5)
        await self.broadcast_state()

    async def reset_state(self):
        """Reset Echo state to idle"""
        self.state = EchoBrainState()
        self.update_system_metrics()
        await self.broadcast_state()

    async def simulate_thought(self, thought_type: str):
        """Simulate a thought process"""
        thoughts = {
            "logic": "Analyzing logical patterns...",
            "creative": "Generating creative concepts...",
            "analysis": "Processing complex analysis...",
            "memory": "Retrieving memory patterns..."
        }

        self.state.thought_type = thought_type
        self.state.current_thought = thoughts.get(thought_type, "Thinking...")
        self.state.activity_level = 80
        self.state.neurons_active = 30

        if thought_type in ["logic", "analysis"]:
            self.state.left_hemisphere_active = True
        else:
            self.state.right_hemisphere_active = True

        await self.broadcast_state()

    def update_system_metrics(self):
        """Update system resource metrics"""
        self.state.cpu_usage = psutil.cpu_percent()
        self.state.memory_usage = psutil.virtual_memory().percent
        self.state.last_update = datetime.now().isoformat()

    async def broadcast_state(self):
        """Broadcast state to all WebSocket clients"""
        message = {
            "type": "state_update",
            "state": asdict(self.state),
            "timestamp": datetime.now().isoformat()
        }

        for client in self.websocket_clients[:]:
            try:
                await client.send_json(message)
            except:
                self.websocket_clients.remove(client)

    async def check_service_health(self, url: str, display_name: str):
        """Check health of a service"""
        try:
            async with aiohttp.ClientSession() as session:
                health_url = f"{url}/health" if not url.endswith("/health") else url
                async with session.get(health_url, timeout=2) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "status": "healthy",
                            "service": display_name,
                            "details": data
                        }
                    else:
                        return {
                            "status": "degraded",
                            "service": display_name,
                            "code": response.status
                        }
        except asyncio.TimeoutError:
            return {
                "status": "timeout",
                "service": display_name
            }
        except Exception as e:
            return {
                "status": "offline",
                "service": display_name,
                "error": str(e)
            }

def create_app() -> FastAPI:
    """Create and configure the FastAPI app"""
    control_api = EchoControlAPI()
    return control_api.app

# HTML for testing dashboard connectivity
dashboard_test_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Echo Control Test</title>
    <style>
        body { background: #1a1a1a; color: #0ff; font-family: monospace; padding: 20px; }
        .status { margin: 10px 0; padding: 10px; background: #0a0a0a; border: 1px solid #0ff; }
        .healthy { border-color: #0f0; }
        .offline { border-color: #f00; }
        button { background: #0a0a0a; color: #0ff; border: 1px solid #0ff; padding: 10px; cursor: pointer; }
        button:hover { background: #0ff; color: #000; }
    </style>
</head>
<body>
    <h1>Echo Brain Control Test</h1>
    <div id="status" class="status">Connecting...</div>
    <button onclick="testChat()">Test Chat</button>
    <button onclick="testControl()">Test Control</button>
    <button onclick="checkServices()">Check Services</button>
    <div id="results"></div>

    <script>
        async function checkHealth() {
            try {
                const response = await fetch('/api/echo/health');
                const data = await response.json();
                document.getElementById('status').innerHTML = `
                    Status: ${data.status}<br>
                    Thought: ${data.current_thought || 'idle'}<br>
                    Activity: ${data.activity_level}%
                `;
                document.getElementById('status').className = 'status healthy';
            } catch (error) {
                document.getElementById('status').innerHTML = 'Offline: ' + error;
                document.getElementById('status').className = 'status offline';
            }
        }

        async function testChat() {
            const response = await fetch('/api/echo/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: 'Test message from dashboard'})
            });
            const data = await response.json();
            document.getElementById('results').innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
        }

        async function testControl() {
            const response = await fetch('/api/echo/control', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({command: 'simulate_thought', params: {type: 'creative'}})
            });
            const data = await response.json();
            document.getElementById('results').innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
        }

        async function checkServices() {
            const response = await fetch('/api/echo/services');
            const data = await response.json();
            document.getElementById('results').innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
        }

        // Check health on load
        checkHealth();
        setInterval(checkHealth, 2000);
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    app = create_app()

    # Add test endpoint
    @app.get("/test")
    async def test_page():
        return HTMLResponse(dashboard_test_html)

    print("Starting Echo Control API...")
    print("Dashboard test page: http://localhost:8309/test")
    print("Health endpoint: http://localhost:8309/api/echo/health")
    uvicorn.run(app, host="0.0.0.0", port=8309)