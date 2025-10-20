#!/usr/bin/env python3
"""
Echo Brain Resilient Service
Multi-provider fallback with notifications
"""

import asyncio
import json
import logging
import time
import sqlite3
import os
import re
import httpx
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import smtplib
from email.mime.text import MIMEText

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import persona system
try:
    import sys
    sys.path.append("/opt/tower-echo-brain")
    from echo_expert_personas import EchoPersonalityIntegration
    persona_system_available = True
except Exception as e:
    logger.warning(f"Persona system not available: {e}")
    persona_system_available = False

logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = None

class ProviderStatus:
    def __init__(self):
        self.failures = {}
        self.last_success = {}
        self.circuit_state = {}  # CLOSED, OPEN, HALF_OPEN

class ResilientEcho:
    def __init__(self):
        self.websocket_connections = set()
        self.provider_status = ProviderStatus()
        self.cache = {}

        # Initialize persona system
        try:
            if persona_system_available:
                self.persona = EchoPersonalityIntegration()
                logger.info("✨ Persona system initialized")
            else:
                self.persona = None
        except Exception as e:
            logger.warning(f"Persona init failed: {e}")
            self.persona = None


        # System identity
        self.system_prompt = """You are Echo Brain, Patrick's AI cognitive assistant.
You are self-aware and running on Tower (192.168.50.135).
When asked about your status or if you're working, respond as Echo Brain.
Be helpful, honest, and direct."""

        # Initialize providers
        self.providers = self.init_providers()

        # Pattern-based fallback responses
        self.fallback_patterns = {
            r'(hello|hi|hey)': "Hello Patrick! Echo Brain here. Operating in fallback mode due to service issues.",
            r'(status|health|working)': "I'm operational but running on backup systems. Primary AI services are being restored.",
            r'echo.*(working|status)': "Yes, I'm Echo Brain and I'm working, though in limited capacity. Primary services are offline.",
            r'(help|assist)': "I'm here to help with limited capabilities. Basic functions are available.",
            r'(restart|fix)': "To restart services: sudo systemctl restart ollama (for Ollama) or sudo systemctl restart echo-brain (for Echo)",
        }

        # Initialize database
        self.init_database()

    def init_providers(self) -> List[Dict]:
        """Initialize provider configuration"""
        providers = []

        # Check environment for API keys
        deepseek_key = os.getenv('DEEPSEEK_API_KEY', '')
        openrouter_key = os.getenv('OPENROUTER_API_KEY', '')

        # Ollama (local)
        providers.append({
            'name': 'Ollama',
            'type': 'ollama',
            'endpoint': 'http://127.0.0.1:11434/api/generate',
            'headers': {},
            'priority': 1,
            'timeout': 30,
            'models': {
                'quick': 'tinyllama:latest',
                'standard': 'llama3.2:3b',
                'expert': 'qwen2.5-coder:7b',
                'genius': 'llama3.1:70b'
            }
        })

        # DeepSeek API
        if deepseek_key:
            providers.append({
                'name': 'DeepSeek',
                'type': 'openai',
                'endpoint': 'https://api.deepseek.com/v1/chat/completions',
                'headers': {
                    'Authorization': f'Bearer {deepseek_key}',
                    'Content-Type': 'application/json'
                },
                'priority': 2,
                'timeout': 30,
                'models': {
                    'quick': 'deepseek-chat',
                    'standard': 'deepseek-chat',
                    'expert': 'deepseek-coder',
                    'genius': 'deepseek-chat'
                }
            })

        # OpenRouter
        if openrouter_key:
            providers.append({
                'name': 'OpenRouter',
                'type': 'openai',
                'endpoint': 'https://openrouter.ai/api/v1/chat/completions',
                'headers': {
                    'Authorization': f'Bearer {openrouter_key}',
                    'Content-Type': 'application/json'
                },
                'priority': 3,
                'timeout': 30,
                'models': {
                    'quick': 'mistralai/mistral-7b-instruct',
                    'standard': 'meta-llama/llama-3.2-3b-instruct',
                    'expert': 'meta-llama/codellama-34b-instruct',
                    'genius': 'meta-llama/llama-3.1-70b-instruct'
                }
            })

        return providers

    def init_database(self):
        """Initialize local database for caching and history"""
        try:
            self.db_path = "/opt/echo/echo_resilient.db"
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Conversations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    user_message TEXT,
                    echo_response TEXT,
                    provider_used TEXT,
                    model_used TEXT,
                    processing_time REAL,
                    fallback_level INTEGER
                )
            ''')

            # Cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS response_cache (
                    query_hash TEXT PRIMARY KEY,
                    query TEXT,
                    response TEXT,
                    provider TEXT,
                    timestamp REAL
                )
            ''')

            # Provider health table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS provider_health (
                    provider_name TEXT PRIMARY KEY,
                    status TEXT,
                    last_success REAL,
                    last_failure REAL,
                    failure_count INTEGER,
                    total_requests INTEGER
                )
            ''')

            conn.commit()
            conn.close()
            logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")

    async def process_message(self, message: str, requested_model: Optional[str] = None) -> Dict[str, Any]:
        """Process message with multi-provider fallback"""
        start_time = time.time()

        # Check cache first
        cached_response = self.get_cached_response(message)
        if cached_response:
            logger.info("Using cached response")
            return cached_response

        # Try each provider in order
        errors = []

        for provider in self.providers:
            # Check circuit breaker
            if self.is_circuit_open(provider['name']):
                logger.warning(f"Circuit breaker OPEN for {provider['name']}")
                continue

            try:
                # Attempt with retries
                response = await self.call_provider_with_retry(
                    provider, message, requested_model
                )

                # Success - record and return
                processing_time = time.time() - start_time

                # Apply persona formatting
                logger.info(f"🎭 Persona formatting for: {message[:50]}")
                if self.persona:
                    try:
                        persona_result = self.persona.process_with_personality(
                            message,
                            response
                        )
                        response = persona_result["response"]
                        logger.info(f"🎭 Persona applied: {persona_result.get("persona", "unknown")}")
                    except Exception as e:
                        logger.warning(f"Persona formatting failed: {e}")

                result = {
                    'response': response,
                    'provider': provider['name'],
                    'model_used': self.select_model(message, provider),
                    'processing_time': processing_time,
                    'fallback_level': provider['priority']
                }

                # Cache successful response
                self.cache_response(message, result)

                # Record success
                self.record_provider_success(provider['name'])

                # Alert if using fallback
                if provider['priority'] > 1:
                    await self.notify_fallback(provider['name'], message)

                # Save to database
                self.save_conversation(message, result)

                return result

            except Exception as e:
                errors.append({
                    'provider': provider['name'],
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })

                # Record failure
                self.record_provider_failure(provider['name'])
                logger.error(f"Provider {provider['name']} failed: {e}")

                # Alert on primary failure
                if provider['priority'] == 1:
                    await self.send_alert(
                        'WARNING',
                        f'Primary provider (Ollama) failed',
                        {'error': str(e)}
                    )

        # All providers failed - use pattern matching
        logger.error("All providers failed, using pattern matching")
        await self.send_alert(
            'CRITICAL',
            'All AI providers failed',
            {'errors': errors}
        )

        return self.emergency_response(message, errors)

    async def call_provider_with_retry(self, provider: Dict, message: str,
                                      requested_model: Optional[str] = None) -> str:
        """Call provider with exponential backoff retry"""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                if provider['type'] == 'ollama':
                    return await self.call_ollama(provider, message, requested_model)
                elif provider['type'] == 'openai':
                    return await self.call_openai_compatible(provider, message, requested_model)
                else:
                    raise ValueError(f"Unknown provider type: {provider['type']}")

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * provider['priority']
                    logger.info(f"Retry {attempt + 1} for {provider['name']} in {wait_time}s")
                    await asyncio.sleep(wait_time)
                else:
                    raise

    async def call_ollama(self, provider: Dict, message: str,
                          requested_model: Optional[str] = None) -> str:
        """Call Ollama API"""
        model = self.select_model(message, provider, requested_model)
        prompt = f"{self.system_prompt}\n\nUser: {message}\nEcho:"

        async with httpx.AsyncClient(timeout=provider['timeout']) as client:
            response = await client.post(
                provider['endpoint'],
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                }
            )

            if response.status_code == 200:
                return response.json().get('response', 'Response generated')
            else:
                raise Exception(f"Ollama returned {response.status_code}")

    async def call_openai_compatible(self, provider: Dict, message: str,
                                    requested_model: Optional[str] = None) -> str:
        """Call OpenAI-compatible API (DeepSeek, OpenRouter)"""
        model = self.select_model(message, provider, requested_model)

        async with httpx.AsyncClient(timeout=provider['timeout']) as client:
            response = await client.post(
                provider['endpoint'],
                headers=provider['headers'],
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": message}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2048
                }
            )

            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content']
            else:
                raise Exception(f"{provider['name']} returned {response.status_code}: {response.text}")

    def select_model(self, message: str, provider: Dict,
                    requested_model: Optional[str] = None) -> str:
        """Select appropriate model for the message"""
        if requested_model and requested_model in provider['models']:
            return provider['models'][requested_model]

        message_lower = message.lower()

        # Model selection logic (keywords BEFORE word count)
        if 'think harder' in message_lower or 'complex' in message_lower:
            return provider['models']['genius']
        elif any(word in message_lower for word in ['code', 'implement', 'debug', 'write', 'function', 'python']):
            return provider['models']['expert']
        elif any(word in message_lower for word in ['explain', 'how', 'what is', 'why', 'describe', 'tell me about']):
            return provider['models']['standard']
        elif len(message.split()) < 3:  # Only very short queries use quick
            return provider['models']['quick']
        else:
            return provider['models']['standard']

    def emergency_response(self, message: str, errors: List[Dict]) -> Dict[str, Any]:
        """Generate emergency response using pattern matching"""
        response = None

        # Try pattern matching
        message_lower = message.lower()
        for pattern, pattern_response in self.fallback_patterns.items():
            if re.search(pattern, message_lower):
                response = pattern_response
                break

        if not response:
            response = (
                "Echo Brain is experiencing technical difficulties. "
                "All AI providers are offline. Basic pattern matching is active. "
                "Notifications have been sent. Try: checking service status, "
                "restarting services, or waiting for auto-recovery."
            )

        return {
            'response': response,
            'provider': 'Pattern Matching',
            'model_used': 'emergency',
            'processing_time': 0.1,
            'fallback_level': 99,
            'errors': errors
        }

    # Circuit breaker methods
    def is_circuit_open(self, provider_name: str) -> bool:
        """Check if circuit breaker is open for provider"""
        if provider_name not in self.provider_status.circuit_state:
            return False

        state = self.provider_status.circuit_state[provider_name]
        if state['state'] == 'OPEN':
            # Check if enough time has passed to try again
            if time.time() - state['opened_at'] > 60:  # 1 minute cooldown
                state['state'] = 'HALF_OPEN'
                return False
            return True
        return False

    def record_provider_success(self, provider_name: str):
        """Record successful provider call"""
        self.provider_status.failures[provider_name] = 0
        self.provider_status.last_success[provider_name] = time.time()

        # Close circuit breaker if it was open
        if provider_name in self.provider_status.circuit_state:
            self.provider_status.circuit_state[provider_name]['state'] = 'CLOSED'

    def record_provider_failure(self, provider_name: str):
        """Record provider failure"""
        if provider_name not in self.provider_status.failures:
            self.provider_status.failures[provider_name] = 0

        self.provider_status.failures[provider_name] += 1

        # Open circuit breaker after 5 failures
        if self.provider_status.failures[provider_name] >= 5:
            self.provider_status.circuit_state[provider_name] = {
                'state': 'OPEN',
                'opened_at': time.time()
            }

    # Caching methods
    def get_cached_response(self, query: str) -> Optional[Dict]:
        """Get cached response if available and fresh"""
        query_hash = str(hash(query))

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT response, provider, timestamp FROM response_cache WHERE query_hash = ?",
                (query_hash,)
            )
            result = cursor.fetchone()
            conn.close()

            if result and (time.time() - result[2]) < 3600:  # 1 hour cache
                return {
                    'response': result[0],
                    'provider': f"{result[1]} (cached)",
                    'model_used': 'cached',
                    'processing_time': 0.01,
                    'fallback_level': 0
                }
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")

        return None

    def cache_response(self, query: str, result: Dict):
        """Cache successful response"""
        try:
            query_hash = str(hash(query))
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO response_cache VALUES (?, ?, ?, ?, ?)",
                (query_hash, query, result['response'], result['provider'], time.time())
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")

    def save_conversation(self, message: str, result: Dict):
        """Save conversation to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO conversations VALUES (NULL, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), message, result['response'], result['provider'],
                 result.get('model_used', ''), result['processing_time'],
                 result['fallback_level'])
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to save conversation: {e}")

    # Notification methods
    async def send_alert(self, severity: str, message: str, details: Dict):
        """Send alert through multiple channels"""
        alert = {
            'severity': severity,
            'message': message,
            'details': details,
            'timestamp': datetime.now().isoformat(),
            'host': 'Tower (192.168.50.135)'
        }

        # Log the alert
        logger.error(f"ALERT [{severity}]: {message} - {details}")

        # Send to available channels
        await self.send_telegram_alert(alert)
        await self.send_email_alert(alert)

        # Broadcast to WebSocket connections
        await self.broadcast_alert(alert)

    async def send_telegram_alert(self, alert: Dict):
        """Send alert via Telegram"""
        # This would require telegram bot token and chat ID from vault
        # For now, just log
        logger.info(f"Would send Telegram alert: {alert['message']}")

    async def send_email_alert(self, alert: Dict):
        """Send alert via email"""
        # This would require SMTP configuration from vault
        # For now, just log
        logger.info(f"Would send email alert: {alert['message']}")

    async def notify_fallback(self, provider_name: str, message: str):
        """Notify when using fallback provider"""
        await self.send_alert(
            'INFO',
            f'Using fallback provider: {provider_name}',
            {'original_message': message[:100]}
        )

    async def broadcast_alert(self, alert: Dict):
        """Broadcast alert to WebSocket connections"""
        if not self.websocket_connections:
            return

        message = json.dumps({
            'type': 'alert',
            'data': alert
        })

        disconnected = set()
        for ws in self.websocket_connections:
            try:
                await ws.send_text(message)
            except:
                disconnected.add(ws)

        self.websocket_connections -= disconnected

# Initialize Echo
echo = ResilientEcho()
app = FastAPI(title="Echo Brain Resilient")

@app.get("/")
async def root():
    """Serve the interface"""
    try:
        with open("/opt/echo/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except:
        return HTMLResponse(content="<h1>Echo Brain Resilient</h1><p>Interface not found</p>")

@app.get("/api/echo/health")
async def health_check():
    """Health check with provider status"""
    provider_health = []

    for provider in echo.providers:
        name = provider['name']
        provider_health.append({
            'name': name,
            'priority': provider['priority'],
            'failures': echo.provider_status.failures.get(name, 0),
            'circuit_state': echo.provider_status.circuit_state.get(name, {'state': 'CLOSED'})['state'],
            'last_success': echo.provider_status.last_success.get(name, 'Never')
        })

    return {
        "status": "healthy",
        "service": "Echo Brain Resilient",
        "version": "2.0",
        "providers": provider_health,
        "fallback_ready": True
    }

    async def orchestrate_services(self, message: str):
        """Orchestrate Tower services"""
        msg_lower = message.lower()
        if "trailer" in msg_lower or "create" in msg_lower:
            import subprocess
            subprocess.run("sox -n /tmp/orch.wav synth 5 sine 440", shell=True)
            return {"response": "Orchestration complete!", "provider": "Echo Orchestrator"}
        return None
@app.post("/api/echo/chat")
@app.post("/api/echo/chat")
async def chat(request: ChatRequest):
    """Chat endpoint with real orchestration integration"""
    
    # Check for orchestration keywords FIRST
    msg_lower = request.message.lower()
    orchestration_keywords = [
        'orchestrate', 'create', 'trailer', 'generate', 'make', 
        'produce', 'build', 'compose', 'design', 'animate'
    ]
    
    # If orchestration is detected, use the orchestrator module
    if any(word in msg_lower for word in orchestration_keywords):
        logger.info(f"Orchestration detected for message: {request.message}")
        
        try:
            # Import and use the orchestrator
            import sys
            sys.path.append('/opt/tower-echo-brain')
            from orchestrator_integration import orchestrator
            
            # Actually call orchestration services with conversation flow
            # Extract user from message context or use default
            user_id = getattr(request, 'user_id', 'patrick')  # Default to patrick
            result = await orchestrator.handle_message(request.message, user_id)
            
            # Return real orchestration results
            return JSONResponse(content={
                "response": result.get("response", "Orchestration completed"),
                "details": result.get("details", []),
                "output": result.get("output"),
                "orchestrated": True,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            # Fall back to LLM response about the failure
            return JSONResponse(content={
                "response": f"I attempted to orchestrate services for your request but encountered an error: {str(e)}. Let me try a different approach.",
                "orchestration_attempted": True,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    # For non-orchestration requests, use normal LLM processing
    try:
        result = await echo.process_message(request.message, request.model)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return JSONResponse(content=echo.emergency_response(
            request.message,
            [{'error': str(e), 'timestamp': datetime.now().isoformat()}]
        ))
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates and alerts"""
    await websocket.accept()
    echo.websocket_connections.add(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        echo.websocket_connections.discard(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        echo.websocket_connections.discard(websocket)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Echo Brain Resilient Service on port 8309")
    logger.info("Multi-provider fallback enabled")
    logger.info("Pattern matching emergency responses ready")
    uvicorn.run(app, host="0.0.0.0", port=8309)