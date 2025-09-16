#!/usr/bin/env python3
"""
Echo Brain Unified Service - Refactored Architecture
Consolidates 5 fragmented Echo services into single intelligent router
Implements dynamic model escalation from 1B to 70B parameters
"""

import asyncio
import aiohttp
import logging
import json
import psycopg2
import uuid
import re
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from echo_brain_thoughts import echo_brain
from echo_autonomous_evolution import echo_autonomous_evolution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    context: Optional[Dict] = {}
    intelligence_level: Optional[str] = "auto"
    user_id: Optional[str] = "default"  # For conversation tracking
    conversation_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    model_used: str
    intelligence_level: str
    processing_time: float
    escalation_path: List[str]
    requires_clarification: bool = False
    clarifying_questions: List[str] = []
    conversation_id: str
    intent: Optional[str] = None
    confidence: float = 0.0

class EchoIntelligenceRouter:
    """Core intelligence routing system for Echo Brain"""
    
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_hierarchy = {
            "quick": "tinyllama:latest",        # 1B parameters
            "standard": "llama3.2:3b",          # 3B parameters
            "professional": "mistral:7b",       # 7B parameters
            "expert": "qwen2.5-coder:32b",     # 32B parameters  
            "genius": "llama3.1:70b",          # 70B parameters
        }
        self.specialized_models = {
            "coding": "deepseek-coder-v2:16b",
            "creative": "mixtral:8x7b", 
            "analysis": "codellama:70b"
        }
        self.escalation_history = []
        
    def analyze_complexity(self, query: str, context: Dict) -> str:
        """Analyze query complexity to determine optimal model"""
        complexity_score = 0.0
        
        # Basic query analysis
        complexity_score += len(query.split()) * 0.3
        complexity_score += query.count('?') * 3
        complexity_score += query.count('.') * 1
        
        # Technical complexity indicators
        technical_terms = [
            'database', 'architecture', 'algorithm', 'implementation',
            'refactor', 'optimization', 'integration', 'system'
        ]
        complexity_score += sum(8 for term in technical_terms if term.lower() in query.lower())
        
        # Programming language detection
        code_terms = ['python', 'javascript', 'sql', 'function', 'class', 'async']
        if any(term in query.lower() for term in code_terms):
            complexity_score += 15
            
        # Context complexity
        if context.get('previous_failures', 0) > 0:
            complexity_score += 20  # Escalate if previous attempts failed
            
        if context.get('user_expertise') == 'expert':
            complexity_score += 10
            
        # Route based on score (adjusted for better escalation)
        if complexity_score < 8:
            return "quick"
        elif complexity_score < 25: 
            return "standard"
        elif complexity_score < 40:
            return "professional"
        elif complexity_score < 60:
            return "expert" 
        else:
            return "genius"
    
    def detect_specialization(self, query: str) -> Optional[str]:
        """Detect if query requires specialized model"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['code', 'program', 'function', 'debug']):
            return "coding"
        elif any(term in query_lower for term in ['creative', 'story', 'write', 'imagine']):
            return "creative"  
        elif any(term in query_lower for term in ['analyze', 'data', 'research', 'study']):
            return "analysis"
        return None
    
    def _calculate_complexity_score(self, query: str, context: Dict) -> float:
        """Calculate numerical complexity score for brain visualization"""
        score = 0.0
        
        # Basic query analysis
        score += len(query.split()) * 0.3
        score += query.count('?') * 3
        score += query.count('.') * 1
        
        # Technical complexity indicators (increased scoring)
        technical_terms = [
            'database', 'architecture', 'algorithm', 'implementation',
            'refactor', 'optimization', 'integration', 'system', 'distributed',
            'microservice', 'scalable', 'performance', 'design', 'patterns'
        ]
        score += sum(12 for term in technical_terms if term.lower() in query.lower())
        
        # Programming language detection (increased weight)
        code_terms = ['python', 'javascript', 'sql', 'function', 'class', 'async']
        if any(term in query.lower() for term in code_terms):
            score += 20
            
        # Context complexity
        if context.get('previous_failures', 0) > 0:
            score += 20
            
        return min(score, 100.0)
    
    async def query_model(self, model: str, prompt: str, max_tokens: int = 2048) -> Dict:
        """Query specific Ollama model"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.7
                    }
                }
                
                start_time = asyncio.get_event_loop().time()
                async with session.post(self.ollama_url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        processing_time = asyncio.get_event_loop().time() - start_time
                        return {
                            "success": True,
                            "response": result.get("response", ""),
                            "processing_time": processing_time,
                            "model": model
                        }
                    else:
                        return {"success": False, "error": f"HTTP {response.status}"}
                        
        except Exception as e:
            logger.error(f"Model query failed for {model}: {e}")
            return {"success": False, "error": str(e)}
    
    async def progressive_escalation(self, query: str, context: Dict) -> Dict:
        """Implement progressive model escalation with visible brain activity"""
        escalation_path = []
        
        # 🧠 START THINKING - Begin neural visualization
        thought_id = await echo_brain.start_thinking("query_processing", query)
        
        # 🧠 PROCESS INPUT
        await echo_brain.think_about_input(thought_id, query)
        
        # Start with complexity-based model selection
        initial_level = self.analyze_complexity(query, context)
        complexity_score = self._calculate_complexity_score(query, context)
        
        # 🧠 ANALYZE COMPLEXITY 
        await echo_brain.analyze_complexity(thought_id, complexity_score / 100.0)
        
        # Check for specialization override
        specialization = self.detect_specialization(query)
        if specialization:
            model = self.specialized_models[specialization]
            escalation_path.append(f"specialized_{specialization}")
            # 🧠 DECISION: SPECIALIZED MODEL
            await echo_brain.make_decision(thought_id, "specialization", f"Using {specialization} specialist")
        else:
            model = self.model_hierarchy[initial_level]
            escalation_path.append(initial_level)
            # 🧠 DECISION: STANDARD MODEL
            await echo_brain.make_decision(thought_id, "escalation", f"Using {initial_level} intelligence level")
        
        # 🧠 GENERATE RESPONSE
        await echo_brain.generate_response(thought_id, f"{initial_level} response")
        
        # Attempt query
        result = await self.query_model(model, query)
        
        if result["success"]:
            # 🧠 SUCCESS - Complete thinking
            await echo_brain.finish_thinking(thought_id)
            return {
                **result,
                "intelligence_level": initial_level,
                "escalation_path": escalation_path,
                "thought_id": thought_id,
                "brain_activity": echo_brain.get_brain_state()
            }
        else:
            # 🧠 ESCALATION NEEDED
            await echo_brain.emotional_response(thought_id, "concern", "Initial model failed")
            
            # Escalation fallback
            if initial_level != "genius":
                logger.info(f"Escalating from {initial_level} to genius mode")
                escalation_path.append("genius")
                
                # 🧠 ESCALATION DECISION
                await echo_brain.make_decision(thought_id, "escalation", "Upgrading to genius level")
                
                fallback_result = await self.query_model(self.model_hierarchy["genius"], query)
                await echo_brain.finish_thinking(thought_id)
                
                return {
                    **fallback_result, 
                    "intelligence_level": "genius",
                    "escalation_path": escalation_path,
                    "thought_id": thought_id,
                    "brain_activity": echo_brain.get_brain_state()
                }
            else:
                await echo_brain.finish_thinking(thought_id)
                return {
                    **result,
                    "thought_id": thought_id,
                    "brain_activity": echo_brain.get_brain_state()
                }

class ConversationManager:
    """Manages conversation context and intent recognition"""
    
    def __init__(self):
        self.conversations = {}  # In-memory for now, can move to Redis/DB later
        self.intent_patterns = {
            "code_modification": [
                r"modify|change|update|edit|fix.*code|file",
                r"add.*function|method|class",
                r"refactor|rewrite"
            ],
            "debugging": [
                r"error|bug|issue|problem|fail|crash",
                r"debug|troubleshoot|diagnose",
                r"not working|broken"
            ],
            "architecture": [
                r"design|architecture|structure|organize",
                r"system|framework|pattern",
                r"how.*build|implement.*system"
            ],
            "ci_cd": [
                r"deploy|deployment|ci/cd|pipeline",
                r"build|test.*automation",
                r"git|version control|workflow"
            ],
            "explanation": [
                r"what.*is|how.*does|explain|understand",
                r"learn|tutorial|guide",
                r"difference between"
            ],
            "delegation": [
                r"delegate.*to|assign.*to|route.*to",
                r"send.*to.*agent|forward.*to",
                r"security.*agent|architect.*agent",
                r"have.*agent.*do|agent.*handle"
            ]
        }
        
        self.clarifying_questions = {
            "code_modification": [
                "Which specific files need to be modified?",
                "What exact functionality should be added or changed?",
                "Are there any constraints or requirements I should know about?",
                "Should this integrate with existing code patterns?"
            ],
            "debugging": [
                "What specific error message are you seeing?",
                "When does this error occur (what triggers it)?",
                "What was the last working state?",
                "Have you tried any solutions already?"
            ],
            "delegation": [
                "Which agent should handle this task?",
                "What type of task should this be classified as (design, security, development, etc.)?",
                "What priority level should this task have?",
                "Should this task be monitored for completion?"
            ],
            "architecture": [
                "What's the main goal of this system?",
                "Are there performance or scalability requirements?",
                "Should this integrate with your existing Tower services?",
                "What's your timeline for implementation?"
            ],
            "ci_cd": [
                "What type of project are we deploying?",
                "Do you have existing CI/CD infrastructure?",
                "What environments need to be supported?",
                "Are there specific testing requirements?"
            ],
            "explanation": [
                "What's your current level of understanding with this topic?",
                "Are you looking for a high-level overview or technical details?",
                "Is this for a specific project or general learning?",
                "Do you prefer code examples or conceptual explanations?"
            ]
        }
    
    def classify_intent(self, query: str, conversation_history: List[Dict] = []) -> tuple[str, float]:
        """Classify user intent with confidence score"""
        query_lower = query.lower()
        
        # Check conversation context for intent continuation
        if conversation_history:
            last_intent = conversation_history[-1].get("intent")
            if last_intent and any(pattern in query_lower for pattern in ["yes", "no", "continue", "more"]):
                return last_intent, 0.9
        
        # Pattern-based intent classification
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                score += matches * 0.3
            
            if score > 0:
                intent_scores[intent] = min(score, 1.0)
        
        if not intent_scores:
            return "general", 0.5
        
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        return best_intent[0], best_intent[1]
    
    def needs_clarification(self, intent: str, confidence: float, query: str) -> bool:
        """Determine if query needs clarification"""
        # Low confidence always needs clarification
        if confidence < 0.6:
            return True
        
        # Check for vague queries even with good intent classification
        vague_indicators = [
            len(query.split()) < 5,  # Very short queries
            "help" in query.lower() and len(query.split()) < 8,
            query.count("?") == 0 and intent != "explanation",  # No questions but unclear intent
        ]
        
        return any(vague_indicators)
    
    def get_clarifying_questions(self, intent: str, query: str) -> List[str]:
        """Get relevant clarifying questions for intent"""
        questions = self.clarifying_questions.get(intent, [
            "Can you provide more details about what you're trying to accomplish?",
            "What specific aspects would you like me to focus on?",
            "Are there any constraints or preferences I should know about?"
        ])
        
        # Limit to 2-3 most relevant questions to avoid overwhelming
        return questions[:3]
    
    def update_conversation(self, conversation_id: str, user_query: str, 
                          intent: str, response: str, requires_clarification: bool):
        """Update conversation history"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = {
                "history": [],
                "created_at": datetime.now(),
                "last_intent": None
            }
        
        self.conversations[conversation_id]["history"].append({
            "user_query": user_query,
            "intent": intent,
            "response": response,
            "requires_clarification": requires_clarification,
            "timestamp": datetime.now()
        })
        
        self.conversations[conversation_id]["last_intent"] = intent
        
        # Keep conversation history manageable (last 10 interactions)
        if len(self.conversations[conversation_id]["history"]) > 10:
            self.conversations[conversation_id]["history"] = \
                self.conversations[conversation_id]["history"][-10:]
    
    def get_conversation_context(self, conversation_id: str) -> Dict:
        """Get conversation context for better processing"""
        if conversation_id not in self.conversations:
            return {}
        
        conv = self.conversations[conversation_id]
        return {
            "history": conv["history"],
            "last_intent": conv.get("last_intent"),
            "interaction_count": len(conv["history"])
        }

class OrchestratorClient:
    """Client for communicating with the Tower Orchestrator Service"""
    
    def __init__(self, orchestrator_url: str = "http://localhost:8400"):
        self.orchestrator_url = orchestrator_url
        self.task_types = {
            "security": "security",
            "architecture": "design",
            "coding": "development",
            "debugging": "review",
            "deployment": "deployment",
            "monitoring": "monitoring",
            "analysis": "analysis"
        }
    
    async def check_orchestrator_health(self) -> Dict:
        """Check if orchestrator service is available"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.orchestrator_url}/health", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {"available": True, "data": data}
                    return {"available": False, "error": f"HTTP {response.status}"}
        except Exception as e:
            logger.warning(f"Orchestrator not available: {e}")
            return {"available": False, "error": str(e)}
    
    async def list_available_agents(self) -> List[Dict]:
        """Get list of available agents from orchestrator"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.orchestrator_url}/agents", timeout=5) as response:
                    if response.status == 200:
                        return await response.json()
                    return []
        except Exception as e:
            logger.error(f"Failed to get agents: {e}")
            return []
    
    async def delegate_task(self, task_description: str, task_type: str, priority: int = 5) -> Dict:
        """Delegate a task to the orchestrator"""
        try:
            # Map Echo intents to orchestrator task types
            orchestrator_task_type = self.task_types.get(task_type, "analysis")
            
            payload = {
                "type": orchestrator_task_type,
                "description": task_description,
                "priority": priority,
                "requirements": {
                    "source": "echo_brain",
                    "delegated_at": datetime.now().isoformat(),
                    "original_intent": task_type
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.orchestrator_url}/submit_task",
                    json=payload,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"✅ Task delegated to orchestrator: {result['task_id']}")
                        return {
                            "success": True,
                            "task_id": result["task_id"],
                            "status": result["status"],
                            "orchestrator_task_type": orchestrator_task_type
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to delegate task: {error_text}")
                        return {"success": False, "error": error_text}
                        
        except Exception as e:
            logger.error(f"Task delegation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def detect_delegation_intent(self, query: str) -> Optional[Dict]:
        """Detect if a query contains delegation intent and extract task details"""
        delegation_patterns = [
            r"delegate.*?(?:to|this).*?(?:agent|security|architect)",
            r"send.*?(?:to|this).*?(?:security|architect|development).*?agent",
            r"route.*?(?:to|this).*?(?:agent|specialist)",
            r"have.*?(?:agent|security|architect).*?(?:do|handle|process)",
            r"assign.*?(?:to|this).*?(?:agent|team|specialist)"
        ]
        
        # Check for delegation patterns
        for pattern in delegation_patterns:
            if re.search(pattern, query.lower()):
                # Extract task type
                task_type = "analysis"  # default
                
                if any(term in query.lower() for term in ["security", "secure", "audit", "vulnerability"]):
                    task_type = "security"
                elif any(term in query.lower() for term in ["architecture", "design", "structure", "architect"]):
                    task_type = "architecture"
                elif any(term in query.lower() for term in ["code", "coding", "develop", "implement", "programming"]):
                    task_type = "coding"
                elif any(term in query.lower() for term in ["debug", "fix", "error", "bug", "troubleshoot"]):
                    task_type = "debugging"
                elif any(term in query.lower() for term in ["deploy", "deployment", "release"]):
                    task_type = "deployment"
                elif any(term in query.lower() for term in ["monitor", "monitoring", "observe", "metrics"]):
                    task_type = "monitoring"
                
                # Extract priority (default to 5)
                priority = 5
                if any(term in query.lower() for term in ["urgent", "critical", "asap", "immediate"]):
                    priority = 9
                elif any(term in query.lower() for term in ["high priority", "important"]):
                    priority = 7
                elif any(term in query.lower() for term in ["low priority", "when you can"]):
                    priority = 3
                
                return {
                    "is_delegation": True,
                    "task_type": task_type,
                    "priority": priority,
                    "description": query
                }
        
        return None

class EchoDatabase:
    """Unified database manager for Echo learning"""
    
    def __init__(self):
        self.db_config = {
            "host": "192.168.50.135",
            "database": "tower_consolidated", 
            "user": "patrick"
        }
    
    async def log_interaction(self, query: str, response: str, model_used: str, 
                            processing_time: float, escalation_path: List[str],
                            conversation_id: Optional[str] = None, user_id: str = "default",
                            intent: Optional[str] = None, confidence: float = 0.0,
                            requires_clarification: bool = False, 
                            clarifying_questions: Optional[List[str]] = None):
        """Log interaction for learning improvement"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO echo_unified_interactions 
                (query, response, model_used, processing_time, escalation_path, 
                 conversation_id, user_id, intent, confidence, requires_clarification, 
                 clarifying_questions)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (query, response, model_used, processing_time, 
                  json.dumps(escalation_path), conversation_id or "", user_id, intent or "", 
                  confidence, requires_clarification, json.dumps(clarifying_questions or [])))
                  
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Database logging failed: {e}")
    
    async def create_tables_if_needed(self):
        """Create unified Echo tables"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS echo_unified_interactions (
                    id SERIAL PRIMARY KEY,
                    conversation_id VARCHAR(100),
                    user_id VARCHAR(100) DEFAULT 'default',
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    model_used VARCHAR(100) NOT NULL,
                    processing_time FLOAT NOT NULL,
                    escalation_path JSONB,
                    intent VARCHAR(50),
                    confidence FLOAT,
                    requires_clarification BOOLEAN DEFAULT FALSE,
                    clarifying_questions JSONB,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS echo_conversations (
                    id SERIAL PRIMARY KEY,
                    conversation_id VARCHAR(100) UNIQUE NOT NULL,
                    user_id VARCHAR(100) DEFAULT 'default',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_interaction TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    intent_history JSONB,
                    context JSONB
                )
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            logger.info("Echo unified database tables initialized")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")

# FastAPI Application
app = FastAPI(
    title="Echo Brain Unified Service",
    description="Consolidated Echo intelligence with dynamic 1B-70B parameter scaling",
    version="1.0.0"
)

# Global instances
router = EchoIntelligenceRouter()
database = EchoDatabase()
conversation_manager = ConversationManager()
orchestrator_client = OrchestratorClient()
evolution_system = echo_autonomous_evolution

@app.on_event("startup")
async def startup():
    """Initialize service on startup"""
    await database.create_tables_if_needed()
    
    # Start autonomous evolution system
    asyncio.create_task(evolution_system.start_continuous_evolution())
    
    logger.info("🧠 Echo Brain Unified Service - Started")
    logger.info("📊 Intelligence Levels: 1B → 70B parameters")
    logger.info("🚀 Dynamic escalation enabled")
    logger.info("🔄 Autonomous evolution system activated")

@app.get("/api/echo/health")
async def health_check():
    """Health check endpoint with orchestrator integration status"""
    
    # Check orchestrator health
    orchestrator_health = await orchestrator_client.check_orchestrator_health()
    
    return {
        "status": "healthy",
        "service": "Echo Brain Unified with Orchestrator Integration",
        "intelligence_levels": list(router.model_hierarchy.keys()),
        "specialized_models": list(router.specialized_models.keys()),
        "max_parameters": "70B",
        "orchestrator": {
            "available": orchestrator_health["available"],
            "status": orchestrator_health.get("data", {}).get("status") if orchestrator_health["available"] else "unavailable",
            "agents_online": orchestrator_health.get("data", {}).get("agents_online", 0) if orchestrator_health["available"] else 0,
            "delegation_enabled": orchestrator_health["available"]
        },
        "features": {
            "conversational_delegation": True,
            "intent_recognition": True,
            "dynamic_escalation": True,
            "autonomous_evolution": True
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/echo/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Main conversational query processing endpoint with intent recognition"""
    
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Get conversation context
        conv_context = conversation_manager.get_conversation_context(conversation_id)
        
        # Classify intent and confidence
        intent, confidence = conversation_manager.classify_intent(
            request.query, 
            conv_context.get("history", [])
        )
        
        logger.info(f"Intent: {intent}, Confidence: {confidence:.2f}")
        
        # 🎯 Check for delegation intent FIRST
        delegation_info = orchestrator_client.detect_delegation_intent(request.query)
        if delegation_info and delegation_info.get("is_delegation"):
            logger.info(f"🔄 Delegation detected: {delegation_info['task_type']} (Priority: {delegation_info['priority']})")
            
            # Check if orchestrator is available
            orchestrator_health = await orchestrator_client.check_orchestrator_health()
            if not orchestrator_health["available"]:
                # Orchestrator unavailable - handle normally but inform user
                response_text = (f"⚠️ I detected you want to delegate this task, but the orchestrator service "
                               f"is currently unavailable ({orchestrator_health['error']}). "
                               f"I'll handle this myself instead.\n\n")
                
                # Continue with normal processing but add the warning
                request.context["orchestrator_unavailable"] = True
                request.context["delegation_warning"] = response_text
            else:
                # Delegate to orchestrator
                delegation_result = await orchestrator_client.delegate_task(
                    request.query, 
                    delegation_info["task_type"], 
                    delegation_info["priority"]
                )
                
                if delegation_result["success"]:
                    # Task successfully delegated
                    response_text = (
                        f"✅ **Task Delegated Successfully**\n\n"
                        f"**Task ID**: `{delegation_result['task_id']}`\n"
                        f"**Agent Type**: {delegation_result['orchestrator_task_type']}\n"
                        f"**Priority**: {delegation_info['priority']}/10\n"
                        f"**Status**: {delegation_result['status']}\n\n"
                        f"Your task has been routed to the appropriate specialized agent. "
                        f"The agent will process it according to its capabilities and current workload.\n\n"
                        f"**Original Request**: {request.query}"
                    )
                    
                    response = QueryResponse(
                        response=response_text,
                        model_used="orchestrator",
                        intelligence_level="delegation",
                        processing_time=asyncio.get_event_loop().time() - start_time,
                        escalation_path=["delegation", delegation_result['orchestrator_task_type']],
                        requires_clarification=False,
                        clarifying_questions=[],
                        conversation_id=conversation_id,
                        intent="delegation",
                        confidence=1.0
                    )
                    
                    # Update conversation history
                    conversation_manager.update_conversation(
                        conversation_id, request.query, "delegation", response_text, False
                    )
                    
                    # Log interaction
                    await database.log_interaction(
                        request.query, response_text, "orchestrator", 
                        response.processing_time, response.escalation_path,
                        conversation_id, request.user_id, "delegation", 1.0,
                        False, []
                    )
                    
                    return response
                else:
                    # Delegation failed - fall back to normal processing
                    response_text = (f"⚠️ Failed to delegate task to orchestrator ({delegation_result['error']}). "
                                   f"I'll handle this myself instead.\n\n")
                    request.context["delegation_failed"] = True
                    request.context["delegation_warning"] = response_text
        
        # Check if clarification is needed
        needs_clarification = conversation_manager.needs_clarification(
            intent, confidence, request.query
        )
        
        if needs_clarification:
            # Return clarifying questions instead of processing
            clarifying_questions = conversation_manager.get_clarifying_questions(intent, request.query)
            
            response_text = "I want to make sure I understand exactly what you need. Let me ask a few questions:\n\n"
            for i, question in enumerate(clarifying_questions, 1):
                response_text += f"{i}. {question}\n"
            
            response = QueryResponse(
                response=response_text,
                model_used="conversation_manager",
                intelligence_level="clarification",
                processing_time=asyncio.get_event_loop().time() - start_time,
                escalation_path=["clarification"],
                requires_clarification=True,
                clarifying_questions=clarifying_questions,
                conversation_id=conversation_id,
                intent=intent,
                confidence=confidence
            )
            
            # Update conversation history
            conversation_manager.update_conversation(
                conversation_id, request.query, intent, response_text, True
            )
            
            # Log interaction
            await database.log_interaction(
                request.query, response_text, "conversation_manager", 
                response.processing_time, ["clarification"],
                conversation_id, request.user_id, intent, confidence,
                True, clarifying_questions
            )
            
            return response
        
        else:
            # Process with full context for better results
            enhanced_context = {
                **request.context,
                "conversation_history": conv_context.get("history", []),
                "intent": intent,
                "confidence": confidence,
                "user_id": request.user_id
            }
            
            # Process query through intelligence router
            result = await router.progressive_escalation(request.query, enhanced_context)
            
            if not result["success"]:
                raise HTTPException(status_code=500, detail=result.get("error", "Processing failed"))
            
            # Create enhanced response
            response = QueryResponse(
                response=result["response"],
                model_used=result["model"],
                intelligence_level=result["intelligence_level"],
                processing_time=result["processing_time"],
                escalation_path=result["escalation_path"],
                requires_clarification=False,
                clarifying_questions=[],
                conversation_id=conversation_id,
                intent=intent,
                confidence=confidence
            )
            
            # Update conversation history
            conversation_manager.update_conversation(
                conversation_id, request.query, intent, result["response"], False
            )
            
            # Log interaction for learning
            await database.log_interaction(
                request.query, result["response"], result["model"],
                result["processing_time"], result["escalation_path"],
                conversation_id, request.user_id, intent, confidence,
                False, []
            )
            
            return response
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/echo/brain")
async def get_brain_activity():
    """Get current brain visualization state"""
    return {
        "brain_visualization": echo_brain.get_brain_state(),
        "thought_history_count": len(echo_brain.thought_history),
        "active_neurons": len(echo_brain.active_neurons),
        "service": "Echo Brain Neural Visualization",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/echo/thoughts/{thought_id}")
async def get_thought_stream(thought_id: str):
    """Get detailed thought stream for a specific thought"""
    thought_stream = echo_brain.get_thought_stream(thought_id)
    if not thought_stream:
        raise HTTPException(status_code=404, detail="Thought not found")
    
    return {
        "thought_id": thought_id,
        "thought_stream": thought_stream,
        "neuron_count": len(thought_stream),
        "service": "Echo Brain Thought Visualization"
    }

@app.get("/api/echo/stats")
async def get_statistics():
    """Get Echo Brain usage statistics"""
    try:
        conn = psycopg2.connect(**database.db_config)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_queries,
                AVG(processing_time) as avg_processing_time,
                model_used,
                COUNT(*) as usage_count
            FROM echo_unified_interactions 
            GROUP BY model_used
            ORDER BY usage_count DESC
        """)
        
        stats = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return {
            "statistics": [
                {
                    "model": row[2],
                    "usage_count": row[3],
                    "avg_processing_time": float(row[1]) if row[1] else 0
                }
                for row in stats
            ],
            "total_queries": stats[0][0] if stats else 0
        }
        
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        return {"error": str(e)}

@app.get("/api/echo/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history and context"""
    try:
        context = conversation_manager.get_conversation_context(conversation_id)
        if not context:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {
            "conversation_id": conversation_id,
            "history": context["history"],
            "last_intent": context.get("last_intent"),
            "interaction_count": context.get("interaction_count", 0)
        }
    except Exception as e:
        logger.error(f"Failed to retrieve conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/echo/execute")
async def execute_task(request: dict):
    """Execute a task based on conversational understanding - CI/CD integration point"""
    try:
        conversation_id = request.get("conversation_id")
        if not conversation_id:
            raise HTTPException(status_code=400, detail="conversation_id required")
        
        # Get conversation context to understand what user wants
        context = conversation_manager.get_conversation_context(conversation_id)
        if not context or not context.get("history"):
            raise HTTPException(status_code=400, detail="No conversation context found")
        
        # Get the latest intent and requirements from conversation
        latest_interaction = context["history"][-1]
        intent = latest_interaction.get("intent")
        
        execution_plan = {
            "conversation_id": conversation_id,
            "intent": intent,
            "status": "ready_for_execution",
            "steps": []
        }
        
        # Generate execution steps based on intent
        if intent == "code_modification":
            execution_plan["steps"] = [
                "Analyze existing code structure",
                "Identify files to modify",
                "Implement changes with proper error handling",
                "Run tests to verify functionality",
                "Commit changes with descriptive message"
            ]
        elif intent == "ci_cd":
            execution_plan["steps"] = [
                "Analyze project structure and requirements",
                "Set up CI/CD pipeline configuration",
                "Configure automated testing",
                "Set up deployment scripts",
                "Test pipeline execution"
            ]
        elif intent == "debugging":
            execution_plan["steps"] = [
                "Reproduce the error condition",
                "Analyze error logs and stack traces",
                "Identify root cause",
                "Implement fix with proper testing",
                "Verify fix resolves the issue"
            ]
        else:
            execution_plan["steps"] = [
                "Analyze requirements from conversation",
                "Create implementation plan", 
                "Execute step-by-step",
                "Validate results",
                "Provide feedback to user"
            ]
        
        return execution_plan
        
    except Exception as e:
        logger.error(f"Execution planning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/echo/conversations")
async def list_conversations(user_id: str = "default", limit: int = 10):
    """List recent conversations for a user"""
    try:
        # Get conversations from memory (in production, query database)
        user_conversations = []
        for conv_id, conv_data in conversation_manager.conversations.items():
            if conv_data.get("history"):
                # Check if any interaction has this user_id (simplified check)
                user_conversations.append({
                    "conversation_id": conv_id,
                    "created_at": conv_data["created_at"].isoformat(),
                    "last_intent": conv_data.get("last_intent"),
                    "interaction_count": len(conv_data["history"]),
                    "last_query": conv_data["history"][-1]["user_query"][:100] + "..." if conv_data["history"] else ""
                })
        
        # Sort by creation date and limit
        user_conversations.sort(key=lambda x: x["created_at"], reverse=True)
        return {"conversations": user_conversations[:limit]}
        
    except Exception as e:
        logger.error(f"Failed to list conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/echo/stream")
async def stream_brain_activity():
    """Stream real-time brain activity using Server-Sent Events"""
    
    async def event_generator():
        """Generate Server-Sent Events for brain activity"""
        while True:
            try:
                # Get current brain state
                brain_state = echo_brain.get_brain_state()
                
                # Format as SSE event
                event_data = json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "brain_state": brain_state,
                    "service": "Echo Brain Stream"
                })
                
                yield f"data: {event_data}\n\n"
                
                # Wait before next update
                await asyncio.sleep(0.5)  # 2 updates per second
                
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                break
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

@app.post("/api/echo/stream-query")
async def stream_query_processing(request: QueryRequest):
    """Process query with real-time streaming of thought process"""
    
    async def query_stream_generator():
        """Generate real-time streaming of query processing"""
        try:
            # Generate conversation ID if not provided
            conversation_id = request.conversation_id or str(uuid.uuid4())
            
            # Send initial event
            yield f"data: {json.dumps({'type': 'start', 'conversation_id': conversation_id, 'query': request.query})}\n\n"
            
            # Get conversation context
            conv_context = conversation_manager.get_conversation_context(conversation_id)
            
            # Classify intent and confidence
            intent, confidence = conversation_manager.classify_intent(
                request.query, 
                conv_context.get("history", [])
            )
            
            yield f"data: {json.dumps({'type': 'intent', 'intent': intent, 'confidence': confidence})}\n\n"
            
            # Check if clarification is needed
            needs_clarification = conversation_manager.needs_clarification(
                intent, confidence, request.query
            )
            
            if needs_clarification:
                clarifying_questions = conversation_manager.get_clarifying_questions(intent, request.query)
                yield f"data: {json.dumps({'type': 'clarification', 'questions': clarifying_questions})}\n\n"
                
                response_text = "I want to make sure I understand exactly what you need. Let me ask a few questions:\n\n"
                for i, question in enumerate(clarifying_questions, 1):
                    response_text += f"{i}. {question}\n"
                
                yield f"data: {json.dumps({'type': 'response', 'response': response_text, 'requires_clarification': True})}\n\n"
                
            else:
                # Process with streaming brain activity
                enhanced_context = {
                    **request.context,
                    "conversation_history": conv_context.get("history", []),
                    "intent": intent,
                    "confidence": confidence,
                    "user_id": request.user_id
                }
                
                # Start thinking process
                thought_id = await echo_brain.start_thinking("stream_query_processing", request.query)
                yield f"data: {json.dumps({'type': 'thinking_start', 'thought_id': thought_id})}\n\n"
                
                # Process with visible brain activity
                result = await router.progressive_escalation(request.query, enhanced_context)
                
                # Stream brain activity during processing
                brain_state = echo_brain.get_brain_state()
                yield f"data: {json.dumps({'type': 'brain_activity', 'brain_state': brain_state})}\n\n"
                
                if result["success"]:
                    yield f"data: {json.dumps({'type': 'response', 'response': result['response'], 'model_used': result['model'], 'processing_time': result['processing_time']})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'error', 'error': result.get('error', 'Processing failed')})}\n\n"
            
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
            
        except Exception as e:
            logger.error(f"Stream query error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        query_stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

# Orchestrator Integration API Endpoints

@app.get("/api/echo/orchestrator/status")
async def get_orchestrator_status():
    """Get orchestrator service status and health"""
    try:
        health_check = await orchestrator_client.check_orchestrator_health()
        if health_check["available"]:
            return {
                "available": True,
                "orchestrator_data": health_check["data"],
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "available": False,
                "error": health_check["error"],
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Failed to get orchestrator status: {e}")
        return {"available": False, "error": str(e)}

@app.get("/api/echo/agents")
async def list_available_agents():
    """Get list of available agents from orchestrator"""
    try:
        agents = await orchestrator_client.list_available_agents()
        return {
            "agents": agents,
            "total_agents": len(agents),
            "online_agents": len([a for a in agents if a.get("status") == "online"]),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get agents: {e}")
        return {"error": str(e), "agents": []}

class DelegateTaskRequest(BaseModel):
    task_description: str
    task_type: str = "analysis"
    priority: int = 5

@app.post("/api/echo/delegate")
async def delegate_task_directly(request: DelegateTaskRequest):
    """Directly delegate a task to the orchestrator (bypasses conversational interface)"""
    try:
        result = await orchestrator_client.delegate_task(
            request.task_description,
            request.task_type,
            request.priority
        )
        
        if result["success"]:
            return {
                "success": True,
                "task_id": result["task_id"],
                "orchestrator_task_type": result["orchestrator_task_type"],
                "status": result["status"],
                "message": f"Task successfully delegated to {result['orchestrator_task_type']} agent",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "error": result["error"],
                "timestamp": datetime.now().isoformat()
            }
    
    except Exception as e:
        logger.error(f"Task delegation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Autonomous Evolution API Endpoints

@app.get("/api/echo/evolution/status")
async def get_evolution_status():
    """Get autonomous evolution system status"""
    try:
        return evolution_system.get_evolution_status()
    except Exception as e:
        logger.error(f"Failed to get evolution status: {e}")
        return {"error": str(e)}

@app.post("/api/echo/evolution/trigger")
async def trigger_manual_evolution(request: dict):
    """Manually trigger an evolution cycle"""
    try:
        reason = request.get("reason", "manual_api_trigger")
        cycle_id = evolution_system.trigger_manual_evolution(reason)
        return {
            "success": True,
            "cycle_id": str(cycle_id),
            "message": "Evolution cycle triggered successfully"
        }
    except Exception as e:
        logger.error(f"Failed to trigger evolution: {e}")
        return {"error": str(e)}

@app.get("/api/echo/evolution/git-status")
async def get_git_status():
    """Get git repository status for autonomous improvements"""
    try:
        git_status = evolution_system.git_manager.get_git_status()
        evolution_status = evolution_system.git_manager.get_autonomous_evolution_status()
        
        return {
            "git_status": git_status,
            "evolution_capabilities": evolution_status["capabilities"],
            "deployment_config": evolution_status["deployment_config"],
            "safety_config": evolution_status["safety_config"],
            "recent_improvements": evolution_status["recent_improvements"]
        }
    except Exception as e:
        logger.error(f"Failed to get git status: {e}")
        return {"error": str(e)}

@app.post("/api/echo/evolution/self-analysis")
async def trigger_self_analysis(request: dict):
    """Trigger Echo's self-analysis system"""
    try:
        from echo_self_analysis import AnalysisDepth
        
        depth_str = request.get("depth", "functional")
        depth = AnalysisDepth(depth_str)
        context = request.get("context", {})
        
        analysis_result = await evolution_system.self_analysis.conduct_self_analysis(
            depth=depth,
            trigger_context=context
        )
        
        return {
            "analysis_id": analysis_result.analysis_id,
            "timestamp": analysis_result.timestamp.isoformat(),
            "depth": analysis_result.depth.value,
            "awareness_level": analysis_result.awareness_level.value,
            "insights": analysis_result.insights,
            "action_items": analysis_result.action_items,
            "recursive_observations": analysis_result.recursive_observations,
            "capabilities": [
                {
                    "name": cap.capability_name,
                    "current_level": cap.current_level,
                    "desired_level": cap.desired_level,
                    "gap": cap.desired_level - cap.current_level,
                    "improvement_path": cap.improvement_path
                }
                for cap in analysis_result.capabilities
            ]
        }
    except Exception as e:
        logger.error(f"Failed to trigger self-analysis: {e}")
        return {"error": str(e)}

@app.get("/api/echo/evolution/learning-metrics")
async def get_learning_metrics():
    """Get Echo's learning progress metrics"""
    try:
        import psycopg2
        import psycopg2.extras
        
        conn = psycopg2.connect(**database.db_config)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get evolution cycles
        cur.execute("""
            SELECT COUNT(*) as total_cycles,
                   SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_cycles,
                   AVG(EXTRACT(EPOCH FROM (end_time - start_time))) as avg_duration_seconds
            FROM echo_evolution_cycles 
            WHERE end_time IS NOT NULL
        """)
        evolution_stats = cur.fetchone()
        
        # Get learning metrics
        cur.execute("""
            SELECT metric_type, count, last_updated
            FROM echo_learning_metrics
            ORDER BY last_updated DESC
        """)
        learning_metrics = cur.fetchall()
        
        # Get recent self-analysis results
        cur.execute("""
            SELECT analysis_id, timestamp, depth, insights, action_items
            FROM echo_self_analysis
            ORDER BY timestamp DESC
            LIMIT 5
        """)
        recent_analyses = cur.fetchall()
        
        conn.close()
        
        return {
            "evolution_statistics": dict(evolution_stats) if evolution_stats else {},
            "learning_metrics": [dict(metric) for metric in learning_metrics],
            "recent_analyses": [dict(analysis) for analysis in recent_analyses],
            "current_state": evolution_system.learning_state,
            "system_metrics": evolution_system.metrics
        }
        
    except Exception as e:
        logger.error(f"Failed to get learning metrics: {e}")
        return {"error": str(e)}

# Update conversation tracking to trigger learning milestones
async def update_conversation_tracking_with_evolution(query: str, response: str, model_used: str, 
                                                     processing_time: float, conversation_id: str,
                                                     user_id: str = "default"):
    """Enhanced conversation tracking that updates evolution system"""
    try:
        # Update conversation count for evolution triggers
        evolution_system.update_conversation_count()
        
        # Calculate performance score based on response quality
        performance_score = min(1.0, len(response) / 100)  # Simple heuristic
        if processing_time < 2.0:
            performance_score += 0.1
        if "error" not in response.lower():
            performance_score += 0.1
            
        evolution_system.update_performance_metric(performance_score)
        
        # Log the interaction
        await database.log_interaction(
            query, response, model_used, processing_time, 
            [], conversation_id, user_id
        )
        
    except Exception as e:
        logger.error(f"Failed to update conversation tracking: {e}")

if __name__ == "__main__":
    logger.info("🚀 Starting Echo Brain Unified Service")
    logger.info("🔗 Consolidating 5 fragmented Echo services")
    logger.info("🧠 Dynamic intelligence scaling: 1B → 70B parameters")
    
    uvicorn.run(
        "echo_unified_service:app",
        host="0.0.0.0",
        port=8309,
        reload=False,
        workers=1
    )