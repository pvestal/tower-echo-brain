#!/usr/bin/env python3
"""
Enhanced Echo Service with Tower System Management
Extends the basic Echo service with service management capabilities
"""

import logging
import json
import psycopg2
import psycopg2.extras
import asyncio
import subprocess
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import aiohttp
from typing import Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request Models
class ChatRequest(BaseModel):
    message: str
    context: Optional[dict] = {}
    project_id: Optional[str] = None

class ServiceManagementRequest(BaseModel):
    service_name: str
    action: str  # start, stop, restart, status

class LogRequest(BaseModel):
    service_name: str
    lines: Optional[int] = 100


# HashiCorp Vault Client Integration
class VaultClient:
    def __init__(self):
        self.vault_addr = "http://127.0.0.1:8200"
        self.vault_token = "***REMOVED***"
        
    async def get_secret(self, path: str, field: str = None) -> Union[str, Dict]:
        """Get secret from HashiCorp Vault"""
        try:
            headers = {"X-Vault-Token": self.vault_token}
            url = f"{self.vault_addr}/v1/{path}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        secret_data = data.get("data", {}).get("data", {})
                        return secret_data.get(field) if field else secret_data
                    else:
                        logger.error(f"Vault error {response.status} for {path}")
                        return None
        except Exception as e:
            logger.error(f"Vault connection error: {e}")
            return None

# Web Search Integration
class WebSearchService:
    def __init__(self, vault_client: VaultClient):
        self.vault = vault_client
        
    async def search_web(self, query: str, provider: str = "zillow") -> Dict[str, Any]:
        """Search web using RapidAPI services"""
        try:
            # Get RapidAPI key from vault
            api_key = await self.vault.get_secret("oauth/data/rapidapi", "api_key")
            if not api_key:
                return {"error": "No RapidAPI key found in vault"}
                
            headers = {
                "X-RapidAPI-Key": api_key,
                "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
            }
            
            if provider == "zillow":
                url = f"https://zillow-com1.p.rapidapi.com/propertyExtendedSearch?location={query}"
            else:
                return {"error": f"Unknown provider: {provider}"}
                
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "success": True,
                            "provider": provider,
                            "results": data.get("props", [])[:5],  # Limit to 5 results
                            "total_found": len(data.get("props", []))
                        }
                    else:
                        return {"error": f"API error: {response.status}"}
                        
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return {"error": str(e)}

# Credit Monitoring Service
class CreditMonitoringService:
    def __init__(self):
        self.data_sources = [
            {"name": "Chase Bank", "category": "credit_card_bank", "importance": 0.95},
            {"name": "Bank of America", "category": "credit_card_bank", "importance": 0.95},
            {"name": "American Express", "category": "credit_card", "importance": 0.9},
            {"name": "Capital One", "category": "credit_card_bank", "importance": 0.9},
            {"name": "County Court Records", "category": "public_records", "importance": 0.9},
            {"name": "Austin Energy", "category": "utility", "importance": 0.3},
            {"name": "AT&T", "category": "telecom", "importance": 0.4}
        ]
    
    async def get_monitoring_strategy(self) -> Dict[str, Any]:
        """Get credit monitoring strategy"""
        high_impact = [ds for ds in self.data_sources if ds["importance"] >= 0.8]
        
        return {
            "total_sources": len(self.data_sources),
            "high_impact_sources": len(high_impact),
            "high_impact_list": high_impact[:5],
            "coverage_recommendation": "Focus on major banks and credit cards first",
            "next_steps": [
                "Set up API monitoring for major banks",
                "Monitor public records for negative items",
                "Add utility payments for credit boost"
            ]
        }


app = FastAPI(title="Enhanced Echo Service with Tower Management")

class EnhancedEchoService:
    def __init__(self):
        self.db_config = {
            "host": "***REMOVED***",
            "database": "tower_consolidated", 
            "user": "patrick",
            "password": "admin123"
        }
        
        # Load project knowledge
        self.project_knowledge = self._load_project_knowledge()
        
        self.responses = {
            "hello": "Hello! I'm Enhanced Echo with Tower management capabilities.",
            "how are you": "I'm doing great! I can now manage Tower services too.",
            "what time is it": f"The current time is {datetime.now().strftime('%I:%M %p')}",
            "what can you do": "I can chat, manage Tower services, view logs, and monitor system health.",
            "services": "I can manage all Tower services: start, stop, restart, and check status.",
            "help": "Try: 'services', 'status', or ask me to 'restart tower-dashboard'",
            "default": "I heard: '{message}'. I can help with conversations and Tower service management."
        }
        
        # Allowed Tower services for management
        self.allowed_services = [
            "tower-dashboard", "tower-auth", "tower-echo-brain", "tower-kb", 
            "tower-anime", "tower-agent-manager", "tower-crypto-trader", 
            "tower-loan-search", "tower-deepseek-api", "tower-apple-music"
        ]
    
    def _load_project_knowledge(self) -> dict:
        """Load project-specific knowledge and character constraints"""
        try:
            knowledge_file = Path("/opt/tower-echo-brain/config/project_knowledge.json")
            if knowledge_file.exists():
                with open(knowledge_file, 'r') as f:
                    return json.load(f)
            else:
                logger.warning("Project knowledge file not found, using minimal defaults")
                return {"projects": {}}
        except Exception as e:
            logger.error(f"Failed to load project knowledge: {e}")
            return {"projects": {}}
    
    def get_project_context(self, project_id: str) -> Dict[str, Any]:
        """Get project-specific context and constraints"""
        if not project_id or project_id not in self.project_knowledge.get("projects", {}):
            return {}
        
        return self.project_knowledge["projects"][project_id]
    
    def get_project_conversation_history(self, project_id: str, limit: int = 10) -> List[Dict]:
        """Get recent conversation history for a specific project"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            if project_id:
                cursor.execute("""
                    SELECT message, response, created_at 
                    FROM echo_conversations 
                    WHERE project_id = %s 
                    ORDER BY created_at DESC 
                    LIMIT %s
                """, (project_id, limit))
            else:
                # For non-project conversations, get general ones
                cursor.execute("""
                    SELECT message, response, created_at 
                    FROM echo_conversations 
                    WHERE project_id IS NULL 
                    ORDER BY created_at DESC 
                    LIMIT %s
                """, (limit,))
            
            conversations = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return [dict(conv) for conv in conversations]
            
        except Exception as e:
            logger.error(f"Failed to get project conversation history: {e}")
            return []
    
    def get_project_characters(self, project_id: str) -> List[Dict]:
        """Get characters specific to a project"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            if project_id:
                cursor.execute("""
                    SELECT name, description, personality, appearance 
                    FROM anime_characters 
                    WHERE project_id = %s
                """, (project_id,))
            else:
                cursor.execute("""
                    SELECT name, description, personality, appearance 
                    FROM anime_characters 
                    WHERE project_id IS NULL
                """)
            
            characters = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return [dict(char) for char in characters]
            
        except Exception as e:
            logger.error(f"Failed to get project characters: {e}")
            return []
    
    def get_response(self, message: str, project_id: str = None) -> str:
        message_lower = message.lower().strip()
        
        # Check for service management requests in chat
        if any(word in message_lower for word in ["restart", "start", "stop", "service"]):
            return "I can help with service management! Use the dashboard controls or ask me to check specific services."
        
        # If this is a project-specific request, handle with project context
        if project_id:
            project_context = self.get_project_context(project_id)
            if project_context:
                return self._get_project_aware_response(message, project_id, project_context)
        
        # Check for exact matches first
        for key, response in self.responses.items():
            if key in message_lower:
                if key == "default":
                    return response.format(message=message)
                return response
        
        # Default response
        return self.responses["default"].format(message=message)
    
    def _get_project_aware_response(self, message: str, project_id: str, project_context: Dict) -> str:
        """Generate project-aware response using comprehensive project knowledge"""
        # Get detailed project information
        project_name = project_context.get("name", "Unknown Project")
        theme = project_context.get("theme", "")
        setting = project_context.get("setting", "")
        vocabulary = project_context.get("vocabulary", [])
        forbidden_elements = project_context.get("forbidden_elements", [])
        main_characters = project_context.get("main_characters", {})
        story_arcs = project_context.get("story_arcs", {})
        world_building = project_context.get("world_building", {})
        
        message_lower = message.lower()
        
        # Check for forbidden elements first
        for forbidden in forbidden_elements:
            if forbidden.lower() in message_lower:
                return f"I notice you mentioned '{forbidden}' which doesn't fit the {project_name} universe. Let's focus on {theme} instead. What aspect of the {setting} would you like to explore?"
        
        # Character-specific responses
        for char_name, char_data in main_characters.items():
            if char_name.lower() in message_lower or any(word in message_lower for word in char_name.lower().split()):
                return self._generate_character_response(char_name, char_data, message, project_context)
        
        # Story arc responses
        if any(word in message_lower for word in ["story", "plot", "arc", "season", "episode"]):
            return self._generate_story_response(story_arcs, message, project_context)
        
        # World-building responses
        if any(word in message_lower for word in ["location", "setting", "world", "place", "where"]):
            return self._generate_worldbuilding_response(world_building, message, project_context)
        
        # Relationship queries
        if any(word in message_lower for word in ["relationship", "interaction", "together", "between"]):
            return self._generate_relationship_response(main_characters, message, project_context)
        
        # Theme-based responses
        themes_mentioned = [theme for theme in project_context.get("themes", []) if theme.lower() in message_lower]
        if themes_mentioned:
            return self._generate_theme_response(themes_mentioned[0], project_context, message)
        
        # Prompt enhancement
        if "enhance" in message_lower and "prompt" in message_lower:
            return self._enhance_prompt_for_project(message, project_context)
        
        # General project-aware response with rich context
        relevant_vocab = [v for v in vocabulary if v.lower() in message_lower]
        if relevant_vocab:
            return f"In the world of {project_name}, I can help you explore {', '.join(relevant_vocab[:3])} within the context of {theme}. The rich characters and complex relationships in {setting} offer many storytelling opportunities. What specific aspect would you like to develop?"
        else:
            return f"Welcome to {project_name}! This {theme} story takes place in {setting}. I have deep knowledge of the main characters like {', '.join(list(main_characters.keys())[:3])}, their complex relationships, and the unfolding story arcs. What would you like to explore or create?"
    
    def _enhance_prompt_for_project(self, message: str, project_context: Dict) -> str:
        """Enhance a prompt specifically for the project context"""
        theme = project_context.get("theme", "")
        setting = project_context.get("setting", "")
        tone = project_context.get("tone", "")
        vocabulary = project_context.get("vocabulary", [])
        
        # Extract the prompt from the message
        prompt_start = message.lower().find("prompt")
        if prompt_start != -1:
            # Try to extract the actual prompt
            prompt_text = message[prompt_start + 6:].strip().strip(":")
            if prompt_text:
                # Enhance with project-specific elements
                enhanced = f"Enhanced for {project_context.get('name', '')}: {prompt_text}"
                enhanced += f" [Setting: {setting}] [Theme: {theme}] [Tone: {tone}]"
                if len(vocabulary) > 0:
                    relevant_vocab = vocabulary[:5]  # Use first 5 vocabulary items
                    enhanced += f" [Keywords: {', '.join(relevant_vocab)}]"
                return enhanced
        
        return f"To enhance a prompt for {project_context.get('name', '')}, please provide the base prompt you'd like me to improve with {theme} elements."
    
    def _create_character_for_project(self, message: str, project_context: Dict, existing_characters: List[Dict]) -> str:
        """Create a character that fits the project context"""
        character_archetypes = project_context.get("character_archetypes", [])
        setting = project_context.get("setting", "")
        theme = project_context.get("theme", "")
        
        existing_names = [char['name'] for char in existing_characters]
        
        if existing_names:
            char_list = ", ".join(existing_names)
            response = f"For {project_context.get('name', '')}, you already have these characters: {char_list}. "
        else:
            response = f"Creating a new character for {project_context.get('name', '')}. "
        
        response += f"Based on the {theme} theme, I suggest character types like: {', '.join(character_archetypes[:4])}. "
        response += f"The character should fit within {setting}. What type of character role are you looking for?"
        
        return response
    
    def _create_story_for_project(self, message: str, project_context: Dict) -> str:
        """Create a story that fits the project context"""
        themes = project_context.get("themes", [])
        setting = project_context.get("setting", "")
        tone = project_context.get("tone", "")
        
        response = f"For {project_context.get('name', '')} story development:\n"
        response += f"Setting: {setting}\n"
        response += f"Tone: {tone}\n"
        if themes:
            response += f"Themes to explore: {', '.join(themes[:4])}\n"
        response += "What specific story element would you like me to help develop?"
        
        return response
    
    def _generate_character_response(self, char_name: str, char_data: Dict, message: str, project_context: Dict) -> str:
        """Generate detailed response about a specific character"""
        role = char_data.get("role", "Unknown role")
        age = char_data.get("age", "Unknown age")
        background = char_data.get("background", "")
        personality = char_data.get("personality", "")
        appearance = char_data.get("appearance", "")
        voice = char_data.get("voice", "")
        relationships = char_data.get("relationships", "")
        motivation = char_data.get("motivation", "")
        
        message_lower = message.lower()
        project_name = project_context.get("name", "")
        
        # Specific queries about character aspects
        if "daughter" in message_lower and char_name.lower() == "kenji tanaka":
            return f"Kenji Tanaka's relationship with his 16-year-old daughter is deeply strained and tragic. {relationships} His daughter sees him as a monster because of his ruthless debt collection methods, and this perception haunts him despite his professional success. The divorce happened because his wife couldn't handle the stress of his work. This fractured relationship represents the personal cost of his professional dedication to debt collection - he's achieved a 95% collection rate but lost his family in the process. The irony is that while he's expert at pressuring others about their obligations, he's failed in his most important personal obligation as a father."
        
        if "relationship" in message_lower:
            return f"In {project_name}, {char_name} ({role}, age {age}) has complex relationships: {relationships} Their {role} position means they interact with others through the lens of {motivation}"
        
        if "personality" in message_lower or "character" in message_lower:
            return f"{char_name} is a {role} in {project_name}. {personality} Their background: {background} This personality drives their actions throughout the story as they navigate {project_context.get('theme', '')}."
        
        if "appearance" in message_lower or "look" in message_lower:
            return f"{char_name}'s appearance reflects their role and personality: {appearance} This visual presentation is carefully crafted to support their function in {project_name} as {role}."
        
        if "voice" in message_lower or "speak" in message_lower:
            return f"How {char_name} speaks is crucial to their character: {voice} This vocal style supports their role as {role} and helps establish the {project_context.get('tone', '')} tone of {project_name}."
        
        # General character information
        return f"{char_name} is a fascinating character in {project_name}. {role}, age {age}. {background} Their personality: {personality} Key motivation: {motivation} They represent important themes in this {project_context.get('theme', '')} story. What specific aspect of {char_name} interests you most?"
    
    def _generate_story_response(self, story_arcs: Dict, message: str, project_context: Dict) -> str:
        """Generate response about story arcs and plot"""
        project_name = project_context.get("name", "")
        
        if not story_arcs:
            return f"I can help develop story arcs for {project_name}. Based on the {project_context.get('theme', '')} theme, what kind of story progression are you interested in exploring?"
        
        # Return information about available arcs
        arc_summaries = []
        for arc_name, arc_data in story_arcs.items():
            setting = arc_data.get("setting", "")
            conflict = arc_data.get("main_conflict", "")
            arc_summaries.append(f"**{arc_name}**: {setting} - {conflict}")
        
        return f"The {project_name} story unfolds across multiple compelling arcs:\n\n" + "\n\n".join(arc_summaries) + f"\n\nEach arc explores different aspects of {project_context.get('theme', '')}. Which storyline would you like to explore further?"
    
    def _generate_worldbuilding_response(self, world_building: Dict, message: str, project_context: Dict) -> str:
        """Generate response about world-building and locations"""
        project_name = project_context.get("name", "")
        locations = world_building.get("locations", {})
        themes = world_building.get("themes", {})
        
        if not locations:
            return f"The {project_name} universe takes place in {project_context.get('setting', '')}. I can help you develop specific locations and world-building elements that support the {project_context.get('theme', '')} themes."
        
        message_lower = message.lower()
        
        # Check for specific location mentions
        for location_name, location_desc in locations.items():
            if location_name.lower() in message_lower:
                return f"**{location_name}** in {project_name}: {location_desc} This location is crucial to the {project_context.get('theme', '')} atmosphere and provides the perfect backdrop for character interactions."
        
        # General world-building response
        location_list = list(locations.keys())[:4]
        return f"The world of {project_name} is richly detailed with key locations including: {', '.join(location_list)}. Each location serves the {project_context.get('theme', '')} theme and provides unique opportunities for character development and plot advancement. Which aspect of the world-building interests you most?"
    
    def _generate_relationship_response(self, main_characters: Dict, message: str, project_context: Dict) -> str:
        """Generate response about character relationships and interactions"""
        project_name = project_context.get("name", "")
        
        # Try to identify which characters are being asked about
        mentioned_chars = []
        for char_name in main_characters.keys():
            if char_name.lower() in message.lower():
                mentioned_chars.append(char_name)
        
        if len(mentioned_chars) >= 2:
            char1, char2 = mentioned_chars[0], mentioned_chars[1]
            char1_data = main_characters[char1]
            char2_data = main_characters[char2]
            return f"The relationship between {char1} and {char2} in {project_name} is complex and layered. {char1} ({char1_data.get('role', '')}) and {char2} ({char2_data.get('role', '')}) interact within the context of {project_context.get('theme', '')}. Their different motivations - {char1} driven by '{char1_data.get('motivation', '')}' while {char2} is motivated by '{char2_data.get('motivation', '')}' - create rich dramatic potential for conflict and collaboration."
        
        elif len(mentioned_chars) == 1:
            char_name = mentioned_chars[0]
            char_data = main_characters[char_name]
            relationships = char_data.get("relationships", "")
            return f"{char_name}'s relationships in {project_name} are central to their character development: {relationships} These connections drive much of the dramatic tension in the {project_context.get('theme', '')} storyline."
        
        else:
            # General relationship overview
            char_names = list(main_characters.keys())
            return f"The character relationships in {project_name} are intricate and drive the {project_context.get('theme', '')} narrative. The main characters - {', '.join(char_names)} - each have complex connections that create conflict, alliance, and character growth. Which specific relationship dynamic interests you most?"
    
    def _generate_theme_response(self, theme: str, project_context: Dict, message: str) -> str:
        """Generate response about specific themes"""
        project_name = project_context.get("name", "")
        world_building = project_context.get("world_building", {})
        theme_details = world_building.get("themes", {})
        
        # Check if we have detailed theme information
        theme_key = theme.lower().replace(" ", "_")
        if theme_key in theme_details:
            return f"The theme of '{theme}' in {project_name} is explored through: {theme_details[theme_key]} This theme permeates the character development, plot progression, and world-building, creating a cohesive narrative that examines {project_context.get('tone', '')} aspects of human nature and society."
        
        return f"'{theme}' is a crucial theme in {project_name}. This concept is woven throughout the {project_context.get('theme', '')} narrative, influencing character motivations, plot developments, and the overall tone of the story. How would you like to explore this theme further in your creative work?"
    
    def log_conversation(self, user_message: str, echo_response: str, project_id: str = None):
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO echo_conversations 
                (message, response, hemisphere, model_used, created_at, session_id, project_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                user_message,
                echo_response, 
                "enhanced",
                "echo_enhanced_v1",
                datetime.now(),
                f"voice_{datetime.now().strftime('%Y%m%d_%H')}",
                project_id
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            logger.debug(f"Conversation logged successfully for project: {project_id}")
            
        except Exception as e:
            logger.error(f"Failed to log conversation: {e}")

    async def run_command(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute system command with timeout"""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            
            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode('utf-8', errors='replace'),
                "stderr": stderr.decode('utf-8', errors='replace'),
                "return_code": process.returncode,
                "command": command
            }
        except asyncio.TimeoutError:
            return {"success": False, "error": f"Command timed out after {timeout}s"}
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {"success": False, "error": str(e)}

    async def manage_service(self, service_name: str, action: str) -> Dict[str, Any]:
        """Manage Tower systemd services"""
        if service_name not in self.allowed_services:
            return {"success": False, "error": f"Service '{service_name}' not allowed for management"}
            
        if action not in ["start", "stop", "restart", "status", "enable", "disable"]:
            return {"success": False, "error": f"Invalid action: {action}"}
        
        command = f"sudo systemctl {action} {service_name}"
        result = await self.run_command(command)
        
        return {
            "success": result["success"],
            "service": service_name,
            "action": action,
            "output": result.get("stdout", ""),
            "error": result.get("stderr", ""),
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_service_logs(self, service_name: str, lines: int = 100) -> Dict[str, Any]:
        """Get service logs via journalctl"""
        if service_name not in self.allowed_services:
            return {"success": False, "error": f"Service '{service_name}' not allowed"}
        
        command = f"sudo journalctl -u {service_name} --no-pager -l -n {lines}"
        result = await self.run_command(command, timeout=10)
        
        return {
            "success": result["success"],
            "service": service_name,
            "logs": result.get("stdout", ""),
            "lines_requested": lines,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_all_service_status(self) -> Dict[str, Any]:
        """Get status of all Tower services"""
        services_status = {}
        
        for service in self.allowed_services:
            result = await self.manage_service(service, "status")
            services_status[service] = {
                "active": "Active: active" in result.get("output", ""),
                "status_output": result.get("output", ""),
                "last_checked": datetime.now().isoformat()
            }
        
        return {
            "success": True,
            "services": services_status,
            "total_services": len(self.allowed_services),
            "active_count": sum(1 for s in services_status.values() if s["active"]),
            "timestamp": datetime.now().isoformat()
        }

echo_service = EnhancedEchoService()

# Initialize new services
vault_client = VaultClient()
web_search_service = WebSearchService(vault_client)
credit_monitoring_service = CreditMonitoringService()

# ============ ORIGINAL ECHO ENDPOINTS ============

@app.get("/", response_class=HTMLResponse)
async def get_interface():
    try:
        with open("/opt/tower-echo-brain/static/echo_voice_interface.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>Enhanced Echo Service</h1><p>Management interface available at /health</p>")

@app.get("/api/health")
async def health_check():
    try:
        # Test database connection
        conn = psycopg2.connect(**echo_service.db_config)
        conn.close()
        db_status = True
    except:
        db_status = False
    
    return {
        "status": "healthy",
        "service": "Enhanced Echo Service with Tower Management & Project Context",
        "timestamp": datetime.now().isoformat(),
        "database_connected": db_status,
        "management_enabled": True,
        "project_isolation_enabled": True,
        "allowed_services": echo_service.allowed_services,
        "supported_projects": list(echo_service.project_knowledge.get("projects", {}).keys())
    }

@app.get("/api/projects")
async def get_projects():
    """Get available projects and their contexts"""
    try:
        projects = echo_service.project_knowledge.get("projects", {})
        return {
            "success": True,
            "projects": {
                project_id: {
                    "name": project_data.get("name"),
                    "theme": project_data.get("theme"),
                    "setting": project_data.get("setting"),
                    "character_count": len(echo_service.get_project_characters(project_id))
                }
                for project_id, project_data in projects.items()
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Projects endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/project/{project_id}/context")
async def get_project_context_endpoint(project_id: str):
    """Get full context for a specific project"""
    try:
        project_context = echo_service.get_project_context(project_id)
        if not project_context:
            raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
        
        characters = echo_service.get_project_characters(project_id)
        conversation_history = echo_service.get_project_conversation_history(project_id, limit=20)
        
        return {
            "success": True,
            "project_id": project_id,
            "context": project_context,
            "characters": characters,
            "recent_conversations": len(conversation_history),
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Project context endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        user_message = request.message.strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="Empty message")
        
        project_id = request.project_id
        echo_response = echo_service.get_response(user_message, project_id)
        
        # Log conversation with project context
        echo_service.log_conversation(user_message, echo_response, project_id)
        
        return {
            "user_message": user_message,
            "echo_response": echo_response,
            "model": "echo_enhanced_v1",
            "project_id": project_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            user_message = message_data.get("message", "")
            project_id = message_data.get("project_id")
            if not user_message:
                continue
            
            echo_response = echo_service.get_response(user_message, project_id)
            echo_service.log_conversation(user_message, echo_response, project_id)
            
            await websocket.send_text(json.dumps({
                "user_message": user_message,
                "echo_response": echo_response,
                "model": "echo_enhanced_v1",
                "project_id": project_id,
                "timestamp": datetime.now().isoformat()
            }))
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("WebSocket connection closed")

# ============ NEW TOWER MANAGEMENT ENDPOINTS ============

@app.post("/api/system/service")
async def manage_tower_service(request: ServiceManagementRequest):
    """Manage Tower services (start/stop/restart/status)"""
    try:
        result = await echo_service.manage_service(request.service_name, request.action)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Service management failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Service management error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/logs/{service_name}")
async def get_service_logs_endpoint(service_name: str, lines: int = 100):
    """Get service logs via journalctl"""
    try:
        result = await echo_service.get_service_logs(service_name, lines)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to get logs"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Log retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/status")
async def get_all_service_status_endpoint():
    """Get status of all Tower services"""
    try:
        result = await echo_service.get_all_service_status()
        return result
        
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/info")
async def get_system_info():
    """Get basic system information"""
    try:
        # Get disk usage
        disk_result = await echo_service.run_command("df -h /")
        
        # Get memory usage  
        mem_result = await echo_service.run_command("free -h")
        
        # Get load average
        load_result = await echo_service.run_command("uptime")
        
        return {
            "success": True,
            "disk_usage": disk_result.get("stdout", ""),
            "memory_usage": mem_result.get("stdout", ""),
            "load_average": load_result.get("stdout", ""),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"System info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/vault/get/{path:path}")
async def get_vault_secret(path: str, field: Optional[str] = None):
    """Get secret from HashiCorp Vault"""
    try:
        secret = await vault_client.get_secret(path, field)
        if secret is None:
            raise HTTPException(status_code=404, detail="Secret not found")
        return {"success": True, "data": secret}
    except Exception as e:
        logger.error(f"Vault get error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/echo/web-search")
async def echo_web_search(request: Dict[str, Any]):
    """Web search endpoint for Echo"""
    try:
        query = request.get("query", "")
        provider = request.get("provider", "zillow")
        
        if not query:
            raise HTTPException(status_code=400, detail="Query required")
            
        results = await web_search_service.search_web(query, provider)
        return results
        
    except Exception as e:
        logger.error(f"Echo web search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/echo/credit-monitoring")
async def echo_credit_monitoring():
    """Credit monitoring strategy endpoint"""
    try:
        strategy = await credit_monitoring_service.get_monitoring_strategy()
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "credit_monitoring": strategy
        }
    except Exception as e:
        logger.error(f"Credit monitoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/echo/chat")
async def echo_chat(request: Dict[str, Any]):
    """Enhanced chat endpoint with web search integration"""
    try:
        message = request.get("message", "")
        context = request.get("context", {})
        
        # Check if message requests web search
        if "search" in message.lower() or context.get("type") == "web_search":
            # Extract search query
            search_query = message.replace("search for", "").replace("search", "").strip()
            if not search_query:
                search_query = "Austin TX properties"
                
            search_results = await web_search_service.search_web(search_query)
            
            return {
                "success": True,
                "echo_response": f"I found {search_results.get('total_found', 0)} results for '{search_query}'",
                "search_results": search_results,
                "timestamp": datetime.now().isoformat()
            }
        
        # Default echo response
        return {
            "success": True,
            "echo_response": f"Echo received: {message}",
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Echo chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Enhanced Echo Service with Tower Management on port 8309")
    uvicorn.run(app, host="0.0.0.0", port=8309, log_level="info")