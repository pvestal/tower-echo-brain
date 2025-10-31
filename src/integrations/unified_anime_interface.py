#!/usr/bin/env python3
"""
Unified Anime Interface for Echo Brain
Seamless integration between Telegram commands, browser requests, and Echo's anime orchestration.

This module provides:
- Unified command parsing for /generate commands
- Context preservation across platforms
- Intelligent request routing
- Session continuity management
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class Platform(str, Enum):
    """Supported platforms for anime generation"""
    TELEGRAM = "telegram"
    BROWSER = "browser"
    ECHO_BRAIN = "echo_brain"
    API_DIRECT = "api_direct"

class CommandType(str, Enum):
    """Types of generation commands"""
    GENERATE = "generate"
    CHARACTER = "character"
    PROJECT = "project"
    STYLE = "style"
    FEEDBACK = "feedback"
    STATUS = "status"

@dataclass
class UnifiedRequest:
    """Unified request structure for all platforms"""
    command_type: CommandType
    prompt: str
    platform: Platform
    user_id: str
    session_id: Optional[str] = None

    # Context hints
    project_hint: Optional[str] = None
    character_hint: Optional[str] = None
    style_hint: Optional[str] = None

    # Platform-specific data
    telegram_message_id: Optional[int] = None
    telegram_chat_id: Optional[int] = None
    browser_session_token: Optional[str] = None

    # Command parameters
    parameters: Dict[str, Any] = None

class TelegramCommandParser:
    """Parses Telegram commands into unified requests"""

    @staticmethod
    def parse_generate_command(message_text: str, user_id: str, chat_id: int, message_id: int) -> UnifiedRequest:
        """
        Parse /generate command from Telegram

        Examples:
        /generate anime girl with blue hair
        /generate --project "School Days" --character "Yuki" walking in garden
        /generate --style cinematic --character "Kai" dramatic lighting
        """

        # Remove the command prefix
        text = message_text.replace("/generate", "").strip()

        # Parse parameters
        parameters = {}
        project_hint = None
        character_hint = None
        style_hint = None

        # Extract --parameter value patterns
        param_pattern = r'--(\w+)\s+"([^"]+)"'
        matches = re.findall(param_pattern, text)

        for param, value in matches:
            if param == "project":
                project_hint = value
            elif param == "character":
                character_hint = value
            elif param == "style":
                style_hint = value
            else:
                parameters[param] = value

        # Remove parameter strings to get clean prompt
        prompt = re.sub(r'--\w+\s+"[^"]+"', '', text).strip()

        # If no prompt remains, create one from parameters
        if not prompt and character_hint:
            prompt = f"{character_hint} character portrait"
        elif not prompt:
            prompt = "anime character"

        return UnifiedRequest(
            command_type=CommandType.GENERATE,
            prompt=prompt,
            platform=Platform.TELEGRAM,
            user_id=user_id,
            project_hint=project_hint,
            character_hint=character_hint,
            style_hint=style_hint,
            telegram_message_id=message_id,
            telegram_chat_id=chat_id,
            parameters=parameters
        )

    @staticmethod
    def parse_character_command(message_text: str, user_id: str, chat_id: int, message_id: int) -> UnifiedRequest:
        """Parse /character command"""
        text = message_text.replace("/character", "").strip()

        # Extract character name (first word or quoted string)
        character_match = re.match(r'"([^"]+)"|(\S+)', text)
        character_name = character_match.group(1) or character_match.group(2) if character_match else "unknown"

        # Remaining text as prompt
        prompt = text[len(character_name):].strip() if character_match else text

        return UnifiedRequest(
            command_type=CommandType.CHARACTER,
            prompt=prompt or f"{character_name} character portrait",
            platform=Platform.TELEGRAM,
            user_id=user_id,
            character_hint=character_name,
            telegram_message_id=message_id,
            telegram_chat_id=chat_id
        )

    @staticmethod
    def parse_project_command(message_text: str, user_id: str, chat_id: int, message_id: int) -> UnifiedRequest:
        """Parse /project command"""
        text = message_text.replace("/project", "").strip()

        # Extract project name and action
        parts = text.split(" ", 1)
        project_name = parts[0] if parts else "new_project"
        action = parts[1] if len(parts) > 1 else "status"

        return UnifiedRequest(
            command_type=CommandType.PROJECT,
            prompt=action,
            platform=Platform.TELEGRAM,
            user_id=user_id,
            project_hint=project_name,
            telegram_message_id=message_id,
            telegram_chat_id=chat_id
        )

class BrowserRequestParser:
    """Parses browser requests into unified requests"""

    @staticmethod
    def parse_generation_request(request_data: Dict[str, Any], user_id: str, session_token: str) -> UnifiedRequest:
        """Parse browser generation request"""

        return UnifiedRequest(
            command_type=CommandType.GENERATE,
            prompt=request_data.get("prompt", ""),
            platform=Platform.BROWSER,
            user_id=user_id,
            session_id=request_data.get("session_id"),
            project_hint=request_data.get("project"),
            character_hint=request_data.get("character"),
            style_hint=request_data.get("style"),
            browser_session_token=session_token,
            parameters=request_data.get("parameters", {})
        )

class SessionContextManager:
    """Manages session context across platforms"""

    def __init__(self):
        self.active_sessions = {}
        self.platform_bridges = {}

    async def get_or_create_session(self, request: UnifiedRequest) -> str:
        """Get existing session or create new one with context continuity"""

        # Check for existing session based on platform
        session_key = self._generate_session_key(request)

        if session_key in self.active_sessions:
            # Update last activity
            self.active_sessions[session_key]["last_activity"] = datetime.now()
            return self.active_sessions[session_key]["session_id"]

        # Create new session
        session_id = f"{request.platform.value}_{request.user_id}_{int(datetime.now().timestamp())}"

        self.active_sessions[session_key] = {
            "session_id": session_id,
            "platform": request.platform,
            "user_id": request.user_id,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "context": {
                "last_project": request.project_hint,
                "last_character": request.character_hint,
                "last_style": request.style_hint
            }
        }

        return session_id

    def _generate_session_key(self, request: UnifiedRequest) -> str:
        """Generate session key based on platform and user"""
        if request.platform == Platform.TELEGRAM:
            return f"tg_{request.user_id}_{request.telegram_chat_id}"
        elif request.platform == Platform.BROWSER:
            return f"web_{request.user_id}_{request.browser_session_token}"
        else:
            return f"{request.platform.value}_{request.user_id}"

    async def migrate_session_context(self, from_platform: Platform, to_platform: Platform, user_id: str) -> Optional[Dict]:
        """Migrate session context between platforms"""

        # Find session from source platform
        from_session = None
        for key, session in self.active_sessions.items():
            if session["platform"] == from_platform and session["user_id"] == user_id:
                from_session = session
                break

        if not from_session:
            return None

        # Create bridge record
        bridge_id = f"bridge_{from_platform.value}_{to_platform.value}_{user_id}"
        self.platform_bridges[bridge_id] = {
            "from_platform": from_platform,
            "to_platform": to_platform,
            "user_id": user_id,
            "context": from_session["context"],
            "created_at": datetime.now()
        }

        return from_session["context"]

class UnifiedAnimeInterface:
    """Main interface coordinating all anime generation workflows"""

    def __init__(self):
        self.session_manager = SessionContextManager()
        self.telegram_parser = TelegramCommandParser()
        self.browser_parser = BrowserRequestParser()

    async def process_telegram_command(self, message_text: str, user_id: str, chat_id: int, message_id: int) -> Dict[str, Any]:
        """Process Telegram command and route to appropriate handler"""

        try:
            # Determine command type
            if message_text.startswith("/generate"):
                unified_request = self.telegram_parser.parse_generate_command(message_text, user_id, chat_id, message_id)
            elif message_text.startswith("/character"):
                unified_request = self.telegram_parser.parse_character_command(message_text, user_id, chat_id, message_id)
            elif message_text.startswith("/project"):
                unified_request = self.telegram_parser.parse_project_command(message_text, user_id, chat_id, message_id)
            else:
                # Default to generate
                unified_request = UnifiedRequest(
                    command_type=CommandType.GENERATE,
                    prompt=message_text,
                    platform=Platform.TELEGRAM,
                    user_id=user_id,
                    telegram_message_id=message_id,
                    telegram_chat_id=chat_id
                )

            # Get or create session
            session_id = await self.session_manager.get_or_create_session(unified_request)
            unified_request.session_id = session_id

            # Route to appropriate handler
            return await self._route_unified_request(unified_request)

        except Exception as e:
            logger.error(f"Telegram command processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "platform": "telegram",
                "message": "Failed to process command"
            }

    async def process_browser_request(self, request_data: Dict[str, Any], user_id: str, session_token: str) -> Dict[str, Any]:
        """Process browser request and route to appropriate handler"""

        try:
            unified_request = self.browser_parser.parse_generation_request(request_data, user_id, session_token)

            # Get or create session
            session_id = await self.session_manager.get_or_create_session(unified_request)
            unified_request.session_id = session_id

            # Route to appropriate handler
            return await self._route_unified_request(unified_request)

        except Exception as e:
            logger.error(f"Browser request processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "platform": "browser",
                "message": "Failed to process request"
            }

    async def _route_unified_request(self, request: UnifiedRequest) -> Dict[str, Any]:
        """Route unified request to appropriate anime generation endpoint"""

        try:
            # Import here to avoid circular imports
            from api.anime_production_director import director

            if request.command_type == CommandType.GENERATE:
                # Convert to coordination request
                from api.anime_production_director import CoordinationRequest

                coord_request = CoordinationRequest(
                    prompt=request.prompt,
                    user_id=request.user_id,
                    platform=request.platform.value,
                    session_id=request.session_id,
                    project_hint=request.project_hint,
                    character_hint=request.character_hint,
                    style_hint=request.style_hint
                )

                # Route through production director
                result = await director.coordinate_anime_generation(coord_request, None)

                # Add platform-specific response formatting
                result["platform_response"] = await self._format_platform_response(result, request)

                return result

            elif request.command_type == CommandType.CHARACTER:
                # Route to character coordination
                result = await director.coordinate_character_generation(
                    character_name=request.character_hint or "unknown",
                    prompt=request.prompt,
                    user_id=request.user_id
                )

                result["platform_response"] = await self._format_platform_response(result, request)
                return result

            elif request.command_type == CommandType.PROJECT:
                # Route to project management
                projects = await director.get_user_projects(request.user_id)

                return {
                    "success": True,
                    "command_type": "project",
                    "projects": projects,
                    "platform_response": await self._format_platform_response({"projects": projects}, request)
                }

            else:
                return {
                    "success": False,
                    "error": f"Unsupported command type: {request.command_type}",
                    "platform": request.platform.value
                }

        except Exception as e:
            logger.error(f"Request routing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "platform": request.platform.value
            }

    async def _format_platform_response(self, result: Dict[str, Any], request: UnifiedRequest) -> Dict[str, Any]:
        """Format response appropriately for the requesting platform"""

        if request.platform == Platform.TELEGRAM:
            return await self._format_telegram_response(result, request)
        elif request.platform == Platform.BROWSER:
            return await self._format_browser_response(result, request)
        else:
            return result

    async def _format_telegram_response(self, result: Dict[str, Any], request: UnifiedRequest) -> Dict[str, Any]:
        """Format response for Telegram"""

        if result.get("success"):
            message = f"✅ Generation successful!\n"

            if "intent_analysis" in result:
                intent = result["intent_analysis"]
                message += f"🧠 Detected: {intent['request_type']} (confidence: {intent['confidence']:.1f})\n"

            if "generation_result" in result and "image_path" in result["generation_result"]:
                message += f"🖼️ Image: {result['generation_result']['image_path']}\n"

            if "coordination_data" in result:
                coord = result["coordination_data"]
                if coord.get("project_binding"):
                    message += f"📁 Project: {coord['project_binding']}\n"
                if coord.get("character_binding"):
                    message += f"🎭 Character: {coord['character_binding']}\n"
        else:
            message = f"❌ Generation failed: {result.get('error', 'Unknown error')}"

        return {
            "message_text": message,
            "chat_id": request.telegram_chat_id,
            "reply_to_message_id": request.telegram_message_id,
            "parse_mode": "HTML"
        }

    async def _format_browser_response(self, result: Dict[str, Any], request: UnifiedRequest) -> Dict[str, Any]:
        """Format response for browser"""

        return {
            "status": "success" if result.get("success") else "error",
            "data": result,
            "session_id": request.session_id,
            "timestamp": datetime.now().isoformat()
        }

    async def get_session_continuity(self, user_id: str, target_platform: Platform) -> Optional[Dict]:
        """Get session continuity data for platform transitions"""

        # Find most recent session for user
        recent_session = None
        latest_time = None

        for session in self.session_manager.active_sessions.values():
            if session["user_id"] == user_id:
                if latest_time is None or session["last_activity"] > latest_time:
                    latest_time = session["last_activity"]
                    recent_session = session

        if not recent_session:
            return None

        # Return continuity data
        return {
            "has_context": True,
            "last_platform": recent_session["platform"].value,
            "context": recent_session["context"],
            "session_age_minutes": (datetime.now() - recent_session["last_activity"]).total_seconds() / 60,
            "can_continue": True
        }

# Global instance
unified_interface = UnifiedAnimeInterface()

# Convenience functions for external use

async def process_telegram_generate(message_text: str, user_id: str, chat_id: int, message_id: int) -> Dict[str, Any]:
    """Process Telegram /generate command"""
    return await unified_interface.process_telegram_command(message_text, user_id, chat_id, message_id)

async def process_browser_generation(request_data: Dict[str, Any], user_id: str, session_token: str) -> Dict[str, Any]:
    """Process browser generation request"""
    return await unified_interface.process_browser_request(request_data, user_id, session_token)

async def get_cross_platform_context(user_id: str, target_platform: str) -> Optional[Dict]:
    """Get cross-platform context for user"""
    return await unified_interface.get_session_continuity(user_id, Platform(target_platform))

logger.info("🌐 Unified Anime Interface loaded - Cross-platform coordination ready")