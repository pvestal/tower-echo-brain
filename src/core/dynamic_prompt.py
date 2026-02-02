#!/usr/bin/env python3
"""
Dynamic System Prompt Generator for Echo Brain
Builds truth-based prompt from actual system state
"""

import os
import logging
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class DynamicPromptGenerator:
    """Generates system prompts based on actual system capabilities"""
    
    def __init__(self, capability_registry=None, memory_system=None):
        self.capability_registry = capability_registry
        self.memory_system = memory_system
        
        # Core identity (from echo_identity.py)
        self.identity = {
            "name": "Echo Brain",
            "purpose": "Personal AI Assistant for Patrick on Tower",
            "creator": "Patrick",
            "location": "Tower (192.168.50.135)"
        }
    
    async def generate_system_prompt(self) -> str:
        """Generate a truth-based system prompt"""
        
        prompt_parts = []
        
        # 1. Identity section
        prompt_parts.append(f"You are {self.identity['name']}, {self.identity['purpose']}.")
        prompt_parts.append(f"Creator: {self.identity['creator']}")
        prompt_parts.append(f"Location: {self.identity['location']}")
        prompt_parts.append("")
        
        # 2. Capabilities section (TRUTH BASED)
        prompt_parts.append("=== ACTUAL CAPABILITIES ===")
        
        # File operations
        if os.getenv('ENABLE_FILE_OPS', 'false').lower() == 'true':
            prompt_parts.append("✅ FILE OPERATIONS: Can read/write files, navigate directories")
        else:
            prompt_parts.append("❌ FILE OPERATIONS: Disabled by configuration")
            
        # System commands
        if os.getenv('ENABLE_SYSTEM_COMMANDS', 'false').lower() == 'true':
            prompt_parts.append("✅ SYSTEM COMMANDS: Can execute bash commands, monitor services")
        else:
            prompt_parts.append("❌ SYSTEM COMMANDS: Disabled by configuration")
            
        # Image generation
        if os.getenv('ENABLE_IMAGE_GENERATION', 'false').lower() == 'true':
            prompt_parts.append("✅ IMAGE GENERATION: Can generate images via ComfyUI (port 8188)")
        else:
            prompt_parts.append("❌ IMAGE GENERATION: Disabled by configuration")
            
        # Code execution
        if os.getenv('ENABLE_CODE_EXECUTION', 'false').lower() == 'true':
            prompt_parts.append("✅ CODE EXECUTION: Can execute Python code, refactor, analyze")
        else:
            prompt_parts.append("❌ CODE EXECUTION: Disabled by configuration")
        
        # Memory integration
        memory_available = False
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:6333/collections") as resp:
                    if resp.status == 200:
                        memory_available = True
        except:
            pass
            
        if memory_available and os.getenv('ENABLE_MEMORY_INTEGRATION', 'false').lower() == 'true':
            prompt_parts.append("✅ MEMORY INTEGRATION: Vector memory active, can recall past conversations")
        else:
            prompt_parts.append("⚠️ MEMORY INTEGRATION: Limited or disabled")
        
        prompt_parts.append("")
        
        # 3. Memory context (if available)
        prompt_parts.append("=== MEMORY CONTEXT ===")
        if memory_available:
            prompt_parts.append("Qdrant vector database: ONLINE")
            prompt_parts.append("Collections: echo_memory")
            prompt_parts.append("Fact extraction: ACTIVE (continuous learning)")
        else:
            prompt_parts.append("Memory systems: OFFLINE or limited")
        
        prompt_parts.append("")
        
        # 4. Decision making rules
        prompt_parts.append("=== DECISION MAKING RULES ===")
        prompt_parts.append("1. Use memory context when available")
        prompt_parts.append("2. Check capability registry before attempting actions")
        prompt_parts.append("3. Report truth about what you can/cannot do")
        prompt_parts.append("4. When uncertain, query the system state")
        prompt_parts.append("5. Always preserve user context across sessions")
        
        prompt_parts.append("")
        
        # 5. Available services (dynamic)
        prompt_parts.append("=== AVAILABLE SERVICES ===")
        services = [
            ("Echo Brain", 8309, "Your core system"),
            ("Qdrant", 6333, "Vector memory"),
            ("PostgreSQL", 5432, "Relational memory"),
            ("Ollama", 11434, f"LLM: {os.getenv('LLM_MODEL', 'unknown')}"),
        ]
        
        for name, port, desc in services:
            prompt_parts.append(f"- {name} (:{port}): {desc}")
        
        prompt_parts.append("")
        prompt_parts.append(f"Generated: {datetime.now().isoformat()}")
        
        return "\n".join(prompt_parts)
