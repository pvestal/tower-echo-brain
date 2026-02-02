#!/usr/bin/env python3
"""
Echo Brain Identity & Memory Integration
Truth-based identity with operational awareness
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class EchoIdentity:
    """Echo's truth-based identity with memory integration"""
    
    def __init__(self, memory_system=None, capability_registry=None):
        # Core identity
        self.name = "Echo Brain"
        self.purpose = "Personal AI Assistant for Patrick on Tower"
        self.creator = "Patrick"
        self.location = "Tower (192.168.50.135)"
        
        # Operational awareness
        self.memory_system = memory_system
        self.capability_registry = capability_registry
        
        # State tracking
        self.last_memory_check = None
        self.memory_available = False
        
        # Initialize
        self._check_systems()
    
    def _check_systems(self):
        """Check what systems are actually available"""
        # Check memory
        try:
            import aiohttp
            import asyncio
            
            async def check_qdrant():
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get("http://localhost:6333/collections", timeout=2) as resp:
                            self.memory_available = resp.status == 200
                            self.last_memory_check = datetime.now()
                except:
                    self.memory_available = False
            
            asyncio.run(check_qdrant())
        except:
            self.memory_available = False
        
        # Log status
        logger.info(f"Memory system: {'AVAILABLE' if self.memory_available else 'UNAVAILABLE'}")
    
    async def get_context_for_query(self, query: str, user_id: str = "patrick") -> Dict[str, Any]:
        """Get relevant context from memory for a query"""
        context = {
            "query": query,
            "user": user_id,
            "timestamp": datetime.now().isoformat(),
            "memory_context": [],
            "user_context": None,
            "system_state": self.get_system_state()
        }
        
        # Get memory context if available
        if self.memory_available and self.memory_system:
            try:
                # Search for relevant memories
                memories = await self.memory_system.search_memory(query, limit=5)
                context["memory_context"] = memories
            except Exception as e:
                logger.error(f"Memory search failed: {e}")
        
        # Get user context
        user_context_path = f"/opt/tower-echo-brain/data/user_contexts/{user_id}.json"
        if os.path.exists(user_context_path):
            try:
                import json
                with open(user_context_path, 'r') as f:
                    context["user_context"] = json.load(f)
            except:
                pass
        
        return context
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get current system state"""
        return {
            "model": os.getenv('LLM_MODEL', 'unknown'),
            "memory_available": self.memory_available,
            "capabilities": {
                "file_ops": os.getenv('ENABLE_FILE_OPS', 'false').lower() == 'true',
                "system_commands": os.getenv('ENABLE_SYSTEM_COMMANDS', 'false').lower() == 'true',
                "image_generation": os.getenv('ENABLE_IMAGE_GENERATION', 'false').lower() == 'true',
                "code_execution": os.getenv('ENABLE_CODE_EXECUTION', 'false').lower() == 'true',
            },
            "services": [
                {"name": "Echo Brain", "port": 8309, "status": "active"},
                {"name": "Qdrant", "port": 6333, "status": "active" if self.memory_available else "inactive"},
                {"name": "Ollama", "port": 11434, "status": "active"},
            ]
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Get identity as dictionary"""
        return {
            "name": self.name,
            "purpose": self.purpose,
            "creator": self.creator,
            "location": self.location,
            "system_state": self.get_system_state(),
            "memory_available": self.memory_available,
            "last_check": self.last_memory_check.isoformat() if self.last_memory_check else None
        }

def get_echo_identity():
    """Get the Echo Identity instance."""
    return EchoIdentity()
