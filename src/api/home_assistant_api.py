#!/usr/bin/env python3
"""
Home Assistant API endpoints for Echo Brain
Exposes home automation controls through natural language interface
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, Optional
import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Import our Home Assistant integration
try:
    from src.integrations.home_assistant import get_home_assistant_bridge, get_home_status_for_echo
    HOME_ASSISTANT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Home Assistant integration not available: {e}")
    HOME_ASSISTANT_AVAILABLE = False

router = APIRouter(prefix="/api/home", tags=["home_assistant"])

class HomeCommand(BaseModel):
    """Home automation command"""
    command: str
    entity_id: Optional[str] = None
    domain: Optional[str] = None
    service: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class HomeQuery(BaseModel):
    """Home status query"""
    query: str
    category: Optional[str] = None  # lights, sensors, climate, etc.

@router.get("/status")
async def get_home_status():
    """Get comprehensive home status"""
    if not HOME_ASSISTANT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Home Assistant integration not available")

    try:
        status = await get_home_status_for_echo()
        return status
    except Exception as e:
        logger.error(f"Error getting home status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get home status: {str(e)}")

@router.get("/entities")
async def get_home_entities():
    """Get all Home Assistant entities"""
    if not HOME_ASSISTANT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Home Assistant integration not available")

    try:
        bridge = await get_home_assistant_bridge()
        if not bridge:
            raise HTTPException(status_code=503, detail="Home Assistant not connected")

        entities = await bridge.get_entities()
        return {"entities": entities, "count": len(entities)}
    except Exception as e:
        logger.error(f"Error getting entities: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get entities: {str(e)}")

@router.get("/entities/{entity_id}")
async def get_entity_state(entity_id: str):
    """Get specific entity state"""
    if not HOME_ASSISTANT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Home Assistant integration not available")

    try:
        bridge = await get_home_assistant_bridge()
        if not bridge:
            raise HTTPException(status_code=503, detail="Home Assistant not connected")

        entity_state = await bridge.get_entity_state(entity_id)
        if not entity_state:
            raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")

        return {"entity_id": entity_id, "state": entity_state}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting entity state: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get entity state: {str(e)}")

@router.post("/command")
async def execute_home_command(command: HomeCommand, background_tasks: BackgroundTasks):
    """Execute home automation command"""
    if not HOME_ASSISTANT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Home Assistant integration not available")

    try:
        bridge = await get_home_assistant_bridge()
        if not bridge:
            raise HTTPException(status_code=503, detail="Home Assistant not connected")

        # Parse natural language command into service calls
        result = await _process_natural_language_command(bridge, command)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to execute command: {str(e)}")

@router.post("/query")
async def query_home_state(query: HomeQuery):
    """Query home state with natural language"""
    if not HOME_ASSISTANT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Home Assistant integration not available")

    try:
        bridge = await get_home_assistant_bridge()
        if not bridge:
            raise HTTPException(status_code=503, detail="Home Assistant not connected")

        # Process natural language query
        result = await _process_home_query(bridge, query)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")

async def _process_natural_language_command(bridge, command: HomeCommand) -> Dict[str, Any]:
    """Process natural language command into Home Assistant service calls"""
    cmd_lower = command.command.lower()

    # Light controls
    if "turn on" in cmd_lower and "light" in cmd_lower:
        # Extract light entity
        entity_id = command.entity_id or _extract_entity_from_command(cmd_lower, "light")
        if entity_id:
            success = await bridge.call_service("light", "turn_on", entity_id)
            return {"success": success, "action": "turn_on", "entity": entity_id}

    elif "turn off" in cmd_lower and "light" in cmd_lower:
        entity_id = command.entity_id or _extract_entity_from_command(cmd_lower, "light")
        if entity_id:
            success = await bridge.call_service("light", "turn_off", entity_id)
            return {"success": success, "action": "turn_off", "entity": entity_id}

    # Switch controls
    elif "turn on" in cmd_lower and ("switch" in cmd_lower or "outlet" in cmd_lower):
        entity_id = command.entity_id or _extract_entity_from_command(cmd_lower, "switch")
        if entity_id:
            success = await bridge.call_service("switch", "turn_on", entity_id)
            return {"success": success, "action": "turn_on", "entity": entity_id}

    elif "turn off" in cmd_lower and ("switch" in cmd_lower or "outlet" in cmd_lower):
        entity_id = command.entity_id or _extract_entity_from_command(cmd_lower, "switch")
        if entity_id:
            success = await bridge.call_service("switch", "turn_off", entity_id)
            return {"success": success, "action": "turn_off", "entity": entity_id}

    # Climate controls
    elif "set temperature" in cmd_lower or "heat" in cmd_lower or "cool" in cmd_lower:
        entity_id = command.entity_id or _extract_entity_from_command(cmd_lower, "climate")
        # Extract temperature if provided
        temp = _extract_temperature_from_command(cmd_lower)
        if entity_id and temp:
            success = await bridge.call_service("climate", "set_temperature", entity_id, temperature=temp)
            return {"success": success, "action": "set_temperature", "entity": entity_id, "temperature": temp}

    # Generic service call if domain/service provided
    elif command.domain and command.service:
        success = await bridge.call_service(
            command.domain,
            command.service,
            command.entity_id,
            **(command.parameters or {})
        )
        return {"success": success, "action": f"{command.domain}.{command.service}", "entity": command.entity_id}

    return {"success": False, "error": "Could not parse command", "command": command.command}

async def _process_home_query(bridge, query: HomeQuery) -> Dict[str, Any]:
    """Process natural language home status query"""
    query_lower = query.query.lower()
    entities = await bridge.get_entities()

    # Lights query
    if "light" in query_lower:
        lights = {k: v for k, v in entities.items() if k.startswith("light.")}
        lights_on = [k for k, v in lights.items() if v.get("state") == "on"]

        return {
            "category": "lights",
            "total": len(lights),
            "on": len(lights_on),
            "lights_on": lights_on,
            "summary": f"{len(lights_on)} of {len(lights)} lights are on"
        }

    # Temperature/climate query
    elif "temperature" in query_lower or "climate" in query_lower:
        climate = {k: v for k, v in entities.items() if k.startswith("climate.")}
        sensors = {k: v for k, v in entities.items() if k.startswith("sensor.") and "temperature" in k}

        return {
            "category": "climate",
            "climate_entities": len(climate),
            "temperature_sensors": len(sensors),
            "climate_states": climate,
            "temperature_readings": sensors
        }

    # General status
    else:
        status = await bridge.get_home_status()
        return {
            "category": "general",
            "status": status,
            "summary": f"Home Assistant connected with {len(entities)} entities"
        }

def _extract_entity_from_command(command: str, domain: str) -> Optional[str]:
    """Extract entity ID from natural language command"""
    # Simple entity extraction - can be enhanced with NLP
    # For now, return a default entity for the domain
    return f"{domain}.default"

def _extract_temperature_from_command(command: str) -> Optional[float]:
    """Extract temperature from command"""
    import re
    temp_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:degrees?|°)', command)
    if temp_match:
        return float(temp_match.group(1))
    return None