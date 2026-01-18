#!/usr/bin/env python3
"""
Home Assistant WebSocket Integration for Echo Brain
Provides real-time home automation awareness and control
"""
import asyncio
import json
import logging
import websockets
import aiohttp
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class HomeAssistantConfig:
    """Home Assistant connection configuration"""
    host: str = "192.168.50.135"  # Default tower server
    port: int = 8123
    token: Optional[str] = None
    ssl: bool = False

    @property
    def http_url(self) -> str:
        protocol = "https" if self.ssl else "http"
        return f"{protocol}://{self.host}:{self.port}"

    @property
    def ws_url(self) -> str:
        protocol = "wss" if self.ssl else "ws"
        return f"{protocol}://{self.host}:{self.port}/api/websocket"

class HomeAssistantBridge:
    """WebSocket bridge to Home Assistant for real-time integration"""

    def __init__(self, config: HomeAssistantConfig):
        self.config = config
        self.websocket = None
        self.connected = False
        self.message_id = 1
        self.subscriptions = {}
        self.entities = {}
        self.last_update = None

    async def connect(self) -> bool:
        """Connect to Home Assistant WebSocket API"""
        try:
            logger.info(f"🏠 Connecting to Home Assistant at {self.config.ws_url}")

            # Test HTTP connectivity first
            if not await self._test_http_connection():
                logger.warning("🏠 Home Assistant HTTP endpoint not accessible")
                return False

            # Connect to WebSocket
            self.websocket = await websockets.connect(self.config.ws_url)

            # Perform authentication handshake
            auth_response = await self._authenticate()
            if not auth_response:
                logger.error("🏠 Home Assistant authentication failed")
                return False

            logger.info("✅ Home Assistant WebSocket connected")
            self.connected = True

            # Start listening for events
            asyncio.create_task(self._listen_for_events())

            # Load initial state
            await self._load_initial_state()

            return True

        except Exception as e:
            logger.error(f"❌ Failed to connect to Home Assistant: {e}")
            return False

    async def _test_http_connection(self) -> bool:
        """Test HTTP connection to Home Assistant"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{self.config.http_url}/api/") as response:
                    return response.status == 200
        except Exception as e:
            logger.debug(f"Home Assistant HTTP test failed: {e}")
            return False

    async def _authenticate(self) -> bool:
        """Authenticate with Home Assistant WebSocket"""
        try:
            # Receive auth_required message
            auth_required = await self.websocket.recv()
            auth_data = json.loads(auth_required)

            if auth_data.get("type") != "auth_required":
                return False

            # Send authentication
            if self.config.token:
                auth_message = {
                    "type": "auth",
                    "access_token": self.config.token
                }
                await self.websocket.send(json.dumps(auth_message))

                # Receive auth result
                auth_result = await self.websocket.recv()
                result_data = json.loads(auth_result)

                return result_data.get("type") == "auth_ok"
            else:
                logger.warning("🏠 No Home Assistant token provided")
                return False

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False

    async def _listen_for_events(self):
        """Listen for Home Assistant events"""
        try:
            while self.connected and self.websocket:
                message = await self.websocket.recv()
                await self._handle_message(json.loads(message))

        except websockets.exceptions.ConnectionClosed:
            logger.warning("🏠 Home Assistant WebSocket connection closed")
            self.connected = False
        except Exception as e:
            logger.error(f"Error listening for events: {e}")

    async def _handle_message(self, message: Dict):
        """Handle incoming WebSocket message"""
        msg_type = message.get("type")

        if msg_type == "event":
            event_data = message.get("event", {})
            entity_id = event_data.get("data", {}).get("entity_id")

            if entity_id:
                # Update entity state
                new_state = event_data.get("data", {}).get("new_state")
                if new_state:
                    self.entities[entity_id] = new_state
                    logger.debug(f"🏠 Updated {entity_id}: {new_state.get('state')}")

        elif msg_type == "result":
            # Handle command responses
            request_id = message.get("id")
            if request_id in self.subscriptions:
                success = message.get("success", False)
                logger.debug(f"🏠 Command {request_id} result: {success}")

    async def _load_initial_state(self):
        """Load initial state of all entities"""
        try:
            message = {
                "id": self._next_id(),
                "type": "get_states"
            }
            await self.websocket.send(json.dumps(message))
            logger.info("🏠 Requested initial Home Assistant state")

        except Exception as e:
            logger.error(f"Failed to load initial state: {e}")

    def _next_id(self) -> int:
        """Get next message ID"""
        current_id = self.message_id
        self.message_id += 1
        return current_id

    async def get_entities(self) -> Dict[str, Any]:
        """Get all Home Assistant entities"""
        return self.entities.copy()

    async def get_entity_state(self, entity_id: str) -> Optional[Dict]:
        """Get state of specific entity"""
        return self.entities.get(entity_id)

    async def call_service(self, domain: str, service: str, entity_id: str = None, **service_data) -> bool:
        """Call Home Assistant service"""
        try:
            message = {
                "id": self._next_id(),
                "type": "call_service",
                "domain": domain,
                "service": service,
            }

            if entity_id:
                message["target"] = {"entity_id": entity_id}

            if service_data:
                message["service_data"] = service_data

            await self.websocket.send(json.dumps(message))
            logger.info(f"🏠 Called service {domain}.{service}")
            return True

        except Exception as e:
            logger.error(f"Failed to call service: {e}")
            return False

    async def get_home_status(self) -> Dict[str, Any]:
        """Get comprehensive home status for Echo Brain"""
        if not self.connected:
            return {"status": "disconnected", "message": "Home Assistant not connected"}

        try:
            # Categorize entities
            lights = {k: v for k, v in self.entities.items() if k.startswith("light.")}
            sensors = {k: v for k, v in self.entities.items() if k.startswith("sensor.")}
            switches = {k: v for k, v in self.entities.items() if k.startswith("switch.")}
            climate = {k: v for k, v in self.entities.items() if k.startswith("climate.")}

            return {
                "status": "connected",
                "last_update": self.last_update,
                "entity_count": len(self.entities),
                "summary": {
                    "lights": len([l for l in lights.values() if l.get("state") == "on"]),
                    "total_lights": len(lights),
                    "sensors": len(sensors),
                    "switches_on": len([s for s in switches.values() if s.get("state") == "on"]),
                    "climate_zones": len(climate)
                },
                "entities": {
                    "lights": lights,
                    "sensors": sensors,
                    "switches": switches,
                    "climate": climate
                }
            }

        except Exception as e:
            logger.error(f"Error getting home status: {e}")
            return {"status": "error", "error": str(e)}

    async def disconnect(self):
        """Disconnect from Home Assistant"""
        self.connected = False
        if self.websocket:
            await self.websocket.close()
            logger.info("🏠 Disconnected from Home Assistant")

# Global instance
_ha_bridge: Optional[HomeAssistantBridge] = None

async def get_home_assistant_bridge() -> Optional[HomeAssistantBridge]:
    """Get or create Home Assistant bridge instance"""
    global _ha_bridge

    if _ha_bridge is None or not _ha_bridge.connected:
        # Try to load config (you can add token from environment or config file)
        config = HomeAssistantConfig()

        # Check for token in environment
        import os
        token = os.getenv("HOME_ASSISTANT_TOKEN")
        if token:
            config.token = token

        _ha_bridge = HomeAssistantBridge(config)

        # Attempt connection (non-blocking)
        connected = await _ha_bridge.connect()
        if not connected:
            logger.warning("🏠 Home Assistant bridge not available")
            return None

    return _ha_bridge

async def get_home_status_for_echo() -> Dict[str, Any]:
    """Get home status formatted for Echo Brain queries"""
    bridge = await get_home_assistant_bridge()
    if not bridge:
        return {
            "available": False,
            "message": "Home Assistant integration not configured or not accessible"
        }

    status = await bridge.get_home_status()
    return {
        "available": True,
        "home_assistant": status
    }

# CLI testing function
async def test_home_assistant_connection():
    """Test Home Assistant connection from command line"""
    print("🏠 Testing Home Assistant connection...")

    bridge = await get_home_assistant_bridge()
    if bridge:
        status = await bridge.get_home_status()
        print(f"✅ Home Assistant connected: {status}")
    else:
        print("❌ Home Assistant connection failed")

if __name__ == "__main__":
    asyncio.run(test_home_assistant_connection())