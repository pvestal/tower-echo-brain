#!/usr/bin/env python3
"""
Echo Capability Coordinator
Connects user requests to actual capability execution
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from .capability_registry import CapabilityRegistry

logger = logging.getLogger(__name__)

class EchoCapabilityCoordinator:
    """
    Coordinates between Echo Brain chat and autonomous capabilities
    Intercepts action requests and executes them through the capability registry
    """

    def __init__(self, capability_registry: CapabilityRegistry):
        self.capability_registry = capability_registry
        self.action_patterns = self._init_action_patterns()

    def _init_action_patterns(self) -> List[Dict]:
        """Initialize patterns that trigger capabilities"""
        return [
            {
                "patterns": [
                    r"restart.*?(tower[\w-]+)",
                    r"restart.*?the\s+(tower[\w-]+)",
                    r"can you restart.*?(tower[\w-]+)",
                    r"please restart.*?(tower[\w-]+)",
                    r"fix.*?(tower[\w-]+)",
                    r"repair.*?(tower[\w-]+)"
                ],
                "capability": "autonomous_repair",
                "extract": lambda m: {"service_name": m.group(1).strip('?'), "issue_type": "restart"},
                "description": "Service restart/repair operations"
            },
            {
                "patterns": [
                    r"generate.*image.*(.+)",
                    r"create.*anime.*(.+)",
                    r"make.*image.*(.+)",
                    r"anime.*production.*(.+)"
                ],
                "capability": "image_generation",
                "extract": lambda m: {"prompt": m.group(1).strip(), "style": "anime"},
                "description": "Image/anime generation"
            },
            {
                "patterns": [
                    r"send.*notification.*(.+)",
                    r"notify.*(.+)",
                    r"alert.*(.+)"
                ],
                "capability": "send_notification",
                "extract": lambda m: {"message": m.group(1).strip(), "channel": "ntfy"},
                "description": "Send notifications"
            },
            {
                "patterns": [
                    r"check.*services?.*status",
                    r"service.*health.*check",
                    r"monitor.*services?",
                    r"what.*services?.*running"
                ],
                "capability": "service_monitoring",
                "extract": lambda m: {},
                "description": "Service health monitoring"
            },
            {
                "patterns": [
                    r"analyze.*code.*(.+)",
                    r"review.*code.*(.+)",
                    r"improve.*code.*(.+)"
                ],
                "capability": "code_analysis",
                "extract": lambda m: {"file_path": m.group(1).strip() if m.group(1) else None},
                "description": "Code analysis and review"
            },
            {
                "patterns": [
                    r"diagnose.*system",
                    r"check.*system.*health",
                    r"system.*status"
                ],
                "capability": "autonomous_repair",
                "extract": lambda m: {"issue_type": "diagnose"},
                "description": "System diagnosis"
            }
        ]

    async def process_request(self, user_query: str, user: str = "anonymous") -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Process user request to determine if it requires capability execution

        Returns:
            (should_execute, execution_result)
            - should_execute: True if this was an action request
            - execution_result: Result of capability execution or None
        """

        query_lower = user_query.lower().strip()

        # Check each action pattern
        for action_config in self.action_patterns:
            for pattern in action_config["patterns"]:
                match = re.search(pattern, query_lower)
                if match:
                    logger.info(f"🎯 Action detected: {action_config['description']} for user {user}")

                    # Extract parameters
                    params = action_config["extract"](match)
                    params["user"] = user

                    # Execute capability
                    result = await self._execute_capability(
                        action_config["capability"],
                        params,
                        query_lower
                    )

                    return True, result

        # No action pattern matched
        return False, None

    async def _execute_capability(self, capability_name: str, params: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """Execute a capability through the registry"""

        try:
            # Check if capability exists
            if not self.capability_registry.get_capability(capability_name):
                return {
                    "success": False,
                    "error": f"Capability '{capability_name}' not available",
                    "action": "capability_execution",
                    "query": original_query
                }

            logger.info(f"🚀 Executing capability: {capability_name} with params: {params}")

            # Execute the capability
            result = await self.capability_registry.execute_capability(capability_name, **params)

            # Add metadata
            result["action"] = "capability_execution"
            result["capability"] = capability_name
            result["query"] = original_query
            result["autonomous"] = True

            if result.get("success"):
                logger.info(f"✅ Capability {capability_name} executed successfully")
            else:
                logger.warning(f"❌ Capability {capability_name} failed: {result.get('error', 'Unknown error')}")

            return result

        except Exception as e:
            logger.error(f"💥 Error executing capability {capability_name}: {e}")
            return {
                "success": False,
                "error": f"Execution error: {str(e)}",
                "action": "capability_execution",
                "capability": capability_name,
                "query": original_query
            }

    def format_execution_response(self, result: Dict[str, Any], original_query: str) -> str:
        """Format capability execution result for user response"""

        if not result:
            return "I wasn't able to process that request."

        capability = result.get("capability", "unknown")

        if result.get("success"):
            # Success responses
            if capability == "autonomous_repair":
                # Check if it's a diagnosis or repair
                if "system_status" in result:
                    # System diagnosis
                    status = result.get("system_status", {})
                    active_services = [s for s, state in status.items() if state == "active"]
                    inactive_services = [s for s, state in status.items() if state != "active"]

                    response = f"📊 System Diagnosis Complete:\n"
                    response += f"✅ Active Services ({len(active_services)}): {', '.join(active_services)}\n"
                    if inactive_services:
                        response += f"❌ Inactive Services ({len(inactive_services)}): {', '.join(inactive_services)}"
                    return response
                else:
                    # Service restart
                    service = result.get("service", "service")
                    return f"✅ Successfully restarted {service}. The service should now be operational."

            elif capability == "service_monitoring":
                summary = result.get("summary", {})
                total = summary.get("total", 0)
                healthy = summary.get("healthy", 0)
                return f"📊 Service Status: {healthy}/{total} services are healthy. Health rate: {summary.get('health_rate', 0):.1f}%"

            elif capability == "send_notification":
                channels = result.get("channels_successful", 0)
                return f"📢 Notification sent successfully via {channels} channel(s)."

            elif capability == "code_analysis":
                issues = len(result.get("issues", []))
                suggestions = len(result.get("suggestions", []))
                return f"🔍 Code analysis complete: Found {issues} issues and {suggestions} improvement suggestions."

            else:
                return f"✅ {capability.replace('_', ' ').title()} completed successfully."

        else:
            # Error responses
            error = result.get("error", "Unknown error")
            return f"❌ Failed to execute {capability.replace('_', ' ')}: {error}"

    def get_capability_status(self) -> Dict[str, Any]:
        """Get status of all registered capabilities"""

        if not self.capability_registry:
            return {"error": "Capability registry not available"}

        stats = self.capability_registry.get_statistics()
        capabilities = self.capability_registry.list_capabilities()

        return {
            "total_capabilities": len(capabilities),
            "active_capabilities": len([c for c in capabilities if c.status.value == "active"]),
            "action_patterns": len(self.action_patterns),
            "capabilities": [
                {
                    "name": c.name,
                    "type": c.type.value,
                    "status": c.status.value,
                    "usage_count": c.usage_count,
                    "success_rate": c.success_rate
                }
                for c in capabilities
            ],
            "statistics": stats
        }

    def can_handle_query(self, query: str) -> bool:
        """Check if a query can be handled by capabilities (without executing)"""

        query_lower = query.lower().strip()

        for action_config in self.action_patterns:
            for pattern in action_config["patterns"]:
                if re.search(pattern, query_lower):
                    return True

        return False

    def get_supported_actions(self) -> List[str]:
        """Get list of supported action descriptions"""
        return [action["description"] for action in self.action_patterns]

# Global coordinator instance - will be initialized by startup
capability_coordinator = None

def get_capability_coordinator() -> Optional[EchoCapabilityCoordinator]:
    """Get the global capability coordinator"""
    return capability_coordinator

def initialize_coordinator(capability_registry: CapabilityRegistry) -> EchoCapabilityCoordinator:
    """Initialize the global capability coordinator"""
    global capability_coordinator
    capability_coordinator = EchoCapabilityCoordinator(capability_registry)
    logger.info("🎯 Echo Capability Coordinator initialized")
    return capability_coordinator