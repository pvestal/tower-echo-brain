"""
Capability Registry System
Manages and tracks all autonomous capabilities of Echo Brain
"""

from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import asyncio
import logging

logger = logging.getLogger(__name__)

class CapabilityType(Enum):
    """Types of capabilities Echo Brain can have"""
    CODE_EXECUTION = "code_execution"
    SELF_MODIFICATION = "self_modification"
    FINANCIAL = "financial"
    TRAINING = "training"
    WEB_SCRAPING = "web_scraping"
    API_INTEGRATION = "api_integration"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"
    COMMUNICATION = "communication"
    ANALYSIS = "analysis"

class CapabilityStatus(Enum):
    """Status of a capability"""
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"
    INITIALIZING = "initializing"

@dataclass
class Capability:
    """Represents a single capability"""
    name: str
    type: CapabilityType
    description: str
    handler: Optional[Callable] = None
    status: CapabilityStatus = CapabilityStatus.INITIALIZING
    requirements: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_used: Optional[datetime] = None
    usage_count: int = 0
    success_rate: float = 0.0

class CapabilityRegistry:
    """Central registry for all Echo Brain capabilities"""

    def __init__(self):
        self.capabilities: Dict[str, Capability] = {}
        self.capability_groups: Dict[CapabilityType, List[str]] = {}
        self.execution_history = []

    def register_capability(
        self,
        name: str,
        capability_type: CapabilityType,
        description: str,
        handler: Optional[Callable] = None,
        requirements: List[str] = None,
        permissions: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Register a new capability

        Args:
            name: Unique name for the capability
            capability_type: Type of capability
            description: Human-readable description
            handler: Function to execute the capability
            requirements: List of required dependencies
            permissions: List of required permissions
            metadata: Additional metadata

        Returns:
            True if registration successful
        """

        if name in self.capabilities:
            logger.warning(f"Capability {name} already registered")
            return False

        capability = Capability(
            name=name,
            type=capability_type,
            description=description,
            handler=handler,
            requirements=requirements or [],
            permissions=permissions or [],
            metadata=metadata or {}
        )

        # Check if all requirements are met
        if self._check_requirements(capability):
            capability.status = CapabilityStatus.ACTIVE
        else:
            capability.status = CapabilityStatus.DISABLED

        self.capabilities[name] = capability

        # Add to group
        if capability_type not in self.capability_groups:
            self.capability_groups[capability_type] = []
        self.capability_groups[capability_type].append(name)

        logger.info(f"Registered capability: {name} ({capability_type.value})")
        return True

    def _check_requirements(self, capability: Capability) -> bool:
        """Check if all requirements for a capability are met"""

        for requirement in capability.requirements:
            # Check for required modules
            if requirement.startswith("module:"):
                module_name = requirement.replace("module:", "")
                try:
                    __import__(module_name)
                except ImportError:
                    logger.warning(f"Missing module for {capability.name}: {module_name}")
                    return False

            # Check for required services
            elif requirement.startswith("service:"):
                service_name = requirement.replace("service:", "")
                # TODO: Check if service is running
                pass

            # Check for required files
            elif requirement.startswith("file:"):
                file_path = requirement.replace("file:", "")
                from pathlib import Path
                if not Path(file_path).exists():
                    logger.warning(f"Missing file for {capability.name}: {file_path}")
                    return False

        return True

    async def execute_capability(
        self,
        name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a capability

        Args:
            name: Name of the capability
            **kwargs: Arguments to pass to the handler

        Returns:
            Execution results
        """

        if name not in self.capabilities:
            return {
                "success": False,
                "error": f"Capability {name} not found"
            }

        capability = self.capabilities[name]

        if capability.status != CapabilityStatus.ACTIVE:
            return {
                "success": False,
                "error": f"Capability {name} is {capability.status.value}"
            }

        if not capability.handler:
            return {
                "success": False,
                "error": f"Capability {name} has no handler"
            }

        # Execute the capability
        start_time = datetime.now()
        try:
            # Handle both sync and async handlers
            if asyncio.iscoroutinefunction(capability.handler):
                result = await capability.handler(**kwargs)
            else:
                result = capability.handler(**kwargs)

            execution_time = (datetime.now() - start_time).total_seconds()

            # Update statistics
            capability.usage_count += 1
            capability.last_used = datetime.now()

            # Update success rate
            success = result.get("success", False) if isinstance(result, dict) else True
            capability.success_rate = (
                (capability.success_rate * (capability.usage_count - 1) + (1 if success else 0))
                / capability.usage_count
            )

            # Record execution
            self.execution_history.append({
                "capability": name,
                "timestamp": datetime.now().isoformat(),
                "execution_time": execution_time,
                "success": success,
                "args": kwargs
            })

            return result

        except Exception as e:
            logger.error(f"Error executing capability {name}: {e}")
            capability.status = CapabilityStatus.ERROR

            return {
                "success": False,
                "error": str(e),
                "capability": name
            }

    def get_capability(self, name: str) -> Optional[Capability]:
        """Get a capability by name"""
        return self.capabilities.get(name)

    def list_capabilities(
        self,
        capability_type: Optional[CapabilityType] = None,
        status: Optional[CapabilityStatus] = None
    ) -> List[Capability]:
        """
        List capabilities with optional filtering

        Args:
            capability_type: Filter by type
            status: Filter by status

        Returns:
            List of capabilities
        """

        capabilities = list(self.capabilities.values())

        if capability_type:
            capabilities = [c for c in capabilities if c.type == capability_type]

        if status:
            capabilities = [c for c in capabilities if c.status == status]

        return capabilities

    def enable_capability(self, name: str) -> bool:
        """Enable a capability"""

        if name not in self.capabilities:
            return False

        capability = self.capabilities[name]

        if self._check_requirements(capability):
            capability.status = CapabilityStatus.ACTIVE
            logger.info(f"Enabled capability: {name}")
            return True

        return False

    def disable_capability(self, name: str) -> bool:
        """Disable a capability"""

        if name not in self.capabilities:
            return False

        self.capabilities[name].status = CapabilityStatus.DISABLED
        logger.info(f"Disabled capability: {name}")
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get capability statistics"""

        active_count = len([c for c in self.capabilities.values() if c.status == CapabilityStatus.ACTIVE])
        total_executions = sum(c.usage_count for c in self.capabilities.values())

        # Group by type
        by_type = {}
        for cap_type in CapabilityType:
            caps = self.list_capabilities(capability_type=cap_type)
            by_type[cap_type.value] = {
                "count": len(caps),
                "active": len([c for c in caps if c.status == CapabilityStatus.ACTIVE])
            }

        # Most used capabilities
        most_used = sorted(
            self.capabilities.values(),
            key=lambda x: x.usage_count,
            reverse=True
        )[:5]

        return {
            "total_capabilities": len(self.capabilities),
            "active_capabilities": active_count,
            "total_executions": total_executions,
            "by_type": by_type,
            "most_used": [
                {
                    "name": c.name,
                    "usage_count": c.usage_count,
                    "success_rate": c.success_rate
                }
                for c in most_used
            ],
            "recent_executions": self.execution_history[-10:]
        }

    def export_capabilities(self) -> Dict[str, Any]:
        """Export capability definitions for persistence"""

        return {
            "capabilities": [
                {
                    "name": c.name,
                    "type": c.type.value,
                    "description": c.description,
                    "status": c.status.value,
                    "requirements": c.requirements,
                    "permissions": c.permissions,
                    "metadata": c.metadata,
                    "usage_count": c.usage_count,
                    "success_rate": c.success_rate,
                    "last_used": c.last_used.isoformat() if c.last_used else None
                }
                for c in self.capabilities.values()
            ],
            "execution_history": self.execution_history[-100:]  # Keep last 100 executions
        }

    def import_capabilities(self, data: Dict[str, Any]):
        """Import capability definitions"""

        for cap_data in data.get("capabilities", []):
            # Skip if handler not available
            if cap_data["name"] not in self.capabilities:
                logger.warning(f"Skipping import of {cap_data['name']} - no handler registered")
                continue

            # Update statistics
            capability = self.capabilities[cap_data["name"]]
            capability.usage_count = cap_data.get("usage_count", 0)
            capability.success_rate = cap_data.get("success_rate", 0.0)
            if cap_data.get("last_used"):
                capability.last_used = datetime.fromisoformat(cap_data["last_used"])

        # Import execution history
        self.execution_history = data.get("execution_history", [])