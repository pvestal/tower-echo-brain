#!/usr/bin/env python3
"""
Extracts entities from conversation for future reference resolution.
"""

import re
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class EntityExtractor:
    """
    Extracts named entities from queries and responses.

    Focuses on Tower-specific entities:
    - Service names
    - File paths
    - Error types
    - Ports
    """

    # Known Tower services
    KNOWN_SERVICES = {
        "anime_production", "anime-production", "comfyui", "auth_service", "auth-service",
        "echo_brain", "echo-brain", "nginx", "postgresql", "redis", "qdrant", "ollama",
        "tower-echo-brain", "tower-anime-production", "tower-comfyui", "tower-auth",
        "tower-kb", "tower-dashboard", "tower-apple-music", "tower-notification",
        "vault", "veteran-guardian", "telegram-bot"
    }

    # Service-related patterns
    SERVICE_PATTERNS = [
        r"(\w+[-_]?\w*)\s+(?:is|are)\s+(?:broken|down|failed|unhealthy|stopped)",
        r"(?:service|container|process)\s+['\"]?(\w+[-_]?\w*)['\"]?",
        r"(?:restart|stop|start|check)\s+(\w+[-_]?\w*)",
        r"port\s+(\d{4,5})",
    ]

    # File path patterns
    FILE_PATTERNS = [
        r"(/[\w./\-_]+\.(?:py|js|ts|json|yaml|yml|conf|log))",
        r"file\s+['\"]?(/[\w./\-_]+)['\"]?",
    ]

    def __init__(self):
        self.service_patterns = [re.compile(p, re.IGNORECASE) for p in self.SERVICE_PATTERNS]
        self.file_patterns = [re.compile(p) for p in self.FILE_PATTERNS]

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract all entities from text.

        Returns dict like:
        {
            "service": "anime_production",
            "status": "broken",
            "port": "8188",
            "file": "/opt/tower/config.py"
        }
        """
        entities = {}

        # Extract services
        services = self._extract_services(text)
        if services:
            entities["service"] = services[0]
            if len(services) > 1:
                entities["services"] = services

        # Check for broken/unhealthy status
        if re.search(r"\b(broken|down|failed|unhealthy|stopped)\b", text, re.IGNORECASE):
            if "service" in entities:
                entities["broken_service"] = entities["service"]
                entities["status"] = "broken"

        # Extract ports
        ports = re.findall(r"\bport\s+(\d{4,5})\b", text, re.IGNORECASE)
        if ports:
            entities["port"] = ports[0]

        # Extract files
        files = self._extract_files(text)
        if files:
            entities["file"] = files[0]

        logger.debug(f"Extracted entities: {entities}")
        return entities

    def _extract_services(self, text: str) -> List[str]:
        """Extract service names from text."""
        found = set()

        # Check for known services
        text_lower = text.lower()
        for service in self.KNOWN_SERVICES:
            # Check for exact match or as part of compound word
            service_lower = service.lower()
            if service_lower in text_lower:
                # Verify it's a word boundary (not part of another word)
                pattern = r"\b" + re.escape(service_lower) + r"\b"
                if re.search(pattern, text_lower):
                    found.add(service)

        # Check patterns for additional services
        for pattern in self.service_patterns[:3]:  # Skip port pattern
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, str) and len(match) > 2:
                    # Normalize service names
                    normalized = match.lower().replace('-', '_')
                    found.add(normalized)

        return list(found)

    def _extract_files(self, text: str) -> List[str]:
        """Extract file paths from text."""
        found = []
        for pattern in self.file_patterns:
            matches = pattern.findall(text)
            found.extend(matches)
        return found