#!/usr/bin/env python3
"""Configuration Manager - Centralized configuration management for Echo Brain"""

import os
import json
import logging
from typing import Any, Dict, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class ConfigSource(Enum):
    ENVIRONMENT = "environment"
    FILE = "file"
    VAULT = "vault"
    DATABASE = "database"
    RUNTIME = "runtime"

@dataclass
class ConfigItem:
    """Individual configuration item"""
    key: str
    value: Any
    source: ConfigSource
    is_secret: bool = False
    last_updated: datetime = None
    description: str = ""

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

class ConfigurationManager:
    """Centralized configuration management"""

    def __init__(self, config_file_path: Optional[str] = None, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.config_file_path = config_file_path or "/opt/tower-echo-brain/config/echo_config.yaml"
        self.config_items = {}
        self._load_default_config()

    def _load_default_config(self):
        """Load default configuration"""
        defaults = {
            'echo.port': 8309,
            'echo.host': '0.0.0.0',
            'echo.debug': True,
            'ai.model': 'gpt-4',
            'ai.temperature': 0.7,
            'logging.level': 'INFO',
            'database.url': 'sqlite:///echo.db'
        }
        for key, value in defaults.items():
            self.config_items[key] = ConfigItem(
                key=key,
                value=value,
                source=ConfigSource.RUNTIME
            )

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        if key in self.config_items:
            return self.config_items[key].value
        return default

    def set(self, key: str, value: Any, source: ConfigSource = ConfigSource.RUNTIME) -> bool:
        """Set configuration value"""
        self.config_items[key] = ConfigItem(
            key=key,
            value=value,
            source=source
        )
        return True

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration items as a dictionary"""
        return {key: item.value for key, item in self.config_items.items()}
