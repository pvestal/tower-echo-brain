#!/usr/bin/env python3
"""Logging System - Structured logging for Echo Brain"""

import logging
import json
import asyncio
from typing import Any, Dict, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class StructuredLogger:
    """Structured logging with JSON output"""

    def __init__(self, name: str = 'echo_brain', config_manager=None):
        self.name = name
        self.config_manager = config_manager
        self.logger = logging.getLogger(name)
        self.setup_handlers()

    def setup_handlers(self):
        """Setup logging handlers"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    async def log(self, level: str, message: str, context: Dict[str, Any] = None) -> None:
        """Async log with structured data"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'logger': self.name,
            'message': message,
            'context': context or {}
        }

        # Log based on level
        log_json = json.dumps(log_entry)
        if level == 'ERROR':
            self.logger.error(log_json)
        elif level == 'WARNING':
            self.logger.warning(log_json)
        elif level == 'DEBUG':
            self.logger.debug(log_json)
        else:
            self.logger.info(log_json)

    async def info(self, message: str, context: Dict[str, Any] = None) -> None:
        """Log info message"""
        await self.log('INFO', message, context)

    async def error(self, message: str, context: Dict[str, Any] = None) -> None:
        """Log error message"""
        await self.log('ERROR', message, context)

    async def warning(self, message: str, context: Dict[str, Any] = None) -> None:
        """Log warning message"""
        await self.log('WARNING', message, context)

    async def debug(self, message: str, context: Dict[str, Any] = None) -> None:
        """Log debug message"""
        await self.log('DEBUG', message, context)
