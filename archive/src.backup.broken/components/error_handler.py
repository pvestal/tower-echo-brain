#!/usr/bin/env python3
"""Error Handler - Centralized error handling for Echo Brain"""

import logging
import traceback
from typing import Any, Dict, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorCategory(Enum):
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    NETWORK = "network"
    DATABASE = "database"
    AI_MODEL = "ai_model"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"

class ErrorHandler:
    """Handles errors with recovery strategies"""

    def __init__(self, config_manager=None, logger=None):
        self.config_manager = config_manager
        self.logger = logger or logging.getLogger(__name__)
        self.recovery_strategies = {}

    async def handle(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle error with recovery attempt"""
        error_type = type(error).__name__
        self.logger.error(f"Handling error: {error_type} - {str(error)}")

        result = {
            'error_type': error_type,
            'error_message': str(error),
            'context': context or {},
            'recovery_attempted': False,
            'recovered': False,
            'traceback': traceback.format_exc()
        }

        # Attempt recovery based on error type
        if error_type in self.recovery_strategies:
            try:
                recovery = self.recovery_strategies[error_type]
                recovery_result = await recovery(error, context)
                result['recovery_attempted'] = True
                result['recovered'] = recovery_result.get('success', False)
                result['recovery_details'] = recovery_result
            except Exception as e:
                result['recovery_error'] = str(e)

        return result

    def categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize error"""
        error_msg = str(error).lower()

        if 'validation' in error_msg or 'invalid' in error_msg:
            return ErrorCategory.VALIDATION
        elif 'auth' in error_msg or 'permission' in error_msg:
            return ErrorCategory.AUTHENTICATION
        elif 'network' in error_msg or 'connection' in error_msg:
            return ErrorCategory.NETWORK
        elif 'database' in error_msg or 'sql' in error_msg:
            return ErrorCategory.DATABASE
        elif 'model' in error_msg or 'ai' in error_msg:
            return ErrorCategory.AI_MODEL
        elif 'config' in error_msg:
            return ErrorCategory.CONFIGURATION
        else:
            return ErrorCategory.UNKNOWN
