#!/usr/bin/env python3
"""Input Processor - Handles all types of input for Echo Brain"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class InputType(Enum):
    CHAT_MESSAGE = "chat_message"
    API_REQUEST = "api_request"
    VOICE_INPUT = "voice_input"
    AUTONOMOUS_TRIGGER = "autonomous_trigger"
    EXTERNAL_DATA = "external_data"
    WEBHOOK = "webhook"
    SYSTEM_EVENT = "system_event"

class InputProcessor:
    """Processes various types of input"""

    def __init__(self, config_manager, logger=None):
        self.config_manager = config_manager
        self.logger = logger or logging.getLogger(__name__)
        self.supported_input_types = list(InputType)
        self.input_validators = {}
        self.input_transformers = {}

    async def process(self, raw_input: Any, input_type: InputType) -> Dict[str, Any]:
        """Process input based on type"""
        self.logger.info(f"Processing input of type: {input_type.value}")

        # Basic processing
        result = {
            'type': input_type.value,
            'content': raw_input,
            'processed_at': datetime.now().isoformat(),
            'status': 'processed',
            'metadata': {}
        }

        # Type-specific processing
        if input_type == InputType.CHAT_MESSAGE:
            result['metadata']['message_length'] = len(str(raw_input if isinstance(raw_input, str) else raw_input.get('content', '')))
        elif input_type == InputType.API_REQUEST:
            result['metadata']['method'] = raw_input.get('method', 'GET')

        return result

    def validate(self, raw_input: Any, input_type: InputType) -> bool:
        """Validate input"""
        return True

    def transform(self, raw_input: Any, input_type: InputType) -> Any:
        """Transform input"""
        return raw_input
