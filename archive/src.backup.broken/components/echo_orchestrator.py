#!/usr/bin/env python3
"""Echo Orchestrator - Main coordinator for Echo Brain components"""

import asyncio
import logging
from typing import Any, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class EchoOrchestrator:
    """Orchestrates Echo Brain components"""

    def __init__(self, input_processor=None, output_generator=None,
                 config_manager=None, error_handler=None, logger=None):
        self.input_processor = input_processor
        self.output_generator = output_generator
        self.config_manager = config_manager
        self.error_handler = error_handler
        self.logger = logger or logging.getLogger(__name__)
        self.initialized = False

    async def initialize(self) -> bool:
        """Initialize orchestrator"""
        try:
            self.logger.info("Initializing Echo Orchestrator")

            # Verify all components are available
            if not all([self.input_processor, self.output_generator,
                       self.config_manager, self.error_handler]):
                raise ValueError("Missing required components")

            self.initialized = True
            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False

    async def process_request(self, raw_input: Any, input_type: str) -> Dict[str, Any]:
        """Process a request through the pipeline"""
        if not self.initialized:
            await self.initialize()

        try:
            # Process input
            from src.processors.input_processor import InputType
            processed = await self.input_processor.process(
                raw_input,
                InputType[input_type.upper()]
            )

            # Generate AI response (mock for now)
            ai_response = f"Processed: {processed.get('content', 'No content')}"

            # Generate output
            output = await self.output_generator.generate(
                processed,
                ai_response,
                {'model': 'echo-brain-v1'}
            )

            return {
                'success': True,
                'input': processed,
                'output': output,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            # Handle error
            if self.error_handler:
                error_result = await self.error_handler.handle(e, {'raw_input': raw_input})
                return {
                    'success': False,
                    'error': error_result,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
