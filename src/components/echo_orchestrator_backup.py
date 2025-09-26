#!/usr/bin/env python3
"""
Echo Orchestrator - Main service orchestrator with modular architecture
Integrates InputProcessor, OutputGenerator, ConfigurationManager, ErrorHandler, and Logger
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

# Import modular components
from src.managers.configuration_manager import ConfigurationManager, get_config_manager
from src.processors.input_processor import InputProcessor, InputType, ProcessedInput
from src.generators.output_generator import OutputGenerator, OutputType, DeliveryMethod, GeneratedOutput
from src.components.dependency_container import DependencyContainer, get_container, configure_container
from src.components.error_handler import ErrorHandler, get_error_handler, ErrorContext, handle_errors
from src.components.logging_system import EchoLogger, get_logger, LogCategory

@dataclass
class EchoRequest:
    """Standardized request format"""
    query: str
    user_id: str
    conversation_id: Optional[str] = None
    input_type: InputType = InputType.CHAT_MESSAGE
    context: Dict[str, Any] = None
    priority: int = 5
    requires_auth: bool = False
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.conversation_id is None:
            self.conversation_id = f"{self.input_type.value}_{datetime.now().timestamp()}"

@dataclass
class EchoResponse:
    """Standardized response format"""
    response: str
    conversation_id: str
    model_used: str
    processing_time: float
    confidence: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class EchoOrchestrator:
    """
    Main Echo Brain orchestrator with modular architecture
    Coordinates between InputProcessor, AI models, and OutputGenerator
    """
    
    def __init__(self, container: DependencyContainer = None):
        # Initialize container and dependencies
        self.container = container or configure_container()
        self.logger = get_logger("echo_orchestrator")
        self.error_handler = get_error_handler()
        
        # Initialize components (will be injected)
        self.config_manager: Optional[ConfigurationManager] = None
        self.input_processor: Optional[InputProcessor] = None
        self.output_generator: Optional[OutputGenerator] = None
        
        # Performance tracking
        self.request_count = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        
        # Initialize the service
        asyncio.create_task(self._initialize())
    
    async def _initialize(self):
        """Initialize the orchestrator with dependency injection"""
        try:
            self.logger.info("Initializing Echo Orchestrator...", category=LogCategory.SYSTEM)
            
            # Resolve dependencies
            self.config_manager = await self.container.resolve(ConfigurationManager)
            self.input_processor = await self.container.resolve(InputProcessor)
            self.output_generator = await self.container.resolve(OutputGenerator)
            
            # Update logger with configuration
            self.logger.update_context(
                service="echo_orchestrator",
                version="2.0.0",
                environment="production"
            )
            
            self.logger.info("Echo Orchestrator initialized successfully", category=LogCategory.SYSTEM)
            
        except Exception as e:
            self.logger.critical("Failed to initialize Echo Orchestrator", 
                               category=LogCategory.SYSTEM, exc_info=e)
            raise
    
    @handle_errors()
    async def process_request(self, request: EchoRequest) -> EchoResponse:
        """
        Main request processing method
        Handles the complete flow from input to output
        """
        start_time = datetime.now()
        conversation_logger = self.logger.with_context(
            conversation_id=request.conversation_id,
            user_id=request.user_id,
            request_id=f"req_{start_time.timestamp()}"
        )
        
        try:
            conversation_logger.info(
                f"Processing request: {request.query[:100]}...",
                category=LogCategory.USER_ACTION,
                input_type=request.input_type.value,
                priority=request.priority
            )
            
            # Step 1: Process input
            processed_input = await self._process_input(request, conversation_logger)
            if not processed_input:
                return self._create_error_response(
                    request, "Input processing failed", start_time
                )
            
            # Step 2: Generate AI response
            ai_response, model_info = await self._generate_ai_response(
                processed_input, conversation_logger
            )
            if not ai_response:
                return self._create_error_response(
                    request, "AI response generation failed", start_time
                )
            
            # Step 3: Generate output
            generated_output = await self._generate_output(
                processed_input, ai_response, model_info, conversation_logger
            )
            if not generated_output:
                return self._create_error_response(
                    request, "Output generation failed", start_time
                )
            
            # Step 4: Deliver output (if not HTTP response)
            if generated_output.delivery_method != DeliveryMethod.HTTP_RESPONSE:
                delivery_success = await self.output_generator.deliver(generated_output)
                if not delivery_success:
                    conversation_logger.warning("Output delivery failed", 
                                              category=LogCategory.EXTERNAL_SERVICE)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update metrics
            self._update_metrics(processing_time, success=True)
            
            # Log success
            conversation_logger.info(
                f"Request processed successfully in {processing_time:.3f}s",
                category=LogCategory.PERFORMANCE,
                processing_time_ms=processing_time * 1000,
                model_used=model_info.get("model_name", "unknown"),
                confidence=model_info.get("confidence", 0.0)
            )
            
            # Create response
            return EchoResponse(
                response=str(generated_output.content),
                conversation_id=request.conversation_id,
                model_used=model_info.get("model_name", "unknown"),
                processing_time=processing_time,
                confidence=model_info.get("confidence", 0.0),
                success=True,
                metadata={
                    "input_type": request.input_type.value,
                    "output_type": generated_output.output_type.value,
                    "delivery_method": generated_output.delivery_method.value,
                    "processing_steps": ["input_processing", "ai_generation", "output_generation"]
                }
            )
            
        except Exception as e:
            # Handle errors
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(processing_time, success=False)
            
            error_context = ErrorContext(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                additional_data={"request_query": request.query}
            )
            
            await self.error_handler.handle_error(e, error_context)
            
            conversation_logger.error(
                f"Request processing failed: {str(e)}",
                category=LogCategory.SYSTEM,
                exc_info=e,
                processing_time_ms=processing_time * 1000
            )
            
            return self._create_error_response(request, str(e), start_time)
    
    async def _process_input(self, request: EchoRequest, logger: EchoLogger) -> Optional[ProcessedInput]:
        """Process the input using InputProcessor"""
        try:
            logger.debug("Starting input processing", category=LogCategory.BUSINESS_LOGIC)
            
            # Convert request to raw input format
            raw_input = {
                "query": request.query,
                "user_id": request.user_id,
                "conversation_id": request.conversation_id,
                "context": request.context,
                "priority": request.priority,
                "requires_auth": request.requires_auth
            }
            
            # Process with InputProcessor
            processed = await self.input_processor.process(raw_input, request.input_type)
            
            if processed:
                logger.debug(
                    "Input processing completed",
                    category=LogCategory.BUSINESS_LOGIC,
                    input_length=len(request.query),
                    processed_type=processed.input_type.value
                )
            
            return processed
            
        except Exception as e:
            logger.error("Input processing error", category=LogCategory.BUSINESS_LOGIC, exc_info=e)
            return None
    
    
    async def _generate_output(self, processed_input: ProcessedInput, ai_response: str, 
                              model_info: Dict[str, Any], logger: EchoLogger) -> Optional[GeneratedOutput]:
        """Generate output using OutputGenerator"""
        try:
            logger.debug("Starting output generation", category=LogCategory.BUSINESS_LOGIC)
            
            # Generate output
            output = await self.output_generator.generate(processed_input, ai_response, model_info)
            
            if output:
                logger.debug(
                    "Output generation completed",
                    category=LogCategory.BUSINESS_LOGIC,
                    output_type=output.output_type.value,
                    delivery_method=output.delivery_method.value
                )
            
            return output
            
        except Exception as e:
            logger.error("Output generation error", category=LogCategory.BUSINESS_LOGIC, exc_info=e)
            return None
    
    def _create_error_response(self, request: EchoRequest, error_message: str, start_time: datetime) -> EchoResponse:
        """Create an error response"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return EchoResponse(
            response=f"I apologize, but I encountered an error: {error_message}",
            conversation_id=request.conversation_id,
            model_used="error_handler",
            processing_time=processing_time,
            confidence=0.0,
            success=False,
            error_message=error_message,
            metadata={
                "error": True,
                "input_type": request.input_type.value
            }
        )
    
    def _update_metrics(self, processing_time: float, success: bool):
        """Update performance metrics"""
        self.request_count += 1
        self.total_processing_time += processing_time
        
        if not success:
            self.error_count += 1
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the orchestrator"""
        try:
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "2.0.0",
                "components": {
                    "config_manager": self.config_manager is not None,
                    "input_processor": self.input_processor is not None,
                    "output_generator": self.output_generator is not None,
                    "error_handler": self.error_handler is not None,
                    "logger": self.logger is not None
                },
                "metrics": {
                    "total_requests": self.request_count,
                    "error_count": self.error_count,
                    "error_rate": self.error_count / max(self.request_count, 1),
                    "average_processing_time": self.total_processing_time / max(self.request_count, 1)
                }
            }
            
            # Check if all components are healthy
            if not all(health_status["components"].values()):
                health_status["status"] = "degraded"
            
            return health_status
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        try:
            stats = {
                "orchestrator": {
                    "request_count": self.request_count,
                    "error_count": self.error_count,
                    "total_processing_time": self.total_processing_time,
                    "average_processing_time": self.total_processing_time / max(self.request_count, 1)
                },
                "container": self.container.get_container_stats(),
                "error_handler": self.error_handler.get_error_stats(),
                "logger": self.logger.get_metrics()
            }
            
            # Add component stats
            if self.input_processor:
                stats["input_processor"] = self.input_processor.get_processing_stats()
            
            if self.output_generator:
                stats["output_generator"] = self.output_generator.get_generation_stats()
            
            if self.config_manager:
                stats["config_manager"] = self.config_manager.get_stats()
            
            return stats
            
        except Exception as e:
            self.logger.error("Error getting stats", category=LogCategory.SYSTEM, exc_info=e)
            return {"error": str(e)}

# Global orchestrator instance
_orchestrator_instance: Optional[EchoOrchestrator] = None

async def get_orchestrator() -> EchoOrchestrator:
    """Get the global orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = EchoOrchestrator()
        # Give it a moment to initialize
        await asyncio.sleep(0.1)
    return _orchestrator_instance

# Convenience function for simple requests
async def process_query(query: str, user_id: str = "default", **kwargs) -> EchoResponse:
    """Convenience function to process a simple query"""
    orchestrator = await get_orchestrator()
    request = EchoRequest(query=query, user_id=user_id, **kwargs)
    return await orchestrator.process_request(request)
    async def _generate_ai_response(self, processed_input: ProcessedInput, logger: EchoLogger) -> tuple[Optional[str], Dict[str, Any]]:
        """Generate AI response using Ollama directly"""
        try:
            logger.debug("Starting AI response generation", category=LogCategory.BUSINESS_LOGIC)
            
            import requests
            
            # Prepare prompt
            prompt = f"""You are Echo, an intelligent AI assistant. Please respond to the following query:

Query: {processed_input.content}

Context: {processed_input.context}

Provide a helpful, informative response."""
            
            # Query Ollama directly
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "qwen2.5-coder:7b",
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    ai_response = response.json().get("response", "").strip()
                    
                    if ai_response:
                        # Model info
                        model_info = {
                            "model_name": "qwen2.5-coder:7b",
                            "confidence": 0.8,
                            "processing_method": "ollama_direct"
                        }
                        
                        logger.debug(
                            "AI response generated successfully",
                            category=LogCategory.BUSINESS_LOGIC,
                            response_length=len(ai_response),
                            model=model_info["model_name"]
                        )
                        
                        return ai_response, model_info
                    else:
                        logger.warning("Empty response from Ollama", category=LogCategory.EXTERNAL_SERVICE)
                else:
                    logger.error(f"Ollama request failed with status {response.status_code}", 
                                category=LogCategory.EXTERNAL_SERVICE)
            
            except requests.exceptions.RequestException as e:
                logger.error(f"Ollama connection error: {str(e)}", category=LogCategory.NETWORK)
            
            # Fallback response
            fallback_response = f"I received your message: '{processed_input.content}'. The AI system is currently being refactored with a new modular architecture featuring InputProcessor, OutputGenerator, ConfigurationManager, ErrorHandler, and structured logging. This is a fallback response to demonstrate the new system is working."
            
            model_info = {
                "model_name": "fallback_response",
                "confidence": 0.5,
                "processing_method": "fallback"
            }
            
            logger.info("Using fallback response", category=LogCategory.BUSINESS_LOGIC)
            return fallback_response, model_info
            
        except Exception as e:
            logger.error("AI response generation error", category=LogCategory.BUSINESS_LOGIC, exc_info=e)
            return None, {}
