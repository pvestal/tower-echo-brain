#!/usr/bin/env python3
"""
Abstract interface for Large Language Model (LLM) systems in Echo Brain.
Provides standardized LLM operations with Ollama abstraction.
Patrick Vestal - December 9, 2025
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, AsyncIterable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .ml_model_interface import MLModelInterface, ModelType, StreamingMLModelInterface


class MessageRole(Enum):
    """Message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class ResponseFormat(Enum):
    """LLM response formats."""
    TEXT = "text"
    JSON = "json"
    CODE = "code"
    MARKDOWN = "markdown"


@dataclass
class ChatMessage:
    """Single message in conversation."""
    role: MessageRole
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None


@dataclass
class ChatRequest:
    """Request for LLM chat completion."""
    messages: List[ChatMessage]
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    response_format: ResponseFormat = ResponseFormat.TEXT
    context_window: Optional[int] = None
    stream: bool = False


@dataclass
class ChatResponse:
    """Response from LLM chat completion."""
    content: str
    model_name: str
    usage: Dict[str, int]  # token counts
    finish_reason: str
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StreamChunk:
    """Single chunk from streaming response."""
    content: str
    delta: str
    finish_reason: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMInterface(StreamingMLModelInterface):
    """Abstract interface for Large Language Models."""

    def __init__(self, model_name: str, provider: str = "ollama"):
        """Initialize LLM interface."""
        super().__init__(model_name, ModelType.LLM)
        self._provider = provider
        self._context_window = 4096
        self._max_tokens = 2048

    @property
    def provider(self) -> str:
        """Get LLM provider name."""
        return self._provider

    @property
    def context_window(self) -> int:
        """Get context window size."""
        return self._context_window

    @property
    def max_tokens(self) -> int:
        """Get maximum generation tokens."""
        return self._max_tokens

    @abstractmethod
    async def chat_completion(
        self,
        request: ChatRequest,
        context: Optional[Dict[str, Any]] = None
    ) -> ChatResponse:
        """Generate chat completion.

        Args:
            request: ChatRequest with messages and parameters
            context: Optional context information

        Returns:
            ChatResponse with generated content
        """
        pass

    @abstractmethod
    async def chat_completion_stream(
        self,
        request: ChatRequest,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterable[StreamChunk]:
        """Generate streaming chat completion.

        Args:
            request: ChatRequest with messages and parameters
            context: Optional context information

        Yields:
            StreamChunk objects with partial responses
        """
        pass

    @abstractmethod
    async def simple_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate simple text completion.

        Args:
            prompt: Input prompt text
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            context: Optional context information

        Returns:
            Generated text completion
        """
        pass

    @abstractmethod
    async def analyze_sentiment(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze sentiment of text.

        Args:
            text: Text to analyze
            context: Optional context information

        Returns:
            Dictionary with sentiment analysis results
        """
        pass

    @abstractmethod
    async def extract_keywords(
        self,
        text: str,
        max_keywords: int = 10,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Extract keywords from text.

        Args:
            text: Text to analyze
            max_keywords: Maximum number of keywords
            context: Optional context information

        Returns:
            List of extracted keywords
        """
        pass

    @abstractmethod
    async def summarize_text(
        self,
        text: str,
        max_length: int = 100,
        style: str = "concise",
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Summarize text content.

        Args:
            text: Text to summarize
            max_length: Maximum summary length
            style: Summary style (concise, detailed, bullet_points)
            context: Optional context information

        Returns:
            Text summary
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and capabilities.

        Returns:
            Dictionary with model information
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        pass

    async def validate_chat_messages(self, messages: List[ChatMessage]) -> bool:
        """Validate chat message format.

        Args:
            messages: List of ChatMessage objects to validate

        Returns:
            True if valid, False otherwise
        """
        if not messages:
            return False

        total_tokens = 0
        for message in messages:
            if not isinstance(message.role, MessageRole):
                return False
            if not isinstance(message.content, str) or len(message.content.strip()) == 0:
                return False
            total_tokens += self.count_tokens(message.content)

        # Check if total tokens exceed context window
        return total_tokens <= self.context_window


class OllamaLLMInterface(LLMInterface):
    """Interface specific to Ollama LLM provider."""

    def __init__(self, model_name: str):
        """Initialize Ollama LLM interface."""
        super().__init__(model_name, "ollama")

    @abstractmethod
    async def pull_model(
        self,
        model_name: str,
        insecure: bool = False,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Pull model from Ollama registry.

        Args:
            model_name: Name of model to pull
            insecure: Allow insecure connections
            context: Optional context information

        Returns:
            True if pulled successfully, False otherwise
        """
        pass

    @abstractmethod
    async def list_local_models(self) -> List[str]:
        """List locally available models.

        Returns:
            List of model names
        """
        pass

    @abstractmethod
    async def delete_model(self, model_name: str) -> bool:
        """Delete local model.

        Args:
            model_name: Name of model to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        pass

    @abstractmethod
    async def copy_model(
        self,
        source_name: str,
        destination_name: str
    ) -> bool:
        """Copy/rename local model.

        Args:
            source_name: Source model name
            destination_name: Destination model name

        Returns:
            True if copied successfully, False otherwise
        """
        pass

    @abstractmethod
    async def show_model_info(self, model_name: str) -> Dict[str, Any]:
        """Show detailed model information.

        Args:
            model_name: Name of model to inspect

        Returns:
            Dictionary with model details
        """
        pass

    @abstractmethod
    async def create_modelfile(
        self,
        model_name: str,
        modelfile_content: str
    ) -> bool:
        """Create model from Modelfile.

        Args:
            model_name: Name for new model
            modelfile_content: Modelfile content

        Returns:
            True if created successfully, False otherwise
        """
        pass

    @abstractmethod
    async def get_server_status(self) -> Dict[str, Any]:
        """Get Ollama server status.

        Returns:
            Dictionary with server status information
        """
        pass


class CodeLLMInterface(LLMInterface):
    """Interface for code-specialized LLMs."""

    @abstractmethod
    async def generate_code(
        self,
        prompt: str,
        language: str,
        style: str = "clean",
        include_comments: bool = True,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate code from description.

        Args:
            prompt: Code description/requirements
            language: Programming language
            style: Code style (clean, concise, verbose)
            include_comments: Whether to include comments
            context: Optional context information

        Returns:
            Generated code
        """
        pass

    @abstractmethod
    async def explain_code(
        self,
        code: str,
        language: Optional[str] = None,
        detail_level: str = "medium",
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Explain code functionality.

        Args:
            code: Code to explain
            language: Optional programming language
            detail_level: Level of detail (low, medium, high)
            context: Optional context information

        Returns:
            Code explanation
        """
        pass

    @abstractmethod
    async def review_code(
        self,
        code: str,
        language: Optional[str] = None,
        focus_areas: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Review code for issues and improvements.

        Args:
            code: Code to review
            language: Optional programming language
            focus_areas: Areas to focus on (security, performance, style)
            context: Optional context information

        Returns:
            Dictionary with review findings
        """
        pass

    @abstractmethod
    async def fix_code(
        self,
        code: str,
        error_message: str,
        language: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Fix code based on error message.

        Args:
            code: Code with issues
            error_message: Error description
            language: Optional programming language
            context: Optional context information

        Returns:
            Fixed code
        """
        pass

    @abstractmethod
    async def refactor_code(
        self,
        code: str,
        refactor_goals: List[str],
        language: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Refactor code for improvements.

        Args:
            code: Code to refactor
            refactor_goals: Goals (readability, performance, maintainability)
            language: Optional programming language
            context: Optional context information

        Returns:
            Refactored code
        """
        pass

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Get list of supported programming languages.

        Returns:
            List of supported language names
        """
        pass


class ConversationalLLMInterface(LLMInterface):
    """Interface for conversational LLMs with memory."""

    @abstractmethod
    async def start_conversation(
        self,
        conversation_id: str,
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Start a new conversation session.

        Args:
            conversation_id: Unique conversation identifier
            system_prompt: Optional system prompt
            context: Optional context information

        Returns:
            True if started successfully, False otherwise
        """
        pass

    @abstractmethod
    async def continue_conversation(
        self,
        conversation_id: str,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ChatResponse:
        """Continue existing conversation.

        Args:
            conversation_id: Conversation identifier
            message: User message
            context: Optional context information

        Returns:
            ChatResponse with assistant reply
        """
        pass

    @abstractmethod
    async def get_conversation_history(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[ChatMessage]:
        """Get conversation history.

        Args:
            conversation_id: Conversation identifier
            limit: Optional limit on messages returned

        Returns:
            List of ChatMessage objects
        """
        pass

    @abstractmethod
    async def clear_conversation(self, conversation_id: str) -> bool:
        """Clear conversation history.

        Args:
            conversation_id: Conversation identifier

        Returns:
            True if cleared successfully, False otherwise
        """
        pass

    @abstractmethod
    async def summarize_conversation(
        self,
        conversation_id: str,
        style: str = "concise"
    ) -> str:
        """Summarize conversation history.

        Args:
            conversation_id: Conversation identifier
            style: Summary style

        Returns:
            Conversation summary
        """
        pass

    @abstractmethod
    async def get_conversation_sentiment(
        self,
        conversation_id: str
    ) -> Dict[str, Any]:
        """Analyze conversation sentiment over time.

        Args:
            conversation_id: Conversation identifier

        Returns:
            Dictionary with sentiment analysis
        """
        pass