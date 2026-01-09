#!/usr/bin/env python3
"""
Conversation Management Interface Protocol
Defines contracts for conversation handling and context management
"""

from typing import Protocol, runtime_checkable, List, Dict, Optional, Any, AsyncGenerator
from datetime import datetime
from enum import Enum

class ConversationStatus(Enum):
    """Conversation status enumeration"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"
    PAUSED = "paused"

class MessageType(Enum):
    """Message type enumeration"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    ERROR = "error"

@runtime_checkable
class ConversationManagerInterface(Protocol):
    """
    Protocol for conversation management systems

    Defines standardized methods for handling conversations,
    context management, and message processing.
    """

    async def create_conversation(self, user_id: str, title: Optional[str] = None,
                                metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new conversation

        Args:
            user_id: User identifier
            title: Optional conversation title
            metadata: Optional conversation metadata

        Returns:
            str: Conversation ID
        """
        ...

    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get conversation details

        Args:
            conversation_id: Conversation identifier

        Returns:
            Optional[Dict]: Conversation data or None if not found
        """
        ...

    async def list_conversations(self, user_id: str, status: Optional[ConversationStatus] = None,
                               limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List user conversations

        Args:
            user_id: User identifier
            status: Optional status filter
            limit: Maximum conversations to return
            offset: Number of conversations to skip

        Returns:
            List[Dict]: List of conversations
        """
        ...

    async def add_message(self, conversation_id: str, message_type: MessageType,
                         content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a message to conversation

        Args:
            conversation_id: Conversation identifier
            message_type: Type of message
            content: Message content
            metadata: Optional message metadata

        Returns:
            str: Message ID
        """
        ...

    async def get_messages(self, conversation_id: str, limit: int = 100,
                         offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get conversation messages

        Args:
            conversation_id: Conversation identifier
            limit: Maximum messages to return
            offset: Number of messages to skip

        Returns:
            List[Dict]: List of messages
        """
        ...

    async def update_conversation(self, conversation_id: str,
                                updates: Dict[str, Any]) -> bool:
        """
        Update conversation metadata

        Args:
            conversation_id: Conversation identifier
            updates: Dictionary of fields to update

        Returns:
            bool: True if updated successfully
        """
        ...

    async def archive_conversation(self, conversation_id: str) -> bool:
        """
        Archive a conversation

        Args:
            conversation_id: Conversation identifier

        Returns:
            bool: True if archived successfully
        """
        ...

    async def delete_conversation(self, conversation_id: str,
                                soft_delete: bool = True) -> bool:
        """
        Delete a conversation

        Args:
            conversation_id: Conversation identifier
            soft_delete: Whether to soft delete (mark as deleted) or hard delete

        Returns:
            bool: True if deleted successfully
        """
        ...

    async def search_conversations(self, user_id: str, query: str,
                                 limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search conversations by content

        Args:
            user_id: User identifier
            query: Search query
            limit: Maximum results

        Returns:
            List[Dict]: Matching conversations
        """
        ...

    async def get_conversation_context(self, conversation_id: str,
                                     max_messages: int = 10) -> str:
        """
        Get formatted conversation context

        Args:
            conversation_id: Conversation identifier
            max_messages: Maximum messages to include in context

        Returns:
            str: Formatted conversation context
        """
        ...

    async def get_conversation_summary(self, conversation_id: str) -> Optional[str]:
        """
        Get or generate conversation summary

        Args:
            conversation_id: Conversation identifier

        Returns:
            Optional[str]: Conversation summary or None
        """
        ...


@runtime_checkable
class ContextManagerInterface(Protocol):
    """
    Protocol for context management systems
    """

    async def store_context(self, context_id: str, context_data: Dict[str, Any],
                          ttl: Optional[int] = None) -> bool:
        """
        Store context data

        Args:
            context_id: Context identifier
            context_data: Context data to store
            ttl: Optional time-to-live in seconds

        Returns:
            bool: True if stored successfully
        """
        ...

    async def get_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve context data

        Args:
            context_id: Context identifier

        Returns:
            Optional[Dict]: Context data or None if not found
        """
        ...

    async def update_context(self, context_id: str,
                           updates: Dict[str, Any]) -> bool:
        """
        Update context data

        Args:
            context_id: Context identifier
            updates: Updates to apply

        Returns:
            bool: True if updated successfully
        """
        ...

    async def delete_context(self, context_id: str) -> bool:
        """
        Delete context data

        Args:
            context_id: Context identifier

        Returns:
            bool: True if deleted successfully
        """
        ...

    async def list_contexts(self, pattern: Optional[str] = None) -> List[str]:
        """
        List available context IDs

        Args:
            pattern: Optional pattern to filter context IDs

        Returns:
            List[str]: List of context IDs
        """
        ...

    async def merge_contexts(self, target_id: str, source_ids: List[str]) -> bool:
        """
        Merge multiple contexts into a target context

        Args:
            target_id: Target context ID
            source_ids: List of source context IDs to merge

        Returns:
            bool: True if merged successfully
        """
        ...


@runtime_checkable
class MessageProcessorInterface(Protocol):
    """
    Protocol for message processing systems
    """

    async def process_message(self, message: str, context: Dict[str, Any]) -> str:
        """
        Process a message and generate response

        Args:
            message: Input message
            context: Processing context

        Returns:
            str: Processed response
        """
        ...

    async def process_streaming(self, message: str,
                              context: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """
        Process message with streaming response

        Args:
            message: Input message
            context: Processing context

        Yields:
            str: Response chunks
        """
        ...

    def can_handle_message(self, message: str, context: Dict[str, Any]) -> bool:
        """
        Check if processor can handle this message

        Args:
            message: Input message
            context: Processing context

        Returns:
            bool: True if can handle this message
        """
        ...

    def get_processor_info(self) -> Dict[str, Any]:
        """
        Get processor information

        Returns:
            Dict: Processor metadata
        """
        ...


@runtime_checkable
class ConversationAnalyzerInterface(Protocol):
    """
    Protocol for conversation analysis systems
    """

    async def analyze_sentiment(self, conversation_id: str) -> Dict[str, Any]:
        """
        Analyze conversation sentiment

        Args:
            conversation_id: Conversation identifier

        Returns:
            Dict: Sentiment analysis results
        """
        ...

    async def extract_topics(self, conversation_id: str) -> List[str]:
        """
        Extract main topics from conversation

        Args:
            conversation_id: Conversation identifier

        Returns:
            List[str]: List of extracted topics
        """
        ...

    async def detect_intent(self, message: str) -> Dict[str, Any]:
        """
        Detect intent from a message

        Args:
            message: Input message

        Returns:
            Dict: Intent detection results
        """
        ...

    async def summarize_conversation(self, conversation_id: str,
                                   max_length: int = 500) -> str:
        """
        Generate conversation summary

        Args:
            conversation_id: Conversation identifier
            max_length: Maximum summary length

        Returns:
            str: Conversation summary
        """
        ...

    async def identify_action_items(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Identify action items from conversation

        Args:
            conversation_id: Conversation identifier

        Returns:
            List[Dict]: List of action items with priorities
        """
        ...

    async def get_conversation_insights(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get comprehensive conversation insights

        Args:
            conversation_id: Conversation identifier

        Returns:
            Dict: Conversation insights including patterns, trends, etc.
        """
        ...