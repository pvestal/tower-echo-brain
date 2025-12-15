#!/usr/bin/env python3
"""
Business Logic Middleware for Echo Brain API
Centralized middleware for applying Patrick's business logic patterns to all API responses
"""
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class BusinessLogicMiddleware:
    """
    Middleware for applying business logic patterns to API responses.
    Provides centralized pattern application across all endpoints.
    """

    def __init__(self):
        self.conversation_manager = None
        self.application_stats = {
            'total_requests': 0,
            'patterns_applied': 0,
            'middleware_errors': 0
        }
        logger.info("✅ BusinessLogicMiddleware initialized")

    def initialize(self, conversation_manager):
        """Initialize middleware with conversation manager instance"""
        self.conversation_manager = conversation_manager
        logger.info("🔧 BusinessLogicMiddleware connected to ConversationManager")

    def apply_to_response(self, query: str, response: str, endpoint_type: str = "query") -> str:
        """
        Apply business logic patterns to any API response.

        Args:
            query: Original user query
            response: Base response from API endpoint
            endpoint_type: Type of endpoint (for stats tracking)

        Returns:
            Response with business logic patterns applied
        """
        self.application_stats['total_requests'] += 1

        # Guard clause - no conversation manager available
        if not self.conversation_manager:
            logger.warning("⚠️ BusinessLogicMiddleware: ConversationManager not available")
            return response

        # Guard clause - empty query or response
        if not query or not query.strip():
            logger.debug("🔧 BusinessLogicMiddleware: Empty query, skipping pattern application")
            return response

        if not response or not response.strip():
            logger.debug("🔧 BusinessLogicMiddleware: Empty response, skipping pattern application")
            return response

        try:
            # Apply business logic patterns via conversation manager
            response_with_patterns = self.conversation_manager.apply_business_logic(query, response)

            # Track successful applications
            if response_with_patterns != response:
                self.application_stats['patterns_applied'] += 1
                logger.debug(f"🧠 BusinessLogicMiddleware: Patterns applied to {endpoint_type} response")

            return response_with_patterns

        except Exception as e:
            self.application_stats['middleware_errors'] += 1
            logger.error(f"BusinessLogicMiddleware error for {endpoint_type} endpoint: {e}")
            return response  # Return original response on error

    def get_middleware_stats(self) -> Dict[str, Any]:
        """Get statistics about middleware performance"""
        total_requests = max(1, self.application_stats['total_requests'])

        return {
            **self.application_stats,
            'application_rate': (self.application_stats['patterns_applied'] / total_requests) * 100,
            'error_rate': (self.application_stats['middleware_errors'] / total_requests) * 100,
            'success_rate': ((total_requests - self.application_stats['middleware_errors']) / total_requests) * 100
        }

    def reset_stats(self):
        """Reset middleware statistics"""
        self.application_stats = {
            'total_requests': 0,
            'patterns_applied': 0,
            'middleware_errors': 0
        }
        logger.info("🔧 BusinessLogicMiddleware: Statistics reset")

# Global middleware instance
business_logic_middleware = BusinessLogicMiddleware()

def apply_business_logic_to_response(query: str, response: str, endpoint_type: str = "query") -> str:
    """
    Convenience function for applying business logic patterns.
    Can be easily imported and used by any API endpoint.
    """
    return business_logic_middleware.apply_to_response(query, response, endpoint_type)