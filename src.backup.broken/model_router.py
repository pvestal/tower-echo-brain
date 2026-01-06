
class ModelRouter:
    """Routes queries to appropriate models based on complexity"""

    def __init__(self):
        """Initialize the ModelRouter with empty model registry"""
        self.models = {}
        self.complexity = 0
        logger.info("ModelRouter initialized")

    def calculate_complexity(self, text: str) -> float:
        """
        Calculate the complexity score of input text.

        Args:
            text: The input text to analyze

        Returns:
            float: Complexity score based on word count and thresholds
        """
        logger.info(f"Calculating complexity for text of length {len(text)}")
        words = text.split()
        complexity = len(words) * 0.1
        if len(words) > 100:
            complexity *= 1.5
        if len(words) > 500:
            complexity *= 2
        return complexity

    def select_model(self, query: str) -> str:
        """
        Select the appropriate model based on query complexity.

        Args:
            query: The input query to route

        Returns:
            str: Name of the selected model
        """
        logger.info("Selecting model for query")
        complexity = self.calculate_complexity(query)
        if complexity < 10:
            return "simple_model"
        elif complexity < 50:
            return "medium_model"
        else:
            return "complex_model"

    async def route_query(self, query: str, context: dict = None) -> dict:
        """
        Route query using intelligent database-driven routing.

        Args:
            query: The query to route and execute
            context: Optional context for the query

        Returns:
            dict: Response with success status, model used, and result
        """
        import httpx
        import time

        try:
            # Use the intelligent database router
            from src.core.db_model_router import intelligent_router

            # Get conversation history from context
            conversation_history = []
            if context and isinstance(context, dict):
                conversation_history = context.get('conversation_history', [])

            # Get intelligent routing decision
            start_time = time.time()
            decision = intelligent_router.route_query(query, conversation_history)

            logger.info(f"Intelligent routing: {decision.model} (complexity: {decision.complexity_score}, "
                       f"intent: {decision.intent}, domain: {decision.domain})")

            # Call Ollama API with the selected model
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": decision.model,
                        "prompt": query,
                        "stream": False,
                        "options": {
                            "temperature": decision.temperature,
                            "num_predict": decision.max_tokens
                        }
                    }
                )

                response_time_ms = int((time.time() - start_time) * 1000)

                if response.status_code == 200:
                    result = response.json()

                    # Log performance
                    intelligent_router.log_performance(
                        decision=decision,
                        response_time_ms=response_time_ms,
                        success=True,
                        tokens_used=len(result.get("response", "").split())
                    )

                    return {
                        "success": True,
                        "model_used": decision.model,
                        "tier": f"{decision.intent or 'general'}/{decision.domain or 'general'}",
                        "complexity_score": decision.complexity_score,
                        "response": result.get("response", ""),
                        "reasoning": decision.reasoning,
                        "requires_context": decision.requires_context,
                        "complexity_details": {
                            "word_count": len(query.split()),
                            "complexity_score": decision.complexity_score,
                            "intent": decision.intent,
                            "domain": decision.domain
                        }
                    }
                else:
                    logger.error(f"Ollama API error: {response.status_code}")
                    return {
                        "success": False,
                        "error": f"API error: {response.status_code}",
                        "model_used": decision.model
                    }

        except Exception as e:
            logger.error(f"Error routing query: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model_used": "none"
            }

# Add import for logging
import logging
logger = logging.getLogger(__name__)
