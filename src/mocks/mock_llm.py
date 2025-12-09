#!/usr/bin/env python3
"""
Mock LLM implementation for testing Echo Brain without ML dependencies.
Returns realistic fake data for all LLM operations.
Patrick Vestal - December 9, 2025
"""

import asyncio
import random
import time
from typing import List, Dict, Any, Optional, Union, AsyncIterable
from datetime import datetime

from ..interfaces.llm_interface import (
    LLMInterface, ChatRequest, ChatResponse, StreamChunk, ChatMessage,
    MessageRole, ResponseFormat, OllamaLLMInterface, CodeLLMInterface,
    ConversationalLLMInterface
)
from ..interfaces.ml_model_interface import ModelStatus, ModelMetadata, ModelType


class MockLLM(LLMInterface):
    """Mock LLM implementation for testing."""

    def __init__(self, model_name: str = "mock-llama", provider: str = "mock"):
        """Initialize mock LLM."""
        super().__init__(model_name, provider)
        self._context_window = 4096
        self._max_tokens = 2048
        self._status = ModelStatus.READY

        # Mock model metadata
        self._metadata = ModelMetadata(
            name=model_name,
            version="1.0.0-mock",
            model_type=ModelType.LLM,
            parameters=7000000000,  # 7B parameters
            memory_usage=4096,  # 4GB
            compute_requirements={"gpu_memory": "4GB", "cpu_cores": 4},
            capabilities=["text_generation", "chat", "code", "analysis"],
            created_at=datetime.now()
        )

        # Mock responses for different types of queries
        self._mock_responses = {
            "greeting": [
                "Hello! I'm here to help you with any questions or tasks.",
                "Hi there! How can I assist you today?",
                "Greetings! What can I help you with?"
            ],
            "code": [
                "Here's a Python implementation:\n\n```python\ndef example_function():\n    return 'Hello World'\n```",
                "I can help you with that code. Here's an approach:\n\n```python\n# Your code here\npass\n```",
                "Here's a solution in Python:\n\n```python\nimport os\nprint('Working directory:', os.getcwd())\n```"
            ],
            "analysis": [
                "Based on my analysis, I can see several key points:\n1. Data shows positive trends\n2. Performance metrics are within normal ranges\n3. No critical issues detected",
                "My analysis indicates:\n• System is functioning normally\n• All parameters are within expected ranges\n• No immediate action required",
                "Analysis complete. Key findings:\n- All systems operational\n- Performance: Good\n- Status: Healthy"
            ],
            "explanation": [
                "Let me explain this concept:\n\nThis is a complex topic that involves multiple components working together to achieve the desired outcome.",
                "To understand this better, think of it as a system with interconnected parts that each serve a specific purpose.",
                "The key principle here is that each component builds upon the previous one to create a cohesive whole."
            ],
            "default": [
                "I understand your question. Let me provide a helpful response based on the information you've shared.",
                "That's an interesting point. Here's my perspective on this topic.",
                "I can help with that. Based on your request, here's what I recommend.",
                "Thank you for your question. Here's a comprehensive response to address your needs."
            ]
        }

    async def load(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Mock model loading."""
        self._status = ModelStatus.LOADING
        await asyncio.sleep(0.1)  # Simulate loading time
        self._status = ModelStatus.READY
        return True

    async def unload(self) -> bool:
        """Mock model unloading."""
        self._status = ModelStatus.UNLOADED
        return True

    async def predict(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """Mock prediction - delegates to chat completion."""
        if isinstance(input_data, str):
            response = await self.simple_completion(input_data, context=context)
            return response
        elif isinstance(input_data, ChatRequest):
            return await self.chat_completion(input_data, context)
        else:
            return "Mock LLM response for unknown input type"

    async def batch_predict(self, inputs: List[Any], context: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Mock batch prediction."""
        results = []
        for input_data in inputs:
            result = await self.predict(input_data, context)
            results.append(result)
        return results

    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        if isinstance(input_data, str):
            return len(input_data.strip()) > 0
        elif isinstance(input_data, ChatRequest):
            return len(input_data.messages) > 0
        return False

    async def health_check(self) -> Dict[str, Any]:
        """Mock health check."""
        return {
            "status": "healthy",
            "model_loaded": self._status == ModelStatus.READY,
            "memory_usage": 2048,  # MB
            "last_request": datetime.now().isoformat(),
            "requests_processed": random.randint(100, 1000)
        }

    async def predict_stream(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> AsyncIterable[Dict[str, Any]]:
        """Mock streaming prediction."""
        response_text = await self.simple_completion(str(input_data), context=context)
        words = response_text.split()

        for i, word in enumerate(words):
            yield {
                "token": word + " ",
                "partial_response": " ".join(words[:i+1]),
                "finished": i == len(words) - 1,
                "metadata": {"token_index": i}
            }
            await asyncio.sleep(0.01)  # Simulate streaming delay

    async def chat_completion(self, request: ChatRequest, context: Optional[Dict[str, Any]] = None) -> ChatResponse:
        """Mock chat completion."""
        # Simulate processing time
        processing_start = time.time()
        await asyncio.sleep(random.uniform(0.1, 0.3))

        # Determine response type based on last message content
        last_message = request.messages[-1].content.lower() if request.messages else ""
        response_category = self._categorize_query(last_message)

        # Select mock response
        response_content = random.choice(self._mock_responses.get(response_category, self._mock_responses["default"]))

        # Add some variation based on temperature
        if request.temperature > 0.8:
            response_content += f"\n\n*Note: High creativity mode engaged (temperature: {request.temperature})*"
        elif request.temperature < 0.3:
            response_content += f"\n\n*Note: Focused response mode (temperature: {request.temperature})*"

        processing_time = time.time() - processing_start

        return ChatResponse(
            content=response_content,
            model_name=self.name,
            usage={
                "prompt_tokens": sum(len(msg.content.split()) for msg in request.messages),
                "completion_tokens": len(response_content.split()),
                "total_tokens": sum(len(msg.content.split()) for msg in request.messages) + len(response_content.split())
            },
            finish_reason="stop",
            processing_time=processing_time,
            metadata={"temperature": request.temperature, "category": response_category}
        )

    async def chat_completion_stream(self, request: ChatRequest, context: Optional[Dict[str, Any]] = None) -> AsyncIterable[StreamChunk]:
        """Mock streaming chat completion."""
        # Get the full response first
        full_response = await self.chat_completion(request, context)
        words = full_response.content.split()

        accumulated_content = ""
        for i, word in enumerate(words):
            delta = word + " "
            accumulated_content += delta

            yield StreamChunk(
                content=accumulated_content,
                delta=delta,
                finish_reason="stop" if i == len(words) - 1 else None,
                metadata={"token_index": i, "total_tokens": len(words)}
            )
            await asyncio.sleep(0.05)  # Simulate streaming delay

    async def simple_completion(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, context: Optional[Dict[str, Any]] = None) -> str:
        """Mock simple completion."""
        # Simulate processing delay
        await asyncio.sleep(random.uniform(0.05, 0.15))

        category = self._categorize_query(prompt.lower())
        response = random.choice(self._mock_responses.get(category, self._mock_responses["default"]))

        # Truncate if max_tokens specified
        if max_tokens:
            words = response.split()[:max_tokens]
            response = " ".join(words)

        return response

    async def analyze_sentiment(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mock sentiment analysis."""
        await asyncio.sleep(0.1)

        # Simple mock sentiment based on text content
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
        negative_words = ["bad", "terrible", "awful", "horrible", "disappointing", "poor"]

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            sentiment = "positive"
            score = 0.7 + random.uniform(0, 0.3)
        elif negative_count > positive_count:
            sentiment = "negative"
            score = 0.1 + random.uniform(0, 0.3)
        else:
            sentiment = "neutral"
            score = 0.4 + random.uniform(0, 0.2)

        return {
            "sentiment": sentiment,
            "confidence": score,
            "score": score,
            "positive_probability": score if sentiment == "positive" else 1 - score,
            "negative_probability": score if sentiment == "negative" else 1 - score,
            "neutral_probability": 0.5 if sentiment == "neutral" else random.uniform(0.1, 0.3)
        }

    async def extract_keywords(self, text: str, max_keywords: int = 10, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Mock keyword extraction."""
        await asyncio.sleep(0.1)

        # Simple mock keyword extraction
        words = text.lower().split()
        # Filter out common stop words
        stop_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "up", "about", "into", "through", "during", "before", "after", "above", "below", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "should", "could", "can", "may", "might", "must", "shall", "a", "an", "this", "that", "these", "those"}

        keywords = [word for word in words if len(word) > 3 and word not in stop_words]

        # Remove duplicates and limit
        unique_keywords = list(dict.fromkeys(keywords))[:max_keywords]

        return unique_keywords

    async def summarize_text(self, text: str, max_length: int = 100, style: str = "concise", context: Optional[Dict[str, Any]] = None) -> str:
        """Mock text summarization."""
        await asyncio.sleep(0.1)

        if style == "bullet_points":
            return "• Key point 1: Main topic overview\n• Key point 2: Important details\n• Key point 3: Conclusion and implications"
        elif style == "detailed":
            return f"This text discusses several important concepts. The main theme revolves around the central topic, with supporting details that provide context and depth. The conclusion ties together the various elements to form a comprehensive understanding."
        else:  # concise
            words = text.split()
            if len(words) <= 20:
                return text
            return " ".join(words[:20]) + "..."

    def get_model_info(self) -> Dict[str, Any]:
        """Get mock model information."""
        return {
            "name": self.name,
            "provider": self.provider,
            "version": "1.0.0-mock",
            "parameters": "7B",
            "context_window": self.context_window,
            "max_tokens": self.max_tokens,
            "capabilities": ["text_generation", "chat", "analysis", "code"],
            "languages": ["English", "Python", "JavaScript", "SQL"],
            "model_size": "4.1GB",
            "quantization": "Q4_K_M",
            "architecture": "llama"
        }

    def count_tokens(self, text: str) -> int:
        """Mock token counting (rough approximation)."""
        return len(text.split()) + text.count(',') + text.count('.') + text.count('!')

    def _categorize_query(self, query: str) -> str:
        """Categorize query to select appropriate mock response."""
        query_lower = query.lower()

        if any(word in query_lower for word in ["hello", "hi", "hey", "greetings"]):
            return "greeting"
        elif any(word in query_lower for word in ["code", "python", "function", "class", "def", "import"]):
            return "code"
        elif any(word in query_lower for word in ["analyze", "analysis", "examine", "study", "review"]):
            return "analysis"
        elif any(word in query_lower for word in ["explain", "what is", "how does", "tell me about"]):
            return "explanation"
        else:
            return "default"


class MockOllamaLLM(MockLLM, OllamaLLMInterface):
    """Mock Ollama LLM implementation."""

    def __init__(self, model_name: str = "mock-ollama-llama"):
        """Initialize mock Ollama LLM."""
        super().__init__(model_name, "ollama")

    async def pull_model(self, model_name: str, insecure: bool = False, context: Optional[Dict[str, Any]] = None) -> bool:
        """Mock model pulling."""
        await asyncio.sleep(1.0)  # Simulate download time
        return True

    async def list_local_models(self) -> List[str]:
        """Mock local model listing."""
        return [
            "llama3.1:8b",
            "llama3.1:70b",
            "qwen2.5-coder:32b",
            "mixtral:8x7b",
            "deepseek-coder:latest",
            "codellama:70b"
        ]

    async def delete_model(self, model_name: str) -> bool:
        """Mock model deletion."""
        await asyncio.sleep(0.2)
        return True

    async def copy_model(self, source_name: str, destination_name: str) -> bool:
        """Mock model copying."""
        await asyncio.sleep(0.5)
        return True

    async def show_model_info(self, model_name: str) -> Dict[str, Any]:
        """Mock model info display."""
        return {
            "modelfile": "FROM llama3.1\nPARAMETER temperature 0.7",
            "parameters": "7B",
            "template": "{{ .System }}\n{{ .Prompt }}",
            "system": "You are a helpful AI assistant.",
            "details": {
                "format": "gguf",
                "family": "llama",
                "families": ["llama"],
                "parameter_size": "8.0B",
                "quantization_level": "Q4_0"
            },
            "size": 4661000000,
            "modified_at": "2025-12-09T10:00:00Z",
            "digest": "mock-digest-123456"
        }

    async def create_modelfile(self, model_name: str, modelfile_content: str) -> bool:
        """Mock modelfile creation."""
        await asyncio.sleep(0.5)
        return True

    async def get_server_status(self) -> Dict[str, Any]:
        """Mock server status."""
        return {
            "status": "running",
            "version": "0.1.0-mock",
            "models_loaded": random.randint(1, 5),
            "total_memory": "16GB",
            "available_memory": f"{random.randint(8, 15)}GB",
            "gpu_available": True,
            "gpu_memory": "12GB VRAM"
        }


class MockCodeLLM(MockLLM, CodeLLMInterface):
    """Mock code-specialized LLM."""

    def __init__(self, model_name: str = "mock-code-llama"):
        """Initialize mock code LLM."""
        super().__init__(model_name, "mock-code")

    async def generate_code(self, prompt: str, language: str, style: str = "clean", include_comments: bool = True, context: Optional[Dict[str, Any]] = None) -> str:
        """Mock code generation."""
        await asyncio.sleep(0.2)

        comment_prefix = {
            "python": "#",
            "javascript": "//",
            "java": "//",
            "cpp": "//",
            "c": "//",
            "rust": "//",
            "go": "//"
        }.get(language.lower(), "#")

        if language.lower() == "python":
            code = f"""def example_function():
    {comment_prefix} Generated code based on: {prompt[:50]}...
    {comment_prefix} Style: {style}
    return "Hello, World!"

if __name__ == "__main__":
    result = example_function()
    print(result)"""
        elif language.lower() == "javascript":
            code = f"""function exampleFunction() {{
    {comment_prefix} Generated code based on: {prompt[:50]}...
    {comment_prefix} Style: {style}
    return "Hello, World!";
}}

console.log(exampleFunction());"""
        else:
            code = f"""{comment_prefix} Generated {language} code
{comment_prefix} Based on: {prompt[:50]}...
{comment_prefix} Style: {style}

// Your {language} code here
"""

        if not include_comments:
            # Remove comment lines
            code = "\n".join(line for line in code.split("\n") if not line.strip().startswith(comment_prefix))

        return code

    async def explain_code(self, code: str, language: Optional[str] = None, detail_level: str = "medium", context: Optional[Dict[str, Any]] = None) -> str:
        """Mock code explanation."""
        await asyncio.sleep(0.1)

        if detail_level == "high":
            return f"""This code performs several important operations:

1. Function Definition: The code defines a function that encapsulates the main logic.
2. Data Processing: It processes input data according to specified parameters.
3. Return Statement: The function returns the processed result.
4. Error Handling: Appropriate error handling is implemented.
5. Best Practices: The code follows {language or 'language'} best practices.

The implementation is efficient and maintainable, using standard library functions where appropriate."""
        elif detail_level == "low":
            return "This code defines a function that processes data and returns a result."
        else:  # medium
            return f"This {language or 'code'} implements a function that takes input, processes it according to the specified logic, and returns the result. The implementation follows standard conventions and includes proper error handling."

    async def review_code(self, code: str, language: Optional[str] = None, focus_areas: Optional[List[str]] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mock code review."""
        await asyncio.sleep(0.2)

        focus_areas = focus_areas or ["style", "performance", "security"]

        issues = []
        suggestions = []

        if "style" in focus_areas:
            issues.append({
                "type": "style",
                "severity": "minor",
                "line": random.randint(1, 10),
                "message": "Consider using more descriptive variable names"
            })
            suggestions.append("Follow PEP 8 style guidelines for better readability")

        if "performance" in focus_areas:
            suggestions.append("Consider using list comprehension for better performance")

        if "security" in focus_areas:
            issues.append({
                "type": "security",
                "severity": "low",
                "line": random.randint(5, 15),
                "message": "Validate input parameters to prevent potential issues"
            })

        return {
            "overall_score": random.uniform(7.5, 9.5),
            "issues": issues,
            "suggestions": suggestions,
            "strengths": [
                "Clean code structure",
                "Good function organization",
                "Proper error handling"
            ],
            "metrics": {
                "complexity": random.randint(3, 8),
                "maintainability": random.uniform(8.0, 10.0),
                "readability": random.uniform(7.5, 9.5)
            }
        }

    async def fix_code(self, code: str, error_message: str, language: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> str:
        """Mock code fixing."""
        await asyncio.sleep(0.1)

        return f"""# Fixed code - addressed: {error_message[:50]}...

{code}

# Fix applied:
# - Corrected syntax error
# - Added proper error handling
# - Updated variable names for clarity
"""

    async def refactor_code(self, code: str, refactor_goals: List[str], language: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> str:
        """Mock code refactoring."""
        await asyncio.sleep(0.3)

        return f"""# Refactored code with goals: {', '.join(refactor_goals)}

{code}

# Refactoring applied:
# - Improved readability through better naming
# - Enhanced performance with optimized algorithms
# - Increased maintainability with modular structure
"""

    def get_supported_languages(self) -> List[str]:
        """Mock supported languages."""
        return [
            "python", "javascript", "typescript", "java", "cpp", "c", "rust",
            "go", "php", "ruby", "swift", "kotlin", "scala", "r", "sql", "bash"
        ]


class MockConversationalLLM(MockLLM, ConversationalLLMInterface):
    """Mock conversational LLM with memory."""

    def __init__(self, model_name: str = "mock-conversational-llama"):
        """Initialize mock conversational LLM."""
        super().__init__(model_name, "mock-conversational")
        self._conversations: Dict[str, List[ChatMessage]] = {}

    async def start_conversation(self, conversation_id: str, system_prompt: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> bool:
        """Start mock conversation."""
        self._conversations[conversation_id] = []
        if system_prompt:
            self._conversations[conversation_id].append(
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt, timestamp=datetime.now())
            )
        return True

    async def continue_conversation(self, conversation_id: str, message: str, context: Optional[Dict[str, Any]] = None) -> ChatResponse:
        """Continue mock conversation."""
        if conversation_id not in self._conversations:
            await self.start_conversation(conversation_id)

        # Add user message
        user_message = ChatMessage(role=MessageRole.USER, content=message, timestamp=datetime.now())
        self._conversations[conversation_id].append(user_message)

        # Generate response based on conversation history
        history_length = len(self._conversations[conversation_id])
        if history_length > 10:
            response_content = "That's a great continuation of our discussion. Based on our conversation history, I can provide more personalized insights."
        elif history_length > 5:
            response_content = "I remember our previous exchanges. Let me build on what we've discussed."
        else:
            response_content = "Thank you for starting this conversation. I'm here to help!"

        # Create assistant response
        assistant_message = ChatMessage(role=MessageRole.ASSISTANT, content=response_content, timestamp=datetime.now())
        self._conversations[conversation_id].append(assistant_message)

        return ChatResponse(
            content=response_content,
            model_name=self.name,
            usage={
                "prompt_tokens": sum(len(msg.content.split()) for msg in self._conversations[conversation_id]),
                "completion_tokens": len(response_content.split()),
                "total_tokens": sum(len(msg.content.split()) for msg in self._conversations[conversation_id]) + len(response_content.split())
            },
            finish_reason="stop",
            processing_time=random.uniform(0.1, 0.3),
            metadata={"conversation_id": conversation_id, "history_length": history_length}
        )

    async def get_conversation_history(self, conversation_id: str, limit: Optional[int] = None) -> List[ChatMessage]:
        """Get mock conversation history."""
        if conversation_id not in self._conversations:
            return []

        messages = self._conversations[conversation_id]
        if limit:
            return messages[-limit:]
        return messages.copy()

    async def clear_conversation(self, conversation_id: str) -> bool:
        """Clear mock conversation."""
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            return True
        return False

    async def summarize_conversation(self, conversation_id: str, style: str = "concise") -> str:
        """Mock conversation summarization."""
        if conversation_id not in self._conversations:
            return "No conversation found."

        messages = self._conversations[conversation_id]
        user_messages = [msg for msg in messages if msg.role == MessageRole.USER]

        if style == "detailed":
            return f"This conversation included {len(messages)} total messages with {len(user_messages)} user interactions. Topics covered include general assistance and information exchange."
        else:
            return f"Conversation summary: {len(user_messages)} user messages exchanged covering various topics."

    async def get_conversation_sentiment(self, conversation_id: str) -> Dict[str, Any]:
        """Mock conversation sentiment analysis."""
        if conversation_id not in self._conversations:
            return {"error": "Conversation not found"}

        messages = self._conversations[conversation_id]
        user_messages = [msg for msg in messages if msg.role == MessageRole.USER]

        return {
            "overall_sentiment": "positive",
            "sentiment_trend": "stable",
            "user_satisfaction": random.uniform(0.7, 0.9),
            "engagement_level": "high" if len(user_messages) > 5 else "medium",
            "key_emotions": ["curiosity", "satisfaction", "engagement"],
            "conversation_quality": random.uniform(0.8, 1.0)
        }