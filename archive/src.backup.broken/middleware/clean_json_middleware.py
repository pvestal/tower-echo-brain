#!/usr/bin/env python3
"""
Clean JSON Middleware for Echo Brain
Strips narrative wrappers from LLM responses to return pure JSON
"""

import json
import re
import logging
from typing import Callable
from fastapi import Request
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

class CleanJSONMiddleware(BaseHTTPMiddleware):
    """Middleware to clean narrative wrappers from LLM JSON responses"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process responses to strip narrative wrappers"""

        # Only process JSON responses from echo endpoints
        if not request.url.path.startswith("/api/echo/"):
            return await call_next(request)

        # Call the endpoint
        response = await call_next(request)

        # Only process successful JSON responses
        if response.status_code != 200:
            return response

        # Check content type
        content_type = response.headers.get('content-type', '')
        if 'application/json' not in content_type:
            return response

        try:
            # Read the response body
            body = b""
            async for chunk in response.body_iterator:
                body += chunk

            # Parse the JSON
            data = json.loads(body.decode('utf-8'))

            # Clean the response field if it exists
            if isinstance(data, dict) and 'response' in data:
                original = data['response']
                cleaned = self.clean_llm_response(original)

                if cleaned != original:
                    logger.info(f"🧹 Cleaned response - removed wrapper from: {original[:50]}...")
                    data['response'] = cleaned

                    # Create new response with cleaned data
                    return Response(
                        content=json.dumps(data),
                        media_type="application/json",
                        status_code=response.status_code
                    )

            # Return original response if no cleaning needed
            return Response(
                content=body,
                media_type="application/json",
                status_code=response.status_code
            )

        except Exception as e:
            logger.debug(f"Middleware processing skipped: {e}")
            # If anything goes wrong, return the original response
            return response

    def clean_llm_response(self, response_text: str) -> str:
        """
        Clean narrative wrappers from LLM responses
        """
        if not response_text:
            return response_text

        original = response_text

        # Pattern 1: Markdown code blocks with JSON
        # Matches ```json ... ``` or ``` ... ```
        code_block_pattern = r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```'
        match = re.search(code_block_pattern, response_text, re.DOTALL)
        if match:
            try:
                json_str = match.group(1).strip()
                # Validate it's actual JSON
                json.loads(json_str)
                logger.debug("Extracted JSON from markdown code block")
                return json_str
            except:
                pass

        # Pattern 2: Look for clean JSON objects or arrays
        # Try to find JSON that starts with { or [
        json_patterns = [
            r'^[\s]*(\{(?:[^{}]|(?:\{[^{}]*\}))*\})[\s]*$',  # JSON object
            r'^[\s]*(\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\])[\s]*$',  # JSON array
        ]

        for pattern in json_patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                try:
                    json_str = match.group(1)
                    json.loads(json_str)
                    return json_str
                except:
                    pass

        # Pattern 3: Extract JSON from narrative
        # Look for JSON at the end after narrative text
        narrative_end_pattern = r'.*?(\{[^}]+\}|\[[^\]]+\])[\s]*$'
        match = re.search(narrative_end_pattern, response_text, re.DOTALL)
        if match:
            try:
                json_str = match.group(1)
                json.loads(json_str)

                # Only clean if there was actual narrative before it
                prefix = response_text[:match.start(1)]
                if prefix.strip():
                    logger.debug(f"Removed narrative: {prefix[:30]}...")
                    return json_str
            except:
                pass

        # Pattern 4: Try to parse the entire response as JSON
        try:
            cleaned = response_text.strip()
            json.loads(cleaned)
            return cleaned
        except:
            pass

        # No cleaning possible
        return original