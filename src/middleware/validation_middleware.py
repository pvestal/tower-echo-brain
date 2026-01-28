#!/usr/bin/env python3
"""
Request Validation and Sanitization Middleware for Echo Brain
Provides comprehensive input validation, sanitization, and security checks
"""

import re
import html
import logging
import ipaddress
from typing import Dict, Any, Optional, List, Union
from urllib.parse import unquote
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import json

logger = logging.getLogger(__name__)

# Security patterns
SUSPICIOUS_PATTERNS = [
    r'<script[^>]*>.*?</script>',  # Script tags
    r'javascript:',  # JavaScript URLs
    r'vbscript:',  # VBScript URLs
    r'onload\s*=',  # Event handlers
    r'onclick\s*=',
    r'onerror\s*=',
    r'eval\s*\(',  # Code execution
    r'exec\s*\(',
    r'system\s*\(',
    r'\.\./',  # Directory traversal
    r'\.\.\\',
    r'union\s+select',  # SQL injection
    r'drop\s+table',
    r'delete\s+from',
    r'insert\s+into',
    r'update\s+set',
    r'<iframe[^>]*>',  # Iframe injection
    r'<object[^>]*>',  # Object injection
    r'<embed[^>]*>',   # Embed injection
]

# Compiled regex patterns for performance
COMPILED_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in SUSPICIOUS_PATTERNS]

# File extension blacklist
DANGEROUS_EXTENSIONS = [
    '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js',
    '.jar', '.sh', '.ps1', '.php', '.asp', '.jsp', '.py', '.rb'
]

# Maximum sizes
MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
MAX_HEADER_SIZE = 8192  # 8KB
MAX_URL_LENGTH = 2048
MAX_JSON_DEPTH = 10

class SecurityException(Exception):
    """Security-related exception"""
    pass

class RequestValidator:
    """Validates and sanitizes incoming requests"""

    def __init__(self):
        self.blocked_ips = set()
        self.suspicious_requests = {}

    def validate_ip_address(self, ip: str) -> bool:
        """Validate IP address format and check against blocklist"""
        try:
            ip_obj = ipaddress.ip_address(ip)

            # Check if IP is in private ranges (allow internal traffic)
            if ip_obj.is_private or ip_obj.is_loopback:
                return True

            # Check against blocklist
            if ip in self.blocked_ips:
                return False

            return True
        except ValueError:
            return False

    def sanitize_string(self, value: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        if not isinstance(value, str):
            return str(value)[:max_length]

        # URL decode
        value = unquote(value)

        # HTML escape
        value = html.escape(value)

        # Remove null bytes
        value = value.replace('\x00', '')

        # Limit length
        value = value[:max_length]

        return value

    def check_for_malicious_patterns(self, text: str) -> List[str]:
        """Check text for malicious patterns"""
        found_patterns = []

        for i, pattern in enumerate(COMPILED_PATTERNS):
            if pattern.search(text):
                found_patterns.append(SUSPICIOUS_PATTERNS[i])

        return found_patterns

    def validate_file_upload(self, filename: str, content_type: str) -> bool:
        """Validate file uploads"""
        if not filename:
            return False

        # Check file extension
        for ext in DANGEROUS_EXTENSIONS:
            if filename.lower().endswith(ext):
                return False

        # Check content type
        safe_content_types = [
            'image/jpeg', 'image/png', 'image/gif', 'image/webp',
            'text/plain', 'text/csv', 'application/json',
            'application/pdf'
        ]

        if content_type not in safe_content_types:
            return False

        return True

    def validate_json_structure(self, data: Any, depth: int = 0) -> bool:
        """Validate JSON structure and depth"""
        if depth > MAX_JSON_DEPTH:
            raise SecurityException("JSON depth limit exceeded")

        if isinstance(data, dict):
            for key, value in data.items():
                if not isinstance(key, str) or len(key) > 100:
                    return False
                if not self.validate_json_structure(value, depth + 1):
                    return False
        elif isinstance(data, list):
            for item in data:
                if not self.validate_json_structure(item, depth + 1):
                    return False
        elif isinstance(data, str):
            if len(data) > 10000:  # 10KB max for string values
                return False
            # Check for malicious patterns in string values
            malicious_patterns = self.check_for_malicious_patterns(data)
            if malicious_patterns:
                logger.warning(f"Malicious patterns detected in JSON: {malicious_patterns}")
                return False

        return True

    async def validate_request(self, request: Request) -> Dict[str, Any]:
        """Main request validation function"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        try:
            # Get client IP
            client_ip = request.headers.get("X-Real-IP") or \
                       request.headers.get("X-Forwarded-For", "").split(",")[0] or \
                       getattr(request.client, 'host', 'unknown')

            # Validate IP
            if not self.validate_ip_address(client_ip):
                validation_result["valid"] = False
                validation_result["errors"].append(f"Invalid or blocked IP: {client_ip}")
                return validation_result

            # Check request size
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > MAX_REQUEST_SIZE:
                validation_result["valid"] = False
                validation_result["errors"].append("Request size too large")
                return validation_result

            # Validate URL
            url_str = str(request.url)
            if len(url_str) > MAX_URL_LENGTH:
                validation_result["valid"] = False
                validation_result["errors"].append("URL too long")
                return validation_result

            # Check URL for malicious patterns
            malicious_patterns = self.check_for_malicious_patterns(url_str)
            if malicious_patterns:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Malicious patterns in URL: {malicious_patterns}")
                return validation_result

            # Validate headers
            for header_name, header_value in request.headers.items():
                if len(header_value) > MAX_HEADER_SIZE:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Header too large: {header_name}")
                    return validation_result

                # Check headers for malicious patterns
                malicious_patterns = self.check_for_malicious_patterns(header_value)
                if malicious_patterns:
                    validation_result["warnings"].append(f"Suspicious header {header_name}: {malicious_patterns}")

            # Validate query parameters
            for param_name, param_value in request.query_params.items():
                sanitized_value = self.sanitize_string(param_value)
                if sanitized_value != param_value:
                    validation_result["warnings"].append(f"Query parameter sanitized: {param_name}")

                malicious_patterns = self.check_for_malicious_patterns(param_value)
                if malicious_patterns:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Malicious patterns in query param {param_name}: {malicious_patterns}")

            # Validate JSON body if present
            if request.headers.get("content-type", "").startswith("application/json"):
                try:
                    body = await request.body()
                    if body:
                        json_data = json.loads(body)
                        if not self.validate_json_structure(json_data):
                            validation_result["valid"] = False
                            validation_result["errors"].append("Invalid JSON structure")
                except json.JSONDecodeError:
                    validation_result["valid"] = False
                    validation_result["errors"].append("Invalid JSON format")
                except SecurityException as e:
                    validation_result["valid"] = False
                    validation_result["errors"].append(str(e))

            # Log validation results
            if not validation_result["valid"]:
                logger.warning(f"Request validation failed for {client_ip}: {validation_result['errors']}")
            elif validation_result["warnings"]:
                logger.info(f"Request warnings for {client_ip}: {validation_result['warnings']}")

        except Exception as e:
            logger.error(f"Request validation error: {e}")
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation error: {str(e)}")

        return validation_result

# Global validator instance
request_validator = RequestValidator()

async def validate_request_middleware(request: Request, call_next):
    """Middleware function for request validation"""

    # Skip validation for specific endpoints (like health checks)
    skip_validation_paths = ["/health", "/metrics"]
    if any(request.url.path.startswith(path) for path in skip_validation_paths):
        return await call_next(request)

    # Validate request
    validation_result = await request_validator.validate_request(request)

    if not validation_result["valid"]:
        logger.error(f"Request blocked: {validation_result['errors']}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Request validation failed",
                "details": validation_result["errors"][:3],  # Limit error details
                "timestamp": "2026-01-28T00:00:00.000Z"
            }
        )

    # Add validation warnings to request state
    if validation_result["warnings"]:
        request.state.validation_warnings = validation_result["warnings"]

    response = await call_next(request)

    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"

    return response

def sanitize_response_data(data: Any) -> Any:
    """Sanitize response data before sending"""
    if isinstance(data, dict):
        return {key: sanitize_response_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [sanitize_response_data(item) for item in data]
    elif isinstance(data, str):
        return request_validator.sanitize_string(data)
    else:
        return data