"""
Security Director for AI Assist Board of Directors System

This module provides comprehensive security evaluation capabilities including
vulnerability scanning, OWASP Top 10 compliance checking, and security
best practices assessment.

Author: AI Assist Board of Directors System
Created: 2025-09-16
Version: 1.0.0
"""

import logging
import re
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from src.routing.base_director import DirectorBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecurityDirector(DirectorBase):
    """
    Security Director specializing in cybersecurity evaluation and vulnerability assessment.

    This director provides comprehensive security analysis including:
    - SQL injection detection
    - XSS vulnerability scanning
    - Authentication/authorization checks
    - Sensitive data exposure detection
    - Input validation assessment
    - Encryption requirements checking
    - OWASP Top 10 compliance verification
    """

    def __init__(self):
        """Initialize the Security Director with cybersecurity expertise."""
        super().__init__(
            name="SecurityDirector",
            expertise="Cybersecurity, Vulnerability Assessment, OWASP Compliance, Security Architecture",
            version="1.0.0"
        )

        # Initialize security-specific tracking
        self.vulnerability_signatures = self._load_vulnerability_signatures()
        self.severity_weights = {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.5,
            "low": 0.2
        }

        logger.info(f"SecurityDirector initialized with {len(self.knowledge_base['vulnerability_patterns'])} vulnerability patterns")

    def evaluate(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a task from a cybersecurity perspective.

        Args:
            task (Dict[str, Any]): Task information including code, description, requirements
            context (Dict[str, Any]): Additional context including user info, system state

        Returns:
            Dict[str, Any]: Comprehensive security evaluation result
        """
        try:
            # Extract code and relevant information
            code_content = task.get("code", "")
            task_type = task.get("type", "unknown")
            description = task.get("description", "")

            # Perform comprehensive security analysis
            security_findings = self._perform_security_scan(code_content, task_type, description)

            # Calculate overall security score
            security_score = self._calculate_security_score(security_findings)

            # Generate OWASP compliance assessment
            owasp_compliance = self._assess_owasp_compliance(security_findings)

            # Determine confidence based on analysis completeness
            confidence_factors = {
                "code_coverage": 0.9 if code_content else 0.3,
                "vulnerability_detection": 0.8 if security_findings else 0.5,
                "pattern_matching": 0.7,
                "context_completeness": 0.8 if context.get("requirements") else 0.4
            }
            confidence = self.calculate_confidence(confidence_factors)

            # Generate recommendations
            recommendations_dict = self._generate_security_recommendations(security_findings, task_type)
            # Convert to string format expected by registry
            recommendations = [f"{rec['description']} ({rec['implementation_notes']})" for rec in recommendations_dict]

            # Create detailed reasoning
            reasoning_factors = [
                f"Detected {len(security_findings)} security findings",
                f"Overall security score: {security_score:.2f}/10",
                f"OWASP compliance: {owasp_compliance['compliant_categories']}/{len(owasp_compliance['assessments'])} categories",
                f"Highest severity finding: {self._get_highest_severity(security_findings)}"
            ]

            reasoning = self.generate_reasoning(
                f"Security evaluation completed with {confidence:.1%} confidence",
                reasoning_factors,
                context
            )

            # Compile evaluation result
            evaluation_result = {
                "assessment": self._generate_security_assessment(security_score, security_findings),
                "confidence": confidence,
                "reasoning": reasoning,
                "recommendations": recommendations,
                "security_findings": security_findings,
                "security_score": security_score,
                "owasp_compliance": owasp_compliance,
                "risk_factors": self._identify_risk_factors(security_findings),
                "estimated_effort": self._estimate_remediation_effort(security_findings),
                "next_steps": self._suggest_next_steps(security_findings)
            }

            # Update metrics
            self.update_metrics(evaluation_result)

            logger.info(f"Security evaluation completed: {len(security_findings)} findings, score: {security_score:.2f}")

            return evaluation_result

        except Exception as e:
            logger.error(f"Error during security evaluation: {str(e)}")
            return {
                "assessment": "Security evaluation failed due to internal error",
                "confidence": 0.1,
                "reasoning": f"An error occurred during security analysis: {str(e)}",
                "recommendations": ["Retry security evaluation with valid input"],
                "security_findings": [],
                "security_score": 0.0,
                "error": str(e)
            }

    def load_knowledge(self) -> Dict[str, List[str]]:
        """
        Load comprehensive cybersecurity knowledge base.

        Returns:
            Dict[str, List[str]]: Security-focused knowledge base
        """
        return {
            "security_best_practices": [
                "Implement input validation and sanitization for all user inputs",
                "Use parameterized queries to prevent SQL injection attacks",
                "Encode output data to prevent Cross-Site Scripting (XSS)",
                "Implement proper authentication and session management",
                "Use HTTPS for all data transmission",
                "Store passwords using strong hashing algorithms (bcrypt, Argon2)",
                "Implement proper access controls and authorization checks",
                "Sanitize file uploads and restrict file types",
                "Use Content Security Policy (CSP) headers",
                "Implement rate limiting and DDoS protection",
                "Log security events and monitor for suspicious activity",
                "Keep all dependencies and frameworks updated",
                "Use secure random number generators for tokens and keys",
                "Implement proper error handling without information leakage",
                "Use secure cookie attributes (HttpOnly, Secure, SameSite)",
                "Implement Cross-Site Request Forgery (CSRF) protection",
                "Validate and sanitize all API inputs",
                "Use principle of least privilege for all access controls",
                "Implement multi-factor authentication where appropriate",
                "Encrypt sensitive data at rest and in transit",
                "Use secure communication protocols (TLS 1.2+)",
                "Implement proper key management and rotation",
                "Conduct regular security testing and code reviews",
                "Use security headers (HSTS, X-Frame-Options, etc.)",
                "Implement proper session timeout and invalidation",
                "Use secure development lifecycle (SDL) practices",
                "Implement data loss prevention measures",
                "Use intrusion detection and prevention systems",
                "Conduct regular vulnerability assessments",
                "Implement proper backup and recovery procedures",
                "Use network segmentation and firewalls",
                "Implement proper logging and audit trails",
                "Use security-focused testing methodologies",
                "Implement proper data classification and handling"
            ],
            "vulnerability_patterns": [
                "Unsanitized user input in SQL queries",
                "Direct HTML output without encoding",
                "Hardcoded credentials in source code",
                "Weak password requirements or storage",
                "Missing authentication checks on sensitive endpoints",
                "Insecure direct object references",
                "Missing authorization verification",
                "Unvalidated redirects and forwards",
                "Insecure cryptographic implementations",
                "Information disclosure through error messages",
                "Missing CSRF protection on state-changing operations",
                "Insecure file upload functionality",
                "XML External Entity (XXE) vulnerabilities",
                "Server-Side Request Forgery (SSRF) vulnerabilities",
                "Insecure deserialization of untrusted data",
                "Command injection vulnerabilities",
                "Path traversal vulnerabilities",
                "Race condition vulnerabilities",
                "Buffer overflow conditions",
                "Integer overflow vulnerabilities",
                "Use of deprecated or insecure functions",
                "Missing input length validation",
                "Insufficient session management",
                "Weak random number generation"
            ],
            "security_risk_factors": [
                "Public-facing web applications",
                "Handling of sensitive personal data",
                "Financial transaction processing",
                "User authentication and authorization systems",
                "File upload and processing functionality",
                "Database operations with user input",
                "Third-party API integrations",
                "Mobile application backends",
                "Cloud infrastructure deployments",
                "Legacy system integrations",
                "High-privilege user operations",
                "Cross-domain data sharing",
                "Real-time communication systems",
                "Payment processing systems",
                "Healthcare data processing (HIPAA compliance required)"
            ],
            "security_hardening_strategies": [
                "Implement defense-in-depth security architecture",
                "Use Web Application Firewalls (WAF) for protection",
                "Deploy intrusion detection and prevention systems",
                "Implement comprehensive logging and monitoring",
                "Use security information and event management (SIEM)",
                "Conduct regular penetration testing and vulnerability assessments",
                "Implement automated security testing in CI/CD pipelines",
                "Use container security scanning and runtime protection",
                "Deploy endpoint detection and response (EDR) solutions",
                "Implement zero-trust network architecture",
                "Use threat intelligence feeds for proactive defense",
                "Implement security orchestration and automated response (SOAR)"
            ],
            "owasp_top_10": [
                "A01 Broken Access Control",
                "A02 Cryptographic Failures",
                "A03 Injection",
                "A04 Insecure Design",
                "A05 Security Misconfiguration",
                "A06 Vulnerable and Outdated Components",
                "A07 Identification and Authentication Failures",
                "A08 Software and Data Integrity Failures",
                "A09 Security Logging and Monitoring Failures",
                "A10 Server-Side Request Forgery (SSRF)"
            ]
        }

    def scan_sql_injection(self, code_content: str) -> List[Dict[str, Any]]:
        """
        Scan for SQL injection vulnerabilities.

        Args:
            code_content (str): Code to analyze

        Returns:
            List[Dict[str, Any]]: List of SQL injection findings
        """
        findings = []

        # SQL injection patterns
        sql_patterns = [
            (r'execute\s*\(\s*["\'].*?\+.*?["\']', "Direct string concatenation in SQL execute", "high"),
            (r'query\s*\(\s*["\'].*?\+.*?["\']', "String concatenation in SQL query", "high"),
            (r'SELECT.*?\+.*?FROM', "SQL SELECT with string concatenation", "critical"),
            (r'INSERT.*?\+.*?VALUES', "SQL INSERT with string concatenation", "critical"),
            (r'UPDATE.*?\+.*?SET', "SQL UPDATE with string concatenation", "critical"),
            (r'DELETE.*?\+.*?WHERE', "SQL DELETE with string concatenation", "critical"),
            (r'WHERE.*?\+.*?=', "SQL WHERE clause with concatenation", "high"),
            (r'["\'].*?\%s.*?["\'].*?\%', "String formatting in SQL queries", "medium"),
            (r'format\s*\(.*?SELECT.*?\)', "String format in SQL SELECT", "high"),
            (r'f["\'].*?SELECT.*?\{.*?\}.*?["\']', "F-string in SQL query", "medium")
        ]

        for pattern, description, severity in sql_patterns:
            matches = re.finditer(pattern, code_content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                findings.append({
                    "type": "sql_injection",
                    "severity": severity,
                    "description": description,
                    "pattern": pattern,
                    "match": match.group(0),
                    "line": code_content[:match.start()].count('\n') + 1,
                    "recommendation": "Use parameterized queries or prepared statements"
                })

        return findings

    def check_authentication(self, code_content: str) -> List[Dict[str, Any]]:
        """
        Check for authentication and authorization vulnerabilities.

        Args:
            code_content (str): Code to analyze

        Returns:
            List[Dict[str, Any]]: List of authentication findings
        """
        findings = []

        # Authentication patterns
        auth_patterns = [
            (r'password\s*==\s*["\'].*?["\']', "Hardcoded password comparison", "critical"),
            (r'if\s+user\s*==\s*["\']admin["\']', "Hardcoded admin check", "high"),
            (r'jwt\.decode\s*\([^,]*\)', "JWT decode without verification", "high"),
            (r'session\[.*?\]\s*=.*?without.*?validation', "Session assignment without validation", "medium"),
            (r'@app\.route.*?methods.*?POST.*?(?!.*@login_required)', "POST endpoint without auth", "medium"),
            (r'def.*?(?!.*@require_auth).*?(?!.*@login_required)', "Function without auth decorator", "low"),
            (r'password.*?=.*?request\.', "Password from request without validation", "high"),
            (r'token\s*=\s*request\.args\.get', "Token from URL parameters", "medium"),
            (r'if.*?request\.headers\.get\(["\']authorization["\']', "Manual auth header check", "low")
        ]

        for pattern, description, severity in auth_patterns:
            matches = re.finditer(pattern, code_content, re.IGNORECASE)
            for match in matches:
                findings.append({
                    "type": "authentication",
                    "severity": severity,
                    "description": description,
                    "pattern": pattern,
                    "match": match.group(0),
                    "line": code_content[:match.start()].count('\n') + 1,
                    "recommendation": "Implement proper authentication and authorization controls"
                })

        return findings

    def detect_sensitive_data(self, code_content: str) -> List[Dict[str, Any]]:
        """
        Detect sensitive data exposure vulnerabilities.

        Args:
            code_content (str): Code to analyze

        Returns:
            List[Dict[str, Any]]: List of sensitive data findings
        """
        findings = []

        # Sensitive data patterns
        sensitive_patterns = [
            (r'api[_-]?key\s*=\s*["\'][^"\']{20,}["\']', "Hardcoded API key", "critical"),
            (r'secret[_-]?key\s*=\s*["\'][^"\']{10,}["\']', "Hardcoded secret key", "critical"),
            (r'password\s*=\s*["\'][^"\']{3,}["\']', "Hardcoded password", "critical"),
            (r'private[_-]?key\s*=\s*["\'].*?["\']', "Hardcoded private key", "critical"),
            (r'aws[_-]?access[_-]?key.*?=.*?["\']AKIA[0-9A-Z]{16}["\']', "AWS Access Key", "critical"),
            (r'credit[_-]?card.*?\d{4}.*?\d{4}.*?\d{4}.*?\d{4}', "Credit card number", "critical"),
            (r'ssn.*?\d{3}-\d{2}-\d{4}', "Social Security Number", "critical"),
            (r'email.*?=.*?["\'][^"\']*@[^"\']*["\']', "Email address in code", "low"),
            (r'phone.*?\d{3}.*?\d{3}.*?\d{4}', "Phone number", "low"),
            (r'token\s*=\s*["\'][A-Za-z0-9+/]{50,}={0,2}["\']', "Hardcoded token", "high")
        ]

        for pattern, description, severity in sensitive_patterns:
            matches = re.finditer(pattern, code_content, re.IGNORECASE)
            for match in matches:
                findings.append({
                    "type": "sensitive_data",
                    "severity": severity,
                    "description": description,
                    "pattern": pattern,
                    "match": match.group(0)[:50] + "..." if len(match.group(0)) > 50 else match.group(0),
                    "line": code_content[:match.start()].count('\n') + 1,
                    "recommendation": "Store sensitive data in environment variables or secure key management"
                })

        return findings

    def verify_encryption(self, code_content: str) -> List[Dict[str, Any]]:
        """
        Verify encryption implementation and requirements.

        Args:
            code_content (str): Code to analyze

        Returns:
            List[Dict[str, Any]]: List of encryption findings
        """
        findings = []

        # Encryption patterns
        encryption_patterns = [
            (r'hashlib\.md5\(', "Weak hashing algorithm MD5", "high"),
            (r'hashlib\.sha1\(', "Weak hashing algorithm SHA1", "medium"),
            (r'DES\.|des\.|3DES', "Weak encryption algorithm DES/3DES", "high"),
            (r'RC4\.|rc4\.', "Weak encryption algorithm RC4", "high"),
            (r'ssl_context\.check_hostname\s*=\s*False', "SSL hostname verification disabled", "high"),
            (r'verify\s*=\s*False', "SSL verification disabled", "high"),
            (r'random\.random\(\)', "Weak random number generation", "medium"),
            (r'AES\.new\([^,]*,\s*AES\.MODE_ECB', "AES ECB mode usage", "medium"),
            (r'urllib\.request\.urlopen\(.*?http://', "HTTP usage instead of HTTPS", "medium"),
            (r'requests\.get\(.*?http://', "HTTP request instead of HTTPS", "low")
        ]

        for pattern, description, severity in encryption_patterns:
            matches = re.finditer(pattern, code_content, re.IGNORECASE)
            for match in matches:
                findings.append({
                    "type": "encryption",
                    "severity": severity,
                    "description": description,
                    "pattern": pattern,
                    "match": match.group(0),
                    "line": code_content[:match.start()].count('\n') + 1,
                    "recommendation": "Use strong encryption algorithms and proper implementation"
                })

        return findings

    def assess_input_validation(self, code_content: str) -> List[Dict[str, Any]]:
        """
        Assess input validation implementation.

        Args:
            code_content (str): Code to analyze

        Returns:
            List[Dict[str, Any]]: List of input validation findings
        """
        findings = []

        # Input validation patterns
        validation_patterns = [
            (r'request\.args\.get\([^)]*\)(?!.*(?:int\(|float\(|validate|sanitize))', "Unvalidated request parameter", "medium"),
            (r'request\.form\.get\([^)]*\)(?!.*(?:int\(|float\(|validate|sanitize))', "Unvalidated form input", "medium"),
            (r'request\.json\.get\([^)]*\)(?!.*(?:validate|sanitize))', "Unvalidated JSON input", "medium"),
            (r'eval\s*\(', "Use of eval() function", "critical"),
            (r'exec\s*\(', "Use of exec() function", "critical"),
            (r'os\.system\s*\(', "Direct OS command execution", "critical"),
            (r'subprocess\..*?shell\s*=\s*True', "Shell command execution", "high"),
            (r'open\s*\([^)]*\+.*?\)', "File path concatenation", "medium"),
            (r'__import__\s*\(', "Dynamic import usage", "medium"),
            (r'pickle\.loads\s*\(', "Unsafe deserialization", "high")
        ]

        for pattern, description, severity in validation_patterns:
            matches = re.finditer(pattern, code_content, re.IGNORECASE)
            for match in matches:
                findings.append({
                    "type": "input_validation",
                    "severity": severity,
                    "description": description,
                    "pattern": pattern,
                    "match": match.group(0),
                    "line": code_content[:match.start()].count('\n') + 1,
                    "recommendation": "Implement proper input validation and sanitization"
                })

        return findings

    def _perform_security_scan(self, code_content: str, task_type: str, description: str) -> List[Dict[str, Any]]:
        """
        Perform comprehensive security scan combining all security checks.

        Args:
            code_content (str): Code to analyze
            task_type (str): Type of task
            description (str): Task description

        Returns:
            List[Dict[str, Any]]: Combined security findings
        """
        all_findings = []

        if code_content:
            # Run all security scans
            all_findings.extend(self.scan_sql_injection(code_content))
            all_findings.extend(self.check_authentication(code_content))
            all_findings.extend(self.detect_sensitive_data(code_content))
            all_findings.extend(self.verify_encryption(code_content))
            all_findings.extend(self.assess_input_validation(code_content))

            # Add XSS scanning
            all_findings.extend(self._scan_xss_vulnerabilities(code_content))

        # Add contextual findings based on task type and description
        all_findings.extend(self._analyze_contextual_risks(task_type, description))

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        all_findings.sort(key=lambda x: severity_order.get(x.get("severity", "low"), 3))

        return all_findings

    def _scan_xss_vulnerabilities(self, code_content: str) -> List[Dict[str, Any]]:
        """Scan for XSS vulnerabilities."""
        findings = []

        xss_patterns = [
            (r'innerHTML\s*=.*?\+', "Dynamic innerHTML assignment", "high"),
            (r'innerHTML\s*=.*?["\'].*?\+.*?["\']', "Dynamic innerHTML with concatenation", "high"),
            (r'document\.write\s*\(', "Use of document.write", "medium"),
            (r'eval\s*\(.*?user.*?\)', "eval() with user input", "critical"),
            (r'v-html\s*=.*?\{\{.*?\}\}', "Vue v-html with dynamic content", "high"),
            (r'dangerouslySetInnerHTML', "React dangerouslySetInnerHTML usage", "high"),
            (r'<%=.*?%>', "Unescaped template output", "medium"),
            (r'render_template_string\s*\(', "Template string rendering", "medium"),
            (r'res\.send\s*\(["\'].*?\+.*?req\..*?\+.*?["\']', "Direct output of user input", "high"),
            (r'\.innerHTML\s*=\s*.*?\.value', "innerHTML assignment from user input", "high")
        ]

        for pattern, description, severity in xss_patterns:
            matches = re.finditer(pattern, code_content, re.IGNORECASE)
            for match in matches:
                findings.append({
                    "type": "xss",
                    "severity": severity,
                    "description": description,
                    "pattern": pattern,
                    "match": match.group(0),
                    "line": code_content[:match.start()].count('\n') + 1,
                    "recommendation": "Escape or sanitize output, use safe templating methods"
                })

        return findings

    def _analyze_contextual_risks(self, task_type: str, description: str) -> List[Dict[str, Any]]:
        """Analyze risks based on task context."""
        findings = []

        high_risk_keywords = {
            "payment": "critical",
            "financial": "high",
            "authentication": "high",
            "login": "high",
            "user data": "high",
            "database": "medium",
            "api": "medium",
            "upload": "medium"
        }

        desc_lower = description.lower()
        for keyword, severity in high_risk_keywords.items():
            if keyword in desc_lower:
                findings.append({
                    "type": "contextual_risk",
                    "severity": severity,
                    "description": f"High-risk functionality detected: {keyword}",
                    "recommendation": f"Apply enhanced security measures for {keyword} functionality",
                    "context": keyword
                })

        return findings

    def _calculate_security_score(self, findings: List[Dict[str, Any]]) -> float:
        """Calculate overall security score (0-10 scale)."""
        if not findings:
            return 8.5  # Good baseline if no issues found

        # Weight findings by severity
        penalty = 0
        for finding in findings:
            severity = finding.get("severity", "low")
            penalty += {
                "critical": 3.0,
                "high": 2.0,
                "medium": 1.0,
                "low": 0.3
            }.get(severity, 0.3)

        # Calculate score
        score = max(0.0, 10.0 - penalty)
        return round(score, 2)

    def _assess_owasp_compliance(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess OWASP Top 10 compliance."""
        owasp_categories = self.knowledge_base["owasp_top_10"]
        assessments = {}

        # Map finding types to OWASP categories
        finding_to_owasp = {
            "sql_injection": "A03 Injection",
            "xss": "A03 Injection",
            "authentication": "A07 Identification and Authentication Failures",
            "sensitive_data": "A02 Cryptographic Failures",
            "encryption": "A02 Cryptographic Failures",
            "input_validation": "A03 Injection"
        }

        # Initialize all categories as compliant
        for category in owasp_categories:
            assessments[category] = {"compliant": True, "findings": []}

        # Mark categories as non-compliant based on findings
        for finding in findings:
            finding_type = finding.get("type", "")
            owasp_category = finding_to_owasp.get(finding_type)
            if owasp_category and owasp_category in assessments:
                assessments[owasp_category]["compliant"] = False
                assessments[owasp_category]["findings"].append(finding)

        compliant_count = sum(1 for assessment in assessments.values() if assessment["compliant"])

        return {
            "assessments": assessments,
            "compliant_categories": compliant_count,
            "total_categories": len(owasp_categories),
            "compliance_percentage": (compliant_count / len(owasp_categories)) * 100
        }

    def _generate_security_recommendations(self, findings: List[Dict[str, Any]], task_type: str) -> List[Dict[str, Any]]:
        """Generate specific security recommendations."""
        recommendations = []

        # Group findings by type
        findings_by_type = {}
        for finding in findings:
            finding_type = finding.get("type", "unknown")
            if finding_type not in findings_by_type:
                findings_by_type[finding_type] = []
            findings_by_type[finding_type].append(finding)

        # Generate recommendations for each type
        for finding_type, type_findings in findings_by_type.items():
            severity = max(type_findings, key=lambda x: self.severity_weights.get(x.get("severity", "low"), 0))["severity"]

            recommendations.append({
                "category": f"security_{finding_type}",
                "priority": "critical" if severity in ["critical", "high"] else "medium",
                "description": f"Address {len(type_findings)} {finding_type.replace('_', ' ')} vulnerabilities",
                "implementation_notes": self._get_implementation_notes(finding_type),
                "expected_impact": "Reduced security risk and improved compliance"
            })

        # Add general security recommendations
        if findings:
            recommendations.extend([
                {
                    "category": "security_testing",
                    "priority": "high",
                    "description": "Implement automated security testing",
                    "implementation_notes": "Add SAST/DAST tools to CI/CD pipeline",
                    "expected_impact": "Early detection of security vulnerabilities"
                },
                {
                    "category": "security_review",
                    "priority": "medium",
                    "description": "Conduct security code review",
                    "implementation_notes": "Have security expert review implementation",
                    "expected_impact": "Identification of additional security issues"
                }
            ])

        return recommendations

    def _get_implementation_notes(self, finding_type: str) -> str:
        """Get specific implementation notes for finding types."""
        implementation_guides = {
            "sql_injection": "Replace string concatenation with parameterized queries or prepared statements",
            "xss": "Implement output encoding and Content Security Policy (CSP)",
            "authentication": "Use established authentication frameworks and follow security best practices",
            "sensitive_data": "Move sensitive data to environment variables or secure key management systems",
            "encryption": "Replace weak algorithms with AES-256, SHA-256+, and proper SSL/TLS configuration",
            "input_validation": "Implement input sanitization, validation, and use allow-lists where possible"
        }

        return implementation_guides.get(finding_type, "Consult security best practices and OWASP guidelines")

    def _identify_risk_factors(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Identify overall risk factors based on findings."""
        risk_factors = []

        if findings:
            severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            for finding in findings:
                severity = finding.get("severity", "low")
                severity_counts[severity] += 1

            if severity_counts["critical"] > 0:
                risk_factors.append(f"{severity_counts['critical']} critical security vulnerabilities detected")

            if severity_counts["high"] > 2:
                risk_factors.append(f"Multiple high-severity vulnerabilities ({severity_counts['high']} found)")

            finding_types = set(f.get("type", "") for f in findings)
            if len(finding_types) > 3:
                risk_factors.append(f"Multiple vulnerability types present ({len(finding_types)} different types)")

        return risk_factors or ["No significant security risk factors identified"]

    def _estimate_remediation_effort(self, findings: List[Dict[str, Any]]) -> str:
        """Estimate effort required to remediate security findings."""
        if not findings:
            return "Minimal - maintain current security posture"

        effort_points = sum(
            {"critical": 8, "high": 5, "medium": 3, "low": 1}.get(
                finding.get("severity", "low"), 1
            ) for finding in findings
        )

        if effort_points >= 20:
            return "High - significant security remediation required (2-4 weeks)"
        elif effort_points >= 10:
            return "Medium - moderate security improvements needed (1-2 weeks)"
        else:
            return "Low - minor security enhancements required (1-3 days)"

    def _suggest_next_steps(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Suggest immediate next steps based on findings."""
        next_steps = []

        if not findings:
            next_steps.extend([
                "Conduct comprehensive penetration testing",
                "Implement security monitoring and logging",
                "Regular security training for development team"
            ])
        else:
            # Priority based on critical/high findings
            critical_high = [f for f in findings if f.get("severity") in ["critical", "high"]]

            if critical_high:
                next_steps.extend([
                    f"URGENT: Address {len(critical_high)} critical/high severity vulnerabilities immediately",
                    "Temporarily restrict access to affected systems if possible",
                    "Conduct emergency security review with development team"
                ])
            else:
                next_steps.extend([
                    "Prioritize remediation of identified security issues",
                    "Implement additional security testing in development workflow",
                    "Schedule follow-up security assessment after fixes"
                ])

        return next_steps

    def _generate_security_assessment(self, security_score: float, findings: List[Dict[str, Any]]) -> str:
        """Generate overall security assessment summary."""
        if security_score >= 9.0:
            return f"Excellent security posture with score {security_score}/10. Minimal vulnerabilities detected."
        elif security_score >= 7.0:
            return f"Good security posture with score {security_score}/10. Some improvements recommended."
        elif security_score >= 5.0:
            return f"Moderate security posture with score {security_score}/10. Several vulnerabilities need attention."
        elif security_score >= 3.0:
            return f"Poor security posture with score {security_score}/10. Significant vulnerabilities detected."
        else:
            return f"Critical security issues detected with score {security_score}/10. Immediate action required."

    def _get_highest_severity(self, findings: List[Dict[str, Any]]) -> str:
        """Get the highest severity level from findings."""
        if not findings:
            return "none"

        severity_priority = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        highest = min(findings, key=lambda x: severity_priority.get(x.get("severity", "low"), 3))
        return highest.get("severity", "low")

    def _load_vulnerability_signatures(self) -> Dict[str, List[str]]:
        """Load vulnerability detection signatures."""
        return {
            "sql_injection_signatures": [
                "union select", "drop table", "exec xp_", "sp_executesql",
                "insert into", "delete from", "update set", "' or '1'='1",
                "' or 1=1--", "'; drop table", "' union all select"
            ],
            "xss_signatures": [
                "<script", "javascript:", "onerror=", "onload=",
                "eval(", "setTimeout(", "setInterval(", "document.write"
            ],
            "command_injection_signatures": [
                "system(", "exec(", "shell_exec(", "passthru(", "eval(",
                "os.system", "subprocess.call", "os.popen", "commands."
            ]
        }

    def get_security_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive security summary for this director.

        Returns:
            Dict[str, Any]: Security-focused performance summary
        """
        base_summary = self.get_performance_summary()

        # Add security-specific metrics
        security_metrics = {
            "vulnerability_patterns_loaded": len(self.knowledge_base.get("vulnerability_patterns", [])),
            "security_best_practices": len(self.knowledge_base.get("security_best_practices", [])),
            "owasp_categories_covered": len(self.knowledge_base.get("owasp_top_10", [])),
            "signature_patterns": sum(len(sigs) for sigs in self.vulnerability_signatures.values())
        }

        base_summary.update(security_metrics)
        return base_summary