"""
Performance Director for Echo Brain Board of Directors System

This module provides comprehensive performance evaluation capabilities including
time complexity analysis, space complexity assessment, database query optimization,
caching strategies evaluation, and resource usage monitoring.

Author: Echo Brain Board of Directors System
Created: 2025-09-16
Version: 1.0.0
"""

import logging
import re
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from .base_director import DirectorBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceDirector(DirectorBase):
    """
    Performance Director specializing in performance optimization and efficiency evaluation.

    This director provides comprehensive performance analysis including:
    - Time complexity analysis (Big O notation)
    - Space complexity assessment
    - Database query optimization
    - Caching strategy evaluation
    - Resource usage monitoring
    - Bottleneck detection
    - Memory leak identification
    - CPU utilization analysis
    """

    def __init__(self):
        """Initialize the Performance Director with optimization expertise."""
        super().__init__(
            name="PerformanceDirector",
            expertise="Performance Optimization, Algorithm Analysis, Database Optimization, Caching Strategies",
            version="1.0.0"
        )

        # Initialize performance-specific tracking
        self.complexity_patterns = self._load_complexity_patterns()
        self.performance_weights = {
            "critical": 1.0,    # O(n!) or worse
            "high": 0.8,        # O(2^n) or O(n^3)
            "medium": 0.5,      # O(n^2) or O(n log n)
            "low": 0.2          # O(n) or O(log n)
        }

        logger.info(f"PerformanceDirector initialized with {len(self.knowledge_base.get('performance_patterns', []))} performance patterns")

    def evaluate(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a task from a performance optimization perspective.

        Args:
            task (Dict[str, Any]): Task information including code, description, requirements
            context (Dict[str, Any]): Additional context including user info, system state

        Returns:
            Dict[str, Any]: Comprehensive performance evaluation result
        """
        try:
            # Extract code and relevant information
            code_content = task.get("code", "")
            task_type = task.get("type", "unknown")
            description = task.get("description", "")

            # Perform comprehensive performance analysis
            performance_findings = self._perform_performance_analysis(code_content, task_type, description)

            # Calculate overall performance score
            performance_score = self._calculate_performance_score(performance_findings)

            # Generate optimization assessment
            optimization_opportunities = self._assess_optimization_opportunities(performance_findings)

            # Determine confidence based on analysis completeness
            confidence_factors = {
                "code_coverage": 0.9 if code_content else 0.3,
                "complexity_analysis": 0.8 if performance_findings.get("complexity_issues") else 0.5,
                "pattern_matching": 0.7,
                "context_completeness": 0.8 if context.get("requirements") else 0.4
            }
            confidence = self.calculate_confidence(confidence_factors)

            # Generate recommendations
            recommendations_dict = self._generate_performance_recommendations(performance_findings, task_type)
            # Convert to string format expected by registry
            recommendations = [f"{rec['description']} ({rec['implementation']})" for rec in recommendations_dict]

            # Create detailed reasoning
            reasoning_factors = [
                f"Detected {len(performance_findings.get('issues', []))} performance issues",
                f"Overall performance score: {performance_score:.2f}/10",
                f"Time complexity: {performance_findings.get('time_complexity', 'Unknown')}",
                f"Space complexity: {performance_findings.get('space_complexity', 'Unknown')}",
                f"Optimization opportunities: {len(optimization_opportunities)}"
            ]

            reasoning = self.generate_reasoning(
                assessment=f"Performance evaluation completed with {len(performance_findings.get('issues', []))} issues identified",
                factors=reasoning_factors + [
                    f"Database queries: {performance_findings.get('db_query_count', 0)}",
                    f"Caching coverage: {performance_findings.get('caching_score', 0):.1f}%",
                    f"Resource efficiency: {performance_findings.get('resource_efficiency', 'Unknown')}",
                    f"Performance risk level: {self._calculate_risk_level(performance_findings)}"
                ],
                context=context
            )

            # Record evaluation
            evaluation_result = {
                "director": self.name,
                "task_type": task_type,
                "timestamp": datetime.now().isoformat(),
                "performance_score": performance_score,
                "confidence": confidence,
                "findings": performance_findings,
                "recommendations": recommendations,
                "optimization_opportunities": optimization_opportunities,
                "reasoning": reasoning,
                "metadata": {
                    "evaluation_duration": "computed",
                    "patterns_checked": len(self.knowledge_base["performance_patterns"]),
                    "optimization_strategies_available": len(self.knowledge_base["optimization_strategies"])
                }
            }

            self.evaluation_history.append(evaluation_result)
            logger.info(f"Performance evaluation completed with score: {performance_score:.2f}")

            return evaluation_result

        except Exception as e:
            logger.error(f"Error during performance evaluation: {str(e)}")
            return self._create_error_response(e, task_type)

    def analyze_time_complexity(self, code: str) -> Dict[str, Any]:
        """
        Analyze time complexity of given code.

        Args:
            code (str): Code to analyze

        Returns:
            Dict[str, Any]: Time complexity analysis results
        """
        complexity_indicators = {
            "O(1)": [r"return\s+\w+", r"^\s*\w+\s*=", r"print\("],
            "O(log n)": [r"while.*//=.*2", r"while.*\*=.*2", r"binary.*search"],
            "O(n)": [r"for.*in.*range\(.*\):", r"while.*<.*len\(", r"\.count\(", r"\.index\("],
            "O(n log n)": [r"\.sort\(\)", r"sorted\(", r"merge.*sort", r"quick.*sort"],
            "O(n^2)": [r"for.*in.*:\s*for.*in.*:", r"nested.*loop"],
            "O(2^n)": [r"fibonacci.*recursive", r"def.*\(.*n.*\).*return.*\(.*n-1.*\).*\+.*\(.*n-2.*\)"],
            "O(n!)": [r"permutation", r"factorial.*recursive"]
        }

        detected_complexity = "O(1)"  # Default assumption
        complexity_score = 10.0  # Best case

        for complexity, patterns in complexity_indicators.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                    detected_complexity = complexity
                    # Assign scores based on complexity
                    if complexity == "O(1)":
                        complexity_score = 10.0
                    elif complexity == "O(log n)":
                        complexity_score = 9.0
                    elif complexity == "O(n)":
                        complexity_score = 8.0
                    elif complexity == "O(n log n)":
                        complexity_score = 6.0
                    elif complexity == "O(n^2)":
                        complexity_score = 4.0
                    elif complexity == "O(2^n)":
                        complexity_score = 2.0
                    elif complexity == "O(n!)":
                        complexity_score = 1.0
                    break

        return {
            "complexity": detected_complexity,
            "score": complexity_score,
            "analysis": f"Detected time complexity: {detected_complexity}",
            "improvement_potential": complexity_score < 8.0
        }

    def check_memory_usage(self, code: str) -> Dict[str, Any]:
        """
        Check memory usage patterns in code.

        Args:
            code (str): Code to analyze

        Returns:
            Dict[str, Any]: Memory usage analysis results
        """
        memory_issues = []
        memory_score = 10.0

        # Check for common memory issues
        memory_patterns = {
            "large_list_creation": [r"range\(\d{4,}\)", r"list\(.*range\(\d{4,}\)"],
            "string_concatenation": [r"\+\s*=\s*.*str", r".*\.join\(.*\)"],
            "recursive_without_memoization": [r"def.*\(.*n.*\).*return.*\(.*n-1.*\)"],
            "global_variables": [r"^global\s+\w+", r"globals\(\)"],
            "memory_leaks": [r"while\s+True:", r"infinite.*loop"]
        }

        for issue_type, patterns in memory_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code, re.MULTILINE):
                    memory_issues.append({
                        "type": issue_type,
                        "severity": "medium" if issue_type != "memory_leaks" else "high",
                        "description": f"Potential {issue_type.replace('_', ' ')} detected"
                    })
                    memory_score -= 1.5 if issue_type != "memory_leaks" else 3.0

        return {
            "memory_score": max(0, memory_score),
            "issues": memory_issues,
            "space_complexity": self._estimate_space_complexity(code),
            "recommendations": self._get_memory_recommendations(memory_issues)
        }

    def optimize_queries(self, code: str) -> Dict[str, Any]:
        """
        Analyze and suggest database query optimizations.

        Args:
            code (str): Code containing database queries

        Returns:
            Dict[str, Any]: Query optimization analysis
        """
        query_issues = []
        optimization_score = 10.0

        # Database query patterns to check
        query_patterns = {
            "n_plus_one": [r"for.*in.*:.*\.get\(", r"for.*in.*:.*\.filter\("],
            "missing_indexes": [r"\.filter\(.*=.*\)", r"WHERE.*="],
            "select_all": [r"SELECT \*", r"\.all\(\)"],
            "no_pagination": [r"\.all\(\)", r"SELECT.*without.*LIMIT"],
            "inefficient_joins": [r"JOIN.*JOIN.*JOIN", r"\.select_related\(.*,.*,"],
            "subquery_instead_of_join": [r"SELECT.*\(SELECT", r"\.filter\(.*__in=.*\.filter"]
        }

        for issue_type, patterns in query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                    query_issues.append({
                        "type": issue_type,
                        "severity": "high" if issue_type in ["n_plus_one", "missing_indexes"] else "medium",
                        "description": f"Potential {issue_type.replace('_', ' ')} problem detected"
                    })
                    optimization_score -= 2.0 if issue_type in ["n_plus_one", "missing_indexes"] else 1.0

        return {
            "optimization_score": max(0, optimization_score),
            "issues": query_issues,
            "recommendations": self._get_query_recommendations(query_issues)
        }

    def assess_caching(self, code: str) -> Dict[str, Any]:
        """
        Assess caching strategies and opportunities.

        Args:
            code (str): Code to analyze for caching

        Returns:
            Dict[str, Any]: Caching assessment results
        """
        caching_opportunities = []
        caching_score = 5.0  # Neutral starting point

        # Check for existing caching implementations
        caching_implementations = {
            "redis_cache": [r"redis\.", r"Redis\(", r"@cache"],
            "memory_cache": [r"@lru_cache", r"@functools\.lru_cache", r"cache\["],
            "database_cache": [r"cache_key", r"\.cache\(", r"cached_result"],
            "cdn_cache": [r"Cache-Control", r"ETag", r"cloudflare"]
        }

        # Check for caching opportunities
        cacheable_patterns = {
            "expensive_computation": [r"def.*calculate", r"def.*compute", r"for.*in.*range\(\d{3,}\)"],
            "database_queries": [r"\.get\(", r"\.filter\(", r"SELECT"],
            "api_calls": [r"requests\.", r"urllib", r"http"],
            "file_operations": [r"open\(", r"\.read\(", r"\.load\("]
        }

        # Check existing implementations
        for cache_type, patterns in caching_implementations.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    caching_score += 2.0
                    break

        # Check for opportunities
        for opportunity_type, patterns in cacheable_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    caching_opportunities.append({
                        "type": opportunity_type,
                        "priority": "high" if opportunity_type in ["database_queries", "api_calls"] else "medium",
                        "description": f"Consider caching for {opportunity_type.replace('_', ' ')}"
                    })
                    break

        return {
            "caching_score": min(10.0, caching_score),
            "opportunities": caching_opportunities,
            "recommendations": self._get_caching_recommendations(caching_opportunities)
        }

    def detect_bottlenecks(self, code: str) -> Dict[str, Any]:
        """
        Detect potential performance bottlenecks.

        Args:
            code (str): Code to analyze

        Returns:
            Dict[str, Any]: Bottleneck detection results
        """
        bottlenecks = []
        bottleneck_score = 10.0

        # Bottleneck patterns
        bottleneck_patterns = {
            "blocking_io": [r"time\.sleep\(", r"\.join\(\)", r"input\("],
            "synchronous_calls": [r"requests\.get", r"requests\.post", r"urllib"],
            "inefficient_loops": [r"for.*in.*range\(\d{4,}\)", r"while.*len\(.*\)\s*>\s*\d{3,}"],
            "large_data_processing": [r"\.sort\(\)", r"sorted\(.*len\(.*\)\s*>\s*\d{4,}\)"],
            "recursive_depth": [r"def.*\(.*n.*\).*return.*\(.*n-1.*\).*\(.*n-2.*\)"],
            "regex_complexity": [r"re\.search\(.*\*.*\*", r"re\.findall\(.*\+.*\+"]
        }

        for bottleneck_type, patterns in bottleneck_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code, re.MULTILINE):
                    severity = "high" if bottleneck_type in ["blocking_io", "recursive_depth"] else "medium"
                    bottlenecks.append({
                        "type": bottleneck_type,
                        "severity": severity,
                        "description": f"Potential {bottleneck_type.replace('_', ' ')} bottleneck"
                    })
                    bottleneck_score -= 2.0 if severity == "high" else 1.0

        return {
            "bottleneck_score": max(0, bottleneck_score),
            "bottlenecks": bottlenecks,
            "recommendations": self._get_bottleneck_recommendations(bottlenecks)
        }

    def load_knowledge(self) -> Dict[str, List[str]]:
        """Load performance-specific knowledge base."""
        return {
            "performance_patterns": [
                "Use appropriate data structures (dict for O(1) lookup, set for O(1) membership)",
                "Implement caching for expensive computations and database queries",
                "Use generators and iterators for memory efficiency",
                "Optimize database queries with proper indexing and pagination",
                "Implement connection pooling for database connections",
                "Use asynchronous programming for I/O-bound operations",
                "Profile code to identify actual bottlenecks before optimizing",
                "Use lazy loading for large datasets",
                "Implement proper error handling to avoid performance degradation",
                "Use batch operations for multiple database writes",
                "Optimize regular expressions and use compiled patterns",
                "Use appropriate sorting algorithms for different data sizes",
                "Implement proper memory management in long-running processes",
                "Use CDN and browser caching for web applications",
                "Optimize image and asset sizes for faster loading",
                "Use compression for data transmission",
                "Implement database query result caching",
                "Use index-based database queries instead of table scans",
                "Optimize JSON serialization/deserialization",
                "Use appropriate HTTP methods and status codes",
                "Implement request batching for API calls",
                "Use connection keep-alive for HTTP requests",
                "Optimize CSS and JavaScript for faster page loads",
                "Use database connection pooling",
                "Implement proper pagination for large result sets",
                "Use streaming for large file processing",
                "Optimize algorithm complexity where possible",
                "Use memoization for recursive functions",
                "Implement proper resource cleanup and disposal",
                "Use appropriate timeout values for external calls"
            ],
            "performance_anti_patterns": [
                "N+1 query problem in database operations",
                "Using SELECT * instead of specific columns",
                "Loading entire datasets into memory unnecessarily",
                "Using string concatenation in loops",
                "Not using database indexes on frequently queried columns",
                "Implementing recursive functions without memoization",
                "Using synchronous calls for I/O operations",
                "Not implementing proper caching strategies",
                "Using inefficient sorting algorithms for large datasets",
                "Not cleaning up resources and connections",
                "Using polling instead of event-driven architectures",
                "Implementing complex regex patterns without optimization",
                "Not using connection pooling for databases",
                "Loading large files entirely into memory",
                "Using nested loops with high time complexity",
                "Not implementing proper error handling causing retries",
                "Using inappropriate data structures for operations",
                "Not using lazy loading for optional data",
                "Implementing custom solutions instead of optimized libraries",
                "Not considering memory leaks in long-running processes"
            ],
            "risk_factors": [
                "High time complexity algorithms (O(n^2) or worse)",
                "Memory-intensive operations without bounds",
                "Blocking I/O operations in single-threaded applications",
                "Lack of caching for expensive operations",
                "Inefficient database query patterns",
                "Unbounded data structure growth",
                "Recursive functions without proper base cases",
                "Synchronous external API calls",
                "Large file processing without streaming",
                "Complex regular expressions",
                "Inefficient serialization/deserialization",
                "Missing database indexes on query columns",
                "Connection leaks in database operations",
                "Memory leaks in long-running processes",
                "CPU-intensive operations blocking main thread"
            ],
            "optimization_strategies": [
                "Implement algorithmic improvements to reduce time complexity",
                "Add caching layers (memory, Redis, CDN) for frequently accessed data",
                "Optimize database queries with proper indexing and query structure",
                "Use asynchronous programming for I/O-bound operations",
                "Implement connection pooling for database and HTTP connections",
                "Use streaming for large data processing",
                "Add monitoring and profiling to identify actual bottlenecks",
                "Implement lazy loading for optional or expensive operations",
                "Use appropriate data structures for specific operations",
                "Add compression for data transmission and storage"
            ]
        }

    def _load_complexity_patterns(self) -> Dict[str, List[str]]:
        """Load algorithm complexity detection patterns."""
        return {
            "constant": [r"return\s+\w+", r"^\s*\w+\s*="],
            "logarithmic": [r"while.*//=.*2", r"binary.*search"],
            "linear": [r"for.*in.*range\(", r"while.*<.*len\("],
            "linearithmic": [r"\.sort\(\)", r"merge.*sort"],
            "quadratic": [r"for.*in.*:\s*for.*in.*:"],
            "exponential": [r"fibonacci.*recursive"],
            "factorial": [r"permutation", r"factorial.*recursive"]
        }

    def _perform_performance_analysis(self, code: str, task_type: str, description: str) -> Dict[str, Any]:
        """Perform comprehensive performance analysis."""
        findings = {
            "issues": [],
            "time_complexity": "O(1)",
            "space_complexity": "O(1)",
            "db_query_count": 0,
            "caching_score": 5.0,
            "resource_efficiency": "Unknown"
        }

        if code:
            # Time complexity analysis
            complexity_analysis = self.analyze_time_complexity(code)
            findings["time_complexity"] = complexity_analysis["complexity"]

            # Memory usage analysis
            memory_analysis = self.check_memory_usage(code)
            findings["space_complexity"] = memory_analysis["space_complexity"]
            findings["issues"].extend(memory_analysis["issues"])

            # Database optimization analysis
            query_analysis = self.optimize_queries(code)
            findings["issues"].extend(query_analysis["issues"])

            # Caching assessment
            caching_analysis = self.assess_caching(code)
            findings["caching_score"] = caching_analysis["caching_score"]
            findings["issues"].extend([{"type": "caching", "opportunities": caching_analysis["opportunities"]}])

            # Bottleneck detection
            bottleneck_analysis = self.detect_bottlenecks(code)
            findings["issues"].extend(bottleneck_analysis["bottlenecks"])

        return findings

    def _calculate_performance_score(self, findings: Dict[str, Any]) -> float:
        """Calculate overall performance score based on findings."""
        base_score = 10.0

        # Deduct points based on issues
        for issue in findings.get("issues", []):
            severity = issue.get("severity", "medium")
            base_score -= self.performance_weights.get(severity, 0.5)

        # Factor in complexity
        complexity = findings.get("time_complexity", "O(1)")
        if "n!" in complexity:
            base_score -= 4.0
        elif "2^n" in complexity:
            base_score -= 3.0
        elif "n^2" in complexity:
            base_score -= 2.0
        elif "n log n" in complexity:
            base_score -= 1.0

        # Factor in caching score
        caching_score = findings.get("caching_score", 5.0)
        base_score = (base_score * 0.7) + (caching_score * 0.3)

        return max(0.0, min(10.0, base_score))

    def _assess_optimization_opportunities(self, findings: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Assess optimization opportunities based on findings."""
        opportunities = []

        # Check for algorithmic improvements
        complexity = findings.get("time_complexity", "O(1)")
        if any(bad_complexity in complexity for bad_complexity in ["n^2", "2^n", "n!"]):
            opportunities.append({
                "type": "algorithmic_improvement",
                "priority": "high",
                "description": f"Consider algorithm optimization to reduce {complexity} complexity",
                "impact": "high"
            })

        # Check for caching opportunities
        if findings.get("caching_score", 5.0) < 7.0:
            opportunities.append({
                "type": "caching_implementation",
                "priority": "medium",
                "description": "Implement caching for frequently accessed data",
                "impact": "medium"
            })

        # Check for database optimizations
        db_issues = [issue for issue in findings.get("issues", []) if "query" in issue.get("type", "")]
        if db_issues:
            opportunities.append({
                "type": "database_optimization",
                "priority": "high",
                "description": "Optimize database queries and add proper indexing",
                "impact": "high"
            })

        return opportunities

    def _generate_performance_recommendations(self, findings: Dict[str, Any], task_type: str) -> List[Dict[str, Any]]:
        """Generate specific performance recommendations."""
        recommendations = []

        # Algorithm-specific recommendations
        complexity = findings.get("time_complexity", "O(1)")
        if "n^2" in complexity:
            recommendations.append({
                "type": "algorithm",
                "priority": "high",
                "description": "Consider using more efficient algorithms or data structures",
                "implementation": "Use hash tables for O(1) lookup, implement divide-and-conquer approaches"
            })

        # Memory recommendations
        memory_issues = [issue for issue in findings.get("issues", []) if "memory" in issue.get("type", "")]
        if memory_issues:
            recommendations.append({
                "type": "memory",
                "priority": "medium",
                "description": "Implement memory optimization strategies",
                "implementation": "Use generators, implement proper cleanup, avoid global variables"
            })

        # Database recommendations
        db_issues = [issue for issue in findings.get("issues", []) if "query" in issue.get("type", "")]
        if db_issues:
            recommendations.append({
                "type": "database",
                "priority": "high",
                "description": "Optimize database operations",
                "implementation": "Add indexes, use pagination, implement query batching"
            })

        # Caching recommendations
        if findings.get("caching_score", 5.0) < 6.0:
            recommendations.append({
                "type": "caching",
                "priority": "medium",
                "description": "Implement comprehensive caching strategy",
                "implementation": "Add Redis caching, implement memoization, use CDN for static assets"
            })

        return recommendations

    def _calculate_risk_level(self, findings: Dict[str, Any]) -> str:
        """Calculate performance risk level."""
        risk_score = 0

        # High complexity algorithms are high risk
        complexity = findings.get("time_complexity", "O(1)")
        if any(bad_complexity in complexity for bad_complexity in ["n!", "2^n"]):
            risk_score += 3
        elif "n^2" in complexity:
            risk_score += 2

        # Count critical issues
        critical_issues = len([issue for issue in findings.get("issues", [])
                             if issue.get("severity") == "high"])
        risk_score += critical_issues

        if risk_score >= 4:
            return "High"
        elif risk_score >= 2:
            return "Medium"
        else:
            return "Low"

    def _estimate_space_complexity(self, code: str) -> str:
        """Estimate space complexity of code."""
        if re.search(r"for.*in.*:\s*for.*in.*:", code):
            return "O(n^2)"
        elif re.search(r"list\(.*range\(", code):
            return "O(n)"
        elif re.search(r"def.*\(.*n.*\).*return.*\(.*n-1.*\)", code):
            return "O(n)"
        else:
            return "O(1)"

    def _get_memory_recommendations(self, memory_issues: List[Dict[str, Any]]) -> List[str]:
        """Get specific memory optimization recommendations."""
        recommendations = []

        for issue in memory_issues:
            issue_type = issue.get("type", "")
            if issue_type == "large_list_creation":
                recommendations.append("Use generators or iterators instead of creating large lists")
            elif issue_type == "string_concatenation":
                recommendations.append("Use join() for string concatenation in loops")
            elif issue_type == "recursive_without_memoization":
                recommendations.append("Implement memoization for recursive functions")
            elif issue_type == "global_variables":
                recommendations.append("Minimize global variable usage, consider class-based approaches")
            elif issue_type == "memory_leaks":
                recommendations.append("Implement proper resource cleanup and avoid infinite loops")

        return recommendations

    def _get_query_recommendations(self, query_issues: List[Dict[str, Any]]) -> List[str]:
        """Get specific database query optimization recommendations."""
        recommendations = []

        for issue in query_issues:
            issue_type = issue.get("type", "")
            if issue_type == "n_plus_one":
                recommendations.append("Use select_related() or prefetch_related() to avoid N+1 queries")
            elif issue_type == "missing_indexes":
                recommendations.append("Add database indexes on frequently queried columns")
            elif issue_type == "select_all":
                recommendations.append("Select only necessary columns instead of using SELECT *")
            elif issue_type == "no_pagination":
                recommendations.append("Implement pagination for large result sets")
            elif issue_type == "inefficient_joins":
                recommendations.append("Optimize join operations and consider query restructuring")

        return recommendations

    def _get_caching_recommendations(self, opportunities: List[Dict[str, Any]]) -> List[str]:
        """Get specific caching implementation recommendations."""
        recommendations = []

        for opportunity in opportunities:
            opportunity_type = opportunity.get("type", "")
            if opportunity_type == "expensive_computation":
                recommendations.append("Implement memoization or Redis caching for expensive computations")
            elif opportunity_type == "database_queries":
                recommendations.append("Add query result caching with appropriate TTL")
            elif opportunity_type == "api_calls":
                recommendations.append("Cache API responses with proper invalidation strategies")
            elif opportunity_type == "file_operations":
                recommendations.append("Cache file contents when appropriate")

        return recommendations

    def _get_bottleneck_recommendations(self, bottlenecks: List[Dict[str, Any]]) -> List[str]:
        """Get specific bottleneck resolution recommendations."""
        recommendations = []

        for bottleneck in bottlenecks:
            bottleneck_type = bottleneck.get("type", "")
            if bottleneck_type == "blocking_io":
                recommendations.append("Use asynchronous I/O operations to avoid blocking")
            elif bottleneck_type == "synchronous_calls":
                recommendations.append("Implement async/await for external API calls")
            elif bottleneck_type == "inefficient_loops":
                recommendations.append("Optimize loop logic and consider vectorized operations")
            elif bottleneck_type == "large_data_processing":
                recommendations.append("Use streaming or batch processing for large datasets")
            elif bottleneck_type == "recursive_depth":
                recommendations.append("Implement iterative solutions or tail recursion optimization")

        return recommendations

    def _create_error_response(self, error: Exception, task_type: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "director": self.name,
            "task_type": task_type,
            "timestamp": datetime.now().isoformat(),
            "performance_score": 0.0,
            "confidence": 0.0,
            "error": str(error),
            "status": "error",
            "recommendations": [f"Resolve evaluation error before proceeding (Address the following error: {str(error)})"]
        }