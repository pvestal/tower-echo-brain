#!/usr/bin/env python3
"""
Database Query Optimization Utilities for Echo Brain System

Features:
- Query performance analysis
- Automatic query optimization suggestions
- N+1 query detection and prevention
- Query plan analysis
- Database index recommendations
"""

import re
import time
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class QueryAnalysis:
    """Analysis result for a database query"""
    query_hash: str
    original_query: str
    query_type: str  # SELECT, INSERT, UPDATE, DELETE
    table_accessed: List[str]
    has_where_clause: bool
    has_joins: bool
    has_subqueries: bool
    estimated_complexity: int  # 1-10 scale
    optimization_suggestions: List[str] = field(default_factory=list)
    index_suggestions: List[str] = field(default_factory=list)
    execution_plan: Optional[Dict] = None

@dataclass
class NPlusOneDetection:
    """Detection result for N+1 query patterns"""
    detected: bool = False
    parent_query: str = ""
    repeated_query_pattern: str = ""
    occurrence_count: int = 0
    suggestions: List[str] = field(default_factory=list)

class QueryOptimizer:
    """
    Database query optimization and performance analysis
    """

    def __init__(self):
        self.query_patterns = defaultdict(list)
        self.query_timings = defaultdict(list)
        self.n_plus_one_detector = defaultdict(int)
        self.common_optimization_rules = [
            {
                'pattern': r'SELECT \* FROM',
                'suggestion': 'Avoid SELECT *, specify only needed columns',
                'severity': 'medium'
            },
            {
                'pattern': r'ORDER BY.*LIMIT',
                'suggestion': 'Consider adding an index on ORDER BY columns for pagination',
                'severity': 'medium'
            },
            {
                'pattern': r'WHERE.*LIKE.*%.*%',
                'suggestion': 'LIKE with leading wildcards prevents index usage, consider full-text search',
                'severity': 'high'
            },
            {
                'pattern': r'LEFT JOIN.*WHERE.*IS NULL',
                'suggestion': 'Consider using NOT EXISTS instead of LEFT JOIN...IS NULL',
                'severity': 'low'
            },
            {
                'pattern': r'COUNT\(\*\).*WHERE',
                'suggestion': 'COUNT(*) with WHERE can be expensive on large tables, consider approximations',
                'severity': 'medium'
            }
        ]

    def analyze_query(self, query: str, execution_time: float = None) -> QueryAnalysis:
        """
        Analyze a query for optimization opportunities
        """
        query_clean = self._normalize_query(query)
        query_hash = hashlib.md5(query_clean.encode()).hexdigest()

        # Basic query parsing
        query_upper = query_clean.upper()

        # Determine query type
        query_type = 'UNKNOWN'
        if query_upper.startswith('SELECT'):
            query_type = 'SELECT'
        elif query_upper.startswith('INSERT'):
            query_type = 'INSERT'
        elif query_upper.startswith('UPDATE'):
            query_type = 'UPDATE'
        elif query_upper.startswith('DELETE'):
            query_type = 'DELETE'

        # Extract table names (simplified approach)
        tables = self._extract_table_names(query_clean)

        # Analyze query characteristics
        has_where = 'WHERE' in query_upper
        has_joins = any(join in query_upper for join in ['JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN'])
        has_subqueries = '(' in query and 'SELECT' in query_upper

        # Estimate complexity
        complexity = self._calculate_complexity(query_clean, has_joins, has_subqueries, tables)

        # Generate optimization suggestions
        suggestions = self._get_optimization_suggestions(query_clean)
        index_suggestions = self._get_index_suggestions(query_clean, tables)

        # Store for pattern analysis
        if execution_time:
            self.query_timings[query_hash].append(execution_time)

        analysis = QueryAnalysis(
            query_hash=query_hash,
            original_query=query,
            query_type=query_type,
            table_accessed=tables,
            has_where_clause=has_where,
            has_joins=has_joins,
            has_subqueries=has_subqueries,
            estimated_complexity=complexity,
            optimization_suggestions=suggestions,
            index_suggestions=index_suggestions
        )

        return analysis

    def detect_n_plus_one(
        self,
        queries: List[Tuple[str, float]],
        time_window: float = 1.0
    ) -> NPlusOneDetection:
        """
        Detect N+1 query patterns in a sequence of queries
        """
        if len(queries) < 3:
            return NPlusOneDetection()

        # Look for patterns where one query is followed by many similar queries
        query_patterns = []
        for query, timestamp in queries:
            normalized = self._normalize_query_for_n_plus_one(query)
            query_patterns.append((normalized, timestamp))

        # Group queries by time windows
        time_groups = []
        current_group = [query_patterns[0]]

        for i in range(1, len(query_patterns)):
            query, timestamp = query_patterns[i]
            prev_query, prev_timestamp = query_patterns[i-1]

            if timestamp - prev_timestamp <= time_window:
                current_group.append((query, timestamp))
            else:
                if len(current_group) > 2:
                    time_groups.append(current_group)
                current_group = [(query, timestamp)]

        if len(current_group) > 2:
            time_groups.append(current_group)

        # Analyze each time group for N+1 patterns
        for group in time_groups:
            pattern_counts = defaultdict(int)
            for query, _ in group:
                pattern_counts[query] += 1

            # Look for repeated patterns (N+1 indicator)
            for pattern, count in pattern_counts.items():
                if count >= 3:  # Threshold for N+1 detection
                    suggestions = [
                        "Consider using JOIN or subquery to fetch related data in single query",
                        "Implement eager loading or batch loading for related entities",
                        "Use database-level solutions like materialized views for complex aggregations",
                        f"Query executed {count} times in {time_window}s window - likely N+1 pattern"
                    ]

                    return NPlusOneDetection(
                        detected=True,
                        repeated_query_pattern=pattern,
                        occurrence_count=count,
                        suggestions=suggestions
                    )

        return NPlusOneDetection()

    async def analyze_execution_plan(self, pool, query: str) -> Optional[Dict]:
        """
        Get PostgreSQL execution plan for query analysis
        """
        try:
            explain_query = f"EXPLAIN (FORMAT JSON, ANALYZE, BUFFERS) {query}"
            async with pool.acquire_connection() as connection:
                result = await connection.fetchval(explain_query)
                return result[0] if result else None
        except Exception as e:
            logger.warning(f"Could not get execution plan: {e}")
            return None

    def get_index_recommendations(
        self,
        slow_queries: List[Dict],
        threshold_time: float = 1.0
    ) -> List[Dict]:
        """
        Generate index recommendations based on slow query analysis
        """
        recommendations = []

        for query_info in slow_queries:
            query = query_info.get('query', '')
            avg_time = query_info.get('avg_time', 0)

            if avg_time >= threshold_time:
                analysis = self.analyze_query(query, avg_time)

                if analysis.index_suggestions:
                    recommendations.append({
                        'query_hash': analysis.query_hash,
                        'query_sample': query[:200],
                        'avg_execution_time': avg_time,
                        'suggested_indexes': analysis.index_suggestions,
                        'reason': f"Query averaging {avg_time:.3f}s execution time",
                        'tables_affected': analysis.table_accessed
                    })

        return recommendations

    def _normalize_query(self, query: str) -> str:
        """Normalize query for analysis"""
        # Remove extra whitespace and comments
        query = re.sub(r'\s+', ' ', query.strip())
        query = re.sub(r'--.*$', '', query, flags=re.MULTILINE)
        query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)
        return query

    def _normalize_query_for_n_plus_one(self, query: str) -> str:
        """Normalize query for N+1 detection (parameters become placeholders)"""
        normalized = self._normalize_query(query)

        # Replace parameters with placeholders
        normalized = re.sub(r'\$\d+', '$?', normalized)  # PostgreSQL parameters
        normalized = re.sub(r"'[^']*'", "'?'", normalized)  # String literals
        normalized = re.sub(r'\b\d+\b', '?', normalized)  # Numbers

        return normalized

    def _extract_table_names(self, query: str) -> List[str]:
        """Extract table names from query (simplified approach)"""
        tables = []

        # Simple regex patterns for table extraction
        # This is a simplified approach - a real implementation would use a SQL parser
        patterns = [
            r'FROM\s+(\w+)',
            r'JOIN\s+(\w+)',
            r'UPDATE\s+(\w+)',
            r'INSERT\s+INTO\s+(\w+)',
            r'DELETE\s+FROM\s+(\w+)'
        ]

        query_upper = query.upper()
        for pattern in patterns:
            matches = re.findall(pattern, query_upper)
            tables.extend(matches)

        return list(set(tables))  # Remove duplicates

    def _calculate_complexity(
        self,
        query: str,
        has_joins: bool,
        has_subqueries: bool,
        tables: List[str]
    ) -> int:
        """Calculate query complexity on 1-10 scale"""
        complexity = 1

        # Base complexity factors
        if len(tables) > 3:
            complexity += 2
        elif len(tables) > 1:
            complexity += 1

        if has_joins:
            join_count = len(re.findall(r'JOIN', query.upper()))
            complexity += min(join_count * 1.5, 3)

        if has_subqueries:
            subquery_count = query.count('(')  # Rough estimate
            complexity += min(subquery_count * 1, 2)

        # Additional complexity factors
        if 'GROUP BY' in query.upper():
            complexity += 1
        if 'ORDER BY' in query.upper():
            complexity += 1
        if 'HAVING' in query.upper():
            complexity += 1
        if 'UNION' in query.upper():
            complexity += 2

        return min(complexity, 10)

    def _get_optimization_suggestions(self, query: str) -> List[str]:
        """Get optimization suggestions for a query"""
        suggestions = []

        # Apply common optimization rules
        for rule in self.common_optimization_rules:
            if re.search(rule['pattern'], query, re.IGNORECASE):
                suggestions.append(f"[{rule['severity'].upper()}] {rule['suggestion']}")

        # Additional specific suggestions
        if 'SELECT *' in query.upper():
            suggestions.append("[MEDIUM] Specify exact columns instead of SELECT *")

        if re.search(r'WHERE.*IN\s*\([^)]*SELECT', query, re.IGNORECASE):
            suggestions.append("[HIGH] Consider EXISTS instead of IN with subquery for better performance")

        return suggestions

    def _get_index_suggestions(self, query: str, tables: List[str]) -> List[str]:
        """Generate index suggestions for a query"""
        suggestions = []

        # Extract WHERE clause columns
        where_match = re.search(r'WHERE\s+(.+?)(?:\s+GROUP BY|\s+ORDER BY|\s+LIMIT|$)',
                               query, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1)

            # Find column references in WHERE clause
            column_patterns = re.findall(r'(\w+)\s*[=<>]', where_clause)
            for table in tables:
                for column in column_patterns:
                    suggestions.append(f"Consider index on {table}.{column}")

        # Extract ORDER BY columns
        order_match = re.search(r'ORDER BY\s+([^)]+?)(?:\s+LIMIT|$)', query, re.IGNORECASE)
        if order_match:
            order_columns = order_match.group(1)
            order_cols = [col.strip().split()[0] for col in order_columns.split(',')]
            for table in tables:
                if len(order_cols) > 1:
                    suggestions.append(f"Consider composite index on {table}({', '.join(order_cols)})")

        # JOIN conditions
        join_matches = re.findall(r'JOIN\s+\w+.*?ON\s+([^)]+?)(?:\s+WHERE|\s+GROUP|\s+ORDER|$)',
                                query, re.IGNORECASE | re.DOTALL)
        for join_condition in join_matches:
            # Extract column references from join conditions
            join_cols = re.findall(r'(\w+\.\w+)', join_condition)
            for join_col in join_cols:
                suggestions.append(f"Consider index on {join_col}")

        return list(set(suggestions))  # Remove duplicates


# Singleton instance
_optimizer = None

def get_optimizer() -> QueryOptimizer:
    """Get global query optimizer instance"""
    global _optimizer
    if _optimizer is None:
        _optimizer = QueryOptimizer()
    return _optimizer

async def analyze_query(query: str, execution_time: float = None) -> QueryAnalysis:
    """Analyze a query for optimization opportunities"""
    optimizer = get_optimizer()
    return optimizer.analyze_query(query, execution_time)

async def detect_n_plus_one(queries: List[Tuple[str, float]]) -> NPlusOneDetection:
    """Detect N+1 query patterns"""
    optimizer = get_optimizer()
    return optimizer.detect_n_plus_one(queries)

async def get_index_recommendations(slow_queries: List[Dict]) -> List[Dict]:
    """Get index recommendations for slow queries"""
    optimizer = get_optimizer()
    return optimizer.get_index_recommendations(slow_queries)