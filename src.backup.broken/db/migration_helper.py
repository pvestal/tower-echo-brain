#!/usr/bin/env python3
"""
Migration Helper for Database Connection Pool Implementation

This utility helps migrate existing psycopg2 direct connection code
to use the new AsyncPG connection pool for better performance.
"""

import re
import os
import logging
import asyncio
from typing import List, Dict, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class DatabaseMigrationHelper:
    """
    Helper class to migrate code from psycopg2 direct connections to AsyncPG pools
    """

    def __init__(self, project_root: str = "/opt/tower-echo-brain/src"):
        self.project_root = Path(project_root)
        self.files_to_migrate = []
        self.migration_patterns = [
            {
                'name': 'Direct psycopg2.connect usage',
                'old_pattern': r'psycopg2\.connect\(\*\*[^)]+\)',
                'new_pattern': 'async with acquire_connection() as connection:',
                'requires_async': True,
                'description': 'Replace direct psycopg2.connect with pooled connections'
            },
            {
                'name': 'Cursor creation pattern',
                'old_pattern': r'cursor = conn\.cursor\([^)]*\)',
                'new_pattern': 'cursor = connection.cursor()',
                'requires_async': False,
                'description': 'Update cursor creation for pool connections'
            },
            {
                'name': 'Connection close pattern',
                'old_pattern': r'conn\.close\(\)',
                'new_pattern': '# Connection automatically returned to pool',
                'requires_async': False,
                'description': 'Remove manual connection closing (handled by pool)'
            },
            {
                'name': 'Manual commit pattern',
                'old_pattern': r'conn\.commit\(\)',
                'new_pattern': '# Auto-commit handled by pool context',
                'requires_async': False,
                'description': 'Remove manual commits (handled automatically)'
            }
        ]

    async def scan_codebase(self) -> Dict[str, List[Dict]]:
        """
        Scan the codebase for files that need migration
        """
        migration_candidates = {}

        # Find all Python files
        python_files = list(self.project_root.rglob("*.py"))

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for psycopg2 usage
                if 'psycopg2' in content:
                    issues = self._analyze_file_content(content, str(file_path))
                    if issues:
                        migration_candidates[str(file_path)] = issues

            except Exception as e:
                logger.warning(f"Could not analyze file {file_path}: {e}")

        return migration_candidates

    def _analyze_file_content(self, content: str, file_path: str) -> List[Dict]:
        """
        Analyze file content for migration issues
        """
        issues = []

        for pattern in self.migration_patterns:
            matches = re.findall(pattern['old_pattern'], content, re.MULTILINE)
            if matches:
                for match in matches:
                    issues.append({
                        'type': pattern['name'],
                        'pattern': match,
                        'suggested_fix': pattern['new_pattern'],
                        'requires_async': pattern['requires_async'],
                        'description': pattern['description'],
                        'line_number': self._find_line_number(content, match)
                    })

        # Check for async compatibility
        if issues and 'async def' not in content and any(issue['requires_async'] for issue in issues):
            issues.append({
                'type': 'Missing async function',
                'pattern': 'Function needs to be async',
                'suggested_fix': 'Convert function to async def',
                'requires_async': True,
                'description': 'Function must be async to use connection pool',
                'line_number': 1
            })

        return issues

    def _find_line_number(self, content: str, pattern: str) -> int:
        """
        Find line number where pattern occurs
        """
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if pattern in line:
                return i + 1
        return 1

    async def generate_migration_plan(self) -> Dict[str, any]:
        """
        Generate a comprehensive migration plan
        """
        candidates = await self.scan_codebase()

        # Categorize files by migration complexity
        easy_migrations = []
        complex_migrations = []
        requires_async_conversion = []

        for file_path, issues in candidates.items():
            async_required = any(issue['requires_async'] for issue in issues)
            issue_count = len(issues)

            file_info = {
                'file_path': file_path,
                'issues': issues,
                'issue_count': issue_count,
                'async_required': async_required
            }

            if async_required:
                requires_async_conversion.append(file_info)
            elif issue_count <= 3:
                easy_migrations.append(file_info)
            else:
                complex_migrations.append(file_info)

        migration_plan = {
            'summary': {
                'total_files': len(candidates),
                'easy_migrations': len(easy_migrations),
                'complex_migrations': len(complex_migrations),
                'async_conversions_needed': len(requires_async_conversion)
            },
            'files': {
                'easy_migrations': easy_migrations,
                'complex_migrations': complex_migrations,
                'async_conversions': requires_async_conversion
            },
            'recommendations': self._generate_recommendations(candidates)
        }

        return migration_plan

    def _generate_recommendations(self, candidates: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Generate migration recommendations based on analysis
        """
        recommendations = []

        total_files = len(candidates)
        total_issues = sum(len(issues) for issues in candidates.values())

        if total_files > 20:
            recommendations.append({
                'priority': 'high',
                'action': 'Phase the migration',
                'description': f'With {total_files} files to migrate, consider phasing the migration',
                'timeline': '2-3 weeks'
            })

        if total_issues > 100:
            recommendations.append({
                'priority': 'medium',
                'action': 'Create migration scripts',
                'description': f'{total_issues} total issues detected - automated migration scripts recommended',
                'timeline': '1 week'
            })

        recommendations.extend([
            {
                'priority': 'high',
                'action': 'Start with high-traffic endpoints',
                'description': 'Migrate main API endpoints first for immediate performance improvement',
                'timeline': '1 week'
            },
            {
                'priority': 'medium',
                'action': 'Update imports',
                'description': 'Replace psycopg2 imports with async_database imports',
                'timeline': '2 days'
            },
            {
                'priority': 'low',
                'action': 'Add performance monitoring',
                'description': 'Implement monitoring for migrated endpoints',
                'timeline': '3 days'
            }
        ])

        return recommendations

    async def create_migration_patch(self, file_path: str) -> Optional[str]:
        """
        Create a migration patch for a specific file
        """
        try:
            with open(file_path, 'r') as f:
                original_content = f.read()

            modified_content = original_content

            # Apply migration patterns
            for pattern in self.migration_patterns:
                modified_content = re.sub(
                    pattern['old_pattern'],
                    pattern['new_pattern'],
                    modified_content,
                    flags=re.MULTILINE
                )

            # Add necessary imports if not present
            if 'from ..db.async_database import' not in modified_content:
                import_line = "from ..db.async_database import get_async_database, execute_query, execute_one, execute_command\n"

                # Find where to insert import
                lines = modified_content.split('\n')
                import_index = 0
                for i, line in enumerate(lines):
                    if line.startswith('import ') or line.startswith('from '):
                        import_index = i
                    elif line.strip() and not line.startswith('#'):
                        break

                lines.insert(import_index + 1, import_line)
                modified_content = '\n'.join(lines)

            if modified_content != original_content:
                return modified_content
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to create migration patch for {file_path}: {e}")
            return None

    async def validate_migration(self, file_path: str) -> Dict[str, any]:
        """
        Validate that a migrated file follows best practices
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            validation_results = {
                'file_path': file_path,
                'valid': True,
                'issues': [],
                'score': 100
            }

            # Check for common issues
            checks = [
                {
                    'name': 'Uses connection pooling',
                    'pattern': r'acquire_connection\(\)|execute_query\(|execute_one\(|execute_command\(',
                    'required': True,
                    'weight': 30
                },
                {
                    'name': 'No direct psycopg2.connect',
                    'pattern': r'psycopg2\.connect\(',
                    'required': False,
                    'weight': 25
                },
                {
                    'name': 'Proper async usage',
                    'pattern': r'async def.*await',
                    'required': True,
                    'weight': 20
                },
                {
                    'name': 'Error handling',
                    'pattern': r'try:|except:',
                    'required': True,
                    'weight': 15
                },
                {
                    'name': 'Uses async database imports',
                    'pattern': r'from.*async_database import',
                    'required': True,
                    'weight': 10
                }
            ]

            for check in checks:
                if check['required']:
                    if not re.search(check['pattern'], content):
                        validation_results['issues'].append(f"Missing: {check['name']}")
                        validation_results['score'] -= check['weight']
                else:
                    if re.search(check['pattern'], content):
                        validation_results['issues'].append(f"Found deprecated pattern: {check['name']}")
                        validation_results['score'] -= check['weight']

            validation_results['valid'] = validation_results['score'] >= 80

            return validation_results

        except Exception as e:
            return {
                'file_path': file_path,
                'valid': False,
                'issues': [f"Validation error: {e}"],
                'score': 0
            }

    async def create_performance_test(self, file_path: str) -> str:
        """
        Create a performance test comparing old vs new implementation
        """
        test_name = Path(file_path).stem

        test_code = f'''#!/usr/bin/env python3
"""
Performance test for {test_name} migration
Compares old psycopg2 direct connections vs new AsyncPG pool
"""

import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

# Old implementation (for comparison)
import psycopg2

# New implementation
from ..db.async_database import get_async_database, execute_query

async def test_old_vs_new_performance():
    """Test performance comparison"""

    # Test parameters
    num_queries = 100
    concurrent_requests = 10

    print(f"Performance Test: {{num_queries}} queries with {{concurrent_requests}} concurrent requests")

    # Test new async pool implementation
    print("Testing new AsyncPG pool...")
    async_times = await test_async_performance(num_queries, concurrent_requests)

    print("Testing old psycopg2 direct connections...")
    sync_times = await test_sync_performance(num_queries, concurrent_requests)

    # Calculate statistics
    async_avg = statistics.mean(async_times)
    async_median = statistics.median(async_times)
    sync_avg = statistics.mean(sync_times)
    sync_median = statistics.median(sync_times)

    improvement = ((sync_avg - async_avg) / sync_avg) * 100

    print("\\nResults:")
    print(f"Async Pool - Avg: {{async_avg:.3f}}s, Median: {{async_median:.3f}}s")
    print(f"Direct Conn - Avg: {{sync_avg:.3f}}s, Median: {{sync_median:.3f}}s")
    print(f"Performance Improvement: {{improvement:.1f}}%")

    return {{
        'async_avg': async_avg,
        'sync_avg': sync_avg,
        'improvement_percent': improvement,
        'async_times': async_times,
        'sync_times': sync_times
    }}

async def test_async_performance(num_queries: int, concurrent_requests: int):
    """Test async pool performance"""
    times = []

    async def run_query():
        start = time.time()
        result = await execute_query("SELECT 1 as test")
        end = time.time()
        return end - start

    # Run concurrent batches
    for batch in range(0, num_queries, concurrent_requests):
        batch_size = min(concurrent_requests, num_queries - batch)
        tasks = [run_query() for _ in range(batch_size)]
        batch_times = await asyncio.gather(*tasks)
        times.extend(batch_times)

    return times

async def test_sync_performance(num_queries: int, concurrent_requests: int):
    """Test sync psycopg2 performance"""
    from ..security.credential_validator import get_secure_db_config

    def run_sync_query():
        start = time.time()
        try:
            conn = psycopg2.connect(**get_secure_db_config())
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Sync query error: {{e}}")
        end = time.time()
        return end - start

    times = []

    # Use ThreadPoolExecutor to simulate concurrent requests
    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        for batch in range(0, num_queries, concurrent_requests):
            batch_size = min(concurrent_requests, num_queries - batch)
            futures = [executor.submit(run_sync_query) for _ in range(batch_size)]
            batch_times = [future.result() for future in futures]
            times.extend(batch_times)

    return times

if __name__ == "__main__":
    asyncio.run(test_old_vs_new_performance())
'''

        return test_code


# Global migration helper instance
_migration_helper = None

async def get_migration_helper() -> DatabaseMigrationHelper:
    """Get global migration helper instance"""
    global _migration_helper
    if _migration_helper is None:
        _migration_helper = DatabaseMigrationHelper()
    return _migration_helper

async def run_migration_analysis():
    """Run complete migration analysis"""
    helper = await get_migration_helper()
    plan = await helper.generate_migration_plan()

    logger.info(f"Migration Analysis Complete:")
    logger.info(f"  Files to migrate: {plan['summary']['total_files']}")
    logger.info(f"  Easy migrations: {plan['summary']['easy_migrations']}")
    logger.info(f"  Complex migrations: {plan['summary']['complex_migrations']}")
    logger.info(f"  Async conversions needed: {plan['summary']['async_conversions_needed']}")

    return plan