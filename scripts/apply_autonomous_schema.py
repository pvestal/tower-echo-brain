#!/usr/bin/env python3
"""
Autonomous Core Database Schema Application Script

This script applies the autonomous core database schema to the PostgreSQL database.
It handles creating tables, indexes, triggers, and comments while gracefully
handling existing objects.

Features:
- Reads schema.sql from the autonomous module
- Connects to PostgreSQL using environment variables or defaults
- Creates tables and objects if they don't exist
- Reports what was created vs what was skipped
- Validates schema after application
- Creates backup before making changes (optional)

Usage:
    python apply_autonomous_schema.py [options]

Options:
    --dry-run       : Show what would be done without making changes
    --backup        : Create backup before applying schema
    --force         : Drop and recreate existing objects
    --verbose       : Enable verbose logging
    --validate-only : Only validate existing schema
"""

import asyncio
import asyncpg
import argparse
import logging
import os
import sys
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SchemaApplicator:
    """
    Applies the autonomous core database schema to PostgreSQL.

    This class handles the complete process of reading the schema file,
    connecting to the database, and applying the schema objects while
    providing detailed reporting.
    """

    def __init__(self, args):
        """Initialize the SchemaApplicator with command line arguments."""
        self.args = args
        self.connection = None

        # Database configuration
        self.db_config = {
            'host': os.environ.get('ECHO_BRAIN_DB_HOST', '192.168.50.135'),
            'port': int(os.environ.get('ECHO_BRAIN_DB_PORT', '5432')),
            'database': os.environ.get('ECHO_BRAIN_DB_NAME', 'echo_brain'),
            'user': os.environ.get('ECHO_BRAIN_DB_USER', 'patrick'),
            'password': os.environ.get('ECHO_BRAIN_DB_PASSWORD', os.getenv('DB_PASSWORD', ''))
        }

        # Schema file path
        self.schema_file = os.path.join(
            os.path.dirname(__file__),
            '..', 'src', 'autonomous', 'schema.sql'
        )

        # Results tracking
        self.results = {
            'tables_created': [],
            'tables_skipped': [],
            'indexes_created': [],
            'indexes_skipped': [],
            'functions_created': [],
            'functions_skipped': [],
            'triggers_created': [],
            'triggers_skipped': [],
            'comments_added': [],
            'errors': []
        }

        # Set logging level based on verbosity
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        logger.info("SchemaApplicator initialized")

    async def connect(self) -> bool:
        """
        Connect to the PostgreSQL database.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.connection = await asyncpg.connect(**self.db_config)
            logger.info(f"Connected to database {self.db_config['database']} on {self.db_config['host']}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False

    async def disconnect(self):
        """Disconnect from the database."""
        if self.connection:
            await self.connection.close()
            logger.info("Disconnected from database")

    def read_schema_file(self) -> Optional[str]:
        """
        Read the schema SQL file.

        Returns:
            Optional[str]: Schema SQL content or None if failed
        """
        try:
            if not os.path.exists(self.schema_file):
                logger.error(f"Schema file not found: {self.schema_file}")
                return None

            with open(self.schema_file, 'r') as f:
                content = f.read()

            logger.info(f"Read schema file: {self.schema_file} ({len(content)} characters)")
            return content

        except Exception as e:
            logger.error(f"Failed to read schema file: {e}")
            return None

    def parse_schema_statements(self, schema_content: str) -> Dict[str, List[str]]:
        """
        Parse schema content into categorized SQL statements.

        Args:
            schema_content: Raw SQL schema content

        Returns:
            Dict containing categorized SQL statements
        """
        statements = {
            'tables': [],
            'indexes': [],
            'functions': [],
            'triggers': [],
            'comments': []
        }

        # Split by statements (semicolon followed by whitespace or end of line)
        raw_statements = re.split(r';\s*\n', schema_content)

        for stmt in raw_statements:
            stmt = stmt.strip()
            if not stmt or stmt.startswith('--'):
                continue

            # Categorize statements
            if re.match(r'CREATE\s+TABLE', stmt, re.IGNORECASE):
                statements['tables'].append(stmt + ';')
            elif re.match(r'CREATE\s+.*INDEX', stmt, re.IGNORECASE):
                statements['indexes'].append(stmt + ';')
            elif re.match(r'CREATE\s+.*FUNCTION', stmt, re.IGNORECASE):
                statements['functions'].append(stmt + ';')
            elif re.match(r'CREATE\s+TRIGGER', stmt, re.IGNORECASE):
                statements['triggers'].append(stmt + ';')
            elif re.match(r'COMMENT\s+ON', stmt, re.IGNORECASE):
                statements['comments'].append(stmt + ';')

        logger.debug(f"Parsed schema into {sum(len(stmts) for stmts in statements.values())} statements")
        return statements

    async def object_exists(self, object_type: str, object_name: str,
                           schema_name: str = 'public') -> bool:
        """
        Check if a database object exists.

        Args:
            object_type: Type of object ('table', 'index', 'function', 'trigger')
            object_name: Name of the object
            schema_name: Schema name (default 'public')

        Returns:
            bool: True if object exists, False otherwise
        """
        try:
            if object_type == 'table':
                query = """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = $1 AND table_name = $2
                    )
                """
            elif object_type == 'index':
                query = """
                    SELECT EXISTS (
                        SELECT FROM pg_indexes
                        WHERE schemaname = $1 AND indexname = $2
                    )
                """
            elif object_type == 'function':
                query = """
                    SELECT EXISTS (
                        SELECT FROM information_schema.routines
                        WHERE routine_schema = $1 AND routine_name = $2
                    )
                """
            elif object_type == 'trigger':
                # For triggers, we need the table name too
                query = """
                    SELECT EXISTS (
                        SELECT FROM information_schema.triggers
                        WHERE trigger_schema = $1 AND trigger_name = $2
                    )
                """
            else:
                return False

            result = await self.connection.fetchval(query, schema_name, object_name)
            return bool(result)

        except Exception as e:
            logger.error(f"Error checking if {object_type} '{object_name}' exists: {e}")
            return False

    def extract_object_name(self, sql_statement: str, object_type: str) -> Optional[str]:
        """
        Extract object name from SQL statement.

        Args:
            sql_statement: SQL CREATE statement
            object_type: Type of object being created

        Returns:
            Optional[str]: Object name or None if not found
        """
        try:
            if object_type == 'table':
                match = re.search(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)',
                                sql_statement, re.IGNORECASE)
            elif object_type == 'index':
                match = re.search(r'CREATE\s+.*INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)',
                                sql_statement, re.IGNORECASE)
            elif object_type == 'function':
                match = re.search(r'CREATE\s+.*FUNCTION\s+(\w+)',
                                sql_statement, re.IGNORECASE)
            elif object_type == 'trigger':
                match = re.search(r'CREATE\s+TRIGGER\s+(\w+)',
                                sql_statement, re.IGNORECASE)
            else:
                return None

            return match.group(1) if match else None

        except Exception as e:
            logger.error(f"Error extracting {object_type} name: {e}")
            return None

    async def execute_statement(self, statement: str, object_type: str,
                               object_name: str) -> bool:
        """
        Execute a SQL statement with error handling.

        Args:
            statement: SQL statement to execute
            object_type: Type of object being created
            object_name: Name of the object

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.args.dry_run:
                logger.info(f"DRY-RUN: Would execute {object_type} creation for '{object_name}'")
                return True

            await self.connection.execute(statement)
            logger.debug(f"Successfully executed {object_type} creation for '{object_name}'")
            return True

        except Exception as e:
            error_msg = f"Failed to create {object_type} '{object_name}': {e}"
            logger.error(error_msg)
            self.results['errors'].append(error_msg)
            return False

    async def apply_statements(self, statements: List[str], object_type: str) -> Tuple[int, int]:
        """
        Apply a list of SQL statements of a specific type.

        Args:
            statements: List of SQL statements
            object_type: Type of objects being created

        Returns:
            Tuple[int, int]: (created_count, skipped_count)
        """
        created_count = 0
        skipped_count = 0

        for statement in statements:
            object_name = self.extract_object_name(statement, object_type)
            if not object_name:
                logger.warning(f"Could not extract {object_type} name from statement")
                continue

            # Check if object already exists (unless forcing recreation)
            if not self.args.force and await self.object_exists(object_type, object_name):
                logger.info(f"Skipping {object_type} '{object_name}' (already exists)")
                skipped_count += 1
                self.results[f'{object_type}s_skipped'].append(object_name)
                continue

            # Drop existing object if forcing recreation
            if self.args.force and await self.object_exists(object_type, object_name):
                drop_statement = f"DROP {object_type.upper()} IF EXISTS {object_name} CASCADE;"
                if not self.args.dry_run:
                    try:
                        await self.connection.execute(drop_statement)
                        logger.info(f"Dropped existing {object_type} '{object_name}'")
                    except Exception as e:
                        logger.warning(f"Could not drop {object_type} '{object_name}': {e}")

            # Execute creation statement
            if await self.execute_statement(statement, object_type, object_name):
                created_count += 1
                self.results[f'{object_type}s_created'].append(object_name)
                logger.info(f"Created {object_type} '{object_name}'")
            else:
                skipped_count += 1
                self.results[f'{object_type}s_skipped'].append(object_name)

        return created_count, skipped_count

    async def apply_comments(self, comment_statements: List[str]) -> int:
        """
        Apply COMMENT statements.

        Args:
            comment_statements: List of COMMENT SQL statements

        Returns:
            int: Number of comments applied
        """
        applied_count = 0

        for statement in comment_statements:
            try:
                if not self.args.dry_run:
                    await self.connection.execute(statement)
                applied_count += 1
                self.results['comments_added'].append(statement[:50] + "...")
                logger.debug(f"Applied comment: {statement[:50]}...")
            except Exception as e:
                logger.warning(f"Failed to apply comment: {e}")

        return applied_count

    async def create_backup(self) -> Optional[str]:
        """
        Create a backup of the database before applying schema.

        Returns:
            Optional[str]: Backup file path or None if failed
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"/tmp/autonomous_schema_backup_{timestamp}.sql"

            cmd = [
                "pg_dump",
                "-h", self.db_config['host'],
                "-p", str(self.db_config['port']),
                "-U", self.db_config['user'],
                "-d", self.db_config['database'],
                "--schema-only",
                "-f", backup_file
            ]

            # Set password via environment variable
            env = os.environ.copy()
            env['PGPASSWORD'] = self.db_config['password']

            if not self.args.dry_run:
                result = subprocess.run(cmd, env=env, capture_output=True, text=True)

                if result.returncode != 0:
                    logger.error(f"Backup failed: {result.stderr}")
                    return None

                logger.info(f"Created backup: {backup_file}")
                return backup_file
            else:
                logger.info(f"DRY-RUN: Would create backup at {backup_file}")
                return backup_file

        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None

    async def validate_schema(self) -> bool:
        """
        Validate that the schema was applied correctly.

        Returns:
            bool: True if validation passed, False otherwise
        """
        try:
            expected_tables = [
                'autonomous_goals',
                'autonomous_tasks',
                'autonomous_approvals',
                'autonomous_audit_log'
            ]

            for table in expected_tables:
                if not await self.object_exists('table', table):
                    logger.error(f"Validation failed: Table '{table}' not found")
                    return False

            # Check for essential indexes
            essential_indexes = [
                'idx_autonomous_goals_status',
                'idx_autonomous_tasks_goal_id',
                'idx_autonomous_audit_log_timestamp'
            ]

            missing_indexes = []
            for index in essential_indexes:
                if not await self.object_exists('index', index):
                    missing_indexes.append(index)

            if missing_indexes:
                logger.warning(f"Some indexes not found: {missing_indexes}")

            # Test basic operations
            await self.connection.execute("SELECT 1 FROM autonomous_goals LIMIT 0")
            await self.connection.execute("SELECT 1 FROM autonomous_tasks LIMIT 0")
            await self.connection.execute("SELECT 1 FROM autonomous_audit_log LIMIT 0")

            logger.info("Schema validation passed")
            return True

        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False

    def print_summary(self):
        """Print a summary of the schema application results."""
        print("\n" + "="*70)
        print("AUTONOMOUS SCHEMA APPLICATION SUMMARY")
        print("="*70)

        if self.args.dry_run:
            print("\n🔍 DRY RUN MODE - No changes were made")

        # Tables
        if self.results['tables_created']:
            print(f"\n✅ Tables Created ({len(self.results['tables_created'])}):")
            for table in self.results['tables_created']:
                print(f"   • {table}")

        if self.results['tables_skipped']:
            print(f"\n⏭️  Tables Skipped ({len(self.results['tables_skipped'])}):")
            for table in self.results['tables_skipped']:
                print(f"   • {table} (already exists)")

        # Indexes
        if self.results['indexes_created']:
            print(f"\n📊 Indexes Created ({len(self.results['indexes_created'])}):")
            for index in self.results['indexes_created']:
                print(f"   • {index}")

        if self.results['indexes_skipped']:
            print(f"\n⏭️  Indexes Skipped ({len(self.results['indexes_skipped'])}):")
            for index in self.results['indexes_skipped']:
                print(f"   • {index}")

        # Functions and Triggers
        if self.results['functions_created']:
            print(f"\n⚙️  Functions Created ({len(self.results['functions_created'])}):")
            for func in self.results['functions_created']:
                print(f"   • {func}")

        if self.results['triggers_created']:
            print(f"\n🔄 Triggers Created ({len(self.results['triggers_created'])}):")
            for trigger in self.results['triggers_created']:
                print(f"   • {trigger}")

        # Comments
        if self.results['comments_added']:
            print(f"\n📝 Comments Added: {len(self.results['comments_added'])}")

        # Errors
        if self.results['errors']:
            print(f"\n❌ Errors ({len(self.results['errors'])}):")
            for error in self.results['errors']:
                print(f"   • {error}")

        # Statistics
        total_created = (len(self.results['tables_created']) +
                        len(self.results['indexes_created']) +
                        len(self.results['functions_created']) +
                        len(self.results['triggers_created']))

        total_skipped = (len(self.results['tables_skipped']) +
                        len(self.results['indexes_skipped']) +
                        len(self.results['functions_skipped']) +
                        len(self.results['triggers_skipped']))

        print(f"\n📊 Summary:")
        print(f"   • Total Objects Created: {total_created}")
        print(f"   • Total Objects Skipped: {total_skipped}")
        print(f"   • Comments Applied: {len(self.results['comments_added'])}")
        print(f"   • Errors: {len(self.results['errors'])}")

        print(f"\n🔗 Database: {self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}")
        print(f"👤 User: {self.db_config['user']}")
        print(f"⏰ Applied at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n" + "="*70)

    async def run(self) -> bool:
        """
        Run the complete schema application process.

        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Starting autonomous schema application")

        try:
            # Read schema file
            schema_content = self.read_schema_file()
            if not schema_content:
                return False

            # Parse schema statements
            statements = self.parse_schema_statements(schema_content)

            # Connect to database
            if not await self.connect():
                return False

            # Validate-only mode
            if self.args.validate_only:
                success = await self.validate_schema()
                return success

            # Create backup if requested
            if self.args.backup:
                backup_file = await self.create_backup()
                if not backup_file:
                    logger.error("Backup creation failed, aborting")
                    return False

            # Apply schema objects in order
            logger.info("Applying tables...")
            await self.apply_statements(statements['tables'], 'table')

            logger.info("Applying indexes...")
            await self.apply_statements(statements['indexes'], 'index')

            logger.info("Applying functions...")
            await self.apply_statements(statements['functions'], 'function')

            logger.info("Applying triggers...")
            await self.apply_statements(statements['triggers'], 'trigger')

            logger.info("Applying comments...")
            await self.apply_comments(statements['comments'])

            # Validate schema after application
            if not self.args.dry_run:
                if not await self.validate_schema():
                    logger.error("Schema validation failed after application")
                    return False

            # Print summary
            self.print_summary()

            logger.info("Schema application completed successfully")
            return True

        except Exception as e:
            logger.error(f"Schema application failed: {e}")
            return False

        finally:
            await self.disconnect()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply autonomous core database schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run to see what would be done
    python apply_autonomous_schema.py --dry-run

    # Apply schema with backup
    python apply_autonomous_schema.py --backup

    # Force recreate existing objects
    python apply_autonomous_schema.py --force

    # Only validate existing schema
    python apply_autonomous_schema.py --validate-only

Environment Variables:
    ECHO_BRAIN_DB_HOST     : Database host (default: 192.168.50.135)
    ECHO_BRAIN_DB_PORT     : Database port (default: 5432)
    ECHO_BRAIN_DB_NAME     : Database name (default: tower_consolidated)
    ECHO_BRAIN_DB_USER     : Database user (default: patrick)
    ECHO_BRAIN_DB_PASSWORD : Database password (required)
        """
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )

    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create backup before applying schema'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Drop and recreate existing objects'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing schema without applying changes'
    )

    return parser.parse_args()


async def main():
    """Main entry point for the schema application script."""
    args = parse_arguments()

    print("Echo Brain Autonomous Schema Applicator")
    print("=" * 42)

    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made\n")

    applicator = SchemaApplicator(args)
    success = await applicator.run()

    if success:
        if args.validate_only:
            print("\n✅ Schema validation passed!")
        elif args.dry_run:
            print("\n✅ Dry run completed successfully!")
        else:
            print("\n✅ Schema applied successfully!")
        return 0
    else:
        print("\n❌ Schema application failed. Check logs for details.")
        return 1


if __name__ == "__main__":
    # Run the schema applicator
    exit_code = asyncio.run(main())
    sys.exit(exit_code)