#!/usr/bin/env python3
"""
Database Migration Script for Echo Brain
Migrates Echo Brain to use tower_consolidated database with proper schema
"""

import psycopg2
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EchoBrainDatabaseMigration:
    """Handles database migration and setup for Echo Brain"""

    def __init__(self):
        self.db_config = {
            "host": "localhost",
            "database": "tower_consolidated",
            "user": "patrick",
            "password": "***REMOVED***",
            "port": 5432
        }

    def test_connection(self):
        """Test database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            cur.execute("SELECT version()")
            version = cur.fetchone()
            logger.info(f"✅ Connected to PostgreSQL: {version[0][:50]}...")
            cur.close()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False

    def check_existing_tables(self):
        """Check what tables already exist"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            # Check for Echo tables
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name LIKE 'echo_%'
                ORDER BY table_name;
            """)

            echo_tables = cur.fetchall()

            if echo_tables:
                logger.info(f"Found {len(echo_tables)} existing Echo tables:")
                for table in echo_tables:
                    logger.info(f"  - {table[0]}")
            else:
                logger.info("No existing Echo tables found")

            # Check for anime tables
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name LIKE 'anime_%'
                ORDER BY table_name;
            """)

            anime_tables = cur.fetchall()

            if anime_tables:
                logger.info(f"Found {len(anime_tables)} anime tables that can be linked:")
                for table in anime_tables:
                    logger.info(f"  - {table[0]}")

            cur.close()
            conn.close()
            return echo_tables, anime_tables

        except Exception as e:
            logger.error(f"Error checking tables: {e}")
            return [], []

    def apply_schema(self):
        """Apply the Echo Brain schema"""
        schema_file = Path(__file__).parent / "echo_brain_complete_schema.sql"

        if not schema_file.exists():
            logger.error(f"Schema file not found: {schema_file}")
            return False

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            logger.info("Applying Echo Brain schema...")

            # Read and execute schema
            with open(schema_file, 'r') as f:
                schema_sql = f.read()

            # Execute the schema
            cur.execute(schema_sql)
            conn.commit()

            logger.info("✅ Schema applied successfully")

            # Verify tables were created
            cur.execute("""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name LIKE 'echo_%'
            """)

            table_count = cur.fetchone()[0]
            logger.info(f"✅ Created/verified {table_count} Echo tables")

            cur.close()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"❌ Schema application failed: {e}")
            if conn:
                conn.rollback()
                conn.close()
            return False

    def update_configuration(self):
        """Update Echo Brain configuration to use tower_consolidated"""
        env_file = Path("/opt/tower-echo-brain/.env")

        logger.info("Updating Echo Brain configuration...")

        # Create/update .env file
        env_content = f"""# Echo Brain Database Configuration
DB_NAME=tower_consolidated
DB_USER=patrick
DB_PASSWORD=***REMOVED***
DB_HOST=localhost
DB_PORT=5432

# Use tower_consolidated for all Echo Brain operations
USE_VAULT=false

# Qdrant vector database
QDRANT_HOST=localhost
QDRANT_PORT=6333
"""

        try:
            with open(env_file, 'w') as f:
                f.write(env_content)
            logger.info(f"✅ Updated configuration at {env_file}")
        except Exception as e:
            logger.error(f"❌ Failed to update configuration: {e}")
            return False

        # Also update the database.py fallback defaults
        database_py = Path("/opt/tower-echo-brain/src/db/database.py")
        if database_py.exists():
            try:
                content = database_py.read_text()
                # Update the default database name
                content = content.replace(
                    'os.environ.get("DB_NAME", "echo_brain")',
                    'os.environ.get("DB_NAME", "tower_consolidated")'
                )
                database_py.write_text(content)
                logger.info("✅ Updated database.py defaults")
            except Exception as e:
                logger.warning(f"⚠️ Could not update database.py: {e}")

        return True

    def migrate_existing_data(self):
        """Migrate any existing data from other databases"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            # Check if there's any anime conversation data to migrate
            cur.execute("""
                SELECT COUNT(*) FROM anime_conversation_context
                WHERE conversation_id IS NOT NULL
            """)
            result = cur.fetchone()

            if result and result[0] > 0:
                logger.info(f"Found {result[0]} anime conversations to link")

                # Link anime conversations
                cur.execute("""
                    INSERT INTO echo_conversations (conversation_id, username, created_at, context)
                    SELECT
                        conversation_id,
                        'patrick',
                        created_at,
                        jsonb_build_object(
                            'source', 'anime_system',
                            'project_id', project_id,
                            'character_mentioned', character_mentioned
                        )
                    FROM anime_conversation_context
                    WHERE conversation_id IS NOT NULL
                    ON CONFLICT (conversation_id) DO UPDATE SET
                        context = echo_conversations.context ||
                                 jsonb_build_object('anime_linked', true),
                        last_interaction = CURRENT_TIMESTAMP
                """)

                migrated = cur.rowcount
                conn.commit()
                logger.info(f"✅ Linked {migrated} anime conversations")

            cur.close()
            conn.close()
            return True

        except Exception as e:
            logger.warning(f"⚠️ Data migration skipped: {e}")
            return True  # Not critical if migration fails

    def verify_setup(self):
        """Verify the database is properly set up"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            # Test inserting a conversation
            test_id = "test_migration_verify"
            cur.execute("""
                INSERT INTO echo_conversations (conversation_id, username)
                VALUES (%s, 'test_user')
                ON CONFLICT (conversation_id) DO UPDATE SET
                    last_interaction = CURRENT_TIMESTAMP
                RETURNING id
            """, (test_id,))

            result = cur.fetchone()
            if result:
                logger.info("✅ Test insert successful")

                # Clean up test
                cur.execute("""
                    DELETE FROM echo_conversations
                    WHERE conversation_id = %s
                """, (test_id,))

            conn.commit()
            cur.close()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            return False

    def run_migration(self):
        """Run the complete migration process"""
        logger.info("="*60)
        logger.info("Echo Brain Database Migration")
        logger.info("="*60)

        # Step 1: Test connection
        logger.info("\nStep 1: Testing database connection...")
        if not self.test_connection():
            logger.error("Cannot proceed without database connection")
            return False

        # Step 2: Check existing tables
        logger.info("\nStep 2: Checking existing tables...")
        echo_tables, anime_tables = self.check_existing_tables()

        # Step 3: Apply schema
        logger.info("\nStep 3: Applying Echo Brain schema...")
        if not self.apply_schema():
            logger.error("Schema application failed")
            return False

        # Step 4: Update configuration
        logger.info("\nStep 4: Updating configuration...")
        if not self.update_configuration():
            logger.error("Configuration update failed")
            return False

        # Step 5: Migrate existing data
        if anime_tables:
            logger.info("\nStep 5: Migrating existing data...")
            self.migrate_existing_data()
        else:
            logger.info("\nStep 5: No existing data to migrate")

        # Step 6: Verify setup
        logger.info("\nStep 6: Verifying setup...")
        if not self.verify_setup():
            logger.error("Verification failed")
            return False

        logger.info("\n" + "="*60)
        logger.info("✅ Migration completed successfully!")
        logger.info("Echo Brain is now using tower_consolidated database")
        logger.info("="*60)

        # Print next steps
        logger.info("\nNext steps:")
        logger.info("1. Restart Echo Brain service: sudo systemctl restart tower-echo-brain")
        logger.info("2. Test the memory system: python /opt/tower-echo-brain/tests/quick_memory_test.py")
        logger.info("3. Run full test suite: python /opt/tower-echo-brain/tests/run_memory_tests.py")

        return True


if __name__ == "__main__":
    migration = EchoBrainDatabaseMigration()
    success = migration.run_migration()
    sys.exit(0 if success else 1)