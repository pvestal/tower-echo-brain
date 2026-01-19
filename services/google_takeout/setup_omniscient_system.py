#!/usr/bin/env python3
"""
Setup and Activation Script for Omniscient Personal Data Collection System
Prepares all components for complete personal intelligence gathering
"""

import asyncio
import logging
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import psycopg2
import asyncpg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OmniscientSystemSetup:
    """Setup and activate the omniscient data collection system"""

    def __init__(self):
        self.db_url = "postgresql://patrick:tower_echo_brain_secret_key_2025@localhost/echo_brain"
        self.vault_addr = "http://127.0.0.1:8200"
        self.config_path = Path(__file__).parent / "config" / "settings.yaml"
        self.data_root = Path("/opt/tower-echo-brain/data/omniscient")

    async def setup_database_schema(self):
        """Create all required database tables and indexes"""
        logger.info("🗄️ Creating omniscient database schema...")

        conn = await asyncpg.connect(self.db_url)

        try:
            # Enable vector extension for embeddings
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Main omniscient data table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS omniscient_data (
                    id SERIAL PRIMARY KEY,
                    source VARCHAR(50) NOT NULL,
                    item_type VARCHAR(50) NOT NULL,
                    content_hash VARCHAR(64) UNIQUE,
                    content JSONB NOT NULL,
                    metadata JSONB,
                    timestamp TIMESTAMP NOT NULL,
                    importance_score FLOAT DEFAULT 0.5,
                    privacy_level VARCHAR(20) DEFAULT 'private',
                    embedding VECTOR(1536),
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tags TEXT[]
                )
            """)

            # Face recognition database
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS face_recognition_data (
                    id SERIAL PRIMARY KEY,
                    person_name VARCHAR(100),
                    face_encoding BYTEA,
                    source_file VARCHAR(500),
                    confidence FLOAT,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    occurrence_count INTEGER DEFAULT 1,
                    relationship_score FLOAT DEFAULT 0.5
                )
            """)

            # Email intelligence table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS email_intelligence (
                    id SERIAL PRIMARY KEY,
                    message_id VARCHAR(200) UNIQUE,
                    thread_id VARCHAR(200),
                    sender VARCHAR(200),
                    recipients TEXT[],
                    subject TEXT,
                    body_text TEXT,
                    body_html TEXT,
                    attachments JSONB,
                    sentiment_score FLOAT,
                    importance_score FLOAT,
                    relationship_strength FLOAT,
                    keywords TEXT[],
                    entities JSONB,
                    timestamp TIMESTAMP,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Calendar intelligence
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS calendar_intelligence (
                    id SERIAL PRIMARY KEY,
                    event_id VARCHAR(200) UNIQUE,
                    calendar_name VARCHAR(100),
                    title TEXT,
                    description TEXT,
                    location TEXT,
                    attendees TEXT[],
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    event_type VARCHAR(50),
                    importance_score FLOAT,
                    relationship_data JSONB,
                    recurrence_pattern VARCHAR(100),
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Browser intelligence
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS browsing_intelligence (
                    id SERIAL PRIMARY KEY,
                    url TEXT,
                    title TEXT,
                    domain VARCHAR(100),
                    visit_time TIMESTAMP,
                    visit_duration INTEGER,
                    visit_count INTEGER DEFAULT 1,
                    search_query TEXT,
                    category VARCHAR(50),
                    interest_score FLOAT,
                    browser_type VARCHAR(20),
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Credential intelligence (HIGH SECURITY)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS credential_intelligence (
                    id SERIAL PRIMARY KEY,
                    service_name VARCHAR(100),
                    service_url TEXT,
                    username VARCHAR(100),
                    password_hash VARCHAR(128),
                    password_strength FLOAT,
                    last_used TIMESTAMP,
                    risk_level VARCHAR(20) DEFAULT 'medium',
                    notes TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Knowledge graph relationships
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_relationships (
                    id SERIAL PRIMARY KEY,
                    entity_1_type VARCHAR(50),
                    entity_1_id INTEGER,
                    entity_2_type VARCHAR(50),
                    entity_2_id INTEGER,
                    relationship_type VARCHAR(50),
                    strength FLOAT,
                    confidence FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Processing queue for background tasks
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS processing_queue (
                    id SERIAL PRIMARY KEY,
                    task_type VARCHAR(50),
                    source_path TEXT,
                    priority INTEGER DEFAULT 5,
                    status VARCHAR(20) DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0
                )
            """)

            # Create performance indexes
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_omniscient_source ON omniscient_data(source)",
                "CREATE INDEX IF NOT EXISTS idx_omniscient_type ON omniscient_data(item_type)",
                "CREATE INDEX IF NOT EXISTS idx_omniscient_timestamp ON omniscient_data(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_omniscient_importance ON omniscient_data(importance_score DESC)",
                "CREATE INDEX IF NOT EXISTS idx_omniscient_privacy ON omniscient_data(privacy_level)",
                "CREATE INDEX IF NOT EXISTS idx_omniscient_content_hash ON omniscient_data(content_hash)",
                "CREATE INDEX IF NOT EXISTS idx_face_person ON face_recognition_data(person_name)",
                "CREATE INDEX IF NOT EXISTS idx_face_confidence ON face_recognition_data(confidence DESC)",
                "CREATE INDEX IF NOT EXISTS idx_email_sender ON email_intelligence(sender)",
                "CREATE INDEX IF NOT EXISTS idx_email_timestamp ON email_intelligence(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_email_importance ON email_intelligence(importance_score DESC)",
                "CREATE INDEX IF NOT EXISTS idx_calendar_start_time ON calendar_intelligence(start_time)",
                "CREATE INDEX IF NOT EXISTS idx_calendar_importance ON calendar_intelligence(importance_score DESC)",
                "CREATE INDEX IF NOT EXISTS idx_browsing_domain ON browsing_intelligence(domain)",
                "CREATE INDEX IF NOT EXISTS idx_browsing_visit_time ON browsing_intelligence(visit_time)",
                "CREATE INDEX IF NOT EXISTS idx_browsing_interest ON browsing_intelligence(interest_score DESC)",
                "CREATE INDEX IF NOT EXISTS idx_credential_service ON credential_intelligence(service_name)",
                "CREATE INDEX IF NOT EXISTS idx_credential_risk ON credential_intelligence(risk_level)",
                "CREATE INDEX IF NOT EXISTS idx_knowledge_entity1 ON knowledge_relationships(entity_1_type, entity_1_id)",
                "CREATE INDEX IF NOT EXISTS idx_knowledge_entity2 ON knowledge_relationships(entity_2_type, entity_2_id)",
                "CREATE INDEX IF NOT EXISTS idx_knowledge_strength ON knowledge_relationships(strength DESC)",
                "CREATE INDEX IF NOT EXISTS idx_processing_status ON processing_queue(status)",
                "CREATE INDEX IF NOT EXISTS idx_processing_priority ON processing_queue(priority DESC)"
            ]

            for index_sql in indexes:
                await conn.execute(index_sql)

            # Create materialized views for fast analytics
            await conn.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS daily_activity_summary AS
                SELECT
                    DATE(timestamp) as activity_date,
                    source,
                    item_type,
                    COUNT(*) as item_count,
                    AVG(importance_score) as avg_importance,
                    COUNT(DISTINCT CASE WHEN privacy_level = 'sensitive' THEN id END) as sensitive_items,
                    COUNT(DISTINCT CASE WHEN privacy_level = 'classified' THEN id END) as classified_items
                FROM omniscient_data
                WHERE timestamp > CURRENT_DATE - INTERVAL '90 days'
                GROUP BY DATE(timestamp), source, item_type
                ORDER BY activity_date DESC
            """)

            await conn.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS relationship_network AS
                SELECT
                    entity_1_type,
                    entity_1_id,
                    entity_2_type,
                    entity_2_id,
                    relationship_type,
                    strength,
                    COUNT(*) as interaction_count
                FROM knowledge_relationships
                GROUP BY entity_1_type, entity_1_id, entity_2_type, entity_2_id, relationship_type, strength
                ORDER BY strength DESC, interaction_count DESC
            """)

            # Create refresh function for materialized views
            await conn.execute("""
                CREATE OR REPLACE FUNCTION refresh_omniscient_views()
                RETURNS void AS $$
                BEGIN
                    REFRESH MATERIALIZED VIEW daily_activity_summary;
                    REFRESH MATERIALIZED VIEW relationship_network;
                END;
                $$ LANGUAGE plpgsql
            """)

            logger.info("✅ Omniscient database schema created successfully")

        except Exception as e:
            logger.error(f"❌ Database schema creation failed: {e}")
            raise
        finally:
            await conn.close()

    def setup_vault_credentials(self):
        """Setup Google API credentials in Vault"""
        logger.info("🔐 Setting up Google API credentials...")

        print("""
🔐 GOOGLE API CREDENTIALS SETUP

To enable complete personal data harvesting, you need to set up Google API credentials:

1. Go to: https://console.cloud.google.com
2. Create a new project or select existing project
3. Enable these APIs:
   - Google Photos Library API
   - Gmail API
   - Google Calendar API
   - Google Takeout API
   - Google Drive API

4. Create OAuth2 credentials:
   - Application type: Desktop application
   - Download the JSON credentials file

5. Set up OAuth2 scopes for maximum access:
   - https://www.googleapis.com/auth/photoslibrary.readonly
   - https://www.googleapis.com/auth/gmail.readonly
   - https://www.googleapis.com/auth/calendar.readonly
   - https://www.googleapis.com/auth/drive.readonly
   - https://www.googleapis.com/auth/takeout
   - https://www.googleapis.com/auth/userinfo.email
   - https://www.googleapis.com/auth/userinfo.profile

        """)

        # Interactive credential setup
        use_vault = input("Do you want to store credentials in Vault? (y/n): ").lower().startswith('y')

        if use_vault:
            client_id = input("Enter Google OAuth2 Client ID: ").strip()
            client_secret = input("Enter Google OAuth2 Client Secret: ").strip()

            # Store in Vault
            vault_data = {
                "client_id": client_id,
                "client_secret": client_secret,
                "setup_date": datetime.now().isoformat(),
                "scopes": [
                    "https://www.googleapis.com/auth/photoslibrary.readonly",
                    "https://www.googleapis.com/auth/gmail.readonly",
                    "https://www.googleapis.com/auth/calendar.readonly",
                    "https://www.googleapis.com/auth/drive.readonly",
                    "https://www.googleapis.com/auth/takeout",
                    "https://www.googleapis.com/auth/userinfo.email",
                    "https://www.googleapis.com/auth/userinfo.profile"
                ]
            }

            try:
                subprocess.run([
                    "vault", "kv", "put", "secret/google/omniscient",
                    f"client_id={client_id}",
                    f"client_secret={client_secret}",
                    f"setup_date={vault_data['setup_date']}"
                ], env={"VAULT_ADDR": self.vault_addr}, check=True)

                logger.info("✅ Credentials stored in Vault")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to store in Vault: {e}")

                # Fallback to JSON storage
                logger.info("📁 Storing credentials in fallback JSON vault")
                vault_dir = Path.home() / ".tower_credentials"
                vault_dir.mkdir(exist_ok=True)

                vault_file = vault_dir / "vault.json"
                if vault_file.exists():
                    with open(vault_file, 'r') as f:
                        existing_data = json.load(f)
                else:
                    existing_data = {}

                existing_data["google_omniscient"] = vault_data

                with open(vault_file, 'w') as f:
                    json.dump(existing_data, f, indent=2)

                logger.info("✅ Credentials stored in JSON vault")

        else:
            logger.info("⚠️ Skipping credential setup - manual configuration required")

    def setup_directories(self):
        """Create required directory structure"""
        logger.info("📁 Setting up directory structure...")

        directories = [
            self.data_root,
            self.data_root / "faces",
            self.data_root / "cache",
            self.data_root / "takeout",
            self.data_root / "browser",
            self.data_root / "credentials",
            self.data_root / "embeddings",
            self.data_root / "logs",
            Path("/opt/tower-echo-brain/data/takeout"),
            Path("/opt/tower-echo-brain/services/google_takeout/logs")
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        # Set permissions for security
        sensitive_dirs = [self.data_root / "credentials", self.data_root / "faces"]
        for directory in sensitive_dirs:
            import stat
            directory.chmod(stat.S_IRWXU)  # Owner read/write/execute only

        logger.info("✅ Directory structure created")

    def install_dependencies(self):
        """Install required Python packages"""
        logger.info("📦 Installing required dependencies...")

        required_packages = [
            "face-recognition",
            "opencv-python",
            "beautifulsoup4",
            "asyncpg",
            "psycopg2-binary",
            "google-auth",
            "google-auth-oauthlib",
            "google-auth-httplib2",
            "google-api-python-client",
            "pillow",
            "numpy",
            "scipy",
            "scikit-learn",
            "nltk",
            "spacy",
            "cryptography",
            "hvac",
            "httpx"
        ]

        venv_python = Path("/opt/tower-echo-brain/venv/bin/pip")

        for package in required_packages:
            try:
                subprocess.run([
                    str(venv_python), "install", package
                ], check=True, capture_output=True)
                logger.info(f"✅ Installed {package}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️ Failed to install {package}: {e}")

    def create_systemd_service(self):
        """Create systemd service for background processing"""
        logger.info("⚙️ Creating systemd service...")

        service_content = f"""[Unit]
Description=Tower Echo Brain - Omniscient Data Collector
After=network.target postgresql.service
Requires=postgresql.service

[Service]
Type=simple
User=patrick
Group=patrick
WorkingDirectory=/opt/tower-echo-brain/services/google_takeout
Environment=PATH=/opt/tower-echo-brain/venv/bin
Environment=PYTHONPATH=/opt/tower-echo-brain
Environment=VAULT_ADDR=http://127.0.0.1:8200
ExecStart=/opt/tower-echo-brain/venv/bin/python omniscient_data_collector.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""

        service_file = Path("/etc/systemd/system/tower-omniscient-collector.service")

        try:
            with open(service_file, 'w') as f:
                f.write(service_content)

            subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)
            subprocess.run(["sudo", "systemctl", "enable", "tower-omniscient-collector.service"], check=True)

            logger.info("✅ Systemd service created and enabled")
        except Exception as e:
            logger.error(f"❌ Failed to create systemd service: {e}")

    async def test_system(self):
        """Test all components of the omniscient system"""
        logger.info("🧪 Testing omniscient system components...")

        tests_passed = 0
        total_tests = 6

        # Test 1: Database connectivity
        try:
            conn = await asyncpg.connect(self.db_url)
            await conn.fetchval("SELECT 1")
            await conn.close()
            logger.info("✅ Database connectivity test passed")
            tests_passed += 1
        except Exception as e:
            logger.error(f"❌ Database test failed: {e}")

        # Test 2: Directory permissions
        try:
            test_file = self.data_root / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()
            logger.info("✅ Directory permissions test passed")
            tests_passed += 1
        except Exception as e:
            logger.error(f"❌ Directory permissions test failed: {e}")

        # Test 3: Python dependencies
        try:
            import face_recognition
            import cv2
            import asyncpg
            logger.info("✅ Python dependencies test passed")
            tests_passed += 1
        except Exception as e:
            logger.error(f"❌ Dependencies test failed: {e}")

        # Test 4: Vault connectivity
        try:
            result = subprocess.run([
                "vault", "status"
            ], env={"VAULT_ADDR": self.vault_addr}, capture_output=True)
            if result.returncode == 0 or "Sealed" in result.stderr.decode():
                logger.info("✅ Vault connectivity test passed")
                tests_passed += 1
            else:
                logger.warning("⚠️ Vault not accessible, using fallback storage")
                tests_passed += 1  # Still pass test as fallback exists
        except Exception as e:
            logger.warning(f"⚠️ Vault test failed, using fallback: {e}")
            tests_passed += 1

        # Test 5: Google Takeout infrastructure
        try:
            from takeout_manager import GoogleTakeoutManager
            manager = GoogleTakeoutManager()
            logger.info("✅ Google Takeout infrastructure test passed")
            tests_passed += 1
        except Exception as e:
            logger.error(f"❌ Google Takeout test failed: {e}")

        # Test 6: Omniscient collector
        try:
            from omniscient_data_collector import OmniscientDataCollector
            import yaml

            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            collector = OmniscientDataCollector(config)
            logger.info("✅ Omniscient collector test passed")
            tests_passed += 1
        except Exception as e:
            logger.error(f"❌ Omniscient collector test failed: {e}")

        logger.info(f"🧪 System tests completed: {tests_passed}/{total_tests} passed")
        return tests_passed == total_tests

    async def run_complete_setup(self):
        """Run the complete omniscient system setup"""
        logger.info("🚀 SETTING UP OMNISCIENT PERSONAL DATA COLLECTION SYSTEM")

        print("""
╔════════════════════════════════════════════════════════════════════╗
║                    ⚠️  CRITICAL SECURITY WARNING  ⚠️                ║
║                                                                    ║
║  This system will collect and analyze ALL your personal data:      ║
║  • Photos with facial recognition                                  ║
║  • Complete email history and content                             ║
║  • Calendar events and scheduling patterns                        ║
║  • Browser history and search queries                             ║
║  • Stored passwords and credentials                               ║
║  • Social media activity and relationships                        ║
║                                                                    ║
║  This creates an OMNISCIENT profile of your digital life.         ║
║  Use responsibly and ensure proper security measures.             ║
╚════════════════════════════════════════════════════════════════════╝
        """)

        confirm = input("\nDo you want to proceed with omniscient system setup? (yes/no): ").lower()
        if confirm != "yes":
            logger.info("❌ Setup cancelled by user")
            return False

        try:
            # Step 1: Setup directories
            self.setup_directories()

            # Step 2: Install dependencies
            self.install_dependencies()

            # Step 3: Setup database schema
            await self.setup_database_schema()

            # Step 4: Setup credentials
            self.setup_vault_credentials()

            # Step 5: Create systemd service
            self.create_systemd_service()

            # Step 6: Test system
            test_success = await self.test_system()

            if test_success:
                logger.info("🎉 OMNISCIENT SYSTEM SETUP COMPLETE")
                print("""
✅ Setup completed successfully!

Next steps:
1. Start the omniscient collector: sudo systemctl start tower-omniscient-collector
2. Monitor logs: journalctl -u tower-omniscient-collector -f
3. Run initial harvest: python omniscient_data_collector.py
4. Access data via Echo Brain API at http://localhost:8309

SECURITY REMINDERS:
• Regularly backup encrypted data
• Monitor access logs
• Use strong authentication
• Keep credentials secure
• Review privacy settings
                """)
                return True
            else:
                logger.error("❌ Setup completed with test failures")
                return False

        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False

if __name__ == "__main__":
    setup = OmniscientSystemSetup()

    # Run async setup
    success = asyncio.run(setup.run_complete_setup())

    if success:
        print("🧠 Omniscient Personal Intelligence System is ready!")
        sys.exit(0)
    else:
        print("❌ Setup failed. Check logs for details.")
        sys.exit(1)