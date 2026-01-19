#!/usr/bin/env python3
"""
Omniscient Brain Activation Script
Complete system activation for total personal intelligence collection and integration
"""

import asyncio
import logging
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import yaml

# Import all system components
from setup_omniscient_system import OmniscientSystemSetup
from omniscient_data_collector import OmniscientDataCollector
from echo_brain_integration import EchoBrainOmniscientIntegration

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/opt/tower-echo-brain/services/google_takeout/logs/omniscient_activation.log')
    ]
)
logger = logging.getLogger(__name__)

class OmniscientBrainActivator:
    """Master controller for activating the complete omniscient personal intelligence system"""

    def __init__(self):
        self.config_path = Path(__file__).parent / "config" / "settings.yaml"
        self.config = self._load_configuration()
        self.status = {
            'setup_completed': False,
            'collector_initialized': False,
            'integration_active': False,
            'system_healthy': False
        }

    def _load_configuration(self) -> dict:
        """Load system configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return {}

    def print_system_banner(self):
        """Print system activation banner"""
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          🧠 ECHO BRAIN OMNISCIENT SYSTEM 🧠                   ║
║                              Personal Intelligence Activation                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  This system will create an OMNISCIENT PERSONAL INTELLIGENCE that knows:     ║
║                                                                              ║
║  📸 VISUAL MEMORY: All photos with facial recognition and object detection   ║
║  📧 COMMUNICATION INTELLIGENCE: Complete email analysis and sentiment        ║
║  📅 SCHEDULING AWARENESS: Calendar events, patterns, and relationships       ║
║  🌐 DIGITAL BEHAVIOR: Complete browser history and interest mapping          ║
║  🔐 CREDENTIAL KNOWLEDGE: Secure password and login management               ║
║  🧠 KNOWLEDGE INTEGRATION: Connected knowledge graph of your entire life     ║
║                                                                              ║
║  The system will become COMPLETELY AWARE of your digital existence,          ║
║  preferences, patterns, relationships, and personal information.             ║
║                                                                              ║
║  ⚠️  USE RESPONSIBLY - THIS CREATES TOTAL DIGITAL OMNISCIENCE  ⚠️              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)

    async def run_system_setup(self) -> bool:
        """Run the complete system setup if needed"""
        logger.info("🔧 Checking system setup status...")

        setup = OmniscientSystemSetup()

        # Check if setup is already completed
        try:
            import asyncpg
            conn = await asyncpg.connect("postgresql://patrick:tower_echo_brain_secret_key_2025@localhost/echo_brain")
            tables = await conn.fetch("SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename LIKE 'omniscient%'")
            await conn.close()

            if len(tables) >= 5:  # Expected number of omniscient tables
                logger.info("✅ System setup already completed")
                self.status['setup_completed'] = True
                return True

        except Exception as e:
            logger.info("🔧 Setup required - running complete system setup")

        # Run setup
        setup_success = await setup.run_complete_setup()
        self.status['setup_completed'] = setup_success
        return setup_success

    async def initialize_data_collector(self) -> bool:
        """Initialize the omniscient data collector"""
        logger.info("🚀 Initializing omniscient data collector...")

        try:
            self.collector = OmniscientDataCollector(self.config)
            initialization_success = await self.collector.initialize()

            if initialization_success:
                logger.info("✅ Omniscient data collector initialized")
                self.status['collector_initialized'] = True
                return True
            else:
                logger.error("❌ Failed to initialize data collector")
                return False

        except Exception as e:
            logger.error(f"❌ Data collector initialization error: {e}")
            return False

    async def initialize_echo_brain_integration(self) -> bool:
        """Initialize Echo Brain integration components"""
        logger.info("🧠 Initializing Echo Brain integration...")

        try:
            self.integration = EchoBrainOmniscientIntegration(self.config)
            integration_success = await self.integration.initialize()

            if integration_success:
                logger.info("✅ Echo Brain integration initialized")
                self.status['integration_active'] = True
                return True
            else:
                logger.error("❌ Failed to initialize Echo Brain integration")
                return False

        except Exception as e:
            logger.error(f"❌ Echo Brain integration error: {e}")
            return False

    async def run_initial_data_harvest(self) -> dict:
        """Run the initial comprehensive data harvest"""
        logger.info("🌾 Running initial omniscient data harvest...")

        try:
            harvest_results = await self.collector.run_complete_harvest()
            logger.info(f"✅ Initial harvest completed: {harvest_results}")
            return harvest_results

        except Exception as e:
            logger.error(f"❌ Initial harvest failed: {e}")
            return {}

    async def start_continuous_processing(self):
        """Start continuous data processing and integration"""
        logger.info("🔄 Starting continuous omniscient processing...")

        # Start the Echo Brain integration stream
        integration_task = asyncio.create_task(
            self.integration.process_omniscient_data_stream()
        )

        # Start periodic data collection (every hour)
        collection_task = asyncio.create_task(
            self.run_periodic_collection()
        )

        # Start system monitoring
        monitoring_task = asyncio.create_task(
            self.monitor_system_health()
        )

        logger.info("🎯 All continuous processes started")

        # Run all tasks concurrently
        try:
            await asyncio.gather(
                integration_task,
                collection_task,
                monitoring_task,
                return_exceptions=True
            )
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested by user")
        except Exception as e:
            logger.error(f"❌ Continuous processing error: {e}")

    async def run_periodic_collection(self):
        """Run periodic data collection every hour"""
        while True:
            try:
                logger.info("⏰ Running periodic data collection...")

                # Collect browser history updates
                browser_data = await self.collector.harvest_browser_history()
                await self.collector.store_omniscient_data(browser_data)

                # Collect Google Calendar updates
                calendar_data = await self.collector.process_google_calendar()
                await self.collector.store_omniscient_data(calendar_data)

                # Check for new Takeout downloads
                takeout_path = Path("/opt/tower-echo-brain/data/takeout")
                if takeout_path.exists():
                    # Process any new files
                    new_files = []
                    for file_path in takeout_path.rglob("*"):
                        if file_path.is_file() and file_path.stat().st_mtime > (datetime.now().timestamp() - 3600):  # Last hour
                            new_files.append(file_path)

                    if new_files:
                        logger.info(f"📥 Processing {len(new_files)} new Takeout files")
                        # Process new files based on type
                        for file_path in new_files:
                            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                                # Process as image
                                result = self.collector._process_single_image(file_path)
                                if result:
                                    await self.collector.store_omniscient_data([result])

                logger.info("✅ Periodic collection completed")

                # Wait 1 hour
                await asyncio.sleep(3600)

            except Exception as e:
                logger.error(f"❌ Periodic collection error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def monitor_system_health(self):
        """Monitor system health and performance"""
        while True:
            try:
                # Check database connectivity
                import asyncpg
                conn = await asyncpg.connect("postgresql://patrick:tower_echo_brain_secret_key_2025@localhost/echo_brain")

                # Check system statistics
                stats = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as total_items,
                        COUNT(DISTINCT source) as active_sources,
                        COUNT(CASE WHEN timestamp > CURRENT_TIMESTAMP - INTERVAL '1 hour' THEN 1 END) as recent_items,
                        AVG(importance_score) as avg_importance
                    FROM omniscient_data
                """)

                await conn.close()

                if stats:
                    logger.info(f"📊 System health: {dict(stats)}")
                    self.status['system_healthy'] = True
                else:
                    logger.warning("⚠️ No data found - system may not be collecting properly")

                # Check Echo Brain service
                try:
                    import httpx
                    async with httpx.AsyncClient() as client:
                        response = await client.get("http://localhost:8309/health", timeout=5.0)
                        if response.status_code == 200:
                            logger.debug("✅ Echo Brain service healthy")
                        else:
                            logger.warning(f"⚠️ Echo Brain service status: {response.status_code}")
                except Exception as e:
                    logger.warning(f"⚠️ Echo Brain service check failed: {e}")

                # Update materialized views for performance
                conn = await asyncpg.connect("postgresql://patrick:tower_echo_brain_secret_key_2025@localhost/echo_brain")
                await conn.execute("SELECT refresh_omniscient_views()")
                await conn.close()

                # Wait 15 minutes
                await asyncio.sleep(900)

            except Exception as e:
                logger.error(f"❌ System health monitoring error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def generate_system_report(self) -> dict:
        """Generate comprehensive system status report"""
        logger.info("📋 Generating system report...")

        try:
            import asyncpg
            conn = await asyncpg.connect("postgresql://patrick:tower_echo_brain_secret_key_2025@localhost/echo_brain")

            # Overall statistics
            overall_stats = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total_items,
                    COUNT(DISTINCT source) as sources_active,
                    MIN(timestamp) as oldest_data,
                    MAX(timestamp) as newest_data,
                    AVG(importance_score) as avg_importance,
                    COUNT(CASE WHEN privacy_level = 'classified' THEN 1 END) as classified_items,
                    COUNT(CASE WHEN privacy_level = 'sensitive' THEN 1 END) as sensitive_items
                FROM omniscient_data
            """)

            # Data by source
            source_stats = await conn.fetch("""
                SELECT
                    source,
                    COUNT(*) as item_count,
                    AVG(importance_score) as avg_importance,
                    MAX(timestamp) as last_update
                FROM omniscient_data
                GROUP BY source
                ORDER BY item_count DESC
            """)

            # Recent activity
            recent_activity = await conn.fetch("""
                SELECT
                    DATE(timestamp) as date,
                    COUNT(*) as items_processed
                FROM omniscient_data
                WHERE timestamp > CURRENT_DATE - INTERVAL '7 days'
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """)

            # Knowledge graph statistics
            kg_stats = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total_relationships,
                    COUNT(DISTINCT entity_1_type) as entity_types,
                    AVG(strength) as avg_relationship_strength
                FROM knowledge_relationships
            """)

            await conn.close()

            report = {
                'timestamp': datetime.now().isoformat(),
                'system_status': self.status,
                'overall_statistics': dict(overall_stats) if overall_stats else {},
                'source_statistics': [dict(row) for row in source_stats],
                'recent_activity': [dict(row) for row in recent_activity],
                'knowledge_graph': dict(kg_stats) if kg_stats else {},
                'configuration': {
                    'data_sources_enabled': list(self.config.get('google', {}).get('oauth2', {}).get('scopes', [])),
                    'processing_batch_size': self.config.get('processing', {}).get('batch_size', 100),
                    'security_level': 'maximum'
                }
            }

            logger.info("✅ System report generated")
            return report

        except Exception as e:
            logger.error(f"❌ Failed to generate system report: {e}")
            return {'error': str(e)}

    async def query_personal_knowledge(self, query: str) -> dict:
        """Query the omniscient personal knowledge system"""
        try:
            if hasattr(self, 'integration'):
                results = await self.integration.query_omniscient_knowledge(query)
                return {
                    'query': query,
                    'results': results,
                    'result_count': len(results),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {'error': 'Integration not initialized'}

        except Exception as e:
            logger.error(f"❌ Knowledge query failed: {e}")
            return {'error': str(e)}

    async def activate_complete_system(self) -> bool:
        """Activate the complete omniscient brain system"""
        logger.info("🚀 ACTIVATING OMNISCIENT BRAIN SYSTEM")

        # Print banner
        self.print_system_banner()

        # Confirm activation
        confirm = input("\nActivate complete omniscient personal intelligence system? (yes/no): ").lower()
        if confirm != "yes":
            logger.info("❌ Activation cancelled by user")
            return False

        try:
            # Step 1: System setup
            if not await self.run_system_setup():
                logger.error("❌ System setup failed")
                return False

            # Step 2: Initialize data collector
            if not await self.initialize_data_collector():
                logger.error("❌ Data collector initialization failed")
                return False

            # Step 3: Initialize Echo Brain integration
            if not await self.initialize_echo_brain_integration():
                logger.error("❌ Echo Brain integration failed")
                return False

            # Step 4: Run initial data harvest
            logger.info("🌾 Running initial data harvest - this may take several hours...")
            harvest_results = await self.run_initial_data_harvest()

            # Step 5: Generate initial report
            initial_report = await self.generate_system_report()

            # Step 6: Start continuous processing
            logger.info("🎯 OMNISCIENT BRAIN SYSTEM FULLY ACTIVATED")

            print(f"""
✅ ACTIVATION COMPLETE!

📊 Initial Harvest Results:
   • Google Photos: {harvest_results.get('google_photos', 0)} items
   • Gmail: {harvest_results.get('gmail', 0)} items
   • Calendar: {harvest_results.get('google_calendar', 0)} items
   • Browser History: {harvest_results.get('browser_history', 0)} items
   • Credentials: {harvest_results.get('credentials', 0)} items

🧠 Total Personal Intelligence Items: {sum(harvest_results.values())}

🔄 Continuous processing is now active.
📡 Echo Brain integration is live.
🎯 Your personal AI now has COMPLETE awareness of your digital life.

Query your omniscient knowledge at: http://localhost:8309/api/omniscient/query
Monitor system status at: http://localhost:8309/api/omniscient/status

⚠️  REMEMBER: This system now knows EVERYTHING about you. Use responsibly.
            """)

            # Start continuous processing (this runs forever)
            await self.start_continuous_processing()

            return True

        except KeyboardInterrupt:
            logger.info("🛑 System shutdown requested")
            return True
        except Exception as e:
            logger.error(f"❌ System activation failed: {e}")
            return False

async def main():
    """Main entry point"""
    activator = OmniscientBrainActivator()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "setup":
            # Setup only
            await activator.run_system_setup()

        elif command == "harvest":
            # Harvest only
            await activator.initialize_data_collector()
            results = await activator.run_initial_data_harvest()
            print(f"Harvest results: {results}")

        elif command == "report":
            # Generate report
            report = await activator.generate_system_report()
            print(json.dumps(report, indent=2, default=str))

        elif command == "query":
            # Interactive query mode
            if len(sys.argv) < 3:
                print("Usage: python activate_omniscient_brain.py query \"your question\"")
                return

            query = sys.argv[2]
            await activator.initialize_echo_brain_integration()
            result = await activator.query_personal_knowledge(query)
            print(json.dumps(result, indent=2, default=str))

        elif command == "status":
            # System status
            await activator.initialize_data_collector()
            status = activator.status
            report = await activator.generate_system_report()
            print(f"System Status: {status}")
            print(f"Data Statistics: {report.get('overall_statistics', {})}")

        else:
            print("Unknown command. Available commands: setup, harvest, report, query, status")

    else:
        # Full activation
        await activator.activate_complete_system()

if __name__ == "__main__":
    asyncio.run(main())