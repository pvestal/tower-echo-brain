#!/usr/bin/env python3
"""
Anime Production Orchestration Startup Script

This script initializes and starts the complete anime production orchestration system,
including all necessary components and services.

Usage:
    python start_anime_orchestration.py [options]

Options:
    --port PORT         API service port (default: 8320)
    --host HOST         API service host (default: 0.0.0.0)
    --no-dashboard      Skip serving the dashboard
    --dev               Development mode with debug logging
    --check-deps        Check dependencies and exit

Dependencies:
    - Echo Brain service (localhost:8309)
    - ComfyUI service (localhost:8188)
    - PostgreSQL database (echo_brain schema)
    - Python dependencies (FastAPI, uvicorn, asyncpg, etc.)
"""

import argparse
import asyncio
import logging
import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from autonomous.anime_orchestration_service import app
    from autonomous.core import AutonomousCore
    from autonomous.anime_production_orchestrator import AnimeProductionOrchestrator
    import uvicorn
    import asyncpg
    import aiohttp
except ImportError as e:
    print(f"ERROR: Missing required dependency: {e}")
    print("Please install required packages:")
    print("pip install fastapi uvicorn asyncpg aiohttp")
    sys.exit(1)


class OrchestrationStarter:
    """Handles initialization and startup of the anime production orchestration system"""

    def __init__(self, args):
        self.args = args
        self.setup_logging()

        # Service configurations
        self.services = {
            'echo_brain': {'url': 'http://localhost:8309', 'required': True},
            'comfyui': {'url': 'http://localhost:8188', 'required': True},
            'ollama': {'url': 'http://localhost:11434', 'required': False}
        }

        # Database configuration
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'echo_brain',
            'user': 'patrick',
            'password': os.environ.get('ECHO_BRAIN_DB_PASSWORD', os.getenv("TOWER_DB_PASSWORD", os.getenv("DB_PASSWORD", "")))
        }

        # Required directories
        self.required_dirs = [
            '/mnt/1TB-storage/ComfyUI/output',
            '/mnt/1TB-storage/models/loras',
            '/tmp/anime_production',
            '/opt/tower-echo-brain/static'
        ]

    def setup_logging(self):
        """Configure logging based on arguments"""
        level = logging.DEBUG if self.args.dev else logging.INFO
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        logging.basicConfig(level=level, format=format_str)
        self.logger = logging.getLogger(__name__)

    async def check_dependencies(self) -> Dict[str, bool]:
        """Check all system dependencies"""
        self.logger.info("Checking system dependencies...")
        results = {}

        # Check database connection
        try:
            conn = await asyncpg.connect(**self.db_config)
            await conn.close()
            results['database'] = True
            self.logger.info("✓ Database connection successful")
        except Exception as e:
            results['database'] = False
            self.logger.error(f"✗ Database connection failed: {e}")

        # Check external services
        async with aiohttp.ClientSession() as session:
            for service_name, config in self.services.items():
                try:
                    async with session.get(config['url'], timeout=5) as response:
                        if response.status < 500:
                            results[service_name] = True
                            self.logger.info(f"✓ {service_name.upper()} service accessible")
                        else:
                            results[service_name] = False
                            self.logger.warning(f"⚠ {service_name.upper()} service returned {response.status}")
                except Exception as e:
                    results[service_name] = False
                    if config['required']:
                        self.logger.error(f"✗ {service_name.upper()} service not accessible: {e}")
                    else:
                        self.logger.warning(f"⚠ {service_name.upper()} service not accessible: {e}")

        # Check required directories
        for dir_path in self.required_dirs:
            path = Path(dir_path)
            try:
                path.mkdir(parents=True, exist_ok=True)
                if os.access(path, os.W_OK):
                    results[f'dir_{path.name}'] = True
                    self.logger.info(f"✓ Directory accessible: {dir_path}")
                else:
                    results[f'dir_{path.name}'] = False
                    self.logger.error(f"✗ Directory not writable: {dir_path}")
            except Exception as e:
                results[f'dir_{path.name}'] = False
                self.logger.error(f"✗ Directory creation failed: {dir_path} - {e}")

        # Check Python packages
        required_packages = ['fastapi', 'uvicorn', 'asyncpg', 'aiohttp', 'pydantic']
        for package in required_packages:
            try:
                __import__(package)
                results[f'pkg_{package}'] = True
                self.logger.info(f"✓ Python package: {package}")
            except ImportError:
                results[f'pkg_{package}'] = False
                self.logger.error(f"✗ Missing Python package: {package}")

        # Check system tools
        system_tools = ['ffmpeg', 'ffprobe']  # Required for video processing
        for tool in system_tools:
            try:
                subprocess.run([tool, '-version'], capture_output=True, check=True)
                results[f'tool_{tool}'] = True
                self.logger.info(f"✓ System tool: {tool}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                results[f'tool_{tool}'] = False
                self.logger.warning(f"⚠ System tool not found: {tool}")

        return results

    def check_critical_dependencies(self, results: Dict[str, bool]) -> bool:
        """Check if critical dependencies are met"""
        critical = ['database', 'echo_brain', 'comfyui']
        missing_critical = [dep for dep in critical if not results.get(dep, False)]

        if missing_critical:
            self.logger.error(f"Critical dependencies missing: {missing_critical}")
            return False

        return True

    async def initialize_database(self):
        """Initialize database schema if needed"""
        self.logger.info("Initializing database schema...")

        try:
            # Test basic connection first
            conn = await asyncpg.connect(**self.db_config)

            # Check if autonomous tables exist
            table_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'autonomous_goals'
                )
            """)

            if not table_exists:
                self.logger.info("Creating autonomous core schema...")
                schema_file = Path(__file__).parent.parent / 'src' / 'autonomous' / 'schema.sql'

                if schema_file.exists():
                    with open(schema_file, 'r') as f:
                        schema_sql = f.read()
                    await conn.execute(schema_sql)
                    self.logger.info("✓ Autonomous core schema created")
                else:
                    self.logger.warning("Schema file not found, assuming schema exists")
            else:
                self.logger.info("✓ Database schema already exists")

            await conn.close()
            return True

        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            return False

    def create_systemd_service(self):
        """Create systemd service file for the orchestration service"""
        service_content = f"""[Unit]
Description=Anime Production Orchestration Service
After=network.target postgresql.service
Wants=postgresql.service

[Service]
Type=simple
User=patrick
Group=patrick
WorkingDirectory=/opt/tower-echo-brain
Environment=PYTHONPATH=/opt/tower-echo-brain/src
Environment=ECHO_BRAIN_DB_PASSWORD={self.db_config['password']}
ExecStart=/usr/bin/python3 /opt/tower-echo-brain/scripts/start_anime_orchestration.py --port {self.args.port} --host {self.args.host}
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""

        service_file = Path('/etc/systemd/system/tower-anime-orchestration.service')

        try:
            # Check if we have write permissions
            if os.access('/etc/systemd/system', os.W_OK):
                with open(service_file, 'w') as f:
                    f.write(service_content)
                self.logger.info(f"✓ Systemd service file created: {service_file}")

                # Reload systemd
                subprocess.run(['sudo', 'systemctl', 'daemon-reload'], check=True)
                self.logger.info("✓ Systemd daemon reloaded")

                return True
            else:
                self.logger.warning("Cannot create systemd service (no write permission)")
                self.logger.info("To create service manually, run:")
                self.logger.info(f"sudo tee {service_file} > /dev/null << 'EOF'")
                self.logger.info(service_content)
                self.logger.info("EOF")
                self.logger.info("sudo systemctl daemon-reload")
                return False

        except Exception as e:
            self.logger.error(f"Failed to create systemd service: {e}")
            return False

    def create_startup_script(self):
        """Create convenience startup script"""
        script_content = f"""#!/bin/bash
# Anime Production Orchestration Startup Script
# Generated automatically

cd /opt/tower-echo-brain

# Set environment variables
export PYTHONPATH=/opt/tower-echo-brain/src
export ECHO_BRAIN_DB_PASSWORD="{self.db_config['password']}"

# Start the orchestration service
python3 scripts/start_anime_orchestration.py \\
    --port {self.args.port} \\
    --host {self.args.host} \\
    "$@"
"""

        script_file = Path('/opt/tower-echo-brain/start_anime_orchestration.sh')

        try:
            with open(script_file, 'w') as f:
                f.write(script_content)

            # Make executable
            script_file.chmod(0o755)
            self.logger.info(f"✓ Startup script created: {script_file}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create startup script: {e}")
            return False

    async def start_service(self):
        """Start the orchestration service"""
        self.logger.info("Starting Anime Production Orchestration Service...")

        try:
            # Configure uvicorn
            config = uvicorn.Config(
                app,
                host=self.args.host,
                port=self.args.port,
                log_level="debug" if self.args.dev else "info",
                access_log=True,
                reload=self.args.dev
            )

            server = uvicorn.Server(config)

            # Add static file serving for dashboard
            if not self.args.no_dashboard:
                from fastapi.staticfiles import StaticFiles
                static_dir = Path('/opt/tower-echo-brain/static')
                if static_dir.exists():
                    app.mount("/dashboard", StaticFiles(directory=static_dir, html=True), name="dashboard")
                    self.logger.info(f"✓ Dashboard available at http://{self.args.host}:{self.args.port}/dashboard/")

            self.logger.info(f"✓ Service starting on http://{self.args.host}:{self.args.port}")
            self.logger.info(f"✓ API documentation: http://{self.args.host}:{self.args.port}/docs")

            await server.serve()

        except Exception as e:
            self.logger.error(f"Failed to start service: {e}")
            return False

    async def run(self):
        """Main execution flow"""
        self.logger.info("=" * 60)
        self.logger.info("Anime Production Orchestration System Startup")
        self.logger.info("=" * 60)

        # Check dependencies
        results = await self.check_dependencies()

        if self.args.check_deps:
            # Print summary and exit
            print("\nDependency Check Summary:")
            print("-" * 40)

            for key, status in results.items():
                status_str = "✓ OK" if status else "✗ FAIL"
                print(f"{key.replace('_', ' ').title()}: {status_str}")

            critical_ok = self.check_critical_dependencies(results)
            print(f"\nCritical Dependencies: {'✓ OK' if critical_ok else '✗ FAIL'}")

            return 0 if critical_ok else 1

        # Check critical dependencies
        if not self.check_critical_dependencies(results):
            self.logger.error("Cannot start service due to missing critical dependencies")
            return 1

        # Initialize database
        if not await self.initialize_database():
            self.logger.error("Database initialization failed")
            return 1

        # Create service files if requested
        if self.args.create_service:
            self.create_systemd_service()

        self.create_startup_script()

        # Start the service
        try:
            await self.start_service()
            return 0
        except KeyboardInterrupt:
            self.logger.info("Service stopped by user")
            return 0
        except Exception as e:
            self.logger.error(f"Service failed: {e}")
            return 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Start the Anime Production Orchestration System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_anime_orchestration.py                 # Start with defaults
  python start_anime_orchestration.py --port 8321    # Custom port
  python start_anime_orchestration.py --dev          # Development mode
  python start_anime_orchestration.py --check-deps   # Check dependencies only

The service provides:
  • REST API for project management
  • Real-time WebSocket updates
  • Interactive dashboard UI
  • Integration with Echo Brain and ComfyUI
  • Autonomous workflow orchestration
        """
    )

    parser.add_argument(
        '--port', type=int, default=8320,
        help='API service port (default: 8320)'
    )

    parser.add_argument(
        '--host', default='0.0.0.0',
        help='API service host (default: 0.0.0.0)'
    )

    parser.add_argument(
        '--no-dashboard', action='store_true',
        help='Skip serving the dashboard'
    )

    parser.add_argument(
        '--dev', action='store_true',
        help='Development mode with debug logging and auto-reload'
    )

    parser.add_argument(
        '--check-deps', action='store_true',
        help='Check dependencies and exit'
    )

    parser.add_argument(
        '--create-service', action='store_true',
        help='Create systemd service file'
    )

    args = parser.parse_args()

    # Create and run the starter
    starter = OrchestrationStarter(args)

    try:
        exit_code = asyncio.run(starter.run())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nStartup interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()