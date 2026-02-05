"""
System Model for Echo Brain
Maintains a live model of the Tower system.
Knows what's running, how things connect, current health.
"""

import asyncio
import asyncpg
import psutil
import subprocess
import json
import httpx
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

from .schemas import (
    Service, ServiceStatus, ServiceDependency, ServiceHealth,
    NetworkMap, Schema, ServiceType
)

logger = logging.getLogger(__name__)


class SystemModel:
    """
    Maintains a live model of the Tower system.
    Knows what's running, how things connect, current health.
    """

    def __init__(self, db_config: Dict[str, str] = None):
        self.db_config = db_config or {
            "host": "localhost",
            "database": "echo_brain",
            "user": "patrick",
            "password": "RP78eIrW7cI2jYvL5akt1yurE"
        }
        self._pool = None

    async def get_db_pool(self):
        """Get or create database connection pool"""
        if not self._pool:
            self._pool = await asyncpg.create_pool(
                **self.db_config,
                min_size=2,
                max_size=10,
                timeout=10
            )
        return self._pool

    async def close(self):
        """Clean up connections"""
        if self._pool:
            await self._pool.close()

    def _run_command(self, command: str) -> Dict[str, Any]:
        """Run a shell command and return result"""
        try:
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=10
            )
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout.strip(),
                'stderr': result.stderr.strip(),
                'returncode': result.returncode
            }
        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e),
                'returncode': -1
            }

    async def discover_services(self) -> List[Service]:
        """
        Find all Tower services:
        - systemd units matching tower-*
        - Docker containers
        - Listening ports
        - Process names
        """
        services = []

        # Discover systemd services
        systemd_services = await self._discover_systemd_services()
        services.extend(systemd_services)

        # Discover Docker containers
        docker_services = await self._discover_docker_services()
        services.extend(docker_services)

        # Discover listening processes
        port_services = await self._discover_port_services()
        services.extend(port_services)

        # Remove duplicates and update database
        unique_services = self._deduplicate_services(services)
        await self._update_services_database(unique_services)

        return unique_services

    async def _discover_systemd_services(self) -> List[Service]:
        """Find Tower systemd services"""
        services = []

        try:
            # List all tower-* services
            result = self._run_command("systemctl list-units --type=service --no-pager tower-*")

            if result['success']:
                lines = result['stdout'].split('\n')
                for line in lines:
                    if 'tower-' in line and '.service' in line:
                        parts = line.split()
                        if parts:
                            service_name = parts[0].replace('.service', '')

                            # Get detailed status
                            status_result = self._run_command(f"systemctl is-active {service_name}")
                            status = status_result['stdout'] if status_result['success'] else 'unknown'

                            # Get service file path
                            show_result = self._run_command(f"systemctl show -p FragmentPath {service_name}")
                            config_path = None
                            if show_result['success'] and 'FragmentPath=' in show_result['stdout']:
                                config_path = show_result['stdout'].split('FragmentPath=')[1].strip()

                            services.append(Service(
                                name=service_name,
                                service_type=ServiceType.SYSTEMD,
                                status=status,
                                config_path=config_path,
                                last_checked=datetime.now(),
                                metadata={'systemd_unit': f"{service_name}.service"}
                            ))

        except Exception as e:
            logger.error(f"Error discovering systemd services: {e}")

        return services

    async def _discover_docker_services(self) -> List[Service]:
        """Find Docker containers related to Tower"""
        services = []

        try:
            result = self._run_command("docker ps --format json")

            if result['success'] and result['stdout']:
                for line in result['stdout'].split('\n'):
                    if line.strip():
                        try:
                            container = json.loads(line)
                            name = container.get('Names', '')
                            image = container.get('Image', '')

                            # Only include Tower-related containers
                            if 'tower' in name.lower() or 'tower' in image.lower():
                                ports = container.get('Ports', '')
                                port = self._extract_port_from_string(ports)

                                services.append(Service(
                                    name=name,
                                    service_type=ServiceType.DOCKER,
                                    port=port,
                                    status=container.get('State', 'unknown'),
                                    last_checked=datetime.now(),
                                    metadata={
                                        'image': image,
                                        'container_id': container.get('ID', ''),
                                        'ports': ports
                                    }
                                ))
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.error(f"Error discovering Docker services: {e}")

        return services

    async def _discover_port_services(self) -> List[Service]:
        """Find services by listening ports"""
        services = []

        # Known Tower ports
        known_ports = {
            8309: 'tower-echo-brain',
            8088: 'tower-auth',
            8188: 'comfyui',
            8307: 'tower-kb',
            8311: 'tower-frontend',
            11434: 'ollama',
            6333: 'qdrant'
        }

        try:
            for port, expected_service in known_ports.items():
                # Check if port is listening
                connections = psutil.net_connections(kind='inet')
                listening = any(
                    conn.laddr.port == port and conn.status == 'LISTEN'
                    for conn in connections
                )

                if listening:
                    # Try to get process info
                    process_name = None
                    pid = None

                    for proc in psutil.process_iter(['pid', 'name']):
                        try:
                            # Get connections for this specific process
                            proc_conns = proc.connections(kind='inet')
                            for conn in proc_conns:
                                if conn.laddr.port == port and conn.status == 'LISTEN':
                                    process_name = proc.info['name']
                                    pid = proc.info['pid']
                                    break
                        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                            continue

                    services.append(Service(
                        name=expected_service,
                        service_type=ServiceType.PROCESS,
                        port=port,
                        status='running',
                        last_checked=datetime.now(),
                        metadata={
                            'process_name': process_name,
                            'pid': pid,
                            'detected_by': 'port_scan'
                        }
                    ))

        except Exception as e:
            logger.error(f"Error discovering port services: {e}")

        return services

    def _extract_port_from_string(self, ports_str: str) -> Optional[int]:
        """Extract port number from Docker ports string"""
        try:
            # Look for patterns like "0.0.0.0:8309->8309/tcp"
            if ':' in ports_str and '->' in ports_str:
                external_part = ports_str.split('->')[0]
                if ':' in external_part:
                    return int(external_part.split(':')[-1])
        except (ValueError, IndexError):
            pass
        return None

    def _deduplicate_services(self, services: List[Service]) -> List[Service]:
        """Remove duplicate services, preferring systemd over others"""
        seen = {}

        # Sort by preference: systemd > docker > process
        type_priority = {ServiceType.SYSTEMD: 0, ServiceType.DOCKER: 1, ServiceType.PROCESS: 2}
        services.sort(key=lambda s: type_priority.get(s.service_type, 3))

        for service in services:
            if service.name not in seen:
                seen[service.name] = service
            else:
                # Merge information from duplicate
                existing = seen[service.name]
                if not existing.port and service.port:
                    existing.port = service.port
                existing.metadata.update(service.metadata)

        return list(seen.values())

    async def _update_services_database(self, services: List[Service]):
        """Update services in database"""
        pool = await self.get_db_pool()

        try:
            async with pool.acquire() as conn:
                for service in services:
                    # Check if service exists
                    existing = await conn.fetchrow(
                        "SELECT id FROM services WHERE name = $1", service.name
                    )

                    if existing:
                        # Update existing
                        await conn.execute(
                            """UPDATE services
                               SET service_type = $2, port = $3, status = $4,
                                   config_path = $5, last_checked = $6, metadata = $7
                               WHERE id = $1""",
                            existing['id'], service.service_type.value, service.port,
                            service.status, service.config_path, service.last_checked,
                            json.dumps(service.metadata)
                        )
                    else:
                        # Insert new
                        await conn.execute(
                            """INSERT INTO services
                               (name, service_type, port, status, config_path, last_checked, metadata)
                               VALUES ($1, $2, $3, $4, $5, $6, $7)""",
                            service.name, service.service_type.value, service.port,
                            service.status, service.config_path, service.last_checked,
                            json.dumps(service.metadata)
                        )

        except Exception as e:
            logger.error(f"Error updating services database: {e}")

    async def get_service_status(self, name: str) -> ServiceStatus:
        """Live status: running, ports, memory, uptime, recent errors"""
        pool = await self.get_db_pool()

        try:
            async with pool.acquire() as conn:
                service = await conn.fetchrow(
                    "SELECT * FROM services WHERE name = $1", name
                )

                if not service:
                    return ServiceStatus(
                        name=name,
                        status='not_found',
                        recent_errors=['Service not found in database']
                    )

                # Get live status based on service type
                status = await self._get_live_service_status(service)

                # Get recent errors from logs
                recent_errors = await self._get_recent_errors(name)

                return ServiceStatus(
                    name=name,
                    status=status['status'],
                    port=service['port'],
                    uptime=status.get('uptime'),
                    memory_mb=status.get('memory_mb'),
                    cpu_percent=status.get('cpu_percent'),
                    recent_errors=recent_errors
                )

        except Exception as e:
            logger.error(f"Error getting service status for {name}: {e}")
            return ServiceStatus(
                name=name,
                status='error',
                recent_errors=[str(e)]
            )

    async def _get_live_service_status(self, service: Dict) -> Dict[str, Any]:
        """Get live status for a service"""
        name = service['name']
        service_type = service['service_type']

        if service_type == 'systemd':
            return await self._get_systemd_status(name)
        elif service_type == 'docker':
            return await self._get_docker_status(name)
        elif service_type == 'process' and service['port']:
            return await self._get_port_status(service['port'])
        else:
            return {'status': 'unknown'}

    async def _get_systemd_status(self, name: str) -> Dict[str, Any]:
        """Get systemd service status"""
        try:
            # Get status
            status_result = self._run_command(f"systemctl is-active {name}")
            status = status_result['stdout'] if status_result['success'] else 'failed'

            # Get detailed info
            show_result = self._run_command(f"systemctl show {name}")

            uptime = None
            memory_mb = None

            if show_result['success']:
                lines = show_result['stdout'].split('\n')
                for line in lines:
                    if line.startswith('ActiveEnterTimestamp='):
                        timestamp_str = line.split('=', 1)[1].strip()
                        if timestamp_str and timestamp_str != 'n/a':
                            try:
                                # Parse systemd timestamp
                                from dateutil import parser
                                start_time = parser.parse(timestamp_str.split(' ')[1:4])
                                uptime_seconds = (datetime.now() - start_time).total_seconds()
                                uptime = f"{int(uptime_seconds // 3600)}h {int((uptime_seconds % 3600) // 60)}m"
                            except:
                                pass
                    elif line.startswith('MemoryCurrent='):
                        try:
                            memory_bytes = int(line.split('=')[1].strip())
                            memory_mb = memory_bytes / 1024 / 1024
                        except:
                            pass

            return {
                'status': status,
                'uptime': uptime,
                'memory_mb': memory_mb
            }

        except Exception as e:
            logger.error(f"Error getting systemd status for {name}: {e}")
            return {'status': 'error'}

    async def _get_docker_status(self, name: str) -> Dict[str, Any]:
        """Get Docker container status"""
        try:
            result = self._run_command(f"docker inspect {name}")

            if result['success']:
                data = json.loads(result['stdout'])
                if data:
                    container = data[0]
                    state = container.get('State', {})

                    status = 'running' if state.get('Running') else 'stopped'

                    # Calculate uptime
                    uptime = None
                    if state.get('StartedAt'):
                        from dateutil import parser
                        start_time = parser.parse(state['StartedAt'])
                        uptime_seconds = (datetime.now() - start_time.replace(tzinfo=None)).total_seconds()
                        uptime = f"{int(uptime_seconds // 3600)}h {int((uptime_seconds % 3600) // 60)}m"

                    return {
                        'status': status,
                        'uptime': uptime
                    }

        except Exception as e:
            logger.error(f"Error getting Docker status for {name}: {e}")

        return {'status': 'unknown'}

    async def _get_port_status(self, port: int) -> Dict[str, Any]:
        """Check if port is listening and get process info"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent', 'connections']):
                try:
                    for conn in proc.info['connections'] or []:
                        if conn.laddr.port == port and conn.status == 'LISTEN':
                            memory_info = proc.info.get('memory_info')
                            return {
                                'status': 'running',
                                'memory_mb': memory_info.rss / 1024 / 1024 if memory_info else None,
                                'cpu_percent': proc.info.get('cpu_percent')
                            }
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            return {'status': 'stopped'}

        except Exception as e:
            logger.error(f"Error getting port status for {port}: {e}")
            return {'status': 'error'}

    async def _get_recent_errors(self, service_name: str) -> List[str]:
        """Get recent error messages for a service"""
        errors = []

        try:
            # Try to get systemd errors
            result = self._run_command(f"journalctl -u {service_name} --since '1 hour ago' --no-pager -p err")

            if result['success'] and result['stdout']:
                lines = result['stdout'].split('\n')[-5:]  # Last 5 error lines
                errors.extend([line for line in lines if line.strip()])

        except Exception as e:
            logger.debug(f"Could not get systemd logs for {service_name}: {e}")

        return errors

    async def get_service_dependencies(self, name: str) -> List[str]:
        """What does this service need to run"""
        pool = await self.get_db_pool()

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT depends_on FROM service_dependencies WHERE service_id = (SELECT id FROM services WHERE name = $1)",
                    name
                )
                return [row['depends_on'] for row in rows]

        except Exception as e:
            logger.error(f"Error getting dependencies for {name}: {e}")
            return []

    async def get_service_config(self, name: str) -> Dict[str, Any]:
        """Configuration from env files, systemd units, config files"""
        pool = await self.get_db_pool()
        config = {}

        try:
            async with pool.acquire() as conn:
                service = await conn.fetchrow(
                    "SELECT * FROM services WHERE name = $1", name
                )

                if not service:
                    return {}

                # Get config from systemd unit
                if service['service_type'] == 'systemd' and service['config_path']:
                    config.update(await self._parse_systemd_config(service['config_path']))

                # Look for .env files
                service_dir = f"/opt/{name}"
                env_file = f"{service_dir}/.env"
                if Path(env_file).exists():
                    config.update(await self._parse_env_file(env_file))

                # Add metadata
                config['metadata'] = json.loads(service['metadata']) if service['metadata'] else {}

        except Exception as e:
            logger.error(f"Error getting config for {name}: {e}")

        return config

    async def _parse_systemd_config(self, unit_path: str) -> Dict[str, Any]:
        """Parse systemd unit file"""
        config = {}

        try:
            with open(unit_path, 'r') as f:
                content = f.read()

            for line in content.split('\n'):
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()

        except Exception as e:
            logger.debug(f"Error parsing systemd config {unit_path}: {e}")

        return config

    async def _parse_env_file(self, env_path: str) -> Dict[str, Any]:
        """Parse .env file"""
        config = {}

        try:
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip().strip('"\'')

        except Exception as e:
            logger.debug(f"Error parsing env file {env_path}: {e}")

        return config

    async def get_database_schema(self, db_name: str) -> Schema:
        """Tables, columns, relationships, indexes"""
        try:
            pool = await asyncpg.create_pool(
                host=self.db_config['host'],
                database=db_name,
                user=self.db_config['user'],
                password=self.db_config['password'],
                min_size=1,
                max_size=2
            )

            async with pool.acquire() as conn:
                # Get tables and columns
                tables = await conn.fetch("""
                    SELECT
                        t.table_name,
                        c.column_name,
                        c.data_type,
                        c.is_nullable,
                        c.column_default
                    FROM information_schema.tables t
                    LEFT JOIN information_schema.columns c ON t.table_name = c.table_name
                    WHERE t.table_schema = 'public' AND t.table_type = 'BASE TABLE'
                    ORDER BY t.table_name, c.ordinal_position
                """)

                # Organize by table
                table_dict = {}
                for row in tables:
                    table_name = row['table_name']
                    if table_name not in table_dict:
                        table_dict[table_name] = {
                            'name': table_name,
                            'columns': []
                        }

                    if row['column_name']:
                        table_dict[table_name]['columns'].append({
                            'name': row['column_name'],
                            'type': row['data_type'],
                            'nullable': row['is_nullable'] == 'YES',
                            'default': row['column_default']
                        })

            await pool.close()

            return Schema(
                database=db_name,
                tables=list(table_dict.values())
            )

        except Exception as e:
            logger.error(f"Error getting schema for {db_name}: {e}")
            return Schema(database=db_name, tables=[])

    async def get_network_map(self) -> NetworkMap:
        """What talks to what, on which ports"""
        pool = await self.get_db_pool()

        try:
            async with pool.acquire() as conn:
                services = await conn.fetch("SELECT * FROM services")

                # Build connections based on known patterns
                connections = []

                for service in services:
                    service_name = service['name']
                    port = service['port']

                    # Known connections
                    if service_name == 'tower-echo-brain':
                        connections.extend([
                            {'from': service_name, 'to': 'postgres', 'port': 5432, 'type': 'database'},
                            {'from': service_name, 'to': 'qdrant', 'port': 6333, 'type': 'vector_db'},
                            {'from': service_name, 'to': 'ollama', 'port': 11434, 'type': 'llm'}
                        ])
                    elif service_name == 'tower-auth':
                        connections.append({'from': service_name, 'to': 'postgres', 'port': 5432, 'type': 'database'})

                return NetworkMap(
                    services=[
                        Service(
                            id=s['id'],
                            name=s['name'],
                            service_type=ServiceType(s['service_type']),
                            port=s['port'],
                            status=s['status'],
                            config_path=s['config_path'],
                            last_checked=s['last_checked'],
                            metadata=json.loads(s['metadata']) if s['metadata'] else {}
                        )
                        for s in services
                    ],
                    connections=connections
                )

        except Exception as e:
            logger.error(f"Error building network map: {e}")
            return NetworkMap(services=[], connections=[])


# Singleton instance
_system_model = None

def get_system_model() -> SystemModel:
    """Get or create singleton instance"""
    global _system_model
    if not _system_model:
        _system_model = SystemModel()
    return _system_model