#!/usr/bin/env python3
"""
Spatial Reasoning for Echo Brain
Enables understanding of code structure as 3D space.
"""

import ast
import networkx as nx
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import logging
import pickle
import json
import re
import os
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class ServiceInfo:
    """Information about a Tower service."""
    name: str
    port: Optional[int]
    path: str
    dependencies: List[str]
    api_endpoints: List[str]
    database_refs: List[str]
    config_files: List[str]

@dataclass
class FileAnalysis:
    """Analysis results for a code file."""
    path: str
    imports: List[str]
    functions: List[str]
    classes: List[str]
    api_calls: List[str]
    database_queries: List[str]
    config_refs: List[str]
    service_name: str
    file_type: str

class SpatialCodeUnderstanding:
    """
    Maps codebase as spatial structure for better navigation and understanding.
    """

    def __init__(self, data_dir: str = "/opt/tower-echo-brain/data"):
        self.graph = nx.DiGraph()
        self.file_positions = {}  # 3D positions for each file
        self.service_map = {}     # Service -> Port mapping
        self.import_graph = {}    # Import relationships
        self.services = {}        # Service name -> ServiceInfo
        self.data_dir = Path(data_dir)
        self.graph_file = self.data_dir / "tower_knowledge_graph.pkl"
        self.metadata_file = self.data_dir / "graph_metadata.json"

        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load existing graph if available
        self.load_graph()

    def analyze_python_file(self, filepath: Path, service_name: str = "unknown") -> FileAnalysis:
        """Extract comprehensive information from Python file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Try to parse AST, but continue even if it fails
            imports = []
            functions = []
            classes = []

            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            imports.append(name.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module)
                    elif isinstance(node, ast.FunctionDef):
                        functions.append(node.name)
                    elif isinstance(node, ast.ClassDef):
                        classes.append(node.name)
            except SyntaxError:
                # Fallback to regex parsing for files with syntax errors
                import_matches = re.findall(r'(?:from\s+(\S+)\s+import|import\s+(\S+))', content)
                for match in import_matches:
                    imports.extend([m for m in match if m])

                func_matches = re.findall(r'def\s+(\w+)\s*\(', content)
                functions.extend(func_matches)

                class_matches = re.findall(r'class\s+(\w+)\s*[:\(]', content)
                classes.extend(class_matches)

            # Detect API calls
            api_calls = []
            api_patterns = [
                r'requests\.[get|post|put|delete]+\([\'"]([^\'"]+)',
                r'curl\s+[^\s]*?([/\w\-]+)',
                r'@app\.route\([\'"]([^\'"]+)',
                r'@router\.[get|post|put|delete]+\([\'"]([^\'"]+)',
                r'fetch\([\'"]([^\'"]+)',
                r'axios\.[get|post|put|delete]+\([\'"]([^\'"]+)'
            ]

            for pattern in api_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                api_calls.extend(matches)

            # Detect database queries
            db_queries = []
            db_patterns = [
                r'SELECT\s+.*?FROM\s+(\w+)',
                r'INSERT\s+INTO\s+(\w+)',
                r'UPDATE\s+(\w+)\s+SET',
                r'DELETE\s+FROM\s+(\w+)',
                r'CREATE\s+TABLE\s+(\w+)',
                r'\.execute\([\'"]([^\'"]*(?:SELECT|INSERT|UPDATE|DELETE)[^\'"]*)',
                r'\.query\([\'"]([^\'"]*(?:SELECT|INSERT|UPDATE|DELETE)[^\'"]*)'
            ]

            for pattern in db_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                db_queries.extend(matches)

            # Detect config references
            config_refs = []
            config_patterns = [
                r'config\[[\'"]([\w_]+)[\'"]',
                r'\.env\.[A-Z_]+',
                r'getenv\([\'"]([^\'"]+)',
                r'ENV\[[\'"]([^\'"]+)',
                r'\.get\([\'"]([^\'"]+)[\'"]'
            ]

            for pattern in config_patterns:
                matches = re.findall(pattern, content)
                config_refs.extend(matches)

            # Determine file type
            file_type = "unknown"
            if "api" in str(filepath).lower() or "routes" in str(filepath).lower():
                file_type = "api"
            elif "frontend" in str(filepath).lower() or filepath.suffix in ['.vue', '.js', '.tsx']:
                file_type = "frontend"
            elif "database" in str(filepath).lower() or "db" in str(filepath).lower():
                file_type = "database"
            elif "config" in str(filepath).lower():
                file_type = "config"
            elif "test" in str(filepath).lower():
                file_type = "test"
            else:
                file_type = "core"

            return FileAnalysis(
                path=str(filepath),
                imports=imports,
                functions=functions,
                classes=classes,
                api_calls=api_calls,
                database_queries=db_queries,
                config_refs=config_refs,
                service_name=service_name,
                file_type=file_type
            )

        except Exception as e:
            logger.warning(f"Error analyzing {filepath}: {e}")
            return FileAnalysis(
                path=str(filepath),
                imports=[],
                functions=[],
                classes=[],
                api_calls=[],
                database_queries=[],
                config_refs=[],
                service_name=service_name,
                file_type="error"
            )

    def discover_tower_services(self) -> List[str]:
        """Discover all Tower service directories."""
        opt_dirs = []

        # Scan /opt for tower-* directories
        opt_path = Path("/opt")
        if opt_path.exists():
            opt_dirs = [str(p) for p in opt_path.glob("tower-*") if p.is_dir()]

        # Add other important paths
        additional_paths = [
            "/home/patrick/Tower",
            "/mnt/1TB-storage/ComfyUI",
            "/opt/echo",
            "/opt/echo-fast"
        ]

        for path in additional_paths:
            if Path(path).exists():
                opt_dirs.append(path)

        logger.info(f"Discovered {len(opt_dirs)} Tower-related directories")
        return opt_dirs

    def extract_service_info(self, service_path: Path) -> ServiceInfo:
        """Extract service information from directory."""
        service_name = service_path.name

        # Try to find port from various sources
        port = None
        port_patterns = [
            r'port[:\s=]+(\d+)',
            r'PORT[:\s=]+(\d+)',
            r'listen[:\s]+(\d+)',
            r'8\d{3}'  # Common pattern for 8xxx ports
        ]

        # Check common config files
        config_files = []
        for config_file in ["app.py", "main.py", "server.py", "*.service", ".env"]:
            config_files.extend(service_path.glob(config_file))

        for config_file in config_files:
            try:
                if config_file.is_file():
                    content = config_file.read_text()
                    for pattern in port_patterns:
                        matches = re.findall(pattern, content)
                        if matches:
                            port = int(matches[0])
                            break
                    if port:
                        break
            except:
                continue

        # Extract dependencies, APIs, database refs
        dependencies = []
        api_endpoints = []
        database_refs = []

        # Scan for common dependency patterns
        for py_file in service_path.rglob("*.py"):
            try:
                content = py_file.read_text()

                # Look for service calls
                service_patterns = [
                    r'requests\.(?:get|post|put|delete)\([\'"].*?:(\d+)',
                    r'http://localhost:(\d+)',
                    r'http://192\.168\.50\.135:(\d+)'
                ]

                for pattern in service_patterns:
                    matches = re.findall(pattern, content)
                    dependencies.extend([f"port-{m}" for m in matches])

                # Look for API definitions
                api_patterns = [
                    r'@app\.route\([\'"]([^\'"]+)',
                    r'@router\.[get|post|put|delete]+\([\'"]([^\'"]+)'
                ]

                for pattern in api_patterns:
                    matches = re.findall(pattern, content)
                    api_endpoints.extend(matches)

                # Look for database references
                db_patterns = [
                    r'psycopg2\.connect|postgresql://',
                    r'sqlite3\.connect',
                    r'redis\.Redis',
                    r'qdrant_client'
                ]

                for pattern in db_patterns:
                    if re.search(pattern, content):
                        database_refs.append(pattern.split('.')[0])

            except:
                continue

        return ServiceInfo(
            name=service_name,
            port=port,
            path=str(service_path),
            dependencies=list(set(dependencies)),
            api_endpoints=list(set(api_endpoints)),
            database_refs=list(set(database_refs)),
            config_files=[str(f) for f in config_files]
        )

    def build_codebase_graph(self, root_paths: Optional[List[str]] = None):
        """Build comprehensive dependency graph of entire Tower codebase."""
        if root_paths is None:
            root_paths = self.discover_tower_services()

        logger.info(f"Building codebase graph for {len(root_paths)} directories")

        total_files = 0
        total_services = 0

        for root in root_paths:
            root_path = Path(root)
            if not root_path.exists():
                logger.warning(f"Path does not exist: {root}")
                continue

            # Extract service info
            service_name = root_path.name
            service_info = self.extract_service_info(root_path)
            self.services[service_name] = service_info
            total_services += 1

            # Add service node to graph
            self.graph.add_node(f"service:{service_name}",
                               type="service",
                               **asdict(service_info))

            # Find all relevant files (not just Python)
            file_patterns = ["*.py", "*.js", "*.vue", "*.ts", "*.tsx", "*.json", "*.yaml", "*.yml"]
            all_files = []

            for pattern in file_patterns:
                all_files.extend(root_path.rglob(pattern))

            # Limit files per service to prevent overwhelming the graph
            max_files_per_service = 200
            if len(all_files) > max_files_per_service:
                logger.info(f"Service {service_name} has {len(all_files)} files, limiting to {max_files_per_service}")
                all_files = all_files[:max_files_per_service]

            for file_path in all_files:
                try:
                    # Skip common unimportant files
                    if any(skip in str(file_path) for skip in [
                        'node_modules', '__pycache__', '.git', 'venv', '.env',
                        'package-lock.json', '.pyc'
                    ]):
                        continue

                    if file_path.suffix == '.py':
                        # Analyze Python files
                        analysis = self.analyze_python_file(file_path, service_name)

                        # Create unique file node ID
                        relative_path = str(file_path.relative_to(root_path))
                        file_node = f"{service_name}::{relative_path}"

                        # Add file node to graph
                        self.graph.add_node(file_node,
                                          type="file",
                                          **asdict(analysis))

                        # Connect file to service
                        self.graph.add_edge(f"service:{service_name}", file_node,
                                          relationship="contains")

                        # Add import edges
                        for imp in analysis.imports:
                            if imp.startswith(('src', 'tower', 'api', 'app')):
                                self.graph.add_edge(file_node, f"module:{imp}",
                                                  relationship="imports")

                        # Add API call edges
                        for api_call in analysis.api_calls:
                            if api_call.startswith('/'):
                                self.graph.add_edge(file_node, f"endpoint:{api_call}",
                                                  relationship="calls")

                        # Add database query edges
                        for db_query in analysis.database_queries:
                            self.graph.add_edge(file_node, f"database:{db_query}",
                                              relationship="queries")

                        total_files += 1

                    else:
                        # Handle non-Python files (basic analysis)
                        relative_path = str(file_path.relative_to(root_path))
                        file_node = f"{service_name}::{relative_path}"

                        self.graph.add_node(file_node,
                                          type="file",
                                          path=str(file_path),
                                          service_name=service_name,
                                          file_type=file_path.suffix[1:] if file_path.suffix else "unknown")

                        # Connect file to service
                        self.graph.add_edge(f"service:{service_name}", file_node,
                                          relationship="contains")
                        total_files += 1

                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
                    continue

        # Add service-to-service relationships
        self._build_service_relationships()

        logger.info(f"Graph built with {self.graph.number_of_nodes()} nodes, "
                   f"{self.graph.number_of_edges()} edges from "
                   f"{total_services} services and {total_files} files")

        # Save the graph
        self.save_graph()

    def _build_service_relationships(self):
        """Build relationships between services based on dependencies."""
        for service_name, service_info in self.services.items():
            service_node = f"service:{service_name}"

            # Add port-based dependencies
            for dep in service_info.dependencies:
                if dep.startswith("port-"):
                    port = dep.replace("port-", "")
                    # Find service with this port
                    for other_name, other_info in self.services.items():
                        if other_info.port and str(other_info.port) == port:
                            self.graph.add_edge(service_node, f"service:{other_name}",
                                              relationship="depends_on",
                                              via=f"port:{port}")

            # Add database dependencies
            for db_ref in service_info.database_refs:
                self.graph.add_edge(service_node, f"database:{db_ref}",
                                  relationship="uses_database")

    def compute_spatial_positions(self):
        """Compute 3D positions for files based on relationships."""
        if not self.graph.nodes():
            return

        # Use spring layout in 3D
        pos_2d = nx.spring_layout(self.graph, k=1, iterations=50)

        # Convert to 3D by adding hierarchy level
        for node in self.graph.nodes():
            x, y = pos_2d[node]

            # Z-axis based on directory depth
            depth = node.count('/') if isinstance(node, str) else 0
            z = depth * 0.1

            self.file_positions[node] = (x, y, z)

        logger.info(f"Computed 3D positions for {len(self.file_positions)} files")

    def find_service_topology(self):
        """Map Tower's service architecture."""
        services = {
            'echo-brain': 8309,
            'anime-production': 8331,
            'knowledge-base': 8307,
            'dashboard': 8080,
            'auth': 8088,
            'plaid': 8089,
            'comfyui': 8188,
            'telegram-bot': None,
            'qdrant': 6333,
            'postgresql': 5432,
        }

        # Build service dependency map
        service_deps = {
            'echo-brain': ['postgresql', 'qdrant', 'knowledge-base'],
            'anime-production': ['comfyui', 'postgresql'],
            'dashboard': ['echo-brain', 'auth'],
            'telegram-bot': ['echo-brain', 'anime-production'],
        }

        for service, deps in service_deps.items():
            for dep in deps:
                self.graph.add_edge(f"service:{service}", f"service:{dep}")

        self.service_map = services
        logger.info(f"Mapped {len(services)} Tower services")

    def spatial_search(self, query: str, radius: float = 1.0) -> List[str]:
        """Find files spatially near a query point."""
        # TODO: Implement spatial proximity search
        # This would find files "near" the query in 3D space

        nearby_files = []

        # For now, return files with query in name
        for node in self.graph.nodes():
            if query.lower() in str(node).lower():
                nearby_files.append(node)

        return nearby_files[:10]

    def get_dependency_chain(self, start: str, end: str) -> List[str]:
        """Find dependency path between two files/services."""
        try:
            path = nx.shortest_path(self.graph, start, end)
            return path
        except nx.NetworkXNoPath:
            return []

    def visualize_region(self, center: str, depth: int = 2) -> Dict:
        """Get spatial region around a file/service."""
        region = {
            'center': center,
            'neighbors': {},
            'positions': {}
        }

        # Get neighbors within depth
        if center in self.graph:
            for d in range(1, depth + 1):
                neighbors_at_depth = []
                for node in self.graph.nodes():
                    try:
                        if nx.shortest_path_length(self.graph, center, node) == d:
                            neighbors_at_depth.append(node)
                    except:
                        pass

                region['neighbors'][d] = neighbors_at_depth

        # Include spatial positions
        for node in [center] + sum(region['neighbors'].values(), []):
            if node in self.file_positions:
                region['positions'][node] = self.file_positions[node]

        return region

    def understand_architecture(self) -> Dict:
        """Understand overall Tower architecture spatially."""
        architecture = {
            'layers': {},
            'services': self.service_map,
            'clusters': {},
            'critical_paths': []
        }

        # Identify layers (frontend, backend, database)
        for node in self.graph.nodes():
            if 'frontend' in str(node) or '.vue' in str(node):
                layer = 'frontend'
            elif 'api' in str(node) or 'routes' in str(node):
                layer = 'api'
            elif 'db' in str(node) or 'database' in str(node):
                layer = 'database'
            else:
                layer = 'core'

            if layer not in architecture['layers']:
                architecture['layers'][layer] = []
            architecture['layers'][layer].append(node)

        # Find critical paths (most connected nodes)
        centrality = nx.degree_centrality(self.graph)
        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        architecture['critical_paths'] = [node for node, _ in top_nodes]

        return architecture

# Test the spatial understanding
if __name__ == "__main__":
    spatial = SpatialCodeUnderstanding()

    # Build graph for Echo Brain
    spatial.build_codebase_graph(["/opt/tower-echo-brain"])

    # Compute positions
    spatial.compute_spatial_positions()

    # Map services
    spatial.find_service_topology()

    # Test search
    results = spatial.spatial_search("memory")
    print(f"Files related to 'memory': {results}")

    # Understand architecture
    arch = spatial.understand_architecture()
    print(f"Critical paths: {arch['critical_paths'][:5]}")