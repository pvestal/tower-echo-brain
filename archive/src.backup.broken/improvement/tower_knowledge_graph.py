#!/usr/bin/env python3
"""
Tower Knowledge Graph Builder - Maps 141,957 files into spatial intelligence.
Designed by Claude Experts: Architect + Infrastructure Specialists
"""

import os
import ast
import json
import pickle
import hashlib
import asyncio
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime
import networkx as nx
import numpy as np
from collections import defaultdict
import psycopg2
from psycopg2.extras import Json
import redis
import httpx

class TowerKnowledgeGraph:
    """Spatial knowledge graph for Tower's entire codebase."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.service_map = {}
        self.api_endpoints = defaultdict(list)
        self.db_connections = defaultdict(set)
        self.import_graph = defaultdict(set)
        self.file_vectors = {}  # File path -> vector embedding

        # Storage backends
        self.db_conn = psycopg2.connect(
            host="localhost",
            database="echo_brain",
            user="patrick",
            password="***REMOVED***"
        )
        self.redis = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.qdrant_url = "http://localhost:6333"

        # Tower directories to scan
        self.tower_paths = [
            "/opt/tower-echo-brain",
            "/opt/tower-anime-production",
            "/opt/tower-auth",
            "/opt/tower-dashboard",
            "/opt/tower-kb",
            "/opt/tower-apple-music",
            "/opt/tower-agent-manager",
            "/opt/tower-crypto-trader",
            "/opt/tower-loan-search",
            "/home/patrick/Tower",
            "/mnt/1TB-storage/ComfyUI"
        ]

        self.stats = {
            "total_files": 0,
            "python_files": 0,
            "js_files": 0,
            "vue_files": 0,
            "services_found": 0,
            "api_endpoints": 0,
            "db_tables": 0,
            "import_relationships": 0
        }

    def scan_codebase(self):
        """Scan entire Tower codebase and build knowledge graph."""
        print(f"Starting Tower codebase scan at {datetime.now()}")

        for base_path in self.tower_paths:
            if not os.path.exists(base_path):
                continue

            print(f"Scanning {base_path}...")
            for root, dirs, files in os.walk(base_path):
                # Skip node_modules, .git, __pycache__
                dirs[:] = [d for d in dirs if d not in {
                    'node_modules', '.git', '__pycache__', 'venv', '.venv'
                }]

                for file in files:
                    filepath = os.path.join(root, file)
                    self.stats["total_files"] += 1

                    if file.endswith('.py'):
                        self._analyze_python_file(filepath)
                        self.stats["python_files"] += 1
                    elif file.endswith(('.js', '.ts')):
                        self._analyze_javascript_file(filepath)
                        self.stats["js_files"] += 1
                    elif file.endswith('.vue'):
                        self._analyze_vue_file(filepath)
                        self.stats["vue_files"] += 1
                    elif file in ('docker-compose.yml', 'Dockerfile'):
                        self._analyze_docker_file(filepath)
                    elif file.endswith(('.service', '.conf')):
                        self._analyze_config_file(filepath)

                    # Add file node to graph
                    rel_path = os.path.relpath(filepath, '/')
                    self.graph.add_node(rel_path, type='file', ext=Path(file).suffix)

                    # Every 1000 files, save progress
                    if self.stats["total_files"] % 1000 == 0:
                        print(f"  Processed {self.stats['total_files']} files...")
                        self._save_checkpoint()

    def _analyze_python_file(self, filepath):
        """Analyze Python file for imports, classes, functions, API endpoints."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Parse AST
            try:
                tree = ast.parse(content)
            except:
                return

            rel_path = os.path.relpath(filepath, '/')

            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        self.import_graph[rel_path].add(node.module)
                        # Create edge in graph
                        self.graph.add_edge(rel_path, node.module, type='imports')
                        self.stats["import_relationships"] += 1

            # Find FastAPI/Flask routes
            if 'router' in content or 'app.route' in content or '@app.' in content:
                routes = self._extract_api_routes(content, filepath)
                for route in routes:
                    self.api_endpoints[rel_path].append(route)
                    self.graph.add_node(route, type='endpoint')
                    self.graph.add_edge(rel_path, route, type='defines')
                    self.stats["api_endpoints"] += 1

            # Find database operations
            if 'psycopg' in content or 'asyncpg' in content or 'SELECT' in content:
                tables = self._extract_db_tables(content)
                for table in tables:
                    self.db_connections[rel_path].add(table)
                    self.graph.add_node(table, type='db_table')
                    self.graph.add_edge(rel_path, table, type='queries')
                    self.stats["db_tables"] += 1

        except Exception as e:
            pass  # Skip files we can't read

    def _analyze_javascript_file(self, filepath):
        """Analyze JS/TS files for components, API calls, imports."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            rel_path = os.path.relpath(filepath, '/')

            # Find imports
            import_lines = [l for l in content.split('\n') if l.strip().startswith('import')]
            for line in import_lines:
                if 'from' in line:
                    module = line.split('from')[1].strip().strip(';').strip('"').strip("'")
                    self.graph.add_edge(rel_path, module, type='imports')

            # Find API calls
            if 'fetch(' in content or 'axios' in content or 'httpx' in content:
                endpoints = self._extract_js_endpoints(content)
                for endpoint in endpoints:
                    self.graph.add_node(endpoint, type='api_call')
                    self.graph.add_edge(rel_path, endpoint, type='calls')

        except Exception:
            pass

    def _analyze_vue_file(self, filepath):
        """Analyze Vue components for structure and dependencies."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            rel_path = os.path.relpath(filepath, '/')
            component_name = Path(filepath).stem

            # Add component node
            self.graph.add_node(component_name, type='vue_component')
            self.graph.add_edge(rel_path, component_name, type='defines')

            # Find props, emits, imports
            if '<script' in content:
                script_content = content.split('<script')[1].split('</script>')[0]
                # Extract imports similar to JS files
                import_lines = [l for l in script_content.split('\n') if 'import' in l]
                for line in import_lines:
                    if 'from' in line:
                        module = line.split('from')[1].strip().strip(';').strip('"').strip("'")
                        self.graph.add_edge(rel_path, module, type='imports')

        except Exception:
            pass

    def _analyze_docker_file(self, filepath):
        """Analyze Docker files for service definitions."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            rel_path = os.path.relpath(filepath, '/')

            # Find services in docker-compose
            if 'services:' in content:
                lines = content.split('\n')
                in_services = False
                for line in lines:
                    if 'services:' in line:
                        in_services = True
                    elif in_services and line and not line[0].isspace():
                        in_services = False
                    elif in_services and line.strip() and not line.strip().startswith('#'):
                        service_name = line.strip().rstrip(':')
                        if service_name:
                            self.graph.add_node(service_name, type='docker_service')
                            self.graph.add_edge(rel_path, service_name, type='defines')
                            self.stats["services_found"] += 1

        except Exception:
            pass

    def _analyze_config_file(self, filepath):
        """Analyze configuration files for service definitions."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            rel_path = os.path.relpath(filepath, '/')

            # Systemd service files
            if filepath.endswith('.service'):
                service_name = Path(filepath).stem
                self.service_map[service_name] = {
                    'file': rel_path,
                    'type': 'systemd'
                }
                self.graph.add_node(service_name, type='systemd_service')
                self.stats["services_found"] += 1

            # Nginx configs
            elif filepath.endswith('.conf'):
                if 'server_name' in content or 'location' in content:
                    self.graph.add_node(rel_path, type='nginx_config')

        except Exception:
            pass

    def _extract_api_routes(self, content: str, filepath: str) -> List[str]:
        """Extract API routes from Python files."""
        routes = []
        lines = content.split('\n')

        for i, line in enumerate(lines):
            if '@router.' in line or '@app.route' in line or '@app.' in line:
                # Try to extract the route
                if '(' in line and ')' in line:
                    route_part = line.split('(')[1].split(')')[0]
                    if '"' in route_part:
                        route = route_part.split('"')[1]
                        routes.append(route)
                    elif "'" in route_part:
                        route = route_part.split("'")[1]
                        routes.append(route)

        return routes

    def _extract_db_tables(self, content: str) -> Set[str]:
        """Extract database table names from SQL queries."""
        tables = set()

        # Common patterns
        patterns = [
            'FROM (\\w+)',
            'INTO (\\w+)',
            'UPDATE (\\w+)',
            'CREATE TABLE (\\w+)',
            'ALTER TABLE (\\w+)',
            'DROP TABLE (\\w+)'
        ]

        import re
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            tables.update(matches)

        # Filter out common SQL keywords
        tables = {t for t in tables if t.lower() not in {
            'select', 'where', 'and', 'or', 'not', 'null', 'true', 'false'
        }}

        return tables

    def _extract_js_endpoints(self, content: str) -> List[str]:
        """Extract API endpoints from JavaScript fetch/axios calls."""
        endpoints = []

        import re
        # Match fetch('...') or axios.get('...')
        patterns = [
            r"fetch\(['\"]([^'\"]+)['\"]",
            r"axios\.\w+\(['\"]([^'\"]+)['\"]",
            r"http://[^\s'\"]+",
            r"https://[^\s'\"]+",
            r"/api/[^\s'\"]*"
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content)
            endpoints.extend(matches)

        return endpoints

    def compute_spatial_positions(self):
        """Compute 3D spatial positions for all nodes using force-directed layout."""
        print("Computing spatial positions...")

        # Use spring layout for initial 2D positions
        pos_2d = nx.spring_layout(self.graph, k=1, iterations=50)

        # Add Z dimension based on node type
        type_heights = {
            'endpoint': 1.0,
            'docker_service': 0.9,
            'systemd_service': 0.9,
            'vue_component': 0.7,
            'db_table': 0.3,
            'file': 0.5
        }

        positions = {}
        for node, (x, y) in pos_2d.items():
            node_type = self.graph.nodes[node].get('type', 'file')
            z = type_heights.get(node_type, 0.5)
            positions[node] = (x, y, z)
            self.graph.nodes[node]['position'] = positions[node]

        return positions

    def _save_checkpoint(self):
        """Save current progress to disk."""
        checkpoint = {
            'graph': self.graph,
            'service_map': self.service_map,
            'api_endpoints': dict(self.api_endpoints),
            'db_connections': {k: list(v) for k, v in self.db_connections.items()},
            'import_graph': {k: list(v) for k, v in self.import_graph.items()},
            'stats': self.stats,
            'timestamp': datetime.now().isoformat()
        }

        # Save to pickle
        checkpoint_path = Path('/opt/tower-echo-brain/data/tower_knowledge_graph_checkpoint.pkl')
        checkpoint_path.parent.mkdir(exist_ok=True)
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)

        # Cache key metrics in Redis
        self.redis.hset('echo:knowledge_graph', 'total_nodes', self.graph.number_of_nodes())
        self.redis.hset('echo:knowledge_graph', 'total_edges', self.graph.number_of_edges())
        self.redis.hset('echo:knowledge_graph', 'services', len(self.service_map))
        self.redis.hset('echo:knowledge_graph', 'last_update', datetime.now().isoformat())

    def _convert_numpy_types(self, obj):
        """Convert numpy types to regular Python types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj

    def save_to_postgres(self):
        """Save graph to PostgreSQL for persistence."""
        cursor = self.db_conn.cursor()

        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tower_knowledge_graph (
                id SERIAL PRIMARY KEY,
                node_id TEXT UNIQUE,
                node_type TEXT,
                node_data JSONB,
                edges JSONB,
                position FLOAT[],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Insert nodes
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            edges = list(self.graph.edges(node))
            position_raw = node_data.get('position', [0, 0, 0])

            # Convert numpy types to regular Python types
            if hasattr(position_raw, '__iter__'):
                position = [float(p) for p in position_raw]
            else:
                position = [0.0, 0.0, 0.0]

            # Clean node_data of numpy types for JSON serialization
            clean_node_data = self._convert_numpy_types(node_data)

            cursor.execute("""
                INSERT INTO tower_knowledge_graph (node_id, node_type, node_data, edges, position)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (node_id) DO UPDATE
                SET node_data = EXCLUDED.node_data,
                    edges = EXCLUDED.edges,
                    position = EXCLUDED.position
            """, (
                node,
                clean_node_data.get('type', 'unknown'),
                Json(clean_node_data),
                Json(edges),
                position
            ))

        self.db_conn.commit()
        print(f"Saved {self.graph.number_of_nodes()} nodes to PostgreSQL")

    def save_final(self):
        """Save final graph to disk and databases."""
        # Save to pickle
        final_path = Path('/opt/tower-echo-brain/data/tower_knowledge_graph.pkl')
        final_path.parent.mkdir(exist_ok=True)

        graph_data = {
            'graph': self.graph,
            'service_map': self.service_map,
            'api_endpoints': dict(self.api_endpoints),
            'db_connections': {k: list(v) for k, v in self.db_connections.items()},
            'import_graph': {k: list(v) for k, v in self.import_graph.items()},
            'stats': self.stats,
            'positions': {n: self.graph.nodes[n].get('position') for n in self.graph.nodes()},
            'created': datetime.now().isoformat()
        }

        with open(final_path, 'wb') as f:
            pickle.dump(graph_data, f)

        # Save to PostgreSQL
        self.save_to_postgres()

        # Update Redis cache
        self._save_checkpoint()

        print(f"""
        ========================================
        TOWER KNOWLEDGE GRAPH COMPLETE
        ========================================
        Total Files Scanned: {self.stats['total_files']:,}
        Python Files: {self.stats['python_files']:,}
        JavaScript Files: {self.stats['js_files']:,}
        Vue Components: {self.stats['vue_files']:,}

        Graph Statistics:
        - Nodes: {self.graph.number_of_nodes():,}
        - Edges: {self.graph.number_of_edges():,}
        - Services: {self.stats['services_found']:,}
        - API Endpoints: {self.stats['api_endpoints']:,}
        - DB Tables: {self.stats['db_tables']:,}
        - Import Relations: {self.stats['import_relationships']:,}

        Saved to:
        - /opt/tower-echo-brain/data/tower_knowledge_graph.pkl
        - PostgreSQL: tower_knowledge_graph table
        - Redis: echo:knowledge_graph keys
        ========================================
        """)

    def query_graph(self, query_type: str, target: str = None):
        """Query the knowledge graph."""
        if query_type == 'dependencies':
            # Get all dependencies of a file/service
            if target in self.graph:
                deps = list(self.graph.successors(target))
                return deps

        elif query_type == 'dependents':
            # Get all files that depend on target
            if target in self.graph:
                deps = list(self.graph.predecessors(target))
                return deps

        elif query_type == 'service_map':
            # Get all services and their endpoints
            services = {}
            for node in self.graph.nodes():
                if self.graph.nodes[node].get('type') in {'docker_service', 'systemd_service'}:
                    services[node] = {
                        'endpoints': [n for n in self.graph.successors(node)
                                    if self.graph.nodes[n].get('type') == 'endpoint'],
                        'tables': [n for n in self.graph.successors(node)
                                 if self.graph.nodes[n].get('type') == 'db_table']
                    }
            return services

        elif query_type == 'stats':
            return self.stats

        return None


def main():
    """Build Tower knowledge graph."""
    builder = TowerKnowledgeGraph()

    print("Building Tower Knowledge Graph...")
    print(f"Scanning {len(builder.tower_paths)} Tower directories...")

    # Scan codebase
    builder.scan_codebase()

    # Compute spatial positions
    builder.compute_spatial_positions()

    # Save everything
    builder.save_final()

    print("\nKnowledge Graph construction complete!")
    print("Access via API: http://localhost:8309/api/echo/improvement/knowledge-graph")


if __name__ == "__main__":
    main()