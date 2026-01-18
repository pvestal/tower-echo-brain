"""Codebase indexer for creating searchable index of code entities"""
import os
import ast
import logging
from typing import Dict, List, Optional
import psycopg2
from psycopg2.extras import DictCursor

logger = logging.getLogger(__name__)

DB_CONFIG = {
    'host': 'localhost',
    'database': 'tower_consolidated',
    'user': 'patrick',
    'password': 'tower_echo_brain_secret_key_2025'
}

class CodebaseIndexer:
    """Indexes Python codebase for searchable entity extraction"""

    def __init__(self):
        self.entities = []

    def index_directory(self, directory: str, exclude_patterns: List[str] = None) -> int:
        """Index all Python files in directory"""
        exclude_patterns = exclude_patterns or [
            '__pycache__', '.git', '.venv', 'venv', 'node_modules'
        ]

        indexed_count = 0

        for root, dirs, files in os.walk(directory):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]

            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    try:
                        entities = self._parse_file(filepath)
                        self.entities.extend(entities)
                        indexed_count += len(entities)
                    except Exception as e:
                        logger.warning(f"Failed to parse {filepath}: {e}")

        return indexed_count

    def _parse_file(self, filepath: str) -> List[Dict]:
        """Parse Python file and extract entities"""
        entities = []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=filepath)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    entities.append({
                        'entity_type': 'function',
                        'entity_name': node.name,
                        'file_path': filepath,
                        'line_number': node.lineno,
                        'signature': self._get_function_signature(node),
                        'docstring': ast.get_docstring(node) or ''
                    })

                elif isinstance(node, ast.ClassDef):
                    entities.append({
                        'entity_type': 'class',
                        'entity_name': node.name,
                        'file_path': filepath,
                        'line_number': node.lineno,
                        'signature': f"class {node.name}",
                        'docstring': ast.get_docstring(node) or ''
                    })

                elif isinstance(node, ast.AsyncFunctionDef):
                    entities.append({
                        'entity_type': 'async_function',
                        'entity_name': node.name,
                        'file_path': filepath,
                        'line_number': node.lineno,
                        'signature': self._get_async_function_signature(node),
                        'docstring': ast.get_docstring(node) or ''
                    })

        except Exception as e:
            logger.error(f"Error parsing {filepath}: {e}")

        return entities

    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature"""
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        return f"def {node.name}({', '.join(args)})"

    def _get_async_function_signature(self, node: ast.AsyncFunctionDef) -> str:
        """Extract async function signature"""
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        return f"async def {node.name}({', '.join(args)})"

    def save_to_database(self) -> int:
        """Save indexed entities to database"""
        try:
            with psycopg2.connect(**DB_CONFIG) as conn:
                with conn.cursor() as cur:
                    # Clear existing entries
                    cur.execute("DELETE FROM codebase_index")

                    # Insert new entries
                    for entity in self.entities:
                        cur.execute("""
                            INSERT INTO codebase_index
                            (entity_type, entity_name, file_path, line_number, signature, docstring)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """, (
                            entity['entity_type'],
                            entity['entity_name'],
                            entity['file_path'],
                            entity['line_number'],
                            entity['signature'],
                            entity['docstring']
                        ))

                    conn.commit()
                    return len(self.entities)

        except Exception as e:
            logger.error(f"Database save failed: {e}")
            return 0

    def search_entities(self, query: str, limit: int = 10) -> List[Dict]:
        """Search indexed entities"""
        try:
            with psycopg2.connect(**DB_CONFIG) as conn:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    cur.execute("""
                        SELECT entity_type, entity_name, file_path, line_number, signature
                        FROM codebase_index
                        WHERE entity_name ILIKE %s
                           OR signature ILIKE %s
                           OR docstring ILIKE %s
                        ORDER BY entity_name
                        LIMIT %s
                    """, (f'%{query}%', f'%{query}%', f'%{query}%', limit))

                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Entity search failed: {e}")
            return []