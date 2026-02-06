"""Schema Indexer Worker - Indexes Echo Brain's database structure for self-awareness"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any

import httpx
import asyncpg

logger = logging.getLogger(__name__)


class SchemaIndexer:
    """Indexes Echo Brain's own database structure for self-awareness"""

    def __init__(self):
        self.db_url = os.environ.get("DATABASE_URL",
            "postgresql://echo:echo_secure_password_123@localhost/echo_brain")
        self.qdrant_url = "http://localhost:6333"
        self.ollama_url = "http://localhost:11434"
        self.collection = "echo_memory"

    async def run_cycle(self):
        """Main worker cycle — called by scheduler"""
        logger.info("🗄️ Schema Indexer starting cycle")

        try:
            conn = await asyncpg.connect(self.db_url)

            # Get all tables in public schema
            tables = await conn.fetch("""
                SELECT
                    t.table_name,
                    t.table_schema,
                    obj_description(c.oid, 'pg_class') as table_comment
                FROM information_schema.tables t
                JOIN pg_catalog.pg_class c ON c.relname = t.table_name
                JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace AND n.nspname = t.table_schema
                WHERE t.table_schema = 'public'
                    AND t.table_type = 'BASE TABLE'
                ORDER BY t.table_name
            """)

            logger.info(f"Found {len(tables)} tables to index")

            tables_indexed = 0
            tables_unchanged = 0

            for table in tables:
                try:
                    # Get detailed table info
                    table_info = await self._get_table_info(conn, table['table_name'])

                    # Check if schema has changed
                    existing = await conn.fetchrow(
                        "SELECT column_info FROM self_schema_index WHERE table_name = $1",
                        table['table_name']
                    )

                    if existing and json.loads(existing['column_info']) == table_info['columns']:
                        tables_unchanged += 1
                        continue

                    # Generate natural language description
                    description = await self._generate_description(table_info)

                    # Embed and store in Qdrant
                    point_id = await self._embed_description(description, table['table_name'])

                    # Store metadata in database
                    await conn.execute("""
                        INSERT INTO self_schema_index
                        (table_name, schema_name, column_info, row_count, index_info, foreign_keys, qdrant_point_id)
                        VALUES ($1, $2, $3::jsonb, $4, $5::jsonb, $6::jsonb, $7)
                        ON CONFLICT (schema_name, table_name)
                        DO UPDATE SET
                            column_info = EXCLUDED.column_info,
                            row_count = EXCLUDED.row_count,
                            index_info = EXCLUDED.index_info,
                            foreign_keys = EXCLUDED.foreign_keys,
                            qdrant_point_id = EXCLUDED.qdrant_point_id,
                            last_indexed_at = NOW()
                    """,
                        table['table_name'], 'public',
                        json.dumps(table_info['columns']),
                        table_info['row_count'],
                        json.dumps(table_info['indexes']),
                        json.dumps(table_info['foreign_keys']),
                        point_id
                    )

                    tables_indexed += 1

                except Exception as e:
                    logger.error(f"Failed to index table {table['table_name']}: {e}")
                    continue

            # Record metrics
            await conn.execute("""
                INSERT INTO self_health_metrics (metric_name, metric_value, metadata)
                VALUES
                    ('schema_indexer_tables_indexed', $1, $2::jsonb),
                    ('schema_indexer_tables_total', $3, $4::jsonb)
            """,
                float(tables_indexed),
                json.dumps({"cycle_time": datetime.now(timezone.utc).isoformat()}),
                float(len(tables)),
                json.dumps({"tables_unchanged": tables_unchanged})
            )

            await conn.close()

            logger.info(f"✅ Schema Indexer completed: {tables_indexed} tables indexed, "
                       f"{tables_unchanged} unchanged")

        except Exception as e:
            logger.error(f"❌ Schema Indexer cycle failed: {e}", exc_info=True)

            # Try to record the failure
            try:
                conn = await asyncpg.connect(self.db_url)
                await conn.execute("""
                    INSERT INTO self_detected_issues
                    (issue_type, severity, source, title, description, related_worker)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """,
                    "worker_failure", "critical", "schema_indexer",
                    "Schema Indexer cycle failed",
                    str(e), "schema_indexer"
                )
                await conn.close()
            except:
                pass

    async def _get_table_info(self, conn: asyncpg.Connection, table_name: str) -> Dict[str, Any]:
        """Get comprehensive information about a table"""

        # Get columns
        columns = await conn.fetch("""
            SELECT
                column_name,
                data_type,
                is_nullable,
                column_default,
                character_maximum_length,
                numeric_precision
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = $1
            ORDER BY ordinal_position
        """, table_name)

        column_info = []
        for col in columns:
            column_info.append({
                "name": col['column_name'],
                "type": col['data_type'],
                "nullable": col['is_nullable'] == 'YES',
                "default": col['column_default'],
                "max_length": col['character_maximum_length'],
                "precision": col['numeric_precision']
            })

        # Get indexes
        indexes = await conn.fetch("""
            SELECT
                i.relname as index_name,
                a.attname as column_name,
                ix.indisunique as is_unique,
                ix.indisprimary as is_primary
            FROM pg_index ix
            JOIN pg_class t ON t.oid = ix.indrelid
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
            WHERE t.relname = $1
            ORDER BY i.relname, a.attnum
        """, table_name)

        # Group indexes by name
        index_info = {}
        for idx in indexes:
            name = idx['index_name']
            if name not in index_info:
                index_info[name] = {
                    "name": name,
                    "columns": [],
                    "unique": idx['is_unique'],
                    "primary": idx['is_primary']
                }
            index_info[name]['columns'].append(idx['column_name'])

        # Get foreign keys
        foreign_keys = await conn.fetch("""
            SELECT
                kcu.column_name,
                ccu.table_name AS referenced_table,
                ccu.column_name AS referenced_column
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_name = $1
        """, table_name)

        fk_info = []
        for fk in foreign_keys:
            fk_info.append({
                "column": fk['column_name'],
                "references_table": fk['referenced_table'],
                "references_column": fk['referenced_column']
            })

        # Get row count
        row_count = await conn.fetchval(f"""
            SELECT reltuples::BIGINT as estimate
            FROM pg_class
            WHERE relname = $1
        """, table_name)

        # If estimate is 0, try actual count (but with timeout)
        if row_count == 0:
            try:
                row_count = await conn.fetchval(
                    f"SELECT COUNT(*) FROM {table_name}",
                    timeout=5.0
                )
            except:
                row_count = 0

        return {
            "table_name": table_name,
            "columns": column_info,
            "indexes": list(index_info.values()),
            "foreign_keys": fk_info,
            "row_count": row_count or 0
        }

    async def _generate_description(self, table_info: Dict[str, Any]) -> str:
        """Generate a natural language description of the table"""

        # Build description
        desc = f"Table '{table_info['table_name']}' "

        # Add row count
        if table_info['row_count'] > 0:
            desc += f"stores {table_info['row_count']:,} records. "
        else:
            desc += "is currently empty. "

        # Add purpose based on table name patterns
        if 'conversation' in table_info['table_name'].lower():
            desc += "This table stores conversation data. "
        elif 'fact' in table_info['table_name'].lower():
            desc += "This table stores extracted facts and knowledge. "
        elif 'vector' in table_info['table_name'].lower():
            desc += "This table stores vector embeddings and search data. "
        elif 'goal' in table_info['table_name'].lower():
            desc += "This table stores autonomous system goals. "
        elif 'notification' in table_info['table_name'].lower():
            desc += "This table stores system notifications. "
        elif 'self_' in table_info['table_name']:
            desc += "This table is part of the self-awareness system. "

        # Add column summary
        desc += f"\n\nColumns ({len(table_info['columns'])}):\n"
        for col in table_info['columns'][:10]:  # Limit to first 10 columns
            desc += f"- {col['name']} ({col['type']}"
            if not col['nullable']:
                desc += ", NOT NULL"
            if col['default']:
                desc += f", default: {col['default'][:30]}"
            desc += ")\n"

        if len(table_info['columns']) > 10:
            desc += f"... and {len(table_info['columns']) - 10} more columns\n"

        # Add indexes
        if table_info['indexes']:
            desc += f"\nIndexes ({len(table_info['indexes'])}):\n"
            for idx in table_info['indexes'][:5]:
                desc += f"- {idx['name']} on columns: {', '.join(idx['columns'])}"
                if idx['primary']:
                    desc += " (PRIMARY KEY)"
                elif idx['unique']:
                    desc += " (UNIQUE)"
                desc += "\n"

        # Add foreign keys
        if table_info['foreign_keys']:
            desc += f"\nForeign key relationships:\n"
            for fk in table_info['foreign_keys'][:5]:
                desc += f"- {fk['column']} references {fk['references_table']}.{fk['references_column']}\n"

        return desc

    async def _embed_description(self, description: str, table_name: str) -> str:
        """Embed the table description and store in Qdrant"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get embedding from Ollama
                embed_response = await client.post(
                    f"{self.ollama_url}/api/embed",
                    json={
                        "model": "mxbai-embed-large:latest",
                        "input": description
                    }
                )

                if embed_response.status_code != 200:
                    logger.error(f"Embedding failed: {embed_response.text}")
                    return ""

                embedding = embed_response.json()["embeddings"][0]

                # Generate unique point ID
                point_id = str(uuid.uuid4())

                # Store in Qdrant
                payload = {
                    "source": "self_schema",
                    "table_name": table_name,
                    "content_type": "database_schema",
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                    "description": description[:1000]  # Store preview
                }

                upsert_response = await client.put(
                    f"{self.qdrant_url}/collections/{self.collection}/points",
                    json={
                        "points": [{
                            "id": point_id,
                            "vector": embedding,
                            "payload": payload
                        }]
                    }
                )

                if upsert_response.status_code == 200:
                    return point_id
                else:
                    logger.error(f"Qdrant upsert failed: {upsert_response.text}")
                    return ""

        except Exception as e:
            logger.error(f"Failed to embed/store description: {e}")
            return ""