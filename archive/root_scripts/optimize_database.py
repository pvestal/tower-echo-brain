#!/usr/bin/env python3
"""Optimize Echo Brain database with proper indexes and query improvements."""

import psycopg2
from psycopg2 import sql
import time

def optimize_database():
    """Add indexes to improve query performance."""

    conn = psycopg2.connect(
        host="localhost",
        database="echo_brain",
        user="patrick",
        password="tower_echo_brain_secret_key_2025"
    )
    cur = conn.cursor()

    indexes = [
        # Conversations table
        ("idx_conversations_user_id", "echo_conversations", "user_id"),
        ("idx_conversations_timestamp", "echo_conversations", "timestamp DESC"),
        ("idx_conversations_composite", "echo_conversations", "user_id, timestamp DESC"),

        # Unified interactions
        ("idx_unified_user_timestamp", "echo_unified_interactions", "user_id, timestamp DESC"),
        ("idx_unified_session", "echo_unified_interactions", "session_id"),
        ("idx_unified_context_type", "echo_unified_interactions", "context_type"),

        # Learning history
        ("idx_learning_timestamp", "learning_history", "timestamp DESC"),
        ("idx_learning_importance", "learning_history", "importance DESC"),
        ("idx_learning_composite", "learning_history", "importance DESC, timestamp DESC"),

        # Context registry
        ("idx_context_conversation", "context_registry", "conversation_id"),
        ("idx_context_created", "context_registry", "created_at DESC"),

        # Task queue
        ("idx_task_status", "task_queue", "status"),
        ("idx_task_priority", "task_queue", "priority DESC"),
        ("idx_task_composite", "task_queue", "status, priority DESC"),

        # Vector memories
        ("idx_vector_conversation", "vector_memories", "conversation_id"),
        ("idx_vector_created", "vector_memories", "created_at DESC"),

        # Agent state
        ("idx_agent_name", "agent_state", "agent_name"),
        ("idx_agent_updated", "agent_state", "updated_at DESC")
    ]

    created_count = 0
    for index_name, table_name, columns in indexes:
        try:
            # Check if index exists
            cur.execute("""
                SELECT 1 FROM pg_indexes
                WHERE indexname = %s
            """, (index_name,))

            if not cur.fetchone():
                print(f"Creating index {index_name} on {table_name}({columns})...")
                create_sql = sql.SQL("CREATE INDEX {} ON {} ({})").format(
                    sql.Identifier(index_name),
                    sql.Identifier(table_name),
                    sql.SQL(columns)
                )
                cur.execute(create_sql)
                conn.commit()
                created_count += 1
                print(f"  ✅ Created index {index_name}")
            else:
                print(f"  ⚠️  Index {index_name} already exists")

        except Exception as e:
            print(f"  ❌ Error creating {index_name}: {e}")
            conn.rollback()

    # Analyze tables for query optimization
    print("\n📊 Analyzing tables for query optimization...")
    tables_to_analyze = [
        "echo_conversations",
        "echo_unified_interactions",
        "learning_history",
        "context_registry",
        "task_queue",
        "vector_memories",
        "agent_state"
    ]

    for table in tables_to_analyze:
        try:
            cur.execute(sql.SQL("ANALYZE {}").format(sql.Identifier(table)))
            conn.commit()
            print(f"  ✅ Analyzed {table}")
        except Exception as e:
            print(f"  ❌ Error analyzing {table}: {e}")
            conn.rollback()

    # Vacuum tables to reclaim space
    print("\n🧹 Vacuuming tables...")
    conn.autocommit = True
    for table in tables_to_analyze:
        try:
            cur.execute(sql.SQL("VACUUM ANALYZE {}").format(sql.Identifier(table)))
            print(f"  ✅ Vacuumed {table}")
        except Exception as e:
            print(f"  ❌ Error vacuuming {table}: {e}")

    # Get table statistics
    print("\n📈 Table Statistics:")
    for table in tables_to_analyze:
        try:
            cur.execute("""
                SELECT
                    pg_size_pretty(pg_total_relation_size(%s)) as size,
                    reltuples::bigint as row_count
                FROM pg_class
                WHERE relname = %s
            """, (table, table))
            result = cur.fetchone()
            if result:
                print(f"  {table}: {result[0]}, ~{result[1]:,} rows")
        except Exception as e:
            print(f"  Error getting stats for {table}: {e}")

    cur.close()
    conn.close()

    print(f"\n✅ Database optimization complete! Created {created_count} new indexes.")

if __name__ == "__main__":
    optimize_database()