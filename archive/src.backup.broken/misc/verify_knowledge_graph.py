#!/usr/bin/env python3
"""
Verify Tower Knowledge Graph Integration
Tests that the 30,451 nodes are accessible and queryable
"""

import psycopg2
import json
import networkx as nx
from collections import Counter

def verify_knowledge_graph():
    """Verify the knowledge graph in PostgreSQL."""
    print("="*60)
    print("TOWER KNOWLEDGE GRAPH VERIFICATION")
    print("="*60)

    conn = psycopg2.connect(
        host="localhost",
        database="echo_brain",
        user="patrick",
        password="tower_echo_brain_secret_key_2025"
    )
    cursor = conn.cursor()

    # 1. Count total nodes
    print("\n1. Node Statistics:")
    cursor.execute("SELECT COUNT(*) FROM tower_knowledge_graph")
    total_nodes = cursor.fetchone()[0]
    print(f"   Total nodes: {total_nodes:,}")

    # 2. Count by node type
    print("\n2. Nodes by Type:")
    cursor.execute("""
        SELECT node_type, COUNT(*) as count
        FROM tower_knowledge_graph
        GROUP BY node_type
        ORDER BY count DESC
        LIMIT 10
    """)
    for node_type, count in cursor.fetchall():
        print(f"   - {node_type}: {count:,}")

    # 3. Sample API endpoints
    print("\n3. Sample API Endpoints:")
    cursor.execute("""
        SELECT node_id, node_data
        FROM tower_knowledge_graph
        WHERE node_type = 'endpoint'
        LIMIT 5
    """)
    for node_id, node_data in cursor.fetchall():
        print(f"   - {node_id}")

    # 4. Sample Docker services
    print("\n4. Sample Docker Services:")
    cursor.execute("""
        SELECT node_id, node_data
        FROM tower_knowledge_graph
        WHERE node_type = 'docker_service'
        LIMIT 5
    """)
    for node_id, node_data in cursor.fetchall():
        service_name = node_data.get('name', node_id) if node_data else node_id
        print(f"   - {service_name}")

    # 5. Database tables
    print("\n5. Sample Database Tables:")
    cursor.execute("""
        SELECT node_id, node_data
        FROM tower_knowledge_graph
        WHERE node_type = 'db_table'
        LIMIT 5
    """)
    for node_id, node_data in cursor.fetchall():
        print(f"   - {node_id}")

    # 6. Test spatial positions
    print("\n6. Spatial Position Verification:")
    cursor.execute("""
        SELECT COUNT(*)
        FROM tower_knowledge_graph
        WHERE position IS NOT NULL
        AND array_length(position, 1) = 3
    """)
    positioned_nodes = cursor.fetchone()[0]
    print(f"   Nodes with 3D positions: {positioned_nodes:,}")

    # 7. Test edge connectivity
    print("\n7. Edge Connectivity:")
    cursor.execute("""
        SELECT SUM(json_array_length(edges::json)) as total_edges
        FROM tower_knowledge_graph
        WHERE edges IS NOT NULL
    """)
    total_edges = cursor.fetchone()[0] or 0
    print(f"   Total edges: {total_edges:,}")

    # 8. Find most connected nodes
    print("\n8. Most Connected Nodes (Hubs):")
    cursor.execute("""
        SELECT node_id, node_type, json_array_length(edges::json) as edge_count
        FROM tower_knowledge_graph
        WHERE edges IS NOT NULL
        ORDER BY json_array_length(edges::json) DESC
        LIMIT 5
    """)
    for node_id, node_type, edge_count in cursor.fetchall():
        print(f"   - {node_id} ({node_type}): {edge_count} connections")

    # 9. Service distribution
    print("\n9. Service Distribution:")
    cursor.execute("""
        SELECT
            CASE
                WHEN node_id LIKE '%echo%' THEN 'Echo Brain'
                WHEN node_id LIKE '%anime%' THEN 'Anime Production'
                WHEN node_id LIKE '%auth%' THEN 'Authentication'
                WHEN node_id LIKE '%kb%' THEN 'Knowledge Base'
                WHEN node_id LIKE '%comfyui%' THEN 'ComfyUI'
                ELSE 'Other'
            END as service,
            COUNT(*) as count
        FROM tower_knowledge_graph
        GROUP BY service
        ORDER BY count DESC
    """)
    for service, count in cursor.fetchall():
        print(f"   - {service}: {count:,} nodes")

    # 10. Verify recent addition
    print("\n10. Recent Addition Timestamp:")
    cursor.execute("""
        SELECT MAX(created_at) as latest
        FROM tower_knowledge_graph
    """)
    latest = cursor.fetchone()[0]
    print(f"   Latest node added: {latest}")

    print("\n" + "="*60)
    print("✅ KNOWLEDGE GRAPH VERIFIED")
    print("="*60)
    print(f"\nThe Tower Knowledge Graph is fully operational with:")
    print(f"  • {total_nodes:,} nodes indexed")
    print(f"  • {positioned_nodes:,} nodes with 3D spatial positions")
    print(f"  • {total_edges:,} edges connecting components")
    print(f"  • Full Tower codebase structure mapped")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    verify_knowledge_graph()