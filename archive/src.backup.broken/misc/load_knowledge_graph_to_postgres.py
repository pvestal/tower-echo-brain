#!/usr/bin/env python3
"""Load existing knowledge graph checkpoint and save to PostgreSQL."""

import pickle
import psycopg2
from psycopg2.extras import Json
import numpy as np
from pathlib import Path

def convert_numpy_types(obj):
    """Convert numpy types to regular Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def main():
    print("Loading knowledge graph checkpoint...")

    # Load checkpoint
    checkpoint_path = Path('/opt/tower-echo-brain/data/tower_knowledge_graph_checkpoint.pkl')
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    graph = checkpoint['graph']
    print(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

    # Connect to PostgreSQL
    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(
        host="localhost",
        database="echo_brain",
        user="patrick",
        password="***REMOVED***"
    )
    cursor = conn.cursor()

    # Create table if not exists
    print("Creating table...")
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
    print("Inserting nodes to PostgreSQL...")
    success = 0
    errors = 0

    for i, node in enumerate(graph.nodes()):
        if i % 1000 == 0:
            print(f"  Processed {i}/{graph.number_of_nodes()} nodes...")

        try:
            node_data = graph.nodes[node]
            edges = list(graph.edges(node))
            position_raw = node_data.get('position', [0, 0, 0])

            # Convert numpy types to regular Python types
            if hasattr(position_raw, '__iter__'):
                position = [float(p) for p in position_raw]
            else:
                position = [0.0, 0.0, 0.0]

            # Clean node_data of numpy types for JSON serialization
            clean_node_data = convert_numpy_types(node_data)

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
            success += 1

        except Exception as e:
            errors += 1
            print(f"  Error on node {node}: {e}")
            continue

    conn.commit()
    print(f"\nCompleted!")
    print(f"  Successfully saved: {success} nodes")
    print(f"  Errors: {errors}")

    # Verify
    cursor.execute("SELECT COUNT(*) FROM tower_knowledge_graph")
    count = cursor.fetchone()[0]
    print(f"  Total in database: {count} nodes")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()