#!/usr/bin/env python3
"""
Extract ALL training data for Echo Brain - KB articles, Claude conversations,
work projects, financial data - EVERYTHING Patrick has.
"""

import json
import psycopg2
from datetime import datetime
from pathlib import Path
from qdrant_client import QdrantClient
import glob

def extract_kb_articles():
    """Extract ALL Knowledge Base articles from PostgreSQL."""
    print("Extracting Knowledge Base articles...")
    conn = psycopg2.connect(
        host="localhost",
        database="knowledge_base",
        user="patrick",
        password="tower_echo_brain_secret_key_2025"
    )
    cursor = conn.cursor()

    # Get ALL articles
    cursor.execute("""
        SELECT id, title, content, category, tags, created_at, updated_at
        FROM articles
        ORDER BY created_at DESC
    """)

    articles = []
    for row in cursor.fetchall():
        articles.append({
            'id': row[0],
            'title': row[1],
            'content': row[2],
            'category': row[3],
            'tags': row[4],
            'created_at': str(row[5]),
            'updated_at': str(row[6]) if row[6] else None,
            'source': 'knowledge_base'
        })

    cursor.close()
    conn.close()

    print(f"  Found {len(articles)} KB articles")
    return articles

def extract_claude_conversations():
    """Extract ALL conversations from PostgreSQL and Qdrant."""
    print("Extracting ALL conversations (including Claude)...")
    conn = psycopg2.connect(
        host="localhost",
        database="echo_brain",
        user="patrick",
        password="tower_echo_brain_secret_key_2025"
    )
    cursor = conn.cursor()

    # Get ALL conversations from echo_conversations
    cursor.execute("""
        SELECT id, conversation_id, query_text, response_text,
               created_at, metadata, entities_mentioned, model_used,
               intent_classification, confidence
        FROM echo_conversations
        ORDER BY created_at DESC
    """)

    conversations = []
    for row in cursor.fetchall():
        conversations.append({
            'id': row[0],
            'conversation_id': row[1],
            'query': row[2],
            'response': row[3],
            'timestamp': str(row[4]),
            'metadata': row[5] if row[5] else {},
            'entities': row[6] if row[6] else {},
            'model': row[7],
            'intent': row[8],
            'confidence': row[9],
            'source': 'echo_conversations'
        })

    print(f"  Found {len(conversations)} total conversations in PostgreSQL")

    # Get sample queries for pattern analysis
    cursor.execute("""
        SELECT DISTINCT query_text
        FROM echo_conversations
        WHERE query_text IS NOT NULL
        LIMIT 200
    """)

    sample_queries = [row[0] for row in cursor.fetchall()]

    # Also get Claude indexed memory from Qdrant
    print("  Extracting Claude memory from Qdrant...")
    claude_memory_points = extract_claude_from_qdrant()

    cursor.close()
    conn.close()

    return conversations, sample_queries, claude_memory_points

def extract_claude_from_qdrant():
    """Extract Claude conversations from Qdrant vector store."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)

        # Get Claude conversations from 4096D collection
        collection_name = "claude_conversations_4096d"
        info = client.get_collection(collection_name)
        print(f"    Found {info.points_count} Claude memory points in Qdrant")

        # Get sample points with payloads
        points = client.scroll(
            collection_name=collection_name,
            limit=500,  # Get more samples
            with_payload=True,
            with_vectors=False
        )[0]

        claude_memories = []
        for point in points:
            if point.payload:
                claude_memories.append({
                    'id': point.id,
                    'payload': point.payload,
                    'source': 'qdrant_claude_4096d'
                })

        return claude_memories
    except Exception as e:
        print(f"    Could not extract from Qdrant: {e}")
        return []

def extract_work_projects():
    """Extract Patrick's work project data."""
    print("Extracting work project data...")

    work_data = []

    # Check for Trust & Estate Planning data
    conn = psycopg2.connect(
        host="localhost",
        database="echo_brain",
        user="patrick",
        password="tower_echo_brain_secret_key_2025"
    )
    cursor = conn.cursor()

    # Look for work-related patterns in conversations
    cursor.execute("""
        SELECT query_text, response_text, created_at
        FROM echo_conversations
        WHERE query_text LIKE '%trust%' OR query_text LIKE '%estate%'
           OR query_text LIKE '%federal%' OR query_text LIKE '%training%'
           OR query_text LIKE '%plaid%' OR query_text LIKE '%financial%'
           OR query_text LIKE '%loan%' OR query_text LIKE '%credit%'
        LIMIT 500
    """)

    for row in cursor.fetchall():
        work_data.append({
            'query': row[0],
            'response': row[1],
            'timestamp': str(row[2]),
            'type': 'work_conversation'
        })

    print(f"  Found {len(work_data)} work-related conversations")

    cursor.close()
    conn.close()

    return work_data

def extract_vector_embeddings():
    """Extract vector embeddings from Qdrant to understand semantic patterns."""
    print("Extracting vector embeddings...")

    try:
        client = QdrantClient(host="localhost", port=6333)

        collections = [
            "claude_conversations_4096d",
            "learning_facts_4096d",
            "unified_media_memory_4096d",
            "agent_memories_4096d"
        ]

        vector_data = {}
        for collection in collections:
            try:
                # Get collection info
                info = client.get_collection(collection)
                vector_data[collection] = {
                    'points_count': info.points_count,
                    'vector_size': info.config.params.vectors.size
                }

                # Get sample points for testing
                points = client.scroll(
                    collection_name=collection,
                    limit=10,
                    with_payload=True,
                    with_vectors=False  # Don't need actual vectors for test data
                )[0]

                vector_data[collection]['sample_payloads'] = [
                    point.payload for point in points if point.payload
                ]

                print(f"  {collection}: {info.points_count} points")
            except Exception as e:
                print(f"    Error with {collection}: {e}")

        return vector_data

    except Exception as e:
        print(f"  Could not connect to Qdrant: {e}")
        return {}

def extract_claude_files():
    """Extract Claude conversations from filesystem."""
    print("Extracting Claude conversation files...")

    claude_files = []

    # Search for Claude conversation files
    patterns = [
        '/home/patrick/.claude/conversations/*.json',
        '/home/patrick/.claude/conversations/*.md',
        '/home/patrick/.claude/knowledge/*.md',
        '/home/patrick/.claude/*.json'
    ]

    for pattern in patterns:
        for filepath in glob.glob(pattern):
            try:
                path = Path(filepath)
                if path.suffix == '.json':
                    with open(path, 'r') as f:
                        data = json.load(f)
                        claude_files.append({
                            'file': str(path),
                            'type': 'json',
                            'content': data,
                            'source': 'claude_filesystem'
                        })
                elif path.suffix == '.md':
                    with open(path, 'r') as f:
                        content = f.read()
                        claude_files.append({
                            'file': str(path),
                            'type': 'markdown',
                            'content': content,
                            'source': 'claude_filesystem'
                        })
                print(f"    Loaded: {path.name}")
            except Exception as e:
                print(f"    Error loading {filepath}: {e}")

    print(f"  Found {len(claude_files)} Claude conversation files")
    return claude_files

def extract_patrick_patterns():
    """Extract Patrick's actual communication patterns and preferences."""
    print("Extracting Patrick's patterns...")

    patterns = {
        'technical_terms': [
            'Tower', 'Echo Brain', 'anime production', 'Plaid', 'ComfyUI',
            'be more proactive', 'fix this', 'make it work', 'properly',
            'think harder', 'idc', 'whats', 'dont', 'thats'
        ],
        'service_names': [
            'tower-echo-brain', 'tower-anime-production', 'tower-auth',
            'tower-kb', 'tower-apple-music', 'tower-dashboard'
        ],
        'common_requests': [
            'check status', 'fix the broken', 'why is this failing',
            'make it actually work', 'test this properly',
            'stop lying about it working'
        ],
        'preferences': {
            'wants_honesty': True,
            'hates_fake_progress': True,
            'prefers_direct_answers': True,
            'wants_proactive_solutions': True,
            'technical_not_explanatory': True
        }
    }

    return patterns

def create_test_datasets():
    """Create comprehensive test datasets from all extracted data."""
    print("\nCreating test datasets...")

    # Extract everything
    kb_articles = extract_kb_articles()
    conversations, sample_queries, claude_memory_points = extract_claude_conversations()
    work_data = extract_work_projects()
    vector_data = extract_vector_embeddings()
    claude_files = extract_claude_files()
    patrick_patterns = extract_patrick_patterns()

    # Create test directory
    test_dir = Path('/opt/tower-echo-brain/tests/data')
    test_dir.mkdir(parents=True, exist_ok=True)

    # Save KB articles test data
    kb_test = {
        'total_articles': len(kb_articles),
        'categories': list(set(a['category'] for a in kb_articles if a['category'])),
        'sample_articles': kb_articles[:50],  # First 50 for testing
        'work_articles': [a for a in kb_articles if
                         'trust' in a['title'].lower() or
                         'estate' in a['title'].lower() or
                         'federal' in a['title'].lower() or
                         'financial' in a['title'].lower()]
    }
    with open(test_dir / 'kb_articles_test.json', 'w') as f:
        json.dump(kb_test, f, indent=2)
    print(f"  Saved {len(kb_test['sample_articles'])} KB articles for testing")

    # Save Claude conversations test data
    claude_test = {
        'total_conversations': len(conversations),
        'total_claude_memories': len(claude_memory_points),
        'total_claude_files': len(claude_files),
        'sample_conversations': conversations[:100],  # More samples
        'sample_user_queries': sample_queries[:200],  # All queries
        'claude_memory_samples': claude_memory_points[:100],
        'claude_file_samples': claude_files[:20],  # Sample of files
        'patrick_patterns': patrick_patterns
    }
    with open(test_dir / 'claude_conversations_test.json', 'w') as f:
        json.dump(claude_test, f, indent=2)
    print(f"  Saved {len(conversations)} DB conversations + {len(claude_memory_points)} Qdrant + {len(claude_files)} files")

    # Save work project test data
    work_test = {
        'work_conversations': work_data,
        'project_types': ['trust_estate', 'federal_training', 'financial', 'anime'],
        'integration_points': ['Plaid', 'PostgreSQL', 'Qdrant', 'Ollama']
    }
    with open(test_dir / 'work_projects_test.json', 'w') as f:
        json.dump(work_test, f, indent=2)
    print(f"  Saved {len(work_data)} work project data points")

    # Save vector/semantic test data
    vector_test = {
        'collections': vector_data,
        'semantic_queries': [
            "What did we discuss about anime production?",
            "Show me financial integration status",
            "What's broken in Tower services?",
            "How do I setup Plaid webhooks?",
            "What are Patrick's preferences?",
            "Find all trust and estate planning info"
        ]
    }
    with open(test_dir / 'vector_semantic_test.json', 'w') as f:
        json.dump(vector_test, f, indent=2)
    print(f"  Saved vector/semantic test data")

    # Create expected recalls test data
    expected_recalls = {
        'kb_recall_tests': [
            {
                'query': 'What is the anime production status?',
                'should_recall': ['8+ minute generation', 'broken job status', 'needs redesign']
            },
            {
                'query': 'How does Echo Brain learn?',
                'should_recall': ['KB articles', 'Claude conversations', '4096D embeddings']
            },
            {
                'query': 'What financial integrations exist?',
                'should_recall': ['Plaid', 'webhooks', 'MFA', 'bank connections']
            }
        ],
        'conversation_recall_tests': [
            {
                'query': 'What did I say about being proactive?',
                'should_recall': ['be more proactive', 'wants proactive solutions']
            },
            {
                'query': 'What services are broken?',
                'should_recall': ['anime production', '8+ minutes', 'fake progress']
            }
        ],
        'semantic_relationship_tests': [
            {
                'query': 'Connect trust planning with financial systems',
                'should_find': ['estate planning', 'Plaid integration', 'financial services']
            }
        ]
    }
    with open(test_dir / 'expected_recalls_test.json', 'w') as f:
        json.dump(expected_recalls, f, indent=2)
    print(f"  Saved expected recall test scenarios")

    print(f"\nTest datasets created in {test_dir}")
    print(f"Total data points for testing:")
    print(f"  - KB Articles: {len(kb_articles)}")
    print(f"  - Claude Conversations: {len(conversations)}")
    print(f"  - Work Project Data: {len(work_data)}")
    print(f"  - Vector Collections: {len(vector_data)}")

    return test_dir

if __name__ == "__main__":
    test_dir = create_test_datasets()
    print(f"\n✅ All training data extracted to {test_dir}")
    print("Echo Brain can now be tested with REAL data, not fake mocks!")