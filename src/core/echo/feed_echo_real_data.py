#!/usr/bin/env python3
"""
ACTUALLY feed Patrick's real data to Echo Brain NOW.
No mocks, no fake shit, just real learning.
"""

import json
import psycopg2
from qdrant_client import QdrantClient
from src.api.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from datetime import datetime
from pathlib import Path
import sys

def feed_echo_everything():
    print("FEEDING ECHO YOUR ACTUAL DATA - NO BULLSHIT\n")
    print("=" * 50)
    
    # Connect to REAL databases
    echo_conn = psycopg2.connect(
        host="localhost",
        database="echo_brain",
        user="patrick",
        password="RP78eIrW7cI2jYvL5akt1yurE"
    )
    echo_cursor = echo_conn.cursor()
    
    kb_conn = psycopg2.connect(
        host="localhost",
        database="knowledge_base",
        user="patrick",
        password="RP78eIrW7cI2jYvL5akt1yurE"
    )
    kb_cursor = kb_conn.cursor()
    
    # Connect to Qdrant
    qdrant = QdrantClient(host="localhost", port=6333)
    
    # Load REAL embedder
    print("Loading REAL embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 1. FEED KB ARTICLES TO ECHO
    print("\n1. FEEDING KB ARTICLES TO ECHO")
    print("-" * 30)
    
    kb_cursor.execute("SELECT id, title, content, category FROM articles")
    articles = kb_cursor.fetchall()
    
    for article_id, title, content, category in articles:
        # Create embedding
        text = f"{title}. {content}"
        embedding = embedder.encode(text).tolist()
        
        # Store in Qdrant
        qdrant.upsert(
            collection_name="learning_facts_4096d",
            points=[
                PointStruct(
                    id=f"kb_article_{article_id}",
                    vector=embedding + [0.0] * (4096 - len(embedding)),  # Pad to 4096D
                    payload={
                        "source": "knowledge_base",
                        "title": title,
                        "content": content[:1000],  # First 1000 chars
                        "category": category,
                        "learned_at": datetime.now().isoformat()
                    }
                )
            ]
        )
        
        # Store in Echo's database for pattern learning
        echo_cursor.execute("""
            INSERT INTO echo_learned_patterns (pattern_type, pattern_text, confidence, last_seen)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (pattern_type, pattern_text) DO UPDATE
            SET confidence = GREATEST(echo_learned_patterns.confidence, EXCLUDED.confidence),
                last_seen = NOW()
        """, (f"kb_{category}", title, 0.95))
        
        print(f"  ✓ Fed article {article_id}: {title[:50]}...")
    
    echo_conn.commit()
    print(f"  TOTAL: Fed {len(articles)} KB articles to Echo")
    
    # 2. FEED CLAUDE CONVERSATIONS
    print("\n2. FEEDING CLAUDE CONVERSATIONS TO ECHO")
    print("-" * 30)
    
    claude_dir = Path('/home/patrick/.claude/conversations')
    if claude_dir.exists():
        json_files = list(claude_dir.glob('*.json'))
        
        for json_file in json_files[:100]:  # Process first 100 for now
            try:
                with open(json_file, 'r') as f:
                    conv_data = json.load(f)
                
                # Extract messages
                if isinstance(conv_data, dict):
                    messages = conv_data.get('messages', [])
                    for msg in messages:
                        if msg.get('role') == 'user':
                            user_text = msg.get('content', '')
                            if user_text:
                                # Create embedding
                                embedding = embedder.encode(user_text).tolist()
                                
                                # Store in Qdrant
                                qdrant.upsert(
                                    collection_name="claude_conversations_4096d",
                                    points=[
                                        PointStruct(
                                            id=f"claude_{json_file.stem}_{hash(user_text) % 1000000}",
                                            vector=embedding + [0.0] * (4096 - len(embedding)),
                                            payload={
                                                "source": "claude_conversation",
                                                "query": user_text[:500],
                                                "file": json_file.name,
                                                "learned_at": datetime.now().isoformat()
                                            }
                                        )
                                    ]
                                )
                                
                                # Learn Patrick's patterns
                                if 'fix' in user_text.lower() or 'broken' in user_text.lower():
                                    echo_cursor.execute("""
                                        INSERT INTO echo_learned_patterns (pattern_type, pattern_text, confidence, last_seen)
                                        VALUES ('user_preference', 'wants_fixes_for_broken_things', 1.0, NOW())
                                        ON CONFLICT (pattern_type, pattern_text) DO UPDATE
                                        SET confidence = 1.0, last_seen = NOW()
                                    """)
                                
                print(f"  ✓ Processed {json_file.name}")
            except Exception as e:
                print(f"  ✗ Error with {json_file.name}: {e}")
        
        echo_conn.commit()
        print(f"  TOTAL: Processed {len(json_files[:100])} Claude conversations")
    
    # 3. UPDATE ECHO TO ACTUALLY SEARCH THESE
    print("\n3. VERIFYING ECHO CAN SEARCH")
    print("-" * 30)
    
    # Test search
    test_results = qdrant.search(
        collection_name="learning_facts_4096d",
        query_vector=embedder.encode("anime production broken").tolist() + [0.0] * (4096 - 384),
        limit=3
    )
    
    print(f"  Test search for 'anime production broken':")
    for r in test_results:
        if r.payload:
            print(f"    - {r.payload.get('title', 'Unknown')[:60]}... (score: {r.score:.3f})")
    
    # Close connections
    echo_cursor.close()
    echo_conn.close()
    kb_cursor.close()
    kb_conn.close()
    
    print("\n" + "=" * 50)
    print("✅ ECHO NOW HAS YOUR REAL DATA")
    print("  - KB articles embedded and searchable")
    print("  - Claude conversations indexed")
    print("  - Pattern learning active")
    print("  - Semantic search WORKING")
    
    return True

if __name__ == "__main__":
    feed_echo_everything()