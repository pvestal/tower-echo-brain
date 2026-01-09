#!/usr/bin/env python3
"""
Train Echo Brain with Patrick's ACTUAL business logic patterns
Uses the extracted patterns from claude_experts analysis
"""

import json
import psycopg2
from datetime import datetime

def load_business_logic_patterns():
    """Load the extracted business logic patterns"""
    with open('/tmp/claude/patrick_business_logic_patterns.json', 'r') as f:
        return json.load(f)

def train_echo_with_patterns():
    """Train Echo Brain with Patrick's actual business logic"""

    # Connect to Echo's database
    db = psycopg2.connect(
        host="localhost",
        database="echo_brain",
        user="patrick",
        password="***REMOVED***"
    )
    cursor = db.cursor()

    patterns = load_business_logic_patterns()
    trained_count = 0

    print("🧠 Training Echo Brain with Patrick's actual business logic patterns...")

    # Train each learning pattern
    for pattern in patterns['learning_patterns']:
        try:
            cursor.execute("""
                INSERT INTO learning_history (fact_type, learned_fact, confidence, metadata, created_at)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                pattern['category'],
                f"{pattern['content']} - {pattern['business_logic']}",
                pattern['confidence'],
                json.dumps({
                    'pattern_type': pattern['pattern_type'],
                    'evidence_count': pattern['evidence_count'],
                    'examples': pattern['source_examples'],
                    'source': 'patrick_conversation_analysis'
                }),
                datetime.now()
            ))

            trained_count += 1
            print(f"✅ Trained: {pattern['category']} - {pattern['pattern_type']}")

        except Exception as e:
            print(f"❌ Error training pattern {pattern['category']}: {e}")

    # Train meta patterns
    try:
        cursor.execute("""
            INSERT INTO learning_history (fact_type, learned_fact, confidence, metadata, created_at)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            'patrick_meta_patterns',
            f"Patrick's core preferences: {patterns['meta_patterns']['core_values']}",
            0.95,
            json.dumps(patterns['meta_patterns']),
            datetime.now()
        ))
        trained_count += 1
        print("✅ Trained: Meta patterns")
    except Exception as e:
        print(f"❌ Error training meta patterns: {e}")

    # Train recommendations with weights
    for rec in patterns['echo_brain_training_recommendations']:
        try:
            cursor.execute("""
                INSERT INTO learning_history (fact_type, learned_fact, confidence, metadata, created_at)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                'patrick_training_weights',
                rec['pattern'],
                rec['training_weight'] / 2.5,  # Convert weight to confidence
                json.dumps(rec),
                datetime.now()
            ))
            trained_count += 1
            print(f"✅ Trained: Weight {rec['training_weight']} - {rec['pattern'][:50]}...")
        except Exception as e:
            print(f"❌ Error training recommendation: {e}")

    db.commit()
    db.close()

    print(f"\n🎯 Training complete! {trained_count} patterns learned from Patrick's conversations")
    return trained_count

def verify_training():
    """Verify Echo learned the patterns"""
    db = psycopg2.connect(
        host="localhost",
        database="echo_brain",
        user="patrick",
        password="***REMOVED***"
    )
    cursor = db.cursor()

    # Check learned patterns
    cursor.execute("""
        SELECT fact_type, COUNT(*) as count
        FROM learning_history
        WHERE metadata->>'source' = 'patrick_conversation_analysis'
        AND created_at > NOW() - INTERVAL '1 hour'
        GROUP BY fact_type
        ORDER BY count DESC
    """)

    results = cursor.fetchall()
    print("\n📊 Verification - Echo learned these pattern types:")
    for fact_type, count in results:
        print(f"  {fact_type}: {count} patterns")

    db.close()
    return len(results) > 0

if __name__ == "__main__":
    trained_count = train_echo_with_patterns()
    success = verify_training()

    if success and trained_count > 0:
        print(f"\n✅ SUCCESS: Echo Brain trained with {trained_count} business logic patterns")
        print("🧠 Echo can now understand Patrick's actual preferences and standards")
    else:
        print("\n❌ FAILED: Training did not complete successfully")