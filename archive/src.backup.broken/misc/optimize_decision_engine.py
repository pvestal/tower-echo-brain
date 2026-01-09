#!/usr/bin/env python3
"""
Optimize Model Decision Engine
Apply optimizations based on test results to improve decision accuracy
"""

import os
import asyncio
import sys
import psycopg2
import json

sys.path.append('/opt/tower-echo-brain')

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "database": "tower_consolidated",
    "user": os.getenv("TOWER_USER", os.getenv("TOWER_USER", "patrick"))
}

def apply_optimizations():
    """Apply optimized thresholds and weights based on test analysis"""

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        print("Applying optimizations to model decision engine...")

        # 1. Adjust complexity thresholds to be more aggressive
        print("\n1. Updating complexity thresholds...")

        # Current issues:
        # - Basic coding (1.0) gets tiny instead of small (need to lower small threshold)
        # - Debugging (5.0) gets small instead of medium (need to lower medium threshold)
        # - Complex design (24.8) gets medium instead of large (need to lower large threshold)
        # - Massive refactoring (38.8) gets large instead of cloud (need to lower cloud threshold)

        optimized_thresholds = [
            ('tiny', 0, 2),      # Reduced from 5 to 2
            ('small', 2, 8),     # Reduced from 5-15 to 2-8
            ('medium', 8, 20),   # Reduced from 15-30 to 8-20
            ('large', 20, 35),   # Reduced from 30-50 to 20-35
            ('cloud', 35, 999)   # Reduced from 50+ to 35+
        ]

        for tier, min_score, max_score in optimized_thresholds:
            cursor.execute("""
                UPDATE complexity_thresholds
                SET min_score = %s, max_score = %s, last_adjusted = CURRENT_TIMESTAMP
                WHERE tier = %s
            """, (min_score, max_score, tier))
            print(f"  {tier}: {min_score} - {max_score}")

        # 2. Adjust feature weights to be more sensitive
        print("\n2. Updating feature weights...")

        # Current weights are not aggressive enough for coding and technical tasks
        optimized_weights = {
            "code_complexity": 3.0,     # Increased from 1.0 - coding tasks need higher impact
            "context_lines": 0.8,       # Increased from 0.5 - context matters more
            "technical_depth": 3.5,     # Increased from 2.0 - technical terms need more weight
            "multi_file": 4.0,          # Increased from 3.0 - multi-file ops are complex
            "architecture": 5.0,        # Increased from 4.0 - architecture decisions are critical
            "debugging": 3.0,           # Doubled from 1.5 - debugging needs capable models
            "optimization": 3.5,        # Increased from 2.5 - optimization is complex
            "security": 3.0,            # Increased from 2.0 - security needs expertise
            "api_design": 3.5,          # Increased from 2.5 - API design is important
            "refactoring": 4.0          # Increased from 3.0 - refactoring is very complex
        }

        for feature, weight in optimized_weights.items():
            cursor.execute("""
                INSERT INTO model_decision_weights (feature, weight, update_count, active)
                VALUES (%s, %s, 1, true)
                ON CONFLICT (feature) DO UPDATE SET
                    weight = EXCLUDED.weight,
                    update_count = model_decision_weights.update_count + 1
            """, (feature, weight))
            print(f"  {feature}: {weight}")

        # 3. Add new feature detection patterns for better analysis
        print("\n3. Creating backup of current settings...")

        # Create backup table for rollback if needed
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS decision_engine_backup AS
            SELECT 'threshold' as type, tier as name, min_score::text, max_score::text,
                   CURRENT_TIMESTAMP as backup_date
            FROM complexity_thresholds
            UNION ALL
            SELECT 'weight' as type, feature as name, weight::text, '' as max_score,
                   CURRENT_TIMESTAMP as backup_date
            FROM model_decision_weights
            WHERE active = true
        """)

        conn.commit()
        print("✅ Optimizations applied successfully!")

        # 4. Show summary of changes
        print("\n📊 OPTIMIZATION SUMMARY:")
        print("=" * 50)
        print("Threshold Changes (more aggressive tiers):")
        print("- Tiny: 0-5 → 0-2")
        print("- Small: 5-15 → 2-8")
        print("- Medium: 15-30 → 8-20")
        print("- Large: 30-50 → 20-35")
        print("- Cloud: 50+ → 35+")
        print()
        print("Weight Increases (more sensitive detection):")
        for feature, weight in optimized_weights.items():
            print(f"- {feature}: increased to {weight}")

        print("\n🔄 Run tests again to validate improvements...")

        conn.close()
        return True

    except Exception as e:
        print(f"❌ Failed to apply optimizations: {e}")
        return False

def create_rollback_script():
    """Create a rollback script in case optimizations need to be reverted"""

    rollback_script = """#!/usr/bin/env python3
'''
Rollback script for model decision engine optimizations
'''

import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "database": "tower_consolidated",
    "user": os.getenv("TOWER_USER", os.getenv("TOWER_USER", "patrick"))
}

def rollback():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Restore original thresholds
    original_thresholds = [
        ('tiny', 0, 5),
        ('small', 5, 15),
        ('medium', 15, 30),
        ('large', 30, 50),
        ('cloud', 50, 999)
    ]

    for tier, min_score, max_score in original_thresholds:
        cursor.execute('''
            UPDATE complexity_thresholds
            SET min_score = %s, max_score = %s
            WHERE tier = %s
        ''', (min_score, max_score, tier))

    # Restore original weights
    original_weights = {
        "code_complexity": 1.0,
        "context_lines": 0.5,
        "technical_depth": 2.0,
        "multi_file": 3.0,
        "architecture": 4.0,
        "debugging": 1.5,
        "optimization": 2.5,
        "security": 2.0,
        "api_design": 2.5,
        "refactoring": 3.0
    }

    for feature, weight in original_weights.items():
        cursor.execute('''
            UPDATE model_decision_weights
            SET weight = %s
            WHERE feature = %s
        ''', (weight, feature))

    conn.commit()
    conn.close()
    print("✅ Rollback completed")

if __name__ == "__main__":
    rollback()
"""

    with open('/opt/tower-echo-brain/rollback_optimizations.py', 'w') as f:
        f.write(rollback_script)

    print("💾 Rollback script created: rollback_optimizations.py")

async def main():
    """Main optimization runner"""

    print("🔧 Model Decision Engine Optimization")
    print("=" * 50)

    # Create rollback script first
    create_rollback_script()

    # Apply optimizations
    success = apply_optimizations()

    if success:
        print("\n✅ Optimization complete! Test with:")
        print("   python3 direct_decision_test.py")
        print("\n🔄 To rollback if needed:")
        print("   python3 rollback_optimizations.py")
    else:
        print("\n❌ Optimization failed!")
        return 1

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))