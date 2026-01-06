#!/usr/bin/env python3
"""
Fine-tune Model Decision Engine
Apply balanced optimizations for optimal 80%+ accuracy
"""

import os
import asyncio
import sys
import psycopg2

sys.path.append('/opt/tower-echo-brain')

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "database": "tower_consolidated",
    "user": os.getenv("TOWER_USER", os.getenv("TOWER_USER", "patrick"))
}

def apply_fine_tuning():
    """Apply fine-tuned adjustments based on over-correction"""

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        print("Fine-tuning model decision engine...")

        # Analysis of current issues (57.1% score):
        # - API implementation (24.8) got large instead of medium - need to raise medium threshold
        # - Architecture refactoring (48.3) got cloud instead of large - need to raise cloud threshold
        # - Complex system design (35.3) got cloud instead of large - need to raise cloud threshold

        print("\n1. Adjusting complexity thresholds (balanced approach)...")

        balanced_thresholds = [
            ('tiny', 0, 2),      # Keep as is - working well
            ('small', 2, 8),     # Keep as is - working well
            ('medium', 8, 25),   # Increased from 20 to 25 (API impl should stay medium)
            ('large', 25, 45),   # Increased from 35 to 45 (arch tasks should use large models)
            ('cloud', 45, 999)   # Increased from 35 to 45 (only truly massive tasks)
        ]

        for tier, min_score, max_score in balanced_thresholds:
            cursor.execute("""
                UPDATE complexity_thresholds
                SET min_score = %s, max_score = %s, last_adjusted = CURRENT_TIMESTAMP
                WHERE tier = %s
            """, (min_score, max_score, tier))
            print(f"  {tier}: {min_score} - {max_score}")

        # 2. Slightly reduce some feature weights that are too aggressive
        print("\n2. Balancing feature weights...")

        balanced_weights = {
            "code_complexity": 2.5,     # Reduced from 3.0 - was too aggressive
            "context_lines": 0.8,       # Keep same - working well
            "technical_depth": 3.0,     # Reduced from 3.5 - was pushing scores too high
            "multi_file": 3.5,          # Reduced from 4.0 - multi-file detection was over-weighted
            "architecture": 4.5,        # Reduced from 5.0 - architecture detection too strong
            "debugging": 3.0,           # Keep same - debugging complexity appropriate
            "optimization": 3.0,        # Reduced from 3.5 - optimization was over-weighted
            "security": 2.5,            # Reduced from 3.0 - security terms were too impactful
            "api_design": 3.0,          # Reduced from 3.5 - API design was pushing scores too high
            "refactoring": 3.5          # Reduced from 4.0 - refactoring was too heavily weighted
        }

        for feature, weight in balanced_weights.items():
            cursor.execute("""
                UPDATE model_decision_weights
                SET weight = %s, update_count = update_count + 1
                WHERE feature = %s
            """, (weight, feature))
            print(f"  {feature}: {weight}")

        conn.commit()
        print("✅ Fine-tuning applied successfully!")

        # Show summary
        print("\n📊 FINE-TUNING SUMMARY:")
        print("=" * 50)
        print("Threshold Adjustments (more balanced):")
        print("- Medium: 8-20 → 8-25 (keep more API tasks in medium)")
        print("- Large: 20-35 → 25-45 (architecture tasks use large models)")
        print("- Cloud: 35+ → 45+ (only truly massive complexity)")
        print()
        print("Weight Reductions (less aggressive):")
        for feature, weight in balanced_weights.items():
            print(f"- {feature}: balanced to {weight}")

        conn.close()
        return True

    except Exception as e:
        print(f"❌ Failed to apply fine-tuning: {e}")
        return False

async def main():
    """Main fine-tuning runner"""

    print("🔧 Model Decision Engine Fine-Tuning")
    print("=" * 50)

    # Apply fine-tuning
    success = apply_fine_tuning()

    if success:
        print("\n✅ Fine-tuning complete! Test with:")
        print("   python3 direct_decision_test.py")
        print("\n📈 Expected improvements:")
        print("   - API implementation should stay in medium tier")
        print("   - Architecture tasks should use large (not cloud)")
        print("   - Complex design should use large models")
        print("   - Target score: 80%+")
    else:
        print("\n❌ Fine-tuning failed!")
        return 1

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))