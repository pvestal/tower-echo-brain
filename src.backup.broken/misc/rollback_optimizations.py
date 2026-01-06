#!/usr/bin/env python3
'''
Rollback script for model decision engine optimizations
'''

import os
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
