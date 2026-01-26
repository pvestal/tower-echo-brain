#!/usr/bin/env python3
"""
LoRA Training Status Manager
Shows status of all poses, styles, actions, and scenes
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from tabulate import tabulate
from datetime import datetime
from pathlib import Path
import sys
import argparse

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'anime_production',
    'user': 'patrick',
    'password': 'RP78eIrW7cI2jYvL5akt1yurE',
    'port': 5432
}

class LoRAStatusManager:
    def __init__(self):
        self.conn = None

    def connect(self):
        """Connect to database"""
        self.conn = psycopg2.connect(**DB_CONFIG)

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def get_all_status(self):
        """Get status of all LoRAs"""
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)

        query = """
        SELECT * FROM lora_training_overview
        ORDER BY category, lora_name;
        """

        cursor.execute(query)
        return cursor.fetchall()

    def get_category_summary(self):
        """Get summary by category"""
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)

        query = """
        SELECT
            c.name as category,
            COUNT(d.id) as total,
            COUNT(CASE WHEN s.status = 'completed' THEN 1 END) as trained,
            COUNT(CASE WHEN s.status = 'training' THEN 1 END) as in_progress,
            COUNT(CASE WHEN s.status = 'queued' THEN 1 END) as queued,
            COUNT(CASE WHEN s.status IS NULL OR s.status = 'not_started' THEN 1 END) as not_started
        FROM lora_categories c
        LEFT JOIN lora_definitions d ON c.id = d.category_id
        LEFT JOIN lora_training_status s ON d.id = s.definition_id
        GROUP BY c.name, c.priority
        ORDER BY c.priority DESC;
        """

        cursor.execute(query)
        return cursor.fetchall()

    def get_training_stats(self):
        """Get overall training statistics"""
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)

        query = "SELECT * FROM get_training_stats();"
        cursor.execute(query)
        return cursor.fetchone()

    def display_status(self, filter_category=None, show_nsfw=True):
        """Display comprehensive status"""
        print("\n" + "="*100)
        print("LORA TRAINING STATUS OVERVIEW")
        print("="*100)

        # Overall stats
        stats = self.get_training_stats()
        print(f"\n📊 Overall Statistics:")
        print(f"  Total LoRAs: {stats['total_loras']}")
        print(f"  ✅ Trained: {stats['trained']}/{stats['total_loras']} ({stats['trained']*100//stats['total_loras']}%)")
        print(f"  🔄 In Progress: {stats['in_progress']}")
        print(f"  ⏳ Queued: {stats['queued']}")
        print(f"  ❌ Not Started: {stats['not_started']}")
        print(f"  ⚠️ Failed: {stats['failed']}")
        print(f"  🔞 NSFW: {stats['nsfw_trained']}/{stats['nsfw_total']} trained")

        # Category summary
        print(f"\n📁 Category Summary:")
        categories = self.get_category_summary()

        cat_table = []
        for cat in categories:
            progress = f"{cat['trained']}/{cat['total']}"
            percentage = (cat['trained'] * 100 // cat['total']) if cat['total'] > 0 else 0
            status_emoji = "✅" if percentage == 100 else "🔄" if cat['in_progress'] > 0 else "❌"

            cat_table.append([
                f"{status_emoji} {cat['category'].upper()}",
                cat['total'],
                progress,
                f"{percentage}%",
                cat['in_progress'],
                cat['queued'],
                cat['not_started']
            ])

        print(tabulate(cat_table,
                      headers=['Category', 'Total', 'Trained', '%', 'Training', 'Queued', 'Todo'],
                      tablefmt='grid'))

        # Detailed status
        print(f"\n📋 Detailed LoRA Status:")
        all_loras = self.get_all_status()

        if filter_category:
            all_loras = [l for l in all_loras if l['category'] == filter_category]

        if not show_nsfw:
            all_loras = [l for l in all_loras if not l['is_nsfw']]

        # Group by category
        current_category = None
        for lora in all_loras:
            if lora['category'] != current_category:
                current_category = lora['category']
                print(f"\n🏷️ {current_category.upper()}")
                print("-" * 80)

            # Status icon
            status_icon = {
                'Ready': '✅',
                'In Progress': '🔄',
                'Queued': '⏳',
                'Failed': '❌',
                'Not Started': '⭕'
            }.get(lora['status_display'], '❓')

            # NSFW indicator
            nsfw = "🔞" if lora['is_nsfw'] else "  "

            # Format output
            name = f"{lora['display_name']:30}"
            trigger = f"[{lora['trigger_word']}]" if lora['trigger_word'] else ""
            status = f"{status_icon} {lora['status_display']:12}"

            print(f"  {nsfw} {name} {trigger:30} {status}")

            if lora['model_path']:
                print(f"      └─ Model: {Path(lora['model_path']).name}")

    def queue_training(self, lora_name, priority=5):
        """Add LoRA to training queue"""
        cursor = self.conn.cursor()

        # Get definition ID
        cursor.execute("""
            SELECT id FROM lora_definitions WHERE name = %s
        """, (lora_name,))

        result = cursor.fetchone()
        if not result:
            print(f"❌ LoRA '{lora_name}' not found in definitions")
            return False

        definition_id = result[0]

        # Add to queue
        cursor.execute("""
            INSERT INTO training_queue (definition_id, priority, status)
            VALUES (%s, %s, 'pending')
            ON CONFLICT DO NOTHING
        """, (definition_id, priority))

        self.conn.commit()
        print(f"✅ Added '{lora_name}' to training queue with priority {priority}")
        return True

    def mark_completed(self, lora_name, model_path):
        """Mark LoRA as completed"""
        cursor = self.conn.cursor()

        cursor.execute("""
            UPDATE lora_training_status
            SET status = 'completed',
                model_path = %s,
                training_completed_at = NOW(),
                updated_at = NOW()
            WHERE definition_id = (SELECT id FROM lora_definitions WHERE name = %s)
        """, (model_path, lora_name))

        self.conn.commit()
        print(f"✅ Marked '{lora_name}' as completed")

def main():
    parser = argparse.ArgumentParser(description='LoRA Training Status Manager')
    parser.add_argument('command', choices=['status', 'queue', 'complete'],
                       help='Command to execute')
    parser.add_argument('--category', help='Filter by category')
    parser.add_argument('--no-nsfw', action='store_true', help='Hide NSFW content')
    parser.add_argument('--lora', help='LoRA name for queue/complete commands')
    parser.add_argument('--priority', type=int, default=5, help='Priority for queue command')
    parser.add_argument('--model-path', help='Model path for complete command')

    args = parser.parse_args()

    manager = LoRAStatusManager()
    manager.connect()

    try:
        if args.command == 'status':
            manager.display_status(
                filter_category=args.category,
                show_nsfw=not args.no_nsfw
            )

        elif args.command == 'queue':
            if not args.lora:
                print("❌ --lora required for queue command")
                sys.exit(1)
            manager.queue_training(args.lora, args.priority)

        elif args.command == 'complete':
            if not args.lora or not args.model_path:
                print("❌ --lora and --model-path required for complete command")
                sys.exit(1)
            manager.mark_completed(args.lora, args.model_path)

    finally:
        manager.close()

if __name__ == "__main__":
    main()