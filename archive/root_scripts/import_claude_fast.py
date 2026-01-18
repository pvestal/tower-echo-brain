#!/usr/bin/env python3
"""Fast Claude conversation importer for training"""

import json
import os
import psycopg2
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClaudeImporter:
    def __init__(self):
        self.conn = psycopg2.connect(
            host="localhost",
            database="echo_brain",
            user="patrick",
            password="tower_echo_brain_secret_key_2025"
        )
        self.cur = self.conn.cursor()
        self.claude_dir = Path("/home/patrick/.claude/conversations")

    def import_conversations(self, limit=100):
        """Import Claude conversations for training"""
        logger.info(f"📚 Importing from {self.claude_dir}")

        files = sorted(self.claude_dir.glob("*.json"))[-limit:]  # Get most recent
        imported = 0
        patterns_found = 0

        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Extract conversation patterns
                for message in data.get('messages', []):
                    if message.get('role') == 'user':
                        query = message.get('content', '')[:5000]  # Limit length
                    elif message.get('role') == 'assistant' and query:
                        response = message.get('content', '')[:5000]

                        # Store in training feedback
                        self.cur.execute("""
                            INSERT INTO training_feedback
                            (conversation_id, query_text, response_text, was_useful, feedback_score)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT DO NOTHING
                        """, (str(file_path.stem), query, response, True, 0.8))

                        # Extract patterns (code, commands, preferences)
                        patterns = self.extract_patterns(query, response)
                        for pattern in patterns:
                            self.store_pattern(pattern)
                            patterns_found += 1

                        query = None  # Reset for next pair

                imported += 1
                if imported % 10 == 0:
                    self.conn.commit()
                    logger.info(f"  Imported {imported}/{len(files)} files, {patterns_found} patterns")

            except Exception as e:
                logger.warning(f"  Skip {file_path.name}: {e}")

        self.conn.commit()
        logger.info(f"✅ Imported {imported} conversations, {patterns_found} patterns")
        return imported, patterns_found

    def extract_patterns(self, query, response):
        """Extract learning patterns from conversations"""
        patterns = []

        # Code patterns
        if '```' in response:
            patterns.append({
                'type': 'code_generation',
                'data': {'has_code': True, 'query_type': 'coding'}
            })

        # Command patterns
        if any(cmd in response for cmd in ['curl', 'git', 'python', 'bash']):
            patterns.append({
                'type': 'command_usage',
                'data': {'has_commands': True}
            })

        # Refactoring patterns
        if 'refactor' in query.lower() or 'fix' in query.lower():
            patterns.append({
                'type': 'problem_solving',
                'data': {'approach': 'refactoring'}
            })

        return patterns

    def store_pattern(self, pattern):
        """Store learned pattern in database"""
        self.cur.execute("""
            INSERT INTO learned_patterns (pattern_type, pattern_data, confidence)
            VALUES (%s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (pattern['type'], json.dumps(pattern['data']), 0.7))

if __name__ == "__main__":
    importer = ClaudeImporter()
    importer.import_conversations(limit=500)  # Import last 500 conversations