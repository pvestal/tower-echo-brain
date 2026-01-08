#!/usr/bin/env python3
"""
Train Echo on Patrick's actual personal data
- Claude conversations (12,228 files)
- Google Photos metadata
- KB articles and preferences
- Personal workflows and patterns
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

# Database connection
import sys
sys.path.append('/opt/tower-echo-brain')
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

class PersonalDataTrainer:
    """Train Echo on Patrick's actual data, not mocks"""

    def __init__(self):
        self.db = psycopg2.connect(
            host="localhost",
            database="echo_brain",
            user="patrick",
            password="tower_echo_brain_secret_key_2025"
        )

    def train_on_claude_conversations(self):
        """Process Patrick's actual Claude conversations"""
        claude_dirs = [
            "/home/patrick/.claude/conversations",
            "/home/patrick/Documents/.claude/conversations",
            "/home/patrick/.claude/memory",
            "/home/patrick/.claude/context"
        ]

        conversation_count = 0
        patterns_learned = {}

        for claude_dir in claude_dirs:
            if os.path.exists(claude_dir):
                print(f"Processing {claude_dir}...")

                for root, dirs, files in os.walk(claude_dir):
                    for file in files:
                        if file.endswith(('.json', '.md', '.txt')):
                            file_path = os.path.join(root, file)
                            try:
                                self._process_conversation_file(file_path, patterns_learned)
                                conversation_count += 1

                                if conversation_count % 100 == 0:
                                    print(f"Processed {conversation_count} conversations...")

                            except Exception as e:
                                logger.warning(f"Error processing {file_path}: {e}")

        self._save_learned_patterns(patterns_learned)
        print(f"✅ Trained on {conversation_count} Claude conversations")
        return conversation_count

    def _process_conversation_file(self, file_path: str, patterns: Dict):
        """Extract Patrick's patterns from conversation file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    data = json.load(f)
                else:
                    data = {'content': f.read()}

            # Extract Patrick's ACTUAL business logic patterns
            content = str(data).lower()

            # Technical stack preferences
            if 'postgresql' in content:
                patterns.setdefault('tech_stack_database', []).append('PostgreSQL')
            if 'vue' in content or 'javascript' in content:
                patterns.setdefault('tech_stack_frontend', []).append('Vue.js')
            if 'python' in content:
                patterns.setdefault('tech_stack_backend', []).append('Python')
            if 'tower' in content:
                patterns.setdefault('infrastructure', []).append('Tower_ecosystem')

            # Quality standards
            if any(word in content for word in ['prove', 'test', 'verify']):
                patterns.setdefault('quality_standards', []).append('proof_required')
            if any(word in content for word in ['mock', 'fake', 'bullshit']):
                patterns.setdefault('quality_standards', []).append('no_fake_implementations')
            if any(word in content for word in ['working', 'actually works', 'tested']):
                patterns.setdefault('quality_standards', []).append('must_actually_work')

            # Communication patterns
            if any(word in content for word in ['lying', 'liar', 'lie']):
                patterns.setdefault('frustration_triggers', []).append('claude_lying')
            if any(word in content for word in ['fucking', 'asshole', 'bitch']):
                patterns.setdefault('frustration_indicators', []).append('patrick_angry')
            if any(word in content for word in ['fix', 'broken', 'not working']):
                patterns.setdefault('problem_solving', []).append('direct_action_required')

            # Project priorities
            if 'anime production' in content:
                patterns.setdefault('project_status', []).append('anime_production_broken')
            if 'echo brain' in content:
                patterns.setdefault('project_focus', []).append('echo_intelligence_development')
            if 'tower services' in content:
                patterns.setdefault('infrastructure_focus', []).append('tower_service_management')

            # Naming preferences
            if any(word in content for word in ['promotional', 'enhanced', 'bulletproof', 'ultimate']):
                patterns.setdefault('naming_standards', []).append('no_promotional_words')
            if 'professional naming' in content or 'descriptive' in content:
                patterns.setdefault('naming_standards', []).append('functional_descriptive_names')

            # Learning about Claude's mistakes
            if 'claude' in content and any(word in content for word in ['mistake', 'wrong', 'error']):
                patterns.setdefault('claude_patterns', []).append('claude_makes_errors')
            if 'session amnesia' in content or 'goldfish' in content:
                patterns.setdefault('claude_patterns', []).append('claude_forgets_context')

        except Exception as e:
            logger.warning(f"Error processing file content: {e}")

    def train_on_kb_articles(self):
        """Learn from Knowledge Base articles about Patrick's work"""
        try:
            cursor = self.db.cursor()
            cursor.execute("SELECT title, content, category, tags FROM articles")
            articles = cursor.fetchall()

            learned_facts = []

            for title, content, category, tags in articles:
                # Extract key facts about Patrick's systems
                facts = {
                    'source': 'kb_article',
                    'title': title,
                    'category': category,
                    'content_summary': content[:200] + "..." if len(content) > 200 else content,
                    'learned_at': datetime.now().isoformat()
                }

                learned_facts.append(facts)

            # Store in Echo's learning table
            for fact in learned_facts:
                cursor.execute("""
                    INSERT INTO learning_history (fact_type, content, source, created_at)
                    VALUES (%s, %s, %s, %s)
                """, ('kb_article', json.dumps(fact), 'knowledge_base', fact['learned_at']))

            self.db.commit()
            print(f"✅ Learned from {len(learned_facts)} KB articles")
            return len(learned_facts)

        except Exception as e:
            logger.error(f"KB training error: {e}")
            return 0

    def train_on_google_photos_metadata(self):
        """Train on actual Google Photos metadata (not mocks)"""
        google_dirs = [
            "/home/patrick/Pictures",
            "/mnt/1TB-storage/Google Photos",
            "/home/patrick/Photos"
        ]

        photo_count = 0
        metadata_learned = {}

        for photo_dir in google_dirs:
            if os.path.exists(photo_dir):
                print(f"Processing photos in {photo_dir}...")

                for root, dirs, files in os.walk(photo_dir):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4', '.mov')):
                            file_path = os.path.join(root, file)

                            # Extract metadata patterns
                            file_stat = os.stat(file_path)
                            metadata = {
                                'filename': file,
                                'size': file_stat.st_size,
                                'created': datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                                'path_pattern': self._extract_path_pattern(file_path)
                            }

                            # Learn organization patterns
                            path_parts = Path(file_path).parts
                            for part in path_parts:
                                if part not in ['home', 'patrick', 'Pictures']:
                                    metadata_learned.setdefault('organization_patterns', []).append(part)

                            photo_count += 1
                            if photo_count % 1000 == 0:
                                print(f"Processed {photo_count} photos...")

        # Store learned photo organization patterns
        if metadata_learned:
            cursor = self.db.cursor()
            cursor.execute("""
                INSERT INTO learning_history (fact_type, content, source, learned_at)
                VALUES (%s, %s, %s, %s)
            """, ('photo_organization', json.dumps(metadata_learned), 'google_photos_metadata', datetime.now().isoformat()))
            self.db.commit()

        print(f"✅ Analyzed {photo_count} photos for organization patterns")
        return photo_count

    def _extract_path_pattern(self, file_path: str) -> str:
        """Extract organization pattern from file path"""
        path = Path(file_path)
        return '/'.join(path.parts[-3:])  # Last 3 path components

    def _save_learned_patterns(self, patterns: Dict):
        """Save Patrick's patterns to Echo's learning database"""
        try:
            cursor = self.db.cursor()

            for pattern_type, values in patterns.items():
                # Count occurrences
                value_counts = {}
                for value in values:
                    value_counts[value] = value_counts.get(value, 0) + 1

                # Store top patterns
                for value, count in sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                    cursor.execute("""
                        INSERT INTO learning_history (fact_type, content, source, learned_at)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """, (
                        f'patrick_{pattern_type}',
                        json.dumps({'pattern': value, 'frequency': count}),
                        'claude_conversations',
                        datetime.now().isoformat()
                    ))

            self.db.commit()
            print(f"✅ Saved {len(patterns)} pattern types to Echo's learning database")

        except Exception as e:
            logger.error(f"Error saving patterns: {e}")

    def run_full_training(self):
        """Train Echo on ALL of Patrick's actual data"""
        print("🧠 Training Echo on Patrick's actual personal data...")
        print("=" * 60)

        # Train on conversations
        conv_count = self.train_on_claude_conversations()

        # Train on KB articles
        kb_count = self.train_on_kb_articles()

        # Train on photo metadata
        photo_count = self.train_on_google_photos_metadata()

        # Summary
        print("\n" + "=" * 60)
        print("🎯 PERSONAL DATA TRAINING COMPLETE")
        print(f"📝 Claude conversations: {conv_count}")
        print(f"📚 KB articles: {kb_count}")
        print(f"📸 Photos analyzed: {photo_count}")
        print("\nEcho now knows Patrick's actual preferences, workflows, and patterns!")

        return {
            'conversations': conv_count,
            'kb_articles': kb_count,
            'photos': photo_count,
            'status': 'completed'
        }


if __name__ == "__main__":
    trainer = PersonalDataTrainer()
    results = trainer.run_full_training()
    print(f"Training results: {results}")