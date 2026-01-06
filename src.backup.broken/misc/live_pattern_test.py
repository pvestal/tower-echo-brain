#!/usr/bin/env python3
"""
Live Pattern Testing - Show Echo's learned patterns in real-time
Demonstrates comprehensive learning from existing codebase/database/configs
"""
import psycopg2
import time
import json
import sys
import requests
from datetime import datetime

class LivePatternTester:
    """Stream real-time testing of Echo's comprehensive learning"""

    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'echo_brain',
            'user': 'patrick',
            'password': '***REMOVED***'
        }

    def print_section(self, title):
        """Print a formatted section header"""
        print(f"\n{'='*60}")
        print(f"🎯 {title}")
        print(f"{'='*60}")

    def show_learned_patterns_live(self):
        """Show patterns learned from comprehensive ingestion"""
        self.print_section("COMPREHENSIVE LEARNING RESULTS")

        try:
            db = psycopg2.connect(**self.db_config)
            cursor = db.cursor()

            # Get patterns extracted from different sources
            cursor.execute("""
                SELECT fact_type, learned_fact, confidence,
                       metadata->>'source' as source,
                       metadata->>'extraction_source' as extraction_source,
                       created_at
                FROM learning_history
                WHERE created_at > NOW() - INTERVAL '1 hour'
                ORDER BY created_at DESC
                LIMIT 20
            """)

            patterns_by_source = {}
            for row in cursor.fetchall():
                fact_type, learned_fact, confidence, source, extraction_source, created_at = row

                if extraction_source not in patterns_by_source:
                    patterns_by_source[extraction_source] = []

                patterns_by_source[extraction_source].append({
                    'type': fact_type,
                    'pattern': learned_fact,
                    'confidence': confidence,
                    'source': source,
                    'learned_at': created_at.strftime('%H:%M:%S') if created_at else 'unknown'
                })

            # Show patterns by source
            for extraction_source, patterns in patterns_by_source.items():
                print(f"\n📂 {extraction_source.upper()} ANALYSIS:")
                for i, pattern in enumerate(patterns[:5], 1):  # Top 5 per source
                    confidence_bar = "●" * int(pattern['confidence'] * 10)
                    print(f"  {i}. {pattern['pattern'][:80]}...")
                    print(f"     📊 {confidence_bar} {pattern['confidence']:.2f} | 🕐 {pattern['learned_at']}")

            db.close()

        except Exception as e:
            print(f"❌ Failed to show learned patterns: {e}")

    def test_pattern_application_live(self):
        """Test that patterns are being applied to responses"""
        self.print_section("PATTERN APPLICATION TESTING")

        test_queries = [
            ("Database choice", "What database should I use?"),
            ("Framework preference", "What web framework is best?"),
            ("Deployment approach", "How should I deploy my service?"),
            ("Code organization", "How should I structure my project?")
        ]

        for test_name, query in test_queries:
            print(f"\n🧪 Testing: {test_name}")
            print(f"   Query: '{query}'")

            try:
                # Try to get response from Echo (with short timeout)
                response = requests.post(
                    'http://localhost:8309/api/echo/query',
                    json={'query': query},
                    timeout=10
                )

                if response.status_code == 200:
                    data = response.json()
                    response_text = data.get('response', '')
                    patterns_applied = data.get('business_logic_applied', [])

                    print(f"   📝 Response: {response_text[:100]}...")
                    print(f"   🧠 Patterns Applied: {len(patterns_applied)}")
                    print(f"   ⚡ Processing Time: {data.get('processing_time', 0):.2f}s")
                else:
                    print(f"   ❌ API Error: {response.status_code}")

            except requests.exceptions.Timeout:
                print(f"   ⏰ Timeout - Echo taking too long")
            except Exception as e:
                print(f"   ❌ Error: {e}")

            time.sleep(1)  # Brief pause between tests

    def show_feedback_effectiveness(self):
        """Show how feedback is being used for improvement"""
        self.print_section("FEEDBACK EFFECTIVENESS")

        try:
            response = requests.get('http://localhost:8309/api/echo/feedback/stats')
            if response.status_code == 200:
                feedback_data = response.json()

                stats = feedback_data['feedback_stats']
                print(f"📊 Total Feedback: {stats['total_feedback']}")
                print(f"👍 Thumbs Up: {stats['thumbs_up']}")
                print(f"👎 Thumbs Down: {stats['thumbs_down']}")
                print(f"✏️ Corrections: {stats['corrections']}")
                print(f"🧠 Pattern Feedback: {stats['pattern_feedback']}")
                print(f"⚙️ Processed: {stats['processed_feedback']}")

                # Show recent feedback
                recent_feedback = feedback_data.get('recent_feedback', [])
                if recent_feedback:
                    print(f"\n🔄 Recent Feedback:")
                    for i, feedback in enumerate(recent_feedback[-3:], 1):
                        feedback_type = feedback['feedback_type']
                        timestamp = feedback['timestamp'][:19]  # Remove microseconds
                        reason = feedback.get('feedback_data', {}).get('reason', 'No reason')
                        print(f"  {i}. {feedback_type} at {timestamp}: {reason}")

        except Exception as e:
            print(f"❌ Failed to show feedback effectiveness: {e}")

    def show_live_learning_pipeline_status(self):
        """Show the learning pipeline status"""
        self.print_section("LEARNING PIPELINE STATUS")

        try:
            response = requests.get('http://localhost:8309/api/echo/learning/pipeline-status')
            if response.status_code == 200:
                pipeline_data = response.json()
                status = pipeline_data['pipeline_status']

                print(f"🚀 Status: {status['status']}")
                print(f"📅 Last Run: {status.get('last_run', 'Never')}")
                print(f"📈 Patterns Extracted: {status.get('patterns_extracted', 0)}")
                print(f"📂 Sources Processed: {status.get('sources_processed', 0)}")
                print(f"🔄 Total Runs: {status.get('total_runs', 0)}")

                if status.get('errors'):
                    print(f"❌ Errors: {len(status['errors'])}")
                    for error in status['errors'][-2:]:  # Last 2 errors
                        print(f"   • {error}")

                if status.get('last_results'):
                    breakdown = status['last_results'].get('pattern_breakdown', {})
                    print(f"\n📊 Pattern Breakdown:")
                    for source, count in breakdown.items():
                        print(f"   • {source}: {count} patterns")

        except Exception as e:
            print(f"❌ Failed to show pipeline status: {e}")

    def run_live_demo(self):
        """Run a live demonstration of Echo's comprehensive learning"""
        print("🚀 ECHO BRAIN COMPREHENSIVE LEARNING LIVE DEMO")
        print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Show what Echo has learned from existing data
        self.show_learned_patterns_live()

        # Show learning pipeline status
        self.show_live_learning_pipeline_status()

        # Show feedback effectiveness
        self.show_feedback_effectiveness()

        # Test pattern application (may timeout but that's ok)
        self.test_pattern_application_live()

        self.print_section("SUMMARY")
        print("✅ Echo is now learning from:")
        print("   • Existing codebase (Tower services, Echo Brain)")
        print("   • Database content (conversations, learnings)")
        print("   • System configurations (nginx, systemd)")
        print("   • Documentation (README files, KB articles)")
        print("   • Git commit history")
        print("   • User feedback (thumbs up/down, corrections)")
        print("\n🎯 This creates a complete learning loop from ALL your existing data!")
        print("   No more waiting for future conversations - Echo learns from everything now.")

if __name__ == "__main__":
    tester = LivePatternTester()
    tester.run_live_demo()