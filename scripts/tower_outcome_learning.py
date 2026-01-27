#!/usr/bin/env python3
"""
Implement learning from repair outcomes for Echo Brain
Track what worked, what didn't, and adapt future responses
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

class TowerOutcomeLearning:
    def __init__(self):
        # Create lightweight SQLite database for outcome tracking
        self.db_path = Path('/opt/tower-echo-brain/data/repair_outcomes.db')
        self.db_path.parent.mkdir(exist_ok=True)
        self.init_database()

    def init_database(self):
        """Initialize repair outcomes database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS repair_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                service TEXT,
                issue TEXT,
                action_taken TEXT,
                success BOOLEAN,
                time_to_resolve REAL,
                error_message TEXT,
                confidence_score REAL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learned_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT UNIQUE,
                success_rate REAL,
                avg_time_to_resolve REAL,
                last_updated TEXT,
                priority INTEGER
            )
        ''')

        conn.commit()
        conn.close()

    def record_repair_attempt(self, service: str, issue: str, action: str,
                             success: bool, time_taken: float = None,
                             error: str = None):
        """Record outcome of a repair attempt"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO repair_attempts
            (timestamp, service, issue, action_taken, success, time_to_resolve, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            service,
            issue,
            action,
            success,
            time_taken,
            error
        ))

        conn.commit()
        conn.close()

        # Update learned patterns
        self.update_patterns(service, issue, action, success)

    def update_patterns(self, service: str, issue: str, action: str, success: bool):
        """Update learned patterns based on outcomes"""
        pattern = f"{service}:{issue}:{action}"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if pattern exists
        cursor.execute('SELECT * FROM learned_patterns WHERE pattern = ?', (pattern,))
        existing = cursor.fetchone()

        if existing:
            # Update existing pattern
            cursor.execute('''
                UPDATE learned_patterns
                SET success_rate = (
                    SELECT AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END)
                    FROM repair_attempts
                    WHERE service = ? AND issue = ? AND action_taken = ?
                ),
                avg_time_to_resolve = (
                    SELECT AVG(time_to_resolve)
                    FROM repair_attempts
                    WHERE service = ? AND issue = ? AND action_taken = ? AND success = 1
                ),
                last_updated = ?
                WHERE pattern = ?
            ''', (service, issue, action, service, issue, action,
                 datetime.now().isoformat(), pattern))
        else:
            # Create new pattern
            cursor.execute('''
                INSERT INTO learned_patterns
                (pattern, success_rate, last_updated, priority)
                VALUES (?, ?, ?, ?)
            ''', (pattern, 1.0 if success else 0.0, datetime.now().isoformat(), 1))

        conn.commit()
        conn.close()

    def get_best_action(self, service: str, issue: str) -> Dict[str, Any]:
        """Get the best action based on learned outcomes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Find successful patterns for this service/issue
        cursor.execute('''
            SELECT action_taken,
                   COUNT(*) as attempts,
                   AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                   AVG(time_to_resolve) as avg_time
            FROM repair_attempts
            WHERE service = ? AND issue LIKE ?
            GROUP BY action_taken
            ORDER BY success_rate DESC, avg_time ASC
            LIMIT 1
        ''', (service, f'%{issue}%'))

        result = cursor.fetchone()
        conn.close()

        if result:
            return {
                'action': result[0],
                'confidence': result[2],
                'avg_time': result[3],
                'attempts': result[1]
            }
        return None

    def generate_report(self) -> str:
        """Generate learning report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get overall stats
        cursor.execute('SELECT COUNT(*), AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) FROM repair_attempts')
        total_attempts, overall_success_rate = cursor.fetchone()

        # Get top patterns
        cursor.execute('''
            SELECT pattern, success_rate, avg_time_to_resolve
            FROM learned_patterns
            WHERE success_rate > 0.7
            ORDER BY success_rate DESC
            LIMIT 10
        ''')
        top_patterns = cursor.fetchall()

        conn.close()

        report = f"""
ECHO BRAIN LEARNING REPORT
Generated: {datetime.now().isoformat()}
=====================================

Overall Statistics:
- Total repair attempts: {total_attempts or 0}
- Overall success rate: {(overall_success_rate or 0) * 100:.1f}%

Top Successful Patterns:
"""
        for pattern, success_rate, avg_time in top_patterns:
            parts = pattern.split(':', 2)  # Split into max 3 parts
            if len(parts) >= 3:
                service, issue, action = parts[0], parts[1], ':'.join(parts[2:])
            else:
                service, issue, action = parts[0] if parts else 'unknown', parts[1] if len(parts) > 1 else 'unknown', 'unknown'
            report += f"- {service} [{issue}]: {action} (Success: {success_rate*100:.0f}%"
            if avg_time:
                report += f", Avg time: {avg_time:.1f}s"
            report += ")\n"

        return report

# Integration with Echo Brain
def integrate_with_echo_brain():
    """Create hooks for Echo Brain to use outcome learning"""

    learner = TowerOutcomeLearning()

    # Example: Record a successful repair
    learner.record_repair_attempt(
        service="tower-echo-brain",
        issue="port conflict",
        action="kill process and restart",
        success=True,
        time_taken=5.2
    )

    # Example: Get best action for an issue
    best_action = learner.get_best_action("tower-echo-brain", "port conflict")
    if best_action:
        print(f"Recommended action: {best_action['action']}")
        print(f"Confidence: {best_action['confidence']*100:.0f}%")

    # Generate report
    print(learner.generate_report())

if __name__ == "__main__":
    print("🧠 Tower Outcome Learning System")
    print("================================")

    learner = TowerOutcomeLearning()

    # Simulate some learning
    repairs = [
        ("tower-anime-production", "port 8305 in use", "change to port 8328", True, 3.0),
        ("tower-apple-music", "port 8315 in use", "change to port 8306", True, 2.5),
        ("tower-echo-brain", "import error", "fix import paths", True, 10.0),
        ("learning-pipeline", "missing directory", "create cache directory", True, 1.0),
        ("tower-kb", "connection refused", "restart service", False, None),
        ("tower-kb", "connection refused", "fix database password", True, 5.0),
    ]

    print("\n📝 Recording repair outcomes...")
    for service, issue, action, success, time in repairs:
        learner.record_repair_attempt(service, issue, action, success, time)
        status = "✅" if success else "❌"
        print(f"   {status} {service}: {action}")

    print("\n🎯 Testing best action retrieval...")
    test_cases = [
        ("tower-kb", "connection refused"),
        ("tower-anime-production", "port"),
    ]

    for service, issue in test_cases:
        best = learner.get_best_action(service, issue)
        if best:
            print(f"   {service} [{issue}]: {best['action']} (confidence: {best['confidence']*100:.0f}%)")

    print("\n" + learner.generate_report())

    print("\n✅ Outcome learning system ready!")
    print("   Database: /opt/tower-echo-brain/data/repair_outcomes.db")
    print("   Echo Brain can now learn from repair attempts!")