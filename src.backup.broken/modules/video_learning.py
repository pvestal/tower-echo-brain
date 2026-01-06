"""Echo learns from successful workflows and improves over time"""
import json
import sqlite3
from pathlib import Path
from typing import Dict

class VideoLearningSystem:
    def __init__(self):
        self.db_path = Path("/opt/tower-echo-brain/workflows.db")
        self.init_db()
        
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''CREATE TABLE IF NOT EXISTS workflow_history (
            id INTEGER PRIMARY KEY,
            workflow TEXT,
            prompt TEXT,
            style TEXT,
            success BOOLEAN,
            quality_score REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        conn.commit()
        conn.close()
        
    def learn_from_success(self, workflow: Dict, prompt: str, style: str):
        """Save successful workflow for future use"""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO workflow_history (workflow, prompt, style, success, quality_score) VALUES (?, ?, ?, ?, ?)",
            (json.dumps(workflow), prompt, style, True, 0.8)
        )
        conn.commit()
        conn.close()
        
    def get_best_workflow_for_style(self, style: str) -> Dict:
        """Get the best performing workflow for a style"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT workflow FROM workflow_history WHERE style = ? AND success = 1 ORDER BY quality_score DESC LIMIT 1",
            (style,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
        return None
        
    def improve_from_feedback(self, workflow_id: int, quality_score: float):
        """Update quality score based on user feedback"""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE workflow_history SET quality_score = ? WHERE id = ?",
            (quality_score, workflow_id)
        )
        conn.commit()
        conn.close()

print("Echo learning system created - will improve with each successful generation")
