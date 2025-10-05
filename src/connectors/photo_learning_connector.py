#!/usr/bin/env python3
"""
Photo Learning Connector
Bridges photo analysis with training pipeline
"""

import sqlite3
import json
import asyncio
from pathlib import Path
from typing import Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PhotoLearningConnector:
    def __init__(self, 
                 photos_db: str = "/opt/tower-echo-brain/photos.db",
                 training_log: str = "/opt/tower-echo-brain/vault_data/training_log.json"):
        self.photos_db = photos_db
        self.training_log = training_log
        
    def analyze_photo_preferences(self) -> Dict:
        conn = sqlite3.connect(self.photos_db)
        cursor = conn.cursor()
        
        insights = {'total_photos': 0, 'analyzed_photos': 0, 'categories': {}}
        
        cursor.execute("SELECT COUNT(*) FROM photos WHERE echo_analyzed = 1")
        insights['analyzed_photos'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM photos")
        insights['total_photos'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT categories, COUNT(*) FROM photos WHERE categories IS NOT NULL GROUP BY categories")
        for cat, count in cursor.fetchall():
            if cat:
                insights['categories'][cat] = count
        
        conn.close()
        return insights
        
    def derive_persona_traits(self, insights: Dict) -> Dict:
        traits = {}
        category_count = len(insights.get('categories', {}))
        traits['curiosity_level'] = min(10, category_count)
        traits['technical_depth'] = min(10, insights.get('analyzed_photos', 0) // 100)
        return traits
        
    def update_training_log(self, insights: Dict, traits: Dict):
        training_data = []
        if Path(self.training_log).exists():
            with open(self.training_log, 'r') as f:
                training_data = json.load(f)
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'source': 'photo_analysis',
            'insights': insights,
            'derived_traits': traits
        }
        
        training_data.append(entry)
        
        with open(self.training_log, 'w') as f:
            json.dump(training_data, f, indent=2)
        
    async def run_learning_cycle(self):
        insights = self.analyze_photo_preferences()
        traits = self.derive_persona_traits(insights)
        self.update_training_log(insights, traits)
        return {'insights': insights, 'traits': traits}
