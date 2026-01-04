#!/usr/bin/env python3
"""
Photo Learning Connector (Schema-Matched)
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
        
        insights = {'total_photos': 0, 'categories': {}}
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM photos")
        insights['total_photos'] = cursor.fetchone()[0]
        
        # Get any other available data
        cursor.execute("PRAGMA table_info(photos)")
        columns = [row[1] for row in cursor.fetchall()]
        insights['columns_available'] = columns
        
        conn.close()
        logger.info(f"Analyzed {insights['total_photos']} photos")
        return insights
        
    def derive_persona_traits(self, insights: Dict) -> Dict:
        traits = {}
        photo_count = insights.get('total_photos', 0)
        
        # Basic trait derivation from photo count
        traits['curiosity_level'] = min(10, photo_count // 100)
        traits['visual_preference'] = min(10, photo_count // 50)
        
        return traits
        
    def update_training_log(self, insights: Dict, traits: Dict):
        training_data = []
        if Path(self.training_log).exists():
            try:
                with open(self.training_log, 'r') as f:
                    data = json.load(f); training_data = data if isinstance(data, list) else []
            except:
                training_data = []
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'source': 'photo_analysis',
            'insights': insights,
            'derived_traits': traits
        }
        
        training_data.append(entry)
        
        with open(self.training_log, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        logger.info(f"Updated training log")
        
    async def run_learning_cycle(self):
        insights = self.analyze_photo_preferences()
        traits = self.derive_persona_traits(insights)
        self.update_training_log(insights, traits)
        return {'insights': insights, 'traits': traits}
