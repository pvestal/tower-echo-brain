#!/usr/bin/env python3
"""
Gradual vector fix for Qdrant - runs in background over time
"""
import requests
import json
import time
import logging
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GradualVectorFixer:
    def __init__(self):
        self.qdrant_url = "http://localhost:6333"
        self.collection = "echo_memory"
        self.ollama_url = "http://localhost:11434"
        self.embedding_model = "mxbai-embed-large:latest"
        self.batch_size = 10  # Small batches to avoid overwhelming
        self.delay_between = 2  # Seconds between batches
        self.max_points = 1000  # Max points to fix per run
        
    def get_points_needing_vectors(self, offset=0, limit=100):
        """Get points without vectors"""
        url = f"{self.qdrant_url}/collections/{self.collection}/points/scroll"
        payload = {
            "offset": offset,
            "limit": limit,
            "with_payload": True,
            "with_vector": True
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            points_needing_vectors = []
            for point in data.get("result", {}).get("points", []):
                if not point.get("vector"):
                    points_needing_vectors.append(point)
            
            return points_needing_vectors, data.get("result", {}).get("next_page_offset")
            
        except Exception as e:
            logger.error(f"Error getting points: {e}")
            return [], None
    
    def generate_embedding(self, text):
        """Generate embedding for text"""
        if not text or len(text.strip()) < 10:
            return None
            
        url = f"{self.ollama_url}/api/embeddings"
        payload = {
            "model": self.embedding_model,
            "prompt": text[:10000]  # Limit text length
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json().get("embedding", [])
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
    
    def update_point_with_vector(self, point_id, vector):
        """Update a single point with its vector"""
        url = f"{self.qdrant_url}/collections/{self.collection}/points"
        payload = {
            "points": [{
                "id": point_id,
                "vector": vector
            }]
        }
        
        try:
            response = requests.put(url, json=payload, timeout=30)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to update point {point_id}: {e}")
            return False
    
    def fix_batch(self, points):
        """Fix a batch of points"""
        fixed = 0
        failed = 0
        
        for point in points:
            point_id = point.get("id")
            text = point.get("payload", {}).get("text", "")
            
            if not text:
                logger.warning(f"Point {point_id} has no text, skipping")
                continue
            
            logger.info(f"Processing point {point_id} ({len(text)} chars)")
            
            vector = self.generate_embedding(text)
            if not vector:
                logger.error(f"Failed to generate vector for point {point_id}")
                failed += 1
                continue
            
            if self.update_point_with_vector(point_id, vector):
                fixed += 1
                logger.info(f"✅ Fixed point {point_id}")
            else:
                failed += 1
                logger.error(f"❌ Failed to update point {point_id}")
            
            # Small delay between points
            time.sleep(0.5)
        
        return fixed, failed
    
    def run_gradual_fix(self, target_points=100):
        """Run gradual fix for specified number of points"""
        logger.info(f"🚀 Starting gradual vector fix for up to {target_points} points")
        
        total_fixed = 0
        total_failed = 0
        offset = 0
        
        while total_fixed < target_points:
            logger.info(f"📋 Getting next batch (offset: {offset})")
            points, next_offset = self.get_points_needing_vectors(offset, self.batch_size)
            
            if not points:
                logger.info("🎉 No more points needing vectors!")
                break
            
            logger.info(f"🔧 Fixing batch of {len(points)} points")
            fixed, failed = self.fix_batch(points)
            
            total_fixed += fixed
            total_failed += failed
            
            logger.info(f"📊 Batch complete: {fixed} fixed, {failed} failed")
            logger.info(f"📈 Total: {total_fixed}/{target_points} points fixed")
            
            if next_offset is None:
                logger.info("📭 No more pages")
                break
            
            offset = next_offset
            
            # Delay between batches
            if total_fixed < target_points:
                logger.info(f"⏳ Waiting {self.delay_between}s before next batch...")
                time.sleep(self.delay_between)
        
        logger.info(f"✅ Gradual fix complete: {total_fixed} points fixed, {total_failed} failed")
        return total_fixed, total_failed

if __name__ == "__main__":
    print("=" * 80)
    print("GRADUAL QDRANT VECTOR FIX")
    print("=" * 80)
    print(f"Start time: {datetime.now().isoformat()}")
    print()
    
    fixer = GradualVectorFixer()
    
    # Check current state first
    points, _ = fixer.get_points_needing_vectors(limit=5)
    if not points:
        print("✅ All points have vectors!")
        sys.exit(0)
    
    print(f"Found {len(points)} points needing vectors (sample)")
    print()
    
    # Ask how many to fix
    try:
        target = int(input("How many points to fix this run? (default: 100): ") or "100")
    except:
        target = 100
    
    print(f"\nFixing up to {target} points...")
    print("This will run in the background and may take a while.")
    print("Check logs at: /var/log/echo-brain-vector-fix.log")
    print()
    
    # Run the fix
    fixed, failed = fixer.run_gradual_fix(target)
    
    print(f"\n🎯 Fix complete: {fixed} points fixed, {failed} failed")
    print(f"End time: {datetime.now().isoformat()}")
