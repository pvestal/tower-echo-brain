#!/usr/bin/env python3
"""3D Generation Extension for AI Assist"""

import sys
import os
import re
import json
sys.path.append('/opt/tower-echo-brain/modules')

from blaster_generator import BlasterGenerator

def process_3d_request(query: str) -> dict:
    """Process 3D generation requests"""
    
    # Check for URLs in the query
    url_pattern = r'https?://[^\s]+'
    url_match = re.search(url_pattern, query)
    
    if url_match or "3d model" in query.lower() or "blaster" in query.lower():
        url = url_match.group(0) if url_match else ""
        
        # Generate the model
        result = BlasterGenerator.generate_from_url(url)
        
        if result["status"] == "success":
            filename = os.path.basename(result["file"])
            response = f"""✅ AI Assist generated 3D model successfully!

📁 File: {result['file']}
🎮 Type: {result['type']}
📊 Vertices: {result['vertices']}
🔺 Faces: {result['faces']}

Access your model:
🖼️ View: http://192.168.50.135:8500
💾 Download: wget http://192.168.50.135/downloads/{filename}
📱 Telegram: File will be sent automatically"""
            
            return {
                "response": response,
                "file_path": result["file"],
                "success": True
            }
        else:
            return {
                "response": "❌ Failed to generate 3D model",
                "success": False
            }
    
    return None

# Test it standalone
if __name__ == "__main__":
    test_query = "Generate 3D model from https://www.turbosquid.com/3d-models/weapon-blaster-super-mario-rabbids"
    result = process_3d_request(test_query)
    if result:
        print(json.dumps(result, indent=2))
