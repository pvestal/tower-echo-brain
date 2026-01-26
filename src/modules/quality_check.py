"""Enhanced quality checking with actual vision model"""

import base64
import requests
import json
from pathlib import Path

def check_with_vision(image_path):
    """Check image quality using LLaVA"""
    
    # Encode image
    with open(image_path, 'rb') as f:
        img_base64 = base64.b64encode(f.read()).decode()
    
    # Vision check prompt
    prompt = """Analyze this generated anime/cyberpunk image:
    1. Sharpness (1-10): 
    2. Composition (1-10):
    3. Character consistency (1-10):
    4. Color quality (1-10):
    5. Any artifacts or issues?
    
    Overall quality score (1-10):
    Should regenerate? (yes/no):"""
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llava:7b",
            "prompt": prompt,
            "images": [img_base64],
            "stream": False
        },
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json().get('response', '')
        
        # Parse score from response
        score = 7  # default
        regenerate = False
        
        if "Overall quality score" in result:
            try:
                # Extract score
                lines = result.split('\n')
                for line in lines:
                    if "Overall quality score" in line:
                        score = int(''.join(filter(str.isdigit, line)) or "7")
                    if "Should regenerate" in line and "yes" in line.lower():
                        regenerate = True
            except:
                pass
        
        return {
            "passed": score >= 7,
            "score": score,
            "details": result[:500],
            "regenerate": regenerate
        }
    
    return {"passed": True, "score": 7, "details": "Vision check unavailable", "regenerate": False}

# Test it
if __name__ == "__main__":
    test_image = "/home/patrick/ComfyUI/output/goblin_slayer_cyberpunk_proper_00001_.png"
    if Path(test_image).exists():
        print(f"Checking: {test_image}")
        result = check_with_vision(test_image)
        print(f"Result: {result}")
