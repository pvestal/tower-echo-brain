import requests
import json
import base64
from pathlib import Path

class EchoVision:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = "llava:7b"
    
    async def analyze_image(self, image_path: str, prompt: str = "Describe this image"):
        """Analyze image using LLaVA"""
        try:
            # Read and encode image
            with open(image_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode()
            
            # Send to LLaVA
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [img_data],
                "stream": False
            }
            
            response = requests.post(self.ollama_url, json=payload)
            result = response.json()
            return result.get('response', 'No analysis available')
            
        except Exception as e:
            return f"Vision analysis failed: {str(e)}"
    
    async def quality_check_anime(self, frame_path: str):
        """Quality check for anime frames"""
        prompt = "Analyze this anime frame for visual quality, art style consistency, and character design. Rate the quality and suggest improvements."
        return await self.analyze_image(frame_path, prompt)
    
    async def analyze_video_frame(self, frame_path: str):
        """Analyze video frame for content and quality"""
        prompt = "Describe the content, quality, and technical aspects of this video frame."
        return await self.analyze_image(frame_path, prompt)
