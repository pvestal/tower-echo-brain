#!/usr/bin/env python3
"""
Intelligent Generation Orchestrator
Uses story_bible data to make smart decisions about models, LoRAs, workflows, and prompts
"""

import json
import requests
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Configuration
COMFYUI_URL = "http://localhost:8188"
QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
COLLECTION_NAME = "story_bible"

WORKFLOW_DIR = "/opt/tower-anime-production/workflows/comfyui"
MODELS_DIR = "/mnt/1TB-storage/models"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GenerationRequest:
    """Parsed user request for generation"""
    characters: List[str]
    scene_type: str
    style: str
    mood: str
    quality: str
    raw_request: str

@dataclass
class ResourceSelection:
    """Selected resources for generation"""
    workflow_file: str
    base_model: str
    loras: List[Dict[str, float]]  # [{"name": "lora.safetensors", "strength": 0.8}]
    positive_prompt: str
    negative_prompt: str
    generation_settings: Dict[str, Any]

class ContentAnalyzer:
    """Analyzes user requests to extract generation requirements"""

    def __init__(self):
        self.character_patterns = [
            r'\b(Kai|Mei|Yuki|Takeshi|Ryuu|Elena)\b',
            r'character\s+(\w+)',
            r'(\w+)\s+character'
        ]

        self.scene_patterns = {
            'action': [r'fight', r'combat', r'battle', r'action', r'attack'],
            'dialogue': [r'talk', r'conversation', r'dialogue', r'speak'],
            'romance': [r'romance', r'kiss', r'love', r'romantic'],
            'casual': [r'casual', r'daily', r'normal', r'everyday']
        }

        self.style_patterns = {
            'cyberpunk': [r'cyberpunk', r'cyber', r'neon', r'futuristic', r'tech'],
            'anime': [r'anime', r'manga', r'japanese'],
            'realistic': [r'realistic', r'photorealistic', r'real']
        }

    def analyze_request(self, request: str) -> GenerationRequest:
        """Parse user request into structured requirements"""
        request_lower = request.lower()

        # Extract characters
        characters = []
        for pattern in self.character_patterns:
            matches = re.findall(pattern, request, re.IGNORECASE)
            characters.extend(matches)

        # Remove duplicates, keep order
        characters = list(dict.fromkeys(characters))

        # Determine scene type
        scene_type = 'casual'  # default
        for scene, patterns in self.scene_patterns.items():
            if any(re.search(pattern, request_lower) for pattern in patterns):
                scene_type = scene
                break

        # Determine style
        style = 'anime'  # default
        for style_name, patterns in self.style_patterns.items():
            if any(re.search(pattern, request_lower) for pattern in patterns):
                style = style_name
                break

        # Determine mood and quality
        mood = 'neutral'
        if any(word in request_lower for word in ['dark', 'serious', 'intense']):
            mood = 'dark'
        elif any(word in request_lower for word in ['bright', 'happy', 'cheerful']):
            mood = 'bright'

        quality = 'high'
        if any(word in request_lower for word in ['quick', 'draft', 'test']):
            quality = 'draft'
        elif any(word in request_lower for word in ['best', 'masterpiece', 'perfect']):
            quality = 'masterpiece'

        return GenerationRequest(
            characters=characters,
            scene_type=scene_type,
            style=style,
            mood=mood,
            quality=quality,
            raw_request=request
        )

class StoryBibleSearcher:
    """Searches story_bible for relevant character and scene data"""

    async def get_character_data(self, character_name: str) -> Optional[Dict]:
        """Get detailed character data from story_bible"""
        try:
            # Generate embedding for search
            embed_resp = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={'model': 'nomic-embed-text', 'prompt': f"{character_name} character description"}
            )

            if embed_resp.status_code != 200:
                return None

            embedding = embed_resp.json()['embedding']

            # Search for character
            search_resp = requests.post(
                f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search",
                json={
                    'vector': embedding,
                    'limit': 10,
                    'with_payload': True,
                    'filter': {
                        'must': [{'key': 'type', 'match': {'value': 'character'}}]
                    }
                }
            )

            if search_resp.status_code == 200:
                results = search_resp.json()['result']

                # Find exact character match
                for result in results:
                    payload = result['payload']
                    if character_name.lower() in payload.get('name', '').lower():
                        return {
                            'name': payload['name'],
                            'content': payload['content'],
                            'project_id': payload.get('project_id'),
                            'score': result['score']
                        }

                # If no exact match, return best match
                if results:
                    payload = results[0]['payload']
                    return {
                        'name': payload['name'],
                        'content': payload['content'],
                        'project_id': payload.get('project_id'),
                        'score': results[0]['score']
                    }

        except Exception as e:
            logger.error(f"Failed to get character data for {character_name}: {e}")

        return None

    async def get_scene_context(self, scene_type: str, style: str) -> List[Dict]:
        """Get relevant scene and workflow data"""
        try:
            query = f"{scene_type} {style} scene workflow"

            embed_resp = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={'model': 'nomic-embed-text', 'prompt': query}
            )

            if embed_resp.status_code != 200:
                return []

            embedding = embed_resp.json()['embedding']

            # Search for scenes and workflows
            search_resp = requests.post(
                f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search",
                json={
                    'vector': embedding,
                    'limit': 5,
                    'with_payload': True
                }
            )

            if search_resp.status_code == 200:
                results = search_resp.json()['result']
                return [
                    {
                        'type': result['payload']['type'],
                        'content': result['payload']['content'],
                        'metadata': result['payload'],
                        'score': result['score']
                    }
                    for result in results
                ]

        except Exception as e:
            logger.error(f"Failed to get scene context: {e}")

        return []

class ResourceSelector:
    """Selects optimal models, LoRAs, and workflows based on requirements"""

    def __init__(self):
        # Model recommendations based on style
        self.model_preferences = {
            'cyberpunk': ['cyberrealistic_v9.safetensors', 'realistic_vision_v51.safetensors'],
            'anime': ['AOM3A1B.safetensors', 'chilloutmix.safetensors'],
            'realistic': ['realistic_vision_v51.safetensors', 'epicrealism_v5.safetensors']
        }

        # Workflow preferences based on scene type and requirements
        self.workflow_preferences = {
            ('action', True): 'anime_30sec_rife_workflow_with_lora.json',  # (scene_type, needs_lora)
            ('action', False): 'ACTION_combat_workflow.json',
            ('dialogue', True): 'anime_30sec_working_workflow.json',
            ('casual', True): 'anime_30sec_rife_workflow_with_lora.json',
            ('default', True): 'anime_30sec_rife_workflow_with_lora.json',
            ('default', False): 'anime_30sec_working_workflow.json'
        }

        # LoRA patterns for character matching
        self.lora_patterns = {
            'kai': ['kai_cyberpunk_slayer.safetensors', 'kai_nakamura_optimized_v1.safetensors'],
            'mei': ['mei_working_v1.safetensors'],
            'cyberpunk': ['cyberpunk_style_proper.safetensors', 'cyberpunk_style.safetensors']
        }

    def select_resources(self, request: GenerationRequest, character_data: List[Dict]) -> ResourceSelection:
        """Select optimal resources for generation"""

        # Determine if we need LoRAs
        needs_lora = bool(request.characters) or request.style == 'cyberpunk'

        # Select workflow
        workflow_key = (request.scene_type, needs_lora)
        if workflow_key not in self.workflow_preferences:
            workflow_key = ('default', needs_lora)

        workflow_file = self.workflow_preferences[workflow_key]

        # Select base model
        preferred_models = self.model_preferences.get(request.style, self.model_preferences['anime'])
        base_model = self._select_available_model(preferred_models)

        # Select LoRAs
        loras = []
        if needs_lora:
            loras = self._select_loras(request, character_data)

        # Build prompts
        positive_prompt = self._build_positive_prompt(request, character_data)
        negative_prompt = self._build_negative_prompt(request)

        # Generation settings based on quality
        generation_settings = self._get_generation_settings(request.quality)

        return ResourceSelection(
            workflow_file=workflow_file,
            base_model=base_model,
            loras=loras,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            generation_settings=generation_settings
        )

    def _select_available_model(self, preferred_models: List[str]) -> str:
        """Select first available model from preferences"""
        try:
            available_models = list(Path(f"{MODELS_DIR}/checkpoints").glob("*.safetensors"))
            available_names = [m.name for m in available_models]

            for preferred in preferred_models:
                if preferred in available_names:
                    return preferred

            # Fallback to first available
            return available_names[0] if available_names else "AOM3A1B.safetensors"

        except Exception as e:
            logger.warning(f"Could not check available models: {e}")
            return "AOM3A1B.safetensors"

    def _select_loras(self, request: GenerationRequest, character_data: List[Dict]) -> List[Dict[str, float]]:
        """Select appropriate LoRAs"""
        loras = []

        try:
            available_loras = list(Path(f"{MODELS_DIR}/loras").glob("*.safetensors"))
            available_names = [l.name for l in available_loras]

            # Character-specific LoRAs
            for char in request.characters:
                char_lower = char.lower()
                for pattern, lora_list in self.lora_patterns.items():
                    if pattern in char_lower:
                        for lora in lora_list:
                            if lora in available_names:
                                loras.append({"name": lora, "strength": 0.8})
                                break

            # Style-specific LoRAs
            if request.style == 'cyberpunk':
                for lora in self.lora_patterns.get('cyberpunk', []):
                    if lora in available_names and not any(l['name'] == lora for l in loras):
                        loras.append({"name": lora, "strength": 0.6})
                        break

        except Exception as e:
            logger.warning(f"Could not select LoRAs: {e}")

        return loras

    def _build_positive_prompt(self, request: GenerationRequest, character_data: List[Dict]) -> str:
        """Build detailed positive prompt using character data"""
        prompt_parts = []

        # Quality prefix
        if request.quality == 'masterpiece':
            prompt_parts.append("masterpiece, best quality, ultra detailed, 8k resolution")
        elif request.quality == 'high':
            prompt_parts.append("masterpiece, best quality, high resolution, detailed")
        else:
            prompt_parts.append("best quality, detailed")

        # Character-specific details
        for char_data in character_data:
            if char_data:
                content = char_data.get('content', '')

                # Extract key visual details from character content
                visual_details = self._extract_visual_details(content)
                if visual_details:
                    prompt_parts.append(visual_details)

        # Scene and style
        scene_descriptors = {
            'action': "dynamic action scene, combat pose, intense movement",
            'dialogue': "conversation scene, character interaction, expressive faces",
            'romance': "romantic scene, intimate lighting, soft atmosphere",
            'casual': "casual scene, natural pose, relaxed atmosphere"
        }

        if request.scene_type in scene_descriptors:
            prompt_parts.append(scene_descriptors[request.scene_type])

        # Style descriptors
        style_descriptors = {
            'cyberpunk': "cyberpunk aesthetic, neon lighting, futuristic technology, dark atmosphere",
            'anime': "anime style, manga aesthetic, Japanese animation style",
            'realistic': "photorealistic, realistic lighting, detailed textures"
        }

        if request.style in style_descriptors:
            prompt_parts.append(style_descriptors[request.style])

        return ", ".join(prompt_parts)

    def _extract_visual_details(self, content: str) -> str:
        """Extract visual description from character content"""
        # Look for appearance-related information
        visual_keywords = ['hair', 'eyes', 'clothing', 'appearance', 'style', 'build', 'height']

        lines = content.split('\n')
        visual_lines = []

        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in visual_keywords):
                # Clean up the line
                clean_line = line.strip()
                if ':' in clean_line:
                    clean_line = clean_line.split(':', 1)[1].strip()
                if clean_line and len(clean_line) > 5:
                    visual_lines.append(clean_line)

        return ", ".join(visual_lines[:3])  # Limit to 3 visual details

    def _build_negative_prompt(self, request: GenerationRequest) -> str:
        """Build negative prompt based on requirements"""
        base_negative = "worst quality, low quality, blurry, ugly, distorted"

        # Add scene-specific negatives
        if request.scene_type == 'action':
            base_negative += ", static pose, standing still, no movement"
        elif request.scene_type == 'dialogue':
            base_negative += ", no expression, blank face, closed mouth"

        # Add style-specific negatives
        if request.style == 'cyberpunk':
            base_negative += ", bright lighting, cheerful, colorful, medieval"
        elif request.style == 'realistic':
            base_negative += ", cartoon, anime, illustration"

        return base_negative

    def _get_generation_settings(self, quality: str) -> Dict[str, Any]:
        """Get generation settings based on quality requirement"""
        settings = {
            'masterpiece': {'steps': 40, 'cfg': 8.0, 'sampler': 'dpmpp_2m'},
            'high': {'steps': 30, 'cfg': 7.5, 'sampler': 'dpmpp_2m'},
            'draft': {'steps': 20, 'cfg': 7.0, 'sampler': 'euler'}
        }

        return settings.get(quality, settings['high'])

class IntelligentOrchestrator:
    """Main orchestration service"""

    def __init__(self):
        self.content_analyzer = ContentAnalyzer()
        self.searcher = StoryBibleSearcher()
        self.selector = ResourceSelector()

    async def generate_intelligently(self, user_request: str) -> Dict[str, Any]:
        """Execute intelligent generation pipeline"""
        logger.info(f"Starting intelligent generation for: {user_request}")

        try:
            # Step 1: Analyze content
            request = self.content_analyzer.analyze_request(user_request)
            logger.info(f"Parsed request: characters={request.characters}, scene={request.scene_type}, style={request.style}")

            # Step 2: Get character data
            character_data = []
            for char in request.characters:
                char_data = await self.searcher.get_character_data(char)
                character_data.append(char_data)
                if char_data:
                    logger.info(f"Found character data for {char}: {char_data['name']} (score: {char_data['score']:.3f})")

            # Step 3: Get scene context
            scene_context = await self.searcher.get_scene_context(request.scene_type, request.style)
            logger.info(f"Retrieved {len(scene_context)} scene context items")

            # Step 4: Select resources
            resources = self.selector.select_resources(request, character_data)
            logger.info(f"Selected resources:")
            logger.info(f"  Workflow: {resources.workflow_file}")
            logger.info(f"  Model: {resources.base_model}")
            logger.info(f"  LoRAs: {resources.loras}")
            logger.info(f"  Prompt: {resources.positive_prompt[:100]}...")

            # Return the orchestration plan (not executing ComfyUI yet for testing)
            return {
                'request_analysis': request.__dict__,
                'character_data': character_data,
                'scene_context': scene_context,
                'resource_selection': resources.__dict__,
                'status': 'orchestration_complete',
                'ready_for_generation': True
            }

        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'ready_for_generation': False
            }

# Test function
async def test_intelligent_orchestration():
    """Test the intelligent orchestration with various requests"""
    orchestrator = IntelligentOrchestrator()

    test_requests = [
        "Generate Kai fighting cyberpunk goblins in Tokyo",
        "Show Mei in a romantic scene",
        "Create a casual dialogue between Yuki and Takeshi"
    ]

    for request in test_requests:
        print(f"\n{'='*60}")
        print(f"Testing: {request}")
        print('='*60)

        result = await orchestrator.generate_intelligently(request)

        if result['ready_for_generation']:
            resources = result['resource_selection']
            print(f"✅ ORCHESTRATION SUCCESS")
            print(f"📁 Workflow: {resources['workflow_file']}")
            print(f"🎨 Model: {resources['base_model']}")
            print(f"🔧 LoRAs: {len(resources['loras'])} selected")
            for lora in resources['loras']:
                print(f"   - {lora['name']} (strength: {lora['strength']})")
            print(f"📝 Prompt: {resources['positive_prompt'][:200]}...")
            print(f"🚫 Negative: {resources['negative_prompt']}")
        else:
            print(f"❌ ORCHESTRATION FAILED: {result.get('error')}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_intelligent_orchestration())