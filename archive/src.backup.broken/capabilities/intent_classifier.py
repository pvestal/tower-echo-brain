"""
Intent Classification Module
Classifies user intents for autonomous decision making
"""

import re
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class IntentClassifier:
    """Classify intents from text input"""

    def __init__(self):
        # Define intent patterns
        self.intent_patterns = {
            'code_generation': [
                r'write.*code',
                r'create.*function',
                r'implement',
                r'generate.*script',
                r'build.*program'
            ],
            'image_generation': [
                r'generate.*image',
                r'create.*picture',
                r'draw',
                r'anime.*character',
                r'comfyui'
            ],
            'analysis': [
                r'analyze',
                r'examine',
                r'investigate',
                r'review',
                r'audit'
            ],
            'learning': [
                r'learn',
                r'train',
                r'improve',
                r'optimize',
                r'enhance'
            ],
            'monitoring': [
                r'monitor',
                r'watch',
                r'track',
                r'observe',
                r'check.*status'
            ],
            'execution': [
                r'run',
                r'execute',
                r'perform',
                r'start',
                r'launch'
            ],
            'query': [
                r'what.*is',
                r'how.*to',
                r'explain',
                r'tell.*me',
                r'describe'
            ]
        }

        # Define complexity indicators
        self.complexity_keywords = {
            'simple': ['basic', 'simple', 'easy', 'quick'],
            'medium': ['standard', 'normal', 'typical'],
            'complex': ['advanced', 'complex', 'detailed', 'comprehensive']
        }

    def classify(self, text: str) -> Dict[str, Any]:
        """Classify the intent of the given text"""

        text_lower = text.lower()

        # Find matching intents
        matched_intents = []
        confidence_scores = {}

        for intent, patterns in self.intent_patterns.items():
            matches = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    matches += 1

            if matches > 0:
                confidence = min(1.0, matches / len(patterns))
                matched_intents.append(intent)
                confidence_scores[intent] = confidence

        # Determine primary intent
        primary_intent = None
        if matched_intents:
            primary_intent = max(confidence_scores, key=confidence_scores.get)

        # Determine complexity
        complexity = self._determine_complexity(text_lower)

        # Extract entities
        entities = self._extract_entities(text)

        return {
            'primary_intent': primary_intent or 'unknown',
            'all_intents': matched_intents,
            'confidence_scores': confidence_scores,
            'complexity': complexity,
            'entities': entities,
            'requires_confirmation': len(matched_intents) > 1,
            'original_text': text
        }

    def _determine_complexity(self, text: str) -> str:
        """Determine the complexity level of the request"""

        for level, keywords in self.complexity_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return level

        # Default based on text length
        word_count = len(text.split())
        if word_count < 10:
            return 'simple'
        elif word_count < 50:
            return 'medium'
        else:
            return 'complex'

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text"""

        entities = {
            'files': [],
            'functions': [],
            'models': [],
            'numbers': []
        }

        # Extract file paths
        file_pattern = r'[/\w]+\.\w+'
        entities['files'] = re.findall(file_pattern, text)

        # Extract function names
        func_pattern = r'\b([a-z_][a-z0-9_]*)\s*\('
        entities['functions'] = re.findall(func_pattern, text, re.IGNORECASE)

        # Extract model names
        model_keywords = ['llama', 'qwen', 'mixtral', 'deepseek', 'gpt']
        for keyword in model_keywords:
            if keyword in text.lower():
                entities['models'].append(keyword)

        # Extract numbers
        number_pattern = r'\b\d+\b'
        entities['numbers'] = re.findall(number_pattern, text)

        return entities

    def suggest_action(self, classification: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest an action based on intent classification"""

        intent = classification['primary_intent']
        complexity = classification['complexity']

        action_map = {
            'code_generation': {
                'action': 'generate_code',
                'model': 'deepseek-coder-v2:16b' if complexity == 'complex' else 'qwen2.5-coder:7b',
                'priority': 5
            },
            'image_generation': {
                'action': 'generate_image',
                'service': 'comfyui',
                'priority': 3
            },
            'analysis': {
                'action': 'analyze',
                'model': 'llama3.1:8b',
                'priority': 4
            },
            'learning': {
                'action': 'train',
                'service': 'lora_training',
                'priority': 2
            },
            'monitoring': {
                'action': 'monitor',
                'service': 'metrics',
                'priority': 1
            },
            'execution': {
                'action': 'execute',
                'service': 'docker',
                'priority': 5
            },
            'query': {
                'action': 'query',
                'model': 'llama3.1:8b',
                'priority': 4
            }
        }

        suggestion = action_map.get(intent, {
            'action': 'unknown',
            'priority': 0
        })

        suggestion['confidence'] = classification['confidence_scores'].get(intent, 0)
        suggestion['entities'] = classification['entities']

        return suggestion