#!/usr/bin/env python3
"""
Style Learning Integration Engine
Connects Echo's preference learning system with anime generation requests.

This engine:
- Learns Patrick's style preferences from generation feedback
- Automatically applies learned preferences to new requests
- Builds feedback loops from generation results to preference updates
- Tracks creative decision patterns and learns from them
- Provides intelligent style suggestions based on learned patterns
"""

import asyncio
import json
import logging
import os
import psycopg2
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import sys
import os
sys.path.append('/opt/tower-echo-brain/src')

from db.database import database

logger = logging.getLogger(__name__)

@dataclass
class StylePreference:
    """Individual style preference with learning metadata"""
    element: str
    category: str  # quality, lighting, composition, character, color, mood
    confidence: float  # 0.0 to 1.0
    usage_count: int
    last_used: datetime
    positive_feedback_count: int
    negative_feedback_count: int
    context_tags: List[str]

@dataclass
class LearnedPreferences:
    """Collection of learned style preferences"""
    user_id: str
    preferences: Dict[str, StylePreference]
    quality_settings: Dict[str, Any]
    generation_patterns: Dict[str, Any]
    learning_confidence: float
    last_updated: datetime

class StyleLearningEngine:
    """Engine for learning and applying Patrick's anime style preferences"""

    def __init__(self):
        self.db_config = {
            "database": os.environ.get("DB_NAME", "echo_brain"),
            "user": os.environ.get("DB_USER", "patrick")
        }

        # Style categories for classification
        self.style_categories = {
            "quality": ["high quality", "detailed", "professional", "cinematic", "masterpiece", "best quality"],
            "lighting": ["dramatic lighting", "soft lighting", "volumetric lighting", "rim lighting", "ambient lighting"],
            "composition": ["close-up", "wide shot", "portrait", "full body", "medium shot", "extreme close-up"],
            "character": ["anime", "manga", "photorealistic", "cartoon", "realistic", "stylized"],
            "color": ["vibrant colors", "muted colors", "monochrome", "warm colors", "cool colors", "neon"],
            "mood": ["dramatic", "peaceful", "energetic", "mysterious", "romantic", "action-packed"]
        }

        # Default quality baselines
        self.default_preferences = {
            "quality": ["high quality", "detailed", "professional anime style"],
            "lighting": ["dramatic lighting"],
            "character": ["anime style", "detailed character design"],
            "technical": ["8k", "ultra detailed", "sharp focus"]
        }

    async def analyze_generation_request(self, prompt: str, user_id: str = "patrick") -> Dict[str, Any]:
        """Analyze generation request and suggest style improvements"""
        try:
            # Get learned preferences for user
            learned_prefs = await self.get_learned_preferences(user_id)

            # Analyze current prompt
            current_elements = self.extract_style_elements(prompt)

            # Suggest improvements based on learning
            suggestions = await self.generate_style_suggestions(current_elements, learned_prefs)

            # Build enhanced prompt
            enhanced_prompt = await self.enhance_prompt_with_learning(prompt, learned_prefs, suggestions)

            return {
                "original_prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "current_elements": current_elements,
                "suggestions": suggestions,
                "learning_confidence": learned_prefs.learning_confidence,
                "applied_preferences": len(suggestions.get("applied", []))
            }

        except Exception as e:
            logger.error(f"Failed to analyze generation request: {e}")
            return {
                "original_prompt": prompt,
                "enhanced_prompt": prompt,
                "error": str(e)
            }

    async def get_learned_preferences(self, user_id: str) -> LearnedPreferences:
        """Get user's learned style preferences"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Get recent style learning data
            cursor.execute("""
                SELECT style_elements, quality_assessment, user_feedback, context_tags, created_at
                FROM anime_echo_style_learning
                WHERE user_id = %s
                AND user_feedback->>'rating' IS NOT NULL
                ORDER BY created_at DESC LIMIT 100
            """, (user_id,))

            learning_data = cursor.fetchall()
            cursor.close()
            conn.close()

            # Process learning data into preferences
            preferences = {}
            quality_settings = {}
            generation_patterns = {}

            total_feedback = len(learning_data)
            positive_count = 0

            for row in learning_data:
                style_elements, quality_assessment, user_feedback, context_tags, created_at = row

                # Parse feedback
                feedback = user_feedback or {}
                if isinstance(feedback, str):
                    feedback = json.loads(feedback)

                rating = feedback.get("rating", 0)
                if rating >= 4:  # Positive feedback
                    positive_count += 1

                    # Process style elements
                    elements = style_elements or []
                    if isinstance(elements, str):
                        elements = json.loads(elements)

                    for element in elements:
                        if element and element.strip():
                            self._update_preference_learning(preferences, element, rating, context_tags or [])

                    # Process quality settings
                    if quality_assessment:
                        assessment = quality_assessment
                        if isinstance(assessment, str):
                            assessment = json.loads(assessment)

                        settings = assessment.get("settings_used", {})
                        if settings and rating >= 4:
                            for key, value in settings.items():
                                if key not in quality_settings:
                                    quality_settings[key] = {"values": [], "confidence": 0}
                                quality_settings[key]["values"].append(value)

            # Calculate quality setting preferences
            for key, data in quality_settings.items():
                values = data["values"]
                if values:
                    # Use most common value
                    quality_settings[key] = {
                        "preferred_value": max(set(values), key=values.count),
                        "confidence": len(values) / total_feedback if total_feedback > 0 else 0
                    }

            # Calculate learning confidence
            learning_confidence = positive_count / total_feedback if total_feedback > 0 else 0.1

            return LearnedPreferences(
                user_id=user_id,
                preferences=preferences,
                quality_settings=quality_settings,
                generation_patterns=generation_patterns,
                learning_confidence=learning_confidence,
                last_updated=datetime.now()
            )

        except Exception as e:
            logger.error(f"Failed to get learned preferences: {e}")
            return LearnedPreferences(
                user_id=user_id,
                preferences={},
                quality_settings={},
                generation_patterns={},
                learning_confidence=0.1,
                last_updated=datetime.now()
            )

    def _update_preference_learning(self, preferences: Dict, element: str, rating: int, context_tags: List[str]):
        """Update preference learning data for an element"""
        element_clean = element.strip().lower()

        if element_clean not in preferences:
            # Categorize the element
            category = self._categorize_style_element(element_clean)

            preferences[element_clean] = StylePreference(
                element=element_clean,
                category=category,
                confidence=0.0,
                usage_count=0,
                last_used=datetime.now(),
                positive_feedback_count=0,
                negative_feedback_count=0,
                context_tags=[]
            )

        pref = preferences[element_clean]
        pref.usage_count += 1
        pref.last_used = datetime.now()

        if rating >= 4:
            pref.positive_feedback_count += 1
        elif rating <= 2:
            pref.negative_feedback_count += 1

        # Update context tags
        for tag in context_tags:
            if tag not in pref.context_tags:
                pref.context_tags.append(tag)

        # Calculate confidence
        total_feedback = pref.positive_feedback_count + pref.negative_feedback_count
        if total_feedback > 0:
            pref.confidence = pref.positive_feedback_count / total_feedback
        else:
            pref.confidence = 0.5

    def _categorize_style_element(self, element: str) -> str:
        """Categorize a style element into its type"""
        element_lower = element.lower()

        for category, keywords in self.style_categories.items():
            for keyword in keywords:
                if keyword.lower() in element_lower:
                    return category

        # Default categorization based on common patterns
        if any(word in element_lower for word in ["lighting", "light", "shadow"]):
            return "lighting"
        elif any(word in element_lower for word in ["shot", "view", "angle", "composition"]):
            return "composition"
        elif any(word in element_lower for word in ["color", "palette", "hue", "saturation"]):
            return "color"
        elif any(word in element_lower for word in ["quality", "detailed", "resolution", "sharp"]):
            return "quality"
        elif any(word in element_lower for word in ["mood", "atmosphere", "feeling", "emotion"]):
            return "mood"
        else:
            return "character"

    def extract_style_elements(self, prompt: str) -> Dict[str, List[str]]:
        """Extract style elements from prompt text"""
        elements = {
            "quality": [],
            "lighting": [],
            "composition": [],
            "character": [],
            "color": [],
            "mood": [],
            "technical": [],
            "other": []
        }

        prompt_lower = prompt.lower()

        # Extract elements by category
        for category, keywords in self.style_categories.items():
            for keyword in keywords:
                if keyword.lower() in prompt_lower:
                    elements[category].append(keyword)

        # Extract technical terms
        technical_terms = ["8k", "4k", "ultra detailed", "sharp focus", "depth of field", "bokeh"]
        for term in technical_terms:
            if term.lower() in prompt_lower:
                elements["technical"].append(term)

        # Extract other style terms using regex
        style_patterns = [
            r"\b(\w+)\s+style\b",
            r"\b(\w+)\s+aesthetic\b",
            r"\bby\s+(\w+)\b",  # Artist names
            r"\b(\w+)\s+art\b"
        ]

        for pattern in style_patterns:
            matches = re.findall(pattern, prompt_lower)
            for match in matches:
                if match not in elements["other"]:
                    elements["other"].append(match)

        return elements

    async def generate_style_suggestions(self, current_elements: Dict, learned_prefs: LearnedPreferences) -> Dict[str, Any]:
        """Generate style suggestions based on learned preferences"""
        suggestions = {
            "add": [],
            "improve": [],
            "remove": [],
            "applied": [],
            "confidence_scores": {}
        }

        try:
            # Suggest additions based on high-confidence preferences
            for element, pref in learned_prefs.preferences.items():
                if pref.confidence >= 0.7 and pref.usage_count >= 3:
                    # Check if element category is missing or weak in current prompt
                    current_category_elements = current_elements.get(pref.category, [])

                    if len(current_category_elements) == 0:
                        suggestions["add"].append({
                            "element": pref.element,
                            "category": pref.category,
                            "confidence": pref.confidence,
                            "reason": f"Highly rated {pref.category} preference"
                        })
                    elif len(current_category_elements) < 2 and pref.element not in str(current_category_elements).lower():
                        suggestions["improve"].append({
                            "element": pref.element,
                            "category": pref.category,
                            "confidence": pref.confidence,
                            "reason": f"Additional {pref.category} enhancement"
                        })

            # Suggest quality improvements
            if len(current_elements.get("quality", [])) == 0:
                for default_quality in self.default_preferences["quality"]:
                    suggestions["add"].append({
                        "element": default_quality,
                        "category": "quality",
                        "confidence": 0.9,
                        "reason": "Default quality enhancement"
                    })

            # Suggest technical improvements
            if len(current_elements.get("technical", [])) == 0:
                for tech_term in self.default_preferences["technical"]:
                    suggestions["add"].append({
                        "element": tech_term,
                        "category": "technical",
                        "confidence": 0.8,
                        "reason": "Technical quality enhancement"
                    })

            # Suggest removals for low-confidence elements (if any detected)
            for element, pref in learned_prefs.preferences.items():
                if pref.confidence <= 0.3 and pref.negative_feedback_count > pref.positive_feedback_count:
                    suggestions["remove"].append({
                        "element": pref.element,
                        "category": pref.category,
                        "confidence": pref.confidence,
                        "reason": "Low-rated element to avoid"
                    })

            # Limit suggestions to avoid prompt bloat
            suggestions["add"] = suggestions["add"][:5]
            suggestions["improve"] = suggestions["improve"][:3]

        except Exception as e:
            logger.error(f"Failed to generate style suggestions: {e}")

        return suggestions

    async def enhance_prompt_with_learning(self, original_prompt: str, learned_prefs: LearnedPreferences,
                                         suggestions: Dict) -> str:
        """Enhance prompt with learned preferences"""
        try:
            enhanced_parts = [original_prompt]

            # Add high-confidence suggestions
            additions = []

            for suggestion in suggestions.get("add", []):
                if suggestion["confidence"] >= 0.7:
                    additions.append(suggestion["element"])
                    suggestions["applied"].append(suggestion)

            for suggestion in suggestions.get("improve", []):
                if suggestion["confidence"] >= 0.8:
                    additions.append(suggestion["element"])
                    suggestions["applied"].append(suggestion)

            # Apply quality settings from learning
            quality_settings = learned_prefs.quality_settings
            if quality_settings:
                # Add quality-related preferences with high confidence
                for setting, data in quality_settings.items():
                    if setting in ["quality_level", "style_preference"] and data.get("confidence", 0) >= 0.6:
                        preferred_value = data.get("preferred_value")
                        if preferred_value and preferred_value not in original_prompt.lower():
                            additions.append(preferred_value)

            # Combine enhancements
            if additions:
                enhanced_parts.extend(additions)

            enhanced_prompt = ", ".join(enhanced_parts)

            # Clean up the prompt
            enhanced_prompt = self._clean_prompt(enhanced_prompt)

            logger.info(f"Enhanced prompt with {len(suggestions.get('applied', []))} learned preferences")
            return enhanced_prompt

        except Exception as e:
            logger.error(f"Failed to enhance prompt with learning: {e}")
            return original_prompt

    def _clean_prompt(self, prompt: str) -> str:
        """Clean and optimize the enhanced prompt"""
        # Remove duplicates while preserving order
        parts = [part.strip() for part in prompt.split(",")]
        seen = set()
        cleaned_parts = []

        for part in parts:
            part_lower = part.lower()
            if part_lower not in seen and part.strip():
                seen.add(part_lower)
                cleaned_parts.append(part)

        # Limit total length to avoid token limits
        cleaned_prompt = ", ".join(cleaned_parts)
        if len(cleaned_prompt) > 1500:  # Reasonable token limit
            cleaned_prompt = ", ".join(cleaned_parts[:20])  # Take first 20 elements

        return cleaned_prompt

    async def record_generation_feedback(self, generation_id: str, prompt_used: str,
                                       style_elements: List[str], generation_result: Dict,
                                       user_feedback: Optional[Dict] = None) -> bool:
        """Record generation feedback for learning"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Build quality assessment
            quality_assessment = {
                "generation_successful": generation_result.get("success", False),
                "generation_time": generation_result.get("generation_time"),
                "settings_used": generation_result.get("settings", {}),
                "output_quality": generation_result.get("quality_score", 0.5)
            }

            # Extract context tags
            context_tags = []
            if generation_result.get("character_name"):
                context_tags.append(f"character:{generation_result['character_name']}")
            if generation_result.get("scene_type"):
                context_tags.append(generation_result["scene_type"])
            if generation_result.get("generation_type"):
                context_tags.append(generation_result["generation_type"])

            # Determine feedback weight
            feedback_weight = 1.0
            if user_feedback:
                rating = user_feedback.get("rating", 0)
                feedback_weight = rating / 5.0  # Normalize to 0-1

            # Insert or update learning record
            cursor.execute("""
                INSERT INTO anime_echo_style_learning
                (generation_id, user_id, prompt_used, style_elements, quality_assessment,
                 user_feedback, context_tags, feedback_weight)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (generation_id)
                DO UPDATE SET
                    user_feedback = EXCLUDED.user_feedback,
                    feedback_weight = EXCLUDED.feedback_weight
            """, (
                generation_id, "patrick", prompt_used,
                json.dumps(style_elements), json.dumps(quality_assessment),
                json.dumps(user_feedback or {}), json.dumps(context_tags),
                feedback_weight
            ))

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"📊 Recorded generation feedback: {generation_id} (weight: {feedback_weight:.2f})")
            return True

        except Exception as e:
            logger.error(f"Failed to record generation feedback: {e}")
            return False

    async def get_style_analytics(self, user_id: str = "patrick") -> Dict[str, Any]:
        """Get analytics on learned style preferences"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Get learning statistics
            cursor.execute("""
                SELECT COUNT(*) as total_generations,
                       AVG(CAST(user_feedback->>'rating' AS FLOAT)) as avg_rating,
                       COUNT(*) FILTER (WHERE user_feedback->>'rating' IS NOT NULL) as rated_generations,
                       COUNT(*) FILTER (WHERE CAST(user_feedback->>'rating' AS FLOAT) >= 4) as positive_ratings
                FROM anime_echo_style_learning
                WHERE user_id = %s
            """, (user_id,))

            stats = cursor.fetchone()
            total_generations, avg_rating, rated_generations, positive_ratings = stats

            # Get most successful style elements
            cursor.execute("""
                SELECT style_elements, COUNT(*) as usage_count,
                       AVG(CAST(user_feedback->>'rating' AS FLOAT)) as avg_rating
                FROM anime_echo_style_learning
                WHERE user_id = %s
                AND user_feedback->>'rating' IS NOT NULL
                GROUP BY style_elements
                HAVING COUNT(*) >= 2
                ORDER BY avg_rating DESC, usage_count DESC
                LIMIT 10
            """, (user_id,))

            top_elements = cursor.fetchall()

            # Get learning trends over time
            cursor.execute("""
                SELECT DATE(created_at) as generation_date,
                       COUNT(*) as daily_generations,
                       AVG(CAST(user_feedback->>'rating' AS FLOAT)) as daily_avg_rating
                FROM anime_echo_style_learning
                WHERE user_id = %s
                AND created_at >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY DATE(created_at)
                ORDER BY generation_date DESC
            """, (user_id,))

            trends = cursor.fetchall()

            cursor.close()
            conn.close()

            # Calculate learning confidence
            learning_confidence = positive_ratings / rated_generations if rated_generations > 0 else 0

            analytics = {
                "learning_statistics": {
                    "total_generations": total_generations or 0,
                    "rated_generations": rated_generations or 0,
                    "average_rating": round(avg_rating or 0, 2),
                    "positive_ratings": positive_ratings or 0,
                    "learning_confidence": round(learning_confidence, 2)
                },
                "top_style_elements": [
                    {
                        "elements": json.loads(row[0]) if row[0] else [],
                        "usage_count": row[1],
                        "average_rating": round(row[2] or 0, 2)
                    }
                    for row in top_elements
                ],
                "learning_trends": [
                    {
                        "date": row[0].isoformat() if row[0] else None,
                        "generations": row[1],
                        "average_rating": round(row[2] or 0, 2)
                    }
                    for row in trends
                ],
                "recommendations": []
            }

            # Generate recommendations
            if learning_confidence < 0.5:
                analytics["recommendations"].append("Provide more feedback to improve learning accuracy")

            if rated_generations < 10:
                analytics["recommendations"].append("Rate more generations to build preference model")

            if avg_rating and avg_rating < 3.5:
                analytics["recommendations"].append("Experiment with different style elements to find preferences")

            return analytics

        except Exception as e:
            logger.error(f"Failed to get style analytics: {e}")
            return {"error": str(e)}

# Global style learning engine
style_learning_engine = StyleLearningEngine()

# Convenience functions
async def analyze_and_enhance_prompt(prompt: str, user_id: str = "patrick") -> Dict[str, Any]:
    """Analyze prompt and enhance with learned preferences"""
    return await style_learning_engine.analyze_generation_request(prompt, user_id)

async def record_style_feedback(generation_id: str, prompt: str, style_elements: List[str],
                              result: Dict, rating: int, feedback: str = None) -> bool:
    """Record user feedback for style learning"""
    feedback_data = {"rating": rating, "feedback": feedback}
    return await style_learning_engine.record_generation_feedback(
        generation_id, prompt, style_elements, result, feedback_data
    )

async def get_user_style_analytics(user_id: str = "patrick") -> Dict[str, Any]:
    """Get user's style learning analytics"""
    return await style_learning_engine.get_style_analytics(user_id)