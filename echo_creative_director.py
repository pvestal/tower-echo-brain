#!/usr/bin/env python3
"""
Echo Creative Director Module - Proactive storytelling assistant
Echo asks questions, helps develop stories, improves characters
NOT technical - this is for directors who don't know technology
"""

import asyncio
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from echo_intention_translator import UserIntentionTranslator

class StoryPhase(Enum):
    """Where we are in story development"""
    INITIAL_IDEA = "initial_idea"
    CHARACTER_DEVELOPMENT = "character_development"
    WORLD_BUILDING = "world_building"
    PLOT_STRUCTURE = "plot_structure"
    SCENE_PLANNING = "scene_planning"
    VISUAL_STYLE = "visual_style"
    READY_TO_CREATE = "ready_to_create"

@dataclass
class StoryProject:
    """A story Echo is helping develop"""
    title: str = ""
    genre: str = ""
    characters: List[Dict] = field(default_factory=list)
    setting: str = ""
    plot_points: List[str] = field(default_factory=list)
    visual_style: str = ""
    mood: str = ""
    duration: int = 30  # seconds
    current_phase: StoryPhase = StoryPhase.INITIAL_IDEA
    conversation_history: List[Dict] = field(default_factory=list)

class EchoCreativeDirector:
    """Echo as a proactive creative assistant for video/anime creation"""

    def __init__(self):
        self.current_project = StoryProject()
        self.questions_asked = 0
        self.suggestions_made = 0

        # NEW: Initialize the intention translator
        self.translator = UserIntentionTranslator()

        # Echo's creative knowledge base
        self.story_patterns = {
            "anime": ["shounen journey", "magical girl", "cyberpunk", "slice of life", "mecha"],
            "drama": ["redemption arc", "love triangle", "coming of age", "family conflict"],
            "action": ["hero's journey", "revenge plot", "heist", "survival", "tournament arc"]
        }

        self.character_archetypes = [
            "reluctant hero", "wise mentor", "comic relief", "mysterious stranger",
            "loyal friend", "rival turned ally", "tragic villain", "innocent child"
        ]

        self.visual_moods = [
            "dark and gritty", "bright and colorful", "dreamlike", "noir",
            "neon cyberpunk", "soft pastels", "high contrast", "ethereal"
        ]

    async def process_user_input(self, user_input: str, video_mode: bool = True) -> str:
        """
        Main entry point - Echo processes input and PROACTIVELY helps develop the story
        """

        if not video_mode:
            return "Video mode is off. Enable it to work on your story together!"

        # NEW: Use the intention translator to extract story elements
        extracted_elements = await self.translator.process_input(user_input)

        # Add to conversation history
        self.current_project.conversation_history.append({
            "user": user_input,
            "phase": self.current_project.current_phase.value,
            "extracted": [{"type": e.element_type.value, "value": e.value} for e in extracted_elements]
        })

        # Generate intelligent response based on what was extracted
        response = await self._generate_intelligent_response(user_input, extracted_elements)

        # Get contextual questions from translator (these are smart!)
        follow_ups = await self.translator.generate_contextual_questions()

        # Offer suggestions based on current story state
        suggestions = await self._generate_smart_suggestions()

        # Combine into Echo's response
        echo_response = f"{response}\n\n"

        if follow_ups:
            echo_response += "🤔 Let me ask you:\n"
            for q in follow_ups:
                echo_response += f"  • {q}\n"
            echo_response += "\n"

        if suggestions:
            echo_response += "💡 What if we tried:\n"
            for s in suggestions:
                echo_response += f"  • {s}\n"
            echo_response += "\n"

        # Show current story summary using translator data
        story_summary = self.translator.get_story_summary()
        if story_summary['completeness_score'] > 0:
            echo_response += await self._show_story_progress(story_summary)

        return echo_response

    async def _generate_intelligent_response(self, user_input: str, extracted_elements) -> str:
        """Generate intelligent response based on what was actually extracted"""

        if not extracted_elements:
            return "Tell me more about your story idea! I'm here to help you develop it."

        # Acknowledge what we understood
        responses = []

        # Character elements
        character_elements = [e for e in extracted_elements if e.element_type.value.startswith('character')]
        if character_elements:
            char_names = [e.value for e in character_elements if e.element_type.value == 'character_name']
            char_traits = [e.value for e in character_elements if e.element_type.value == 'character_trait']
            char_roles = [e.value for e in character_elements if e.element_type.value == 'character_role']

            if char_names:
                responses.append(f"I love the name {char_names[0]}!")
            if char_traits:
                responses.append(f"A {char_traits[0]} character sounds compelling!")
            if char_roles:
                responses.append(f"A {char_roles[0]} protagonist has great story potential!")

        # Setting elements
        setting_elements = [e for e in extracted_elements if e.element_type.value.startswith('setting')]
        if setting_elements:
            locations = [e.value for e in setting_elements if e.element_type.value == 'setting_location']
            times = [e.value for e in setting_elements if e.element_type.value == 'setting_time']

            if locations:
                responses.append(f"{locations[0]} is a great setting!")
            if times:
                responses.append(f"The {times[0]} time period adds so much atmosphere!")

        # Plot elements
        plot_elements = [e for e in extracted_elements if e.element_type.value.startswith('plot')]
        if plot_elements:
            conflicts = [e.value for e in plot_elements if e.element_type.value == 'plot_conflict']
            goals = [e.value for e in plot_elements if e.element_type.value == 'plot_goal']

            if conflicts:
                responses.append(f"Fighting {conflicts[0]} creates immediate tension!")
            if goals:
                responses.append(f"The goal to {goals[0]} gives the story direction!")

        # Genre/mood elements
        genre_elements = [e for e in extracted_elements if e.element_type.value in ['genre', 'mood_tone']]
        if genre_elements:
            for element in genre_elements:
                if element.element_type.value == 'genre':
                    responses.append(f"{element.value.title()} is such a rich genre to explore!")
                elif element.element_type.value == 'mood_tone':
                    responses.append(f"A {element.value} tone will really draw viewers in!")

        if responses:
            return " ".join(responses)
        else:
            return "I'm getting some great ideas from what you've told me!"

    async def _generate_smart_suggestions(self) -> List[str]:
        """Generate suggestions based on current story state"""
        suggestions = []
        story_state = self.translator.get_story_summary()

        # Character suggestions
        if story_state['characters']:
            char = story_state['characters'][0]
            if char['name'] and not char['role']:
                suggestions.append(f"What if {char['name']} is a warrior, mage, or detective?")

            if char['well_defined'] and len(story_state['characters']) == 1:
                suggestions.append(f"Add a mentor or rival for {char['name']} to interact with")

        # Setting suggestions
        if story_state['setting']['location'] and not story_state['setting']['time_period']:
            suggestions.append("Consider when this takes place - future, past, or alternate reality?")

        # Plot suggestions
        if story_state['plot']['main_conflict'] and not story_state['plot']['character_goal']:
            suggestions.append("What does your character hope to achieve by overcoming this challenge?")

        # Visual suggestions
        if story_state['visual_style']['genre'] and not story_state['visual_style']['mood']:
            genre = story_state['visual_style']['genre']
            if genre == 'anime':
                suggestions.append("Try a bright, colorful mood or dark, mysterious atmosphere")
            elif genre == 'cyberpunk':
                suggestions.append("A neon-lit, high-tech visual style would be perfect")

        return suggestions[:2]

    async def _show_story_progress(self, story_summary: Dict) -> str:
        """Show the current story development progress"""
        progress = "📖 **Your Story So Far:**\n"

        # Characters
        if story_summary['characters']:
            for char in story_summary['characters'][:2]:  # Show up to 2 characters
                name = char['name'] if char['name'] else "Your protagonist"
                details = []
                if char['age']:
                    details.append(f"age {char['age']}")
                if char['role']:
                    details.append(char['role'])
                if char['traits']:
                    details.append(', '.join(char['traits'][:2]))

                char_line = f"  • {name}"
                if details:
                    char_line += f" ({', '.join(details)})"
                progress += char_line + "\n"

        # Setting
        setting = story_summary['setting']
        if setting['location'] or setting['time_period']:
            setting_parts = []
            if setting['location']:
                setting_parts.append(setting['location'])
            if setting['time_period']:
                setting_parts.append(setting['time_period'])
            if setting['atmosphere']:
                setting_parts.append(f"{setting['atmosphere']} atmosphere")

            if setting_parts:
                progress += f"  • Setting: {', '.join(setting_parts)}\n"

        # Plot
        plot = story_summary['plot']
        if plot['main_conflict']:
            progress += f"  • Conflict: {plot['main_conflict']}\n"
        if plot['character_goal']:
            progress += f"  • Goal: {plot['character_goal']}\n"

        # Visual style
        visual = story_summary['visual_style']
        if visual['genre'] or visual['mood']:
            visual_parts = []
            if visual['genre']:
                visual_parts.append(visual['genre'])
            if visual['mood']:
                visual_parts.append(f"{visual['mood']} mood")

            if visual_parts:
                progress += f"  • Style: {', '.join(visual_parts)}\n"

        # Progress bar
        score = story_summary['completeness_score']
        progress += f"\n📊 Story Development: {score:.0f}% complete"

        if score >= 80:
            progress += "\n✨ **Ready to create!** Your story has enough detail for video generation!"
        elif score >= 60:
            progress += "\n🎯 **Almost there!** Just a few more details needed."
        elif score >= 40:
            progress += "\n📝 **Good progress!** Keep building your story."

        return progress

    async def _generate_follow_up_questions(self) -> List[str]:
        """Generate proactive questions based on what's missing in the story"""

        questions = []
        phase = self.current_project.current_phase

        if phase == StoryPhase.CHARACTER_DEVELOPMENT:
            if not self.current_project.characters:
                questions.extend([
                    "What's your character's name and age?",
                    "What drives them - what do they want more than anything?",
                    "What's their biggest fear or weakness?"
                ])
            else:
                questions.extend([
                    "What's their backstory - what shaped them?",
                    "Do they have any special abilities or skills?",
                    "Who are their allies and enemies?"
                ])

        elif phase == StoryPhase.WORLD_BUILDING:
            questions.extend([
                "Is this set in the future, past, or alternate reality?",
                "What makes this world unique - what are its rules?",
                "What's the atmosphere - hopeful, dark, mysterious?"
            ])

        elif phase == StoryPhase.PLOT_STRUCTURE:
            questions.extend([
                "What's the main conflict your character faces?",
                "How does the story begin - what disrupts their normal life?",
                "What's at stake if they fail?"
            ])

        elif phase == StoryPhase.VISUAL_STYLE:
            questions.extend([
                "Do you prefer bright colors or darker tones?",
                "Fast action cuts or slow contemplative scenes?",
                "Realistic or more stylized/artistic?"
            ])

        # Limit questions to avoid overwhelming
        self.questions_asked += len(questions[:3])
        return questions[:3]

    async def _generate_suggestions(self) -> List[str]:
        """Proactively suggest improvements and options"""

        suggestions = []

        if self.current_project.genre:
            # Suggest story patterns
            patterns = self.story_patterns.get(self.current_project.genre, [])
            if patterns and self.suggestions_made < 3:
                suggestions.append(f"A {patterns[0]} storyline could work great here")

        if self.current_project.characters:
            # Suggest character improvements
            if len(self.current_project.characters) == 1:
                suggestions.append("Add a rival or mentor character for more dynamic interactions")

            # Suggest character development
            suggestions.append("Give your character a personal item or symbol that represents their journey")

        if not self.current_project.visual_style:
            # Suggest visual styles
            mood = self.visual_moods[self.suggestions_made % len(self.visual_moods)]
            suggestions.append(f"Try a {mood} visual style")

        if self.current_project.plot_points:
            # Suggest plot improvements
            suggestions.append("Add a plot twist in the middle to surprise the audience")
            suggestions.append("Consider a bittersweet ending for emotional impact")

        self.suggestions_made += len(suggestions)
        return suggestions[:2]  # Don't overwhelm with suggestions

    async def _summarize_current_story(self) -> str:
        """Show the user what we've developed so far"""

        summary = "📖 **Your Story So Far:**\n"

        if self.current_project.title:
            summary += f"  Title: {self.current_project.title}\n"

        if self.current_project.genre:
            summary += f"  Genre: {self.current_project.genre}\n"

        if self.current_project.characters:
            summary += f"  Characters: {len(self.current_project.characters)} developed\n"
            for char in self.current_project.characters[:2]:  # Show first 2
                summary += f"    • {char.get('name', 'Unnamed')}: {char.get('role', 'protagonist')}\n"

        if self.current_project.setting:
            summary += f"  Setting: {self.current_project.setting}\n"

        if self.current_project.visual_style:
            summary += f"  Visual Style: {self.current_project.visual_style}\n"

        # Show progress
        phases_complete = 0
        if self.current_project.title: phases_complete += 1
        if self.current_project.characters: phases_complete += 1
        if self.current_project.setting: phases_complete += 1
        if self.current_project.plot_points: phases_complete += 1
        if self.current_project.visual_style: phases_complete += 1

        progress = (phases_complete / 5) * 100
        summary += f"\n📊 Story Development: {progress:.0f}% complete"

        if progress >= 80:
            summary += "\n✨ **We're ready to create your video!** Just say 'let's make it!'"

        return summary

    async def generate_video_from_story(self) -> Dict:
        """
        Convert the developed story into video generation parameters
        Uses the intelligent translator to build rich prompts
        """

        story_summary = self.translator.get_story_summary()

        # Check if we have enough for video generation
        if story_summary['completeness_score'] < 40:
            return {
                "error": "We need more story details first! Tell me about your character and setting.",
                "completeness": story_summary['completeness_score']
            }

        # Use the translator to generate the prompt
        video_prompt = await self.translator.generate_video_prompt()

        if not video_prompt:
            return {"error": "Not enough story elements to create a video yet."}

        # Determine style and mood from what we've learned
        style = "anime"
        mood = "dynamic"

        if story_summary['visual_style']['genre']:
            style = story_summary['visual_style']['genre']

        if story_summary['visual_style']['mood']:
            mood = story_summary['visual_style']['mood']

        return {
            "prompt": video_prompt,
            "duration": self.current_project.duration,
            "style": style,
            "mood": mood,
            "ready": True,
            "completeness": story_summary['completeness_score'],
            "message": f"🎬 Ready to create! Your story is {story_summary['completeness_score']:.0f}% developed.",
            "story_summary": story_summary
        }


# Example of Echo being intelligent and responsive
async def demo_intelligent_echo():
    """Show how Echo now intelligently builds stories from natural language"""

    echo = EchoCreativeDirector()

    print("=== DEMO: Intelligent Echo Creative Director ===\n")

    # Conversation that builds progressively
    conversations = [
        "I want to make an anime about a cyberpunk samurai",
        "Her name is Luna and she's 16 years old",
        "She's brave but mysterious, living in modern Tokyo",
        "She discovers she has magic powers and must fight against dark spirits",
        "The mood should be mysterious but hopeful"
    ]

    for user_input in conversations:
        print(f"👤 User: {user_input}")
        response = await echo.process_user_input(user_input)
        print(f"🤖 Echo: {response}")
        print("-" * 50)

    # Show final video generation
    print("\n🎬 Final Story Assessment:")
    video_params = await echo.generate_video_from_story()
    print(json.dumps(video_params, indent=2))

    print("\n=== Story Elements Extracted ===")
    story_summary = echo.translator.get_story_summary()
    for category, data in story_summary.items():
        if category != 'completeness_score':
            print(f"{category}: {data}")

if __name__ == "__main__":
    asyncio.run(demo_intelligent_echo())