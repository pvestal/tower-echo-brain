#!/usr/bin/env python3
"""
Echo User Intention Translator Module

Parses natural language input to extract story elements and builds stories progressively.
Remembers everything said and generates intelligent contextual questions.
"""

import re
import json
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio

class ElementType(Enum):
    """Types of story elements we can extract"""
    CHARACTER_NAME = "character_name"
    CHARACTER_AGE = "character_age"
    CHARACTER_TRAIT = "character_trait"
    CHARACTER_ROLE = "character_role"
    CHARACTER_APPEARANCE = "character_appearance"
    SETTING_LOCATION = "setting_location"
    SETTING_TIME = "setting_time"
    SETTING_ATMOSPHERE = "setting_atmosphere"
    PLOT_CONFLICT = "plot_conflict"
    PLOT_GOAL = "plot_goal"
    PLOT_OBSTACLE = "plot_obstacle"
    PLOT_TWIST = "plot_twist"
    MOOD_TONE = "mood_tone"
    VISUAL_STYLE = "visual_style"
    GENRE = "genre"
    THEME = "theme"
    RELATIONSHIP = "relationship"

@dataclass
class StoryElement:
    """A single extracted story element"""
    element_type: ElementType
    value: str
    confidence: float
    source_text: str
    context: str = ""

@dataclass
class Character:
    """Character information"""
    name: str = ""
    age: Optional[int] = None
    traits: List[str] = field(default_factory=list)
    role: str = ""
    appearance: str = ""
    relationships: Dict[str, str] = field(default_factory=dict)
    backstory: str = ""
    goals: List[str] = field(default_factory=list)

    def is_well_defined(self) -> bool:
        """Check if character has enough detail"""
        return bool(self.name and (self.traits or self.role or self.appearance))

@dataclass
class Setting:
    """Setting/world information"""
    location: str = ""
    time_period: str = ""
    atmosphere: str = ""
    description: str = ""
    rules: List[str] = field(default_factory=list)

    def is_well_defined(self) -> bool:
        """Check if setting has enough detail"""
        return bool(self.location and (self.time_period or self.atmosphere))

@dataclass
class Plot:
    """Plot structure information"""
    main_conflict: str = ""
    character_goal: str = ""
    obstacles: List[str] = field(default_factory=list)
    twists: List[str] = field(default_factory=list)
    stakes: str = ""
    theme: str = ""

    def is_well_defined(self) -> bool:
        """Check if plot has enough structure"""
        return bool(self.main_conflict and self.character_goal)

@dataclass
class VisualStyle:
    """Visual and mood information"""
    mood: str = ""
    color_palette: str = ""
    art_style: str = ""
    pacing: str = ""
    genre: str = ""

    def is_well_defined(self) -> bool:
        """Check if visual style is defined"""
        return bool(self.mood or self.art_style or self.genre)

class UserIntentionTranslator:
    """Translates natural language into story elements"""

    def __init__(self):
        self.characters: List[Character] = []
        self.setting = Setting()
        self.plot = Plot()
        self.visual_style = VisualStyle()
        self.conversation_memory: List[str] = []
        self.extracted_elements: List[StoryElement] = []
        self.questions_asked: Set[str] = set()

        # Patterns for extracting information
        self.patterns = {
            # Character patterns
            ElementType.CHARACTER_NAME: [
                r"(?:her|his|their|character)\s+name\s+is\s+(\w+)",
                r"(?:named|called)\s+(\w+)",
                r"(\w+)\s+is\s+(?:the|a|an)\s+(?:main\s+)?(?:character|protagonist|hero|heroine)",
                r"meet\s+(\w+)",
                r"about\s+(\w+)\s+who"
            ],
            ElementType.CHARACTER_AGE: [
                r"(?:she's|he's|they're)\s+(\d+)(?:\s+years?\s+old)?",
                r"(\d+)(?:\s+year\s+old|\s+years\s+old)",
                r"age\s+(?:of\s+)?(\d+)"
            ],
            ElementType.CHARACTER_TRAIT: [
                r"(?:she's|he's|they're)\s+(brave|smart|shy|confident|mysterious|powerful|weak|strong|kind|cruel|wise|foolish)",
                r"(?:a|an)\s+(brave|smart|shy|confident|mysterious|powerful|weak|strong|kind|cruel|wise|foolish)\s+(?:person|character|warrior|girl|boy)",
                r"has\s+(?:a\s+)?(brave|smart|shy|confident|mysterious|powerful|weak|strong|kind|cruel|wise|foolish)\s+(?:personality|nature)"
            ],
            ElementType.CHARACTER_ROLE: [
                r"(?:a|an|the)\s+(warrior|mage|samurai|ninja|princess|prince|student|teacher|detective|thief|assassin|healer|leader)",
                r"(?:she's|he's|they're)\s+(?:a|an|the)\s+(warrior|mage|samurai|ninja|princess|prince|student|teacher|detective|thief|assassin|healer|leader)",
                r"works?\s+as\s+(?:a|an|the)\s+(warrior|mage|samurai|ninja|princess|prince|student|teacher|detective|thief|assassin|healer|leader)"
            ],

            # Setting patterns
            ElementType.SETTING_LOCATION: [
                r"(?:in|set\s+in|takes\s+place\s+in)\s+([\w\s]+?)(?:\s+(?:where|and|but|however)|\.|$)",
                r"lives?\s+in\s+([\w\s]+?)(?:\s+(?:where|and|but|however)|\.|$)",
                r"world\s+of\s+([\w\s]+?)(?:\s+(?:where|and|but|however)|\.|$)"
            ],
            ElementType.SETTING_TIME: [
                r"(?:in\s+the\s+)?(future|past|medieval|modern|ancient|renaissance|victorian|cyberpunk)",
                r"(?:year\s+)?(\d{4})",
                r"(?:during\s+the\s+)?(war|peace|renaissance|industrial\s+age)"
            ],
            ElementType.SETTING_ATMOSPHERE: [
                r"(?:it's|atmosphere\s+is|mood\s+is)\s+(dark|bright|mysterious|hopeful|grim|cheerful|ominous|peaceful)",
                r"(?:a|an)\s+(dark|bright|mysterious|hopeful|grim|cheerful|ominous|peaceful)\s+(?:world|place|setting)"
            ],

            # Plot patterns
            ElementType.PLOT_CONFLICT: [
                r"(?:fights?|fighting|battles?|battling)\s+(?:against\s+)?([\w\s]+?)(?:\s+(?:who|and|but|however)|\.|$)",
                r"conflict\s+(?:with|against)\s+([\w\s]+?)(?:\s+(?:who|and|but|however)|\.|$)",
                r"(?:must\s+)?(?:stop|defeat|overcome)\s+([\w\s]+?)(?:\s+(?:who|and|but|however)|\.|$)"
            ],
            ElementType.PLOT_GOAL: [
                r"(?:wants?\s+to|trying\s+to|seeks?\s+to|aims?\s+to)\s+([\w\s]+?)(?:\s+(?:but|and|however)|\.|$)",
                r"(?:goal\s+is\s+to|mission\s+is\s+to)\s+([\w\s]+?)(?:\s+(?:but|and|however)|\.|$)",
                r"(?:must|needs?\s+to)\s+([\w\s]+?)(?:\s+(?:but|and|however)|\.|$)"
            ],

            # Visual/mood patterns
            ElementType.MOOD_TONE: [
                r"(?:tone\s+is|mood\s+is|feels?)\s+(mysterious|hopeful|dark|bright|serious|comedic|dramatic|action-packed)",
                r"(?:a|an)\s+(mysterious|hopeful|dark|bright|serious|comedic|dramatic|action-packed)\s+(?:story|tone|mood)",
                r"mood\s+should\s+be\s+([\w\s]+?)(?:\s+(?:but|and|however)|\.|$)",
                r"should\s+be\s+(mysterious\s+but\s+hopeful|dark\s+but\s+bright|[\w\s]+?)(?:\.|$)"
            ],
            ElementType.GENRE: [
                r"(?:want\s+to\s+make\s+(?:an?\s+)?|it's\s+(?:an?\s+)?)(anime|manga|cyberpunk|fantasy|sci-fi|romance|horror|thriller|comedy|drama)",
                r"(?:genre\s+is|style\s+is)\s+(anime|manga|cyberpunk|fantasy|sci-fi|romance|horror|thriller|comedy|drama)"
            ]
        }

    async def process_input(self, user_input: str) -> List[StoryElement]:
        """Process user input and extract story elements"""
        user_input = user_input.lower().strip()
        self.conversation_memory.append(user_input)

        extracted = []

        # Try each pattern type
        for element_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, user_input, re.IGNORECASE)
                for match in matches:
                    value = match.group(1).strip()
                    if value and len(value) > 1:  # Valid extraction
                        element = StoryElement(
                            element_type=element_type,
                            value=value,
                            confidence=0.8,  # Pattern-based extraction is pretty reliable
                            source_text=match.group(0),
                            context=user_input
                        )
                        extracted.append(element)

        # Store and integrate elements
        for element in extracted:
            await self._integrate_element(element)
            self.extracted_elements.append(element)

        return extracted

    async def _integrate_element(self, element: StoryElement):
        """Integrate extracted element into story structure"""

        if element.element_type == ElementType.CHARACTER_NAME:
            char = self._get_or_create_character(element.value)
            char.name = element.value

        elif element.element_type == ElementType.CHARACTER_AGE:
            char = self._get_main_character()
            if char:
                try:
                    char.age = int(element.value)
                except ValueError:
                    pass

        elif element.element_type == ElementType.CHARACTER_TRAIT:
            char = self._get_main_character()
            if char and element.value not in char.traits:
                char.traits.append(element.value)

        elif element.element_type == ElementType.CHARACTER_ROLE:
            char = self._get_main_character()
            if char:
                char.role = element.value

        elif element.element_type == ElementType.SETTING_LOCATION:
            self.setting.location = element.value

        elif element.element_type == ElementType.SETTING_TIME:
            self.setting.time_period = element.value

        elif element.element_type == ElementType.SETTING_ATMOSPHERE:
            self.setting.atmosphere = element.value

        elif element.element_type == ElementType.PLOT_CONFLICT:
            self.plot.main_conflict = element.value

        elif element.element_type == ElementType.PLOT_GOAL:
            self.plot.character_goal = element.value

        elif element.element_type == ElementType.MOOD_TONE:
            self.visual_style.mood = element.value

        elif element.element_type == ElementType.GENRE:
            self.visual_style.genre = element.value

    def _get_or_create_character(self, name: str) -> Character:
        """Get existing character or create new one"""
        for char in self.characters:
            if char.name.lower() == name.lower():
                return char

        new_char = Character(name=name)
        self.characters.append(new_char)
        return new_char

    def _get_main_character(self) -> Optional[Character]:
        """Get the main character (first one or most detailed)"""
        if not self.characters:
            # Create unnamed main character
            main_char = Character()
            self.characters.append(main_char)
            return main_char
        return self.characters[0]

    async def generate_contextual_questions(self) -> List[str]:
        """Generate intelligent questions based on what we know and don't know"""
        questions = []

        # Character questions
        main_char = self._get_main_character() if self.characters else None

        if main_char:
            if not main_char.name and "what's your character's name" not in self.questions_asked:
                questions.append("What's your character's name?")
                self.questions_asked.add("what's your character's name")

            elif main_char.name and not main_char.age and "how old is" not in str(self.questions_asked):
                questions.append(f"How old is {main_char.name}?")
                self.questions_asked.add(f"how old is {main_char.name}")

            elif main_char.name and not main_char.traits and "what is personality" not in str(self.questions_asked):
                questions.append(f"What is {main_char.name}'s personality like?")
                self.questions_asked.add(f"what is {main_char.name} personality")

            elif main_char.name and not main_char.role and "what does" not in str(self.questions_asked):
                questions.append(f"What does {main_char.name} do? Are they a warrior, student, mage?")
                self.questions_asked.add(f"what does {main_char.name} do")

        # Setting questions
        if not self.setting.location and "where does story take place" not in self.questions_asked:
            questions.append("Where does your story take place?")
            self.questions_asked.add("where does story take place")

        elif self.setting.location and not self.setting.time_period and "when is this set" not in self.questions_asked:
            questions.append("When is this set? Modern day, future, medieval times?")
            self.questions_asked.add("when is this set")

        # Plot questions
        if main_char and main_char.name and not self.plot.main_conflict and "what challenge" not in str(self.questions_asked):
            questions.append(f"What challenge or enemy does {main_char.name} face?")
            self.questions_asked.add(f"what challenge does {main_char.name} face")

        elif self.plot.main_conflict and not self.plot.character_goal and "what does character want" not in self.questions_asked:
            questions.append("What does your character want to achieve?")
            self.questions_asked.add("what does character want")

        # Visual style questions
        if self.visual_style.genre and not self.visual_style.mood and "what mood" not in self.questions_asked:
            questions.append("What mood should this have? Dark and serious, or bright and hopeful?")
            self.questions_asked.add("what mood")

        # Advanced questions only if we have basics
        if main_char and main_char.is_well_defined() and self.setting.is_well_defined():
            if len(self.characters) == 1 and "supporting characters" not in self.questions_asked:
                questions.append(f"Does {main_char.name} have any friends, mentors, or rivals?")
                self.questions_asked.add("supporting characters")

        return questions[:3]  # Limit to avoid overwhelming

    def get_story_summary(self) -> Dict:
        """Get current story state"""
        return {
            "characters": [
                {
                    "name": char.name or "Unnamed protagonist",
                    "age": char.age,
                    "traits": char.traits,
                    "role": char.role,
                    "appearance": char.appearance,
                    "well_defined": char.is_well_defined()
                }
                for char in self.characters
            ] if self.characters else [],
            "setting": {
                "location": self.setting.location,
                "time_period": self.setting.time_period,
                "atmosphere": self.setting.atmosphere,
                "well_defined": self.setting.is_well_defined()
            },
            "plot": {
                "main_conflict": self.plot.main_conflict,
                "character_goal": self.plot.character_goal,
                "obstacles": self.plot.obstacles,
                "well_defined": self.plot.is_well_defined()
            },
            "visual_style": {
                "mood": self.visual_style.mood,
                "genre": self.visual_style.genre,
                "art_style": self.visual_style.art_style,
                "well_defined": self.visual_style.is_well_defined()
            },
            "completeness_score": self._calculate_completeness()
        }

    def _calculate_completeness(self) -> float:
        """Calculate how complete the story is (0-100)"""
        score = 0
        total = 4

        # Character completeness (25%)
        if self.characters and self.characters[0].is_well_defined():
            score += 25

        # Setting completeness (25%)
        if self.setting.is_well_defined():
            score += 25

        # Plot completeness (25%)
        if self.plot.is_well_defined():
            score += 25

        # Visual style completeness (25%)
        if self.visual_style.is_well_defined():
            score += 25

        return score

    async def generate_video_prompt(self) -> str:
        """Generate a rich prompt for video generation"""
        prompt_parts = []

        # Genre first
        if self.visual_style.genre:
            prompt_parts.append(self.visual_style.genre)

        # Character
        if self.characters and self.characters[0].name:
            char = self.characters[0]
            char_desc = char.name
            if char.role:
                char_desc += f" the {char.role}"
            if char.age:
                char_desc += f", age {char.age}"
            if char.traits:
                char_desc += f", {', '.join(char.traits[:2])}"
            prompt_parts.append(char_desc)

        # Setting
        if self.setting.location:
            setting_desc = f"in {self.setting.location}"
            if self.setting.time_period:
                setting_desc += f" during {self.setting.time_period}"
            prompt_parts.append(setting_desc)

        # Action/conflict
        if self.plot.main_conflict:
            prompt_parts.append(f"facing {self.plot.main_conflict}")
        elif self.plot.character_goal:
            prompt_parts.append(f"trying to {self.plot.character_goal}")

        # Mood
        if self.visual_style.mood:
            prompt_parts.append(f"{self.visual_style.mood} atmosphere")

        return ", ".join(prompt_parts)

# Example usage and testing
async def test_translator():
    """Test the intention translator"""
    translator = UserIntentionTranslator()

    # Simulate conversation
    inputs = [
        "I want to make an anime about a cyberpunk samurai",
        "Her name is Luna and she's 16 years old",
        "She's brave but mysterious, living in modern Tokyo",
        "She discovers she has magic powers and must fight against dark spirits",
        "The mood should be mysterious but hopeful"
    ]

    for user_input in inputs:
        print(f"\nUser: {user_input}")

        # Extract elements
        elements = await translator.process_input(user_input)
        print(f"Extracted: {[(e.element_type.value, e.value) for e in elements]}")

        # Generate questions
        questions = await translator.generate_contextual_questions()
        if questions:
            print(f"Questions: {questions}")

        # Show summary
        summary = translator.get_story_summary()
        print(f"Completeness: {summary['completeness_score']}%")

    # Final prompt
    prompt = await translator.generate_video_prompt()
    print(f"\nFinal video prompt: {prompt}")

if __name__ == "__main__":
    asyncio.run(test_translator())