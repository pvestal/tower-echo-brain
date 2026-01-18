"""
UNIFIED MODEL ROUTER - REPLACES ALL 46 CONFIGS
SINGLE SOURCE OF TRUTH
"""
import logging
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class ModelTier(Enum):
    REASONING = "reasoning"
    CODING = "coding"
    CREATIVE = "creative"
    FAST = "fast"

@dataclass
class ModelSelection:
    model_name: str
    tier: str
    reason: str

class UnifiedModelRouter:
    def __init__(self):
        # SIMPLE RULES - NO DATABASE, NO COMPLEXITY
        self.rules = {
            ModelTier.REASONING: "deepseek-r1:8b",
            ModelTier.CODING: "deepseek-coder-v2:16b",
            ModelTier.CREATIVE: "llama3.2:3b",
            ModelTier.FAST: "qwen2.5:3b"
        }
        logger.info("🚀 Unified Router: Created - Replacing 46 configs")

    def select_model(self, query: str) -> ModelSelection:
        """ONE FUNCTION TO RULE THEM ALL"""
        query_lower = query.lower()

        # CHECK FOR REASONING NEEDS FIRST (HIGHEST PRIORITY)
        try:
            import sys
            if '/opt/tower-echo-brain/src' not in sys.path:
                sys.path.insert(0, '/opt/tower-echo-brain/src')
            from reasoning.deepseek_reasoner import should_use_reasoning
            if should_use_reasoning(query):
                return ModelSelection(
                    model_name=self.rules[ModelTier.REASONING],
                    tier="reasoning",
                    reason="Deep reasoning required (DeepSeek with <think> tags)"
                )
        except ImportError:
            # Fallback if reasoner not available
            pass

        # CODING QUERIES
        coding_words = ['code', 'function', 'debug', 'refactor', 'implement', 'write.*code', 'generate.*code', 'python', 'javascript']
        if any(word in query_lower for word in coding_words):
            return ModelSelection(
                model_name="qwen2.5:3b",  # Based on available models
                tier="coding",
                reason="Contains coding keywords"
            )

        # ANIME/CREATIVE QUERIES
        anime_words = ['anime', 'goblin', 'cyber', 'character', 'tokyo', 'scene', 'story', 'creative']
        if any(word in query_lower for word in anime_words):
            return ModelSelection(
                model_name=self.rules[ModelTier.CREATIVE],
                tier="creative",
                reason="Contains anime/creative keywords"
            )

        # REASONING QUERIES (BACKUP CHECK)
        reasoning_words = ['analyze', 'plan', 'design', 'architecture', 'strategy', 'consolidate', 'how.*work', 'why.*work']
        if any(word in query_lower for word in reasoning_words) or len(query) > 100:
            return ModelSelection(
                model_name=self.rules[ModelTier.REASONING],
                tier="reasoning",
                reason="Complex analysis query"
            )

        # DEFAULT: FAST
        return ModelSelection(
            model_name=self.rules[ModelTier.FAST],
            tier="fast",
            reason="Simple query - default fast model"
        )

# SINGLE INSTANCE - SINGLE SOURCE OF TRUTH
unified_router = UnifiedModelRouter()

def get_model_for_query(query: str) -> str:
    """Simple interface - returns just model name"""
    return unified_router.select_model(query).model_name