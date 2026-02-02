"""Memory module for conversation context and entity management"""

# Import modules as they become available
try:
    from src.memory.context_retrieval import ConversationContextRetriever
except ImportError:
    ConversationContextRetriever = None

try:
    from src.memory.pronoun_resolver import PronounResolver
except ImportError:
    PronounResolver = None

try:
    from src.memory.entity_extractor import EntityExtractor
except ImportError:
    EntityExtractor = None

__all__ = ['ConversationContextRetriever', 'PronounResolver', 'EntityExtractor']