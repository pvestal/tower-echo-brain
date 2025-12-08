"""Memory module for conversation context and entity management"""

# Import modules as they become available
try:
    from .context_retrieval import ConversationContextRetriever
except ImportError:
    ConversationContextRetriever = None

try:
    from .pronoun_resolver import PronounResolver
except ImportError:
    PronounResolver = None

try:
    from .entity_extractor import EntityExtractor
except ImportError:
    EntityExtractor = None

__all__ = ['ConversationContextRetriever', 'PronounResolver', 'EntityExtractor']