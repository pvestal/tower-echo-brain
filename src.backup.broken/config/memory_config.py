#!/usr/bin/env python3
"""
Configuration for Echo Brain's conversation memory system.

This configuration controls how Echo remembers and retrieves conversations,
enabling true long-term memory and context awareness.
"""

# ============================================
# CONVERSATION MEMORY SETTINGS
# ============================================

# Maximum number of historical messages to retrieve per query
# Higher values provide more context but increase processing time
MAX_HISTORY_MESSAGES = 50  # Increased from 10 for better context

# Maximum age of conversations to consider (in days)
# Set to None for indefinite retention (recommended for production)
# Set to a number to limit memory to recent conversations only
MAX_HISTORY_AGE_DAYS = None  # None = remember everything forever

# Number of recent entity mentions to track for pronoun resolution
# Used when resolving "it", "that", "them" to actual entities
MAX_ENTITY_LOOKBACK = 10  # How many previous messages to check for entities

# ============================================
# MEMORY PERSISTENCE SETTINGS
# ============================================

# How often to clean up old conversations (if MAX_HISTORY_AGE_DAYS is set)
CLEANUP_INTERVAL_HOURS = 24  # Run cleanup once per day

# Minimum conversations to keep per user regardless of age
MIN_CONVERSATIONS_PER_USER = 100  # Always keep at least this many

# ============================================
# PERFORMANCE TUNING
# ============================================

# Database connection pool settings
DB_POOL_MIN_SIZE = 2
DB_POOL_MAX_SIZE = 10

# Cache settings for frequently accessed conversations
CACHE_TTL_SECONDS = 300  # 5 minutes
CACHE_MAX_ENTRIES = 1000

# ============================================
# LEARNING & PATTERN RECOGNITION
# ============================================

# Enable automatic pattern learning from conversations
ENABLE_PATTERN_LEARNING = True

# Minimum conversation length to consider for pattern extraction
MIN_CONVERSATION_LENGTH_FOR_LEARNING = 3  # At least 3 exchanges

# How often to run pattern extraction (in hours)
PATTERN_EXTRACTION_INTERVAL_HOURS = 6

# ============================================
# PRIVACY & COMPLIANCE
# ============================================

# Conversations to exclude from long-term memory (regex patterns)
EXCLUDED_CONVERSATION_PATTERNS = [
    r"^temp_.*",       # Temporary conversations
    r"^test_.*",       # Test conversations
    r"^debug_.*",      # Debug conversations
]

# Sensitive data patterns to redact before storage
REDACTION_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN pattern
    r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # Credit card pattern
]

# ============================================
# CONVERSATION CONTEXT ENRICHMENT
# ============================================

# Enable Tower service status injection into context
INJECT_SERVICE_STATUS = True

# Enable time-aware context (morning/evening/weekend awareness)
ENABLE_TEMPORAL_CONTEXT = True

# Enable user preference learning
LEARN_USER_PREFERENCES = True

# ============================================
# DEBUGGING & MONITORING
# ============================================

# Enable detailed memory operation logging
DEBUG_MEMORY_OPS = False

# Log slow queries (threshold in seconds)
SLOW_QUERY_THRESHOLD = 1.0

# Enable memory usage statistics collection
COLLECT_MEMORY_STATS = True