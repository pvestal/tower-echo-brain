"""
Provider State Handlers for Echo Brain Contract Tests

Each "given" clause in the consumer tests maps to a handler here.
These set up the database/service state needed for the provider
to satisfy each interaction.

Think of these like test fixtures — they ensure the backend is in
the right state before each contract interaction is replayed.
"""

from typing import Callable, Dict, Any
from unittest.mock import AsyncMock, MagicMock
import json


class ProviderStateManager:
    """
    Manages provider states for contract verification.
    
    Maps consumer-defined state descriptions to setup functions
    that prepare the backend to satisfy those expectations.
    """

    def __init__(self):
        self._handlers: Dict[str, Callable] = {}
        self._register_states()

    def _register_states(self):
        """Register all known provider states."""
        self._handlers = {
            'the system is running': self._state_system_running,
            'the vector store is unreachable': self._state_vector_store_down,
            'the vector store has indexed documents': self._state_has_documents,
            'the vector store has no matching documents': self._state_no_matches,
            'memories exist in the database': self._state_has_memories,
            'the system can accept new memories': self._state_can_create,
            'an ingestion has completed previously': self._state_ingestion_complete,
            'no ingestion has ever run': self._state_never_ingested,
        }

    def handle(self, state: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Set up the provider state for a given interaction.
        
        Returns mock/fixture data that test routes can reference.
        """
        handler = self._handlers.get(state)
        if not handler:
            raise ValueError(
                f"Unknown provider state: '{state}'\n"
                f"Known states: {list(self._handlers.keys())}\n"
                f"Add a handler in provider_states.py"
            )
        return handler(params or {})

    # ─── State Handlers ──────────────────────────────────────────

    def _state_system_running(self, params: dict) -> dict:
        """Baseline: everything healthy, no special setup."""
        return {
            'db_status': 'up',
            'vector_status': 'up',
            'ollama_status': 'up',
        }

    def _state_vector_store_down(self, params: dict) -> dict:
        """Qdrant is unreachable — health check should show degraded."""
        return {
            'db_status': 'up',
            'vector_status': 'down',
            'ollama_status': 'up',
        }

    def _state_has_documents(self, params: dict) -> dict:
        """Vector store has searchable documents."""
        return {
            'seed_documents': [
                {
                    'id': 'mem_abc123',
                    'content': 'The Victron MultiPlus II is configured for...',
                    'score': 0.87,
                    'source': 'claude_conversations',
                    'metadata': {
                        'file': 'conversation_2025-01-15.jsonl',
                        'chunk_index': 3
                    },
                    'created_at': '2025-01-15T10:30:00Z'
                }
            ]
        }

    def _state_no_matches(self, params: dict) -> dict:
        """Vector store exists but query won't match anything."""
        return {'seed_documents': [], 'force_empty': True}

    def _state_has_memories(self, params: dict) -> dict:
        """Database has memory entries for listing."""
        return {
            'seed_memories': [
                {
                    'id': 'mem_001',
                    'content': 'Tower server runs 28 microservices...',
                    'category': 'infrastructure',
                    'source': 'claude_conversations',
                    'created_at': '2025-01-10T08:00:00Z',
                    'updated_at': '2025-01-10T08:00:00Z',
                    'embedding_model': 'nomic-embed-text'
                }
            ],
            'total': 150
        }

    def _state_can_create(self, params: dict) -> dict:
        """System is ready to accept new memory entries."""
        return {'allow_writes': True}

    def _state_ingestion_complete(self, params: dict) -> dict:
        """At least one ingestion run has completed."""
        return {
            'ingestion': {
                'running': False,
                'last_run': '2025-02-10T03:00:00Z',
                'last_run_status': 'success',
                'documents_processed': 347,
                'documents_failed': 2,
                'next_scheduled': '2025-02-11T03:00:00Z'
            }
        }

    def _state_never_ingested(self, params: dict) -> dict:
        """Fresh system — ingestion has never been triggered."""
        return {
            'ingestion': {
                'running': False,
                'last_run': None,
                'last_run_status': None,
                'documents_processed': 0,
                'documents_failed': 0,
                'next_scheduled': None
            }
        }


# Singleton for import convenience
state_manager = ProviderStateManager()
