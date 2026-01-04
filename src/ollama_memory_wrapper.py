#!/usr/bin/env python3
"""
Ollama wrapper that automatically adds memory context to all prompts
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:
    import ollama
    _original_chat = ollama.chat
    _original_generate = ollama.generate
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama not available for memory wrapper")
    ollama = None
    _original_chat = None
    _original_generate = None

def augment_prompt_with_memory(prompt: str) -> str:
    """Add memory context to prompt"""
    try:
        # Don't augment if already has memory context
        if "Relevant memories:" in prompt or "Based on these memories:" in prompt:
            return prompt

        from src.middleware.memory_augmentation_middleware import augment_with_memories
        import uuid
        request_id = str(uuid.uuid4())
        augmented = augment_with_memories(prompt, request_id)
        if augmented != prompt:
            print(f"[OLLAMA WRAPPER] 📚 Added memory context to prompt (request_id: {request_id})")
            logger.info(f"📚 Ollama wrapper: Added memory context (request_id: {request_id})")
        return augmented
    except Exception as e:
        logger.debug(f"Memory augmentation failed: {e}")
        return prompt

def memory_aware_chat(*args, **kwargs):
    """Wrapper for ollama.chat that adds memory context"""
    # Extract messages and augment user prompts
    if 'messages' in kwargs:
        for msg in kwargs['messages']:
            if msg.get('role') == 'user' and 'content' in msg:
                msg['content'] = augment_prompt_with_memory(msg['content'])

    return _original_chat(*args, **kwargs)

def memory_aware_generate(*args, **kwargs):
    """Wrapper for ollama.generate that adds memory context"""
    # Augment the prompt
    if 'prompt' in kwargs:
        kwargs['prompt'] = augment_prompt_with_memory(kwargs['prompt'])
    elif len(args) > 1:
        args = list(args)
        args[1] = augment_prompt_with_memory(args[1])
        args = tuple(args)

    return _original_generate(*args, **kwargs)

# Monkey patch ollama module
def install_memory_wrapper():
    """Install the memory wrapper into ollama module"""
    if not OLLAMA_AVAILABLE:
        logger.warning("Cannot install memory wrapper - ollama not available")
        return False

    print("[OLLAMA WRAPPER] Installing memory augmentation wrapper")
    ollama.chat = memory_aware_chat
    ollama.generate = memory_aware_generate
    logger.info("✅ Ollama memory wrapper installed")
    return True