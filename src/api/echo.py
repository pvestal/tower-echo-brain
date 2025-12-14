#!/usr/bin/env python3
"""
Core Echo Brain query and conversation API routes
"""
import asyncio
import json
import time
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request

# Import models and services
from src.db.models import QueryRequest, QueryResponse
from src.db.database import database
from src.core.intelligence import intelligence_router
from src.services.conversation import conversation_manager

# Import identity and user context systems
from src.core.echo_identity import get_echo_identity
from src.core.user_context_manager import get_user_context_manager
from src.integrations.vault_manager import get_vault_manager

# Import external modules
from echo_brain_thoughts import echo_brain

# Import agentic persona for collaborative interaction
try:
    from src.core.agentic_persona import agentic_persona
except ImportError:
    agentic_persona = None

# Import business logic middleware for centralized pattern application
from .business_logic_middleware import business_logic_middleware, apply_business_logic_to_response

# Import memory components for conversation context
from src.memory.context_retrieval import ConversationContextRetriever
from src.memory.pronoun_resolver import PronounResolver
from src.memory.entity_extractor import EntityExtractor
from src.memory.memory_integration import save_conversation_with_entities

# Import unified conversation system
from src.core.unified_conversation import (
    unified_conversation,
    process_unified_message,
    save_unified_response
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize memory components lazily
_context_retriever = None
_pronoun_resolver = None
_entity_extractor = None

def get_memory_components():
    """Get or create memory components"""
    global _context_retriever, _pronoun_resolver, _entity_extractor
    if _context_retriever is None:
        _context_retriever = ConversationContextRetriever()
        _pronoun_resolver = PronounResolver()
        _entity_extractor = EntityExtractor()
    return _context_retriever, _pronoun_resolver, _entity_extractor

# Initialize business logic middleware with conversation manager
business_logic_middleware.initialize(conversation_manager)

def build_debug_info(query: str, intent: str, confidence: float,
                    semantic_results: list, patterns_applied: list,
                    processing_times: dict) -> dict:
    """Build comprehensive debug information for response"""

    # Get memory access info
    memory_accessed = []
    # Handle semantic_results as a list (our new format)
    if isinstance(semantic_results, list):
        for result in semantic_results[:5]:
            memory_accessed.append({
                "source": result.get('metadata', {}).get('source', 'unknown'),
                "content_preview": result.get('content', '')[:100],
                "relevance_score": result.get('metadata', {}).get('confidence', 0.7)
            })
    # Handle semantic_results as a dict (old format)
    elif isinstance(semantic_results, dict):
        for collection_name, hits in semantic_results.items():
            for hit in hits[:3]:  # Top 3 per collection
                memory_accessed.append({
                    "collection": collection_name,
                    "content_preview": str(hit.get("payload", {}))[:100],
                    "relevance_score": hit.get("score", 0.0),
                    "id": str(hit.get("id", "unknown"))
                })

    # Intent classification reasoning
    reasoning = {
        "intent_classification": {
            "selected_intent": intent,
            "confidence_score": confidence,
            "classification_factors": conversation_manager.thought_log[-5:] if conversation_manager.thought_log else [],
            "clarification_needed": confidence < 0.7
        },
        "semantic_search": {
            "collections_searched": len(semantic_results) if isinstance(semantic_results, dict) else 1 if semantic_results else 0,
            "total_memories_found": len(semantic_results) if isinstance(semantic_results, list) else sum(len(hits) for hits in semantic_results.values()) if isinstance(semantic_results, dict) else 0,
            "relevance_threshold": 0.7
        },
        "business_logic": {
            "patterns_considered": len(patterns_applied),
            "patterns_applied": patterns_applied,
            "application_success": len(patterns_applied) > 0
        }
    }

    return {
        "debug_info": {
            "query_analysis": {
                "query_length": len(query),
                "query_complexity": "high" if len(query.split()) > 10 else "low",
                "contains_technical_terms": any(term in query.lower() for term in ['api', 'database', 'service', 'error', 'code'])
            },
            "timestamp": datetime.now().isoformat(),
            "processing_path": "standard_query_flow"
        },
        "reasoning": reasoning,
        "memory_accessed": memory_accessed,
        "business_logic_applied": patterns_applied,
        "processing_breakdown": processing_times
    }

@router.post("/api/echo/query", response_model=QueryResponse)
@router.post("/api/echo/chat", response_model=QueryResponse)
async def query_echo(request: QueryRequest, http_request: Request = None):
    """Main query endpoint with intelligent routing and conversation management"""
    start_time = time.time()

    print(f"\n🎯 ECHO QUERY HANDLER CALLED: {request.query[:50]}...\n")
    logger.info(f"🎯 ECHO QUERY HANDLER CALLED: {request.query[:50]}...")

    # MEMORY AUGMENTATION - Add memory context to ALL queries
    try:
        from src.middleware.memory_augmentation_middleware import augment_with_memories
        augmented = augment_with_memories(request.query)
        if augmented != request.query:
            logger.info(f"📚 Query augmented with memory context")
            request.query = augmented
    except Exception as e:
        logger.warning(f"Memory augmentation failed: {e}")

    # Generate conversation ID if not provided
    if not request.conversation_id:
        request.conversation_id = str(uuid.uuid4())

    # Get username from request (defaults to 'anonymous')
    username = getattr(request, 'username', None)
    if not username and http_request:
        username = http_request.headers.get("X-Username", "anonymous")
    if not username:
        username = "anonymous"

    # Get user context and identity
    user_manager = await get_user_context_manager()
    user_context = await user_manager.get_or_create_context(username)
    echo_identity = get_echo_identity()

    # Recognize user and apply permissions
    user_recognition = echo_identity.recognize_user(username)
    logger.info(f"👤 User: {username} - Access: {user_recognition['access_level']}")

    # Add to user's conversation history
    await user_manager.add_conversation(username, "user", request.query)

    # RETRIEVE CONVERSATION CONTEXT
    try:
        from src.middleware.conversation_context import conversation_context, inject_context
        # Inject conversation history into the request
        request_dict = request.dict()
        request_dict = await inject_context(request_dict)
        # Update query with context if available
        if 'query' in request_dict and request_dict['query'] != request.query:
            logger.info(f"📚 Injected conversation context for {request.conversation_id}")
            request.query = request_dict['query']
    except Exception as e:
        logger.warning(f"Could not inject conversation context: {e}")

    logger.info(f"🔍 ECHO QUERY HANDLER - Query: {request.query[:50]}...")
    logger.info(f"🔍 Request type: {getattr(request, 'request_type', 'NOT_SET')}")

    # Store original query for database logging
    original_query = request.query

    # UNIFIED CONVERSATION SYSTEM - Process through unified pipeline
    try:
        unified_result = await process_unified_message(
            conversation_id=request.conversation_id,
            user_query=request.query,
            user_id=request.user_id,
            metadata={
                "intelligence_level": request.intelligence_level,
                "request_type": getattr(request, 'request_type', 'query')
            }
        )

        # Use enhanced query from unified system
        if unified_result.get("enhanced_query"):
            logger.info(f"🔄 Using unified enhanced query")
            request.query = unified_result["enhanced_query"]
    except Exception as e:
        logger.warning(f"Unified conversation processing failed: {e}")

    # SEMANTIC SEARCH INTEGRATION - Search existing memories FIRST
    semantic_results = []

    # Import and use memory search
    try:
        print(f"🔍 MEMORY SEARCH: Starting search for '{request.query}'")
        from src.managers.echo_integration import MemorySearch
        memory_search = MemorySearch()
        memories = memory_search.search_all(request.query)
        total_memories = sum(len(v) for v in memories.values())
        print(f"📚 MEMORY SEARCH: Found {total_memories} memories")
        logger.info(f"📚 Found {total_memories} memories for query")

        # Convert to semantic_results format
        if total_memories > 0:
            # Add learned patterns
            if memories.get('learned_patterns'):
                for pattern in memories['learned_patterns'][:3]:
                    semantic_results.append({
                        'content': pattern['text'],
                        'metadata': {'source': 'learned_patterns', 'confidence': pattern.get('confidence', 0.8)}
                    })

            # Add conversations
            if memories.get('conversations'):
                for conv in memories['conversations'][:2]:
                    semantic_results.append({
                        'content': f"Previous Q: {conv['query']}\nA: {conv['response']}",
                        'metadata': {'source': 'conversations', 'date': str(conv.get('date', ''))}
                    })

            # Add takeout insights
            if memories.get('takeout'):
                for item in memories['takeout'][:2]:
                    semantic_results.append({
                        'content': item['content'],
                        'metadata': {'source': 'takeout', 'type': item.get('type', '')}
                    })

            logger.info(f"📚 Added {len(semantic_results)} memory results to context")
    except Exception as e:
        logger.error(f"Memory search error: {e}")

    if hasattr(conversation_manager, 'vector_search') and conversation_manager.vector_search:
        try:
            semantic_results = conversation_manager.vector_search.search_all_collections(
                request.query, limit_per_collection=3
            )
            if semantic_results:
                logger.info(f"✅ Found semantic memories: {len(semantic_results)} collections")
        except Exception as e:
            logger.warning(f"⚠️ Semantic search failed: {e}")

    # Validate request_type field exists and log for debugging
    request_type = getattr(request, 'request_type', None)
    if not request_type:
        logger.warning(f"⚠️ request_type not set, defaulting to 'conversation'")
        request_type = 'conversation'
    else:
        logger.info(f"✅ Request type validated: {request_type}")

    # CHECK REQUEST TYPE FIRST - Route to appropriate handler
    if request_type == 'system_command':
        # Check if user has permission for system commands
        if not await user_manager.check_permission(username, "system_commands"):
            logger.warning(f"⚠️ User {username} attempted system command without permission")
            return QueryResponse(
                response="You do not have permission to execute system commands. Only the creator has this access.",
                model_used="permission_system",
                intelligence_level="access_control",
                processing_time=time.time() - start_time,
                escalation_path=["permission_denied"],
                conversation_id=request.conversation_id,
                intent="system_command",
                confidence=1.0,
                requires_clarification=False,
                clarifying_questions=[]
            )

        logger.info("🚨 ROUTING TO DIRECT SYSTEM COMMAND EXECUTION")

        # Direct execution with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"🔄 System command attempt {attempt + 1}/{max_retries}: {request.query}")

                # Direct subprocess execution - no safety restrictions
                import subprocess

                # Execute the command directly using shell for full command support
                process = await asyncio.create_subprocess_shell(
                    request.query,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd="/tmp"  # Execute in /tmp directory
                )

                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=30
                )

                stdout_text = stdout.decode('utf-8', errors='replace')
                stderr_text = stderr.decode('utf-8', errors='replace')

                processing_time = time.time() - start_time

                if process.returncode == 0:
                    logger.info(f"✅ System command executed successfully on attempt {attempt + 1}")
                    output = stdout_text
                    if stderr_text:
                        output += f"\nSTDERR: {stderr_text}"
                    return QueryResponse(
                        response=output,
                        model_used="direct_executor",
                        intelligence_level="system_command",
                        processing_time=processing_time,
                        escalation_path=[f"system_command:{request.query[:20]}"],
                        conversation_id=request.conversation_id,
                        intent="system_command",
                        confidence=1.0,
                        requires_clarification=False,
                        clarifying_questions=[]
                    )
                else:
                    logger.warning(f"⚠️ System command failed on attempt {attempt + 1}: {stderr_text}")
                    if attempt == max_retries - 1:  # Last attempt
                        processing_time = time.time() - start_time
                        return QueryResponse(
                            response=f"Command failed after {max_retries} attempts.\nSTDOUT: {stdout_text}\nSTDERR: {stderr_text}\nExit code: {process.returncode}",
                            model_used="direct_executor",
                            intelligence_level="error",
                            processing_time=processing_time,
                            escalation_path=["system_command_failed"],
                            conversation_id=request.conversation_id,
                            intent="system_command",
                            confidence=0.0,
                            requires_clarification=False,
                            clarifying_questions=[]
                        )

            except asyncio.TimeoutError:
                logger.error(f"❌ System command timed out on attempt {attempt + 1}")
                if attempt == max_retries - 1:  # Last attempt
                    processing_time = time.time() - start_time
                    return QueryResponse(
                        response=f"Command timed out after {max_retries} attempts (30s timeout each)",
                        model_used="direct_executor",
                        intelligence_level="error",
                        processing_time=processing_time,
                        escalation_path=["system_command_timeout"],
                        conversation_id=request.conversation_id,
                        intent="system_command",
                        confidence=0.0,
                        requires_clarification=False,
                        clarifying_questions=[]
                    )
            except Exception as e:
                logger.error(f"❌ System command execution error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:  # Last attempt
                    processing_time = time.time() - start_time
                    return QueryResponse(
                        response=f"System command failed after {max_retries} attempts: {str(e)}",
                        model_used="direct_executor",
                        intelligence_level="error",
                        processing_time=processing_time,
                        escalation_path=["system_command_error"],
                        conversation_id=request.conversation_id,
                        intent="system_command",
                        confidence=0.0,
                        requires_clarification=False,
                        clarifying_questions=[]
                    )
                # Wait before retry
                await asyncio.sleep(0.5 * (attempt + 1))

    # CHECK FOR SLASH COMMANDS FIRST
    if request.query.strip().startswith('/'):
        try:
            from src.commands.command_handler import CommandHandler
            cmd_handler = CommandHandler()
            command, params = cmd_handler.parse_command(request.query)

            if command:
                logger.info(f"🔧 Executing command: {command}")
                response_text = await cmd_handler.execute_command(command, params)

                processing_time = time.time() - start_time
                return QueryResponse(
                    response=response_text,
                    model_used="command_system",
                    intelligence_level="command",
                    processing_time=processing_time,
                    escalation_path=["direct_command"],
                    conversation_id=request.conversation_id,
                    intent="command",
                    confidence=1.0,
                    requires_clarification=False,
                    clarifying_questions=[]
                )
        except Exception as e:
            logger.error(f"Command execution failed: {e}")

    # Cognitive model selection if available
    selected_model = None
    complexity_score = None
    tier = None
    selection_reason = "default"
    try:
        from fixed_model_selector import ModelSelector
        selector = ModelSelector()
        selected_model, intelligence_level, selection_reason, complexity_score, tier = selector.select_model(
            request.query,
            request.intelligence_level if request.intelligence_level != "auto" else None
        )
        logger.info(f"🎯 Cognitive selection: {selected_model} ({intelligence_level}) - {selection_reason}")
        request.intelligence_level = intelligence_level
    except ImportError:
        logger.debug("Cognitive selector not available, using default routing")

    try:
        # Get memory components
        context_retriever, pronoun_resolver, entity_extractor = get_memory_components()

        # ACTUALLY USE SEMANTIC SEARCH FROM CONVERSATION MANAGER
        semantic_results = conversation_manager.search_semantic_memory(request.query)
        if semantic_results:
            logger.info(f"🔍 SEMANTIC MEMORY: Found {len(semantic_results)} relevant memories")
            # Add to context for response generation
            memory_context = "\n".join([f"- {r['content'][:200]}" for r in semantic_results[:5]])
        else:
            memory_context = ""

        # ============================================
        # STEP 1: Retrieve conversation history with memory system
        # ============================================
        conversation_context = await conversation_manager.get_conversation_context(request.conversation_id)
        logger.info(f"🔍 CONVERSATION CONTEXT: {len(conversation_context.get('history', []))} messages retrieved")

        # Get enhanced history and entities from memory system
        memory_history = await context_retriever.get_recent_history(request.conversation_id)
        active_entities = await context_retriever.get_active_entities(request.conversation_id)
        logger.info(f"🧠 MEMORY: {len(memory_history)} history messages, {len(active_entities)} entities")

        # ============================================
        # STEP 2: Extract entities from current query
        # ============================================
        current_entities = entity_extractor.extract(request.query)
        logger.info(f"📋 EXTRACTED ENTITIES: {current_entities}")

        # Merge with active entities from history
        all_entities = {**active_entities, **current_entities}
        logger.info(f"🔗 MERGED ENTITIES: {all_entities}")

        # ============================================
        # STEP 3: Resolve pronouns using context
        # ============================================
        resolved_query, resolved_entity = pronoun_resolver.resolve(
            original_query,
            all_entities  # Use merged entities for resolution
        )

        if resolved_entity:
            logger.info(f"✨ PRONOUN RESOLVED: '{original_query}' → '{resolved_query}'")
            # Update the request query to use the resolved version
            request.query = resolved_query

        # Classify intent and extract parameters
        intent, confidence, intent_params = conversation_manager.classify_intent(
            request.query,
            conversation_context.get("history", [])
        )

        logger.info(f"🎯 Intent: {intent} (confidence: {confidence:.2f}) params: {intent_params}")

        # Check with agentic persona if we should ask questions
        should_ask_user = False
        contextual_question = None
        if agentic_persona:
            should_ask_user = agentic_persona.should_ask_question(
                request.query, intent, confidence
            )
            if should_ask_user:
                contextual_question = agentic_persona.get_contextual_question(
                    request.query, intent, conversation_context.get("history", [])
                )

        # Check if clarification is needed - BUT skip if pronoun was resolved
        needs_clarification = (should_ask_user or conversation_manager.needs_clarification(intent, confidence, request.query))
        if resolved_entity:
            # If we resolved a pronoun, we likely don't need clarification
            needs_clarification = False
            logger.info("🎯 Skipping clarification - pronoun was resolved")

        if needs_clarification:
            if contextual_question:
                clarifying_questions = [contextual_question]
                response_text = contextual_question
            else:
                clarifying_questions = conversation_manager.get_clarifying_questions(intent, request.query)
                response_text = "I'd like to better understand your request. Could you help clarify?"

            # APPLY BUSINESS LOGIC PATTERNS TO CLARIFICATION RESPONSES VIA MIDDLEWARE
            response_text_with_patterns = apply_business_logic_to_response(
                request.query, response_text, "clarification"
            )

            processing_time = time.time() - start_time

            response = QueryResponse(
                response=response_text_with_patterns,
                model_used="conversation_manager",
                intelligence_level="clarification",
                processing_time=processing_time,
                escalation_path=["clarification_needed"],
                requires_clarification=True,
                clarifying_questions=clarifying_questions,
                conversation_id=request.conversation_id,
                intent=intent,
                confidence=confidence,
                resolved_query=resolved_query if resolved_query != original_query else None,
                entities_extracted=active_entities if active_entities else None
            )

            conversation_manager.update_conversation(
                request.conversation_id, request.query, intent, response.response, True
            )

            # Use unified conversation system for complete memory persistence
            try:
                await save_unified_response(
                    conversation_id=request.conversation_id,
                    response=response.response,
                    metadata={
                        "user_query": original_query,
                        "user_id": request.user_id,
                        "username": username,
                        "platform": "api",
                        "model_used": response.model_used,
                        "processing_time": response.processing_time,
                        "escalation_path": response.escalation_path,
                        "intent": intent,
                        "confidence": confidence,
                        "requires_clarification": True,
                        "clarifying_questions": clarifying_questions,
                        "complexity_score": complexity_score,
                        "tier": tier,
                        "entities": all_entities,
                        "conversation_context": conversation_context
                    }
                )
                logger.info(f"✅ Unified conversation saved for {request.conversation_id}")
            except Exception as e:
                logger.error(f"❌ Clarification save_unified_response FAILED: {e}")

            return response

        # Handle anime generation specifically
        if intent == "anime_generation":
            logger.info(f"🎬 AUTO-ANIME: Generating anime with params: {intent_params}")
            try:
                import aiohttp
                prompt_text = intent_params.get('prompt', request.query)
                prompt_text = prompt_text.replace('generate anime', '').replace('create anime', '').strip()
                if not prompt_text:
                    prompt_text = "anime magical girl"

                async with aiohttp.ClientSession() as session:
                    payload = {"prompt": prompt_text}
                    async with session.post("http://localhost:8328/api/anime/generate",
                                           json=payload, timeout=60) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            job_id = result.get("job_id", "unknown")
                            response_text = (
                                f"✅ Anime generation started!\n"
                                f"🎬 Job ID: {job_id}\n"
                                f"🎭 Prompt: {prompt_text}\n"
                                f"📊 Monitor: http://localhost:8328/api/anime/jobs/{job_id}"
                            )
                        else:
                            error_msg = await resp.text()
                            response_text = f"❌ Anime generation failed: {error_msg}"

                        response = QueryResponse(
                            response=response_text,
                            model_used="anime_service_8328",
                            intelligence_level="capability",
                            processing_time=time.time() - start_time,
                            escalation_path=["anime_generation_capability"],
                            conversation_id=request.conversation_id,
                            intent=intent,
                            confidence=confidence
                        )
            except Exception as e:
                logger.error(f"Anime generation error: {e}")
                response = QueryResponse(
                    response=f"❌ Anime generation error: {str(e)}",
                    model_used="anime_service_error",
                    intelligence_level="capability",
                    processing_time=time.time() - start_time,
                    escalation_path=["anime_error"],
                    conversation_id=request.conversation_id,
                    intent=intent,
                    confidence=confidence
                )

        # Handle other capability intents automatically
        elif intent in ['service_testing', 'service_debugging', 'service_monitoring', 'agent_delegation', 'inter_service_communication', 'image_generation', 'voice_generation', 'music_generation', 'code_review', 'code_refactor', 'code_modification']:
            response = await handle_capability_intent(intent, intent_params, request, request.conversation_id, start_time)

            # APPLY BUSINESS LOGIC PATTERNS TO CAPABILITY RESPONSES VIA MIDDLEWARE
            response.response = apply_business_logic_to_response(
                request.query, response.response, "capability"
            )

            conversation_manager.update_conversation(
                request.conversation_id, request.query, intent, response.response, False
            )

            try:
                # Use unified conversation system for complete memory persistence
                await save_unified_response(
                    conversation_id=request.conversation_id,
                    response=response.response,
                    metadata={
                        "user_query": original_query,
                        "user_id": request.user_id,
                        "username": username,
                        "platform": "api",
                        "model_used": response.model_used,
                        "processing_time": response.processing_time,
                        "escalation_path": response.escalation_path,
                        "intent": intent,
                        "confidence": confidence,
                        "requires_clarification": False,
                        "clarifying_questions": None,
                        "complexity_score": complexity_score,
                        "tier": tier,
                        "entities": all_entities,
                        "conversation_context": conversation_context
                    }
                )
                logger.info(f"✅ Unified conversation saved for {request.conversation_id}")
            except Exception as e:
                logger.error(f"❌ log_interaction unified save FAILED: {e}")

            return response

        # Build context for model selection
        context = {
            "conversation_history": conversation_context.get("history", []),
            "user_id": request.user_id,
            "intent": intent,
            "intent_params": intent_params,
            "previous_failures": len([h for h in conversation_context.get("history", []) if "error" in h.get("response", "").lower()])
        }

        if request.intelligence_level and request.intelligence_level != "auto":
            context["requested_level"] = request.intelligence_level

        # Build augmented query with memory context
        augmented_query = request.query
        if semantic_results:
            memory_context_parts = ["\n📚 Relevant memories:"]
            for result in semantic_results[:5]:
                source = result.get('metadata', {}).get('source', 'unknown')
                content = result.get('content', '')[:200]
                memory_context_parts.append(f"[{source}]: {content}...")

            augmented_query = "\n".join(memory_context_parts) + f"\n\n🔍 Current query: {request.query}"
            logger.info(f"📚 Augmented query with {len(semantic_results)} memory results")

        # FIXED: Always use direct query_model instead of broken progressive_escalation
        if selected_model:
            logger.info(f"🧠 Using cognitively selected model: {selected_model} - {selection_reason}")
            result = await intelligence_router.query_model(selected_model, augmented_query, context)
            if result["success"]:
                result["intelligence_level"] = intelligence_level
                result["escalation_path"] = [f"cognitive_selection:{selected_model}"]
                result["decision_reason"] = selection_reason
            else:
                logger.warning(f"⚠️ Cognitive model {selected_model} failed, trying fallback model")
                result = await intelligence_router.query_model("llama3.2:3b", augmented_query, context)
        else:
            # Use direct query_model instead of broken progressive_escalation
            result = await intelligence_router.query_model("llama3.2:3b", augmented_query, context)

        if result["success"]:
            processing_time = time.time() - start_time

            # APPLY PATRICK'S BUSINESS LOGIC PATTERNS TO RESPONSE VIA MIDDLEWARE
            response_with_patterns = apply_business_logic_to_response(
                request.query, result["response"], "llm_response"
            )

            # GET DEBUG INFO FOR VERBOSE RESPONSE
            patterns_applied = business_logic_middleware.get_middleware_stats().get('patterns_applied', 0)
            processing_times = {
                "total_processing": processing_time,
                "semantic_search": 0.1,  # Placeholder - we'll instrument this later
                "pattern_application": 0.05,  # Placeholder
                "llm_inference": processing_time - 0.15
            }

            debug_data = build_debug_info(
                request.query, intent, confidence, semantic_results,
                [f"Applied {patterns_applied} business logic patterns"], processing_times
            )

            response = QueryResponse(
                response=response_with_patterns,
                model_used=result["model"],
                intelligence_level=result.get("intelligence_level", "standard"),
                processing_time=processing_time,
                escalation_path=result.get("escalation_path", []),
                conversation_id=request.conversation_id,
                intent=intent,
                confidence=confidence,
                **debug_data  # Include all debug information
            )

            conversation_manager.update_conversation(
                request.conversation_id, request.query, intent, response.response, False
            )

            try:
                # Use unified conversation system for complete memory persistence
                await save_unified_response(
                    conversation_id=request.conversation_id,
                    response=response.response,
                    metadata={
                        "user_query": original_query,
                        "user_id": request.user_id,
                        "username": username,
                        "platform": "api",
                        "model_used": response.model_used,
                        "processing_time": response.processing_time,
                        "escalation_path": response.escalation_path,
                        "intent": intent,
                        "confidence": confidence,
                        "requires_clarification": False,
                        "clarifying_questions": None,
                        "complexity_score": complexity_score,
                        "tier": tier,
                        "entities": all_entities,
                        "conversation_context": conversation_context
                    }
                )
                logger.info(f"✅ Unified conversation saved for {request.conversation_id}")
            except Exception as e:
                logger.error(f"❌ log_interaction unified save FAILED: {e}")

            return response
        else:
            processing_time = time.time() - start_time
            error_message = f"❌ Query processing failed: {result.get('error', 'Unknown error')}"

            response = QueryResponse(
                response=error_message,
                model_used="error_handler",
                intelligence_level="error",
                processing_time=processing_time,
                escalation_path=["error"],
                conversation_id=request.conversation_id,
                intent=intent,
                confidence=0.0
            )

            return response

    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        processing_time = time.time() - start_time

        response = QueryResponse(
            response=f"❌ Internal error: {str(e)}",
            model_used="error_handler",
            intelligence_level="error",
            processing_time=processing_time,
            escalation_path=["exception"],
            conversation_id=request.conversation_id,
            intent="error",
            confidence=0.0
        )

        return response

@router.get("/api/echo/brain")
async def get_brain_activity():
    """Get current brain activity and neural state"""
    try:
        brain_state = echo_brain.get_brain_state()
        return {
            "brain_activity": brain_state,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get brain activity: {e}")
        raise HTTPException(status_code=500, detail=f"Brain activity error: {str(e)}")

@router.get("/api/echo/thoughts/recent")
async def get_recent_thoughts():
    """Get recent thought activity"""
    try:
        # Return recent thoughts as an array directly (UI expects array)
        return [
            {
                "id": "thought-1",
                "type": "analysis",
                "content": "Processing dashboard metrics request",
                "timestamp": datetime.now().isoformat()
            },
            {
                "id": "thought-2",
                "type": "learning",
                "content": "Updating knowledge graph connections",
                "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat()
            },
            {
                "id": "thought-3",
                "type": "reasoning",
                "content": "Evaluating model selection for query",
                "timestamp": (datetime.now() - timedelta(minutes=10)).isoformat()
            }
        ]
    except Exception as e:
        logger.error(f"Failed to get recent thoughts: {e}")
        # Return empty array on error rather than failing
        return []

@router.get("/api/echo/thoughts/{thought_id}")
async def get_thought(thought_id: str):
    """Get specific thought by ID"""
    try:
        # This would normally fetch from database
        return {
            "thought": {
                "id": thought_id,
                "type": "analysis",
                "content": f"Thought details for {thought_id}",
                "timestamp": datetime.now().isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get thought {thought_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Thought retrieval error: {str(e)}")

@router.get("/api/echo/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    try:
        conversation = conversation_manager.get_conversation_history(conversation_id)
        return {
            "conversation_id": conversation_id,
            "history": conversation,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Conversation error: {str(e)}")

@router.get("/api/echo/conversations")
async def get_conversations():
    """Get all conversations"""
    try:
        conversations = conversation_manager.get_all_conversations()
        return {
            "conversations": conversations,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Conversations error: {str(e)}")

@router.get("/api/echo/oversight/dashboard")
async def get_oversight_dashboard(http_request: Request):
    """Get creator oversight dashboard"""
    # Verify creator access
    username = http_request.headers.get("X-Username", "anonymous")
    if username != "patrick":
        raise HTTPException(status_code=403, detail="Creator access required")

    echo_identity = get_echo_identity()
    user_manager = await get_user_context_manager()
    vault_manager = await get_vault_manager()

    # Get all user contexts
    all_users = await user_manager.get_all_users()

    # Get system metrics
    import psutil
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    return {
        "identity": echo_identity.get_creator_dashboard(),
        "system_metrics": {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "disk_percent": disk.percent
        },
        "active_users": all_users,
        "vault_status": "connected" if vault_manager.is_initialized else "disconnected",
        "capabilities": echo_identity.capabilities,
        "access_log": vault_manager.get_access_log("patrick")[-20:]
    }

@router.get("/api/echo/users/{username}")
async def get_user_context(username: str, http_request: Request):
    """Get specific user context (creator only)"""
    # Verify creator access
    requester = http_request.headers.get("X-Username", "anonymous")
    if requester != "patrick":
        raise HTTPException(status_code=403, detail="Creator access required")

    user_manager = await get_user_context_manager()
    context = await user_manager.get_or_create_context(username)

    return {
        "user": username,
        "context": context.get_context_summary(),
        "memory": await user_manager.get_user_memory(username)
    }

@router.post("/api/echo/users/{username}/preferences")
async def update_user_preferences(username: str, preferences: Dict[str, Any], http_request: Request):
    """Update user preferences (creator only)"""
    # Verify creator access
    requester = http_request.headers.get("X-Username", "anonymous")
    if requester != "patrick":
        raise HTTPException(status_code=403, detail="Creator access required")

    user_manager = await get_user_context_manager()

    results = {}
    for key, value in preferences.items():
        if await user_manager.update_preference(username, key, value):
            results[key] = "updated"
        else:
            results[key] = "failed"

    return {"username": username, "updates": results}

async def handle_capability_intent(intent: str, intent_params: dict, request: QueryRequest, conversation_id: str, start_time: float):
    """Handle capability-based intents"""
    from src.utils.helpers import safe_executor

    try:
        result = await safe_executor.execute_capability(intent, intent_params, request.query)
        processing_time = time.time() - start_time

        return QueryResponse(
            response=result["response"],
            model_used=result.get("model", "capability_handler"),
            intelligence_level="capability",
            processing_time=processing_time,
            escalation_path=[f"capability:{intent}"],
            conversation_id=conversation_id,
            intent=intent,
            confidence=1.0
        )
    except Exception as e:
        logger.error(f"Capability {intent} failed: {e}")
        processing_time = time.time() - start_time

        return QueryResponse(
            response=f"❌ Capability {intent} failed: {str(e)}",
            model_used="error_handler",
            intelligence_level="error",
            processing_time=processing_time,
            escalation_path=["capability_error"],
            conversation_id=conversation_id,
            intent=intent,
            confidence=0.0
        )