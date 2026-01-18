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

# Import the model router for dynamic model selection
from src.model_router import ModelRouter

# Import Qdrant memory system for vectorization
from src.qdrant_memory import QdrantMemory

# Import identity and user context systems
from src.core.echo_identity import get_echo_identity
from src.core.user_context_manager import get_user_context_manager
from src.integrations.vault_manager import get_vault_manager

# Import external modules
from src.core.echo.echo_brain_thoughts import echo_brain

# Import agentic persona for collaborative interaction
try:
    from src.core.agentic_persona import agentic_persona
except ImportError:
    agentic_persona = None

# Import business logic middleware for centralized pattern application
# TODO: Reimplement business logic middleware after fixing import structure
# from src.services.business_logic_applicator import business_logic_middleware, apply_business_logic_to_response
business_logic_middleware = None
apply_business_logic_to_response = None

# Import memory components for conversation context
from src.memory.context_retrieval import ConversationContextRetriever
from src.memory.pronoun_resolver import PronounResolver
from src.memory.entity_extractor import EntityExtractor
from src.memory.memory_integration import save_conversation_with_entities

# Reasoning module will be loaded directly from file

# Import conversation management system
from src.core.conversation_manager import (
    conversation_handler,
    process_message,
    save_response
)

# Import Tower LLM Executor for delegation
from src.core.tower_llm_executor import tower_executor


def handle_errors(func):
    """Decorator to handle errors gracefully."""
    from functools import wraps
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except asyncio.TimeoutError:
            logger.error(f"Timeout in {func.__name__}")
            return {"error": "Request timeout", "status": "timeout"}
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            return {"error": str(e), "status": "error"}
    return wrapper

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize memory components lazily
_context_retriever = None
_pronoun_resolver = None
_entity_extractor = None

# Initialize the model router for dynamic model selection
model_router = ModelRouter()

# Initialize Qdrant memory system
qdrant_memory = QdrantMemory()

def get_memory_components():
    """Get or create memory components"""
    global _context_retriever, _pronoun_resolver, _entity_extractor
    if _context_retriever is None:
        _context_retriever = ConversationContextRetriever()
        _pronoun_resolver = PronounResolver()
        _entity_extractor = EntityExtractor()
    return _context_retriever, _pronoun_resolver, _entity_extractor

# Initialize business logic middleware with conversation manager
# TODO: Fix after reimplementing business logic
# business_logic_middleware.initialize(conversation_manager)

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
@handle_errors
async def query_echo(request: QueryRequest, http_request: Request = None):
    """Main query endpoint with intelligent routing and conversation management"""
    start_time = time.time()


    # Original complex logic continues below...

    print(f"\n🎯 ECHO QUERY HANDLER CALLED: {request.query[:50]}...\n")
    logger.info(f"🎯 ECHO QUERY HANDLER CALLED: {request.query[:50]}...")

    # Store original query BEFORE any modifications
    original_query = request.query

    # MEMORY AUGMENTATION - Add memory context to ALL queries
    try:
        from src.middleware.memory_augmentation_middleware import augment_with_memories
        # Generate unique request ID for memory isolation
        request_id = str(uuid.uuid4())
        augmented = augment_with_memories(request.query, request_id)
        if augmented != request.query:
            logger.info(f"📚 Query augmented with memory context (request_id: {request_id})")
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

    # ========================================
    # CAPABILITY COORDINATOR - CHECK FOR ACTION REQUESTS
    # ========================================
    logger.info(f"🔍 CHECKING CAPABILITY COORDINATOR for query: {original_query[:50]}...")
    try:
        from src.capabilities.echo_capability_coordinator import get_capability_coordinator
        capability_coordinator = get_capability_coordinator()
        logger.info(f"📋 Capability coordinator instance: {capability_coordinator}")

        if capability_coordinator:
            logger.info(f"🎯 Processing request through capability coordinator...")
            should_execute, execution_result = await capability_coordinator.process_request(
                original_query, username
            )
            logger.info(f"📊 Capability check result - should_execute: {should_execute}, result: {execution_result}")

            if should_execute and execution_result:
                logger.info(f"✅ EXECUTING CAPABILITY: {execution_result.get('capability', 'unknown')}")

                # Format the execution result as a response
                response_text = capability_coordinator.format_execution_response(
                    execution_result, original_query
                )

                processing_time = time.time() - start_time

                # Create response with execution result
                response = QueryResponse(
                    response=response_text,
                    model_used="capability_coordinator",
                    intelligence_level="autonomous",
                    processing_time=processing_time,
                    escalation_path=["capability_execution"],
                    conversation_id=request.conversation_id,
                    intent="action_execution",
                    confidence=1.0,
                    requires_clarification=False,
                    clarifying_questions=[],
                    reasoning={
                        "action_taken": True,
                        "capability": execution_result.get("capability", "unknown"),
                        "success": execution_result.get("success", False)
                    }
                )

                # Log the action execution to database
                await user_manager.add_conversation(username, "assistant", response_text)

                # Log to database for tracking
                try:
                    await database.log_interaction(
                        query=original_query,
                        response=response_text,
                        model_used="capability_coordinator",
                        processing_time=processing_time,
                        escalation_path=["capability_execution"],
                        conversation_id=request.conversation_id,
                        user_id=username,
                        intent="action_execution",
                        confidence=1.0,
                        requires_clarification=False,
                        complexity_score=0.5,
                        tier="capability"
                    )
                    logger.info(f"✅ Capability execution logged to database")
                except Exception as e:
                    logger.error(f"❌ Failed to log capability execution: {e}")

                logger.info(f"🚀 Action executed for {username}: {execution_result.get('capability', 'unknown')}")

                return response
            else:
                logger.info(f"❌ No capability match found for query: {original_query[:50]}")
        else:
            logger.warning("⚠️ Capability coordinator is None - not initialized!")
    except Exception as e:
        logger.error(f"⚠️ Capability coordinator error: {e}", exc_info=True)
        # Continue with normal processing if capability execution fails

    # RETRIEVE CONVERSATION CONTEXT
    try:
        from src.intelligence.conversation_context import ConversationContext, enhance_query_with_context
        # For now, skip context injection to prevent timeouts
        # TODO: Implement proper context injection with ConversationContext class
        logger.info(f"📚 Context injection disabled for performance - using direct query")
    except Exception as e:
        logger.warning(f"Could not load conversation context: {e}")

    logger.info(f"🔍 ECHO QUERY HANDLER - Query: {request.query[:50]}...")
    logger.info(f"🔍 Request type: {getattr(request, 'request_type', 'NOT_SET')}")

    # ========================================
    # KNOWLEDGE RETRIEVAL - SEARCH INDEXED CONTENT FIRST
    # ========================================
    logger.info("🔍 KNOWLEDGE RETRIEVAL: === STARTING ===")
    knowledge_context = ""
    try:
        from qdrant_client import QdrantClient

        # Load embedding service via direct file import (same pattern as reasoning)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "embedding_service",
            "/opt/tower-echo-brain/src/services/embedding_service.py"
        )
        embed_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(embed_module)

        # Initialize embedding service
        embedder = embed_module.EmbeddingService()
        await embedder.initialize()

        # Get query embedding
        query_vector = await embedder.embed_single(original_query)

        # Search multiple collections
        client = QdrantClient(host="localhost", port=6333)
        collections = ['documents', 'code', 'facts', 'solutions', 'photos']
        all_results = []

        for collection in collections:
            try:
                hits = client.search(
                    collection_name=collection,
                    query_vector=query_vector,
                    limit=3,
                    score_threshold=0.4  # Lower threshold for better recall
                )
                for hit in hits:
                    # Handle different content types
                    if collection == 'photos':
                        content = f"Photo: {hit.payload.get('filename', 'unknown')} from {hit.payload.get('date', 'unknown date')} at {hit.payload.get('location', 'unknown location')}"
                    else:
                        content = hit.payload.get('text') or hit.payload.get('content') or hit.payload.get('problem', '')
                    if content and hit.score > 0.4:
                        all_results.append({
                            'source': collection,
                            'content': content[:500],
                            'score': hit.score
                        })
            except Exception as e:
                logger.debug(f"Collection {collection} search failed: {e}")

        # Sort by relevance and take top 5
        all_results.sort(key=lambda x: x['score'], reverse=True)
        top_results = all_results[:5]

        if top_results:
            knowledge_parts = ["📚 Relevant knowledge from your indexed content:"]
            for r in top_results:
                knowledge_parts.append(f"[{r['source']}] (score: {r['score']:.2f}): {r['content'][:300]}...")
            knowledge_context = "\n".join(knowledge_parts)
            logger.info(f"🔍 KNOWLEDGE RETRIEVAL: Found {len(top_results)} results across collections")
        else:
            logger.debug("🔍 KNOWLEDGE RETRIEVAL: No relevant knowledge found in indexed content")

    except Exception as e:
        logger.warning(f"🔍 KNOWLEDGE RETRIEVAL failed: {e}")

    # Create augmented query for reasoning and conversation processing
    query_for_processing = original_query
    if knowledge_context:
        query_for_processing = f"{knowledge_context}\n\n🔍 User question: {original_query}"
        logger.info(f"📚 KNOWLEDGE RETRIEVAL: Augmented query is {len(query_for_processing)} chars")

    # ========================================
    # REASONING PRIORITY - CHECK ENHANCED QUERY
    # ========================================
    logger.info(f"🔍 Checking if reasoning needed for query: {original_query[:100]}...")

    # Load reasoning module directly from file (bypasses import issues)
    reasoning_available = False
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "deepseek_reasoner",
            "/opt/tower-echo-brain/src/reasoning/deepseek_reasoner.py"
        )
        reasoner_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reasoner_module)

        should_use_reasoning = reasoner_module.should_use_reasoning
        execute_reasoning = reasoner_module.execute_reasoning
        reasoning_available = True
        logger.info("✅ Reasoning module loaded via direct file import")
    except Exception as e:
        logger.warning(f"⚠️ Failed to load reasoning module: {e}")

    if reasoning_available and should_use_reasoning(original_query):
        logger.info(f"🧠 REASONING PRIORITY: {original_query[:50]}...")
        try:
            # Use augmented query for reasoning to include knowledge context
            reasoning_result = await execute_reasoning(query_for_processing)

            if reasoning_result.success:
                processing_time = time.time() - start_time

                # Build response with reasoning
                response = QueryResponse(
                    response=reasoning_result.final_answer,
                    model_used=reasoning_result.model_used,
                    intelligence_level="reasoning",
                    processing_time=processing_time,
                    escalation_path=["reasoning_priority"],
                    conversation_id=request.conversation_id,
                    intent="reasoning",
                    confidence=1.0,
                    requires_clarification=False,
                    clarifying_questions=[],
                    reasoning={
                        "steps": reasoning_result.thinking_steps,
                        "model": reasoning_result.model_used
                    }
                )

                # Log to conversation history with original query
                try:
                    conversation_manager.update_conversation(
                        request.conversation_id, original_query, "reasoning", response.response, False
                    )
                except Exception as e:
                    logger.warning(f"Failed to update conversation history: {e}")

                logger.info(f"✅ Reasoning completed with {len(reasoning_result.thinking_steps)} steps")
                return response
            else:
                logger.warning(f"⚠️ Reasoning failed: {reasoning_result.error}, continuing to normal handlers")
        except Exception as e:
            logger.warning(f"⚠️ Reasoning execution failed: {e}, continuing normally")

    # CONVERSATION MANAGER - Process through conversation pipeline
    logger.info(f"🔄 CONVERSATION MANAGER: Receiving query of {len(query_for_processing)} chars")
    try:
        unified_result = await process_message(
            conversation_id=request.conversation_id,
            query_text=query_for_processing,
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
        # BYPASS SIMPLE QUERIES - prevent context contamination
        query_lower = request.query.lower().strip()
        bypass_patterns = [
            'return json', 'return only', 'return:', 'return ', 'json:', 'echo ',
            'print ', 'test', 'hello', 'ping', 'status', '{"', '[', 'get ', 'show ', 'list '
        ]

        if len(request.query) < 15 or any(query_lower.startswith(p) for p in bypass_patterns):
            print(f"📋 BYPASSING MEMORY SEARCH for simple query: {request.query}")
            logger.info(f"📋 Bypassing memory search for simple query: {request.query[:50]}")
        else:
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

        logger.info("🔒 ROUTING TO SECURE SYSTEM COMMAND EXECUTION")

        # Import secure command executor
        from src.security.safe_command_executor import safe_command_executor

        # Execute through secure validator
        try:
            logger.info(f"🔒 Secure command execution: {request.query}")

            # Execute with security validation
            result = await safe_command_executor.execute_command(
                command_string=request.query,
                username=username
            )

            processing_time = time.time() - start_time

            if result['success']:
                logger.info(f"✅ Secure command executed successfully: {result['command']}")

                # Format output
                output_parts = []
                if result.get('stdout'):
                    output_parts.append(result['stdout'])
                if result.get('stderr'):
                    output_parts.append(f"STDERR: {result['stderr']}")

                output = "\n".join(output_parts) if output_parts else "Command completed successfully"

                return QueryResponse(
                    response=output,
                    model_used="secure_command_executor",
                    intelligence_level="system_command",
                    processing_time=processing_time,
                    escalation_path=[f"secure_command:{result['command']}"],
                    conversation_id=request.conversation_id,
                    intent="system_command",
                    confidence=1.0,
                    requires_clarification=False,
                    clarifying_questions=[]
                )
            else:
                logger.warning(f"🚫 Command blocked/failed: {result['error']}")
                processing_time = time.time() - start_time

                # Provide helpful information about allowed commands
                allowed_commands = safe_command_executor.get_allowed_commands()
                allowed_list = "\n".join([
                    f"• {cmd}: {info['description']}"
                    for cmd, info in allowed_commands.items()
                ])

                error_response = f"""❌ {result['error']}

🔒 Security Policy: Only whitelisted commands are allowed for security.

📝 Allowed commands:
{allowed_list}

💡 Example usage:
• ls -la /tmp
• ps aux
• systemctl status nginx
• df -h"""

                return QueryResponse(
                    response=error_response,
                    model_used="secure_command_executor",
                    intelligence_level="security_policy",
                    processing_time=processing_time,
                    escalation_path=["command_blocked"],
                    conversation_id=request.conversation_id,
                    intent="system_command",
                    confidence=1.0,
                    requires_clarification=False,
                    clarifying_questions=[]
                )

        except Exception as e:
            logger.error(f"❌ Secure command executor failed: {e}")
            processing_time = time.time() - start_time
            return QueryResponse(
                response=f"❌ Command execution system error: {str(e)}",
                model_used="secure_command_executor",
                intelligence_level="error",
                processing_time=processing_time,
                escalation_path=["executor_error"],
                conversation_id=request.conversation_id,
                intent="system_command",
                confidence=0.0,
                requires_clarification=False,
                clarifying_questions=[]
            )

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
        # Disabled: archive path with hyphens can't be imported
        # from archive.fixed-naming-cleanup-20251030.fixed_model_selector import ModelSelector
        # selector = ModelSelector()
        # selected_model, intelligence_level, selection_reason, complexity_score, tier = selector.select_model(
        #     request.query,
        #     request.intelligence_level if request.intelligence_level != "auto" else None
        # )
        # logger.info(f"🎯 Cognitive selection: {selected_model} ({intelligence_level}) - {selection_reason}")
        # request.intelligence_level = intelligence_level
        pass
    except ImportError:
        logger.debug("Cognitive selector not available, using default routing")

    try:
        # Get memory components
        context_retriever, pronoun_resolver, entity_extractor = get_memory_components()

        # Re-enabled with domain filtering to prevent cross-domain contamination
        from src.utils.domain_classifier import QueryDomainClassifier

        classifier = QueryDomainClassifier()
        query_domain = classifier.classify(request.query)

        # Search semantic memory
        semantic_results = conversation_manager.search_semantic_memory(request.query)

        # Filter results based on query domain
        if semantic_results:
            filtered_results = [
                r for r in semantic_results
                if classifier.filter_content(r.get('content', ''), query_domain)
            ]

            if filtered_results:
                logger.info(f"🔍 SEMANTIC MEMORY: {len(filtered_results)}/{len(semantic_results)} memories passed domain filter for '{query_domain}' query")
                # Add to context for response generation
                memory_context = "\n".join([f"- {r['content'][:200]}" for r in filtered_results[:5]])
            else:
                logger.info(f"🚫 All {len(semantic_results)} semantic memories filtered out for '{query_domain}' query")
                memory_context = ""

            semantic_results = filtered_results
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

        # Skip clarification for coding queries to enable proper model routing
        coding_keywords = ['code', 'function', 'python', 'javascript', 'java', 'c++', 'ruby', 'php', 'go', 'rust', 'swift', 'kotlin', 'scala', 'haskell', 'deepseek', 'programming', 'algorithm', 'debug', 'fix', 'write', 'implement']
        if any(keyword in request.query.lower() for keyword in coding_keywords):
            needs_clarification = False
            logger.info("🎯 Skipping clarification - coding query detected, routing to unified router")

        if needs_clarification:
            if contextual_question:
                clarifying_questions = [contextual_question]
                response_text = contextual_question
            else:
                clarifying_questions = conversation_manager.get_clarifying_questions(intent, request.query)
                response_text = "I'd like to better understand your request. Could you help clarify?"

            # APPLY BUSINESS LOGIC PATTERNS TO CLARIFICATION RESPONSES VIA MIDDLEWARE
            # TODO: Fix after reimplementing business logic
            # response_text_with_patterns = apply_business_logic_to_response(
            #     request.query, response_text, "clarification"
            # )
            response_text_with_patterns = response_text

            processing_time = time.time() - start_time

            response = QueryResponse(
                response=response_text_with_patterns,
                model_used="clarification_system",
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
                await save_response(
                    conversation_id=request.conversation_id,
                    response=response.response,
                    metadata={
                        "query_text": original_query,
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

                # Store conversation in Qdrant for vector memory
                try:
                    memory_text = f"User: {original_query}\nAssistant: {response.response}"
                    success = await qdrant_memory.store_memory(
                        text=memory_text,
                        metadata={
                            "conversation_id": request.conversation_id,
                            "user_id": request.user_id or "default",
                            "intent": intent,
                            "model_used": response.model_used,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                    if success:
                        logger.info(f"🧠 Stored memory in Qdrant for conversation {request.conversation_id}")
                except Exception as qdrant_error:
                    logger.error(f"❌ Qdrant storage failed: {qdrant_error}")

            except Exception as e:
                logger.error(f"❌ Clarification save_response FAILED: {e}")

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
            # TODO: Fix after reimplementing business logic
            # response.response = apply_business_logic_to_response(
            #     request.query, response.response, "capability"
            # )
            pass  # response.response stays as is

            conversation_manager.update_conversation(
                request.conversation_id, request.query, intent, response.response, False
            )

            try:
                # Use unified conversation system for complete memory persistence
                await save_response(
                    conversation_id=request.conversation_id,
                    response=response.response,
                    metadata={
                        "query_text": original_query,
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
        logger.info(f"🔍 DEBUG: semantic_results count: {len(semantic_results)}")
        logger.info(f"🔍 DEBUG: original query length: {len(request.query)}")

        # Re-enabled with domain filtering - semantic_results already filtered above
        if semantic_results and len(semantic_results) > 0:
            memory_context_parts = ["\n📚 Relevant memories:"]
            for result in semantic_results[:5]:
                source = result.get('metadata', {}).get('source', 'unknown')
                content = result.get('content', '')[:200]
                memory_context_parts.append(f"[{source}]: {content}...")

            augmented_query = "\n".join(memory_context_parts) + f"\n\n🔍 Current query: {request.query}"
            logger.info(f"📚 Augmented query with {len(semantic_results)} domain-filtered memory results")
        else:
            logger.info("📚 No relevant memories passed domain filter for augmentation")

        logger.info(f"🔍 DEBUG: augmented query length: {len(augmented_query)}")

        # ============================================
        # UNIFIED MODEL ROUTING - SINGLE SOURCE OF TRUTH
        # ============================================
        from src.routing.unified_router import unified_router

        # Get model selection from unified router
        selection = unified_router.select_model(augmented_query)
        selected_model = selection.model_name

        # Log the decision
        logger.info(f"🔀 UNIFIED ROUTER: Selected {selected_model} for: {augmented_query[:50]}...")
        logger.info(f"   Reason: {selection.reason}")

        # Execute query with selected model
        try:
            # Check if this is a reasoning query that needs special handling
            if selected_model == "deepseek-r1:8b" and selection.tier == "reasoning":
                # Use actual reasoning execution with proper import path
                import sys
                if '/opt/tower-echo-brain/src' not in sys.path:
                    sys.path.insert(0, '/opt/tower-echo-brain/src')
                from reasoning.deepseek_reasoner import execute_reasoning

                logger.info(f"🧠 Executing DeepSeek reasoning for: {request.query[:50]}...")
                reasoning_result = await execute_reasoning(request.query)

                if reasoning_result.success:
                    result = {
                        "success": True,
                        "response": reasoning_result.final_answer,
                        "model": reasoning_result.model_used,
                        "tier": "reasoning",
                        "decision_reason": selection.reason,
                        "reasoning": {
                            "steps": reasoning_result.thinking_steps,
                            "model": reasoning_result.model_used
                        }
                    }
                    logger.info(f"✅ Reasoning completed with {len(reasoning_result.thinking_steps)} thinking steps")
                else:
                    # Reasoning failed, fallback to standard model
                    logger.warning(f"⚠️ Reasoning failed: {reasoning_result.error}, using fallback")
                    result = await intelligence_router.query_model("qwen2.5:3b", augmented_query, context)
                    result["model"] = "qwen2.5:3b"
                    result["tier"] = "fallback"
                    result["reasoning_error"] = reasoning_result.error
            else:
                # Standard model execution
                result = await intelligence_router.query_model(selected_model, augmented_query, context)
                if result["success"]:
                    result["model"] = selected_model
                    result["tier"] = selection.tier
                    result["decision_reason"] = selection.reason
                else:
                    # Fallback to fast model if selected model fails
                    logger.warning(f"⚠️ Model {selected_model} failed, using fallback qwen2.5:3b")
                    result = await intelligence_router.query_model("qwen2.5:3b", augmented_query, context)
                    result["model"] = "qwen2.5:3b"
                    result["tier"] = "fallback"
        except Exception as e:
            logger.error(f"❌ Unified router error: {e}")
            # Emergency fallback
            result = await intelligence_router.query_model("qwen2.5:3b", augmented_query, context)
            result["model"] = "qwen2.5:3b"
            result["tier"] = "emergency_fallback"

        if result["success"]:
            processing_time = time.time() - start_time

            # APPLY PATRICK'S BUSINESS LOGIC PATTERNS TO RESPONSE VIA MIDDLEWARE
            # TODO: Fix after reimplementing business logic
            # response_with_patterns = apply_business_logic_to_response(
            #     request.query, result["response"], "llm_response"
            # )
            response_with_patterns = result["response"]

            # GET DEBUG INFO FOR VERBOSE RESPONSE
            # patterns_applied = business_logic_middleware.get_middleware_stats().get('patterns_applied', 0)
            patterns_applied = 0
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
                await save_response(
                    conversation_id=request.conversation_id,
                    response=response.response,
                    metadata={
                        "query_text": original_query,
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

                # Store conversation in Qdrant for vector memory
                try:
                    # Create memory text combining query and response
                    memory_text = f"User: {original_query}\nAssistant: {response.response}"

                    # Store in Qdrant with metadata
                    success = await qdrant_memory.store_memory(
                        text=memory_text,
                        metadata={
                            "conversation_id": request.conversation_id,
                            "user_id": request.user_id or "default",
                            "intent": intent,
                            "model_used": response.model_used,
                            "timestamp": datetime.now().isoformat(),
                            "confidence": confidence,
                            "tier": tier if tier else "unknown"
                        }
                    )

                    if success:
                        logger.info(f"🧠 Stored memory in Qdrant for conversation {request.conversation_id}")
                    else:
                        logger.warning(f"⚠️ Failed to store memory in Qdrant")

                except Exception as qdrant_error:
                    logger.error(f"❌ Qdrant storage failed: {qdrant_error}")

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
@handle_errors
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
@handle_errors
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
@handle_errors
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
@handle_errors
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
@handle_errors
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
@handle_errors
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
@handle_errors
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
@handle_errors
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
@router.get("/api/echo/status")
async def get_echo_status():
    """Get Echo Brain system status with recent activity and persona"""
    import psycopg2
    from psycopg2.extras import RealDictCursor
    
    try:
        # Get recent messages from database using psycopg2 directly
        conn = psycopg2.connect(**database.db_config)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT conversation_id, query, response, intent, timestamp
            FROM echo_unified_interactions
            WHERE conversation_id NOT LIKE 'test_%'
              AND conversation_id NOT LIKE 'debug_%'
              AND conversation_id NOT LIKE 'metrics_%'
            ORDER BY timestamp DESC
            LIMIT 5
        """)
        recent_messages = cursor.fetchall()
        
        # Get conversation stats
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT conversation_id) as total_conversations,
                COUNT(*) as total_messages
            FROM echo_unified_interactions
            WHERE timestamp > NOW() - INTERVAL '24 hours'
        """)
        stats = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        # Get agentic persona state if available
        persona_state = "responsive"
        if agentic_persona:
            try:
                persona_state = agentic_persona.current_mode or "responsive"
            except:
                pass
        
        return {
            "status": "active",
            "persona": persona_state,
            "recent_messages": [
                {
                    "conversation_id": msg["conversation_id"],
                    "query_preview": msg["query"][:100] if msg["query"] else "",
                    "response_preview": msg["response"][:100] if msg["response"] else "",
                    "intent": msg["intent"],
                    "timestamp": msg["timestamp"].isoformat() if msg["timestamp"] else None
                }
                for msg in recent_messages
            ],
            "stats_24h": {
                "conversations": stats["total_conversations"] if stats else 0,
                "messages": stats["total_messages"] if stats else 0
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get Echo status: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

async def get_echo_goals():
    """Get Echo Brain's current system goals"""
    try:
        # Get goals from database or configuration
        goals = await database.fetch_all("""
            SELECT goal_id, goal_description, priority, status, created_at
            FROM echo_goals
            ORDER BY priority ASC, created_at DESC
            LIMIT 10
        """)
        
        # If no table exists or no goals, return default goals
        if not goals:
            default_goals = [
                {
                    "goal_id": "assist_patrick",
                    "goal_description": "Provide helpful assistance to Patrick",
                    "priority": 1,
                    "status": "active",
                    "created_at": None
                },
                {
                    "goal_id": "maintain_tower",
                    "goal_description": "Monitor and maintain Tower infrastructure",
                    "priority": 2,
                    "status": "active",
                    "created_at": None
                },
                {
                    "goal_id": "learn_improve",
                    "goal_description": "Continuously learn and improve responses",
                    "priority": 3,
                    "status": "active",
                    "created_at": None
                }
            ]
            return {
                "goals": default_goals,
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "goals": [
                {
                    "goal_id": g["goal_id"],
                    "description": g["goal_description"],
                    "priority": g["priority"],
                    "status": g["status"],
                    "created_at": g["created_at"].isoformat() if g["created_at"] else None
                }
                for g in goals
            ],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        # Return default goals if database query fails
        logger.warning(f"Failed to fetch goals from database, using defaults: {e}")
        default_goals = [
            {
                "goal_id": "assist_patrick",
                "description": "Provide helpful assistance to Patrick",
                "priority": 1,
                "status": "active"
            },
            {
                "goal_id": "maintain_tower",
                "description": "Monitor and maintain Tower infrastructure",
                "priority": 2,
                "status": "active"
            },
            {
                "goal_id": "learn_improve",
                "description": "Continuously learn and improve responses",
                "priority": 3,
                "status": "active"
            }
        ]
        return {
            "goals": default_goals,
            "timestamp": datetime.now().isoformat()
        }

# ============= MISSING ENDPOINTS FOR DASHBOARD =============
import socket
import time as time_module

# Store startup time at module level
_startup_time = time_module.time()

@router.get("/api/coordination/services")
async def get_services():
    """Service discovery for dashboard"""
    services = [
        {"name": "Echo Brain", "port": 8309, "status": "unknown"},
        {"name": "Anime Production", "port": 8328, "status": "unknown"},
        {"name": "ComfyUI", "port": 8188, "status": "unknown"},
        {"name": "Ollama", "port": 11434, "status": "unknown"},
        {"name": "Qdrant", "port": 6333, "status": "unknown"},
        {"name": "Redis", "port": 6379, "status": "unknown"},
    ]
    
    # Check each service
    for svc in services:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', svc["port"]))
            svc["status"] = "running" if result == 0 else "stopped"
            sock.close()
        except:
            svc["status"] = "error"
    
    return {"services": services}

@router.get("/api/theater/agents")
async def get_theater_agents():
    """Get status and list of available autonomous agents"""
    try:
        agents = []

        # DeepSeek Coding Agent
        try:
            from src.agents.deepseek_coding_agent import get_agent
            coding_agent = get_agent()
            agents.append({
                "name": "DeepSeek Coding Agent",
                "type": "coding",
                "status": "active",
                "description": "Autonomous code generation, analysis, and debugging",
                "model": "deepseek-coder-v2:16b",
                "endpoints": ["/api/coding-agent/*"],
                "capabilities": ["code_generation", "debugging", "refactoring", "analysis"]
            })
        except Exception as e:
            agents.append({
                "name": "DeepSeek Coding Agent",
                "type": "coding",
                "status": "error",
                "error": str(e)
            })

        # Reasoning Agent
        try:
            from src.agents.reasoning_agent import ReasoningAgent
            agents.append({
                "name": "Reasoning Agent",
                "type": "reasoning",
                "status": "available",
                "description": "Complex problem analysis and strategic thinking",
                "model": "deepseek-r1:8b",
                "capabilities": ["analysis", "planning", "problem_solving"]
            })
        except Exception as e:
            agents.append({
                "name": "Reasoning Agent",
                "type": "reasoning",
                "status": "error",
                "error": str(e)
            })

        # Narration Agent
        try:
            from src.agents.narration_agent import NarrationAgent
            agents.append({
                "name": "Narration Agent",
                "type": "creative",
                "status": "available",
                "description": "Content creation and narrative generation",
                "model": "llama3.2:3b",
                "capabilities": ["storytelling", "content_creation", "documentation"]
            })
        except Exception as e:
            agents.append({
                "name": "Narration Agent",
                "type": "creative",
                "status": "error",
                "error": str(e)
            })

        # Autonomous Repair Executor
        try:
            from src.tasks.autonomous_repair_executor import RepairExecutor
            agents.append({
                "name": "Autonomous Repair Executor",
                "type": "maintenance",
                "status": "available",
                "description": "Automated system repair and maintenance",
                "capabilities": ["system_repair", "error_detection", "automated_fixes"]
            })
        except Exception as e:
            agents.append({
                "name": "Autonomous Repair Executor",
                "type": "maintenance",
                "status": "error",
                "error": str(e)
            })

        active_count = len([a for a in agents if a["status"] == "active"])
        available_count = len([a for a in agents if a["status"] in ["active", "available"]])

        return {
            "agents": agents,
            "status": "configured" if available_count > 0 else "error",
            "message": f"{active_count} active agents, {available_count} total available",
            "total": len(agents),
            "active": active_count,
            "available": available_count
        }

    except Exception as e:
        return {
            "agents": [],
            "status": "error",
            "message": f"Failed to load theater agents: {str(e)}"
        }

@router.get("/api/echo/git/status")
async def get_git_status():
    """Simple git status endpoint without complex imports"""
    try:
        import subprocess
        import os

        # Change to the echo brain directory
        os.chdir("/opt/tower-echo-brain")

        # Get basic git status
        result = subprocess.run(["git", "status", "--porcelain"],
                              capture_output=True, text=True, timeout=10)

        if result.returncode != 0:
            return {"error": "Not a git repository or git command failed"}

        # Get branch name
        branch_result = subprocess.run(["git", "branch", "--show-current"],
                                     capture_output=True, text=True, timeout=5)
        branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown"

        # Parse status
        lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
        modified = []
        untracked = []
        staged = []

        for line in lines:
            if line:
                status = line[:2]
                file_path = line[3:]

                if status.startswith('??'):
                    untracked.append(file_path)
                elif status[0] in 'MADRC':
                    staged.append(file_path)
                elif status[1] in 'MADRC':
                    modified.append(file_path)

        return {
            "status": "success",
            "repository": "/opt/tower-echo-brain",
            "branch": branch,
            "modified_files": modified,
            "untracked_files": untracked,
            "staged_files": staged,
            "is_clean": len(lines) == 0,
            "git_available": True
        }

    except subprocess.TimeoutExpired:
        return {"error": "Git command timed out"}
    except Exception as e:
        return {"error": f"Git status error: {str(e)}", "git_available": False}

# ============= ROOT LEVEL ENDPOINTS FOR MONITORING =============

@router.get("/health")
async def root_health_check():
    """Root health check for monitoring tools"""
    return {
        "status": "healthy",
        "uptime_seconds": int(time_module.time() - _startup_time),
        "version": "1.0.0",
        "service": "echo-brain"
    }

@router.get("/ready")
async def readiness_check():
    """Readiness probe for kubernetes/monitoring"""
    return {"ready": True, "service": "echo-brain"}

@router.get("/alive")
async def liveness_check():
    """Liveness probe"""
    return {"alive": True}

@router.get("/metrics")
async def get_metrics():
    """Basic metrics endpoint"""
    return {
        "uptime_seconds": int(time_module.time() - _startup_time),
        "endpoints_registered": 20,
        "status": "operational"
    }
