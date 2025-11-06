#!/usr/bin/env python3
"""
Core Echo Brain query and conversation API routes
"""
import asyncio
import json
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks

# Import models and services
from src.db.models import QueryRequest, QueryResponse
from src.db.database import database
from src.core.intelligence import intelligence_router
from src.services.conversation import conversation_manager

# Import external modules
from echo_brain_thoughts import echo_brain

# Import agentic persona for collaborative interaction
try:
    from src.core.agentic_persona import agentic_persona
except ImportError:
    agentic_persona = None

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/api/echo/query", response_model=QueryResponse)
@router.post("/api/echo/chat", response_model=QueryResponse)
async def query_echo(request: QueryRequest):
    """Main query endpoint with intelligent routing and conversation management"""
    start_time = time.time()

    # Generate conversation ID if not provided
    if not request.conversation_id:
        request.conversation_id = str(uuid.uuid4())

    logger.info(f"🧠 Query received: {request.query[:100]}...")
    logger.info(f"🔧 Request type: {request.request_type}")

    # CHECK REQUEST TYPE FIRST - Route to appropriate handler
    if request.request_type == 'system_command':
        logger.info("🚨 ROUTING TO SYSTEM COMMAND EXECUTION")
        from src.utils.helpers import safe_executor
        try:
            result = await safe_executor.execute(request.query, allow_all=False)
            processing_time = time.time() - start_time

            return QueryResponse(
                response=result["output"] if result["success"] else f"Command failed: {result.get('error', 'Unknown error')}",
                model_used="safe_executor",
                intelligence_level="system_command",
                processing_time=processing_time,
                escalation_path=[f"system_command:{request.query[:20]}"],
                conversation_id=request.conversation_id,
                intent="system_command",
                confidence=1.0,
                requires_clarification=False,
                clarifying_questions=[]
            )
        except Exception as e:
            logger.error(f"System command execution failed: {e}")
            processing_time = time.time() - start_time
            return QueryResponse(
                response=f"System command failed: {str(e)}",
                model_used="safe_executor",
                intelligence_level="error",
                processing_time=processing_time,
                escalation_path=["system_command_error"],
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
        # Get conversation context
        conversation_context = await conversation_manager.get_conversation_context(request.conversation_id)

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

        # Check if clarification is needed
        needs_clarification = should_ask_user or conversation_manager.needs_clarification(intent, confidence, request.query)

        if needs_clarification:
            if contextual_question:
                clarifying_questions = [contextual_question]
                response_text = contextual_question
            else:
                clarifying_questions = conversation_manager.get_clarifying_questions(intent, request.query)
                response_text = "I'd like to better understand your request. Could you help clarify?"

            processing_time = time.time() - start_time

            response = QueryResponse(
                response=response_text,
                model_used="conversation_manager",
                intelligence_level="clarification",
                processing_time=processing_time,
                escalation_path=["clarification_needed"],
                requires_clarification=True,
                clarifying_questions=clarifying_questions,
                conversation_id=request.conversation_id,
                intent=intent,
                confidence=confidence
            )

            conversation_manager.update_conversation(
                request.conversation_id, request.query, intent, response.response, True
            )

            return response

        # Handle capability intents automatically
        capability_intents = ['anime_generation', 'service_testing', 'service_debugging', 'service_monitoring', 'agent_delegation', 'inter_service_communication', 'image_generation', 'voice_generation', 'music_generation', 'code_review', 'code_refactor', 'code_modification']
        if intent in capability_intents:
            response = await handle_capability_intent(intent, intent_params, request, request.conversation_id, start_time)

            conversation_manager.update_conversation(
                request.conversation_id, request.query, intent, response.response, False
            )

            try:
                await database.log_interaction(
                    request.query, response.response, response.model_used,
                    response.processing_time, response.escalation_path,
                    request.conversation_id, request.user_id, intent, confidence,
                    complexity_score, tier
                )
            except Exception as e:
                logger.error(f"❌ log_interaction FAILED: {e}")

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

        # Use cognitive model selection if available
        if selected_model:
            logger.info(f"🧠 Using cognitively selected model: {selected_model} - {selection_reason}")
            result = await intelligence_router.query_model(selected_model, request.query)
            if result["success"]:
                result["intelligence_level"] = intelligence_level
                result["escalation_path"] = [f"cognitive_selection:{selected_model}"]
                result["decision_reason"] = selection_reason
            else:
                logger.warning(f"⚠️ Cognitive model {selected_model} failed, falling back to progressive escalation")
                result = await intelligence_router.progressive_escalation(request.query, context)
        else:
            result = await intelligence_router.progressive_escalation(request.query, context)

        if result["success"]:
            processing_time = time.time() - start_time

            response = QueryResponse(
                response=result["response"],
                model_used=result["model"],
                intelligence_level=result.get("intelligence_level", "standard"),
                processing_time=processing_time,
                escalation_path=result.get("escalation_path", []),
                conversation_id=request.conversation_id,
                intent=intent,
                confidence=confidence
            )

            conversation_manager.update_conversation(
                request.conversation_id, request.query, intent, response.response, False
            )

            try:
                await database.log_interaction(
                    request.query, response.response, response.model_used,
                    response.processing_time, response.escalation_path,
                    request.conversation_id, request.user_id, intent, confidence,
                    complexity_score, tier
                )
            except Exception as e:
                logger.error(f"❌ log_interaction FAILED: {e}")

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

@router.get("/api/echo/thoughts/{thought_id}")
async def get_thought(thought_id: str):
    """Get specific thought by ID"""
    try:
        thought = echo_brain.get_thought(thought_id)
        if thought:
            return {"thought": thought, "timestamp": datetime.now().isoformat()}
        else:
            raise HTTPException(status_code=404, detail="Thought not found")
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

async def handle_capability_intent(intent: str, intent_params: dict, request: QueryRequest, conversation_id: str, start_time: float):
    """Handle capability-based intents"""
    from src.services.safe_executor import safe_executor

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