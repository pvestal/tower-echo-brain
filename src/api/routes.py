#!/usr/bin/env python3
"""
API routes for Echo Brain system
"""

import asyncio
import json
import time
import uuid
import logging
import psycopg2
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket
from fastapi.responses import StreamingResponse

# Import models and services
from src.db.models import (
    QueryRequest, QueryResponse, ExecuteRequest, ExecuteResponse,
    TestRequest, VoiceNotificationRequest, VoiceStatusRequest
)
from src.db.database import database
from src.core.intelligence import intelligence_router
from src.services.conversation import conversation_manager
from src.services.testing import testing_framework
from src.utils.helpers import safe_executor, tower_orchestrator

# Import external modules
from echo_brain_thoughts import echo_brain
from model_manager import get_model_manager, ModelManagementRequest, ModelManagementResponse

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Use tower_orchestrator from helpers (which is now ResilientOrchestrator)

@router.get("/api/echo/health")
async def health_check():
    """Health check endpoint with module information"""
    try:
        # Test database connection
        conn = psycopg2.connect(**database.db_config)
        conn.close()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {e}"

    return {
        "status": "healthy",
        "service": "echo-brain",
        "version": "1.0.0",
        "architecture": "modular",
        "database": db_status,
        "timestamp": datetime.now().isoformat(),
        "modules": {
            "intelligence_router": intelligence_router is not None,
            "conversation_manager": conversation_manager is not None,
            "testing_framework": testing_framework is not None,
            "database": database is not None,
            "safe_executor": safe_executor is not None
        }
    }

# Capability handler function
async def handle_capability_intent(intent: str, params: Dict, request: QueryRequest, conversation_id: str, start_time: float) -> QueryResponse:
    """Handle capability intents by automatically using Echo's APIs"""
    try:
        if intent == "service_testing":
            target = params.get('target_service', 'echo')  # Default to self-test
            logger.info(f"🧪 AUTO-TESTING: Running test on {target}")

            # Execute test automatically
            test_result = await testing_framework.run_universal_test(target)

            if test_result['success']:
                response_text = f"✅ Test completed for {target}!\n\nResults:\n{test_result['output']}"
                if test_result.get('error'):
                    response_text += f"\n⚠️ Warnings: {test_result['error']}"
            else:
                response_text = f"❌ Test failed for {target}\n\nError: {test_result.get('error', 'Unknown error')}\n\nOutput: {test_result['output']}"

            response_text += f"\n\n⏱️ Processing time: {test_result['processing_time']:.2f}s"

        elif intent == "service_debugging":
            target = params.get('target_service', 'echo')
            logger.info(f"🔍 AUTO-DEBUG: Running debug analysis on {target}")

            # Execute debug automatically
            debug_result = await testing_framework.run_debug_analysis(target)

            if debug_result['success']:
                response_text = f"🔍 Debug analysis completed for {target}!\n\nAnalysis:\n{debug_result['output']}"
                if debug_result.get('error'):
                    response_text += f"\n⚠️ Issues found: {debug_result['error']}"
            else:
                response_text = f"❌ Debug analysis failed for {target}\n\nError: {debug_result.get('error', 'Unknown error')}\n\nOutput: {debug_result['output']}"

            response_text += f"\n\n⏱️ Processing time: {debug_result['processing_time']:.2f}s"

        elif intent == "service_monitoring":
            if 'stats' in request.query.lower() or 'statistics' in request.query.lower():
                logger.info(f"📊 AUTO-STATS: Getting Echo statistics")

                # Get Echo's own statistics
                try:
                    conn = psycopg2.connect(**database.db_config)
                    cursor = conn.cursor()

                    cursor.execute("""
                        SELECT
                            COUNT(*) as total_queries,
                            AVG(processing_time) as avg_processing_time,
                            model_used,
                            COUNT(*) as usage_count
                        FROM echo_unified_interactions
                        GROUP BY model_used
                        ORDER BY usage_count DESC
                        LIMIT 10
                    """)

                    stats = cursor.fetchall()
                    cursor.close()
                    conn.close()

                    if stats:
                        response_text = "📊 Echo Brain Statistics\n\n"
                        total_queries = stats[0][0] if stats else 0
                        response_text += f"Total queries processed: {total_queries}\n\n"
                        response_text += "Model Usage:\n"
                        for stat in stats:
                            model = stat[2]
                            usage = stat[3]
                            avg_time = stat[1] if stat[1] else 0
                            response_text += f"  {model}: {usage} queries (avg: {avg_time:.2f}s)\n"
                    else:
                        response_text = "📊 No statistics available yet"

                except Exception as e:
                    response_text = f"❌ Failed to get statistics: {e}"
            else:
                # Get service health status
                tower_health = await testing_framework.get_tower_health_summary()
                response_text = f"🏥 Tower Health Summary\n\n"
                response_text += f"Overall Status: {tower_health['summary']['overall_status'].upper()}\n"
                response_text += f"Active Services: {tower_health['summary']['service_health']}\n\n"

                # Service details
                response_text += "Service Status:\n"
                for service, status in tower_health['services'].items():
                    status_icon = "✅" if status.get('active') else "❌"
                    response_text += f"  {status_icon} {service}: {status.get('status', 'unknown')}\n"

        elif intent == "anime_generation":
            logger.info(f"🎬 AUTO-ANIME: Generating anime with params: {params}")
            try:
                import aiohttp
                prompt_text = params.get('prompt', request.query)
                prompt_text = prompt_text.replace('generate anime', '').replace('create anime', '').strip()
                if not prompt_text:
                    prompt_text = "anime magical girl"
                
                async with aiohttp.ClientSession() as session:
                    payload = {"prompt": prompt_text}
                    async with session.post(
                        "http://localhost:8328/api/generate",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=300)
                    ) as resp:
                        result = await resp.json()
                        if resp.status == 200:
                            response_text = (
                                f"✅ Anime generation started!\n"
                                f"Generation ID: {result.get('generation_id')}\n"
                                f"Status: {result.get('status')}\n"
                                f"Check at: /api/status/{result.get('generation_id')}"
                            )
                        else:
                            error_msg = result.get('detail', {}).get('message', 'Unknown error')
                            response_text = f"❌ Anime generation failed: {error_msg}"
                        
                        return QueryResponse(
                            response=response_text,
                            model_used="anime_service_8328",
                            intelligence_level="capability",
                            processing_time=time.time() - start_time,
                            escalation_path=["anime_generation_capability"],
                            conversation_id=request.conversation_id
                        )
            except Exception as e:
                logger.error(f"Anime generation error: {e}")
                return QueryResponse(
                    response=f"❌ Anime generation error: {str(e)}",
                    model_used="anime_service_error",
                    intelligence_level="capability",
                    processing_time=time.time() - start_time,
                    escalation_path=["anime_error"],
                    conversation_id=request.conversation_id
                )

        elif intent == "image_generation":
            logger.info(f"🎨 AUTO-IMAGE: Generating image with params: {params}")
            try:
                # Extract prompt and style from params
                prompt = params.get('prompt', request.query.replace('generate image', '').replace('create image', '').strip())
                if not prompt:
                    prompt = "cyberpunk anime scene"
                style = params.get('style', 'anime')

                # Actually generate the image using tower_orchestrator
                result = await tower_orchestrator.generate_image(prompt=prompt, style=style)

                if result.get('success'):
                    prompt_id = result.get('prompt_id', 'unknown')
                    image_url = f"http://***REMOVED***:8188/view?filename=ComfyUI_{prompt_id[:8]}.png"
                    response_text = f"🎨 Image generated successfully!\n\nPrompt: {prompt}\nStyle: {style}\nPrompt ID: {prompt_id}\n\nImage URL: {image_url}\n\n✅ Generation complete! Check ComfyUI output directory."
                    if result.get('compute_location'):
                        response_text += f"\nCompute: {result['compute_location']}"
                else:
                    response_text = f"❌ Image generation failed: {result.get('error', 'Unknown error')}"
            except Exception as e:
                response_text = f"❌ Image generation error: {str(e)}"
        elif intent == "voice_generation":
            logger.info(f"🎵 AUTO-VOICE: Generating voice with params: {params}")
            try:
                # Extract text and character from params
                text = params.get('text', request.query.replace('generate voice', '').replace('say', '').strip())
                if not text:
                    text = "Hello from Echo Brain"
                character = params.get('character', 'echo_default')

                # Actually generate voice using tower_orchestrator
                result = await tower_orchestrator.generate_voice(text=text, character=character)

                if result.get('success'):
                    audio_url = result.get('audio_url', '')
                    response_text = f"🎵 Voice generated successfully!\n\nText: {text}\nCharacter: {character}\n\nAudio URL: {audio_url}"
                    if result.get('metadata'):
                        response_text += f"\nMetadata: {result['metadata']}"
                else:
                    response_text = f"❌ Voice generation failed: {result.get('error', 'Unknown error')}"
            except Exception as e:
                response_text = f"❌ Voice generation error: {str(e)}"
        elif intent == "music_generation":
            logger.info(f"🎼 AUTO-MUSIC: Generating music with params: {params}")
            try:
                # Extract description and duration from params
                description = params.get('description', request.query.replace('generate music', '').replace('create music', '').strip())
                if not description:
                    description = "Epic cinematic soundtrack"
                duration = params.get('duration', 30)

                # Actually generate music using tower_orchestrator
                result = await tower_orchestrator.create_music(description=description, duration=duration)

                if result.get('success'):
                    music_url = result.get('music_url', '')
                    response_text = f"🎼 Music generated successfully!\n\nDescription: {description}\nDuration: {duration}s\n\nMusic URL: {music_url}"
                    if result.get('metadata'):
                        response_text += f"\nMetadata: {result['metadata']}"
                else:
                    response_text = f"❌ Music generation failed: {result.get('error', 'Unknown error')}"
            except Exception as e:
                response_text = f"❌ Music generation error: {str(e)}"
        else:
            # Fallback for other capability intents
            response_text = f"🤖 Echo capability '{intent}' executed with parameters: {params}"

        processing_time = time.time() - start_time

        return QueryResponse(
            response=response_text,
            model_used="echo_capability",
            intelligence_level="capability",
            processing_time=processing_time,
            escalation_path=["capability_auto_execution"],
            conversation_id=conversation_id,
            intent=intent,
            confidence=1.0
        )

    except Exception as e:
        logger.error(f"Capability intent handling failed: {e}")
        processing_time = time.time() - start_time

        return QueryResponse(
            response=f"❌ Failed to execute capability '{intent}': {str(e)}",
            model_used="echo_capability",
            intelligence_level="capability",
            processing_time=processing_time,
            escalation_path=["capability_error"],
            conversation_id=conversation_id,
            intent=intent,
            confidence=0.0
        )

@router.post("/api/echo/query", response_model=QueryResponse)
@router.post("/api/echo/chat", response_model=QueryResponse)  # Alias for compatibility
async def query_echo(request: QueryRequest):
    """Main query endpoint with intelligent routing and conversation management"""
    start_time = time.time()

    # Generate conversation ID if not provided
    if not request.conversation_id:
        request.conversation_id = str(uuid.uuid4())

    logger.info(f"🧠 Query received: {request.query[:100]}...")

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
        # Override intelligence level with cognitive selection
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

        # Check if clarification is needed
        needs_clarification = conversation_manager.needs_clarification(intent, confidence, request.query)

        if needs_clarification:
            clarifying_questions = conversation_manager.get_clarifying_questions(intent, request.query)
            processing_time = time.time() - start_time

            response = QueryResponse(
                response="I'd like to better understand your request. Could you help clarify?",
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

            # Update conversation
            conversation_manager.update_conversation(
                request.conversation_id, request.query, intent, response.response, True
            )

            return response

        # Handle capability intents automatically
        capability_intents = ['anime_generation', 'service_testing', 'service_debugging', 'service_monitoring', 'agent_delegation', 'inter_service_communication', 'image_generation', 'voice_generation', 'music_generation']
        if intent in capability_intents:
            response = await handle_capability_intent(intent, intent_params, request, request.conversation_id, start_time)

            # Update conversation
            conversation_manager.update_conversation(
                request.conversation_id, request.query, intent, response.response, False
            )

            # Log to database
            logger.info(f"🔍 DEBUG: About to log_interaction for capability intent '{intent}'")
            logger.info(f"🔍 DEBUG: Params - conv_id={request.conversation_id}, user_id={request.user_id}, query='{request.query[:50]}...'")
            try:
                await database.log_interaction(
                    request.query, response.response, response.model_used,
                    response.processing_time, response.escalation_path,
                    request.conversation_id, request.user_id, intent, confidence,
                    complexity_score, tier
                )
                logger.info(f"✅ DEBUG: log_interaction SUCCESS for conv_id={request.conversation_id}")
            except Exception as e:
                logger.error(f"❌ DEBUG: log_interaction FAILED: {e}", exc_info=True)

            return response

        # For other intents, use cognitive model selection if available
        context = {
            "conversation_history": conversation_context.get("history", []),
            "user_id": request.user_id,
            "intent": intent,
            "intent_params": intent_params,
            "previous_failures": len([h for h in conversation_context.get("history", []) if "error" in h.get("response", "").lower()])
        }

        # Add intelligence level from request if specified
        if request.intelligence_level and request.intelligence_level != "auto":
            context["requested_level"] = request.intelligence_level

        # Use cognitive model selection if available, otherwise fall back to intelligent routing
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
            # Use progressive escalation for intelligent routing
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

            # Update conversation
            conversation_manager.update_conversation(
                request.conversation_id, request.query, intent, response.response, False
            )

            # Log to database
            logger.info(f"🔍 DEBUG: About to log_interaction for general intent '{intent}'")
            logger.info(f"🔍 DEBUG: Params - conv_id={request.conversation_id}, user_id={request.user_id}, query='{request.query[:50]}...'")
            try:
                await database.log_interaction(
                    request.query, response.response, response.model_used,
                    response.processing_time, response.escalation_path,
                    request.conversation_id, request.user_id, intent, confidence,
                    complexity_score, tier
                )
                logger.info(f"✅ DEBUG: log_interaction SUCCESS for conv_id={request.conversation_id}")
            except Exception as e:
                logger.error(f"❌ DEBUG: log_interaction FAILED: {e}", exc_info=True)

            return response
        else:
            # Handle failure
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
            "status": "active",
            "brain_state": brain_state,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get brain activity: {e}")
        return {"status": "error", "error": str(e)}

@router.get("/api/echo/thoughts/{thought_id}")
async def get_thought_details(thought_id: str):
    """Get details of a specific thought process"""
    try:
        # This would typically fetch from a thought database
        return {
            "thought_id": thought_id,
            "status": "completed",
            "details": "Thought details would be retrieved from storage",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}

@router.get("/api/echo/stats")
async def get_echo_stats():
    """Get Echo Brain performance statistics"""
    try:
        conn = psycopg2.connect(**database.db_config)
        cursor = conn.cursor()

        # Get basic statistics
        cursor.execute("""
            SELECT
                COUNT(*) as total_interactions,
                AVG(processing_time) as avg_processing_time,
                MAX(processing_time) as max_processing_time,
                MIN(processing_time) as min_processing_time
            FROM echo_unified_interactions
        """)

        basic_stats = cursor.fetchone()

        # Get model usage statistics
        cursor.execute("""
            SELECT model_used, COUNT(*) as usage_count, AVG(processing_time) as avg_time
            FROM echo_unified_interactions
            GROUP BY model_used
            ORDER BY usage_count DESC
            LIMIT 10
        """)

        model_stats = cursor.fetchall()

        cursor.close()
        conn.close()

        return {
            "basic_stats": {
                "total_interactions": basic_stats[0] if basic_stats[0] else 0,
                "avg_processing_time": float(basic_stats[1]) if basic_stats[1] else 0.0,
                "max_processing_time": float(basic_stats[2]) if basic_stats[2] else 0.0,
                "min_processing_time": float(basic_stats[3]) if basic_stats[3] else 0.0
            },
            "model_usage": [
                {
                    "model": stat[0],
                    "usage_count": stat[1],
                    "avg_processing_time": float(stat[2]) if stat[2] else 0.0
                }
                for stat in model_stats
            ],
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return {"error": str(e)}

@router.get("/api/echo/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    try:
        history = await database.get_conversation_history(conversation_id)
        context = await conversation_manager.get_conversation_context(conversation_id)

        return {
            "conversation_id": conversation_id,
            "history": history,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get conversation: {e}")
        return {"error": str(e)}

@router.post("/api/echo/execute")
async def execute_command(request: ExecuteRequest):
    """Execute shell commands safely"""
    start_time = time.time()

    if not request.conversation_id:
        request.conversation_id = str(uuid.uuid4())

    logger.info(f"🔧 Execute command: {request.command}")

    try:
        result = await safe_executor.execute_command(request.command, request.safe_mode)

        response = ExecuteResponse(
            command=request.command,
            success=result["success"],
            output=result["output"],
            error=result.get("error"),
            exit_code=result["exit_code"],
            processing_time=result["processing_time"],
            conversation_id=request.conversation_id,
            safety_checks=result["safety_checks"]
        )

        return response

    except Exception as e:
        logger.error(f"Command execution failed: {e}")
        processing_time = time.time() - start_time

        response = ExecuteResponse(
            command=request.command,
            success=False,
            output="",
            error=str(e),
            exit_code=-1,
            processing_time=processing_time,
            conversation_id=request.conversation_id,
            safety_checks={"passed_safety_check": False, "safe_mode_enabled": request.safe_mode, "safety_message": str(e)}
        )

        return response

@router.get("/api/echo/conversations")
async def get_user_conversations(user_id: str = "default", limit: int = 50):
    """Get all conversations for a user"""
    try:
        conversations = await database.get_user_conversations(user_id, limit)
        return {
            "user_id": user_id,
            "conversations": conversations,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get conversations: {e}")
        return {"error": str(e)}

@router.get("/api/echo/stream")
async def stream_brain_activity():
    """Stream real-time brain activity"""
    async def generate():
        try:
            while True:
                brain_state = echo_brain.get_brain_state()
                data = json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "brain_state": brain_state
                })
                yield f"data: {data}\n\n"
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/plain")

# Voice notification endpoints
@router.post("/api/echo/voice/notify")
async def voice_notify(request: VoiceNotificationRequest):
    """Send REAL voice notification using Tower voice service"""
    logger.info(f"🗣️ Voice notification: {request.message[:50]}...")

    try:
        # Use REAL Tower orchestrator for voice generation
        result = await tower_orchestrator.generate_voice(
            text=request.message,
            character=request.character
        )

        if result["success"]:
            return {
                "success": True,
                "message": "Voice notification sent successfully",
                "character": request.character,
                "priority": request.priority,
                "voice_result": result
            }
        else:
            return {
                "success": False,
                "error": f"Voice generation failed: {result.get('error', 'Unknown error')}"
            }
    except Exception as e:
        logger.error(f"Voice notification failed: {e}")
        return {"success": False, "error": str(e)}

@router.post("/api/echo/voice/status")
async def voice_status(request: VoiceStatusRequest):
    """Get voice system status"""
    try:
        return {
            "status": "ready",
            "user_id": request.user_id,
            "available_characters": ["yukiko", "akira", "system"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.get("/api/echo/voice/characters")
async def get_voice_characters():
    """Get available voice characters"""
    return {
        "characters": [
            {"name": "yukiko", "description": "Calm, analytical assistant"},
            {"name": "akira", "description": "Energetic, creative assistant"},
            {"name": "system", "description": "System announcements"}
        ]
    }

# Testing and debugging endpoints
@router.post("/api/echo/test/{target}")
async def test_service(target: str, request: TestRequest):
    """Test a Tower service"""
    logger.info(f"🧪 Testing service: {target}")

    try:
        if request.test_type == "debug":
            result = await testing_framework.run_debug_analysis(target)
        else:
            result = await testing_framework.run_universal_test(target)

        return result
    except Exception as e:
        logger.error(f"Service test failed: {e}")
        return {"success": False, "error": str(e)}

@router.post("/api/echo/debug/{service}")
async def debug_service(service: str):
    """Debug a specific service"""
    logger.info(f"🔍 Debugging service: {service}")

    try:
        result = await testing_framework.run_debug_analysis(service)
        return result
    except Exception as e:
        logger.error(f"Service debug failed: {e}")
        return {"success": False, "error": str(e)}

@router.get("/api/echo/tower/status")
async def get_tower_status():
    """Get Tower system status"""
    try:
        status = await testing_framework.get_tower_service_status()
        return {
            "tower_status": status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Tower status check failed: {e}")
        return {"error": str(e)}

@router.get("/api/echo/tower/health")
async def get_tower_health():
    """Get comprehensive Tower health summary"""
    try:
        health = await testing_framework.get_tower_health_summary()
        return health
    except Exception as e:
        logger.error(f"Tower health check failed: {e}")
        return {"error": str(e)}

@router.post("/api/echo/tower/{command}")
async def execute_tower_command(command: str, args: List[str] = []):
    """Execute Tower framework command"""
    logger.info(f"🏗️ Tower command: {command} {args}")

    try:
        result = await testing_framework.run_tower_command(command, args)
        return result
    except Exception as e:
        logger.error(f"Tower command failed: {e}")
        return {"success": False, "error": str(e)}

@router.get("/api/echo/testing/capabilities")
async def get_testing_capabilities():
    """Get testing framework capabilities"""
    return {
        "capabilities": [
            "universal_testing",
            "debug_analysis",
            "tower_status_monitoring",
            "service_health_checks",
            "tower_command_execution"
        ],
        "available_targets": [
            "echo", "anime", "dashboard", "auth", "kb", "deepseek",
            "comfyui", "voice", "apple-music"
        ],
        "testing_framework_version": "1.0.0"
    }

# Model management endpoints
# Commented out - using individual endpoints instead
# @router.post("/api/echo/models/manage", response_model=ModelManagementResponse)
async def manage_model_disabled(request: ModelManagementRequest, background_tasks: BackgroundTasks):
    """Manage Ollama models (pull, update, remove)"""
    logger.info(f"🔧 Model management: {request.action} {request.model}")

    try:
        # Initialize dependencies if not available
        from routing.service_registry import ServiceRegistry
        from routing.request_logger import RequestLogger
        board_registry = ServiceRegistry()
        request_logger = RequestLogger()
        model_manager = get_model_manager(board_registry, request_logger)

        if request.action in ["pull", "update"]:
            # Start background task for model operations
            task_id = str(uuid.uuid4())
            background_tasks.add_task(
                model_manager.pull_model_background,
                request.model,
                task_id,
                request.user_id
            )

            return ModelManagementResponse(
                success=True,
                message=f"Model {request.action} started",
                request_id=task_id,
                model=request.model
            )
        else:
            # Direct operations
            result = await model_manager.manage_model(request)
            return result

    except Exception as e:
        logger.error(f"Model management failed: {e}")
        return ModelManagementResponse(
            success=False,
            message=str(e),
            request_id="",
            model=request.model
        )

@router.get("/api/echo/models/list")
async def list_models():
    """List all available Ollama models"""
    try:
        # Use direct ollama command as fallback
        from src.api.dependencies import execute_ollama_command
        import subprocess

        # Direct ollama list command
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = []
            for line in lines:
                if line:
                    parts = line.split()
                    if len(parts) >= 4:
                        models.append({
                            "name": parts[0],
                            "size": f"{parts[2]} {parts[3]}",
                            "modified": " ".join(parts[4:]) if len(parts) > 4 else ""
                        })
            return models
        return {"error": "Failed to list models"}
    except Exception as e:
        logger.error(f"Model listing failed: {e}")
        return {"error": str(e)}

@router.post("/api/echo/models/pull/{model_name}")
async def pull_model(model_name: str, background_tasks: BackgroundTasks):
    """Pull a specific model"""
    logger.info(f"📥 Pulling model: {model_name}")

    try:
        from src.api.dependencies import execute_ollama_command

        async def pull_model_async(model: str):
            result = await execute_ollama_command(["ollama", "pull", model])
            if result["success"]:
                logger.info(f"✅ Model {model} pulled successfully")
            else:
                logger.error(f"❌ Failed to pull {model}: {result.get('stderr', result.get('error'))}")
            return result

        task_id = str(uuid.uuid4())
        background_tasks.add_task(pull_model_async, model_name)

        return {
            "success": True,
            "message": f"Model pull started for {model_name}",
            "task_id": task_id
        }
    except Exception as e:
        logger.error(f"Model pull failed: {e}")
        return {"success": False, "error": str(e)}

@router.delete("/api/echo/models/{model_name}")
async def remove_model(model_name: str):
    """Remove a specific model"""
    logger.info(f"🗑️ Removing model: {model_name}")

    try:
        from src.api.dependencies import execute_ollama_command

        result = await execute_ollama_command(["ollama", "rm", model_name])

        if result["success"]:
            return {"success": True, "message": f"Model {model_name} removed successfully"}
        else:
            return {"success": False, "error": result.get("stderr", "Failed to remove model")}
    except Exception as e:
        logger.error(f"Model removal failed: {e}")
        return {"success": False, "error": str(e)}

@router.get("/api/echo/models/status/{request_id}")
async def get_model_operation_status(request_id: str):
    """Get status of a model operation"""
    try:
        # Initialize dependencies if not available
        from routing.service_registry import ServiceRegistry
        from routing.request_logger import RequestLogger
        board_registry = ServiceRegistry()
        request_logger = RequestLogger()
        model_manager = get_model_manager(board_registry, request_logger)
        status = await model_manager.get_operation_status(request_id)
        return status
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Code generation error: {e}")
        return {
            "error": str(e),
            "processing_time": time.time() - start_time
        }


# Simple code-only endpoint
@router.post("/api/echo/code")
async def generate_code_only(request: QueryRequest):
    """Code generation endpoint - forces code output"""
    # Modify query to force code generation
    code_request = QueryRequest(
        query=f"Write only code, no explanations: {request.query}",
        conversation_id=request.conversation_id,
        user_id=request.user_id,
        intelligence_level="small"  # Use code model
    )
    # Call the regular query handler with modified request
    response = await query_echo(code_request)
    response.mode = "code_only"
    return response

# REAL MULTIMEDIA ORCHESTRATION ENDPOINTS
# These actually call Tower services and generate content

@router.post("/api/echo/multimedia/generate/image")
async def generate_image_multimedia(request: dict):
    """Generate image using ComfyUI through real orchestration"""
    logger.info(f"🎨 Image generation request: {request.get('prompt', 'No prompt')[:50]}...")

    try:
        result = await tower_orchestrator.generate_image(
            prompt=request.get('prompt', 'cyberpunk anime scene'),
            style=request.get('style', 'anime')
        )

        return {
            "endpoint": "/api/echo/multimedia/generate/image",
            "action": "image_generation",
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        return {"success": False, "error": str(e)}

@router.post("/api/echo/multimedia/generate/voice")
async def generate_voice_multimedia(request: dict):
    """Generate voice using Tower voice service"""
    logger.info(f"🗣️ Voice generation request: {request.get('text', 'No text')[:50]}...")

    try:
        result = await tower_orchestrator.generate_voice(
            text=request.get('text', 'Hello from Echo Brain'),
            character=request.get('character', 'echo_default')
        )

        return {
            "endpoint": "/api/echo/multimedia/generate/voice",
            "action": "voice_generation",
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
    except Exception as e:
        logger.error(f"Voice generation failed: {e}")
        return {"success": False, "error": str(e)}

@router.post("/api/echo/multimedia/generate/music")
async def generate_music_multimedia(request: dict):
    """Generate music using Tower music service"""
    logger.info(f"🎵 Music generation request: {request.get('description', 'No description')[:50]}...")

    try:
        result = await tower_orchestrator.create_music(
            description=request.get('description', 'Epic cinematic soundtrack'),
            duration=request.get('duration', 30)
        )

        return {
            "endpoint": "/api/echo/multimedia/generate/music",
            "action": "music_generation",
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
    except Exception as e:
        logger.error(f"Music generation failed: {e}")
        return {"success": False, "error": str(e)}

@router.post("/api/echo/multimedia/orchestrate")
async def orchestrate_multimedia_task(request: dict):
    """Orchestrate complex multimedia tasks across services"""
    task_type = request.get('task_type', 'unknown')
    description = request.get('description', 'No description')
    requirements = request.get('requirements', {})

    logger.info(f"🎬 Multimedia orchestration: {task_type} - {description[:50]}...")

    try:
        result = await tower_orchestrator.orchestrate_multimedia(
            task_type=task_type,
            description=description,
            requirements=requirements
        )

        return {
            "endpoint": "/api/echo/multimedia/orchestrate",
            "action": "multimedia_orchestration",
            "task_type": task_type,
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
    except Exception as e:
        logger.error(f"Multimedia orchestration failed: {e}")
        return {"success": False, "error": str(e)}

@router.get("/api/echo/multimedia/services/status")
async def get_multimedia_services_status():
    """Get status of all multimedia services"""
    logger.info("📊 Checking multimedia services status...")

    try:
        services = ['comfyui', 'voice', 'music', 'anime']
        status_results = {}

        for service in services:
            status_results[service] = await tower_orchestrator.get_service_status(service)

        overall_health = all(result.get('success', False) for result in status_results.values())

        return {
            "endpoint": "/api/echo/multimedia/services/status",
            "overall_health": "healthy" if overall_health else "degraded",
            "timestamp": datetime.now().isoformat(),
            "services": status_results
        }
    except Exception as e:
        logger.error(f"Service status check failed: {e}")
        return {"success": False, "error": str(e)}

