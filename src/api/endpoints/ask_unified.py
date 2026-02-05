"""
Unified Ask Endpoint - Uses the unified knowledge layer
This replaces the fragmented ask implementation
"""
from typing import Dict, Any
from datetime import datetime
import logging
import httpx

logger = logging.getLogger(__name__)

async def ask_unified(request: Dict[str, Any]):
    """Main Q&A endpoint using unified knowledge layer"""
    question = request.get("question", "")
    use_context = request.get("use_context", True)
    verbose = request.get("verbose", True)

    # Use unified knowledge layer
    from src.core.unified_knowledge import get_unified_knowledge
    knowledge = get_unified_knowledge()

    debug_info = {
        "question": question,
        "steps": [],
        "search_terms": []
    }

    try:
        if use_context:
            # Extract search terms for debugging
            search_terms = knowledge.extract_search_terms(question)
            debug_info["search_terms"] = search_terms
            debug_info["steps"].append(f"Extracted search terms: {search_terms}")

            # Get unified context from all sources
            context = await knowledge.get_context(
                query=question,
                max_facts=5,
                max_vectors=3,
                max_conversations=3
            )

            debug_info["steps"].append(
                f"Retrieved {len(context['facts'])} facts, "
                f"{len(context['vectors'])} vectors, "
                f"{len(context['conversations'])} conversations"
            )

            # Format for LLM
            enhanced_prompt = knowledge.format_for_llm(context, question)

            # Track sources
            sources = []
            for fact in context['facts']:
                sources.append({
                    "type": "fact",
                    "content": fact.content[:100],
                    "confidence": fact.confidence
                })
            for vec in context['vectors']:
                sources.append({
                    "type": "vector",
                    "content": vec.content[:100],
                    "confidence": vec.confidence
                })
            for conv in context['conversations']:
                sources.append({
                    "type": "conversation",
                    "content": conv.content[:100],
                    "role": conv.metadata.get('role', 'unknown')
                })

        else:
            enhanced_prompt = question
            sources = []
            context = None

        # Send to Ollama
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "mistral:7b",
                    "prompt": enhanced_prompt,
                    "stream": False
                }
            )

            answer = response.json().get("response", "")

        result = {
            "answer": answer,
            "model": "mistral:7b",
            "context_used": use_context and context is not None,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        }

        if verbose and context:
            result["debug"] = {
                **debug_info,
                "total_sources": context.get('total_sources', 0),
                "prompt_length": len(enhanced_prompt)
            }

        return result

    except Exception as e:
        logger.error(f"Error in ask_unified: {e}")
        return {
            "answer": f"Error processing request: {str(e)}",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }