#!/usr/bin/env python3
"""
Fix the ask endpoint to use unified knowledge layer
Run this to patch the echo_main_router.py file
"""
import sys

# Read the current file
with open('/opt/tower-echo-brain/src/api/endpoints/echo_main_router.py', 'r') as f:
    content = f.read()

# Find and replace the ask function
new_ask_function = '''@router.post("/ask")
async def ask(request: Dict[str, Any]):
    """Main Q&A endpoint - WITH UNIFIED KNOWLEDGE LAYER"""
    question = request.get("question", "")
    use_context = request.get("use_context", True)
    verbose = request.get("verbose", True)

    # Use unified knowledge layer
    from src.core.unified_knowledge import get_unified_knowledge
    knowledge = get_unified_knowledge()

    debug_info = {"question": question, "steps": [], "search_terms": []}

    try:
        if use_context:
            # Get unified context from all sources
            context = await knowledge.get_context(
                query=question,
                max_facts=5,
                max_vectors=3,
                max_conversations=3
            )

            # Track what was found
            debug_info["search_terms"] = knowledge.extract_search_terms(question)
            debug_info["steps"].append(
                f"Found {len(context['facts'])} facts, "
                f"{len(context['vectors'])} vectors, "
                f"{len(context['conversations'])} conversations"
            )

            # Build enhanced prompt with unified context
            enhanced_prompt = knowledge.format_for_llm(context, question)

            # Track sources for transparency
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
            # No context requested
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

        # Build response
        result = {
            "answer": answer,
            "model": "mistral:7b",
            "context_used": use_context and context is not None,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        }

        # Add debug info if verbose
        if verbose and context:
            result["debug"] = {
                **debug_info,
                "total_sources": context.get('total_sources', 0),
                "prompt_length": len(enhanced_prompt) if use_context else 0
            }

        return result

    except Exception as e:
        logger.error(f"Error in ask endpoint: {e}")
        return {
            "answer": f"Error processing request: {str(e)}",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }'''

# Find the start of the ask function
start_marker = '@router.post("/ask")'
start_pos = content.find(start_marker)

if start_pos == -1:
    print("ERROR: Could not find ask endpoint")
    sys.exit(1)

# Find the next @router or the end of the function
next_router_pos = content.find('@router.', start_pos + len(start_marker))
if next_router_pos == -1:
    next_router_pos = len(content)

# Replace the function
new_content = content[:start_pos] + new_ask_function + '\n\n' + content[next_router_pos:]

# Write back
with open('/opt/tower-echo-brain/src/api/endpoints/echo_main_router.py', 'w') as f:
    f.write(new_content)

print("✅ Ask endpoint updated to use unified knowledge layer")
print("Now restart the service to test: sudo systemctl restart tower-echo-brain")