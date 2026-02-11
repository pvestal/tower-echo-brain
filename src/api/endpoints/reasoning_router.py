"""
Echo Brain Reasoning API with proper vector retrieval
Uses ParallelRetriever to search both Qdrant and PostgreSQL
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging
import time
import httpx
import os

from src.context_assembly.retriever import ParallelRetriever

router = APIRouter(prefix="", tags=["reasoning"])
logger = logging.getLogger(__name__)

# Initialize retriever
retriever = ParallelRetriever()

class AskRequest(BaseModel):
    """Request for asking Echo Brain a question"""
    question: str
    context: Optional[str] = None

@router.post("/ask")
async def ask_question(request: AskRequest):
    """
    Ask Echo Brain a question and get a synthesized response using proper retrieval
    """
    try:
        logger.info(f"🤖 Processing question: {request.question[:100]}...")
        start_time = time.time()

        # Ensure retriever is initialized
        if not retriever.pg_pool:
            await retriever.initialize()

        # Use ParallelRetriever to search all sources
        retrieval_result = await retriever.retrieve(
            query=request.question,
            max_results=10
        )

        # Extract sources for context
        sources = retrieval_result.get("sources", [])

        # Build context from retrieved sources
        context_pieces = []
        for source in sources[:5]:  # Use top 5 sources
            content = source.get("content", "")
            source_type = source.get("source", "unknown")
            score = source.get("score", 0)

            # Prioritize authoritative sources
            metadata = source.get("metadata", {})
            if metadata.get("authoritative", False):
                # Put authoritative sources first
                context_pieces.insert(0, f"[AUTHORITATIVE - {source_type}]: {content}")
            else:
                context_pieces.append(f"[{source_type} - score: {score:.2f}]: {content}")

        context = "\n\n".join(context_pieces)

        # Generate response using Ollama
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        prompt = f"""You are Echo Brain, Patrick's personal AI assistant running on the Tower server.

Based on the following retrieved context from your knowledge base:

{context}

Question: {request.question}

Provide a specific, accurate answer based on the context. If the context contains authoritative information (marked as [AUTHORITATIVE]), prioritize that information. Be concise and factual."""

        # Call Ollama for generation
        async with httpx.AsyncClient(timeout=60.0) as client:
            llm_start = time.time()
            resp = await client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": "mistral:7b",
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3  # Lower temperature for factual responses
                }
            )
            llm_time = (time.time() - llm_start) * 1000

            answer = resp.json().get("response", "Error generating response")

        total_time = (time.time() - start_time) * 1000

        # Format sources for response
        source_list = []
        for source in sources[:5]:
            source_info = source.get("source", "unknown")
            if source.get("metadata", {}).get("source_path"):
                source_info = source["metadata"]["source_path"]
            source_list.append(source_info)

        return {
            "question": request.question,
            "answer": answer,
            "confidence": max(s.get("score", 0) for s in sources) if sources else 0.1,
            "memories_used": len(sources),
            "sources": source_list,
            "reasoning_time_ms": int(total_time),
            "model_used": "mistral:7b",
            # Enhanced metrics
            "domain": retrieval_result.get("domain", "unknown"),
            "domain_confidence": retrieval_result.get("domain_confidence", 0),
            "total_found": retrieval_result.get("total_found", 0),
            "retrieval_ms": retrieval_result.get("retrieval_ms", 0),
            "llm_generation_time_ms": int(llm_time),
            "embedding_model_used": "nomic-embed-text"
        }

    except Exception as e:
        logger.error(f"Failed to process question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.on_event("startup")
async def startup():
    """Initialize retriever on startup"""
    await retriever.initialize()
    logger.info("Reasoning router initialized with ParallelRetriever")

@router.on_event("shutdown")
async def shutdown():
    """Clean shutdown"""
    await retriever.shutdown()
    logger.info("Reasoning router shutdown complete")