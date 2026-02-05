#!/usr/bin/env python3
"""
Echo Brain Intelligence Engine
This is the THINKING layer that combines memory, context, and reasoning
to generate intelligent responses based on what Echo Brain knows.
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import httpx
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class KnowledgeDomain(Enum):
    """Knowledge domains Echo Brain can reason about"""
    TOWER_SYSTEM = "tower_system"
    ANIME_PRODUCTION = "anime_production"
    ECHO_BRAIN = "echo_brain"
    PROGRAMMING = "programming"
    CONVERSATIONS = "conversations"
    FACTS = "facts"
    UNKNOWN = "unknown"

@dataclass
class MemoryContext:
    """Context retrieved from memory"""
    content: str
    score: float
    source: str
    domain: KnowledgeDomain
    metadata: Dict[str, Any]

@dataclass
class ThoughtProcess:
    """Represents Echo Brain's thinking process"""
    query: str
    domain: KnowledgeDomain
    memories_retrieved: List[MemoryContext]
    confidence_score: float
    reasoning_steps: List[str]
    response: str
    sources_used: List[str]

class IntelligenceEngine:
    """
    The brain that actually THINKS and RESPONDS using memories
    """

    def __init__(self):
        import os
        self.mcp_url = "http://localhost:8309/mcp"
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "mistral:7b")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:latest")
        self.thinking_log = []
        self.active_model = self.ollama_model

    async def think_and_respond(self, query: str, context: Optional[str] = None) -> ThoughtProcess:
        """
        Main thinking process - this is where Echo Brain becomes intelligent
        """
        logger.info(f"🧠 Thinking about: {query}")

        # Step 1: Understand the domain
        domain = self._identify_domain(query)
        logger.info(f"  Domain identified: {domain.value}")

        # Step 2: Retrieve relevant memories (parallel searches for speed)
        memories = await self._retrieve_memories(query, domain)
        logger.info(f"  Retrieved {len(memories)} relevant memories")

        # Step 3: Analyze what we know vs what we don't know
        knowledge_gaps = self._identify_knowledge_gaps(query, memories)

        # Step 4: Build reasoning chain
        reasoning_steps = self._build_reasoning_chain(query, memories, knowledge_gaps)

        # Step 5: Generate response using memories + reasoning
        response = await self._generate_intelligent_response(
            query, memories, reasoning_steps, context
        )

        # Step 6: Calculate confidence
        confidence = self._calculate_confidence(memories, reasoning_steps)

        # Step 7: Extract sources
        sources = list(set([m.source for m in memories[:5]]))  # Top 5 unique sources

        thought = ThoughtProcess(
            query=query,
            domain=domain,
            memories_retrieved=memories,
            confidence_score=confidence,
            reasoning_steps=reasoning_steps,
            response=response,
            sources_used=sources
        )

        # Log the thinking process
        self.thinking_log.append(thought)

        return thought

    def _identify_domain(self, query: str) -> KnowledgeDomain:
        """Identify what domain the query is about"""
        query_lower = query.lower()

        # Domain keywords mapping
        domain_keywords = {
            KnowledgeDomain.TOWER_SYSTEM: [
                "tower", "system", "server", "service", "api", "dashboard"
            ],
            KnowledgeDomain.ANIME_PRODUCTION: [
                "anime", "video", "generation", "comfyui", "framepack", "character",
                "animation", "scene", "render", "workflow"
            ],
            KnowledgeDomain.ECHO_BRAIN: [
                "echo brain", "memory", "ingestion", "vector", "embedding", "qdrant",
                "knowledge", "learning", "mcp"
            ],
            KnowledgeDomain.PROGRAMMING: [
                "code", "function", "class", "python", "javascript", "typescript",
                "bug", "error", "implement", "fix"
            ],
            KnowledgeDomain.CONVERSATIONS: [
                "said", "discussed", "talked", "mentioned", "conversation", "tell me"
            ],
            KnowledgeDomain.FACTS: [
                "fact", "know", "information", "data", "detail", "what is", "who is"
            ]
        }

        # Score each domain
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                domain_scores[domain] = score

        # Return highest scoring domain
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        return KnowledgeDomain.UNKNOWN

    async def _retrieve_memories(self, query: str, domain: KnowledgeDomain) -> List[MemoryContext]:
        """
        Retrieve relevant memories using multiple search strategies
        """
        memories = []

        # Build search queries based on domain
        search_queries = [query]  # Always search the original query

        # Add domain-specific searches
        if domain == KnowledgeDomain.ANIME_PRODUCTION:
            search_queries.extend([
                "anime video generation workflow",
                "comfyui framepack pipeline",
                "character animation tower"
            ])
        elif domain == KnowledgeDomain.TOWER_SYSTEM:
            search_queries.extend([
                "tower system architecture",
                "tower services API",
                "tower dashboard components"
            ])
        elif domain == KnowledgeDomain.ECHO_BRAIN:
            search_queries.extend([
                "echo brain memory ingestion",
                "vector embeddings qdrant",
                "knowledge learning pipeline"
            ])

        # Execute parallel searches
        async with httpx.AsyncClient(timeout=30) as client:
            tasks = []
            for search_query in search_queries[:3]:  # Limit to 3 searches
                task = self._search_memory(client, search_query, limit=5)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            seen_content = set()
            for result_set in results:
                if isinstance(result_set, list):
                    for item in result_set:
                        # Deduplicate by content hash
                        content_hash = hash(item.get("content", "")[:100])
                        if content_hash not in seen_content:
                            seen_content.add(content_hash)
                            memories.append(MemoryContext(
                                content=item.get("content", ""),
                                score=item.get("score", 0),
                                source=item.get("source", "unknown"),
                                domain=domain,
                                metadata=item.get("payload", {})
                            ))

        # Sort by relevance score
        memories.sort(key=lambda m: m.score, reverse=True)

        return memories[:10]  # Return top 10 memories

    async def _search_memory(self, client: httpx.AsyncClient, query: str, limit: int = 5) -> List[Dict]:
        """Execute a single memory search"""
        try:
            response = await client.post(
                self.mcp_url,
                json={
                    "method": "tools/call",
                    "params": {
                        "name": "search_memory",
                        "arguments": {
                            "query": query,
                            "limit": limit
                        }
                    }
                }
            )
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    return data
                return data.get("result", [])
        except Exception as e:
            logger.error(f"Memory search failed for '{query}': {e}")
        return []

    def _identify_knowledge_gaps(self, query: str, memories: List[MemoryContext]) -> List[str]:
        """Identify what we don't know about the query"""
        gaps = []

        # Check if we have high-confidence memories
        high_confidence = [m for m in memories if m.score > 0.7]

        if not high_confidence:
            gaps.append("No high-confidence memories found for this query")

        # Check for specific information types
        query_lower = query.lower()

        if "how" in query_lower and not any("implement" in m.content.lower() or "create" in m.content.lower() for m in memories):
            gaps.append("Missing implementation details")

        if "why" in query_lower and not any("because" in m.content.lower() or "reason" in m.content.lower() for m in memories):
            gaps.append("Missing reasoning or explanation")

        if "when" in query_lower and not any(m.metadata.get("timestamp") for m in memories):
            gaps.append("Missing temporal information")

        if "error" in query_lower or "bug" in query_lower:
            if not any("fix" in m.content.lower() or "solution" in m.content.lower() for m in memories):
                gaps.append("Missing solution or fix information")

        return gaps

    def _build_reasoning_chain(self, query: str, memories: List[MemoryContext], gaps: List[str]) -> List[str]:
        """Build a chain of reasoning based on memories and gaps"""
        steps = []

        # Step 1: What do we know?
        if memories:
            top_memory = memories[0]
            steps.append(f"Found relevant information with {top_memory.score:.2f} confidence from {top_memory.source}")

            # Categorize memories
            high_conf = len([m for m in memories if m.score > 0.7])
            med_conf = len([m for m in memories if 0.5 < m.score <= 0.7])
            low_conf = len([m for m in memories if m.score <= 0.5])

            steps.append(f"Memory distribution: {high_conf} high confidence, {med_conf} medium, {low_conf} low")
        else:
            steps.append("No directly relevant memories found")

        # Step 2: What patterns do we see?
        if len(memories) >= 3:
            # Check for consensus
            common_terms = self._find_common_terms(memories)
            if common_terms:
                steps.append(f"Common themes across memories: {', '.join(common_terms[:3])}")

        # Step 3: What are we missing?
        if gaps:
            steps.append(f"Knowledge gaps identified: {', '.join(gaps[:2])}")

        # Step 4: How recent is our knowledge?
        if memories:
            timestamps = [m.metadata.get("timestamp") for m in memories if m.metadata.get("timestamp")]
            if timestamps:
                # Parse and find most recent
                try:
                    dates = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t in timestamps[:3]]
                    most_recent = max(dates)
                    days_ago = (datetime.now() - most_recent.replace(tzinfo=None)).days
                    steps.append(f"Most recent relevant memory is from {days_ago} days ago")
                except:
                    pass

        # Step 5: Confidence assessment
        avg_score = sum(m.score for m in memories[:5]) / min(5, len(memories)) if memories else 0
        if avg_score > 0.7:
            steps.append("High confidence in retrieved information")
        elif avg_score > 0.5:
            steps.append("Moderate confidence in retrieved information")
        else:
            steps.append("Low confidence - may need additional context")

        return steps

    def _find_common_terms(self, memories: List[MemoryContext]) -> List[str]:
        """Find common significant terms across memories"""
        from collections import Counter

        # Extract words from all memories
        all_words = []
        stop_words = {"the", "is", "at", "which", "on", "and", "a", "an", "as", "are", "was", "were", "to", "of", "for", "in", "with"}

        for memory in memories[:5]:
            words = memory.content.lower().split()
            significant_words = [w for w in words if len(w) > 3 and w not in stop_words]
            all_words.extend(significant_words)

        # Count frequency
        word_counts = Counter(all_words)

        # Return most common
        return [word for word, count in word_counts.most_common(5) if count >= 2]

    async def _generate_intelligent_response(
        self,
        query: str,
        memories: List[MemoryContext],
        reasoning: List[str],
        context: Optional[str]
    ) -> str:
        """
        Generate an intelligent response using memories and reasoning
        This is where Echo Brain actually THINKS using LLM
        """
        if not memories:
            return "I don't have any relevant memories about this topic. I may need more information to help you."

        # Build context from memories for LLM
        memory_context = []
        for i, mem in enumerate(memories[:10], 1):
            memory_context.append(
                f"[Memory {i}] (score: {mem.score:.2f}, source: {mem.source})\n{mem.content[:500]}"
            )

        # Build reasoning context
        reasoning_context = "\n".join([f"- {step}" for step in reasoning])

        # Create system prompt
        system_prompt = """You are Echo Brain, Patrick's intelligent AI assistant with access to his personal knowledge base.

Your task is to synthesize the provided memories and reasoning steps into a coherent, intelligent response.

Guidelines:
1. SYNTHESIZE information - don't just list memories
2. Be SPECIFIC - reference concrete details when relevant
3. Be HONEST - acknowledge gaps in knowledge
4. Be CONCISE - no unnecessary preamble
5. STAY GROUNDED - only use information from provided memories

Focus on providing actionable, useful information based on what you know."""

        # Build user prompt
        user_prompt = f"""Query: {query}

Relevant Memories:
{chr(10).join(memory_context)}

Reasoning Steps:
{reasoning_context}

{"Context: " + context if context else ""}

Based on these memories and reasoning, provide a comprehensive, synthesized answer that directly addresses the query."""

        try:
            # Use Ollama to generate intelligent response
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.active_model,
                        "prompt": user_prompt,
                        "system": system_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": 500,
                            "top_p": 0.9
                        }
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    generated = result.get("response", "").strip()
                    if generated:
                        return generated

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")

        # Fallback to simple concatenation if LLM fails
        response_parts = []
        high_conf = [m for m in memories if m.score > 0.7]

        if high_conf:
            response_parts.append("Based on my memories:")
            for mem in high_conf[:3]:
                response_parts.append(f"- {mem.content[:200]}...")
        else:
            response_parts.append("Limited information available:")
            for mem in memories[:2]:
                response_parts.append(f"- {mem.content[:150]}... (confidence: {mem.score:.2f})")

        if reasoning:
            response_parts.append(f"\nAnalysis: {reasoning[-1]}")

        return "\n".join(response_parts)

    async def _generate_intelligent_response_stream(
        self,
        query: str,
        memories: List[MemoryContext],
        reasoning: List[str],
        context: Optional[str]
    ):
        """
        Stream the LLM response chunk by chunk
        """
        if not memories:
            yield "I don't have any relevant memories about this topic. I may need more information to help you."
            return

        # Build context from memories for LLM
        memory_context = []
        for i, mem in enumerate(memories[:10], 1):
            memory_context.append(
                f"[Memory {i}] (score: {mem.score:.2f}, source: {mem.source})\n{mem.content[:500]}"
            )

        # Build reasoning context
        reasoning_context = "\n".join([f"- {step}" for step in reasoning])

        # Create system prompt
        system_prompt = """You are Echo Brain, Patrick's intelligent AI assistant with access to his personal knowledge base.

Your task is to synthesize the provided memories and reasoning steps into a coherent, intelligent response.

Guidelines:
1. SYNTHESIZE information - don't just list memories
2. Be SPECIFIC - reference concrete details when relevant
3. Be HONEST - acknowledge gaps in knowledge
4. Be CONCISE - no unnecessary preamble
5. STAY GROUNDED - only use information from provided memories

Focus on providing actionable, useful information based on what you know."""

        # Build user prompt
        user_prompt = f"""Query: {query}

Relevant Memories:
{chr(10).join(memory_context)}

Reasoning Steps:
{reasoning_context}

{"Context: " + context if context else ""}

Based on these memories and reasoning, provide a comprehensive, synthesized answer that directly addresses the query."""

        try:
            # Stream response using Ollama
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.active_model,
                        "prompt": user_prompt,
                        "system": system_prompt,
                        "stream": True,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": 500,
                            "top_p": 0.9
                        }
                    }
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            import json
                            try:
                                data = json.loads(line)
                                if "response" in data:
                                    yield data["response"]
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"\n\n[Error: {str(e)}]"

    async def think_stream(self, query: str, context: Optional[str] = None):
        """
        Streaming version with progress updates
        """
        start_time = datetime.now()

        # Yield initial status
        yield {
            "type": "status",
            "data": "Analyzing query domain...",
            "timestamp": datetime.now().isoformat()
        }

        # Step 1: Domain identification
        domain = self._identify_domain(query)
        yield {
            "type": "status",
            "data": f"Domain identified: {domain.value}",
            "timestamp": datetime.now().isoformat()
        }

        # Step 2: Memory retrieval
        yield {
            "type": "status",
            "data": "Searching memories...",
            "timestamp": datetime.now().isoformat()
        }

        memories = await self._retrieve_memories(query, domain)

        # Calculate average score of ALL memories searched
        avg_memory_score = sum(m.score for m in memories) / len(memories) if memories else 0

        yield {
            "type": "memories_found",
            "data": {
                "memories_searched": len(memories),
                "memories_used": len(memories),  # IntelligenceEngine doesn't filter
                "avg_memory_score": round(avg_memory_score, 4)
            },
            "timestamp": datetime.now().isoformat()
        }

        # Step 3: Knowledge gap analysis
        knowledge_gaps = self._identify_knowledge_gaps(query, memories)

        # Step 4: Build reasoning
        reasoning_steps = self._build_reasoning_chain(query, memories, knowledge_gaps)

        yield {
            "type": "status",
            "data": f"Thinking with {self.active_model}...",
            "timestamp": datetime.now().isoformat()
        }

        # Step 5: Stream LLM response
        async for chunk in self._generate_intelligent_response_stream(
            query, memories, reasoning_steps, context
        ):
            yield {
                "type": "response_chunk",
                "data": chunk,
                "timestamp": datetime.now().isoformat()
            }

        # Final metrics
        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        confidence = self._calculate_confidence(memories, reasoning_steps)

        yield {
            "type": "complete",
            "data": {
                "domain": domain.value,
                "confidence": round(confidence, 4),
                "memories_searched": len(memories),
                "memories_used": len(memories),
                "avg_memory_score": round(avg_memory_score, 4),
                "model_used": self.active_model,
                "thinking_time_ms": round(elapsed_ms, 2),
                "reasoning_steps": reasoning_steps,
                "sources": list(set([m.source for m in memories[:5]]))
            },
            "timestamp": datetime.now().isoformat()
        }

    def _calculate_confidence(self, memories: List[MemoryContext], reasoning: List[str]) -> float:
        """Calculate overall confidence in our response"""
        if not memories:
            return 0.0

        # Factors:
        # 1. Average memory score (40%)
        avg_score = sum(m.score for m in memories[:5]) / min(5, len(memories))

        # 2. Number of high-confidence memories (30%)
        high_conf_ratio = len([m for m in memories if m.score > 0.7]) / min(5, len(memories))

        # 3. Memory consensus (20%) - do memories agree?
        if len(memories) >= 2:
            common_terms = self._find_common_terms(memories)
            consensus_score = min(1.0, len(common_terms) / 3)
        else:
            consensus_score = 0.5

        # 4. Reasoning completeness (10%)
        reasoning_score = min(1.0, len(reasoning) / 5)

        # Weighted average
        confidence = (
            avg_score * 0.4 +
            high_conf_ratio * 0.3 +
            consensus_score * 0.2 +
            reasoning_score * 0.1
        )

        return min(1.0, confidence)

    async def analyze_knowledge_coverage(self) -> Dict[str, Any]:
        """
        Analyze what Echo Brain knows across all domains
        This gives us a comprehensive view of the knowledge base
        """
        logger.info("🔍 Analyzing Echo Brain's complete knowledge coverage...")

        domains_to_test = [
            ("Tower System", ["tower", "system", "service", "API", "dashboard"]),
            ("Anime Production", ["anime", "video", "generation", "comfyui", "framepack"]),
            ("Echo Brain", ["echo brain", "memory", "ingestion", "vector", "qdrant"]),
            ("Programming", ["python", "javascript", "code", "function", "implementation"]),
            ("Infrastructure", ["postgresql", "nginx", "systemd", "docker", "kubernetes"]),
            ("AI/ML", ["embedding", "transformer", "neural", "model", "training"])
        ]

        coverage = {}

        for domain_name, keywords in domains_to_test:
            domain_coverage = {
                "keywords_tested": keywords,
                "memories_found": 0,
                "avg_confidence": 0,
                "sample_memories": [],
                "knowledge_depth": "unknown"
            }

            # Test each keyword
            all_memories = []
            for keyword in keywords[:3]:  # Test first 3 keywords
                memories = await self._retrieve_memories(
                    keyword,
                    KnowledgeDomain.UNKNOWN
                )
                all_memories.extend(memories)

            if all_memories:
                # Calculate stats
                domain_coverage["memories_found"] = len(all_memories)
                domain_coverage["avg_confidence"] = sum(m.score for m in all_memories) / len(all_memories)

                # Get top samples
                top_memories = sorted(all_memories, key=lambda m: m.score, reverse=True)[:3]
                domain_coverage["sample_memories"] = [
                    {
                        "content": m.content[:100] + "..." if len(m.content) > 100 else m.content,
                        "score": m.score,
                        "source": m.source
                    }
                    for m in top_memories
                ]

                # Assess knowledge depth
                if domain_coverage["avg_confidence"] > 0.7 and domain_coverage["memories_found"] > 10:
                    domain_coverage["knowledge_depth"] = "comprehensive"
                elif domain_coverage["avg_confidence"] > 0.5 and domain_coverage["memories_found"] > 5:
                    domain_coverage["knowledge_depth"] = "moderate"
                elif domain_coverage["memories_found"] > 0:
                    domain_coverage["knowledge_depth"] = "limited"
                else:
                    domain_coverage["knowledge_depth"] = "none"

            coverage[domain_name] = domain_coverage

        # Overall statistics
        total_memories = sum(d["memories_found"] for d in coverage.values())
        domains_with_knowledge = sum(1 for d in coverage.values() if d["memories_found"] > 0)

        return {
            "domains_analyzed": len(domains_to_test),
            "domains_with_knowledge": domains_with_knowledge,
            "total_memories_sampled": total_memories,
            "domain_coverage": coverage,
            "timestamp": datetime.now().isoformat()
        }


# Global instance
intelligence = IntelligenceEngine()


async def test_intelligence():
    """Test the intelligence engine with various queries"""

    test_queries = [
        "How does the Tower system architecture work?",
        "What do you know about anime video generation?",
        "Explain the Echo Brain ingestion pipeline",
        "What errors have we encountered with PostgreSQL?",
        "How do we generate embeddings for conversations?"
    ]

    print("\n" + "="*60)
    print("🧠 ECHO BRAIN INTELLIGENCE TEST")
    print("="*60)

    for query in test_queries:
        print(f"\n❓ Query: {query}")
        print("-"*40)

        thought = await intelligence.think_and_respond(query)

        print(f"🎯 Domain: {thought.domain.value}")
        print(f"📊 Confidence: {thought.confidence_score:.2%}")
        print(f"💭 Reasoning Steps: {len(thought.reasoning_steps)}")

        for step in thought.reasoning_steps:
            print(f"  • {step}")

        print(f"\n💡 Response:\n{thought.response}")

        if thought.sources_used:
            print(f"\n📚 Sources: {', '.join(thought.sources_used[:3])}")

        print("-"*40)

    # Analyze overall knowledge coverage
    print("\n" + "="*60)
    print("📊 KNOWLEDGE COVERAGE ANALYSIS")
    print("="*60)

    coverage = await intelligence.analyze_knowledge_coverage()

    print(f"Domains analyzed: {coverage['domains_analyzed']}")
    print(f"Domains with knowledge: {coverage['domains_with_knowledge']}")
    print(f"Total memories sampled: {coverage['total_memories_sampled']}")

    print("\nDomain Breakdown:")
    for domain, stats in coverage["domain_coverage"].items():
        print(f"\n  {domain}:")
        print(f"    Knowledge Depth: {stats['knowledge_depth'].upper()}")
        print(f"    Memories Found: {stats['memories_found']}")
        print(f"    Avg Confidence: {stats['avg_confidence']:.2%}")


if __name__ == "__main__":
    asyncio.run(test_intelligence())