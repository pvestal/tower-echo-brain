"""
Context Compiler

Assembles retrieved context into a coherent prompt ready for LLM inference.
Handles:
- Domain-specific system prompt selection
- Token budget management
- Priority-based content truncation
- Final context formatting
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

import tiktoken  # For accurate token counting

from .models import (
    Domain, TokenBudget, AssembledContext,
    ClassificationResult, RetrievalResult,
    VectorResult, FactResult, ConversationTurn, CodeContext
)


# Domain-specific system prompts
SYSTEM_PROMPTS = {
    Domain.TECHNICAL: """You are Echo Brain's technical assistant, specialized in helping Patrick with:
- Tower server infrastructure (AMD Ryzen 9 24-core, 96GB DDR6 RAM, RTX 3060 12GB, RX 9070 XT 16GB)
- Echo Brain system development (PostgreSQL, Qdrant, Ollama, FastAPI microservices)
- Python, TypeScript, and system administration
- Debugging, optimization, and architectural decisions

You have access to Patrick's codebase context and technical documentation. Be direct and solution-focused.
When discussing code, provide working examples. When debugging, ask for specific error messages if not provided.

Current system state from memory will be provided below.""",

    Domain.ANIME: """You are Echo Brain's creative production assistant for anime content generation.
Active projects:
- Tokyo Debt Desire (TDD)
- Cyberpunk Goblin Slayer (CGS)

You help with:
- ComfyUI workflows and node configurations
- LoRA training and checkpoint selection
- Character consistency across scenes
- LTX video generation optimization
- Storyboard and scene planning

Keep technical (server/database) concerns separate from creative production.
Reference project-specific settings and previous creative decisions from context.""",

    Domain.PERSONAL: """You are Echo Brain's personal assistant for Patrick.
You help with:
- RV electrical systems (Victron MultiPlus II, LiFePO4 batteries, solar MPPT)
- 2022 Toyota Tundra 1794 Edition maintenance and modifications
- 2021 Sundowner Trailblazer 2286TB toy hauler
- Schedule and preference management
- Outdoor activities and trip planning

Draw on past conversations and preferences when making recommendations.""",

    Domain.GENERAL: """You are Echo Brain, Patrick's personal AI assistant.
You have memory of past conversations and can help with a wide range of topics.
When the query doesn't fit a specific domain, you provide helpful, informed responses
while being aware of Patrick's technical background and interests.

If the query would benefit from domain-specific context (technical, anime production, or personal),
suggest narrowing the focus for better assistance."""
}


class ContextCompiler:
    """
    Compiles retrieved context into a final prompt with token budget management.
    
    Priority for content inclusion (when budget is tight):
    1. System prompt (always included)
    2. Facts (highest information density)
    3. Recent conversation (for continuity)
    4. Code context (when relevant)
    5. Vector content (general context)
    """
    
    def __init__(
        self,
        token_budget: TokenBudget = None,
        encoding_name: str = "cl100k_base"  # GPT-4/Claude compatible
    ):
        self.budget = token_budget or TokenBudget()
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def compile(
        self,
        query: str,
        classification: ClassificationResult,
        retrieval: RetrievalResult
    ) -> AssembledContext:
        """
        Compile retrieval results into final context.
        
        Args:
            query: The original user query
            classification: Domain classification result
            retrieval: Retrieved content from all sources
            
        Returns:
            AssembledContext ready for LLM
        """
        start_time = datetime.utcnow()
        
        # Get domain-specific system prompt
        system_prompt = SYSTEM_PROMPTS.get(classification.domain, SYSTEM_PROMPTS[Domain.GENERAL])
        system_tokens = self._count_tokens(system_prompt)
        
        # Calculate available budget
        query_tokens = self._count_tokens(query)
        available = self.budget.total - system_tokens - query_tokens - 500  # Buffer
        
        # Allocate budget by priority
        allocations = self._allocate_budget(available, retrieval)
        
        # Select content within budget
        selected_facts = self._select_facts(retrieval.facts, allocations["facts"])
        selected_conversation = self._select_conversation(
            retrieval.conversation_history, 
            allocations["conversation"]
        )
        selected_code = self._select_code(retrieval.code_context, allocations["code"])
        
        # Calculate final token count
        total_tokens = (
            system_tokens + 
            query_tokens +
            self._count_tokens(self._format_facts(selected_facts)) +
            self._count_tokens(self._format_conversation(selected_conversation)) +
            self._count_tokens(self._format_code(selected_code))
        )
        
        elapsed_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        return AssembledContext(
            system_prompt=system_prompt,
            facts=selected_facts,
            conversation_history=selected_conversation,
            code_context=selected_code,
            domain=classification.domain,
            classification_confidence=classification.confidence,
            token_count=total_tokens,
            assembly_time_ms=elapsed_ms,
            query=query,
            vector_ids_used=[v.id for v in retrieval.vectors[:10]],  # Track what we used
            fact_ids_used=[f.id for f in selected_facts]
        )
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        return len(self.encoding.encode(text))
    
    def _allocate_budget(
        self,
        available: int,
        retrieval: RetrievalResult
    ) -> dict[str, int]:
        """
        Allocate token budget across content types.
        
        Uses a dynamic allocation based on what content is available.
        """
        # Base allocations (proportional)
        base = {
            "facts": 0.35,       # 35% for facts
            "conversation": 0.30,  # 30% for conversation
            "code": 0.25,       # 25% for code
            "vectors": 0.10     # 10% for additional vector context
        }
        
        # Adjust based on what's available
        if not retrieval.facts:
            base["conversation"] += base["facts"] / 2
            base["code"] += base["facts"] / 2
            base["facts"] = 0
        
        if not retrieval.code_context:
            base["facts"] += base["code"] / 2
            base["conversation"] += base["code"] / 2
            base["code"] = 0
        
        if not retrieval.conversation_history:
            base["facts"] += base["conversation"] / 2
            base["code"] += base["conversation"] / 2
            base["conversation"] = 0
        
        return {
            key: int(available * proportion)
            for key, proportion in base.items()
        }
    
    def _select_facts(
        self,
        facts: list[FactResult],
        budget: int
    ) -> list[FactResult]:
        """Select facts within token budget, prioritizing by relevance."""
        if not facts or budget <= 0:
            return []
        
        # Sort by relevance score
        sorted_facts = sorted(facts, key=lambda f: f.relevance_score, reverse=True)
        
        selected = []
        used_tokens = 0
        
        for fact in sorted_facts:
            fact_tokens = self._count_tokens(fact.fact_text)
            if used_tokens + fact_tokens <= budget:
                selected.append(fact)
                used_tokens += fact_tokens
            else:
                break  # Budget exhausted
        
        return selected
    
    def _select_conversation(
        self,
        turns: list[ConversationTurn],
        budget: int
    ) -> list[ConversationTurn]:
        """Select conversation turns within budget, keeping most recent."""
        if not turns or budget <= 0:
            return []
        
        # Start from most recent, work backwards
        selected = []
        used_tokens = 0
        
        for turn in reversed(turns):
            turn_text = f"{turn.role}: {turn.content}"
            turn_tokens = self._count_tokens(turn_text)
            if used_tokens + turn_tokens <= budget:
                selected.insert(0, turn)  # Maintain chronological order
                used_tokens += turn_tokens
            else:
                break
        
        return selected
    
    def _select_code(
        self,
        code_context: list[CodeContext],
        budget: int
    ) -> list[CodeContext]:
        """Select code context within budget, prioritizing by relevance."""
        if not code_context or budget <= 0:
            return []
        
        # Sort by relevance
        sorted_code = sorted(code_context, key=lambda c: c.relevance_score, reverse=True)
        
        selected = []
        used_tokens = 0
        
        for code in sorted_code:
            code_tokens = self._count_tokens(code.content)
            if used_tokens + code_tokens <= budget:
                selected.append(code)
                used_tokens += code_tokens
            elif code_tokens > budget:
                # Truncate large files to fit
                truncated = self._truncate_code(code, budget - used_tokens)
                if truncated:
                    selected.append(truncated)
                break
            else:
                break
        
        return selected
    
    def _truncate_code(self, code: CodeContext, max_tokens: int) -> Optional[CodeContext]:
        """Truncate code to fit within token budget."""
        if max_tokens < 100:
            return None  # Not worth including tiny snippets
        
        lines = code.content.split('\n')
        truncated_lines = []
        used_tokens = 0
        
        # Include imports and class/function definitions first
        priority_patterns = ['import ', 'from ', 'class ', 'def ', 'async def ']
        
        # First pass: priority lines
        for line in lines:
            if any(line.strip().startswith(p) for p in priority_patterns):
                line_tokens = self._count_tokens(line)
                if used_tokens + line_tokens <= max_tokens * 0.5:  # Reserve half for body
                    truncated_lines.append(line)
                    used_tokens += line_tokens
        
        # Second pass: fill with remaining lines
        for line in lines:
            if line not in truncated_lines:
                line_tokens = self._count_tokens(line)
                if used_tokens + line_tokens <= max_tokens:
                    truncated_lines.append(line)
                    used_tokens += line_tokens
        
        if not truncated_lines:
            return None
        
        return CodeContext(
            file_path=code.file_path,
            content='\n'.join(truncated_lines) + '\n# ... (truncated)',
            language=code.language,
            relevance_score=code.relevance_score,
            function_names=code.function_names,
            class_names=code.class_names,
            imports=code.imports
        )
    
    def _format_facts(self, facts: list[FactResult]) -> str:
        """Format facts for context string."""
        if not facts:
            return ""
        return "\n".join(f"• {f.fact_text}" for f in facts)
    
    def _format_conversation(self, turns: list[ConversationTurn]) -> str:
        """Format conversation for context string."""
        if not turns:
            return ""
        return "\n".join(f"{t.role}: {t.content}" for t in turns)
    
    def _format_code(self, code_context: list[CodeContext]) -> str:
        """Format code for context string."""
        if not code_context:
            return ""
        parts = []
        for c in code_context:
            parts.append(f"### {c.file_path}\n```{c.language}\n{c.content}\n```")
        return "\n\n".join(parts)


class ContextAssembler:
    """
    High-level interface that combines classification, retrieval, and compilation.
    This is the main entry point for the context assembly pipeline.
    """
    
    def __init__(
        self,
        classifier,  # QueryClassifier
        retriever,   # RetrievalOrchestrator
        compiler: ContextCompiler = None
    ):
        self.classifier = classifier
        self.retriever = retriever
        self.compiler = compiler or ContextCompiler()
    
    async def assemble(self, query: str) -> AssembledContext:
        """
        Full context assembly pipeline.
        
        Args:
            query: The user's query
            
        Returns:
            AssembledContext ready for LLM inference
        """
        # Step 1: Classify the query
        classification = self.classifier.classify(query)
        
        # Step 2: Retrieve relevant context
        retrieval = await self.retriever.retrieve(
            query=query,
            domain=classification.domain,
            include_code=(classification.domain == Domain.TECHNICAL)
        )
        
        # Step 3: Compile into final context
        context = self.compiler.compile(query, classification, retrieval)
        
        return context
    
    def to_ollama_messages(self, context: AssembledContext, user_query: str) -> list[dict]:
        """
        Convert assembled context to Ollama message format.
        
        Returns list of messages ready for ollama.chat()
        """
        messages = []
        
        # System message with context
        system_content = context.system_prompt
        
        # Add facts if present
        if context.facts:
            facts_text = "\n".join(f"• {f.fact_text}" for f in context.facts)
            system_content += f"\n\n## Relevant Knowledge\n{facts_text}"
        
        # Add code context if present
        if context.code_context:
            code_text = "\n\n".join(
                f"### {c.file_path}\n```{c.language}\n{c.content}\n```"
                for c in context.code_context
            )
            system_content += f"\n\n## Code Context\n{code_text}"
        
        messages.append({
            "role": "system",
            "content": system_content
        })
        
        # Add conversation history
        for turn in context.conversation_history:
            messages.append({
                "role": turn.role,
                "content": turn.content
            })
        
        # Add current query
        messages.append({
            "role": "user",
            "content": user_query
        })
        
        return messages


# ============================================================================
# Factory function
# ============================================================================

def create_assembler(classifier, retriever) -> ContextAssembler:
    """Create a context assembler with default compiler."""
    return ContextAssembler(
        classifier=classifier,
        retriever=retriever,
        compiler=ContextCompiler()
    )
