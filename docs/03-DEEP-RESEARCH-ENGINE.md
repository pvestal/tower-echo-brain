# TASK: Build Deep Research Engine for Echo Brain

## PREREQUISITES

- Web search (Prompt 01) must be deployed and working
- Document ingestion (Prompt 02) should be done but is not strictly required

## CONTEXT

This is Echo Brain's answer to Perplexity Deep Research. It takes a complex question, breaks it into sub-questions, iteratively searches web + memory, evaluates findings, and produces a cited research report. The key difference from Perplexity: Echo Brain searches BOTH the live web AND 509K+ personal memory vectors simultaneously.

## SYSTEM FACTS

```
Echo Brain:       /opt/tower-echo-brain/
Web Search:       /api/echo/search/web (SearXNG at localhost:8888)
Vector Search:    Qdrant at localhost:6333, echo_memory collection (768D)
Agents:           5 agents with hot-reload from markdown definitions
Reasoning Model:  deepseek-r1:8b via Ollama
General Model:    mistral:7b or equivalent via Ollama
SSE Streaming:    Already implemented in /ask/stream
Ollama:           localhost:11434
API Keys:         Anthropic, OpenAI, DeepSeek available in Vault
```

## DESIGN

```
User asks: "Should I upgrade my RV batteries to Battleborn or build a custom LiFePO4 pack?"

Step 1: DECOMPOSE
├── "Battleborn LiFePO4 RV battery specs pricing 2025 2026"
├── "DIY LiFePO4 battery pack RV build cost components"
├── "Victron MultiPlus compatibility LiFePO4 brands"
├── "RV lithium battery failure modes safety BMS"
└── "Sundowner Trailblazer battery compartment dimensions"

Step 2: SEARCH (parallel per sub-question)
├── Web Search → SearXNG results for each query
├── Memory Search → Qdrant results (Patrick's own RV work, past decisions)
└── Facts Search → structured facts about Patrick's RV setup

Step 3: EVALUATE
├── Do we have enough to answer comprehensively? 
├── Are there contradictions between sources?
└── What gaps remain? → Generate follow-up queries → Loop back to Step 2

Step 4: SYNTHESIZE
├── Merge all findings with source attribution
├── Web sources: [W1], [W2], etc.
├── Memory sources: [M1], [M2], etc.
├── Facts: [F1], [F2], etc.
└── Produce structured report with recommendation
```

## PHASE 1: UNDERSTAND EXISTING ARCHITECTURE

```bash
# Understand the ask pipeline
grep -rn "async def.*ask\|process_message\|generate_response" /opt/tower-echo-brain/src/ --include="*.py" | head -20

# Check existing agent collaboration
grep -rn "collaborate\|orchestrat\|multi.*step\|research" /opt/tower-echo-brain/src/ --include="*.py" | head -15

# Check LangGraph or similar
/opt/tower-echo-brain/venv/bin/pip list | grep -i "langgraph\|langchain\|graph"

# Check how Ollama is called
grep -rn "ollama\|11434\|generate\|chat.*completion" /opt/tower-echo-brain/src/ --include="*.py" | head -20

# Check the SearchService from Prompt 01
cat /opt/tower-echo-brain/src/services/search_service.py | head -50

# Check existing SSE streaming implementation
grep -rn "SSE\|EventSource\|StreamingResponse\|stream" /opt/tower-echo-brain/src/ --include="*.py" | head -15
```

## PHASE 2: BUILD RESEARCH ENGINE

Create `/opt/tower-echo-brain/src/services/research_engine.py`

### Core Architecture

```python
"""
Multi-step research engine that combines web search and personal memory.
Decomposes complex questions, iteratively searches, and synthesizes cited reports.
"""

class ResearchEngine:
    """
    Flow:
    1. Decompose question into 3-6 searchable sub-questions
    2. For each sub-question, parallel search: web + memory + facts
    3. Evaluate: do we have enough? Are there contradictions?
    4. If insufficient, generate follow-up questions and loop (max 3 iterations)
    5. Synthesize all findings into a cited report
    """
    
    MAX_ITERATIONS = 3
    MAX_SUB_QUESTIONS = 6
    MIN_SOURCES_FOR_COMPLETION = 5
    
    def __init__(self, search_service, memory_service, fact_service, llm_service):
        ...
    
    async def research(self, question: str, depth: str = "standard",
                       on_progress: Callable = None) -> ResearchReport:
        """
        Run a complete research session.
        
        depth: "quick" (1 iteration), "standard" (up to 2), "deep" (up to 3)
        on_progress: callback for SSE streaming progress events
        """
        ...
    
    async def _decompose(self, question: str) -> list[SubQuestion]:
        """Use reasoning model to break question into searchable sub-questions."""
        # Prompt deepseek-r1 to generate 3-6 focused queries
        # Each query should be 3-8 words, optimized for search engines
        # Include a "purpose" field explaining what this sub-question answers
        ...
    
    async def _search_all_sources(self, sub_question: SubQuestion) -> FindingSet:
        """Parallel search across web, memory, and facts for one sub-question."""
        web_task = self.search_service.search(sub_question.query, num_results=5)
        memory_task = self._search_memory(sub_question.query, limit=5)
        fact_task = self._search_facts(sub_question.query, limit=3)
        
        web_results, memory_results, fact_results = await asyncio.gather(
            web_task, memory_task, fact_task,
            return_exceptions=True
        )
        # Handle exceptions gracefully — a failed web search shouldn't kill the whole research
        ...
    
    async def _fetch_key_pages(self, web_results: list[SearchResult], max_pages: int = 3) -> list[WebDocument]:
        """Fetch full content from the most promising web results."""
        # Use search_service.search_and_fetch for detailed content
        # Prioritize: official docs > authoritative sources > forums
        ...
    
    async def _evaluate_sufficiency(self, question: str, 
                                      all_findings: list[FindingSet]) -> Evaluation:
        """Determine if gathered evidence is sufficient to answer comprehensively."""
        # Prompt the reasoning model:
        # - Is there enough information to answer the question thoroughly?
        # - Are there contradictions that need resolution?
        # - What specific aspects are still unanswered?
        # - If not sufficient, what follow-up queries would help?
        # Return: sufficient (bool), gaps (list), follow_up_queries (list)
        ...
    
    async def _synthesize(self, question: str, 
                           all_findings: list[FindingSet]) -> ResearchReport:
        """Produce final report with inline citations."""
        # Build a source index:
        #   [W1] = web result 1 (title, URL)
        #   [M1] = memory result 1 (source, date)
        #   [F1] = fact (subject, predicate, object)
        
        # Prompt the general model with ALL findings + source index
        # Instruct it to:
        #   - Write a comprehensive answer
        #   - Cite sources inline: "The battery capacity is 200Ah [W2] which aligns with your existing setup [M1]"
        #   - Note any contradictions between sources
        #   - Provide a clear recommendation if the question asks for one
        ...
```

### Data Models

```python
@dataclass
class SubQuestion:
    query: str          # Search-optimized query (3-8 words)
    purpose: str        # What this sub-question is meant to answer
    iteration: int      # Which research iteration generated this

@dataclass
class FindingSet:
    sub_question: SubQuestion
    web_results: list[SearchResult]
    web_documents: list[WebDocument]  # Full fetched content
    memory_results: list[MemoryResult]
    fact_results: list[FactResult]
    
@dataclass
class Evaluation:
    sufficient: bool
    confidence: float       # 0-1
    gaps: list[str]         # What's still missing
    follow_up_queries: list[SubQuestion]
    
@dataclass  
class ResearchReport:
    question: str
    answer: str             # Full report with inline citations
    sources: list[Source]   # All sources used, with [W1] [M1] [F1] keys
    iterations: int         # How many search iterations were needed
    sub_questions: list[SubQuestion]
    total_search_time_ms: float
    total_sources_consulted: int
    depth: str
```

### LLM Integration

**CRITICAL DECISION:** The research engine should use Ollama local models by default, but support API fallback for complex research:

```python
# Default: Use local Ollama (free, private)
# For decomposition: deepseek-r1:8b (reasoning)
# For synthesis: mistral:7b or whatever general model is configured

# Optional: Route to Claude API for complex research
# Check if question complexity warrants API call
# Use Claude Haiku for cost efficiency (~$0.034 per research session)
# Only if: depth="deep" AND local model produces low-confidence synthesis
```

Find the existing LLM calling pattern in the codebase and use it:

```bash
grep -rn "ollama\|generate\|chat.*api\|anthropic\|openai" /opt/tower-echo-brain/src/services/ --include="*.py" | head -20
grep -rn "class.*LLM\|class.*Model\|async def.*generate" /opt/tower-echo-brain/src/ --include="*.py" | head -15
```

## PHASE 3: API ENDPOINTS

```
POST /api/echo/research
  Body: { "question": "...", "depth": "standard" }  
  Response: { "job_id": "uuid", "status": "started" }
  
GET /api/echo/research/{job_id}
  Response: { "status": "running|complete|failed", "report": {...}, "progress": {...} }

GET /api/echo/research/{job_id}/stream
  Response: SSE stream of progress events:
    event: decomposing
    data: {"sub_questions": [...]}
    
    event: searching  
    data: {"sub_question": "...", "web_results": 5, "memory_results": 3}
    
    event: evaluating
    data: {"iteration": 1, "sufficient": false, "gaps": [...]}
    
    event: synthesizing
    data: {"sources_count": 15}
    
    event: complete
    data: {"report": {...}}

GET /api/echo/research/history
  Response: List of past research sessions with questions and summaries
```

### Job Management

Research jobs can take 30-120 seconds. They need:
- Background execution (don't block the API)
- Progress tracking via SSE
- Result storage in PostgreSQL for later retrieval
- Cancellation support

```sql
CREATE TABLE IF NOT EXISTS research_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question TEXT NOT NULL,
    depth VARCHAR(20) DEFAULT 'standard',
    status VARCHAR(20) DEFAULT 'pending',
    progress JSONB DEFAULT '{}',
    report JSONB,
    error_message TEXT,
    iterations INTEGER DEFAULT 0,
    sources_consulted INTEGER DEFAULT 0,
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    total_time_ms FLOAT
);
```

## PHASE 4: MCP TOOL

```python
@mcp.tool()
async def deep_research(
    question: str,
    depth: str = "standard",
) -> str:
    """Run multi-step research combining web search and personal memory.
    
    Decomposes the question, iteratively searches multiple sources,
    and produces a comprehensive cited report.
    
    Args:
        question: The research question (complex questions work best)
        depth: "quick" (1 iteration), "standard" (up to 2), "deep" (up to 3)
    
    Returns:
        Research report with citations [W1] for web, [M1] for memory, [F1] for facts
    """
    # Start research job
    # Poll for completion (with timeout)
    # Return formatted report
```

## PHASE 5: VERIFY

```bash
echo "=== TEST 1: Quick Research ==="
JOB=$(curl -s -X POST http://localhost:8309/api/echo/research \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the current recommended approach for RV lithium battery upgrades?", "depth": "quick"}' | python3 -c "import sys,json; print(json.load(sys.stdin).get('job_id',''))")

echo "Job ID: $JOB"
echo "Waiting for completion..."
for i in $(seq 1 30); do
  STATUS=$(curl -s "http://localhost:8309/api/echo/research/$JOB" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','unknown'))")
  echo "  [$i] Status: $STATUS"
  [ "$STATUS" = "complete" ] && break
  [ "$STATUS" = "failed" ] && break
  sleep 5
done

echo ""
echo "=== TEST 2: Get Report ==="
curl -s "http://localhost:8309/api/echo/research/$JOB" | python3 -c "
import sys, json
d = json.load(sys.stdin)
if d.get('status') == 'complete':
    report = d.get('report', {})
    answer = report.get('answer', '')
    sources = report.get('sources', [])
    print(f'PASS: Report generated')
    print(f'  Length: {len(answer)} chars')
    print(f'  Sources: {len(sources)}')
    print(f'  Iterations: {report.get(\"iterations\", 0)}')
    print(f'  Preview: {answer[:300]}...')
    web_sources = [s for s in sources if s.get('type') == 'web']
    mem_sources = [s for s in sources if s.get('type') == 'memory']
    print(f'  Web sources: {len(web_sources)}, Memory sources: {len(mem_sources)}')
else:
    print(f'Status: {d.get(\"status\")}')
    print(f'Error: {d.get(\"error_message\",\"none\")}')
"

echo ""
echo "=== TEST 3: SSE Stream ==="
timeout 60 curl -s -N "http://localhost:8309/api/echo/research/$JOB/stream" | head -20

echo ""
echo "=== TEST 4: Research History ==="
curl -s http://localhost:8309/api/echo/research/history | python3 -c "
import sys, json; d=json.load(sys.stdin)
print(f'PASS: {len(d)} research sessions in history') if d else print('CHECK: no history yet')
"
```

## CONSTRAINTS

- Use existing LLM calling patterns from the codebase
- Use existing SSE streaming infrastructure  
- Use the SearchService from Prompt 01 for web search
- Use existing Qdrant/memory search for memory results
- Background jobs via asyncio.create_task (or existing job pattern if one exists)
- Total research should complete within 120 seconds for "standard" depth
- Local Ollama models by default — API fallback is optional enhancement
- Do NOT make this dependent on LangGraph unless it's already installed and used

## DONE WHEN

- [ ] ResearchEngine class created with decompose → search → evaluate → synthesize loop
- [ ] Research jobs table in PostgreSQL
- [ ] POST /api/echo/research starts a job and returns job_id
- [ ] GET /api/echo/research/{id} returns status and report
- [ ] SSE streaming shows progress events
- [ ] Research combines web + memory + facts sources
- [ ] Report includes inline citations [W1] [M1] [F1]
- [ ] MCP `deep_research` tool registered
- [ ] Verification tests pass
