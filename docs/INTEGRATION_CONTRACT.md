# Echo Brain ↔ Tower Anime Production Integration Contract

## Echo Brain Server
- **Host:** `***REMOVED***`
- **Port:** `8309`
- **Base URL:** `http://***REMOVED***:8309`

---

## API Endpoints tower-anime-production SHOULD Call

### 1. Main Chat/Query Endpoint
```
POST /api/echo/chat
POST /api/echo/query
```

**Request:**
```json
{
  "query": "string - the prompt/question",
  "user_id": "string - user identifier",
  "conversation_id": "string - optional, for context continuity",
  "intelligence_level": "auto | basic | advanced | reasoning"
}
```

**Response:**
```json
{
  "response": "string - Echo Brain's response",
  "model_used": "string - which LLM was used",
  "intelligence_level": "string - actual level used",
  "complexity_score": "number - calculated complexity",
  "processing_time": "number - seconds",
  "conversation_id": "string",
  "reasoning": {
    "steps": ["array of thinking steps if reasoning model used"],
    "model": "string"
  }
}
```

### 2. Delegation to Tower LLMs (Save Opus Tokens)
```
POST /api/echo/delegate/to-tower
```

**Request:**
```json
{
  "task": "string - task description",
  "context": {"optional": "context dict"},
  "model": "qwen2.5-coder:7b | deepseek-coder:latest | codellama:7b",
  "priority": "normal | high | low"
}
```

**Response:**
```json
{
  "success": true,
  "task": "string",
  "model": "string",
  "commands_executed": 0,
  "results": [],
  "execution_history": [],
  "timestamp": "string"
}
```

### 3. Health Check
```
GET /api/echo/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "ISO datetime",
  "services": {...}
}
```

### 4. Get Tower Capabilities
```
GET /api/echo/delegate/capabilities
```

**Response:**
```json
{
  "model": "current model",
  "available_models": ["list"],
  "capabilities": ["list"],
  "operations": {"category": ["ops"]}
}
```

---

## Shared Database: `anime_production`

Echo Brain reads from these tables (tower-anime-production should write to them):

### `anime_character_memory`
```sql
character_name VARCHAR
canonical_description TEXT
visual_consistency_score FLOAT
generation_count INT
successful_generations INT
reference_images JSONB
style_elements JSONB
last_generated TIMESTAMP
```

### `user_creative_preferences`
```sql
user_id VARCHAR
preference_type VARCHAR
preference_key VARCHAR
preference_value JSONB
confidence_score FLOAT
```

---

## What Echo Brain Provides Internally

### AnimeMemoryIntegration (`src/modules/generation/anime/anime_memory_integration.py`)
- `get_character_info(character_name)` - Reads character data
- `get_user_preferences(user_id)` - Reads user preferences
- `get_anime_context(query)` - Extracts anime context from queries

### Model Routing (`src/model_router.py`)
- Automatic model escalation based on complexity
- Basic: qwen2.5-coder:7b
- Advanced: qwen2.5-coder:32b
- Reasoning: deepseek-r1:8b or deepseek-r1:70b

---

## What's BROKEN/MISSING

### 1. Entry Point Mismatch
- `start_echo.sh` runs `python echo.py` (doesn't exist at root)
- Actual entry is `src/app_factory.py`
- **FIX:** Update start script or create root echo.py that imports app_factory

### 2. Legacy Imports Commented Out
In `src/app_factory.py`:
```python
# TODO: Fix legacy imports after restructuring
# from src.api.legacy.anime_integration import router as anime_integration_router
```
The anime integration router is disabled.

### 3. Missing Quality Orchestrator Path
`anime_story_orchestrator.py` expects:
```python
sys.path.append('/opt/tower-anime-production/quality')
from src.modules.generation.anime.anime_quality_orchestrator import AnimeQualityOrchestrator
```
This path doesn't exist in current tower-anime-production structure.

### 4. Hardcoded IPs
Multiple files reference `***REMOVED***` directly instead of using env vars.

---

## Recommended Integration Pattern

### From tower-anime-production:
```python
import httpx

ECHO_BRAIN_URL = os.getenv("ECHO_BRAIN_URL", "http://***REMOVED***:8309")

async def ask_echo_brain(prompt: str, conversation_id: str = None) -> dict:
    """Send query to Echo Brain for intelligent response"""
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(
            f"{ECHO_BRAIN_URL}/api/echo/chat",
            json={
                "query": prompt,
                "user_id": "anime_production",
                "conversation_id": conversation_id,
                "intelligence_level": "auto"
            }
        )
        return response.json()

async def delegate_to_tower(task: str, model: str = "qwen2.5-coder:7b") -> dict:
    """Delegate heavy computation to Tower LLMs"""
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{ECHO_BRAIN_URL}/api/echo/delegate/to-tower",
            json={
                "task": task,
                "model": model
            }
        )
        return response.json()
```

---

## Test the Integration
```bash
# Health check
curl http://***REMOVED***:8309/api/echo/health

# Simple query
curl -X POST http://***REMOVED***:8309/api/echo/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What anime characters do you know about?", "user_id": "test"}'

# Check capabilities
curl http://***REMOVED***:8309/api/echo/delegate/capabilities
```
