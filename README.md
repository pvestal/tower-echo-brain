# Echo Brain

AI orchestration system with intelligent model routing, conversation memory, and integration with tower-anime-production.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start Echo Brain
./start_echo.sh

# Or directly
python echo.py
```

API runs at `http://localhost:8309`

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/echo/chat` | POST | Main conversation endpoint |
| `/api/echo/query` | POST | Query endpoint (alias) |
| `/api/echo/health` | GET | Health check |
| `/api/echo/delegate/to-tower` | POST | Delegate to local LLMs |
| `/api/echo/delegate/capabilities` | GET | List available models |

## Architecture

```
tower-echo-brain/
├── echo.py                    # Entry point
├── src/
│   ├── app_factory.py         # FastAPI app creation
│   ├── api/                   # API routes
│   │   ├── echo.py            # Main chat/query endpoints
│   │   ├── delegation_routes.py # Tower LLM delegation
│   │   └── health.py          # Health checks
│   ├── core/                  # Core logic
│   │   ├── intelligence.py    # Model routing
│   │   └── conversation_manager.py
│   ├── memory/                # Conversation memory
│   ├── reasoning/             # DeepSeek reasoning
│   └── modules/
│       └── generation/anime/  # Anime production integration
├── config/                    # Configuration files
├── database/                  # Schema and migrations
└── tests/                     # Test suites
```

## Model Routing

Echo Brain uses intent-based routing via `UnifiedModelRouter`:

| Intent | Model | Notes |
|--------|-------|-------|
| Default/Fast | qwen2.5:3b | 94ms TTFT, greetings, quick questions |
| Coding | qwen2.5-coder:7b | Code generation, debugging |
| Reasoning | deepseek-r1:8b | Complex analysis with `<think>` tags |
| Conversation | llama3.1:8b | General chat |
| Fallback | llama3.2:3b | When database unavailable |

Routing is database-driven via `select_model()` PostgreSQL function, with fallback to pattern matching.

## Integration with tower-anime-production

**Echo Brain calls tower-anime-production** (not the other way around):

```
Echo Brain (8309) ---> tower-anime-production (8328)
                       /api/anime/generate
                       /api/anime/jobs/{id}
                       /api/anime/health
```

When a user asks to "generate anime", Echo Brain:
1. Detects `anime_generation` intent
2. Forwards to `http://localhost:8328/api/anime/generate`
3. Returns job ID for monitoring

**From tower-anime-production calling Echo Brain:**
```python
import httpx

async def ask_echo(prompt: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8309/api/echo/chat",
            json={"query": prompt, "user_id": "anime_production"}
        )
        return response.json()
```

See `docs/INTEGRATION_CONTRACT.md` for full API documentation.

## Database

Connects to PostgreSQL:
- `tower_consolidated` - Main database for routing decisions
- Model selection via `select_model()` function
- Performance logging to `model_performance` table

## Configuration

Environment variables:
```bash
DB_HOST=localhost
DB_NAME=echo_brain
DB_USER=echo_user
DB_PASSWORD=echo_password
ECHO_PORT=8309
ECHO_HOST=0.0.0.0
```

## Development

```bash
# Run tests
./run_tests.sh

# Or with pytest
pytest tests/
```
