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

Echo Brain automatically selects models based on complexity:

| Complexity | Model |
|------------|-------|
| Basic (0-30) | qwen2.5-coder:7b |
| Advanced (30-60) | qwen2.5-coder:32b |
| Reasoning (60+) | deepseek-r1:8b / deepseek-r1:70b |

## Integration with tower-anime-production

Echo Brain provides AI services for anime production:

```python
import httpx

async def ask_echo(prompt: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://192.168.50.135:8309/api/echo/chat",
            json={"query": prompt, "user_id": "anime_production"}
        )
        return response.json()
```

See `docs/INTEGRATION_CONTRACT.md` for full API documentation.

## Database

Connects to PostgreSQL databases:
- `echo_brain` - Echo Brain state and conversations
- `anime_production` - Shared with tower-anime-production

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
