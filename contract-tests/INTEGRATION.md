# Integrating Contract Tests with Echo Brain

## Step 1: Wire Up the Real FastAPI App

In `provider/test_provider_verification.py`, replace the `create_test_app()` stub
with your actual Echo Brain app:

```python
# Replace this:
# app = create_test_app()

# With your real app import:
from echo_brain.main import app  # adjust path to your app

# Then add the provider state endpoint to your app
# (only in test mode — use an environment flag):
import os
if os.getenv('CONTRACT_TEST_MODE'):
    from provider_states import state_manager

    @app.post("/_pact/provider-states")
    async def set_provider_state(body: dict):
        state_name = body.get('state', '')
        action = body.get('action', 'setup')
        if action == 'setup':
            # Inject mock data into your dependency injection layer
            app.state.test_overrides = state_manager.handle(state_name)
        return {"status": "ok"}
```

## Step 2: Adapt Provider States to Your Data Layer

The state handlers in `provider_states.py` currently return raw dicts.
Wire them into however Echo Brain manages its data:

```python
# If you use dependency injection (recommended for FastAPI):
from echo_brain.dependencies import get_db, get_qdrant_client

async def override_qdrant():
    """Return a mock Qdrant client seeded with test data."""
    mock = MockQdrantClient()
    mock.seed(state_manager.current_state.get('seed_documents', []))
    return mock

app.dependency_overrides[get_qdrant_client] = override_qdrant
```

## Step 3: Map Your Real Endpoints

Update the consumer tests to match your actual API paths. Common patterns:

| Consumer test path       | Your actual path (adjust)        |
|--------------------------|----------------------------------|
| `/api/v1/health`         | `/health` or `/api/health`       |
| `/api/v1/query`          | `/api/v1/search` or `/query`     |
| `/api/v1/memories`       | `/api/v1/knowledge`              |
| `/api/v1/ingestion/status` | `/api/v1/pipeline/status`      |

## Step 4: Add to Your CI Pipeline

```yaml
# .github/workflows/contract-tests.yml
name: Contract Tests
on:
  pull_request:
    paths:
      - 'frontend/**'
      - 'backend/**'
      - 'contracts/**'

jobs:
  contract-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: '20' }
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: ./scripts/run_contract_tests.sh
```

## Step 5: When an Endpoint Changes

**Backend change (FastAPI):**
1. Make your change
2. Run `./scripts/run_contract_tests.sh provider`
3. If it fails → you broke the contract → update frontend or negotiate

**Frontend change (Vue needs new field):**
1. Add the expectation in `consumer/tests/echo-brain.consumer.spec.ts`
2. Run `./scripts/run_contract_tests.sh consumer` to generate new contract
3. Run `./scripts/run_contract_tests.sh provider` — it will fail
4. Update FastAPI to satisfy the new contract
5. Both pass → deploy

This is the "consumer-driven" part: the frontend drives what the API must provide.

## Git Hook (Optional)

```bash
# .git/hooks/pre-push
#!/bin/bash
echo "Running contract tests..."
./scripts/run_contract_tests.sh both || {
    echo "Contract tests failed — push blocked"
    exit 1
}
```
