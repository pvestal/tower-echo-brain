# Echo Brain Contract Tests

Consumer-driven contract testing between the Vue.js frontend and FastAPI backend.

## Architecture

```
┌─────────────────┐     contract.json     ┌─────────────────┐
│   Vue Frontend   │ ──── generates ────▶ │   Pact Broker    │
│   (Consumer)     │                       │   (or local)     │
└─────────────────┘                       └────────┬─────────┘
                                                   │
                                          verifies against
                                                   │
                                          ┌────────▼─────────┐
                                          │  FastAPI Backend  │
                                          │   (Provider)      │
                                          └──────────────────┘
```

## Flow

1. **Consumer tests** (Vue/TS) define expected request/response pairs
2. Pact generates a contract JSON file from those expectations
3. **Provider tests** (Python) replay the contract against the real FastAPI app
4. If the provider can't satisfy the contract → test fails → drift caught

## Quick Start

### Consumer side (Vue/TypeScript)
```bash
cd consumer
npm install
npm test
```

### Provider side (Python/FastAPI)
```bash
cd provider
pip install -r requirements.txt
pytest
```

### Both (CI script)
```bash
./scripts/run_contract_tests.sh
```

## Adding New Contract Tests

1. Define the interaction in `consumer/tests/` (what the frontend expects)
2. Run consumer tests to generate updated contract
3. Run provider tests to verify the backend still satisfies it
4. Both sides pass → safe to deploy
