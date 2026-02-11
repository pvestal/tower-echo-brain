# Echo Brain Test Inventory

## How to Run

```bash
# Smoke tests (primary — run first)
pytest tests/echo_brain_smoke_test.py -v

# Contract tests (Pact consumer)
cd contract-tests/consumer && npm test

# Individual test files
pytest tests/<filename>.py -v
```

## Active Test Files

### Core / Smoke
| File | Description | Runner |
|------|-------------|--------|
| `echo_brain_smoke_test.py` | Health, query, memory, embeddings, DB, performance, voice endpoint checks | `pytest -v` |
| `test_echo_brain.py` | Core Echo Brain unit tests | `pytest` |
| `test_router_inclusion.py` | Verifies FastAPI router registration | `pytest` |

### Auth Integration
| File | Description | Runner |
|------|-------------|--------|
| `test_auth_integration.py` | Tower Auth Bridge: Google, GitHub, Apple Music credentials | `python` (standalone) |
| `verify_and_use_integration.py` | End-to-end integration verification | `python` (standalone) |

### ComfyUI / Image Generation
| File | Description | Runner |
|------|-------------|--------|
| `test_comfyui.py` | ComfyUI API connectivity | `pytest` |
| `test_models.py` | Model loading verification | `pytest` |
| `test_better_models.py` | Model quality comparisons | `python` |
| `test_quality_models.py` | Quality scoring tests | `python` |
| `test_realistic_vision.py` | Realistic Vision checkpoint | `python` |
| `test_epicrealism.py` | EpicRealism checkpoint | `python` |
| `test_epic_female.py` | Female character generation | `python` |
| `test_cyberrealistic.py` | CyberRealistic checkpoint | `python` |
| `test_majicmix.py` | MajicMix checkpoint | `python` |
| `test_lora_generator.py` | LoRA generation tests | `python` |
| `test_all_tdd_characters.py` | Full character roster TDD | `python` |
| `test_both_genders_same_model.py` | Gender parity tests | `python` |
| `test_missing_males.py` | Male character coverage gaps | `python` |
| `test_openpose_controlnet.py` | OpenPose ControlNet pipeline | `python` |
| `test_15_images_tweaking.py` | Batch parameter tuning | `python` |
| `test_small_batch.py` | Small batch generation | `python` |

### Diagnostics / Utilities
| File | Description | Runner |
|------|-------------|--------|
| `echo_brain_knowledge_diagnostic.py` | Knowledge retrieval quality | `python` |
| `comprehensive_git_test.py` | Git integration tests | `python` |
| `final_working_test.py` | End-to-end working validation | `python` |
| `quick_test.py` | Quick sanity check | `python` |
| `debug_moltbook.py` | Moltbook-specific debug | `python` |
| `test_now.py` | Ad-hoc test runner | `python` |

### Contract Tests (separate directory)
| File | Description | Runner |
|------|-------------|--------|
| `contract-tests/consumer/tests/echo-brain.consumer.spec.ts` | Pact consumer contracts (health, query, memory, ingestion, voice) | `npm test` |

## Archived Tests (`tests/tmp_archive/`)

Files moved here are superseded or one-shot diagnostics:

| File | Reason Archived |
|------|----------------|
| `test_contracts.py` | Overlaps with Pact consumer spec; uses `/api/v1/` paths; has typo (`AssertionError`) |
| `test_integration_now.py` | Subset of `test_auth_integration.py` |
| `test_existing_oauth.py` | One-shot diagnostic that writes credential files |
| `test_found_token.py` | One-shot diagnostic that writes config files |
| `test_mcp.py` | Simpler MCP test superseded by `mcp_server/test_server.py` |
| *(older archives)* | Various Qdrant, extraction, GPU, and generation test scripts |
