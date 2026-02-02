# Echo Brain Architecture Consolidation Plan

## Executive Summary
Reduce 15 directories → 6 directories while maintaining all functionality

## Current Problems Identified
1. **241 Python files** across 15 directories
2. **27 files** handling auth/memory/routing/vectors
3. **Circular dependency** between api ↔ core
4. **3 directories with ≤1 import** (managers, interfaces, root files)
5. **24 orphaned files** in unused/ including critical integrations

## Proposed Target Architecture

```
src/
├── main.py                 # Single entry point
├── api/                    # All HTTP endpoints
│   ├── endpoints/          # REST endpoints (merge routers/)
│   ├── websockets/         # WebSocket handlers
│   └── middleware/         # Auth, CORS, logging
├── core/                   # Base classes & utilities
│   ├── config.py           # Configuration
│   ├── database.py         # DB connection (from db/)
│   └── utils/              # Shared utilities
├── services/               # Business logic
│   ├── memory.py           # ONE memory system
│   ├── agents.py           # ONE agent system (merge agents/, autonomous/)
│   ├── models.py           # Model management
│   └── auth.py             # ONE auth system
├── integrations/           # External services
│   ├── moltbook.py         # Moltbook API
│   ├── mcp.py              # MCP server
│   └── omniscient.py       # Omniscient pipeline
└── modules/                # Feature modules
    └── [keep existing]     # Board, templates, etc.
```

## Migration Steps

### Phase 1: Quick Wins (Day 1)
```bash
# 1. Merge low-impact directories
mv src/routers/* src/api/endpoints/
mv src/interfaces/* src/api/
mv src/managers/* src/services/

# 2. Consolidate database
mv src/db/* src/core/
rmdir src/db

# 3. Clean obvious duplicates
rm src/unused/echo_vector_memory.py  # Duplicate
rm src/unused/model_router.py        # Duplicate
```

### Phase 2: Service Consolidation (Day 2)
```python
# 1. Create unified memory service
# src/services/memory.py - Merge:
- src/memory/*
- src/core/memory/*
- src/services/memory/*
- Keep best implementation

# 2. Create unified agent service
# src/services/agents.py - Merge:
- src/agents/*
- src/autonomous/*
- src/unused/collaboration_framework.py (if needed)

# 3. Create unified auth service
# src/services/auth.py - Pick ONE from:
- src/api/auth_*.py
- src/core/auth.py
- src/integrations/auth_*.py
```

### Phase 3: Fix Circular Dependencies (Day 3)
```python
# Break api ↔ core circular dependency
# Move shared models to core/models.py
# API should import from core, not vice versa
```

### Phase 4: Recovery & Testing (Day 4)
```bash
# Recover critical unused files
mv src/unused/omniscient_pipeline.py src/integrations/
mv src/unused/collaboration_framework.py src/services/
mv src/unused/integrate_board_api.py src/modules/board/

# Delete remaining unused
rm -rf src/unused/
```

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Directories | 15 | 6 |
| Python files | 241 | ~150 |
| Auth implementations | 7+ | 1 |
| Memory systems | 4 | 1 |
| Router patterns | 3 | 1 |
| Circular dependencies | 1 | 0 |

## Risk Mitigation

1. **Before ANY deletion:**
   ```bash
   # Full backup
   cp -r /opt/tower-echo-brain /opt/tower-echo-brain.backup.$(date +%Y%m%d)

   # Test all endpoints
   python3 test_all_endpoints.py
   ```

2. **Progressive migration:**
   - One service at a time
   - Test after each merge
   - Keep old files in archive/ until verified

3. **Rollback plan:**
   ```bash
   # If something breaks
   git reset --hard HEAD
   # Or restore from backup
   mv /opt/tower-echo-brain.backup.* /opt/tower-echo-brain
   ```

## Implementation Script

```python
#!/usr/bin/env python3
# consolidate.py - Automated consolidation with safety checks

import os
import shutil
from pathlib import Path

def safe_merge(source, target, backup_dir):
    """Safely merge directories with backup"""
    # Implementation here
    pass

def test_endpoints():
    """Test all 12 endpoints still work"""
    # Implementation here
    pass

def update_imports(old_path, new_path):
    """Update all imports after moving files"""
    # Implementation here
    pass

# Execute consolidation
if __name__ == "__main__":
    # Run consolidation steps
    pass
```

## Timeline

- **Day 1:** Quick wins, backup everything
- **Day 2:** Service consolidation
- **Day 3:** Fix circular dependencies
- **Day 4:** Recovery & comprehensive testing
- **Day 5:** Documentation & cleanup

## Expected Outcome

A clean, maintainable architecture with:
- Clear separation of concerns
- Single source of truth per domain
- No circular dependencies
- All endpoints functional
- 40% reduction in code complexity