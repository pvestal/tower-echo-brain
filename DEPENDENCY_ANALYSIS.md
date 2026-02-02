# Echo Brain Dependency Analysis

## Directory Statistics
| Directory | Files | Imported By | Total Imports |
|-----------|-------|-------------|---------------|
| agents | 8 | 3 | 10 |
| api | 36 | 4 | 14 |
| autonomous | 31 | 1 | 5 |
| core | 23 | 6 | 18 |
| integrations | 20 | 4 | 14 |
| interfaces | 8 | 1 | 1 |
| managers | 10 | 1 | 1 |
| memory | 6 | 1 | 4 |
| modules | 39 | 1 | 2 |
| root | 2 | 0 | 0 |
| routers | 10 | 1 | 4 |
| services | 24 | 4 | 11 |

## Dependency Graph
```mermaid
graph TD
    root[root<br/>2 files]
    services[services<br/>24 files]
    core[core<br/>23 files]
    managers[managers<br/>10 files]
    integrations[integrations<br/>20 files]
    api[api<br/>36 files]
    routers[routers<br/>10 files]
    modules[modules<br/>39 files]
    agents[agents<br/>8 files]
    autonomous[autonomous<br/>31 files]
    interfaces[interfaces<br/>8 files]
    memory[memory<br/>6 files]
    api -->|10| core
    routers -->|9| api
    api -->|8| services
    api -->|6| middleware
    api -->|6| agents
    api -->|5| autonomous
    routers -->|5| integrations
    root -->|4| routers
    services -->|4| integrations
    api -->|4| db
    api -->|4| integrations
    api -->|4| memory
    api -->|4| utils
    api -->|3| routing
    autonomous -->|3| agents
    root -->|2| api
    services -->|2| db
    core -->|2| misc
    api -->|2| security
    api -->|2| tasks
    routers -->|2| modules
    modules -->|2| api
    autonomous -->|2| core
    root -->|1| services
    services -->|1| misc
    core -->|1| routing
    core -->|1| utils
    core -->|1| db
    core -->|1| middleware
    core -->|1| models
    core -->|1| api
    core -->|1| legacy
    managers -->|1| execution
    managers -->|1| engines
    managers -->|1| db
    managers -->|1| middleware
    integrations -->|1| core
    api -->|1| model_router
    api -->|1| qdrant_memory
    api -->|1| capabilities
    api -->|1| intelligence
    api -->|1| managers
    api -->|1| commands
    api -->|1| misc
    routers -->|1| services
    modules -->|1| core
    agents -->|1| core
    interfaces -->|1| routing
    memory -->|1| config
```

## Key Insights

### Most Imported Directories
- **core** (18 imports) from: agents, api, autonomous, integrations, modules, unused
- **api** (14 imports) from: core, modules, root, routers
- **integrations** (14 imports) from: api, routers, services, unused
- **services** (11 imports) from: api, root, routers, unused
- **db** (10 imports) from: api, core, managers, services, unused

### Consolidation Candidates (≤1 import)
- **root** (0 imports)
- **managers** (1 imports)
- **interfaces** (1 imports)

### Circular Dependencies
- api ↔ core