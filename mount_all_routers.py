#!/usr/bin/env python3
"""
Mount all missing routers in Echo Brain
This script identifies and adds all unmounted routers to main.py
"""

import os
import re
from pathlib import Path

# Find all router files
api_dir = Path("/opt/tower-echo-brain/src/api")
router_files = []

for py_file in api_dir.glob("*.py"):
    if py_file.name == "__init__.py":
        continue

    content = py_file.read_text()
    if "router = APIRouter()" in content or "router = Router()" in content:
        router_files.append(py_file.stem)
        print(f"Found router: {py_file.stem}")

print(f"\nTotal routers found: {len(router_files)}")

# Generate import statements
imports = []
includes = []

for router_name in sorted(router_files):
    # Skip known problematic or duplicate routers
    if router_name in ["__pycache__", "routes", "echo.py.memory_wiring_backup"]:
        continue

    imports.append(f"    from .api.{router_name} import router as {router_name}_router")

    # Determine prefix based on router name
    if router_name == "models":
        prefix = ""  # Already mounted
    elif router_name == "agents":
        prefix = "/api/echo/agents"
    elif router_name == "diagnostics":
        prefix = "/api/diagnostics"
    elif router_name == "autonomous":
        prefix = "/api/autonomous"
    elif router_name == "codebase":
        prefix = "/api/echo/codebase"
    elif router_name == "echo":
        prefix = ""  # Root level echo endpoints
    elif router_name == "health":
        prefix = "/api/echo"
    elif router_name == "delegation_routes":
        prefix = "/api/delegation"
    elif router_name == "notifications_api":
        prefix = "/api/notifications"
    elif router_name == "db_metrics":
        prefix = "/api/db"
    elif router_name == "git_operations":
        prefix = "/api/git"
    elif router_name == "vault":
        prefix = "/api/vault"
    elif router_name == "knowledge":
        prefix = "/api/knowledge"
    elif router_name == "tasks":
        prefix = "/api/tasks"
    elif router_name == "resilience_status":
        prefix = "/api/resilience"
    elif router_name == "google_calendar_api":
        prefix = "/api/calendar"
    elif router_name == "preferences":
        prefix = "/api/preferences"
    elif router_name == "repair_api":
        prefix = "/api/repair"
    elif router_name == "integrations":
        prefix = "/integrations"
    elif router_name == "home_assistant_api":
        prefix = "/home-assistant"
    elif router_name == "claude_bridge":
        prefix = "/api/claude"
    elif router_name == "solutions":
        prefix = "/api/echo/solutions"
    elif router_name == "system_metrics":
        prefix = "/api/echo/metrics"
    elif router_name == "training_status":
        prefix = ""  # Has its own path
    elif router_name == "secured_routes":
        prefix = ""  # Root level secured
    elif router_name == "system_stub":
        prefix = ""  # Stub, skip
    elif router_name == "takeout_stub":
        prefix = ""  # Stub, skip
    else:
        prefix = f"/api/{router_name.replace('_api', '').replace('_', '-')}"

    if router_name not in ["system_stub", "takeout_stub"]:
        includes.append(f"    app.include_router({router_name}_router, prefix=\"{prefix}\")")

print("\n=== Import statements to add ===")
for imp in imports:
    print(imp)

print("\n=== Include statements to add ===")
for inc in includes:
    print(inc)

print(f"\nTotal routers to mount: {len(includes)}")

# Save the mounting code
with open("/opt/tower-echo-brain/router_mounting_code.txt", "w") as f:
    f.write("# Add these imports after line 189 in main.py:\n\n")
    f.write("# Mount all missing routers\n")
    f.write("try:\n")
    for imp in imports:
        f.write(imp + "\n")
    f.write("\n")
    for inc in includes:
        f.write(inc + "\n")
    f.write("    logger.info(f'✅ Mounted {len(includes)} routers')\n")
    f.write("except Exception as e:\n")
    f.write("    logger.error(f'Failed to mount routers: {e}')\n")

print("\nRouter mounting code saved to router_mounting_code.txt")