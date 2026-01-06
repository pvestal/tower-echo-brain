#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '/opt/tower-echo-brain')

# Now import the unified API
from src.core.echo.echo_unified_api import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8309)
