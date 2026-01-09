#!/usr/bin/env python3
"""
Echo Brain Entry Point

This is the main entry point for running Echo Brain.
It creates and runs the FastAPI application from app_factory.
"""

import os
import sys
import uvicorn

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.app_factory import create_app

# Create the FastAPI application
app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("ECHO_PORT", "8309"))
    host = os.getenv("ECHO_HOST", "0.0.0.0")

    print(f"🧠 Starting Echo Brain on {host}:{port}")

    uvicorn.run(
        "echo:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
