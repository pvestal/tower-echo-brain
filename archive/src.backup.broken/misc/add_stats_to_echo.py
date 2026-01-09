#!/usr/bin/env python3
"""
Script to add comprehensive user statistics tracking to existing AI Assist
"""

import re

def add_stats_to_echo():
    # Read the original file
    with open("echo_working.py", "r") as f:
        content = f.read()
    
    # Add sqlite3 import after existing imports
    import_section = "import uvicorn"
    sqlite_import = "import sqlite3\nfrom datetime import datetime, timedelta\nfrom fastapi import FastAPI, HTTPException, Path"
    content = content.replace("from fastapi import FastAPI, HTTPException", sqlite_import)
    content = content.replace("import uvicorn", "import uvicorn")
    
    # Add database initialization code after the story_states declaration
    db_init_code = ""
