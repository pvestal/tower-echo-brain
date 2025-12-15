#!/usr/bin/env python3
"""
Production Echo Service
Uses symlinked database at /opt/echo/echo_resilient.db -> /opt/tower-echo-brain/echo_migrated.db
"""
import sys
sys.path.insert(0, '/opt/echo')

from src.legacy.echo_resilient_service import app
import uvicorn

if __name__ == '__main__':
    print('🚀 Production Echo Brain Service')
    print('   Running from: /opt/tower-echo-brain/')
    print('   Database: /opt/echo/echo_resilient.db -> /opt/tower-echo-brain/echo_migrated.db')
    print('   Port: 8309')
    
    uvicorn.run(app, host='0.0.0.0', port=8309)
