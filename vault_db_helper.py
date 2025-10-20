#!/usr/bin/env python3
"""
Database credentials helper with Vault fallback
Reads from Vault if available, falls back to environment variables
"""

import os
import psycopg2

def get_db_config():
    """Get database configuration from Vault or environment variables"""
    
    # Try Vault first
    try:
        import hvac
        vault_addr = os.environ.get('VAULT_ADDR', 'http://127.0.0.1:8200')
        vault_token = os.environ.get('VAULT_TOKEN')
        
        if vault_token:
            client = hvac.Client(url=vault_addr, token=vault_token)
            
            if client.is_authenticated():
                secret = client.secrets.kv.v2.read_secret_version(path='tower/database')
                data = secret['data']['data']
                
                return {
                    "host": data.get('host', 'localhost'),
                    "database": data.get('database', 'echo_brain'),
                    "user": data.get('user', 'patrick'),
                    "password": data.get('password')
                }
    except Exception as e:
        # Vault not available, fall back to environment
        pass
    
    # Fallback to environment variables
    return {
        "host": os.environ.get("DB_HOST", "localhost"),
        "database": os.environ.get("DB_NAME", "echo_brain"),
        "user": os.environ.get("DB_USER", "patrick"),
        "password": os.environ.get("DB_PASSWORD")
    }

def test_connection():
    """Test database connection"""
    config = get_db_config()
    
    try:
        conn = psycopg2.connect(**config)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        cursor.close()
        conn.close()
        print(f"✅ Database connection successful!")
        print(f"PostgreSQL version: {version[0]}")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()
