#!/usr/bin/env python3
"""
Migration script to move from /opt/echo/ to /opt/tower-echo-brain/
Preserves all resilience patterns and conversation history
"""
import sys
import sqlite3
import shutil
import os
from pathlib import Path

def migrate_conversations():
    """Migrate conversations from echo_resilient.db"""
    try:
        # Copy database
        shutil.copy2('/opt/echo/echo_resilient.db', '/opt/tower-echo-brain/echo_migrated.db')
        print('✅ Copied echo_resilient.db to echo_migrated.db')
        
        # Verify migration
        conn = sqlite3.connect('/opt/tower-echo-brain/echo_migrated.db')
        cursor = conn.cursor()
        
        # Get table info
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f'✅ Tables migrated: {[t[0] for t in tables]}')
        
        # Get conversation count
        cursor.execute('SELECT COUNT(*) FROM conversations')
        count = cursor.fetchone()[0]
        print(f'✅ Verified {count} conversations migrated')
        
        # Get provider health status
        cursor.execute('SELECT provider_name, status, total_requests FROM provider_health')
        providers = cursor.fetchall()
        for provider, status, requests in providers:
            print(f'   Provider: {provider} - Status: {status} - Requests: {requests}')
        
        conn.close()
        return True
    except Exception as e:
        print(f'❌ Database migration failed: {e}')
        import traceback
        traceback.print_exc()
        return False

def migrate_config():
    """Migrate configuration files"""
    try:
        # Copy configuration
        if os.path.exists('/opt/echo/echo_patrick_config.json'):
            shutil.copy2('/opt/echo/echo_patrick_config.json', '/opt/tower-echo-brain/')
            print('✅ Migrated echo_patrick_config.json')
        
        # Create src directory if it doesn't exist
        os.makedirs('/opt/tower-echo-brain/src/legacy', exist_ok=True)
        
        # Copy resilience service for reference
        shutil.copy2('/opt/echo/echo_resilient_service.py', 
                     '/opt/tower-echo-brain/src/legacy/echo_resilient_service.py')
        print('✅ Copied resilience patterns to src/legacy/')
        
        return True
    except Exception as e:
        print(f'❌ Config migration failed: {e}')
        import traceback
        traceback.print_exc()
        return False

def create_production_service():
    """Create production-ready service wrapper"""
    service_content = '''#!/usr/bin/env python3
"""
Production Echo Service - Migrated from /opt/echo/ with enhanced architecture
Combines battle-tested resilience with modular tower-echo-brain architecture
"""
import sys
import os

# Add paths
sys.path.insert(0, '/opt/tower-echo-brain')
sys.path.insert(0, '/opt/echo')

# Import resilient service from /opt/echo/
from echo_resilient_service import ResilientEcho

# Import tower-echo-brain components
try:
    from model_manager import OllamaModelManager
    from echo_expert_personas import EchoPersonalityIntegration
except ImportError as e:
    print(f'Warning: Could not import tower-echo-brain components: {e}')
    OllamaModelManager = None
    EchoPersonalityIntegration = None

class ProductionEchoService:
    """Production service combining both architectures"""
    
    def __init__(self):
        # Use the proven resilient echo from /opt/echo/
        self.resilient_echo = ResilientEcho()
        print('✅ Initialized ResilientEcho from /opt/echo/')
        
        # Optional: Initialize tower-echo-brain components if available
        if EchoPersonalityIntegration:
            try:
                self.persona = EchoPersonalityIntegration()
                print('✅ Initialized Expert Personas')
            except Exception as e:
                print(f'⚠️  Persona init failed: {e}')
                self.persona = None
        
    def run(self):
        """Start the production service"""
        print('🚀 Production Echo Service Started!')
        print('   Database: /opt/tower-echo-brain/echo_migrated.db')
        print('   Resilience: From /opt/echo/echo_resilient_service.py')
        print('   Architecture: /opt/tower-echo-brain/')
        
        # Start the resilient echo service
        self.resilient_echo.run()

if __name__ == '__main__':
    service = ProductionEchoService()
    service.run()
'''
    
    try:
        with open('/opt/tower-echo-brain/production_echo_service.py', 'w') as f:
            f.write(service_content)
        
        os.chmod('/opt/tower-echo-brain/production_echo_service.py', 0o755)
        print('✅ Created production_echo_service.py')
        return True
    except Exception as e:
        print(f'❌ Service creation failed: {e}')
        return False

if __name__ == '__main__':
    print('🚀 Starting Echo Service Migration...')
    print('=' * 60)
    
    success = True
    
    print('\n📦 Step 1: Migrating conversations and history...')
    if not migrate_conversations():
        success = False
    
    print('\n⚙️  Step 2: Migrating configuration files...')
    if not migrate_config():
        success = False
    
    print('\n🏗️  Step 3: Creating production service...')
    if not create_production_service():
        success = False
    
    print('\n' + '=' * 60)
    if success:
        print('🎉 MIGRATION SUCCESSFUL!')
        print('\n📊 What was migrated:')
        print('   ✅ Conversations and history')
        print('   ✅ Circuit breaker patterns') 
        print('   ✅ Multi-provider fallback chain')
        print('   ✅ Provider health monitoring')
        print('   ✅ Configuration and cost controls')
        print('\n🚀 Next steps:')
        print('   1. Create systemd service')
        print('   2. Stop old /opt/echo/ service')
        print('   3. Start new production service')
        print('   4. Test endpoints')
    else:
        print('❌ Migration completed with errors - check output above')
        sys.exit(1)
