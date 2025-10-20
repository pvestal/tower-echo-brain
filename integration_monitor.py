#!/usr/bin/env python3
import sys
import psutil

def check_services():
    services = {
        'echo_resilient': ('/opt/echo/echo_resilient_service.py',),
        'anime_service': ('echo_enhanced_anime_service.py',),
        'control_api': ('service_control_api.py',),
        'kb_service': ('kb_api.py',),
        'auth_service': ('auth_api.py',)
    }
    
    print('\n🔍 Echo Brain Integration Status:\n')
    print(f'{"Service":<20} {"Status":<15} {"PID":<10} {"Uptime":<15}')
    print('=' * 65)
    
    for name, paths in services.items():
        found = False
        for proc in psutil.process_iter(['pid', 'cmdline', 'create_time']):
            try:
                cmdline = proc.info['cmdline'] or []
                cmdline_str = ' '.join(cmdline)
                
                if any(path in cmdline_str for path in paths):
                    pid = proc.info['pid']
                    uptime_seconds = psutil.Process(pid).create_time()
                    from datetime import datetime
                    uptime = datetime.fromtimestamp(uptime_seconds).strftime('%b %d %H:%M')
                    
                    print(f'{name:<20} {"✅ RUNNING":<15} {pid:<10} {uptime:<15}')
                    found = True
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if not found:
            print(f'{name:<20} {"❌ STOPPED":<15} {"N/A":<10} {"N/A":<15}')
    
    print('\n' + '=' * 65)
    print('\n📊 Integration Components:\n')
    
    # Check for key files
    import os
    components = {
        'Model Manager': '/opt/tower-echo-brain/model_manager.py',
        'Resilience Service': '/opt/echo/echo_resilient_service.py',
        'Expert Personas': '/opt/tower-echo-brain/echo_expert_personas.py',
        'JWT Config': '/opt/tower-echo-brain/.env',
        'Control API': '/opt/tower-echo-brain/service_control_api.py'
    }
    
    for name, path in components.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f'✅ {name:<25} ({size:,} bytes)')
        else:
            print(f'❌ {name:<25} (NOT FOUND)')
    
    # Test JWT_SECRET
    print('\n🔐 JWT Configuration:\n')
    try:
        with open('/opt/tower-echo-brain/.env', 'r') as f:
            env_content = f.read()
            if 'JWT_SECRET=' in env_content:
                jwt_lines = [line for line in env_content.split('\n') if line.startswith('JWT_SECRET=')]
                if len(jwt_lines) == 1:
                    jwt_value = jwt_lines[0].split('=', 1)[1]
                    print(f'✅ JWT_SECRET configured: {jwt_value[:15]}...')
                else:
                    print(f'⚠️  Multiple JWT_SECRET entries found: {len(jwt_lines)}')
            else:
                print('❌ JWT_SECRET not found in .env')
    except Exception as e:
        print(f'❌ Error reading .env: {e}')
    
    print('\n' + '=' * 65)

if __name__ == '__main__':
    check_services()

def check_migration():
    """Check migration status"""
    import os
    
    migrations = {
        'Conversations DB': '/opt/tower-echo-brain/echo_migrated.db',
        'Configuration': '/opt/tower-echo-brain/echo_patrick_config.json',
        'Resilience Patterns': '/opt/tower-echo-brain/src/legacy/echo_resilient_service.py',
        'Production Service': '/opt/tower-echo-brain/production_echo_service.py',
        'Systemd Service': '/etc/systemd/system/tower-echo-brain.service',
        'Old Echo DB': '/opt/echo/echo_resilient.db',
        'Migration Script': '/opt/tower-echo-brain/migrate_echo_service.py'
    }
    
    print('\n🔧 Migration Status:\n')
    for item, path in migrations.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f'✅ {item:<25} ({size:,} bytes)')
        else:
            print(f'❌ {item:<25} (NOT FOUND)')
    
    # Check database conversation count
    try:
        import sqlite3
        conn = sqlite3.connect('/opt/tower-echo-brain/echo_migrated.db')
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM conversations')
        count = cursor.fetchone()[0]
        print(f'\n📊 Migrated Data:')
        print(f'   Conversations: {count}')
        
        cursor.execute('SELECT provider_name, total_requests FROM provider_health')
        providers = cursor.fetchall()
        for provider, requests in providers:
            print(f'   {provider}: {requests} requests')
        conn.close()
    except Exception as e:
        print(f'⚠️  Could not read migration database: {e}')

if __name__ == '__main__':
    check_services()
    check_migration()
