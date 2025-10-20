#!/usr/bin/env python3
import sys
import requests
import sqlite3
import psutil
from datetime import datetime

class ProductionDashboard:
    def __init__(self):
        self.service_url = 'http://localhost:8309/api/echo'
        
    def check_service_health(self):
        try:
            response = requests.get(f'{self.service_url}/health', timeout=5)
            return response.status_code == 200, response.json() if response.status_code == 200 else {}
        except Exception as e:
            return False, {'error': str(e)}
    
    def check_database(self):
        try:
            conn = sqlite3.connect('/opt/tower-echo-brain/echo_migrated.db')
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM conversations')
            conv_count = cursor.fetchone()[0]
            cursor.execute('SELECT provider_name, total_requests, failure_count FROM provider_health')
            providers = cursor.fetchall()
            conn.close()
            return True, conv_count, providers
        except Exception as e:
            return False, 0, []
    
    def check_system_resources(self):
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline'] or []
                if 'production_echo_service.py' in ' '.join(cmdline):
                    p = psutil.Process(proc.info['pid'])
                    return {
                        'pid': proc.info['pid'],
                        'memory_mb': p.memory_info().rss / 1024 / 1024,
                        'cpu_percent': p.cpu_percent(interval=0.1),
                        'status': p.status()
                    }
            except:
                continue
        return None
    
    def generate_report(self):
        print(f'🏥 Echo Brain Production Dashboard - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print('=' * 70)
        
        # Service Health
        healthy, health_data = self.check_service_health()
        status = '✅ HEALTHY' if healthy else '❌ UNHEALTHY'
        print(f'Service Status:    {status}')
        
        if healthy:
            print(f'   Version: {health_data.get("version", "Unknown")}')
            print(f'   Fallback Ready: {health_data.get("fallback_ready", False)}')
        
        # Database
        db_ok, conv_count, providers = self.check_database()
        db_status = '✅ OK' if db_ok else '❌ ISSUES'
        print(f'\nDatabase:          {db_status}')
        print(f'   Conversations: {conv_count}')
        print(f'   Providers monitored: {len(providers)}')
        
        if providers:
            print('\n📊 Provider Statistics:')
            for name, requests, failures in providers:
                success_rate = ((requests - failures) / requests * 100) if requests > 0 else 0
                print(f'   • {name:15} {requests:6} requests | {failures:3} failures | {success_rate:.1f}% success')
        
        # System Resources
        resources = self.check_system_resources()
        if resources:
            print(f'\n💾 System Resources:')
            print(f'   PID: {resources["pid"]}')
            print(f'   Memory: {resources["memory_mb"]:.1f} MB')
            print(f'   CPU: {resources["cpu_percent"]:.1f}%')
            print(f'   Status: {resources["status"]}')
        
        # Provider Status from health endpoint
        if healthy and 'providers' in health_data:
            print('\n🔧 Live Provider Status:')
            for provider in health_data['providers']:
                state = provider.get('circuit_state', 'UNKNOWN')
                failures = provider.get('failures', 0)
                emoji = '✅' if state == 'CLOSED' else '⚠️' if state == 'HALF_OPEN' else '❌'
                print(f'   {emoji} {provider["name"]:15} {state:10} (failures: {failures})')
        
        print(f'\n🎯 Overall Status: {"✅ FULLY OPERATIONAL" if healthy and db_ok else "❌ NEEDS ATTENTION"}')
        print('=' * 70)

if __name__ == '__main__':
    dashboard = ProductionDashboard()
    dashboard.generate_report()
