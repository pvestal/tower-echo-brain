#!/usr/bin/env python3
"""
Echo Brain Authentication Integration
Provides unified access to all Patrick's tokens and data
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime
import subprocess
import socket
import logging

logger = logging.getLogger(__name__)

class EchoAuthIntegration:
    def __init__(self):
        self.network_mode = self.detect_network_access()
        self.github_token = self.get_github_token()
        
    def detect_network_access(self):
        """Detect how Echo can be accessed"""
        # Check if we're accessible via different methods
        access_methods = []
        
        # Local network
        if self.test_port('127.0.0.1', 8309):
            access_methods.append('localhost')
        if self.test_port('192.168.50.135', 8309):
            access_methods.append('local_network')
            
        # Tailscale 
        if self.test_port('100.125.174.118', 8309):
            access_methods.append('tailscale')
            
        return access_methods
    
    def test_port(self, host, port, timeout=2):
        """Test if a port is accessible"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except:
            return False
    
    def get_github_token(self):
        """Get GitHub token from CLI"""
        try:
            result = subprocess.run(['gh', 'auth', 'token'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return None
    
    async def get_patrick_data_summary(self):
        """Get summary of Patrick's accessible data"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'network_access': self.network_mode,
            'available_services': {}
        }
        
        # GitHub (always available remotely)
        if self.github_token:
            summary['available_services']['github'] = {
                'status': 'authenticated',
                'token_length': len(self.github_token),
                'capabilities': ['repos', 'issues', 'actions', 'gists']
            }
        
        # Local services (only when on network)
        if 'localhost' in self.network_mode or 'local_network' in self.network_mode:
            summary['available_services'].update({
                'vault': {'status': 'accessible', 'port': 8200},
                'auth_service': {'status': 'accessible', 'port': 8088},
                'apple_music': {'status': 'accessible', 'port': 8315},
                'plaid': {'status': 'accessible', 'port': 8089}
            })
        
        # Tailscale access
        if 'tailscale' in self.network_mode:
            summary['remote_access'] = {
                'method': 'tailscale_vpn',
                'ip': '100.125.174.118',
                'security': 'encrypted_tunnel'
            }
        
        return summary
    
    def get_auth_urls_for_remote(self):
        """Get auth URLs that work when Patrick is remote"""
        if 'tailscale' in self.network_mode:
            base_url = 'http://100.125.174.118'
        elif 'local_network' in self.network_mode:
            base_url = 'http://192.168.50.135'
        else:
            return {'error': 'No remote access configured'}
            
        return {
            'google_oauth': f'{base_url}:8088/auth/google',
            'github_oauth': f'{base_url}:8088/auth/github',
            'apple_music': f'{base_url}:8315/auth',
            'plaid_link': f'{base_url}:8089/auth'
        }

# Global instance for Echo to use
echo_auth = EchoAuthIntegration()
