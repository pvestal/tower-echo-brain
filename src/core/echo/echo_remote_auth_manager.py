#!/usr/bin/env python3
"""
Echo Remote Authentication Manager
Handles authentication when Patrick is remote/off home network

Network scenarios:
1. On home network (192.168.50.x) - Direct access
2. Remote via Tailscale - VPN access to Tower
3. Remote via port forwarding - External access
4. Completely remote - Cloud-based tokens only
"""

import os
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import logging
from typing import Dict, List, Any, Optional, Tuple
import subprocess
import socket

logger = logging.getLogger(__name__)

class EchoRemoteAuthManager:
    """Authentication manager that works remotely and locally"""

    def __init__(self):
        self.network_mode = self.detect_network_mode()
        self.base_urls = self.get_base_urls()
        self.credentials = {}
        self.load_credentials()

    def detect_network_mode(self) -> str:
        """Detect what network mode we're in"""
        try:
            # Check if we're on home network
            if self.can_reach_ip("192.168.50.135", 8309, timeout=2):
                return "home_network"

            # Check if Tower is reachable via Tailscale
            if self.can_reach_tailscale():
                return "tailscale_vpn"

            # Check if we can reach external ports
            if self.can_reach_external():
                return "external_access"

            return "remote_only"

        except Exception as e:
            logger.warning(f"Network detection failed: {e}")
            return "unknown"

    def can_reach_ip(self, ip: str, port: int, timeout: int = 5) -> bool:
        """Test if we can reach a specific IP:port"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((ip, port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def can_reach_tailscale(self) -> bool:
        """Check if Tower is reachable via Tailscale"""
        try:
            # Check if tailscale is running
            result = subprocess.run(['tailscale', 'ip'], capture_output=True, text=True)
            if result.returncode != 0:
                return False

            # Try to get Tower's Tailscale IP
            result = subprocess.run(['tailscale', 'status'], capture_output=True, text=True)
            if 'tower' in result.stdout.lower():
                # Extract Tower's Tailscale IP and test connectivity
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'tower' in line.lower():
                        parts = line.split()
                        if len(parts) > 0:
                            tailscale_ip = parts[0]
                            return self.can_reach_ip(tailscale_ip, 8309, timeout=3)
            return False
        except Exception:
            return False

    def can_reach_external(self) -> bool:
        """Check if we can reach Tower via external/forwarded ports"""
        # These would be your external domain/IP with port forwarding
        external_hosts = [
            ("vestal-garcia.duckdns.org", 8309),
            ("192.168.50.135", 8309),  # If you have port forwarding
        ]

        for host, port in external_hosts:
            if self.can_reach_ip(host, port, timeout=3):
                return True
        return False

    def get_base_urls(self) -> Dict[str, str]:
        """Get base URLs based on network mode"""
        if self.network_mode == "home_network":
            return {
                "echo": "http://192.168.50.135:8309",
                "auth": "http://192.168.50.135:8088",
                "apple_music": "http://192.168.50.135:8315",
                "plaid": "http://192.168.50.135:8089",
                "vault": "http://192.168.50.135:8200"
            }
        elif self.network_mode == "tailscale_vpn":
            # Get actual Tailscale IP
            tailscale_ip = self.get_tailscale_tower_ip()
            return {
                "echo": f"http://{tailscale_ip}:8309",
                "auth": f"http://{tailscale_ip}:8088",
                "apple_music": f"http://{tailscale_ip}:8315",
                "plaid": f"http://{tailscale_ip}:8089",
                "vault": f"http://{tailscale_ip}:8200"
            }
        elif self.network_mode == "external_access":
            return {
                "echo": "https://vestal-garcia.duckdns.org:8309",
                "auth": "https://vestal-garcia.duckdns.org:8088",
                "apple_music": "https://vestal-garcia.duckdns.org:8315",
                "plaid": "https://vestal-garcia.duckdns.org:8089",
                "vault": None  # Vault should never be external
            }
        else:  # remote_only
            return {
                "echo": None,
                "auth": None,
                "apple_music": None,
                "plaid": None,
                "vault": None
            }

    def get_tailscale_tower_ip(self) -> str:
        """Get Tower's Tailscale IP address"""
        try:
            result = subprocess.run(['tailscale', 'status'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            for line in lines:
                if 'tower' in line.lower():
                    parts = line.split()
                    if len(parts) > 0:
                        return parts[0]
            return "192.168.50.135"  # fallback
        except Exception:
            return "192.168.50.135"

    def load_credentials(self):
        """Load credentials based on network access"""
        if self.network_mode in ["home_network", "tailscale_vpn"]:
            # Can access Tower files directly
            self.load_tower_credentials()
        else:
            # Load from local cache or cloud storage
            self.load_cached_credentials()

    def load_tower_credentials(self):
        """Load credentials directly from Tower (when accessible)"""
        try:
            # This would be an SSH call or direct file access via VPN
            import paramiko

            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            if self.network_mode == "tailscale_vpn":
                tower_ip = self.get_tailscale_tower_ip()
            else:
                tower_ip = "192.168.50.135"

            ssh.connect(tower_ip, username=os.getenv("TOWER_USER", "patrick"))

            # Get OAuth config
            sftp = ssh.open_sftp()
            with sftp.open('/opt/tower-auth/credentials/oauth_config.json') as f:
                self.credentials = json.load(f)

            ssh.close()
            logger.info(f"✅ Loaded credentials from Tower via {self.network_mode}")

        except Exception as e:
            logger.error(f"❌ Failed to load Tower credentials: {e}")
            self.load_cached_credentials()

    def load_cached_credentials(self):
        """Load credentials from local cache (for remote access)"""
        try:
            cache_file = Path.home() / '.config' / 'echo-auth' / 'credentials.json'
            if cache_file.exists():
                with open(cache_file) as f:
                    self.credentials = json.load(f)
                logger.info("✅ Loaded cached credentials")
            else:
                logger.warning("⚠️ No cached credentials found")
                self.create_minimal_credentials()
        except Exception as e:
            logger.error(f"❌ Failed to load cached credentials: {e}")

    def create_minimal_credentials(self):
        """Create minimal credentials for remote-only mode"""
        self.credentials = {
            "mode": "remote_only",
            "note": "Limited to cloud-based APIs only when remote",
            "github": {
                "use_cli": True
            },
            "google": {
                "use_oauth_flow": True
            }
        }

    async def test_connectivity(self) -> Dict[str, Any]:
        """Test connectivity to all services"""
        results = {
            "network_mode": self.network_mode,
            "connectivity": {},
            "recommendations": []
        }

        # Test each service
        for service, url in self.base_urls.items():
            if url:
                try:
                    timeout = aiohttp.ClientTimeout(total=5)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.get(f"{url}/health") as response:
                            if response.status == 200:
                                results["connectivity"][service] = "reachable"
                            else:
                                results["connectivity"][service] = f"error_{response.status}"
                except Exception as e:
                    results["connectivity"][service] = f"unreachable: {str(e)}"
            else:
                results["connectivity"][service] = "not_configured"

        # Generate recommendations
        results["recommendations"] = self.generate_remote_recommendations(results["connectivity"])

        return results

    def generate_remote_recommendations(self, connectivity: Dict[str, str]) -> List[str]:
        """Generate recommendations for improving remote access"""
        recommendations = []

        if self.network_mode == "remote_only":
            recommendations.extend([
                "🌐 Consider setting up Tailscale VPN for secure remote access",
                "🔧 Set up port forwarding for key services (8309, 8088)",
                "☁️ Use cloud-based authentication when possible",
                "💾 Cache tokens locally for offline use"
            ])
        elif self.network_mode == "external_access":
            recommendations.extend([
                "🔒 Ensure HTTPS is configured for external access",
                "🛡️ Consider IP whitelisting for sensitive services",
                "🔑 Use strong authentication for exposed services"
            ])
        elif self.network_mode == "tailscale_vpn":
            recommendations.append("✅ Tailscale VPN provides secure access")

        # Service-specific recommendations
        unreachable_services = [k for k, v in connectivity.items() if "unreachable" in v]
        if unreachable_services:
            recommendations.append(f"🔧 Configure access for: {', '.join(unreachable_services)}")

        return recommendations

    async def get_github_data_remote(self) -> Dict[str, Any]:
        """Get GitHub data - works remotely via GitHub CLI or API"""
        try:
            # GitHub CLI works remotely
            token_result = subprocess.run(['gh', 'auth', 'token'],
                                        capture_output=True, text=True)
            if token_result.returncode == 0:
                token = token_result.stdout.strip()

                headers = {"Authorization": f"token {token}"}
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        "https://api.github.com/user/repos?sort=updated&per_page=20",
                        headers=headers
                    ) as response:
                        if response.status == 200:
                            repos = await response.json()
                            return {
                                "repositories": repos,
                                "source": "github_api_remote",
                                "status": "success"
                            }
            return {"error": "GitHub authentication failed"}
        except Exception as e:
            return {"error": f"GitHub remote access failed: {e}"}

    async def get_google_data_remote(self) -> Dict[str, Any]:
        """Get Google data - requires OAuth flow when remote"""
        # When remote, would need to implement OAuth flow
        # or use cached tokens
        return {
            "message": "Google Calendar access requires OAuth setup when remote",
            "status": "needs_oauth",
            "oauth_url": "https://accounts.google.com/o/oauth2/auth"
        }

    async def collect_remote_accessible_data(self) -> Dict[str, Any]:
        """Collect only data that's accessible remotely"""
        logger.info(f"🌐 Collecting data in {self.network_mode} mode...")

        # Test connectivity first
        connectivity = await self.test_connectivity()

        # Collect data from accessible services
        data_collection = {}

        # GitHub always works remotely
        data_collection["github"] = await self.get_github_data_remote()

        # Google requires OAuth setup when remote
        data_collection["google"] = await self.get_google_data_remote()

        # Tower services only if accessible
        if connectivity["connectivity"].get("echo") == "reachable":
            data_collection["echo_status"] = "accessible"
        else:
            data_collection["echo_status"] = "not_accessible"

        return {
            "network_analysis": connectivity,
            "collected_data": data_collection,
            "remote_mode": self.network_mode,
            "timestamp": datetime.utcnow().isoformat()
        }

    def setup_remote_access_guide(self) -> Dict[str, Any]:
        """Generate a guide for setting up remote access"""
        return {
            "title": "Echo Remote Access Setup Guide",
            "current_mode": self.network_mode,
            "options": {
                "tailscale_vpn": {
                    "description": "Secure VPN access to home network",
                    "setup": [
                        "Install Tailscale on remote device",
                        "Login with Tailscale account",
                        "Verify Tower is accessible via Tailscale IP"
                    ],
                    "benefits": ["Secure", "Full access", "No port forwarding needed"]
                },
                "port_forwarding": {
                    "description": "Forward specific ports through router",
                    "setup": [
                        "Configure router to forward ports 8309, 8088, 8315",
                        "Set up HTTPS certificates",
                        "Configure firewall rules"
                    ],
                    "benefits": ["Direct access", "No VPN required"],
                    "risks": ["Security exposure", "Need proper authentication"]
                },
                "cloud_hybrid": {
                    "description": "Use cloud APIs when Tower not accessible",
                    "setup": [
                        "Set up OAuth for Google/GitHub",
                        "Cache tokens locally",
                        "Use GitHub CLI for repository access"
                    ],
                    "benefits": ["Always available", "No network dependencies"]
                }
            },
            "recommendations": self.generate_remote_recommendations({})
        }

# Test function
async def test_remote_auth():
    """Test the remote auth manager"""
    print("🌐 Echo Remote Authentication Manager Test")
    print("=" * 60)

    auth = EchoRemoteAuthManager()

    print(f"🔍 Detected Network Mode: {auth.network_mode}")
    print(f"🌐 Base URLs: {json.dumps(auth.base_urls, indent=2)}")

    # Test connectivity
    connectivity = await auth.test_connectivity()
    print(f"\n📡 Connectivity Test:")
    print(f"  Network Mode: {connectivity['network_mode']}")
    print(f"  Service Status:")
    for service, status in connectivity['connectivity'].items():
        status_icon = "✅" if status == "reachable" else "❌"
        print(f"    {status_icon} {service}: {status}")

    print(f"\n💡 Recommendations:")
    for rec in connectivity['recommendations']:
        print(f"    {rec}")

    # Collect accessible data
    print(f"\n📊 Collecting accessible data...")
    data = await auth.collect_remote_accessible_data()

    print(f"  GitHub: {'✅' if data['collected_data']['github'].get('status') == 'success' else '❌'}")
    print(f"  Echo Status: {data['collected_data']['echo_status']}")

    # Show setup guide
    guide = auth.setup_remote_access_guide()
    print(f"\n📖 Remote Access Options:")
    for option, details in guide['options'].items():
        print(f"  🔧 {option}: {details['description']}")

if __name__ == "__main__":
    asyncio.run(test_remote_auth())