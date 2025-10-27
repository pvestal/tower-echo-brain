#!/usr/bin/env python3
"""
Security Configuration Monitoring for Echo Brain
Monitors firewall, exposed ports, service bindings, and security configurations
"""

import subprocess
import logging
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SecurityIssue:
    """Represents a security configuration issue"""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    category: str  # firewall, port_exposure, service_binding, ssh_config, etc.
    title: str
    description: str
    remediation: str
    detected_at: datetime
    details: Dict = None


class SecurityConfigMonitor:
    """Monitor security configurations and detect issues"""
    
    def __init__(self):
        self.last_check = None
        self.issues_found = []
        
    async def run_all_checks(self) -> List[SecurityIssue]:
        """Run all security configuration checks"""
        issues = []
        
        try:
            # Check firewall status
            firewall_issues = await self._check_firewall()
            issues.extend(firewall_issues)
            
            # Check for exposed ports
            port_issues = await self._check_port_exposure()
            issues.extend(port_issues)
            
            # Check service bindings
            binding_issues = await self._check_service_bindings()
            issues.extend(binding_issues)
            
            # Check SSH configuration
            ssh_issues = await self._check_ssh_config()
            issues.extend(ssh_issues)
            
            # Check fail2ban status
            fail2ban_issues = await self._check_fail2ban()
            issues.extend(fail2ban_issues)
            
            self.last_check = datetime.now()
            self.issues_found = issues
            
            logger.info(f"🔒 Security check complete: {len(issues)} issues found")
            return issues
            
        except Exception as e:
            logger.error(f"❌ Error running security checks: {e}")
            return []
    
    async def _check_firewall(self) -> List[SecurityIssue]:
        """Check UFW firewall status"""
        issues = []
        
        try:
            result = subprocess.run(
                ['sudo', 'ufw', 'status'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            output = result.stdout.lower()
            
            if 'inactive' in output or 'status: inactive' in output:
                issues.append(SecurityIssue(
                    severity='CRITICAL',
                    category='firewall',
                    title='Firewall (UFW) is DISABLED',
                    description='UFW firewall is not active, leaving system exposed to network attacks',
                    remediation='Run: sudo ufw enable && sudo ufw allow 22,80,443/tcp',
                    detected_at=datetime.now(),
                    details={'ufw_status': 'inactive'}
                ))
                logger.warning("🔥 CRITICAL: UFW firewall is DISABLED")
            else:
                logger.info("✅ Firewall (UFW) is active")
                
        except subprocess.TimeoutExpired:
            logger.error("⏱️ Timeout checking firewall status")
        except Exception as e:
            logger.error(f"❌ Error checking firewall: {e}")
            
        return issues
    
    async def _check_port_exposure(self) -> List[SecurityIssue]:
        """Check for services exposed on 0.0.0.0"""
        issues = []
        
        try:
            result = subprocess.run(
                ['ss', '-tlnp'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            exposed_services = []
            for line in result.stdout.split('\n'):
                if '0.0.0.0:' in line:
                    # Extract port and service
                    match = re.search(r'0\.0\.0\.0:(\d+)', line)
                    if match:
                        port = match.group(1)
                        # Skip known safe ports
                        if port not in ['22', '80', '443', '8096', '8188', '8080', '8500', '139', '445']:
                            exposed_services.append((port, line.strip()))
            
            if exposed_services:
                for port, details in exposed_services:
                    issues.append(SecurityIssue(
                        severity='HIGH',
                        category='port_exposure',
                        title=f'Service exposed on 0.0.0.0:{port}',
                        description=f'Service listening on all interfaces instead of localhost only',
                        remediation=f'Bind service to 127.0.0.1:{port} instead of 0.0.0.0:{port}',
                        detected_at=datetime.now(),
                        details={'port': port, 'details': details}
                    ))
                    logger.warning(f"⚠️ HIGH: Service exposed on 0.0.0.0:{port}")
            else:
                logger.info("✅ No unauthorized port exposures detected")
                
        except Exception as e:
            logger.error(f"❌ Error checking port exposure: {e}")
            
        return issues
    
    async def _check_service_bindings(self) -> List[SecurityIssue]:
        """Check critical service bindings (PostgreSQL, Redis, etc.)"""
        issues = []
        
        critical_services = {
            '5432': 'PostgreSQL',
            '6379': 'Redis',
            '27017': 'MongoDB',
            '3306': 'MySQL',
            '8200': 'Vault'
        }
        
        try:
            result = subprocess.run(
                ['ss', '-tlnp'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            for port, service_name in critical_services.items():
                if f'0.0.0.0:{port}' in result.stdout:
                    issues.append(SecurityIssue(
                        severity='CRITICAL',
                        category='service_binding',
                        title=f'{service_name} exposed on 0.0.0.0:{port}',
                        description=f'{service_name} database/service should only listen on localhost',
                        remediation=f'Configure {service_name} to bind to 127.0.0.1 only',
                        detected_at=datetime.now(),
                        details={'service': service_name, 'port': port}
                    ))
                    logger.critical(f"🔥 CRITICAL: {service_name} exposed on 0.0.0.0:{port}")
                    
        except Exception as e:
            logger.error(f"❌ Error checking service bindings: {e}")
            
        return issues
    
    async def _check_ssh_config(self) -> List[SecurityIssue]:
        """Check SSH configuration for security issues"""
        issues = []
        
        try:
            with open('/etc/ssh/sshd_config', 'r') as f:
                config = f.read()
            
            # Check PermitRootLogin
            if re.search(r'^\s*PermitRootLogin\s+yes', config, re.MULTILINE):
                issues.append(SecurityIssue(
                    severity='HIGH',
                    category='ssh_config',
                    title='SSH allows root login',
                    description='PermitRootLogin is set to yes, allowing direct root SSH access',
                    remediation='Set PermitRootLogin to prohibit-password or no in /etc/ssh/sshd_config',
                    detected_at=datetime.now(),
                    details={'setting': 'PermitRootLogin yes'}
                ))
                logger.warning("⚠️ HIGH: SSH allows root login")
            
            # Check PasswordAuthentication
            if re.search(r'^\s*PasswordAuthentication\s+yes', config, re.MULTILINE):
                logger.info("ℹ️ MEDIUM: SSH password authentication enabled (consider key-only)")
            else:
                logger.info("✅ SSH configuration looks secure")
                
        except Exception as e:
            logger.error(f"❌ Error checking SSH config: {e}")
            
        return issues
    
    async def _check_fail2ban(self) -> List[SecurityIssue]:
        """Check if fail2ban is installed and running"""
        issues = []
        
        try:
            result = subprocess.run(
                ['systemctl', 'is-active', 'fail2ban'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                issues.append(SecurityIssue(
                    severity='MEDIUM',
                    category='fail2ban',
                    title='fail2ban is not running',
                    description='fail2ban service provides brute-force protection but is not active',
                    remediation='Install and enable fail2ban: sudo apt install fail2ban && sudo systemctl enable --now fail2ban',
                    detected_at=datetime.now(),
                    details={'status': 'inactive'}
                ))
                logger.warning("⚠️ MEDIUM: fail2ban is not running")
            else:
                logger.info("✅ fail2ban is running")
                
        except Exception as e:
            logger.error(f"❌ Error checking fail2ban: {e}")
            
        return issues


# Quick test
if __name__ == '__main__':
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        monitor = SecurityConfigMonitor()
        issues = await monitor.run_all_checks()
        
        print(f"\n{'='*60}")
        print(f"Security Check Results: {len(issues)} issues found")
        print(f"{'='*60}\n")
        
        for issue in issues:
            print(f"[{issue.severity}] {issue.title}")
            print(f"  Category: {issue.category}")
            print(f"  Description: {issue.description}")
            print(f"  Remediation: {issue.remediation}")
            print()
    
    asyncio.run(test())
