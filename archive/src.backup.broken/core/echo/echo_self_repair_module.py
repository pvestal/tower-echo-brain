#!/usr/bin/env python3
"""
Echo Self-Repair Module
Gives Echo the ability to diagnose and repair himself
"""

import subprocess
import json
import asyncio
import aiohttp
from typing import Dict, List, Tuple

class EchoSelfRepair:
    """Echo Brain Self-Diagnostic and Repair System"""
    
    def __init__(self):
        self.expected_services = {
            "echo": {"port": 8309, "process": "echo.py", "critical": True},
            "dashboard": {"port": 8080, "process": "spa-server.py", "critical": True},
            "echo-viz": {"port": 8313, "process": "cognitive_streaming", "critical": False},
            "comfyui": {"port": 8188, "process": "main.py", "critical": False},
            "voice": {"port": 8312, "process": "voice_websocket", "critical": False},
            "kb": {"port": 8307, "process": "kb.py", "critical": True}
        }
        
    async def self_diagnose(self) -> Dict:
        """Complete self-diagnostic of Echo system"""
        diagnosis = {
            "timestamp": asyncio.get_running_loop().time(),
            "services": {},
            "nginx_status": await self.check_nginx_health(),
            "system_health": "unknown",
            "repair_actions": []
        }
        
        for service_name, config in self.expected_services.items():
            service_status = await self.check_service_health(service_name, config)
            diagnosis["services"][service_name] = service_status
            
            if not service_status["healthy"] and config["critical"]:
                diagnosis["repair_actions"].append({
                    "service": service_name,
                    "issue": service_status["issue"],
                    "action": "restart_service"
                })
        
        # Overall health assessment
        critical_services_down = sum(1 for name, config in self.expected_services.items() 
                                   if config["critical"] and not diagnosis["services"][name]["healthy"])
        
        if critical_services_down == 0:
            diagnosis["system_health"] = "healthy"
        elif critical_services_down <= 2:
            diagnosis["system_health"] = "degraded"
        else:
            diagnosis["system_health"] = "critical"
            
        return diagnosis
    
    async def check_service_health(self, service_name: str, config: Dict) -> Dict:
        """Check if a specific service is healthy"""
        port = config["port"]
        process_name = config["process"]
        
        # Check if port is listening
        port_listening = await self.is_port_listening(port)
        
        # Check if process is running
        process_running = await self.is_process_running(process_name)
        
        # Check if service responds to HTTP
        http_responsive = await self.test_http_endpoint(port)
        
        healthy = port_listening and process_running and http_responsive
        
        issue = None
        if not process_running:
            issue = "process_not_running"
        elif not port_listening:
            issue = "port_not_listening"  
        elif not http_responsive:
            issue = "http_not_responsive"
            
        return {
            "healthy": healthy,
            "port_listening": port_listening,
            "process_running": process_running,
            "http_responsive": http_responsive,
            "issue": issue
        }
    
    async def is_port_listening(self, port: int) -> bool:
        """Check if port is listening"""
        try:
            result = subprocess.run(
                ["lsof", "-i", f":{port}"],
                capture_output=True, text=True
            )
            return result.returncode == 0 and "LISTEN" in result.stdout
        except:
            return False
    
    async def is_process_running(self, process_name: str) -> bool:
        """Check if process is running"""
        try:
            result = subprocess.run(
                ["pgrep", "-f", process_name],
                capture_output=True, text=True
            )
            return result.returncode == 0
        except:
            return False
    
    async def test_http_endpoint(self, port: int) -> bool:
        """Test if HTTP endpoint responds"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://127.0.0.1:{port}/", 
                    timeout=aiohttp.ClientTimeout(total=3)
                ) as response:
                    return response.status < 500
        except:
            return False
    
    async def check_nginx_health(self) -> Dict:
        """Check nginx configuration and health"""
        try:
            # Test nginx config
            config_result = subprocess.run(
                ["sudo", "nginx", "-t"],
                capture_output=True, text=True
            )
            
            # Check nginx status
            status_result = subprocess.run(
                ["systemctl", "status", "nginx"],
                capture_output=True, text=True
            )
            
            return {
                "config_valid": config_result.returncode == 0,
                "service_running": "active (running)" in status_result.stdout,
                "config_errors": config_result.stderr if config_result.returncode != 0 else None
            }
        except:
            return {
                "config_valid": False,
                "service_running": False,
                "config_errors": "Unable to check nginx"
            }
    
    async def self_repair(self, diagnosis: Dict) -> Dict:
        """Attempt to repair identified issues"""
        repair_results = []
        
        for action in diagnosis["repair_actions"]:
            service = action["service"]
            issue = action["issue"] 
            
            if action["action"] == "restart_service":
                result = await self.restart_service(service, issue)
                repair_results.append({
                    "service": service,
                    "action": "restart",
                    "success": result["success"],
                    "details": result["details"]
                })
        
        return {
            "repairs_attempted": len(repair_results),
            "repairs_successful": sum(1 for r in repair_results if r["success"]),
            "repair_details": repair_results
        }
    
    async def restart_service(self, service_name: str, issue: str) -> Dict:
        """Restart a specific service"""
        config = self.expected_services[service_name]
        
        try:
            if service_name == "echo":
                # Special handling for Echo (self-restart)
                subprocess.Popen([
                    "bash", "-c",
                    "sleep 2 && cd /opt/tower-echo-brain && export JWT_SECRET=tower_echo_secret_2025 && python3 echo.py &"
                ])
                return {"success": True, "details": "Echo self-restart initiated"}
                
            elif service_name == "dashboard":
                # Restart dashboard service
                subprocess.run([
                    "bash", "-c", 
                    "cd /opt/tower-dashboard && python3 spa-server.py &"
                ], check=True)
                return {"success": True, "details": "Dashboard restarted"}
                
            else:
                return {"success": False, "details": f"No restart procedure for {service_name}"}
                
        except Exception as e:
            return {"success": False, "details": f"Restart failed: {str(e)}"}

# Global instance for Echo to use
echo_self_repair = EchoSelfRepair()

    async def execute_system_command(self, command: str) -> Dict:
        """Actually execute system commands (REAL execution power for Echo)"""
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": str(e)
            }
    
    async def actually_repair_now(self) -> Dict:
        """Execute real repairs with actual system commands"""
        diagnosis = await self.self_diagnose()
        repair_results = []
        
        # Kill any Echo processes on wrong ports
        if not diagnosis["services"]["echo"]["healthy"]:
            kill_result = await self.execute_system_command("pkill -f echo.py")
            repair_results.append({"action": "kill_old_echo", "result": kill_result})
        
        # Start Echo with proper config
        start_cmd = "cd /opt/tower-echo-brain && export JWT_SECRET=tower_echo_secret_2025 && python3 echo.py &"
        start_result = await self.execute_system_command(start_cmd)
        repair_results.append({"action": "start_echo", "result": start_result})
        
        return {
            "diagnosis": diagnosis,
            "repairs_executed": repair_results,
            "echo_has_real_power": True
        }

# Enhanced global instance with REAL execution power
echo_self_repair.execute_system_command = echo_self_repair.__class__.execute_system_command
echo_self_repair.actually_repair_now = echo_self_repair.__class__.actually_repair_now
