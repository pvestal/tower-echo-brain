#!/usr/bin/env python3
"""
Enhanced System Metrics API - Comprehensive hardware monitoring
Includes temperatures, fan speeds, disk I/O, network stats, and advanced visualizations
"""

import psutil
import subprocess
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from fastapi import APIRouter, HTTPException
import time

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


def get_nvidia_detailed_stats() -> Dict:
    """Get detailed NVIDIA GPU statistics including temperature and fan speed"""
    try:
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=temperature.gpu,fan.speed,power.draw,utilization.gpu,utilization.memory,memory.used,memory.total,name',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)

        if result.returncode == 0:
            data = result.stdout.strip().split(', ')
            if len(data) >= 8:
                return {
                    "temperature_c": int(data[0]),
                    "fan_speed_percent": int(data[1]) if data[1] != '[N/A]' else 0,
                    "power_draw_w": float(data[2]),
                    "gpu_utilization_percent": int(data[3]),
                    "memory_utilization_percent": int(data[4]),
                    "memory_used_mb": int(data[5]),
                    "memory_total_mb": int(data[6]),
                    "gpu_name": data[7],
                    "thermal_state": get_thermal_state(int(data[0])),
                    "power_efficiency": round(int(data[3]) / float(data[2]) if float(data[2]) > 0 else 0, 2)
                }
    except Exception as e:
        logger.error(f"Failed to get NVIDIA detailed stats: {e}")

    return {}


def get_thermal_state(temp_c: int) -> str:
    """Determine thermal state based on temperature"""
    if temp_c < 40:
        return "cool"
    elif temp_c < 60:
        return "warm"
    elif temp_c < 75:
        return "hot"
    else:
        return "critical"


def get_amd_gpu_stats() -> Dict:
    """Get AMD GPU statistics using rocm-smi"""
    try:
        result = subprocess.run(['rocm-smi', '--showtemp', '--showpower', '--showuse'],
                              capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            output = result.stdout
            temp_match = re.search(r'Temperature:\s*(\d+)', output)
            power_match = re.search(r'Power:\s*([\d.]+)', output)
            util_match = re.search(r'GPU use:\s*(\d+)', output)

            temp = int(temp_match.group(1)) if temp_match else 0
            power = float(power_match.group(1)) if power_match else 0
            utilization = int(util_match.group(1)) if util_match else 0

            return {
                "temperature_c": temp,
                "power_draw_w": power,
                "gpu_utilization_percent": utilization,
                "thermal_state": get_thermal_state(temp),
                "gpu_name": "AMD RX 9070 XT"
            }
    except Exception as e:
        logger.error(f"Failed to get AMD GPU stats: {e}")

    return {}


def get_disk_stats() -> List[Dict]:
    """Get comprehensive disk statistics"""
    disks = []

    # Get disk usage
    try:
        disk_usage = psutil.disk_usage('/')
        main_disk = {
            "mount_point": "/",
            "device": "nvme0n1p2",
            "total_gb": round(disk_usage.total / (1024**3), 1),
            "used_gb": round(disk_usage.used / (1024**3), 1),
            "free_gb": round(disk_usage.free / (1024**3), 1),
            "usage_percent": round(disk_usage.used / disk_usage.total * 100, 1),
            "disk_type": "NVMe SSD"
        }
        disks.append(main_disk)

        # Additional storage drives
        for mount_point, device, disk_type in [
            ("/mnt/1TB-storage", "sdc2", "SATA SSD"),
            ("/mnt/10TB2", "sdb1", "SATA HDD")
        ]:
            try:
                usage = psutil.disk_usage(mount_point)
                disks.append({
                    "mount_point": mount_point,
                    "device": device,
                    "total_gb": round(usage.total / (1024**3), 1),
                    "used_gb": round(usage.used / (1024**3), 1),
                    "free_gb": round(usage.free / (1024**3), 1),
                    "usage_percent": round(usage.used / usage.total * 100, 1),
                    "disk_type": disk_type
                })
            except:
                pass

    except Exception as e:
        logger.error(f"Failed to get disk stats: {e}")

    # Get I/O statistics
    try:
        disk_io = psutil.disk_io_counters(perdisk=True)
        for disk in disks:
            device = disk["device"].split("/")[-1]  # Extract device name
            if device in disk_io:
                io_stats = disk_io[device]
                disk.update({
                    "read_mb_total": round(io_stats.read_bytes / (1024**2), 1),
                    "write_mb_total": round(io_stats.write_bytes / (1024**2), 1),
                    "read_count": io_stats.read_count,
                    "write_count": io_stats.write_count
                })
    except Exception as e:
        logger.error(f"Failed to get disk I/O stats: {e}")

    return disks


def get_network_stats() -> Dict:
    """Get network interface statistics"""
    try:
        net_io = psutil.net_io_counters(pernic=True)
        network_stats = {}

        # Focus on active interfaces
        for interface, stats in net_io.items():
            if not interface.startswith(('lo', 'docker', 'br-', 'veth')):
                network_stats[interface] = {
                    "bytes_sent_mb": round(stats.bytes_sent / (1024**2), 1),
                    "bytes_recv_mb": round(stats.bytes_recv / (1024**2), 1),
                    "packets_sent": stats.packets_sent,
                    "packets_recv": stats.packets_recv,
                    "errors_in": stats.errin,
                    "errors_out": stats.errout,
                    "interface_type": "ethernet" if "eth" in interface or "en" in interface else "wireless" if "wl" in interface else "other"
                }

        # Get network connections
        connections = len(psutil.net_connections())

        return {
            "interfaces": network_stats,
            "active_connections": connections,
            "total_sent_mb": round(sum(iface["bytes_sent_mb"] for iface in network_stats.values()), 1),
            "total_recv_mb": round(sum(iface["bytes_recv_mb"] for iface in network_stats.values()), 1)
        }

    except Exception as e:
        logger.error(f"Failed to get network stats: {e}")
        return {"interfaces": {}, "active_connections": 0, "total_sent_mb": 0, "total_recv_mb": 0}


def get_process_stats() -> Dict:
    """Get top processes and system activity"""
    try:
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                process_info = proc.info
                if process_info['cpu_percent'] > 0.5:  # Only include active processes
                    processes.append({
                        "pid": process_info['pid'],
                        "name": process_info['name'][:15],  # Truncate name
                        "cpu_percent": round(process_info['cpu_percent'], 1),
                        "memory_percent": round(process_info['memory_percent'], 1)
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Sort by CPU usage and limit to top 5
        processes.sort(key=lambda x: x['cpu_percent'], reverse=True)

        return {
            "top_processes": processes[:5],
            "total_processes": len(psutil.pids()),
            "cpu_cores": psutil.cpu_count(),
            "cpu_freq_mhz": round(psutil.cpu_freq().current) if psutil.cpu_freq() else 0
        }

    except Exception as e:
        logger.error(f"Failed to get process stats: {e}")
        return {"top_processes": [], "total_processes": 0, "cpu_cores": 0, "cpu_freq_mhz": 0}


def get_system_uptime() -> Dict:
    """Get system uptime and load averages"""
    try:
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = datetime.now() - boot_time

        # Get load averages (Linux)
        try:
            load1, load5, load15 = psutil.getloadavg()
        except AttributeError:
            load1 = load5 = load15 = 0

        return {
            "uptime_days": uptime.days,
            "uptime_hours": uptime.seconds // 3600,
            "uptime_minutes": (uptime.seconds % 3600) // 60,
            "boot_time": boot_time.isoformat(),
            "load_avg_1min": round(load1, 2),
            "load_avg_5min": round(load5, 2),
            "load_avg_15min": round(load15, 2),
            "users_logged_in": len(psutil.users())
        }

    except Exception as e:
        logger.error(f"Failed to get uptime stats: {e}")
        return {"uptime_days": 0, "uptime_hours": 0, "uptime_minutes": 0, "load_avg_1min": 0}


@router.get("/api/system/enhanced-metrics")
async def get_enhanced_system_metrics():
    """
    Get comprehensive system metrics including:
    - CPU, Memory, GPU (NVIDIA + AMD)
    - Disk usage and I/O statistics
    - Network interface statistics
    - Process information
    - System uptime and load averages
    - Temperature and power monitoring
    """
    try:
        # Basic system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        # Gather enhanced metrics
        nvidia_stats = get_nvidia_detailed_stats()
        amd_stats = get_amd_gpu_stats()
        disk_stats = get_disk_stats()
        network_stats = get_network_stats()
        process_stats = get_process_stats()
        uptime_stats = get_system_uptime()

        # Calculate health scores
        cpu_health = "good" if cpu_percent < 80 else "warning" if cpu_percent < 95 else "critical"
        memory_health = "good" if memory.percent < 80 else "warning" if memory.percent < 95 else "critical"

        gpu_health = "good"
        if nvidia_stats:
            if nvidia_stats["temperature_c"] > 80:
                gpu_health = "critical"
            elif nvidia_stats["temperature_c"] > 65:
                gpu_health = "warning"

        return {
            "basic_metrics": {
                "cpu_percent": round(cpu_percent, 1),
                "memory_percent": round(memory.percent, 1),
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "swap_percent": round(psutil.swap_memory().percent, 1)
            },
            "gpu_metrics": {
                "nvidia": nvidia_stats,
                "amd": amd_stats
            },
            "disk_metrics": {
                "disks": disk_stats,
                "total_storage_tb": round(sum(disk["total_gb"] for disk in disk_stats) / 1024, 1),
                "total_used_tb": round(sum(disk["used_gb"] for disk in disk_stats) / 1024, 1)
            },
            "network_metrics": network_stats,
            "process_metrics": process_stats,
            "system_info": {
                **uptime_stats,
                "hostname": psutil.users()[0].name if psutil.users() else "tower",
                "platform": "linux"
            },
            "health_status": {
                "cpu": cpu_health,
                "memory": memory_health,
                "gpu": gpu_health,
                "overall": "critical" if "critical" in [cpu_health, memory_health, gpu_health] else
                         "warning" if "warning" in [cpu_health, memory_health, gpu_health] else "good"
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting enhanced system metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced metrics error: {str(e)}")


@router.get("/api/system/thermal-status")
async def get_thermal_status():
    """Get thermal monitoring data for temperature visualization"""
    try:
        thermal_data = {}

        # NVIDIA GPU temperature
        nvidia_stats = get_nvidia_detailed_stats()
        if nvidia_stats:
            thermal_data["nvidia_gpu"] = {
                "temperature": nvidia_stats["temperature_c"],
                "thermal_state": nvidia_stats["thermal_state"],
                "fan_speed": nvidia_stats["fan_speed_percent"],
                "power_draw": nvidia_stats["power_draw_w"]
            }

        # AMD GPU temperature
        amd_stats = get_amd_gpu_stats()
        if amd_stats:
            thermal_data["amd_gpu"] = {
                "temperature": amd_stats["temperature_c"],
                "thermal_state": amd_stats["thermal_state"],
                "power_draw": amd_stats["power_draw_w"]
            }

        # CPU temperature (if available)
        try:
            # Try to get CPU temperature from thermal zones
            result = subprocess.run(['cat', '/sys/class/thermal/thermal_zone0/temp'],
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                cpu_temp = int(result.stdout.strip()) / 1000  # Convert from millidegrees
                thermal_data["cpu"] = {
                    "temperature": round(cpu_temp, 1),
                    "thermal_state": get_thermal_state(int(cpu_temp))
                }
        except:
            pass

        return {
            "thermal_zones": thermal_data,
            "overall_thermal_state": max([zone.get("thermal_state", "cool") for zone in thermal_data.values()],
                                       key=lambda x: {"cool": 0, "warm": 1, "hot": 2, "critical": 3}.get(x, 0)),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting thermal status: {e}")
        raise HTTPException(status_code=500, detail=f"Thermal status error: {str(e)}")