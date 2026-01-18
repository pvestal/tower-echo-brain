#!/usr/bin/env python3
import socket
import sys

def test_remote_connectivity():
    """Test if Echo can be reached remotely"""
    
    # Test local access
    print("🏠 Testing local access (192.168.50.135:8309):")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex(('192.168.50.135', 8309))
        sock.close()
        if result == 0:
            print("  ✅ Local access: WORKING")
        else:
            print("  ❌ Local access: NOT WORKING")
    except Exception as e:
        print(f"  ❌ Local access error: {e}")
    
    # Test Tailscale access
    print("\n🌐 Testing Tailscale access (100.125.174.118:8309):")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex(('100.125.174.118', 8309))
        sock.close()
        if result == 0:
            print("  ✅ Tailscale access: WORKING")
        else:
            print("  ❌ Tailscale access: NOT WORKING")
    except Exception as e:
        print(f"  ❌ Tailscale access error: {e}")
    
    # Test external access
    print("\n🌍 Testing external access (vestal-garcia.duckdns.org:8309):")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('vestal-garcia.duckdns.org', 8309))
        sock.close()
        if result == 0:
            print("  ✅ External access: WORKING")
        else:
            print("  ❌ External access: NOT WORKING (no port forwarding)")
    except Exception as e:
        print(f"  ❌ External access error: {e}")

if __name__ == '__main__':
    test_remote_connectivity()
