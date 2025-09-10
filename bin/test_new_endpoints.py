#!/usr/bin/env python3
"""Test script for new Echo service endpoints"""

def test_imports():
    """Test that all required modules can be imported"""
    try:
        from echo_enhanced_service_updated import VaultClient, WebSearchService, CreditMonitoringService
        print("✅ All classes import successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_endpoint_definitions():
    """Test that endpoints are properly defined"""
    with open("echo_enhanced_service_updated.py", "r") as f:
        content = f.read()
    
    endpoints = [
        "/api/vault/get/{path:path}",
        "/api/echo/web-search", 
        "/api/echo/credit-monitoring",
        "/api/echo/chat"
    ]
    
    all_found = True
    for endpoint in endpoints:
        if endpoint in content:
            print(f"✅ Found endpoint: {endpoint}")
        else:
            print(f"❌ Missing endpoint: {endpoint}")
            all_found = False
    
    return all_found

def test_class_initialization():
    """Test that new classes can be instantiated"""
    try:
        from echo_enhanced_service_updated import VaultClient, WebSearchService, CreditMonitoringService
        
        vault_client = VaultClient()
        print("✅ VaultClient instantiated successfully")
        
        web_search = WebSearchService(vault_client)
        print("✅ WebSearchService instantiated successfully")
        
        credit_monitor = CreditMonitoringService()
        print("✅ CreditMonitoringService instantiated successfully")
        
        return True
    except Exception as e:
        print(f"❌ Instantiation error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Echo Service Updates...")
    print("=" * 40)
    
    test1 = test_imports()
    test2 = test_endpoint_definitions()
    test3 = test_class_initialization()
    
    if test1 and test2 and test3:
        print("\n🎉 ALL TESTS PASSED - Echo service ready for deployment!")
    else:
        print("\n❌ Some tests failed - check errors above")
