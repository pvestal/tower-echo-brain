#!/usr/bin/env python3
"""
Script to update Echo service with new integrations
"""

import re

def update_echo_service():
    # Read current service file
    with open("/opt/tower-echo-brain/bin/echo_enhanced_service.py", "r") as f:
        content = f.read()
    
    # 1. Add new imports after existing typing import
    imports_pattern = r"(from typing import Optional, List, Dict, Any)"
    new_imports = r"\1\nimport aiohttp\nfrom typing import Union"
    content = re.sub(imports_pattern, new_imports, content)
    
    # 2. Find where to insert the new classes (after logging setup but before app creation)
    app_pattern = r"(app = FastAPI\(title=\"Enhanced Echo Service with Tower Management\"\))"
    
    # Define the new classes
    new_classes = """
# HashiCorp Vault Client Integration
class VaultClient:
    def __init__(self):
        self.vault_addr = "http://127.0.0.1:8200"
        self.vault_token = "***REMOVED***"
        
    async def get_secret(self, path: str, field: str = None) -> Union[str, Dict]:
        \"\"\"Get secret from HashiCorp Vault\"\"\"
        try:
            headers = {"X-Vault-Token": self.vault_token}
            url = f"{self.vault_addr}/v1/{path}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        secret_data = data.get("data", {}).get("data", {})
                        return secret_data.get(field) if field else secret_data
                    else:
                        logger.error(f"Vault error {response.status} for {path}")
                        return None
        except Exception as e:
            logger.error(f"Vault connection error: {e}")
            return None

# Web Search Integration
class WebSearchService:
    def __init__(self, vault_client: VaultClient):
        self.vault = vault_client
        
    async def search_web(self, query: str, provider: str = "zillow") -> Dict[str, Any]:
        \"\"\"Search web using RapidAPI services\"\"\"
        try:
            # Get RapidAPI key from vault
            api_key = await self.vault.get_secret("oauth/rapidapi", "api_key")
            if not api_key:
                return {"error": "No RapidAPI key found in vault"}
                
            headers = {
                "X-RapidAPI-Key": api_key,
                "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
            }
            
            if provider == "zillow":
                url = f"https://zillow-com1.p.rapidapi.com/propertyExtendedSearch?location={query}"
            else:
                return {"error": f"Unknown provider: {provider}"}
                
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "success": True,
                            "provider": provider,
                            "results": data.get("props", [])[:5],  # Limit to 5 results
                            "total_found": len(data.get("props", []))
                        }
                    else:
                        return {"error": f"API error: {response.status}"}
                        
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return {"error": str(e)}

# Credit Monitoring Service
class CreditMonitoringService:
    def __init__(self):
        self.data_sources = [
            {"name": "Chase Bank", "category": "credit_card_bank", "importance": 0.95},
            {"name": "Bank of America", "category": "credit_card_bank", "importance": 0.95},
            {"name": "American Express", "category": "credit_card", "importance": 0.9},
            {"name": "Capital One", "category": "credit_card_bank", "importance": 0.9},
            {"name": "County Court Records", "category": "public_records", "importance": 0.9},
            {"name": "Austin Energy", "category": "utility", "importance": 0.3},
            {"name": "AT&T", "category": "telecom", "importance": 0.4}
        ]
    
    async def get_monitoring_strategy(self) -> Dict[str, Any]:
        \"\"\"Get credit monitoring strategy\"\"\"
        high_impact = [ds for ds in self.data_sources if ds["importance"] >= 0.8]
        
        return {
            "total_sources": len(self.data_sources),
            "high_impact_sources": len(high_impact),
            "high_impact_list": high_impact[:5],
            "coverage_recommendation": "Focus on major banks and credit cards first",
            "next_steps": [
                "Set up API monitoring for major banks",
                "Monitor public records for negative items",
                "Add utility payments for credit boost"
            ]
        }

"""
    
    # Insert classes before app creation
    content = re.sub(app_pattern, new_classes + "\n" + r"\1", content)
    
    # 3. Add service initialization after echo_service creation
    service_init_pattern = r"(echo_service = EnhancedEchoService\(\))"
    service_init = r"""\1

# Initialize new services
vault_client = VaultClient()
web_search_service = WebSearchService(vault_client)
credit_monitoring_service = CreditMonitoringService()"""
    
    content = re.sub(service_init_pattern, service_init, content)
    
    # 4. Add new endpoints before if __name__ == "__main__"
    main_pattern = r"(if __name__ == \"__main__\":)"
    
    new_endpoints = """
@app.get("/api/vault/get/{path:path}")
async def get_vault_secret(path: str, field: Optional[str] = None):
    \"\"\"Get secret from HashiCorp Vault\"\"\"
    try:
        secret = await vault_client.get_secret(path, field)
        if secret is None:
            raise HTTPException(status_code=404, detail="Secret not found")
        return {"success": True, "data": secret}
    except Exception as e:
        logger.error(f"Vault get error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/echo/web-search")
async def echo_web_search(request: Dict[str, Any]):
    \"\"\"Web search endpoint for Echo\"\"\"
    try:
        query = request.get("query", "")
        provider = request.get("provider", "zillow")
        
        if not query:
            raise HTTPException(status_code=400, detail="Query required")
            
        results = await web_search_service.search_web(query, provider)
        return results
        
    except Exception as e:
        logger.error(f"Echo web search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/echo/credit-monitoring")
async def echo_credit_monitoring():
    \"\"\"Credit monitoring strategy endpoint\"\"\"
    try:
        strategy = await credit_monitoring_service.get_monitoring_strategy()
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "credit_monitoring": strategy
        }
    except Exception as e:
        logger.error(f"Credit monitoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/echo/chat")
async def echo_chat(request: Dict[str, Any]):
    \"\"\"Enhanced chat endpoint with web search integration\"\"\"
    try:
        message = request.get("message", "")
        context = request.get("context", {})
        
        # Check if message requests web search
        if "search" in message.lower() or context.get("type") == "web_search":
            # Extract search query
            search_query = message.replace("search for", "").replace("search", "").strip()
            if not search_query:
                search_query = "Austin TX properties"
                
            search_results = await web_search_service.search_web(search_query)
            
            return {
                "success": True,
                "echo_response": f"I found {search_results.get('total_found', 0)} results for '{search_query}'",
                "search_results": search_results,
                "timestamp": datetime.now().isoformat()
            }
        
        # Default echo response
        return {
            "success": True,
            "echo_response": f"Echo received: {message}",
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Echo chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

"""
    
    content = re.sub(main_pattern, new_endpoints + "\n" + r"\1", content)
    
    # Write updated file
    with open("/opt/tower-echo-brain/bin/echo_enhanced_service_updated.py", "w") as f:
        f.write(content)
    
    print("Successfully created echo_enhanced_service_updated.py")

if __name__ == "__main__":
    update_echo_service()
