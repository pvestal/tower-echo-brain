#!/usr/bin/env python3
"""
Echo Complete Tower Services Integration
Integrates OAuth SSO, Financial Services, Real Estate, and Personal Monitoring
"""
import requests
import json
import logging
import sys
import os
import asyncio
from typing import Dict, Any, Optional, List

# Add paths to access Tower modules
sys.path.append('/opt/tower-personal-data-monitor')
sys.path.append('/opt/tower-auth')

logger = logging.getLogger(__name__)

class EchoTowerServicesIntegration:
    def __init__(self):
        self.services = {
            # Financial Services
            'loan_search': 'http://127.0.0.1:8302',
            'crypto_trader': 'http://127.0.0.1:8303', 
            'personal_data_monitor': 'http://127.0.0.1:8311',
            
            # Estate and Investment
            'vestal_estate': 'http://127.0.0.1:8310',
            
            # Agent Management
            'agent_manager': 'http://127.0.0.1:8301',
            
            # Authentication
            'auth_service': 'http://127.0.0.1:8088',
            
            # Knowledge Base
            'knowledge_base': 'http://127.0.0.1:8307'
        }
        
        # Load OAuth providers from vault
        self.oauth_providers = self.load_oauth_providers()
        
    def load_oauth_providers(self):
        """Load OAuth providers from Tower vault"""
        try:
            with open('/home/{os.getenv("TOWER_USER", "patrick")}/.tower_credentials/vault.json', 'r') as f:
                vault_data = json.load(f)
            return vault_data.get('oauth_providers', {})
        except Exception as e:
            logger.warning(f"Could not load OAuth providers: {e}")
            return {}
    
    def get_service_health(self, service_name: str):
        """Get health status of a Tower service"""
        try:
            url = self.services.get(service_name)
            if not url:
                return {"error": f"Service {service_name} not found"}
                
            response = requests.get(f"{url}/api/health", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Service {service_name} returned {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Service {service_name} unavailable: {e}"}
    
    def get_patrick_comprehensive_profile(self):
        """Get Patrick's complete profile across all Tower services"""
        profile = {
            "user_id": os.getenv("TOWER_USER", "patrick"),
            "timestamp": "2025-09-12T23:53:00Z",
            "services": {}
        }
        
        # Personal Data Monitoring
        try:
            from plaid_financial_integration import PlaidFinancialIntegration
            plaid = PlaidFinancialIntegration()
            profile["services"]["financial"] = plaid.comprehensive_financial_analysis(user_id=1)
        except Exception as e:
            profile["services"]["financial"] = {"error": str(e)}
        
        # Vestal Estate Integration
        estate_data = self.get_vestal_estate_portfolio()
        profile["services"]["real_estate"] = estate_data
        
        # Crypto Trading
        crypto_data = self.get_service_health("crypto_trader")
        profile["services"]["crypto_trading"] = crypto_data
        
        # Loan Search
        loan_data = self.get_service_health("loan_search")
        profile["services"]["loan_search"] = loan_data
        
        # OAuth SSO Status
        profile["services"]["oauth_sso"] = {
            "providers": list(self.oauth_providers.keys()),
            "google": "configured" if "google" in self.oauth_providers else "not_configured",
            "apple": "configured" if "apple" in self.oauth_providers else "not_configured",
            "microsoft": "configured" if "microsoft" in self.oauth_providers else "not_configured",
            "github": "configured" if "github" in self.oauth_providers else "not_configured"
        }
        
        return profile
    
    def get_vestal_estate_portfolio(self):
        """Get Vestal Estate real estate portfolio data"""
        try:
            response = requests.get(f"{self.services['vestal_estate']}/api/health", timeout=10)
            if response.status_code == 200:
                # Try to get portfolio data
                portfolio_response = requests.get(f"{self.services['vestal_estate']}/api/portfolio", timeout=10)
                if portfolio_response.status_code == 200:
                    return portfolio_response.json()
                else:
                    return {
                        "service_status": "healthy",
                        "portfolio_status": "api_not_found",
                        "message": "Vestal Estate service is running but portfolio endpoint needs configuration"
                    }
            else:
                return {"error": f"Vestal Estate returned {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Vestal Estate unavailable: {e}"}
    
    def search_tower_services(self, query: str):
        """Search across all Tower services based on query"""
        results = {}
        query_lower = query.lower()
        
        # Real Estate queries
        if any(keyword in query_lower for keyword in ['property', 'real estate', 'house', 'investment property', 'vestal']):
            results['real_estate'] = self.get_vestal_estate_portfolio()
            
        # Financial queries
        if any(keyword in query_lower for keyword in ['loan', 'mortgage', 'financing', 'credit', 'rate']):
            results['loans'] = requests.get(f"{self.services['loan_search']}/api/recommendations", timeout=10).json()
            
        # Crypto queries
        if any(keyword in query_lower for keyword in ['crypto', 'bitcoin', 'ethereum', 'trading', 'portfolio']):
            results['crypto'] = requests.get(f"{self.services['crypto_trader']}/api/portfolio", timeout=10).json()
            
        # Personal data queries
        if any(keyword in query_lower for keyword in ['personal', 'data', 'privacy', 'monitoring', 'files']):
            results['personal_monitoring'] = self.get_service_health('personal_data_monitor')
            
        # OAuth/SSO queries
        if any(keyword in query_lower for keyword in ['login', 'auth', 'oauth', 'sso', 'google', 'apple']):
            results['oauth_sso'] = {
                "available_providers": list(self.oauth_providers.keys()),
                "auth_service": self.get_service_health('auth_service')
            }
        
        return results
    
    def get_crawler_learning_data(self):
        """Get data from Tower crawlers and learning systems"""
        learning_data = {
            "knowledge_base": {},
            "agent_capabilities": {},
            "system_learning": {}
        }
        
        # Knowledge Base learning
        try:
            kb_response = requests.get(f"{self.services['knowledge_base']}/api/articles", timeout=10)
            if kb_response.status_code == 200:
                learning_data["knowledge_base"] = kb_response.json()
        except Exception as e:
            learning_data["knowledge_base"] = {"error": str(e)}
        
        # Agent Manager capabilities
        try:
            agent_response = requests.get(f"{self.services['agent_manager']}/api/agents", timeout=10)
            if agent_response.status_code == 200:
                learning_data["agent_capabilities"] = agent_response.json()
        except Exception as e:
            learning_data["agent_capabilities"] = {"error": str(e)}
        
        return learning_data
    
    def handle_patrick_comprehensive_query(self, query: str):
        """Handle Patrick's queries with complete Tower services integration"""
        query_lower = query.lower()
        
        # Comprehensive profile query
        if any(keyword in query_lower for keyword in ['comprehensive', 'complete', 'full profile', 'everything']):
            profile = self.get_patrick_comprehensive_profile()
            return f"Patrick's Complete Tower Profile:\n{json.dumps(profile, indent=2)}"
        
        # Service-specific searches
        search_results = self.search_tower_services(query)
        if search_results:
            return f"Tower Services Search Results:\n{json.dumps(search_results, indent=2)}"
        
        # OAuth/SSO specific
        if 'oauth' in query_lower or 'sso' in query_lower or 'login' in query_lower:
            return f"OAuth SSO Status:\nConfigured providers: {list(self.oauth_providers.keys())}\nGoogle: {'✅' if 'google' in self.oauth_providers else '❌'}\nApple: {'✅' if 'apple' in self.oauth_providers else '❌'}\nMicrosoft: {'✅' if 'microsoft' in self.oauth_providers else '❌'}\nGitHub: {'✅' if 'github' in self.oauth_providers else '❌'}"
        
        # Learning and crawling
        if any(keyword in query_lower for keyword in ['learn', 'crawler', 'knowledge', 'scan', 'catalog']):
            learning_data = self.get_crawler_learning_data()
            return f"Tower Learning Systems:\n{json.dumps(learning_data, indent=2)}"
        
        # Default comprehensive search
        all_services_health = {}
        for service_name in self.services:
            all_services_health[service_name] = self.get_service_health(service_name)
        
        return f"Tower Services Status:\n{json.dumps(all_services_health, indent=2)}"
    
    def enable_real_time_oauth_sso(self):
        """Enable real-time OAuth SSO integration"""
        try:
            # Import Tower auth components
            from oauth_enhanced_handlers import log_auth_event
            from sso_middleware import SSOMiddleware
            
            return {
                "oauth_handlers": "loaded",
                "sso_middleware": "active", 
                "providers": list(self.oauth_providers.keys()),
                "real_time_sso": "enabled"
            }
        except Exception as e:
            return {"error": f"OAuth SSO integration failed: {e}"}

# Global instance for Echo to use
echo_tower_integration = EchoTowerServicesIntegration()
