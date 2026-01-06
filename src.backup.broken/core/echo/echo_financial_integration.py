#!/usr/bin/env python3
"""
Echo Financial Services Integration
"""
import requests
import json
import logging
import sys
import os

# Add path to access Tower modules
sys.path.append('/opt/tower-personal-data-monitor')

logger = logging.getLogger(__name__)

class EchoFinancialIntegration:
    def __init__(self):
        self.services = {
            'loan_search': 'http://127.0.0.1:8302',
            'crypto_trader': 'http://127.0.0.1:8303', 
            'personal_data_monitor': 'http://127.0.0.1:8311'
        }
        
    def get_patrick_financial_profile(self):
        """Get Patrick's comprehensive financial profile"""
        try:
            # Import Plaid integration
            from plaid_financial_integration import PlaidFinancialIntegration
            plaid = PlaidFinancialIntegration()
            
            # Get comprehensive financial analysis
            financial_profile = plaid.comprehensive_financial_analysis(user_id=1)
            
            return {
                "status": "success",
                "financial_profile": financial_profile,
                "real_data": financial_profile.get('real_data', False),
                "plaid_connected": bool(plaid.plaid_credentials),
                "accounts": financial_profile.get('accounts', {}),
                "credit_profile": financial_profile.get('credit_profile', {})
            }
            
        except Exception as e:
            logger.error(f"Financial profile error: {e}")
            return {
                "status": "error", 
                "error": str(e),
                "real_data": False
            }
    
    def get_real_time_crypto_data(self):
        """Get real-time cryptocurrency data"""
        try:
            response = requests.get(f"{self.services['crypto_trader']}/api/portfolio", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Crypto service returned {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Crypto service unavailable: {e}"}
    
    def get_loan_recommendations(self):
        """Get loan recommendations for Patrick"""
        try:
            response = requests.get(f"{self.services['loan_search']}/api/recommendations", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Loan service returned {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Loan service unavailable: {e}"}
    
    def search_financial_opportunities(self, query: str):
        """Search for financial opportunities based on query"""
        results = {}
        
        if any(keyword in query.lower() for keyword in ['crypto', 'bitcoin', 'ethereum', 'trading']):
            results['crypto'] = self.get_real_time_crypto_data()
            
        if any(keyword in query.lower() for keyword in ['loan', 'mortgage', 'financing', 'credit']):
            results['loans'] = self.get_loan_recommendations()
            
        if any(keyword in query.lower() for keyword in ['balance', 'account', 'bank', 'financial']):
            results['financial_profile'] = self.get_patrick_financial_profile()
            
        return results
    
    def handle_patrick_financial_query(self, query: str):
        """Handle Patrick's financial queries with real data"""
        query_lower = query.lower()
        
        # Specific financial queries
        if 'balance' in query_lower or 'account' in query_lower:
            profile = self.get_patrick_financial_profile()
            if profile['status'] == 'success':
                accounts = profile['financial_profile'].get('accounts', {})
                return f"Financial Profile: {json.dumps(accounts, indent=2)}"
            else:
                return f"Unable to access financial data: {profile.get('error', 'Unknown error')}"
        
        elif 'crypto' in query_lower or 'bitcoin' in query_lower:
            crypto_data = self.get_real_time_crypto_data()
            return f"Cryptocurrency Portfolio: {json.dumps(crypto_data, indent=2)}"
            
        elif 'loan' in query_lower or 'mortgage' in query_lower:
            loan_data = self.get_loan_recommendations()
            return f"Loan Recommendations: {json.dumps(loan_data, indent=2)}"
            
        else:
            # General financial search
            results = self.search_financial_opportunities(query)
            return f"Financial Search Results: {json.dumps(results, indent=2)}"

# Global instance for Echo to use
echo_financial = EchoFinancialIntegration()
