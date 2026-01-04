#!/usr/bin/env python3
"""
Echo Financial Context Provider
Provides financial intelligence to Echo Brain responses
"""

import psycopg2
from datetime import datetime, timedelta
from typing import Dict, Optional

class EchoFinancialContext:
    def __init__(self):
        self.db_params = {
            'host': 'localhost',
            'port': 5432,
            'database': 'echo_brain',
            'user': 'patrick'
        }
    
    def get_connection(self):
        return psycopg2.connect(**self.db_params)
    
    def get_spending_summary(self, days=30) -> Dict:
        """Get spending summary for last N days"""
        conn = self.get_connection()
        cur = conn.cursor()
        
        cur.execute('''
            SELECT 
                category,
                COUNT(*) as transaction_count,
                SUM(amount)::numeric(10,2) as total_spent,
                AVG(amount)::numeric(10,2) as avg_transaction
            FROM financial_patterns
            WHERE transaction_date >= CURRENT_DATE - INTERVAL '%s days'
            GROUP BY category
            ORDER BY total_spent DESC
        ''' % days)
        
        categories = []
        total = 0
        for cat, count, spent, avg in cur.fetchall():
            categories.append({
                'category': cat,
                'count': count,
                'total': float(spent),
                'avg': float(avg)
            })
            total += float(spent)
        
        cur.close()
        conn.close()
        
        return {
            'period_days': days,
            'total_spending': total,
            'categories': categories
        }
    
    def get_recurring_bills(self) -> list:
        """Get list of recurring bills"""
        conn = self.get_connection()
        cur = conn.cursor()
        
        cur.execute('''
            SELECT merchant, category, AVG(amount)::numeric(10,2) as avg_amount, COUNT(*) as frequency
            FROM financial_patterns
            WHERE pattern_type = 'recurring'
            GROUP BY merchant, category
            ORDER BY avg_amount DESC
        ''')
        
        bills = []
        for merchant, category, avg_amount, freq in cur.fetchall():
            bills.append({
                'merchant': merchant,
                'category': category,
                'amount': float(avg_amount),
                'frequency': freq
            })
        
        cur.close()
        conn.close()
        
        return bills
    
    def get_category_spending(self, category: str, days=30) -> Dict:
        """Get spending for specific category"""
        conn = self.get_connection()
        cur = conn.cursor()
        
        cur.execute('''
            SELECT 
                merchant,
                COUNT(*) as visits,
                SUM(amount)::numeric(10,2) as total,
                AVG(amount)::numeric(10,2) as avg
            FROM financial_patterns
            WHERE category = %s
            AND transaction_date >= CURRENT_DATE - INTERVAL '%s days'
            GROUP BY merchant
            ORDER BY total DESC
        ''' % ('%s', days), (category,))
        
        merchants = []
        total = 0
        for merchant, visits, spent, avg in cur.fetchall():
            merchants.append({
                'merchant': merchant,
                'visits': visits,
                'total': float(spent),
                'avg': float(avg)
            })
            total += float(spent)
        
        cur.close()
        conn.close()
        
        return {
            'category': category,
            'total_spent': total,
            'merchants': merchants
        }
    
    def get_financial_preferences(self) -> Dict:
        """Get learned spending preferences"""
        conn = self.get_connection()
        cur = conn.cursor()
        
        cur.execute('''
            SELECT preference_key, preference_value, confidence, evidence
            FROM learned_preferences
            WHERE category = 'spending'
            ORDER BY confidence DESC
        ''')
        
        preferences = []
        for key, value, confidence, evidence in cur.fetchall():
            preferences.append({
                'preference': key,
                'value': value,
                'confidence': float(confidence)
            })
        
        cur.close()
        conn.close()
        
        return {'preferences': preferences}
    
    def build_echo_context(self) -> str:
        """Build financial context string for Echo prompt"""
        summary = self.get_spending_summary(30)
        bills = self.get_recurring_bills()
        
        context = f"\nFINANCIAL CONTEXT (Last 30 days):\n"
        context += f"Total spending: ${summary['total_spending']:.2f}\n"
        
        if summary['categories']:
            context += "\nTop spending categories:\n"
            for cat in summary['categories'][:5]:
                context += f"  - {cat['category']}: ${cat['total']:.2f} ({cat['count']} transactions)\n"
        
        if bills:
            context += f"\nRecurring bills ({len(bills)} total):\n"
            for bill in bills[:5]:
                context += f"  - {bill['merchant']}: ${bill['amount']:.2f} ({bill['category']})\n"
        
        return context

if __name__ == '__main__':
    context = EchoFinancialContext()
    print(context.build_echo_context())
