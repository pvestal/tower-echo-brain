#!/usr/bin/env python3
"""
Financial Intelligence Integration
Learns spending patterns from Plaid data and feeds to Echo Brain
"""

import psycopg2
import requests
from datetime import datetime, timedelta
from typing import Dict, List
from collections import defaultdict
import statistics
import json

class FinancialLearner:
    def __init__(self):
        self.db_params = {
            'host': 'localhost',
            'port': 5432,
            'database': 'echo_brain',
            'user': 'patrick'
        }
        self.plaid_api_url = 'http://localhost:8089/api/plaid'
    
    def get_db_connection(self):
        return psycopg2.connect(**self.db_params)
    
    def fetch_plaid_transactions(self, days_back=90):
        """Fetch transactions from Plaid API"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days_back)
            
            response = requests.get(
                f'{self.plaid_api_url}/transactions',
                params={
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('transactions', [])
            else:
                print(f'Plaid API error: {response.status_code}')
                return []
        except Exception as e:
            print(f'Error fetching Plaid transactions: {e}')
            return []
    
    def store_transaction(self, conn, tx):
        """Store transaction in financial_patterns table"""
        try:
            cur = conn.cursor()
            
            category = tx.get('category')
            if isinstance(category, list) and category:
                category = category[0]
            elif not category:
                category = 'Uncategorized'
            
            cur.execute('''
                INSERT INTO financial_patterns 
                (transaction_id, transaction_date, category, amount, merchant, account_id, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (transaction_id) DO UPDATE
                SET metadata = EXCLUDED.metadata
                RETURNING id
            ''', (
                tx.get('transaction_id'),
                tx.get('date'),
                category,
                abs(float(tx.get('amount', 0))),
                tx.get('merchant_name') or tx.get('name'),
                tx.get('account_id'),
                json.dumps({
                    'name': tx.get('name'),
                    'pending': tx.get('pending', False),
                    'payment_channel': tx.get('payment_channel')
                })
            ))
            
            conn.commit()
            cur.close()
            return True
        except Exception as e:
            print(f'Error storing transaction: {e}')
            conn.rollback()
            return False
    
    def detect_recurring_patterns(self, conn):
        """Detect recurring transactions"""
        cur = conn.cursor()
        
        cur.execute('''
            SELECT merchant, category, array_agg(amount) as amounts
            FROM financial_patterns
            WHERE merchant IS NOT NULL
            GROUP BY merchant, category
            HAVING COUNT(*) >= 3
        ''')
        
        for merchant, category, amounts in cur.fetchall():
            if amounts and len(amounts) >= 3:
                avg_amount = statistics.mean(amounts)
                stddev = statistics.stdev(amounts) if len(amounts) > 1 else 0
                
                if avg_amount > 0 and stddev / avg_amount < 0.1:
                    cur.execute('''
                        UPDATE financial_patterns
                        SET pattern_type = 'recurring'
                        WHERE merchant = %s AND category = %s
                    ''', (merchant, category))
        
        conn.commit()
        cur.close()
    
    def learn_spending_preferences(self, conn):
        """Extract spending preferences from transaction patterns"""
        cur = conn.cursor()
        
        cur.execute('''
            SELECT category, merchant, COUNT(*) as frequency, AVG(amount) as avg_spend
            FROM financial_patterns
            WHERE merchant IS NOT NULL
            GROUP BY category, merchant
            HAVING COUNT(*) >= 5
            ORDER BY frequency DESC
            LIMIT 20
        ''')
        
        for category, merchant, frequency, avg_spend in cur.fetchall():
            confidence = min(0.95, frequency / 50.0)
            
            cur.execute('''
                INSERT INTO learned_preferences
                (category, preference_key, preference_value, confidence, source, evidence)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (category, preference_key) DO UPDATE
                SET preference_value = EXCLUDED.preference_value,
                    confidence = EXCLUDED.confidence,
                    evidence = EXCLUDED.evidence,
                    updated_at = CURRENT_TIMESTAMP
            ''', (
                'spending',
                f'frequent_merchant_{category}',
                merchant,
                confidence,
                'financial',
                json.dumps({
                    'frequency': frequency,
                    'avg_spend': float(avg_spend),
                    'category': category
                })
            ))
        
        conn.commit()
        cur.close()
    
    def get_financial_context(self, conn):
        """Get financial context for Echo responses"""
        cur = conn.cursor()
        
        cur.execute('''
            SELECT 
                category,
                COUNT(*) as transaction_count,
                SUM(amount) as total_spent,
                AVG(amount) as avg_transaction
            FROM financial_patterns
            WHERE transaction_date >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY category
            ORDER BY total_spent DESC
        ''')
        
        categories = []
        total_spending = 0
        for category, count, total, avg in cur.fetchall():
            categories.append({
                'category': category,
                'count': count,
                'total': float(total),
                'avg': float(avg)
            })
            total_spending += float(total)
        
        cur.execute('''
            SELECT merchant, category, AVG(amount) as amount
            FROM financial_patterns
            WHERE pattern_type = 'recurring'
            GROUP BY merchant, category
        ''')
        
        recurring = [
            {'merchant': m, 'category': c, 'amount': float(a)}
            for m, c, a in cur.fetchall()
        ]
        
        cur.close()
        
        return {
            'last_30_days_total': total_spending,
            'categories': categories,
            'recurring_bills': recurring,
            'updated_at': datetime.now().isoformat()
        }
    
    def run_ingestion(self):
        """Main ingestion pipeline"""
        print('Starting financial data ingestion...')
        
        transactions = self.fetch_plaid_transactions(days_back=90)
        print(f'Fetched {len(transactions)} transactions from Plaid')
        
        if not transactions:
            print('No transactions to process')
            return None
        
        conn = self.get_db_connection()
        stored = 0
        
        for tx in transactions:
            if self.store_transaction(conn, tx):
                stored += 1
        
        print(f'Stored {stored} transactions')
        
        print('Detecting recurring patterns...')
        self.detect_recurring_patterns(conn)
        
        print('Learning spending preferences...')
        self.learn_spending_preferences(conn)
        
        context = self.get_financial_context(conn)
        print(f'Financial context: Spent in last 30 days across {len(context["categories"])} categories')
        
        conn.close()
        print('Financial ingestion complete!')
        
        return context

if __name__ == '__main__':
    learner = FinancialLearner()
    learner.run_ingestion()
