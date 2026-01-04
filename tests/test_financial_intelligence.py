#!/usr/bin/env python3
import sys
sys.path.insert(0, '/opt/tower-echo-brain/src/integrations')

from src.integrations.echo_financial_context import EchoFinancialContext

# Simulate Echo Brain queries with financial context
financial_context = EchoFinancialContext()

print('=' * 70)
print('ECHO BRAIN - FINANCIAL INTELLIGENCE DEMO')
print('=' * 70)
print()

# Test Query 1: Spending summary
print('Q: "How much did I spend last month?"')
print()
summary = financial_context.get_spending_summary(30)
print(f'A: You spent ${summary["total_spending"]:.2f} in the last 30 days.')
print(f'   Your top category was {summary["categories"][0]["category"]} ')
print(f'   at ${summary["categories"][0]["total"]:.2f}')
print()

# Test Query 2: Category specific
print('-' * 70)
print('Q: "What did I spend on groceries?"')
print()
groceries = financial_context.get_category_spending('Groceries', 30)
print(f'A: You spent ${groceries["total_spent"]:.2f} on groceries in the last 30 days.')
print(f'   You visited:')
for m in groceries['merchants']:
    print(f'   - {m["merchant"]}: {m["visits"]} times, ${m["total"]:.2f} total')
print()

# Test Query 3: Recurring bills
print('-' * 70)
print('Q: "What are my recurring bills?"')
print()
bills = financial_context.get_recurring_bills()
print(f'A: You have {len(bills)} recurring bills:')
total_bills = sum(b['amount'] for b in bills)
for b in bills[:5]:
    print(f'   - {b["merchant"]}: ${b["amount"]:.2f}/month ({b["category"]})')
print(f'   Total recurring: ${total_bills:.2f}/month')
print()

# Test Query 4: Full context
print('-' * 70)
print('Echo\'s Internal Financial Context:')
print(financial_context.build_echo_context())

print('=' * 70)
print('✅ Financial Intelligence Integration Complete!')
print()
print('Next: Integrate with actual Echo Brain LLM prompts')
print('=' * 70)
