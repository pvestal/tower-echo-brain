#!/usr/bin/env python3
"""
AI Assist extension to handle integration queries
Add this to the chat endpoint in echo_working.py
"""

import asyncio
from src.integrations.manager import IntegrationManager

# Initialize the integration manager globally
integration_manager = IntegrationManager()

async def handle_integration_query(message: str, user_id: str = 'patrick'):
    """Check if message is an integration query and handle it"""

    msg_lower = message.lower()

    # Check for volleyball/calendar queries
    if any(word in msg_lower for word in ['volleyball', 'calendar', 'lucia', 'practice', 'game', 'schedule']):
        result = await integration_manager.check_partner_calendar()

        if result.get('volleyball_events'):
            events = result['volleyball_events']
            response = f"🏐 I found {len(events)} volleyball events for Lucia:\n\n"

            for event in events[:5]:  # Show first 5
                response += f"• **{event['summary']}**\n"
                response += f"  📅 {event['start']}\n"
                response += f"  📍 {event.get('location', 'No location specified')}\n\n"

            if len(events) > 5:
                response += f"...and {len(events) - 5} more events"

            return response
        else:
            return "No volleyball events found in the upcoming week."

    # Check for financial queries
    elif any(word in msg_lower for word in ['bank', 'balance', 'money', 'financial', 'spending']):
        result = await integration_manager.get_financial_overview()

        if result.get('status') == 'needs_linking':
            return "💳 You haven't linked any bank accounts yet. Would you like me to help you set that up?"
        elif result.get('error'):
            return f"I couldn't access financial data: {result['error']}"
        else:
            return "Financial integration is configured but no accounts are linked."

    # Check for integration status
    elif 'integration' in msg_lower or 'status' in msg_lower:
        status = await integration_manager.get_integration_status()

        response = "🔌 **Integration Status:**\n\n"
        for service, info in status['integrations'].items():
            emoji = "✅" if info['status'] == 'connected' else "⚠️" if info['status'] == 'configured' else "❌"
            response += f"{emoji} **{service}**: {info['status']}\n"
            if 'note' in info:
                response += f"   _{info['note']}_\n"

        return response

    return None  # Not an integration query

# Test function
async def test_integration():
    queries = [
        "What's Lucia's volleyball schedule?",
        "Do we have volleyball practice this week?",
        "Check my bank balance",
        "What's my integration status?"
    ]

    for query in queries:
        print(f"\n❓ Query: {query}")
        response = await handle_integration_query(query)
        if response:
            print(response)
        else:
            print("(Not an integration query)")

if __name__ == "__main__":
    asyncio.run(test_integration())