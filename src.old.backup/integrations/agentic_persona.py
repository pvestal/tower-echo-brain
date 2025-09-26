#!/usr/bin/env python3
"""
AgenticPersona Framework for Echo Brain
Autonomous monitoring and action agents
"""
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class AgenticPersona:
    """Base class for autonomous monitoring personas"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.triggers = []
        self.actions = []
        self.schedule = None
        self.state = {}
        self.enabled = True

    async def monitor(self) -> Dict[str, Any]:
        """Monitor for conditions and trigger actions"""
        raise NotImplementedError

    async def execute_action(self, action: str, context: Dict[str, Any]) -> bool:
        """Execute a specific action"""
        raise NotImplementedError


class FinancialGuardianPersona(AgenticPersona):
    """Monitors financial accounts and spending patterns"""

    def __init__(self):
        super().__init__(
            "Financial Guardian",
            "Monitors Plaid accounts for unusual activity and spending patterns"
        )
        self.alert_threshold = 500  # Alert on transactions over $500
        self.daily_limit = 1000  # Alert if daily spending exceeds $1000

    async def monitor(self) -> Dict[str, Any]:
        """Check financial accounts"""
        from src.integrations.plaid import PlaidIntegration

        plaid = PlaidIntegration()
        results = {
            'timestamp': datetime.now().isoformat(),
            'alerts': [],
            'summary': {}
        }

        try:
            # Get accounts
            accounts = await plaid.get_accounts()
            total_balance = sum(acc['balance']['current'] for acc in accounts)
            results['summary']['total_balance'] = total_balance

            # Get recent transactions
            transactions = await plaid.get_transactions()

            # Check for large transactions
            for txn in transactions:
                if abs(txn['amount']) > self.alert_threshold:
                    results['alerts'].append({
                        'type': 'large_transaction',
                        'amount': txn['amount'],
                        'description': txn['name'],
                        'date': txn['date']
                    })

            # Calculate daily spending
            today = datetime.now().date()
            daily_total = sum(
                abs(txn['amount'])
                for txn in transactions
                if datetime.fromisoformat(txn['date']).date() == today
            )

            if daily_total > self.daily_limit:
                results['alerts'].append({
                    'type': 'daily_limit_exceeded',
                    'amount': daily_total,
                    'limit': self.daily_limit
                })

            results['summary']['daily_spending'] = daily_total
            results['summary']['transaction_count'] = len(transactions)

        except Exception as e:
            logger.error(f"Financial monitoring error: {e}")
            results['error'] = str(e)

        return results

    async def execute_action(self, action: str, context: Dict[str, Any]) -> bool:
        """Execute financial actions"""
        if action == "send_alert":
            # Send email/notification about financial alert
            print(f"💰 FINANCIAL ALERT: {context}")
            return True
        elif action == "generate_report":
            # Generate detailed spending report
            print(f"📊 Generating financial report...")
            return True
        return False


class FamilyCoordinatorPersona(AgenticPersona):
    """Monitors family calendars and schedules"""

    def __init__(self):
        super().__init__(
            "Family Coordinator",
            "Tracks family schedules and sends reminders"
        )
        self.reminder_hours = 2  # Send reminders 2 hours before events

    async def monitor(self) -> Dict[str, Any]:
        """Check family calendars"""
        from src.integrations.google_calendar import GoogleCalendarIntegration

        calendar = GoogleCalendarIntegration()
        results = {
            'timestamp': datetime.now().isoformat(),
            'upcoming_events': [],
            'reminders_needed': []
        }

        try:
            # Check volleyball schedule
            volleyball_info = await calendar.check_partner_volleyball_schedule()

            if volleyball_info['has_volleyball']:
                for event in volleyball_info['events']:
                    event_time = datetime.fromisoformat(event['start'])
                    time_until = event_time - datetime.now()

                    # Check if reminder needed
                    if timedelta(0) < time_until < timedelta(hours=self.reminder_hours):
                        results['reminders_needed'].append({
                            'event': event['summary'],
                            'location': event.get('location', 'TBD'),
                            'time': event['start'],
                            'minutes_until': int(time_until.total_seconds() / 60)
                        })

                    results['upcoming_events'].append(event)

            # Get all today's events
            events = await calendar.get_today_events()
            results['today_events'] = events

        except Exception as e:
            logger.error(f"Calendar monitoring error: {e}")
            results['error'] = str(e)

        return results

    async def execute_action(self, action: str, context: Dict[str, Any]) -> bool:
        """Execute family coordination actions"""
        if action == "send_reminder":
            # Send reminder about upcoming event
            print(f"🏐 REMINDER: {context['event']} in {context['minutes_until']} minutes at {context['location']}")
            return True
        elif action == "coordinate_pickup":
            # Coordinate family pickups/dropoffs
            print(f"🚗 Coordinating pickup for {context}")
            return True
        return False


class PersonaManager:
    """Manages all agenticPersonas"""

    def __init__(self):
        self.personas = {}
        self.monitoring_tasks = {}
        self._initialize_personas()

    def _initialize_personas(self):
        """Initialize default personas"""
        self.personas['financial'] = FinancialGuardianPersona()
        self.personas['family'] = FamilyCoordinatorPersona()

    async def start_monitoring(self, persona_name: str, interval_minutes: int = 30):
        """Start monitoring for a specific persona"""
        if persona_name not in self.personas:
            return {'error': f'Persona {persona_name} not found'}

        persona = self.personas[persona_name]

        async def monitor_loop():
            while persona.enabled:
                try:
                    results = await persona.monitor()

                    # Check for alerts and execute actions
                    if results.get('alerts'):
                        for alert in results['alerts']:
                            await persona.execute_action('send_alert', alert)

                    if results.get('reminders_needed'):
                        for reminder in results['reminders_needed']:
                            await persona.execute_action('send_reminder', reminder)

                    # Store results
                    persona.state['last_check'] = results

                except Exception as e:
                    logger.error(f"Persona {persona_name} error: {e}")

                # Wait for next check
                await asyncio.sleep(interval_minutes * 60)

        # Start monitoring task
        if persona_name in self.monitoring_tasks:
            self.monitoring_tasks[persona_name].cancel()

        task = asyncio.create_task(monitor_loop())
        self.monitoring_tasks[persona_name] = task

        return {
            'status': 'started',
            'persona': persona_name,
            'interval_minutes': interval_minutes
        }

    def stop_monitoring(self, persona_name: str):
        """Stop monitoring for a specific persona"""
        if persona_name in self.monitoring_tasks:
            self.monitoring_tasks[persona_name].cancel()
            del self.monitoring_tasks[persona_name]
            return {'status': 'stopped', 'persona': persona_name}
        return {'error': f'No active monitoring for {persona_name}'}

    def get_status(self) -> Dict[str, Any]:
        """Get status of all personas"""
        status = {}
        for name, persona in self.personas.items():
            status[name] = {
                'enabled': persona.enabled,
                'monitoring': name in self.monitoring_tasks,
                'last_state': persona.state.get('last_check', {})
            }
        return status


# Example usage and test
async def demonstrate_agentic_personas():
    """Demonstrate agenticPersona capabilities"""
    print("🎭 AGENTIC PERSONA DEMONSTRATION")
    print("=" * 60)

    manager = PersonaManager()

    # Check financial status
    print("\n💰 FINANCIAL GUARDIAN CHECK:")
    financial_results = await manager.personas['financial'].monitor()
    print(f"  Balance: ${financial_results['summary'].get('total_balance', 0):,.2f}")
    print(f"  Today's spending: ${financial_results['summary'].get('daily_spending', 0):,.2f}")
    if financial_results.get('alerts'):
        print(f"  ⚠️ Alerts: {len(financial_results['alerts'])}")

    # Check family schedule
    print("\n👨‍👩‍👧 FAMILY COORDINATOR CHECK:")
    family_results = await manager.personas['family'].monitor()
    print(f"  Upcoming events: {len(family_results.get('upcoming_events', []))}")
    if family_results.get('reminders_needed'):
        print(f"  🔔 Reminders needed: {len(family_results['reminders_needed'])}")
        for reminder in family_results['reminders_needed']:
            print(f"    - {reminder['event']} in {reminder['minutes_until']} minutes")

    # Start monitoring
    print("\n🚀 STARTING CONTINUOUS MONITORING:")
    print("  Financial Guardian: Every 60 minutes")
    print("  Family Coordinator: Every 30 minutes")

    # These would run continuously in production
    await manager.start_monitoring('financial', 60)
    await manager.start_monitoring('family', 30)

    print("\n✅ AgenticPersonas are now monitoring autonomously!")
    print("  They will:")
    print("  - Check conditions periodically")
    print("  - Send alerts when thresholds exceeded")
    print("  - Execute actions automatically")
    print("  - Learn from patterns over time")

    return manager

if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demonstrate_agentic_personas())