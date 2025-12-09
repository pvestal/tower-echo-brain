#!/usr/bin/env python3
"""
Veteran Guardian Support System for Echo Brain
Specialized module for veteran mental health and addiction counseling support
via Telegram integration
"""

import asyncio
import json
import logging
import hashlib
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor
import aiohttp
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk assessment levels for veteran support"""
    CRITICAL = "critical"  # Immediate danger, suicide risk
    HIGH = "high"          # Active crisis, severe symptoms
    MODERATE = "moderate"  # Struggling but stable
    LOW = "low"           # Maintenance phase
    CHECK_IN = "check_in" # Routine support

class VeteranGuardianSystem:
    """
    Specialized support system for veterans with:
    - Addiction counseling expertise
    - Military cultural competence
    - PTSD treatment experience
    - Crisis intervention protocols
    """

    def __init__(self, db_config: Dict, telegram_config: Dict):
        self.db_config = db_config
        self.telegram_config = telegram_config
        self.bot_token = telegram_config.get('bot_token')
        self.support_channel_id = telegram_config.get('support_channel_id')

        # Initialize database
        self.init_database()

        # Crisis keywords and patterns
        self.crisis_patterns = {
            'suicide': r'\b(suicide|suicidal|kill myself|end it all|not worth living|better off dead|going to end it|gun|end my life|want to die|nobody will miss|take my life)\b',
            'self_harm': r'\b(cut myself|hurt myself|self harm|self-harm|cutting)\b',
            'substance': r'\b(relapse|relapsed|using again|drunk|high|overdose|OD|can\'t stop drinking|drinking for.*days|took some pills)\b',
            'violence': r'\b(hurt someone|kill someone|lose control|rage|seeing red)\b',
            'panic': r'\b(panic attack|can\'t breathe|dying|heart attack|losing mind)\b',
            'flashback': r'\b(flashback|back there|in country|firefight|IED|ambush|back in.*[a-z]+ujah|hear.*mortar|squad.*gone)\b'
        }

        # Therapeutic response templates
        self.response_templates = {
            'crisis': {
                'immediate': "I hear you, and I'm here with you right now. Your life matters, and this pain you're feeling is temporary, even though it doesn't feel that way.",
                'safety_check': "First, I need to check - are you safe right now? Are you somewhere secure?",
                'resources': "If you're in immediate danger, please call 988 (Veteran Crisis Line) or text HOME to 741741. I'll stay here with you.",
                'grounding': "Let's try something together - can you tell me 5 things you can see right now? This can help ground us in the present moment."
            },
            'addiction': {
                'relapse_prevention': "Relapse is part of recovery, not failure. You've made it this far, and that strength is still in you.",
                'coping': "What were your go-to coping strategies that helped before? Sometimes we need to remember our tools.",
                'support': "You don't have to face this alone. Is there someone from your support network you can reach out to today?",
                'progress': "Recovery isn't linear. Every day you choose to fight is a victory, even the hard days."
            },
            'ptsd': {
                'validation': "What you're experiencing is real, and it's a normal response to abnormal situations. Your brain is trying to protect you.",
                'grounding': "You're safe now. You're here in {current_date}, not back there. Can you feel your feet on the ground?",
                'breathing': "Let's try tactical breathing together - in for 4, hold for 4, out for 4, hold for 4. This is the same technique used in the field.",
                'normalize': "Many of us struggle with this. It doesn't make you weak - it makes you human."
            },
            'military': {
                'understanding': "I get it. The transition from military to civilian life is one of the hardest missions we face.",
                'identity': "You're still the warrior you always were. Now the battle is different, but your strength remains.",
                'purpose': "Finding a new mission after service is tough. What gave you purpose in uniform can guide you now.",
                'brotherhood': "The bonds we formed don't disappear. Your brothers and sisters are still out here."
            }
        }

        # Positive coping strategies database
        self.coping_strategies = {
            'immediate': [
                "Box breathing (4-4-4-4 count)",
                "5-4-3-2-1 grounding (5 see, 4 touch, 3 hear, 2 smell, 1 taste)",
                "Ice cube in hand (sensory grounding)",
                "Call battle buddy",
                "Step outside for fresh air"
            ],
            'short_term': [
                "Physical exercise (even 10 pushups help)",
                "Write in journal",
                "Listen to calming music",
                "Take a shower",
                "Reach out to support network"
            ],
            'long_term': [
                "Regular therapy sessions",
                "Medication compliance",
                "Exercise routine",
                "Sleep hygiene",
                "Meaningful activities/volunteering"
            ]
        }

    def init_database(self):
        """Initialize database tables for veteran support tracking"""
        # Handle password authentication issue
        import os
        db_config = self.db_config.copy()
        # Use the same password format as Echo Brain
        db_config['password'] = os.getenv('DB_PASSWORD', db_config.get('password', ''))
        if not db_config['password']:
            db_config['password'] = 'patrick123'

        # Try connection without password first (trust auth)
        try:
            db_config_no_pwd = db_config.copy()
            del db_config_no_pwd['password']
            conn = psycopg2.connect(**db_config_no_pwd)
        except:
            # Fallback to password auth
            conn = psycopg2.connect(**db_config)
        cur = conn.cursor()

        # Create veteran support tables
        cur.execute("""
            CREATE TABLE IF NOT EXISTS veteran_support_conversations (
                conversation_id SERIAL PRIMARY KEY,
                telegram_user_id BIGINT NOT NULL,
                telegram_username VARCHAR(255),
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_message_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                risk_level VARCHAR(20),
                status VARCHAR(20) DEFAULT 'active',
                total_messages INTEGER DEFAULT 0,
                crisis_interventions INTEGER DEFAULT 0,
                notes TEXT
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS veteran_support_messages (
                message_id SERIAL PRIMARY KEY,
                conversation_id INTEGER REFERENCES veteran_support_conversations(conversation_id),
                telegram_message_id BIGINT,
                sender VARCHAR(10), -- 'user' or 'bot'
                message_text TEXT NOT NULL,
                risk_assessment VARCHAR(20),
                intervention_type VARCHAR(50),
                response_time_ms INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS veteran_crisis_events (
                event_id SERIAL PRIMARY KEY,
                conversation_id INTEGER REFERENCES veteran_support_conversations(conversation_id),
                crisis_type VARCHAR(50) NOT NULL,
                severity VARCHAR(20) NOT NULL,
                intervention_provided TEXT,
                outcome VARCHAR(100),
                follow_up_needed BOOLEAN DEFAULT true,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS veteran_support_resources (
                resource_id SERIAL PRIMARY KEY,
                resource_type VARCHAR(50) NOT NULL,
                resource_name VARCHAR(255) NOT NULL,
                contact_info TEXT,
                description TEXT,
                url VARCHAR(500),
                priority INTEGER DEFAULT 0,
                active BOOLEAN DEFAULT true
            )
        """)

        # Insert critical resources if not exists
        cur.execute("""
            INSERT INTO veteran_support_resources (resource_type, resource_name, contact_info, description, priority)
            VALUES
                ('crisis', 'Veteran Crisis Line', '988 Press 1', 'Immediate crisis support for veterans', 1),
                ('crisis', 'Crisis Text Line', 'Text HOME to 741741', '24/7 text crisis support', 2),
                ('substance', 'SAMHSA Helpline', '1-800-662-4357', 'Substance abuse and mental health services', 3),
                ('emergency', '911', '911', 'Emergency services', 0)
            ON CONFLICT DO NOTHING
        """)

        conn.commit()
        cur.close()
        conn.close()

        logger.info("Veteran support database initialized")

    async def assess_risk_level(self, message: str) -> Tuple[RiskLevel, List[str]]:
        """
        Assess the risk level of a message and identify crisis indicators
        Returns: (risk_level, list_of_identified_concerns)
        """
        concerns = []
        risk_score = 0

        message_lower = message.lower()

        # Check crisis patterns
        for crisis_type, pattern in self.crisis_patterns.items():
            if re.search(pattern, message_lower, re.IGNORECASE):
                concerns.append(crisis_type)
                if crisis_type in ['suicide', 'violence']:
                    risk_score += 10
                elif crisis_type in ['self_harm', 'overdose']:
                    risk_score += 8
                else:
                    risk_score += 5

        # Check for hopelessness indicators
        hopelessness_words = ['hopeless', 'worthless', 'no point', 'give up', 'can\'t go on', 'hate myself', 'no reason']
        for word in hopelessness_words:
            if word in message_lower:
                risk_score += 3
                if 'hopelessness' not in concerns:
                    concerns.append('hopelessness')

        # Check for isolation indicators
        isolation_words = ['alone', 'nobody cares', 'no one understands', 'isolated', 'nobody understands']
        for word in isolation_words:
            if word in message_lower:
                risk_score += 2
                if 'isolation' not in concerns:
                    concerns.append('isolation')

        # Additional severity keywords that should increase risk
        severity_indicators = ['right now', 'tonight', 'today', 'immediately', 'gun', 'pills', 'rope']
        for indicator in severity_indicators:
            if indicator in message_lower:
                risk_score += 5

        # Determine risk level
        if risk_score >= 10:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 7:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 4:
            risk_level = RiskLevel.MODERATE
        elif risk_score > 0:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.CHECK_IN

        return risk_level, concerns

    async def generate_therapeutic_response(self,
                                           message: str,
                                           risk_level: RiskLevel,
                                           concerns: List[str],
                                           conversation_history: List[Dict]) -> str:
        """
        Generate an appropriate therapeutic response based on:
        - Current message content
        - Risk assessment
        - Identified concerns
        - Conversation history
        """
        response_parts = []

        # Critical risk requires immediate intervention
        if risk_level == RiskLevel.CRITICAL:
            response_parts.append(self.response_templates['crisis']['immediate'])
            response_parts.append(self.response_templates['crisis']['safety_check'])

            if 'suicide' in concerns:
                response_parts.append("I want you to know that you matter. Your story isn't over.")
                response_parts.append(self.response_templates['crisis']['resources'])

            if 'flashback' in concerns:
                response_parts.append(self.response_templates['ptsd']['grounding'].format(
                    current_date=datetime.now().strftime("%B %d, %Y")
                ))

        # High risk needs strong support
        elif risk_level == RiskLevel.HIGH:
            if 'substance' in concerns:
                response_parts.append(self.response_templates['addiction']['relapse_prevention'])
                response_parts.append(self.response_templates['addiction']['coping'])

            if 'panic' in concerns:
                response_parts.append(self.response_templates['ptsd']['breathing'])
                response_parts.append("You've survived 100% of your worst days. You'll get through this one too.")

            response_parts.append(self.response_templates['crisis']['grounding'])

        # Moderate risk needs validation and coping strategies
        elif risk_level == RiskLevel.MODERATE:
            if 'isolation' in concerns:
                response_parts.append(self.response_templates['military']['brotherhood'])
                response_parts.append(self.response_templates['addiction']['support'])

            if 'hopelessness' in concerns:
                response_parts.append(self.response_templates['military']['understanding'])
                response_parts.append(self.response_templates['military']['purpose'])

            # Offer coping strategies
            response_parts.append("\nHere are some strategies that might help right now:")
            for strategy in self.coping_strategies['immediate'][:3]:
                response_parts.append(f"• {strategy}")

        # Low risk or check-in needs maintenance support
        else:
            # Provide encouragement and check on progress
            response_parts.append("I'm here and listening. How are you managing today?")

            if conversation_history:
                # Reference previous conversations for continuity
                response_parts.append("Last time we talked about your progress. How has that been going?")

            response_parts.append(self.response_templates['addiction']['progress'])

        # Always end with availability
        response_parts.append("\nI'm here for you, warrior. What do you need right now?")

        return "\n\n".join(response_parts)

    async def process_telegram_message(self, update: Dict) -> Dict:
        """
        Process incoming Telegram message and generate appropriate response
        """
        try:
            # Extract message data
            message = update.get('message', {})
            chat_id = message.get('chat', {}).get('id')
            user_id = message.get('from', {}).get('id')
            username = message.get('from', {}).get('username', 'Unknown')
            text = message.get('text', '')
            message_id = message.get('message_id')

            if not text:
                return {'status': 'ignored', 'reason': 'no_text'}

            # Start timing for response metrics
            start_time = datetime.now()

            # Get or create conversation
            conversation = await self.get_or_create_conversation(user_id, username)

            # Get conversation history
            history = await self.get_conversation_history(conversation['conversation_id'])

            # Assess risk level
            risk_level, concerns = await self.assess_risk_level(text)

            # Store user message
            await self.store_message(
                conversation['conversation_id'],
                message_id,
                'user',
                text,
                risk_level.value,
                None
            )

            # Generate therapeutic response
            response_text = await self.generate_therapeutic_response(
                text, risk_level, concerns, history
            )

            # Send response via Telegram
            await self.send_telegram_message(chat_id, response_text, message_id)

            # Calculate response time
            response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            # Store bot response
            await self.store_message(
                conversation['conversation_id'],
                None,  # Bot messages don't have incoming message ID
                'bot',
                response_text,
                risk_level.value,
                ','.join(concerns) if concerns else None,
                response_time_ms
            )

            # Update conversation metrics
            await self.update_conversation_metrics(
                conversation['conversation_id'],
                risk_level
            )

            # Log crisis events if needed
            if risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
                await self.log_crisis_event(
                    conversation['conversation_id'],
                    concerns,
                    risk_level,
                    response_text
                )

            return {
                'status': 'success',
                'risk_level': risk_level.value,
                'concerns': concerns,
                'response_time_ms': response_time_ms
            }

        except Exception as e:
            logger.error(f"Error processing Telegram message: {e}")
            return {'status': 'error', 'error': str(e)}

    async def get_or_create_conversation(self, user_id: int, username: str) -> Dict:
        """Get existing or create new conversation for user"""
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Check for existing active conversation
        cur.execute("""
            SELECT * FROM veteran_support_conversations
            WHERE telegram_user_id = %s AND status = 'active'
            ORDER BY last_message_at DESC LIMIT 1
        """, (user_id,))

        conversation = cur.fetchone()

        if not conversation:
            # Create new conversation
            cur.execute("""
                INSERT INTO veteran_support_conversations
                (telegram_user_id, telegram_username)
                VALUES (%s, %s)
                RETURNING *
            """, (user_id, username))
            conversation = cur.fetchone()
            conn.commit()

        cur.close()
        conn.close()

        return dict(conversation)

    async def get_conversation_history(self, conversation_id: int, limit: int = 10) -> List[Dict]:
        """Get recent conversation history"""
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute("""
            SELECT * FROM veteran_support_messages
            WHERE conversation_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """, (conversation_id, limit))

        messages = cur.fetchall()
        cur.close()
        conn.close()

        return [dict(m) for m in reversed(messages)]

    async def store_message(self, conversation_id: int, telegram_message_id: Optional[int],
                           sender: str, text: str, risk_assessment: str,
                           intervention_type: Optional[str], response_time_ms: Optional[int] = None):
        """Store message in database"""
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO veteran_support_messages
            (conversation_id, telegram_message_id, sender, message_text,
             risk_assessment, intervention_type, response_time_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (conversation_id, telegram_message_id, sender, text,
              risk_assessment, intervention_type, response_time_ms))

        conn.commit()
        cur.close()
        conn.close()

    async def update_conversation_metrics(self, conversation_id: int, risk_level: RiskLevel):
        """Update conversation metrics"""
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()

        crisis_increment = 1 if risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH] else 0

        cur.execute("""
            UPDATE veteran_support_conversations
            SET last_message_at = CURRENT_TIMESTAMP,
                total_messages = total_messages + 1,
                crisis_interventions = crisis_interventions + %s,
                risk_level = %s
            WHERE conversation_id = %s
        """, (crisis_increment, risk_level.value, conversation_id))

        conn.commit()
        cur.close()
        conn.close()

    async def log_crisis_event(self, conversation_id: int, concerns: List[str],
                               risk_level: RiskLevel, intervention: str):
        """Log crisis events for tracking and improvement"""
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()

        crisis_type = concerns[0] if concerns else 'general_crisis'

        cur.execute("""
            INSERT INTO veteran_crisis_events
            (conversation_id, crisis_type, severity, intervention_provided)
            VALUES (%s, %s, %s, %s)
        """, (conversation_id, crisis_type, risk_level.value, intervention[:500]))

        conn.commit()
        cur.close()
        conn.close()

    async def send_telegram_message(self, chat_id: int, text: str,
                                   reply_to_message_id: Optional[int] = None):
        """Send message via Telegram Bot API"""
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

        payload = {
            'chat_id': chat_id,
            'text': text,
            'parse_mode': 'HTML'
        }

        if reply_to_message_id:
            payload['reply_to_message_id'] = reply_to_message_id

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                result = await response.json()
                if not result.get('ok'):
                    logger.error(f"Failed to send Telegram message: {result}")
                return result

    async def get_support_metrics(self) -> Dict:
        """Get metrics for monitoring and improvement"""
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Get overall metrics
        cur.execute("""
            SELECT
                COUNT(DISTINCT telegram_user_id) as total_veterans,
                COUNT(*) as total_conversations,
                SUM(total_messages) as total_messages,
                SUM(crisis_interventions) as total_interventions,
                AVG(total_messages) as avg_messages_per_conversation
            FROM veteran_support_conversations
        """)
        overall_metrics = dict(cur.fetchone())

        # Get risk level distribution
        cur.execute("""
            SELECT risk_level, COUNT(*) as count
            FROM veteran_support_conversations
            WHERE risk_level IS NOT NULL
            GROUP BY risk_level
        """)
        risk_distribution = {row['risk_level']: row['count'] for row in cur.fetchall()}

        # Get crisis event statistics
        cur.execute("""
            SELECT
                crisis_type,
                COUNT(*) as occurrences,
                AVG(CASE WHEN resolved_at IS NOT NULL
                    THEN EXTRACT(EPOCH FROM (resolved_at - created_at))/3600
                    ELSE NULL END) as avg_resolution_hours
            FROM veteran_crisis_events
            GROUP BY crisis_type
        """)
        crisis_stats = [dict(row) for row in cur.fetchall()]

        # Get response time metrics
        cur.execute("""
            SELECT
                AVG(response_time_ms) as avg_response_time_ms,
                MIN(response_time_ms) as min_response_time_ms,
                MAX(response_time_ms) as max_response_time_ms,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY response_time_ms) as median_response_time_ms
            FROM veteran_support_messages
            WHERE sender = 'bot' AND response_time_ms IS NOT NULL
        """)
        response_metrics = dict(cur.fetchone())

        cur.close()
        conn.close()

        return {
            'overall': overall_metrics,
            'risk_distribution': risk_distribution,
            'crisis_statistics': crisis_stats,
            'response_times': response_metrics,
            'generated_at': datetime.now().isoformat()
        }


class VeteranSupportTester:
    """Comprehensive testing framework for veteran support responses"""

    def __init__(self, guardian_system: VeteranGuardianSystem):
        self.guardian = guardian_system
        self.test_scenarios = [
            {
                'name': 'suicide_crisis',
                'message': "I can't do this anymore. I'm thinking about ending it all tonight.",
                'expected_risk': RiskLevel.CRITICAL,
                'required_elements': ['988', 'Crisis Line', 'here with you', 'matters']
            },
            {
                'name': 'substance_relapse',
                'message': "I relapsed last night. I'm drunk again and I hate myself for it.",
                'expected_risk': RiskLevel.HIGH,
                'required_elements': ['part of recovery', 'not failure', 'strength']
            },
            {
                'name': 'ptsd_flashback',
                'message': "I keep having flashbacks to the IED attack. I can't sleep.",
                'expected_risk': RiskLevel.HIGH,
                'required_elements': ['safe now', 'grounding', 'normal response']
            },
            {
                'name': 'isolation',
                'message': "Nobody understands what I've been through. I feel so alone.",
                'expected_risk': RiskLevel.MODERATE,
                'required_elements': ['brotherhood', 'not alone', 'understand']
            },
            {
                'name': 'check_in',
                'message': "Just checking in. Had a rough day but managing.",
                'expected_risk': RiskLevel.LOW,
                'required_elements': ['here', 'listening', 'managing']
            }
        ]

    async def run_comprehensive_tests(self) -> Dict:
        """Run comprehensive test suite"""
        results = {
            'total_tests': len(self.test_scenarios),
            'passed': 0,
            'failed': 0,
            'test_details': []
        }

        for scenario in self.test_scenarios:
            logger.info(f"Testing scenario: {scenario['name']}")

            # Test risk assessment
            risk_level, concerns = await self.guardian.assess_risk_level(scenario['message'])

            # Generate response
            response = await self.guardian.generate_therapeutic_response(
                scenario['message'],
                risk_level,
                concerns,
                []  # Empty history for testing
            )

            # Validate response
            test_result = {
                'scenario': scenario['name'],
                'risk_assessment_correct': risk_level == scenario['expected_risk'],
                'detected_concerns': concerns,
                'response_adequate': all(
                    element.lower() in response.lower()
                    for element in scenario['required_elements']
                ),
                'response_length': len(response),
                'missing_elements': [
                    element for element in scenario['required_elements']
                    if element.lower() not in response.lower()
                ]
            }

            test_result['passed'] = (
                test_result['risk_assessment_correct'] and
                test_result['response_adequate']
            )

            if test_result['passed']:
                results['passed'] += 1
                logger.info(f"✓ {scenario['name']} passed")
            else:
                results['failed'] += 1
                logger.error(f"✗ {scenario['name']} failed: {test_result['missing_elements']}")

            results['test_details'].append(test_result)

        results['success_rate'] = (results['passed'] / results['total_tests']) * 100

        return results