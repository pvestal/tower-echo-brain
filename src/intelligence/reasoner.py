"""
Reasoning Engine for Echo Brain
The brain that ties it all together.
"""

import asyncio
import httpx
import json
import logging
import re
from datetime import datetime
from typing import Dict, Any, List, Optional

from .schemas import (
    QueryType, Response, Diagnosis, ActionPlan,
    Step, ActionType, SafetyLevel
)

logger = logging.getLogger(__name__)


class ReasoningEngine:
    """
    Processes queries by:
    1. Understanding intent
    2. Gathering relevant context
    3. Planning actions if needed
    4. Executing or answering
    5. Learning from results
    """

    def __init__(self):
        self.code_intel = None
        self.system_model = None
        self.procedures = None
        self.executor = None
        self.knowledge = None

    def _get_components(self):
        """Lazy load components to avoid circular imports"""
        if not self.code_intel:
            from .code_index import get_code_intelligence
            from .system_model import get_system_model
            from .procedures import get_procedure_library
            from .executor import get_action_executor
            from ..core.unified_knowledge import get_unified_knowledge

            self.code_intel = get_code_intelligence()
            self.system_model = get_system_model()
            self.procedures = get_procedure_library()
            self.executor = get_action_executor()
            self.knowledge = get_unified_knowledge()

    async def process(self, query: str, allow_actions: bool = False,
                     context: Dict[str, Any] = None) -> Response:
        """
        Main entry point for all queries.

        1. Classify query type
        2. Gather context based on type
        3. If action needed and allowed: plan and execute
        4. Generate response using LLM
        5. Learn from interaction
        """
        start_time = datetime.now()
        context = context or {}

        self._get_components()

        try:
            # 1. Classify query type
            query_type = await self._classify_query(query)
            logger.info(f"Classified query as: {query_type.value}")

            # 2. Gather context based on type
            relevant_context = await self._gather_context(query, query_type, context)

            # 3. Plan actions if needed
            action_plan = None
            actions_taken = []

            if allow_actions and self._needs_action(query, query_type):
                action_plan = await self.plan_action(query, relevant_context)

                if action_plan and action_plan.steps:
                    logger.info(f"Executing action plan with {len(action_plan.steps)} steps")

                    # Execute actions
                    for i, step in enumerate(action_plan.steps):
                        try:
                            if step.safety_level == SafetyLevel.DANGEROUS and not action_plan.requires_confirmation:
                                actions_taken.append({
                                    'step': i + 1,
                                    'action': step.action.value,
                                    'success': False,
                                    'message': 'Dangerous action blocked without confirmation'
                                })
                                break

                            # Execute the step
                            result = await self._execute_step(step)
                            actions_taken.append({
                                'step': i + 1,
                                'action': step.action.value,
                                'success': result.get('success', False),
                                'result': result
                            })

                            if not result.get('success', False) and step.safety_level != SafetyLevel.SAFE:
                                logger.warning(f"Step {i+1} failed, stopping execution")
                                break

                        except Exception as e:
                            logger.error(f"Error executing step {i+1}: {e}")
                            actions_taken.append({
                                'step': i + 1,
                                'action': step.action.value,
                                'success': False,
                                'error': str(e)
                            })
                            break

            # 4. Generate response using LLM
            response_text = await self._generate_response(
                query, query_type, relevant_context, actions_taken
            )

            # 5. Calculate execution time
            execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            # 6. Learn from interaction (simplified for now)
            await self._learn_from_interaction(query, query_type, response_text, actions_taken)

            return Response(
                query=query,
                query_type=query_type,
                response=response_text,
                actions_taken=actions_taken,
                confidence=self._calculate_confidence(relevant_context, actions_taken),
                sources=self._extract_sources(relevant_context),
                execution_time_ms=execution_time_ms
            )

        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            return Response(
                query=query,
                query_type=QueryType.GENERAL_KNOWLEDGE,
                response=f"I encountered an error processing your query: {str(e)}",
                actions_taken=[],
                confidence=0.0,
                sources=[],
                execution_time_ms=execution_time_ms
            )

    async def _classify_query(self, query: str) -> QueryType:
        """Classify the type of query"""
        query_lower = query.lower()

        # Self-introspection patterns
        if any(pattern in query_lower for pattern in [
            'what are you', 'who are you', 'tell me about yourself',
            'your capabilities', 'what can you do', 'how do you work'
        ]):
            return QueryType.SELF_INTROSPECTION

        # System query patterns
        if any(pattern in query_lower for pattern in [
            'service status', 'is running', 'check service', 'system health',
            'port', 'process', 'systemctl', 'docker ps', 'memory usage',
            'disk space', 'cpu usage', 'logs', 'errors'
        ]):
            return QueryType.SYSTEM_QUERY

        # Code query patterns
        if any(pattern in query_lower for pattern in [
            'function', 'class', 'method', 'code', 'implementation',
            'how does', 'what does', 'find', 'search code', 'definition',
            'api endpoint', 'route', 'imports'
        ]):
            return QueryType.CODE_QUERY

        # Action request patterns
        if any(pattern in query_lower for pattern in [
            'restart', 'start', 'stop', 'install', 'update', 'fix',
            'deploy', 'configure', 'run', 'execute', 'create',
            'delete', 'modify', 'change'
        ]):
            return QueryType.ACTION_REQUEST

        # Default to general knowledge
        return QueryType.GENERAL_KNOWLEDGE

    async def _gather_context(self, query: str, query_type: QueryType,
                             user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Gather relevant context based on query type"""
        context = {'query_type': query_type, 'user_context': user_context}

        try:
            if query_type == QueryType.SELF_INTROSPECTION:
                # Get Echo Brain's own status and capabilities
                context['echo_brain_info'] = await self._get_echo_brain_info()

            elif query_type == QueryType.SYSTEM_QUERY:
                # Get system status and service information
                context['system_info'] = await self._get_system_context(query)

            elif query_type == QueryType.CODE_QUERY:
                # Search code and get relevant definitions
                context['code_info'] = await self._get_code_context(query)

            elif query_type == QueryType.ACTION_REQUEST:
                # Find relevant procedures and system state
                context['action_context'] = await self._get_action_context(query)

            # Always get general knowledge context
            context['knowledge_context'] = await self.knowledge.get_context(query)

        except Exception as e:
            logger.error(f"Error gathering context: {e}")
            context['error'] = str(e)

        return context

    async def _get_echo_brain_info(self) -> Dict[str, Any]:
        """Get information about Echo Brain itself"""
        try:
            # Get service status
            service_status = await self.system_model.get_service_status('tower-echo-brain')

            # Get recent actions
            recent_actions = await self.executor.get_recent_actions(10)

            # Get available procedures
            procedures = await self.procedures.list_procedures()

            return {
                'service_status': service_status.dict(),
                'recent_actions': recent_actions,
                'available_procedures': procedures[:5],  # Top 5
                'components': {
                    'code_intelligence': 'Available - can analyze Tower codebase',
                    'system_model': 'Available - monitors Tower services',
                    'action_executor': 'Available - can execute safe commands',
                    'procedure_library': f'Available - {len(procedures)} procedures',
                    'unified_knowledge': 'Available - facts, vectors, conversations'
                }
            }
        except Exception as e:
            logger.error(f"Error getting Echo Brain info: {e}")
            return {'error': str(e)}

    async def _get_system_context(self, query: str) -> Dict[str, Any]:
        """Get system-related context"""
        try:
            # Extract service name if mentioned
            service_name = self._extract_service_name(query)

            context = {}

            if service_name:
                # Get specific service info
                status = await self.system_model.get_service_status(service_name)
                dependencies = await self.system_model.get_service_dependencies(service_name)
                config = await self.system_model.get_service_config(service_name)

                context['service_info'] = {
                    'name': service_name,
                    'status': status.dict(),
                    'dependencies': dependencies,
                    'config': config
                }
            else:
                # Get general system info
                services = await self.system_model.discover_services()
                network_map = await self.system_model.get_network_map()

                context['general_info'] = {
                    'active_services': [s.dict() for s in services[:10]],
                    'network_connections': len(network_map.connections)
                }

            return context

        except Exception as e:
            logger.error(f"Error getting system context: {e}")
            return {'error': str(e)}

    async def _get_code_context(self, query: str) -> Dict[str, Any]:
        """Get code-related context"""
        try:
            # Extract function/class name if mentioned
            symbol_name = self._extract_symbol_name(query)

            context = {}

            if symbol_name:
                # Search for specific symbol
                definitions = await self.code_intel.find_definition(symbol_name)
                symbols = await self.code_intel.search_symbols(symbol_name)

                context['symbol_info'] = {
                    'name': symbol_name,
                    'definitions': [d.dict() for d in definitions],
                    'symbols': symbols[:5]
                }
            else:
                # General code search
                search_terms = self._extract_search_terms(query)
                if search_terms:
                    symbols = await self.code_intel.search_symbols(search_terms[0])
                    context['search_results'] = symbols[:10]

            # Get API endpoints if query mentions API/endpoint
            if 'api' in query.lower() or 'endpoint' in query.lower():
                endpoints = await self.code_intel.find_endpoints()
                context['endpoints'] = [e.dict() for e in endpoints[:10]]

            return context

        except Exception as e:
            logger.error(f"Error getting code context: {e}")
            return {'error': str(e)}

    async def _get_action_context(self, query: str) -> Dict[str, Any]:
        """Get context for action requests"""
        try:
            # Find matching procedure
            procedure = await self.procedures.find_procedure(query)

            context = {'matching_procedure': procedure.dict() if procedure else None}

            # Extract service name for service actions
            service_name = self._extract_service_name(query)
            if service_name:
                status = await self.system_model.get_service_status(service_name)
                context['target_service'] = {
                    'name': service_name,
                    'status': status.dict()
                }

            return context

        except Exception as e:
            logger.error(f"Error getting action context: {e}")
            return {'error': str(e)}

    def _extract_service_name(self, query: str) -> Optional[str]:
        """Extract service name from query"""
        # Known Tower services
        services = [
            'tower-echo-brain', 'tower-auth', 'tower-kb', 'tower-frontend',
            'comfyui', 'ollama', 'qdrant', 'nginx', 'postgresql'
        ]

        query_lower = query.lower()
        for service in services:
            if service in query_lower:
                return service

        # Look for generic patterns
        patterns = [
            r'service\s+(\w+[\w-]*)',
            r'(\w+[\w-]*)\s+service',
            r'restart\s+(\w+[\w-]*)',
            r'start\s+(\w+[\w-]*)',
            r'stop\s+(\w+[\w-]*)'
        ]

        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                candidate = match.group(1)
                if candidate in services:
                    return candidate

        return None

    def _extract_symbol_name(self, query: str) -> Optional[str]:
        """Extract function/class name from query"""
        patterns = [
            r'function\s+(\w+)',
            r'class\s+(\w+)',
            r'method\s+(\w+)',
            r'(\w+)\s+function',
            r'(\w+)\s+class',
            r'find\s+(\w+)',
            r'what.*does\s+(\w+)',
            r'how.*does\s+(\w+)'
        ]

        query_lower = query.lower()
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                return match.group(1)

        return None

    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract search terms from query"""
        # Remove common words and extract meaningful terms
        stop_words = {'what', 'is', 'the', 'how', 'does', 'do', 'where', 'find', 'search'}

        words = re.findall(r'\w+', query.lower())
        meaningful = [w for w in words if w not in stop_words and len(w) > 2]

        return meaningful[:3]  # Top 3 terms

    def _needs_action(self, query: str, query_type: QueryType) -> bool:
        """Determine if query requires action execution"""
        if query_type != QueryType.ACTION_REQUEST:
            return False

        # Check for action verbs
        action_verbs = [
            'restart', 'start', 'stop', 'install', 'update', 'fix',
            'deploy', 'configure', 'run', 'execute', 'create',
            'delete', 'modify', 'change'
        ]

        query_lower = query.lower()
        return any(verb in query_lower for verb in action_verbs)

    async def plan_action(self, query: str, context: Dict[str, Any]) -> Optional[ActionPlan]:
        """Break down a request into executable steps"""
        try:
            # Check if we have a matching procedure
            matching_procedure = context.get('action_context', {}).get('matching_procedure')

            if matching_procedure:
                # Use existing procedure
                steps = []
                for step_data in matching_procedure['steps']:
                    steps.append(Step(
                        action=ActionType(step_data['action']),
                        target=step_data['target'],
                        command=step_data['command'],
                        description=step_data['description'],
                        safety_level=SafetyLevel(step_data['safety_level']),
                        timeout_seconds=step_data.get('timeout_seconds', 30)
                    ))

                return ActionPlan(
                    intent=query,
                    steps=steps,
                    estimated_time=len(steps) * 30,  # Rough estimate
                    requires_confirmation=any(
                        s.safety_level != SafetyLevel.SAFE for s in steps
                    )
                )

            # Create ad-hoc action plan
            steps = await self._create_adhoc_plan(query, context)

            if steps:
                return ActionPlan(
                    intent=query,
                    steps=steps,
                    estimated_time=len(steps) * 30,
                    requires_confirmation=any(
                        s.safety_level != SafetyLevel.SAFE for s in steps
                    )
                )

        except Exception as e:
            logger.error(f"Error planning action: {e}")

        return None

    async def _create_adhoc_plan(self, query: str, context: Dict[str, Any]) -> List[Step]:
        """Create an ad-hoc action plan"""
        steps = []
        query_lower = query.lower()

        # Simple pattern matching for common actions
        if 'restart' in query_lower:
            service_name = self._extract_service_name(query)
            if service_name:
                steps = [
                    Step(
                        action=ActionType.SHELL,
                        target="systemd",
                        command=f"systemctl status {service_name}",
                        description="Check current service status",
                        safety_level=SafetyLevel.SAFE
                    ),
                    Step(
                        action=ActionType.SERVICE_MANAGE,
                        target="systemd",
                        command=f"restart {service_name}",
                        description="Restart the service",
                        safety_level=SafetyLevel.NEEDS_CONFIRM
                    )
                ]

        elif 'check' in query_lower and 'status' in query_lower:
            service_name = self._extract_service_name(query)
            if service_name:
                steps = [
                    Step(
                        action=ActionType.SHELL,
                        target="systemd",
                        command=f"systemctl status {service_name}",
                        description="Check service status",
                        safety_level=SafetyLevel.SAFE
                    )
                ]

        return steps

    async def _execute_step(self, step: Step) -> Dict[str, Any]:
        """Execute a single action step"""
        try:
            if step.action == ActionType.SHELL:
                result = await self.executor.execute_shell(
                    step.command,
                    safe=(step.safety_level == SafetyLevel.SAFE),
                    timeout=step.timeout_seconds
                )
                return {
                    'success': result.success,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'exit_code': result.exit_code
                }

            elif step.action == ActionType.SERVICE_MANAGE:
                parts = step.command.split(' ', 1)
                action = parts[0]
                service = parts[1] if len(parts) > 1 else step.target

                result = await self.executor.manage_service(
                    service, action,
                    confirm_dangerous=(step.safety_level != SafetyLevel.DANGEROUS)
                )

                return {
                    'success': result.success,
                    'new_status': result.new_status,
                    'message': result.message
                }

            elif step.action == ActionType.API_CALL:
                parts = step.command.split(' ', 1)
                method = parts[0] if len(parts) > 0 else 'GET'
                url = parts[1] if len(parts) > 1 else step.target

                result = await self.executor.call_api(
                    url, method, timeout=step.timeout_seconds
                )

                return {
                    'success': result.success,
                    'status_code': result.status_code,
                    'response_data': result.response_data
                }

            else:
                return {'success': False, 'error': f'Unsupported action type: {step.action}'}

        except Exception as e:
            logger.error(f"Error executing step: {e}")
            return {'success': False, 'error': str(e)}

    async def _generate_response(self, query: str, query_type: QueryType,
                                context: Dict[str, Any], actions_taken: List[Dict[str, Any]]) -> str:
        """Generate response using LLM with structured context"""
        try:
            # Build context for LLM
            system_prompt = self._build_system_prompt(query_type)
            context_text = self._build_context_text(context, actions_taken)
            user_prompt = f"{context_text}\n\nUser Query: {query}"

            # Call LLM
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "mistral:7b",
                        "prompt": f"{system_prompt}\n\n{user_prompt}",
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "top_p": 0.9
                        }
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', 'No response generated')
                else:
                    return f"Error generating response: {response.status_code}"

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error generating a response: {str(e)}"

    def _build_system_prompt(self, query_type: QueryType) -> str:
        """Build system prompt based on query type"""
        base_prompt = """You are Echo Brain, Patrick's intelligent AI assistant for the Tower system. You have access to real-time information about services, code, and system state."""

        if query_type == QueryType.SELF_INTROSPECTION:
            return f"{base_prompt} You're being asked about your own capabilities and status. Be accurate and specific about what you can do."

        elif query_type == QueryType.SYSTEM_QUERY:
            return f"{base_prompt} You're answering a question about system status or services. Use the provided context to give specific, accurate information."

        elif query_type == QueryType.CODE_QUERY:
            return f"{base_prompt} You're being asked about code. Use the code analysis results to provide accurate information about functions, classes, and implementations."

        elif query_type == QueryType.ACTION_REQUEST:
            return f"{base_prompt} You've been asked to perform an action. Explain what you did and the results clearly."

        else:
            return f"{base_prompt} Answer the question using the provided context and your knowledge."

    def _build_context_text(self, context: Dict[str, Any], actions_taken: List[Dict[str, Any]]) -> str:
        """Build context text for LLM"""
        parts = []

        # Add knowledge context
        if 'knowledge_context' in context:
            knowledge = context['knowledge_context']
            if knowledge.get('prompt_context'):
                parts.append(f"RELEVANT KNOWLEDGE:\n{knowledge['prompt_context']}")

        # Add specific context based on type
        for key, value in context.items():
            if key.endswith('_info') or key.endswith('_context'):
                parts.append(f"{key.upper()}:\n{json.dumps(value, indent=2, default=str)}")

        # Add actions taken
        if actions_taken:
            actions_text = "\n".join([
                f"Step {a['step']}: {a.get('action', 'unknown')} - {'Success' if a.get('success') else 'Failed'}"
                for a in actions_taken
            ])
            parts.append(f"ACTIONS PERFORMED:\n{actions_text}")

        return "\n\n".join(parts)

    def _calculate_confidence(self, context: Dict[str, Any], actions_taken: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for response"""
        confidence = 0.5  # Base confidence

        # Increase confidence if we have relevant context
        if context.get('knowledge_context', {}).get('total_sources', 0) > 0:
            confidence += 0.2

        # Increase if we found specific info
        if any(key.endswith('_info') for key in context.keys()):
            confidence += 0.2

        # Increase if actions succeeded
        if actions_taken:
            success_rate = sum(1 for a in actions_taken if a.get('success')) / len(actions_taken)
            confidence += success_rate * 0.1

        return min(confidence, 1.0)

    def _extract_sources(self, context: Dict[str, Any]) -> List[str]:
        """Extract sources from context"""
        sources = []

        knowledge = context.get('knowledge_context', {})
        for fact in knowledge.get('facts', []):
            sources.append(f"Fact: {fact.source_type}")

        for vec in knowledge.get('vectors', []):
            sources.append(f"Memory: {vec.source_type}")

        return list(set(sources))

    async def _learn_from_interaction(self, query: str, query_type: QueryType,
                                    response: str, actions_taken: List[Dict[str, Any]]):
        """Learn from this interaction (simplified)"""
        try:
            # If actions were taken successfully, potentially learn a new procedure
            if actions_taken and all(a.get('success') for a in actions_taken):
                if len(actions_taken) > 1:  # Multi-step process worth learning
                    steps = [
                        {
                            'action': a.get('action', 'shell'),
                            'command': a.get('result', {}).get('command', ''),
                            'description': f"Step {a['step']}",
                            'safety_level': 'safe'
                        }
                        for a in actions_taken
                    ]

                    await self.procedures.learn_procedure(query, steps, "success")

        except Exception as e:
            logger.debug(f"Error in learning: {e}")

    async def explain_code(self, query: str) -> str:
        """Explain code with actual understanding, not text search"""
        self._get_components()

        try:
            symbol_name = self._extract_symbol_name(query)

            if symbol_name:
                definitions = await self.code_intel.find_definition(symbol_name)
                symbols = await self.code_intel.search_symbols(symbol_name)

                if definitions:
                    # Get the first definition
                    definition = definitions[0]

                    # Get dependencies for the file
                    deps = await self.code_intel.get_dependencies(definition.file_path)

                    explanation = f"**{symbol_name}** is defined in `{definition.file_path}:{definition.line_start}`\n\n"

                    # Find matching symbol details
                    for symbol in symbols:
                        if symbol['name'] == symbol_name or symbol['name'].endswith(f".{symbol_name}"):
                            explanation += f"**Type**: {symbol['type']}\n"
                            if symbol['signature']:
                                explanation += f"**Signature**: `{symbol['signature']}`\n"
                            if symbol['docstring']:
                                explanation += f"**Description**: {symbol['docstring']}\n"
                            break

                    explanation += f"\n**Dependencies**: {', '.join(deps.dependencies[:5])}\n"
                    explanation += f"**Used by**: {', '.join(deps.dependents[:5])}"

                    return explanation

            return f"Could not find detailed information about '{symbol_name}'"

        except Exception as e:
            logger.error(f"Error explaining code: {e}")
            return f"Error analyzing code: {str(e)}"

    async def diagnose(self, issue: str) -> Diagnosis:
        """Systematic diagnosis using procedures"""
        self._get_components()

        try:
            # Find diagnostic procedure
            procedure = await self.procedures.find_procedure(f"diagnose {issue}")

            if procedure:
                # Execute diagnostic procedure
                result = await self.procedures.execute_procedure(procedure, {'issue': issue})

                findings = []
                recommendations = []

                for step_result in result.get('results', []):
                    if step_result.get('success'):
                        stdout = step_result.get('stdout', '')
                        if stdout:
                            findings.append(f"Step {step_result['step']}: {stdout[:200]}")

                        # Extract recommendations based on step type
                        if 'failed' in stdout.lower():
                            recommendations.append(f"Investigate {step_result.get('description', 'issue')}")

                return Diagnosis(
                    issue=issue,
                    findings=findings,
                    root_cause=result.get('error_message'),
                    recommendations=recommendations or ['Run detailed health check'],
                    severity='medium',
                    estimated_fix_time=30
                )

            else:
                # Generic diagnosis
                return Diagnosis(
                    issue=issue,
                    findings=['No specific diagnostic procedure found'],
                    recommendations=['Check service logs', 'Verify configuration', 'Check dependencies'],
                    severity='low',
                    estimated_fix_time=15
                )

        except Exception as e:
            logger.error(f"Error diagnosing issue: {e}")
            return Diagnosis(
                issue=issue,
                findings=[f"Diagnosis error: {str(e)}"],
                recommendations=['Check Echo Brain logs', 'Verify system connectivity'],
                severity='high',
                estimated_fix_time=60
            )


# Singleton instance
_reasoning_engine = None

def get_reasoning_engine() -> ReasoningEngine:
    """Get or create singleton instance"""
    global _reasoning_engine
    if not _reasoning_engine:
        _reasoning_engine = ReasoningEngine()
    return _reasoning_engine