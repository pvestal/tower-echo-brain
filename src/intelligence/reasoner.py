"""
Reasoning Engine for Echo Brain
The brain that ties it all together.
"""

import asyncio
import httpx
import json
import logging
import os
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
            'service status', 'is running', 'are running', 'services running', 'what services',
            'check service', 'system health', 'service health', 'running on',
            'port', 'process', 'systemctl', 'docker ps', 'memory usage',
            'disk space', 'cpu usage', 'logs', 'errors', 'what port'
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
                code_context = await self._get_code_context(query)
                context.update(code_context)  # Merge instead of nesting

            elif query_type == QueryType.ACTION_REQUEST:
                # Find relevant procedures and system state
                context['action_context'] = await self._get_action_context(query)

            # Get domain-filtered context using the pipeline's context layer
            # This prevents contamination by filtering based on query intent
            try:
                from src.pipeline.context_layer import ContextLayer
                if not hasattr(self, '_context_layer'):
                    self._context_layer = ContextLayer()
                    await self._context_layer.initialize()

                # Get properly filtered context based on intent
                context_package = await self._context_layer.retrieve(query)

                # Only use knowledge context if we have relevant sources
                if context_package.total_sources_found > 0:
                    context['knowledge_context'] = {
                        'intent': context_package.intent.value,
                        'sources_found': context_package.total_sources_found,
                        'prompt_context': context_package.assembled_context,
                        'confidence': len(context_package.sources) / 10.0  # Simple confidence metric
                    }
                else:
                    # No relevant context found - don't pollute with unrelated data
                    context['knowledge_context'] = {
                        'intent': context_package.intent.value,
                        'sources_found': 0,
                        'prompt_context': '',
                        'confidence': 0.0
                    }
            except Exception as e:
                logger.warning(f"Pipeline context layer not available, falling back to unfiltered: {e}")
                # Fallback to original unfiltered context (but limit it)
                context['knowledge_context'] = await self.knowledge.get_context(query, max_facts=3, max_vectors=2, max_conversations=2)

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
        """Gather rich code context for code-related queries"""
        try:
            # Extract what they're asking about
            subject = self._extract_code_subject(query)

            context = {}

            if not subject:
                # Fall back to keyword search in symbols
                search_terms = self._extract_search_terms(query)
                if search_terms:
                    symbols = await self.code_intel.search_symbols(search_terms[0])
                    context['search_results'] = symbols[:5]
                return context

            # Search for the symbol in our index
            symbols = await self.code_intel.search_symbols(subject)

            if symbols:
                # Get the best match
                best_match = symbols[0]
                file_path = best_match.get('file_path', '')

                # READ THE ACTUAL SOURCE FILE
                if file_path and os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        lines = content.split('\n')

                        # Extract just the relevant section (class or function)
                        start_line = best_match.get('line_start', 1) - 1  # 0-indexed
                        end_line = best_match.get('line_end', len(lines))

                        # Include some context before and the full definition
                        code_block = '\n'.join(lines[max(0, start_line):min(len(lines), end_line)])

                        # Cap at 3000 chars to not overwhelm the LLM
                        if len(code_block) > 3000:
                            code_block = code_block[:3000] + "\n... (truncated)"

                        context['code_info'] = {
                            'symbol_info': {
                                'name': best_match['name'],
                                'type': best_match['type'],
                                'file_path': file_path,
                                'line_start': best_match.get('line_start'),
                                'line_end': best_match.get('line_end'),
                                'signature': best_match.get('signature', ''),
                                'docstring': best_match.get('docstring', ''),
                                'definitions': [best_match]
                            },
                            'source_code': code_block,
                            'file_path': file_path
                        }

                        # Also get related symbols in the same file
                        all_symbols = await self.code_intel.search_symbols('')
                        all_file_symbols = [s for s in all_symbols if s.get('file_path') == file_path]
                        context['code_info']['related_symbols'] = [
                            {'name': s['name'], 'type': s['type'], 'line': s.get('line_start')}
                            for s in all_file_symbols[:10]
                        ]

                        # Get dependencies
                        deps = await self.code_intel.get_dependencies(file_path)
                        context['code_info']['dependencies'] = deps.dependencies[:10]
                        context['code_info']['dependents'] = deps.dependents[:5]

                    except Exception as e:
                        logger.error(f"Error reading source file {file_path}: {e}")
                        context['code_info'] = {'symbol_info': best_match, 'error': str(e)}
                else:
                    context['code_info'] = {'symbol_info': best_match}

            # Also include any matching endpoints
            try:
                endpoints = await self.code_intel.find_endpoints()
                if subject:
                    relevant_endpoints = [e for e in endpoints
                                         if subject.lower() in e.get('function_name', '').lower()
                                         or subject.lower() in e.get('path_pattern', '').lower()]
                    if relevant_endpoints:
                        context['code_info'] = context.get('code_info', {})
                        context['code_info']['endpoints'] = [e.dict() for e in relevant_endpoints[:5]]
            except Exception as e:
                logger.debug(f"Error getting endpoints: {e}")

            # Get API endpoints if query mentions API/endpoint
            if 'api' in query.lower() or 'endpoint' in query.lower():
                try:
                    endpoints = await self.code_intel.find_endpoints()
                    context['endpoints'] = [e.dict() for e in endpoints[:10]]
                except Exception as e:
                    logger.debug(f"Error getting all endpoints: {e}")

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

    def _extract_code_subject(self, query: str) -> Optional[str]:
        """Extract the code subject (class name, function name) from a query"""
        # First, look for CamelCase words in the original query (case sensitive)
        camel_matches = re.findall(r'[A-Z][a-zA-Z]*(?:[A-Z][a-z]*)*', query)
        if camel_matches:
            # Skip common question words
            question_words = {'What', 'How', 'Why', 'Where', 'When', 'Which', 'Who'}
            for match in camel_matches:
                if match not in question_words:
                    return match
            return camel_matches[0]  # Fallback to first if all are question words

        # If no CamelCase, try to extract from cleaned query
        query_lower = query.lower()

        # Remove common question words
        for prefix in ['what methods does the ', 'what does the ', 'what is the ', 'how does the ',
                       'explain the ', 'describe the ', 'show me the ', 'what methods does ',
                       'what functions does ', 'tell me about ', 'what does ', 'what is ',
                       'how does ', 'explain ', 'describe ', 'show me ']:
            if query_lower.startswith(prefix):
                query_lower = query_lower[len(prefix):]
                break

        # Remove trailing words
        for suffix in [' class have', ' class do', ' class work', ' do',
                       ' have', ' work', ' look like', ' contain',
                       ' class', ' function', ' method']:
            if query_lower.endswith(suffix):
                query_lower = query_lower[:-len(suffix)]
                break

        # Clean up and extract first meaningful word
        words = query_lower.split()
        for word in words:
            if len(word) > 2 and word not in ['the', 'and', 'or', 'with', 'from']:
                return word.capitalize()

        return None

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

        # Add system context in a readable format
        if 'system_info' in context:
            system_info = context['system_info']
            if 'general_info' in system_info:
                services = system_info['general_info'].get('active_services', [])
                parts.append("CURRENT TOWER SERVICES:")
                for service in services:
                    status = service.get('status', 'unknown')
                    port = service.get('port', 'N/A')
                    service_type = service.get('service_type', 'unknown')
                    parts.append(f"  • {service['name']}: {status} (port {port}, {service_type})")

            elif 'service_info' in system_info:
                svc = system_info['service_info']
                parts.append(f"SERVICE INFO FOR {svc['name']}:")
                parts.append(f"  Status: {svc['status'].get('status', 'unknown')}")
                parts.append(f"  Port: {svc['status'].get('port', 'N/A')}")
                if svc.get('dependencies'):
                    parts.append(f"  Dependencies: {', '.join(svc['dependencies'])}")

        # Add code context with actual source code
        if 'code_info' in context:
            code_info = context['code_info']

            if 'source_code' in code_info:
                parts.append(f"SOURCE CODE from {code_info.get('file_path', 'unknown')}:")
                parts.append(f"```python\n{code_info['source_code']}\n```")

            if 'symbol_info' in code_info:
                symbol = code_info['symbol_info']
                parts.append(f"SYMBOL: {symbol.get('name', 'unknown')} ({symbol.get('type', 'unknown')})")
                if symbol.get('signature'):
                    parts.append(f"SIGNATURE: {symbol['signature']}")
                if symbol.get('docstring'):
                    parts.append(f"DOCSTRING: {symbol['docstring']}")

            if 'related_symbols' in code_info:
                parts.append("OTHER SYMBOLS IN THIS FILE:")
                for s in code_info['related_symbols']:
                    parts.append(f"  • {s['name']} ({s['type']}) line {s.get('line', '?')}")

            if 'dependencies' in code_info:
                parts.append(f"IMPORTS: {', '.join(code_info['dependencies'][:8])}")

            if 'dependents' in code_info:
                parts.append(f"USED BY: {', '.join(code_info['dependents'][:5])}")

            if 'endpoints' in code_info:
                parts.append("API ENDPOINTS:")
                for ep in code_info['endpoints']:
                    parts.append(f"  • {ep.get('http_method', '?')} {ep.get('path_pattern', '?')}")

            if 'search_results' in code_info:
                parts.append("CODE SEARCH RESULTS:")
                for result in code_info['search_results'][:5]:
                    parts.append(f"  • {result['name']} ({result['type']}) in {result['file_path']}")

        # Handle standalone endpoints for API queries
        if 'endpoints' in context:
            parts.append("API ENDPOINTS:")
            for endpoint in context['endpoints'][:10]:
                parts.append(f"  • {endpoint.get('http_method', '?')} {endpoint.get('path_pattern', '?')} → {endpoint.get('function_name', '?')}")

        # Add other context as JSON for anything else
        for key, value in context.items():
            if key not in ['knowledge_context', 'system_info', 'code_info'] and (key.endswith('_info') or key.endswith('_context')):
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
            # Extract path from query - could be from URL or in text
            path = self._extract_path(query)

            if path and os.path.exists(path):
                # Explain a specific file
                return await self._explain_file(path)

            # Fall back to symbol search
            symbol_name = self._extract_symbol_name(query)
            if symbol_name:
                return await self._explain_symbol(symbol_name)

            return f"Could not find code to explain from: '{query}'"

        except Exception as e:
            logger.error(f"Error explaining code: {e}")
            return f"Error analyzing code: {str(e)}"

    def _extract_path(self, query: str) -> Optional[str]:
        """Extract file path from query"""
        # Look for patterns like "explain /opt/..." or just paths
        import re

        # Common path patterns
        path_patterns = [
            r'/opt/tower-[^/]+/src/[^\s]+\.py',  # Tower service paths
            r'/[^\s]+\.py',  # Any Python file path
            r'src/[^\s]+\.py',  # Relative src paths
            r'[^\s/]+\.py'  # Just filename.py
        ]

        for pattern in path_patterns:
            match = re.search(pattern, query)
            if match:
                candidate_path = match.group(0)

                # Try absolute path first
                if os.path.exists(candidate_path):
                    return candidate_path

                # Try with /opt/tower-echo-brain prefix
                full_path = f"/opt/tower-echo-brain/{candidate_path}"
                if os.path.exists(full_path):
                    return full_path

                # Try just the filename in common locations
                if '/' not in candidate_path:
                    for search_dir in ['/opt/tower-echo-brain/src', '/opt/tower-echo-brain/src/core']:
                        test_path = f"{search_dir}/{candidate_path}"
                        if os.path.exists(test_path):
                            return test_path

        return None

    async def _explain_file(self, file_path: str) -> str:
        """Explain a specific file"""
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Get symbols from our index for this file
            symbols = await self.code_intel.search_symbols('')  # Get all symbols
            file_symbols = [s for s in symbols if s['file_path'] == file_path]

            # Get dependencies
            deps = await self.code_intel.get_dependencies(file_path)

            # Build explanation
            explanation = f"# {os.path.basename(file_path)}\n\n"
            explanation += f"**Location**: `{file_path}`\n"
            explanation += f"**Lines**: {len(content.splitlines())}\n\n"

            if file_symbols:
                explanation += "**Components:**\n"
                classes = [s for s in file_symbols if s['type'] == 'class']
                functions = [s for s in file_symbols if s['type'] == 'function']

                if classes:
                    explanation += f"  • Classes: {', '.join(c['name'] for c in classes[:5])}\n"
                if functions:
                    explanation += f"  • Functions: {', '.join(f['name'] for f in functions[:5])}\n"
                explanation += "\n"

            if deps.dependencies:
                explanation += f"**Imports**: {', '.join(deps.dependencies[:8])}\n"

            if deps.dependents:
                explanation += f"**Used by**: {', '.join(deps.dependents[:5])}\n"

            # Add first docstring or comment for context
            lines = content.split('\n')
            for line in lines[:20]:
                if line.strip().startswith('"""') or line.strip().startswith("'''"):
                    # Found docstring
                    docstring_lines = []
                    quote_type = '"""' if '"""' in line else "'''"

                    if line.count(quote_type) >= 2:
                        # Single line docstring
                        docstring = line.strip().replace(quote_type, '').strip()
                        explanation += f"\n**Purpose**: {docstring}\n"
                    else:
                        # Multi-line docstring
                        docstring_lines.append(line.replace(quote_type, '').strip())
                        for next_line in lines[lines.index(line)+1:lines.index(line)+10]:
                            docstring_lines.append(next_line.strip())
                            if quote_type in next_line:
                                break
                        docstring = ' '.join(docstring_lines).replace(quote_type, '').strip()
                        explanation += f"\n**Purpose**: {docstring[:200]}...\n"
                    break

            return explanation

        except Exception as e:
            logger.error(f"Error explaining file {file_path}: {e}")
            return f"Error reading file: {str(e)}"

    async def _explain_symbol(self, symbol_name: str) -> str:
        """Explain a specific symbol (function/class)"""
        try:
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

                explanation += f"\n**File Dependencies**: {', '.join(deps.dependencies[:5])}\n"
                explanation += f"**Used by**: {', '.join(deps.dependents[:5])}"

                return explanation

            return f"Could not find definition for '{symbol_name}'"

        except Exception as e:
            logger.error(f"Error explaining symbol {symbol_name}: {e}")
            return f"Error analyzing symbol: {str(e)}"

    async def diagnose(self, issue: str) -> Diagnosis:
        """Systematic diagnosis using procedures"""
        self._get_components()

        try:
            # Find diagnostic procedure
            procedure = await self.procedures.find_procedure(f"diagnose {issue}")

            if procedure:
                # Execute diagnostic procedure with timeout
                try:
                    result = await asyncio.wait_for(
                        self.procedures.execute_procedure(procedure, {'issue': issue}),
                        timeout=60  # 60-second timeout for diagnosis
                    )
                except asyncio.TimeoutError:
                    return Diagnosis(
                        issue=issue,
                        findings=['Diagnosis timed out after 60 seconds'],
                        root_cause='Procedure execution timeout',
                        recommendations=['Check system resources', 'Verify database connectivity'],
                        severity='high',
                        estimated_fix_time=120
                    )

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