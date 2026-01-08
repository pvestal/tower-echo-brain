#!/usr/bin/env python3
"""
Business Logic Applicator for Echo Brain
Applies learned business logic patterns to responses (separated from pattern matching)
"""
import logging
from typing import Dict, List, Optional
import re

logger = logging.getLogger(__name__)

class BusinessLogicApplicator:
    """
    Responsible for applying business logic patterns to responses.
    Separated from pattern matching for better separation of concerns.
    """

    def __init__(self):
        self.application_stats = {
            'total_applications': 0,
            'successful_applications': 0,
            'failed_applications': 0
        }
        logger.info("✅ BusinessLogicApplicator initialized")

    def apply_patterns_to_response(self, query: str, base_response: str, patterns: List[Dict]) -> str:
        """
        Apply business logic patterns to modify a response.

        Args:
            query: Original user query
            base_response: Base response to enhance
            patterns: List of relevant patterns to apply

        Returns:
            Modified response with business logic applied
        """
        if not patterns:
            return base_response

        try:
            self.application_stats['total_applications'] += 1
            modified_response = base_response

            # Group patterns by type for organized application
            pattern_groups = self._group_patterns_by_type(patterns)

            # Apply patterns in order of priority
            if 'preference' in pattern_groups:
                modified_response = self._apply_preference_patterns(
                    query, modified_response, pattern_groups['preference']
                )

            if 'requirement' in pattern_groups:
                modified_response = self._apply_requirement_patterns(
                    query, modified_response, pattern_groups['requirement']
                )

            if 'anti_pattern' in pattern_groups:
                modified_response = self._apply_anti_pattern_filters(
                    query, modified_response, pattern_groups['anti_pattern']
                )

            if 'context' in pattern_groups:
                modified_response = self._apply_context_patterns(
                    query, modified_response, pattern_groups['context']
                )

            self.application_stats['successful_applications'] += 1

            if modified_response != base_response:
                logger.info(f"🧠 Applied {len(patterns)} business logic patterns to response")

            return modified_response

        except Exception as e:
            self.application_stats['failed_applications'] += 1
            logger.error(f"Failed to apply business logic patterns: {e}")
            return base_response

    def _group_patterns_by_type(self, patterns: List[Dict]) -> Dict[str, List[Dict]]:
        """Group patterns by their type for organized application"""
        groups = {}
        for pattern in patterns:
            pattern_type = pattern.get('type', 'context')
            if pattern_type not in groups:
                groups[pattern_type] = []
            groups[pattern_type].append(pattern)
        return groups

    def _apply_preference_patterns(self, query: str, response: str, patterns: List[Dict]) -> str:
        """Apply preference-based patterns (like technology choices)"""
        for pattern in patterns:
            trigger = pattern.get('trigger', '').lower()
            preference = pattern.get('business_logic', '')

            if trigger in query.lower() and preference:
                # Add preference information to response
                if not self._preference_already_mentioned(response, preference):
                    response = self._inject_preference(response, preference)

        return response

    def _apply_requirement_patterns(self, query: str, response: str, patterns: List[Dict]) -> str:
        """Apply requirement-based patterns (like quality standards)"""
        for pattern in patterns:
            requirement = pattern.get('business_logic', '')

            if 'proof' in requirement.lower() and 'test' not in response.lower():
                response = self._add_proof_requirement(response)

            if 'verify' in requirement.lower() and 'verify' not in response.lower():
                response = self._add_verification_language(response)

        return response

    def _apply_anti_pattern_filters(self, query: str, response: str, patterns: List[Dict]) -> str:
        """Apply anti-pattern filters (remove unwanted terms/approaches)"""
        for pattern in patterns:
            anti_pattern = pattern.get('business_logic', '').lower()

            if 'avoid promotional terms' in anti_pattern:
                response = self._remove_promotional_terms(response)

            if 'avoid assumptions' in anti_pattern:
                response = self._add_verification_requirements(response)

        return response

    def _apply_context_patterns(self, query: str, response: str, patterns: List[Dict]) -> str:
        """Apply contextual patterns (like project status awareness)"""
        for pattern in patterns:
            context = pattern.get('business_logic', '')
            trigger = pattern.get('trigger', '').lower()

            if trigger in query.lower() and 'status' in context.lower():
                # Add contextual status information
                response = self._inject_context(response, context)

        return response

    def _preference_already_mentioned(self, response: str, preference: str) -> bool:
        """Check if preference is already mentioned in response"""
        key_terms = preference.lower().split()[:2]  # Check first two words
        return any(term in response.lower() for term in key_terms if len(term) > 3)

    def _inject_preference(self, response: str, preference: str) -> str:
        """Inject preference information into response"""
        # If response is a question, inject before the question
        if response.strip().endswith('?'):
            parts = response.rsplit('?', 1)
            return f"{preference}. {parts[0]}?{parts[1] if len(parts) > 1 else ''}"
        else:
            return f"{preference}. {response}"

    def _add_proof_requirement(self, response: str) -> str:
        """Add proof/testing requirements to response"""
        if 'test' not in response.lower():
            return response.replace('I recommend', 'I recommend testing').replace('suggest', 'suggest verifying')
        return response

    def _add_verification_language(self, response: str) -> str:
        """Add verification language to response"""
        verification_phrases = [
            'Please verify this works',
            'Test this before using',
            'Confirm this meets your needs'
        ]

        for phrase in verification_phrases:
            if phrase.lower() not in response.lower():
                if response.strip().endswith('.'):
                    return f"{response[:-1]} - {phrase.lower()}."
                else:
                    return f"{response} - {phrase.lower()}"

        return response

    def _remove_promotional_terms(self, response: str) -> str:
        """Remove promotional terms from response"""
        promotional_terms = [
            'enhanced', 'improved', 'optimized', 'revolutionary',
            'cutting-edge', 'state-of-the-art', 'next-generation',
            'advanced', 'sophisticated', 'premium', 'ultimate',
            'powerful', 'robust', 'comprehensive', 'unified'
        ]

        for term in promotional_terms:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            response = pattern.sub('', response)

            # Clean up extra spaces
            response = re.sub(r'\s+', ' ', response).strip()

        return response

    def _add_verification_requirements(self, response: str) -> str:
        """Add verification requirements to avoid assumptions"""
        if 'assume' in response.lower():
            response = response.replace('assume', 'verify')

        if not any(word in response.lower() for word in ['test', 'verify', 'check', 'confirm']):
            if response.strip().endswith('.'):
                response = f"{response[:-1]} - please verify this works for your specific use case."
            else:
                response = f"{response} - please verify this works for your specific use case"

        return response

    def _inject_context(self, response: str, context: str) -> str:
        """Inject contextual information into response"""
        # Extract key context information
        context_info = context.split('.')[0]  # First sentence

        if context_info and context_info.lower() not in response.lower():
            if response.strip().endswith('?'):
                parts = response.rsplit('?', 1)
                return f"{context_info}. {parts[0]}?{parts[1] if len(parts) > 1 else ''}"
            else:
                return f"{context_info}. {response}"

        return response

    def get_application_stats(self) -> Dict[str, int]:
        """Get statistics about pattern applications"""
        return {
            **self.application_stats,
            'success_rate': (
                self.application_stats['successful_applications'] /
                max(1, self.application_stats['total_applications'])
            ) * 100
        }