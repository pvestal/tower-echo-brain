#!/usr/bin/env python3
"""
Code Quality Validator for Echo Brain
Detects gibberish output and triggers model reloading for code generation tasks
"""

import re
import ast
import json
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime
import subprocess
import tempfile

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of code validation"""
    is_valid: bool
    confidence: float
    issues: List[str]
    language: Optional[str]
    quality_score: float
    is_gibberish: bool
    requires_reload: bool

class CodeQualityValidator:
    """
    Validates code output quality and detects gibberish responses
    """

    def __init__(self):
        self.gibberish_patterns = [
            # Repeated characters
            re.compile(r'(.)\1{10,}'),
            # Random unicode characters
            re.compile(r'[\u0080-\u009f\u2000-\u206f\ufff0-\uffff]{5,}'),
            # Excessive special characters with no structure
            re.compile(r'[!@#$%^&*()_+=\[\]{}|\\;:",.<>?/`~]{20,}'),
            # Base64-like gibberish
            re.compile(r'^[A-Za-z0-9+/]{100,}={0,2}$'),
            # Binary gibberish
            re.compile(r'^[01\s]{100,}$'),
            # Hex gibberish
            re.compile(r'^[0-9a-fA-F\s]{100,}$')
        ]

        self.code_indicators = {
            'python': {
                'keywords': ['def', 'class', 'import', 'from', 'if', 'else', 'for', 'while', 'return', 'async', 'await'],
                'patterns': [r'def\s+\w+\(', r'class\s+\w+', r'import\s+\w+', r'^\s{4}', r'#.*\n'],
                'extension': '.py'
            },
            'javascript': {
                'keywords': ['function', 'const', 'let', 'var', 'return', 'if', 'else', 'for', 'while', 'async', 'await'],
                'patterns': [r'function\s+\w+\(', r'const\s+\w+\s*=', r'=>\s*{', r'console\.log'],
                'extension': '.js'
            },
            'typescript': {
                'keywords': ['interface', 'type', 'enum', 'namespace', 'function', 'const', 'let'],
                'patterns': [r'interface\s+\w+', r'type\s+\w+\s*=', r':\s*(string|number|boolean)'],
                'extension': '.ts'
            },
            'java': {
                'keywords': ['public', 'private', 'class', 'interface', 'void', 'static', 'final'],
                'patterns': [r'public\s+class', r'public\s+static\s+void\s+main'],
                'extension': '.java'
            },
            'sql': {
                'keywords': ['SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'TABLE'],
                'patterns': [r'SELECT\s+.*\s+FROM', r'WHERE\s+\w+\s*=', r'CREATE\s+TABLE'],
                'extension': '.sql'
            },
            'html': {
                'keywords': ['<html>', '<body>', '<div>', '<span>', '<script>', '<style>'],
                'patterns': [r'<\w+[^>]*>', r'</\w+>', r'<!DOCTYPE'],
                'extension': '.html'
            },
            'css': {
                'keywords': ['display', 'position', 'margin', 'padding', 'color', 'background'],
                'patterns': [r'\.\w+\s*{', r'#\w+\s*{', r':\s*\d+px', r'@media'],
                'extension': '.css'
            },
            'bash': {
                'keywords': ['#!/bin/bash', 'echo', 'if', 'then', 'else', 'fi', 'for', 'do', 'done'],
                'patterns': [r'^\#!/bin/(bash|sh)', r'\$\w+', r'if\s+\[', r'&&', r'\|\|'],
                'extension': '.sh'
            }
        }

        self.min_valid_lines = 3
        self.min_keyword_ratio = 0.05  # At least 5% of words should be keywords
        self.max_gibberish_ratio = 0.2  # Max 20% can match gibberish patterns

    def validate_code(self, content: str, expected_language: Optional[str] = None) -> ValidationResult:
        """
        Validate if content is legitimate code or gibberish
        """
        if not content or len(content.strip()) < 10:
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                issues=["Content too short"],
                language=None,
                quality_score=0.0,
                is_gibberish=True,
                requires_reload=True
            )

        # Check for obvious gibberish patterns
        gibberish_score = self._calculate_gibberish_score(content)
        if gibberish_score > self.max_gibberish_ratio:
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                issues=["High gibberish pattern match"],
                language=None,
                quality_score=0.0,
                is_gibberish=True,
                requires_reload=True
            )

        # Detect language and validate structure
        detected_language = self._detect_language(content, expected_language)

        if not detected_language:
            # No language detected but check if it looks structured
            structure_score = self._analyze_structure(content)
            if structure_score < 0.3:
                return ValidationResult(
                    is_valid=False,
                    confidence=structure_score,
                    issues=["No programming language detected", "Poor structure"],
                    language=None,
                    quality_score=structure_score,
                    is_gibberish=True,
                    requires_reload=True
                )

        # Validate syntax for detected language
        syntax_valid, syntax_issues = self._validate_syntax(content, detected_language)

        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            content,
            detected_language,
            syntax_valid,
            gibberish_score
        )

        # Determine if reload is needed
        requires_reload = quality_score < 0.3 or gibberish_score > 0.1

        return ValidationResult(
            is_valid=syntax_valid and quality_score > 0.5,
            confidence=quality_score,
            issues=syntax_issues,
            language=detected_language,
            quality_score=quality_score,
            is_gibberish=gibberish_score > 0.1,
            requires_reload=requires_reload
        )

    def _calculate_gibberish_score(self, content: str) -> float:
        """Calculate how much content matches gibberish patterns"""
        if not content:
            return 1.0

        total_length = len(content)
        gibberish_length = 0

        for pattern in self.gibberish_patterns:
            matches = pattern.findall(content)
            for match in matches:
                gibberish_length += len(match)

        return min(gibberish_length / total_length, 1.0)

    def _detect_language(self, content: str, expected: Optional[str] = None) -> Optional[str]:
        """Detect programming language from content"""
        if expected and expected.lower() in self.code_indicators:
            # Verify expected language
            indicators = self.code_indicators[expected.lower()]
            if self._check_language_indicators(content, indicators):
                return expected.lower()

        # Try to detect from content
        best_match = None
        best_score = 0

        for language, indicators in self.code_indicators.items():
            score = self._calculate_language_score(content, indicators)
            if score > best_score:
                best_score = score
                best_match = language

        return best_match if best_score > 0.1 else None

    def _check_language_indicators(self, content: str, indicators: Dict) -> bool:
        """Check if content matches language indicators"""
        content_lower = content.lower()

        # Check keywords
        keyword_count = sum(1 for kw in indicators['keywords'] if kw.lower() in content_lower)
        if keyword_count < 2:
            return False

        # Check patterns
        pattern_matches = 0
        for pattern in indicators['patterns']:
            if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
                pattern_matches += 1

        return pattern_matches >= 1

    def _calculate_language_score(self, content: str, indicators: Dict) -> float:
        """Calculate confidence score for a language"""
        content_lower = content.lower()
        words = re.findall(r'\b\w+\b', content_lower)

        if not words:
            return 0.0

        # Keyword matching
        keyword_count = sum(1 for word in words if word in [kw.lower() for kw in indicators['keywords']])
        keyword_ratio = keyword_count / len(words)

        # Pattern matching
        pattern_matches = 0
        for pattern in indicators['patterns']:
            if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
                pattern_matches += 1
        pattern_score = pattern_matches / len(indicators['patterns'])

        # Combined score
        return (keyword_ratio * 0.6 + pattern_score * 0.4)

    def _analyze_structure(self, content: str) -> float:
        """Analyze structural quality of content"""
        lines = content.split('\n')

        # Check for consistent indentation
        indentation_pattern = re.compile(r'^(\s*)')
        indentations = [len(indentation_pattern.match(line).group(1))
                       for line in lines if line.strip()]

        if not indentations:
            return 0.0

        # Check for structure indicators
        structure_score = 0.0

        # Has multiple lines
        if len(lines) > self.min_valid_lines:
            structure_score += 0.2

        # Has consistent indentation
        if len(set(indentations)) > 1:  # Multiple indentation levels
            structure_score += 0.2

        # Has brackets/parentheses balance
        open_brackets = content.count('(') + content.count('[') + content.count('{')
        close_brackets = content.count(')') + content.count(']') + content.count('}')
        if abs(open_brackets - close_brackets) < 3:
            structure_score += 0.2

        # Has semicolons or newlines as statement separators
        if ';' in content or len(lines) > 5:
            structure_score += 0.2

        # Has comments
        if '#' in content or '//' in content or '/*' in content:
            structure_score += 0.2

        return min(structure_score, 1.0)

    def _validate_syntax(self, content: str, language: Optional[str]) -> Tuple[bool, List[str]]:
        """Validate syntax for specific language"""
        if not language:
            return False, ["No language detected"]

        issues = []

        if language == 'python':
            try:
                ast.parse(content)
                return True, []
            except SyntaxError as e:
                issues.append(f"Python syntax error: {e}")
                # Check if it might be a snippet
                try:
                    # Try wrapping in a function
                    ast.parse(f"def test():\n" + "\n".join(f"    {line}" for line in content.split('\n')))
                    return True, ["Appears to be a valid Python snippet"]
                except:
                    return False, issues

        elif language in ['javascript', 'typescript']:
            # Basic JS/TS validation
            if content.count('{') != content.count('}'):
                issues.append("Unbalanced braces")
            if content.count('(') != content.count(')'):
                issues.append("Unbalanced parentheses")
            if content.count('[') != content.count(']'):
                issues.append("Unbalanced brackets")

            return len(issues) == 0, issues

        elif language == 'sql':
            # Basic SQL validation
            sql_upper = content.upper()
            if 'SELECT' in sql_upper and 'FROM' not in sql_upper:
                issues.append("SELECT without FROM clause")
            if sql_upper.count('(') != sql_upper.count(')'):
                issues.append("Unbalanced parentheses in SQL")

            return len(issues) == 0, issues

        else:
            # Generic validation for other languages
            # Check for basic structure
            if content.count('{') == content.count('}') and \
               content.count('(') == content.count(')') and \
               content.count('[') == content.count(']'):
                return True, []
            else:
                return False, ["Unbalanced delimiters"]

    def _calculate_quality_score(
        self,
        content: str,
        language: Optional[str],
        syntax_valid: bool,
        gibberish_score: float
    ) -> float:
        """Calculate overall quality score"""
        score = 0.0

        # Base score from syntax validation
        if syntax_valid:
            score += 0.4

        # Language detection confidence
        if language:
            score += 0.2

        # Inverse gibberish score
        score += (1.0 - gibberish_score) * 0.2

        # Structure analysis
        structure_score = self._analyze_structure(content)
        score += structure_score * 0.2

        return min(score, 1.0)


class ModelReloadManager:
    """
    Manages model reloading when output quality is poor
    """

    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.validator = CodeQualityValidator()
        self.reload_history = []
        self.max_retries = 3

    async def validate_and_reload(
        self,
        output: str,
        expected_type: str,
        model_used: str,
        prompt: str
    ) -> Tuple[str, str]:
        """
        Validate output and reload model if necessary
        Returns: (final_output, model_used)
        """

        # Special handling for code generation tasks
        if expected_type == 'code':
            validation = self.validator.validate_code(output)

            if validation.requires_reload:
                logger.warning(
                    f"🔄 Model {model_used} produced invalid code. "
                    f"Quality score: {validation.quality_score:.2f}, "
                    f"Gibberish: {validation.is_gibberish}"
                )

                # Try reloading with a better model
                return await self._retry_with_better_model(
                    prompt,
                    model_used,
                    validation,
                    expected_type
                )

        return output, model_used

    async def _retry_with_better_model(
        self,
        prompt: str,
        failed_model: str,
        validation: ValidationResult,
        expected_type: str,
        retry_count: int = 0
    ) -> Tuple[str, str]:
        """Retry generation with a better model"""

        if retry_count >= self.max_retries:
            logger.error(f"❌ Max retries reached. Using best effort output.")
            return "", failed_model

        # Log the failure
        self.reload_history.append({
            'timestamp': datetime.now().isoformat(),
            'failed_model': failed_model,
            'validation_score': validation.quality_score,
            'issues': validation.issues,
            'retry_count': retry_count
        })

        # Get next model to try
        next_model = await self._select_better_model(failed_model, expected_type)

        if not next_model:
            logger.error(f"❌ No better model available")
            return "", failed_model

        logger.info(f"🔄 Reloading with model: {next_model}")

        # Enhance prompt for better results
        enhanced_prompt = self._enhance_prompt_for_code(prompt, validation.issues)

        try:
            # Use the model manager to generate with new model
            result = await self.model_manager.generate(
                enhanced_prompt,
                model=next_model,
                temperature=0.3,  # Lower temperature for more deterministic output
                max_tokens=2000
            )

            # Validate the new output
            new_validation = self.validator.validate_code(result)

            if new_validation.requires_reload:
                # Recursively retry with even better model
                return await self._retry_with_better_model(
                    prompt,
                    next_model,
                    new_validation,
                    expected_type,
                    retry_count + 1
                )

            logger.success(f"✅ Model {next_model} produced valid code")
            return result, next_model

        except Exception as e:
            logger.error(f"❌ Error with model {next_model}: {e}")
            return await self._retry_with_better_model(
                prompt,
                next_model,
                validation,
                expected_type,
                retry_count + 1
            )

    async def _select_better_model(self, failed_model: str, task_type: str) -> Optional[str]:
        """Select a better model based on failure and task type"""
        hierarchy = {
            'code': [
                'qwen2.5-coder:32b',
                'deepseek-coder-v2:latest',
                'codellama:34b',
                'mistral:latest',
                'llama3.1:70b'
            ],
            'general': [
                'llama3.1:70b',
                'mistral:latest',
                'llama3.2:3b',
                'tinyllama:latest'
            ]
        }


        # Find current model position
        try:
            current_index = hierarchy.index(failed_model)
        except ValueError:
            # Model not in hierarchy, start from beginning
            current_index = len(hierarchy)

        # Get next better model (earlier in list = better)
        for i in range(current_index):
            candidate = hierarchy[i]
            if await self.model_manager.is_model_available(candidate):
                return candidate

        return None

    def _enhance_prompt_for_code(self, original_prompt: str, issues: List[str]) -> str:
        """Enhance prompt to avoid previous issues"""

        enhancement = "\n\nIMPORTANT: Generate valid, syntactically correct code. "

        if "syntax error" in str(issues).lower():
            enhancement += "Ensure proper syntax with balanced brackets and valid statements. "

        if "gibberish" in str(issues).lower():
            enhancement += "Output must be legitimate code, not random characters. "

        if "No language detected" in issues:
            enhancement += "Clearly indicate the programming language and use proper syntax. "

        enhancement += "Output ONLY the code without any explanation or markdown formatting."

        return original_prompt + enhancement

    def get_reload_stats(self) -> Dict:
        """Get statistics about model reloads"""
        if not self.reload_history:
            return {"total_reloads": 0}

        return {
            "total_reloads": len(self.reload_history),
            "models_failed": list(set(h['failed_model'] for h in self.reload_history)),
            "average_validation_score": sum(h['validation_score'] for h in self.reload_history) / len(self.reload_history),
            "common_issues": self._get_common_issues(),
            "recent_reloads": self.reload_history[-5:]
        }

    def _get_common_issues(self) -> List[str]:
        """Get most common validation issues"""
        all_issues = []
        for history in self.reload_history:
            all_issues.extend(history['issues'])

        from collections import Counter
        issue_counts = Counter(all_issues)
        return [issue for issue, _ in issue_counts.most_common(5)]
