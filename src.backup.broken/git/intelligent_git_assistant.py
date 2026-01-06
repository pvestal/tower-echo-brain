#!/usr/bin/env python3
"""
Intelligent Git Assistant for Echo Brain
Advanced commit message generation, conflict resolution, and code analysis
"""

import asyncio
import logging
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import difflib
import ast
import git
from git import Repo
import yaml

# Import Echo Brain LLM interface
from ..core.echo.llm_interface import LLMInterface

logger = logging.getLogger(__name__)

class ChangeType(Enum):
    """Types of code changes"""
    FEATURE = "feat"
    FIX = "fix"
    DOCS = "docs"
    STYLE = "style"
    REFACTOR = "refactor"
    TEST = "test"
    CHORE = "chore"
    PERF = "perf"
    CI = "ci"
    BUILD = "build"

class ConflictType(Enum):
    """Types of merge conflicts"""
    TEXT_CONFLICT = "text"
    BINARY_CONFLICT = "binary"
    DELETE_MODIFY = "delete_modify"
    RENAME_CONFLICT = "rename"
    SYMLINK_CONFLICT = "symlink"

class Priority(Enum):
    """Priority levels for conflicts"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class CodeChange:
    """Represents a code change"""
    file_path: str
    change_type: ChangeType
    lines_added: int
    lines_removed: int
    function_changes: List[str]
    class_changes: List[str]
    import_changes: List[str]
    description: str
    impact_score: float

@dataclass
class CommitAnalysis:
    """Analysis of changes for commit message generation"""
    primary_change_type: ChangeType
    scope: Optional[str]
    breaking_change: bool
    changes: List[CodeChange]
    affected_components: Set[str]
    suggested_message: str
    detailed_description: str

@dataclass
class ConflictInfo:
    """Information about a merge conflict"""
    file_path: str
    conflict_type: ConflictType
    our_content: str
    their_content: str
    ancestor_content: Optional[str]
    priority: Priority
    auto_resolvable: bool
    suggested_resolution: Optional[str]
    resolution_confidence: float

@dataclass
class ConflictResolution:
    """Result of conflict resolution"""
    file_path: str
    resolution_method: str
    resolved_content: str
    manual_review_required: bool
    explanation: str

class IntelligentGitAssistant:
    """
    Advanced git assistant with AI-powered commit message generation and conflict resolution.

    Features:
    - Semantic commit message generation
    - Intelligent conflict detection and resolution
    - Code change analysis and impact assessment
    - Context-aware suggestions
    - Breaking change detection
    """

    def __init__(self):
        self.llm: Optional[LLMInterface] = None

        # Configuration
        self.commit_config = {
            'max_subject_length': 72,
            'max_body_line_length': 100,
            'include_scope': True,
            'include_breaking_change_footer': True,
            'use_conventional_commits': True,
            'analyze_code_semantics': True
        }

        # Patterns for detecting change types
        self.change_patterns = {
            ChangeType.FEATURE: [
                r'(add|new|create|implement|introduce)',
                r'(feature|functionality|capability)',
                r'class\s+\w+.*:',  # New class definitions
                r'def\s+\w+.*:',    # New function definitions
            ],
            ChangeType.FIX: [
                r'(fix|bug|error|issue|problem)',
                r'(resolve|correct|repair)',
                r'(patch|hotfix)',
            ],
            ChangeType.DOCS: [
                r'\.md$',
                r'\.rst$',
                r'\.txt$',
                r'readme',
                r'(doc|documentation)',
            ],
            ChangeType.TEST: [
                r'test_',
                r'_test\.py',
                r'tests/',
                r'(test|spec|mock)',
            ],
            ChangeType.REFACTOR: [
                r'(refactor|restructure|reorganize)',
                r'(cleanup|clean up)',
                r'(rename|move)',
            ],
            ChangeType.STYLE: [
                r'(format|formatting)',
                r'(style|styling)',
                r'(lint|linting)',
                r'(whitespace|indentation)',
            ],
            ChangeType.CHORE: [
                r'(chore|maintenance)',
                r'(update|upgrade)',
                r'(dependency|dependencies)',
                r'package\.json',
                r'requirements\.txt',
            ]
        }

        # File patterns for scope detection
        self.scope_patterns = {
            'api': [r'api/', r'endpoints/', r'routes/'],
            'auth': [r'auth/', r'authentication/', r'login'],
            'database': [r'db/', r'database/', r'models/', r'schema'],
            'frontend': [r'frontend/', r'ui/', r'components/', r'views/'],
            'backend': [r'backend/', r'server/', r'core/'],
            'config': [r'config/', r'settings/', r'\.env'],
            'docs': [r'docs/', r'documentation/', r'\.md$'],
            'tests': [r'test/', r'tests/', r'spec/'],
            'ci': [r'\.github/', r'\.gitlab-ci', r'Jenkinsfile'],
            'docker': [r'Dockerfile', r'docker-compose', r'\.dockerignore'],
        }

    async def initialize(self) -> bool:
        """Initialize the intelligent git assistant"""
        try:
            logger.info("Initializing Intelligent Git Assistant...")

            # Initialize LLM interface
            self.llm = LLMInterface()
            await self.llm.initialize()

            logger.info("Intelligent Git Assistant initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Intelligent Git Assistant: {e}")
            return False

    async def analyze_changes_for_commit(
        self,
        repo_path: Path,
        staged_files: Optional[List[str]] = None
    ) -> CommitAnalysis:
        """Analyze staged changes and generate commit message suggestions"""
        try:
            repo = Repo(repo_path)

            # Get staged changes
            if staged_files:
                # Analyze specific files
                diffs = []
                for file_path in staged_files:
                    try:
                        diff = repo.git.diff('--cached', '--', file_path)
                        if diff:
                            diffs.append((file_path, diff))
                    except Exception as e:
                        logger.warning(f"Failed to get diff for {file_path}: {e}")
            else:
                # Get all staged changes
                diff_index = repo.index.diff("HEAD")
                diffs = []
                for diff_item in diff_index:
                    file_path = diff_item.a_path or diff_item.b_path
                    try:
                        diff_content = repo.git.diff('--cached', '--', file_path)
                        if diff_content:
                            diffs.append((file_path, diff_content))
                    except Exception as e:
                        logger.warning(f"Failed to get diff for {file_path}: {e}")

            if not diffs:
                return CommitAnalysis(
                    primary_change_type=ChangeType.CHORE,
                    scope=None,
                    breaking_change=False,
                    changes=[],
                    affected_components=set(),
                    suggested_message="chore: no changes detected",
                    detailed_description="No staged changes found."
                )

            # Analyze each changed file
            changes = []
            affected_components = set()

            for file_path, diff_content in diffs:
                change = await self._analyze_file_change(file_path, diff_content)
                changes.append(change)

                # Detect affected components
                component = self._detect_component_scope(file_path)
                if component:
                    affected_components.add(component)

            # Determine primary change type
            primary_change_type = self._determine_primary_change_type(changes)

            # Detect scope
            scope = self._determine_scope(changes, affected_components)

            # Check for breaking changes
            breaking_change = self._detect_breaking_changes(changes)

            # Generate commit message
            suggested_message = await self._generate_commit_message(
                primary_change_type,
                scope,
                breaking_change,
                changes
            )

            # Generate detailed description
            detailed_description = await self._generate_detailed_description(changes)

            return CommitAnalysis(
                primary_change_type=primary_change_type,
                scope=scope,
                breaking_change=breaking_change,
                changes=changes,
                affected_components=affected_components,
                suggested_message=suggested_message,
                detailed_description=detailed_description
            )

        except Exception as e:
            logger.error(f"Failed to analyze changes: {e}")
            return CommitAnalysis(
                primary_change_type=ChangeType.CHORE,
                scope=None,
                breaking_change=False,
                changes=[],
                affected_components=set(),
                suggested_message=f"chore: update files",
                detailed_description=f"Error analyzing changes: {e}"
            )

    async def _analyze_file_change(self, file_path: str, diff_content: str) -> CodeChange:
        """Analyze changes in a specific file"""
        try:
            # Count added/removed lines
            lines_added = len([line for line in diff_content.split('\n') if line.startswith('+')])
            lines_removed = len([line for line in diff_content.split('\n') if line.startswith('-')])

            # Detect change type based on file pattern and content
            change_type = self._detect_change_type(file_path, diff_content)

            # Analyze code structure changes (for Python files)
            function_changes = []
            class_changes = []
            import_changes = []

            if file_path.endswith('.py'):
                function_changes, class_changes, import_changes = self._analyze_python_changes(diff_content)

            # Generate description
            description = self._generate_file_change_description(
                file_path, lines_added, lines_removed, function_changes, class_changes
            )

            # Calculate impact score (0.0 to 1.0)
            impact_score = self._calculate_impact_score(
                lines_added, lines_removed, function_changes, class_changes, file_path
            )

            return CodeChange(
                file_path=file_path,
                change_type=change_type,
                lines_added=lines_added,
                lines_removed=lines_removed,
                function_changes=function_changes,
                class_changes=class_changes,
                import_changes=import_changes,
                description=description,
                impact_score=impact_score
            )

        except Exception as e:
            logger.warning(f"Error analyzing file {file_path}: {e}")
            return CodeChange(
                file_path=file_path,
                change_type=ChangeType.CHORE,
                lines_added=0,
                lines_removed=0,
                function_changes=[],
                class_changes=[],
                import_changes=[],
                description=f"Update {file_path}",
                impact_score=0.1
            )

    def _detect_change_type(self, file_path: str, diff_content: str) -> ChangeType:
        """Detect the type of change based on file path and diff content"""
        # Check file patterns first
        for change_type, patterns in self.change_patterns.items():
            for pattern in patterns:
                if re.search(pattern, file_path, re.IGNORECASE):
                    return change_type

        # Check diff content patterns
        diff_lower = diff_content.lower()
        for change_type, patterns in self.change_patterns.items():
            for pattern in patterns:
                if re.search(pattern, diff_lower):
                    return change_type

        # Default to chore if no pattern matches
        return ChangeType.CHORE

    def _analyze_python_changes(self, diff_content: str) -> Tuple[List[str], List[str], List[str]]:
        """Analyze changes in Python code to detect function and class changes"""
        function_changes = []
        class_changes = []
        import_changes = []

        try:
            for line in diff_content.split('\n'):
                if line.startswith('+') or line.startswith('-'):
                    line_content = line[1:].strip()

                    # Detect function definitions
                    func_match = re.match(r'def\s+(\w+)', line_content)
                    if func_match:
                        func_name = func_match.group(1)
                        if func_name not in function_changes:
                            function_changes.append(func_name)

                    # Detect class definitions
                    class_match = re.match(r'class\s+(\w+)', line_content)
                    if class_match:
                        class_name = class_match.group(1)
                        if class_name not in class_changes:
                            class_changes.append(class_name)

                    # Detect import changes
                    import_match = re.match(r'(?:from\s+\S+\s+)?import\s+(.+)', line_content)
                    if import_match:
                        import_info = import_match.group(1)
                        if import_info not in import_changes:
                            import_changes.append(import_info)

        except Exception as e:
            logger.warning(f"Error analyzing Python changes: {e}")

        return function_changes, class_changes, import_changes

    def _generate_file_change_description(
        self,
        file_path: str,
        lines_added: int,
        lines_removed: int,
        function_changes: List[str],
        class_changes: List[str]
    ) -> str:
        """Generate a description of file changes"""
        parts = []

        if function_changes:
            if len(function_changes) == 1:
                parts.append(f"modify {function_changes[0]}() function")
            else:
                parts.append(f"modify {len(function_changes)} functions")

        if class_changes:
            if len(class_changes) == 1:
                parts.append(f"modify {class_changes[0]} class")
            else:
                parts.append(f"modify {len(class_changes)} classes")

        if not parts:
            if lines_added > 0 and lines_removed > 0:
                parts.append(f"update content (+{lines_added}, -{lines_removed} lines)")
            elif lines_added > 0:
                parts.append(f"add content (+{lines_added} lines)")
            elif lines_removed > 0:
                parts.append(f"remove content (-{lines_removed} lines)")
            else:
                parts.append("modify file")

        description = ", ".join(parts)
        file_name = Path(file_path).name

        return f"{file_name}: {description}"

    def _calculate_impact_score(
        self,
        lines_added: int,
        lines_removed: int,
        function_changes: List[str],
        class_changes: List[str],
        file_path: str
    ) -> float:
        """Calculate impact score for a change (0.0 to 1.0)"""
        score = 0.0

        # Base score from line changes
        total_lines = lines_added + lines_removed
        line_score = min(total_lines / 100.0, 0.5)  # Max 0.5 for line changes
        score += line_score

        # Function changes increase impact
        if function_changes:
            score += min(len(function_changes) * 0.1, 0.3)

        # Class changes have higher impact
        if class_changes:
            score += min(len(class_changes) * 0.2, 0.4)

        # Critical files have higher impact
        if any(pattern in file_path for pattern in ['__init__.py', 'main.py', 'app.py', 'server.py']):
            score += 0.2

        return min(score, 1.0)

    def _determine_primary_change_type(self, changes: List[CodeChange]) -> ChangeType:
        """Determine the primary change type from all changes"""
        if not changes:
            return ChangeType.CHORE

        # Count change types by impact score
        type_scores = {}
        for change in changes:
            change_type = change.change_type
            if change_type not in type_scores:
                type_scores[change_type] = 0.0
            type_scores[change_type] += change.impact_score

        # Return the type with highest cumulative impact
        return max(type_scores.items(), key=lambda x: x[1])[0]

    def _detect_component_scope(self, file_path: str) -> Optional[str]:
        """Detect the component/scope based on file path"""
        for scope, patterns in self.scope_patterns.items():
            for pattern in patterns:
                if re.search(pattern, file_path):
                    return scope
        return None

    def _determine_scope(self, changes: List[CodeChange], affected_components: Set[str]) -> Optional[str]:
        """Determine the scope for the commit"""
        if not self.commit_config['include_scope']:
            return None

        if len(affected_components) == 1:
            return list(affected_components)[0]
        elif len(affected_components) > 1:
            # Multiple components - use most common or most impactful
            return "multiple"
        else:
            # No clear scope detected
            return None

    def _detect_breaking_changes(self, changes: List[CodeChange]) -> bool:
        """Detect if changes include breaking changes"""
        breaking_indicators = [
            'BREAKING CHANGE',
            'breaking change',
            'remove ',
            'delete ',
            'deprecated',
        ]

        for change in changes:
            description_lower = change.description.lower()
            if any(indicator in description_lower for indicator in breaking_indicators):
                return True

            # Check for public API changes in function/class names
            for func in change.function_changes:
                if not func.startswith('_'):  # Public function
                    # This is simplistic - in practice you'd analyze AST changes
                    return False

        return False

    async def _generate_commit_message(
        self,
        primary_change_type: ChangeType,
        scope: Optional[str],
        breaking_change: bool,
        changes: List[CodeChange]
    ) -> str:
        """Generate a commit message using AI and conventional commit format"""
        try:
            if self.llm:
                # Use AI to generate intelligent commit message
                return await self._generate_ai_commit_message(
                    primary_change_type, scope, breaking_change, changes
                )
            else:
                # Fallback to template-based generation
                return self._generate_template_commit_message(
                    primary_change_type, scope, breaking_change, changes
                )

        except Exception as e:
            logger.warning(f"Failed to generate AI commit message: {e}")
            return self._generate_template_commit_message(
                primary_change_type, scope, breaking_change, changes
            )

    async def _generate_ai_commit_message(
        self,
        primary_change_type: ChangeType,
        scope: Optional[str],
        breaking_change: bool,
        changes: List[CodeChange]
    ) -> str:
        """Generate commit message using AI"""
        try:
            # Prepare context for AI
            changes_summary = []
            for change in changes[:5]:  # Limit to top 5 changes
                changes_summary.append({
                    'file': change.file_path,
                    'type': change.change_type.value,
                    'description': change.description,
                    'functions': change.function_changes[:3],
                    'classes': change.class_changes[:3]
                })

            prompt = f"""
Generate a concise git commit message following conventional commit format:
type(scope): description

Context:
- Primary change type: {primary_change_type.value}
- Scope: {scope or 'none'}
- Breaking change: {breaking_change}
- Files changed: {len(changes)}

Changes:
{json.dumps(changes_summary, indent=2)}

Requirements:
- Use conventional commit format: type(scope): description
- Keep subject line under {self.commit_config['max_subject_length']} characters
- Be specific and clear about what was changed
- Focus on the "what" and "why", not the "how"
- If breaking change, start description with "!"

Examples:
- feat(auth): add JWT token validation
- fix(api): resolve user creation error
- docs(readme): update installation instructions
- refactor(database): extract query helpers

Generate only the commit message, no explanation:
"""

            response = await self.llm.query(prompt, max_tokens=100)
            if response and len(response.strip()) > 0:
                # Clean up the response
                message = response.strip().split('\n')[0]
                if len(message) <= self.commit_config['max_subject_length']:
                    return message

            # If AI response is not suitable, fall back to template
            return self._generate_template_commit_message(
                primary_change_type, scope, breaking_change, changes
            )

        except Exception as e:
            logger.warning(f"AI commit message generation failed: {e}")
            return self._generate_template_commit_message(
                primary_change_type, scope, breaking_change, changes
            )

    def _generate_template_commit_message(
        self,
        primary_change_type: ChangeType,
        scope: Optional[str],
        breaking_change: bool,
        changes: List[CodeChange]
    ) -> str:
        """Generate commit message using templates"""
        # Build type and scope
        type_str = primary_change_type.value
        scope_str = f"({scope})" if scope else ""
        breaking_str = "!" if breaking_change else ""

        # Generate description
        if len(changes) == 1:
            change = changes[0]
            if change.function_changes:
                desc = f"update {change.function_changes[0]}() function"
            elif change.class_changes:
                desc = f"update {change.class_changes[0]} class"
            else:
                file_name = Path(change.file_path).stem
                desc = f"update {file_name}"
        else:
            # Multiple files
            if len(changes) <= 3:
                file_names = [Path(c.file_path).stem for c in changes]
                desc = f"update {', '.join(file_names)}"
            else:
                desc = f"update {len(changes)} files"

        # Construct message
        message = f"{type_str}{scope_str}{breaking_str}: {desc}"

        # Truncate if too long
        if len(message) > self.commit_config['max_subject_length']:
            # Try shorter description
            if len(changes) == 1:
                desc = f"update {Path(changes[0].file_path).stem}"
            else:
                desc = f"update {len(changes)} files"
            message = f"{type_str}{scope_str}{breaking_str}: {desc}"

        return message

    async def _generate_detailed_description(self, changes: List[CodeChange]) -> str:
        """Generate detailed description for commit body"""
        if not changes:
            return "No changes detected."

        lines = []

        # Summary
        lines.append(f"Changes in {len(changes)} file(s):")
        lines.append("")

        # List changes by file
        for change in changes:
            file_name = Path(change.file_path).name
            lines.append(f"- {file_name}: {change.description}")

            if change.function_changes or change.class_changes:
                details = []
                if change.function_changes:
                    details.append(f"functions: {', '.join(change.function_changes[:3])}")
                if change.class_changes:
                    details.append(f"classes: {', '.join(change.class_changes[:3])}")
                lines.append(f"  ({'; '.join(details)})")

        # Statistics
        total_added = sum(c.lines_added for c in changes)
        total_removed = sum(c.lines_removed for c in changes)
        if total_added or total_removed:
            lines.append("")
            lines.append(f"Total: +{total_added}/-{total_removed} lines")

        return '\n'.join(lines)

    async def detect_conflicts(self, repo_path: Path) -> List[ConflictInfo]:
        """Detect and analyze merge conflicts"""
        try:
            repo = Repo(repo_path)

            # Get conflicted files
            conflicted_files = []

            # Check for unmerged paths
            unmerged = repo.git.diff('--name-only', '--diff-filter=U').split('\n')
            for file_path in unmerged:
                if file_path.strip():
                    conflicted_files.append(file_path.strip())

            if not conflicted_files:
                return []

            conflicts = []
            for file_path in conflicted_files:
                conflict_info = await self._analyze_conflict(repo, file_path)
                if conflict_info:
                    conflicts.append(conflict_info)

            return conflicts

        except Exception as e:
            logger.error(f"Failed to detect conflicts: {e}")
            return []

    async def _analyze_conflict(self, repo: Repo, file_path: str) -> Optional[ConflictInfo]:
        """Analyze a specific conflict"""
        try:
            file_full_path = Path(repo.working_dir) / file_path

            if not file_full_path.exists():
                return ConflictInfo(
                    file_path=file_path,
                    conflict_type=ConflictType.DELETE_MODIFY,
                    our_content="",
                    their_content="",
                    ancestor_content=None,
                    priority=Priority.HIGH,
                    auto_resolvable=False,
                    suggested_resolution=None,
                    resolution_confidence=0.0
                )

            # Read file content to analyze conflict markers
            try:
                content = file_full_path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                # Binary file conflict
                return ConflictInfo(
                    file_path=file_path,
                    conflict_type=ConflictType.BINARY_CONFLICT,
                    our_content="[BINARY]",
                    their_content="[BINARY]",
                    ancestor_content=None,
                    priority=Priority.MEDIUM,
                    auto_resolvable=False,
                    suggested_resolution="Manual resolution required for binary file",
                    resolution_confidence=0.0
                )

            # Parse conflict markers
            our_content, their_content, ancestor_content = self._parse_conflict_markers(content)

            if not our_content and not their_content:
                # No conflict markers found
                return None

            # Determine conflict type and priority
            conflict_type = ConflictType.TEXT_CONFLICT
            priority = self._determine_conflict_priority(file_path, our_content, their_content)

            # Check if auto-resolvable
            auto_resolvable, suggested_resolution, confidence = await self._analyze_conflict_resolution(
                file_path, our_content, their_content, ancestor_content
            )

            return ConflictInfo(
                file_path=file_path,
                conflict_type=conflict_type,
                our_content=our_content,
                their_content=their_content,
                ancestor_content=ancestor_content,
                priority=priority,
                auto_resolvable=auto_resolvable,
                suggested_resolution=suggested_resolution,
                resolution_confidence=confidence
            )

        except Exception as e:
            logger.error(f"Failed to analyze conflict in {file_path}: {e}")
            return None

    def _parse_conflict_markers(self, content: str) -> Tuple[str, str, Optional[str]]:
        """Parse conflict markers from file content"""
        our_content = ""
        their_content = ""
        ancestor_content = None

        lines = content.split('\n')
        current_section = None

        for line in lines:
            if line.startswith('<<<<<<< '):
                current_section = 'ours'
                continue
            elif line.startswith('||||||| '):
                current_section = 'ancestor'
                continue
            elif line.startswith('======='):
                current_section = 'theirs'
                continue
            elif line.startswith('>>>>>>> '):
                current_section = None
                continue

            if current_section == 'ours':
                our_content += line + '\n'
            elif current_section == 'theirs':
                their_content += line + '\n'
            elif current_section == 'ancestor':
                if ancestor_content is None:
                    ancestor_content = ""
                ancestor_content += line + '\n'

        return our_content.strip(), their_content.strip(), ancestor_content

    def _determine_conflict_priority(self, file_path: str, our_content: str, their_content: str) -> Priority:
        """Determine the priority of a conflict based on file importance and change complexity"""
        # Critical files get high priority
        critical_patterns = [
            '__init__.py',
            'main.py',
            'app.py',
            'server.py',
            'config',
            'settings',
        ]

        for pattern in critical_patterns:
            if pattern in file_path:
                return Priority.CRITICAL

        # Large differences get higher priority
        our_lines = len(our_content.split('\n'))
        their_lines = len(their_content.split('\n'))
        total_lines = our_lines + their_lines

        if total_lines > 50:
            return Priority.HIGH
        elif total_lines > 20:
            return Priority.MEDIUM
        else:
            return Priority.LOW

    async def _analyze_conflict_resolution(
        self,
        file_path: str,
        our_content: str,
        their_content: str,
        ancestor_content: Optional[str]
    ) -> Tuple[bool, Optional[str], float]:
        """Analyze if conflict can be automatically resolved"""
        try:
            # Simple heuristics for auto-resolution

            # Case 1: One side is empty (deletion vs modification)
            if not our_content.strip() and their_content.strip():
                return True, their_content, 0.8
            if not their_content.strip() and our_content.strip():
                return True, our_content, 0.8

            # Case 2: Changes don't overlap (line-by-line)
            our_lines = our_content.split('\n')
            their_lines = their_content.split('\n')

            if ancestor_content:
                ancestor_lines = ancestor_content.split('\n')
                # Check if changes are in different parts of the file
                our_changes = set(range(len(our_lines))) - set(range(len(ancestor_lines)))
                their_changes = set(range(len(their_lines))) - set(range(len(ancestor_lines)))

                if not our_changes.intersection(their_changes):
                    # Non-overlapping changes - can merge
                    merged_content = self._merge_non_overlapping_changes(
                        ancestor_lines, our_lines, their_lines
                    )
                    return True, merged_content, 0.7

            # Case 3: Identical changes
            if our_content == their_content:
                return True, our_content, 1.0

            # Case 4: Use AI for intelligent resolution
            if self.llm:
                resolution = await self._ai_conflict_resolution(
                    file_path, our_content, their_content, ancestor_content
                )
                if resolution:
                    return True, resolution, 0.6

            # No auto-resolution possible
            return False, None, 0.0

        except Exception as e:
            logger.warning(f"Error analyzing conflict resolution: {e}")
            return False, None, 0.0

    def _merge_non_overlapping_changes(
        self,
        ancestor_lines: List[str],
        our_lines: List[str],
        their_lines: List[str]
    ) -> str:
        """Merge non-overlapping changes"""
        # This is a simplified implementation
        # In practice, you'd use a proper 3-way merge algorithm

        # For now, just concatenate unique lines
        all_lines = ancestor_lines[:]

        # Add lines that are in our_lines but not in ancestor
        for line in our_lines:
            if line not in ancestor_lines and line not in all_lines:
                all_lines.append(line)

        # Add lines that are in their_lines but not in ancestor
        for line in their_lines:
            if line not in ancestor_lines and line not in all_lines:
                all_lines.append(line)

        return '\n'.join(all_lines)

    async def _ai_conflict_resolution(
        self,
        file_path: str,
        our_content: str,
        their_content: str,
        ancestor_content: Optional[str]
    ) -> Optional[str]:
        """Use AI to suggest conflict resolution"""
        try:
            if not self.llm:
                return None

            prompt = f"""
Resolve this git merge conflict by choosing the best combination of changes:

File: {file_path}

Our changes:
```
{our_content}
```

Their changes:
```
{their_content}
```

{f'''
Ancestor (common base):
```
{ancestor_content}
```
''' if ancestor_content else ''}

Provide the resolved content that combines both changes appropriately.
Consider:
- Code functionality and correctness
- Maintaining consistency
- Preserving both sets of improvements where possible

Return only the resolved content, no explanations:
"""

            response = await self.llm.query(prompt, max_tokens=1000)
            if response and len(response.strip()) > 0:
                return response.strip()

            return None

        except Exception as e:
            logger.warning(f"AI conflict resolution failed: {e}")
            return None

    async def resolve_conflicts(
        self,
        repo_path: Path,
        conflicts: List[ConflictInfo],
        auto_resolve: bool = True
    ) -> List[ConflictResolution]:
        """Resolve detected conflicts"""
        resolutions = []

        for conflict in conflicts:
            try:
                if auto_resolve and conflict.auto_resolvable and conflict.suggested_resolution:
                    # Apply automatic resolution
                    resolution = await self._apply_conflict_resolution(
                        repo_path,
                        conflict.file_path,
                        conflict.suggested_resolution,
                        automatic=True
                    )
                    resolutions.append(resolution)
                else:
                    # Mark for manual resolution
                    resolution = ConflictResolution(
                        file_path=conflict.file_path,
                        resolution_method="manual",
                        resolved_content="",
                        manual_review_required=True,
                        explanation=f"Manual resolution required for {conflict.conflict_type.value} conflict"
                    )
                    resolutions.append(resolution)

            except Exception as e:
                logger.error(f"Failed to resolve conflict in {conflict.file_path}: {e}")
                resolution = ConflictResolution(
                    file_path=conflict.file_path,
                    resolution_method="failed",
                    resolved_content="",
                    manual_review_required=True,
                    explanation=f"Resolution failed: {e}"
                )
                resolutions.append(resolution)

        return resolutions

    async def _apply_conflict_resolution(
        self,
        repo_path: Path,
        file_path: str,
        resolved_content: str,
        automatic: bool = True
    ) -> ConflictResolution:
        """Apply conflict resolution to file"""
        try:
            file_full_path = Path(repo_path) / file_path

            # Write resolved content
            file_full_path.write_text(resolved_content, encoding='utf-8')

            # Stage the resolved file
            repo = Repo(repo_path)
            repo.index.add([file_path])

            method = "automatic" if automatic else "manual"
            explanation = f"Conflict resolved {'automatically' if automatic else 'manually'}"

            return ConflictResolution(
                file_path=file_path,
                resolution_method=method,
                resolved_content=resolved_content,
                manual_review_required=not automatic,
                explanation=explanation
            )

        except Exception as e:
            logger.error(f"Failed to apply resolution to {file_path}: {e}")
            return ConflictResolution(
                file_path=file_path,
                resolution_method="failed",
                resolved_content="",
                manual_review_required=True,
                explanation=f"Failed to apply resolution: {e}"
            )

    async def get_commit_suggestions(self, repo_path: Path) -> Dict[str, Any]:
        """Get comprehensive commit suggestions"""
        try:
            # Analyze staged changes
            analysis = await self.analyze_changes_for_commit(repo_path)

            # Check for conflicts
            conflicts = await self.detect_conflicts(repo_path)

            return {
                'commit_analysis': asdict(analysis),
                'conflicts': [asdict(c) for c in conflicts],
                'ready_to_commit': len(conflicts) == 0,
                'recommendations': self._generate_commit_recommendations(analysis, conflicts)
            }

        except Exception as e:
            logger.error(f"Failed to get commit suggestions: {e}")
            return {
                'error': str(e),
                'ready_to_commit': False,
                'recommendations': ["Fix errors before committing"]
            }

    def _generate_commit_recommendations(
        self,
        analysis: CommitAnalysis,
        conflicts: List[ConflictInfo]
    ) -> List[str]:
        """Generate recommendations for the commit"""
        recommendations = []

        if conflicts:
            recommendations.append(f"Resolve {len(conflicts)} merge conflicts before committing")
            return recommendations

        if analysis.breaking_change:
            recommendations.append("This includes breaking changes - update version and changelog")

        if len(analysis.changes) > 10:
            recommendations.append("Consider splitting this into smaller, focused commits")

        high_impact_changes = [c for c in analysis.changes if c.impact_score > 0.7]
        if high_impact_changes:
            recommendations.append(f"High-impact changes detected in {len(high_impact_changes)} files")

        if not analysis.scope:
            recommendations.append("Consider adding a scope to make the commit message clearer")

        return recommendations


# Global instance
intelligent_git_assistant = IntelligentGitAssistant()

async def test_intelligent_git_assistant():
    """Test the intelligent git assistant"""
    assistant = intelligent_git_assistant

    # Initialize
    success = await assistant.initialize()
    if not success:
        print("❌ Failed to initialize intelligent git assistant")
        return

    print("✅ Intelligent git assistant initialized")

    # Test with current repository
    repo_path = Path("/opt/tower-echo-brain")

    try:
        # Get commit suggestions
        suggestions = await assistant.get_commit_suggestions(repo_path)

        print(f"\n📊 Commit Analysis:")
        if 'commit_analysis' in suggestions:
            analysis = suggestions['commit_analysis']
            print(f"  Primary type: {analysis['primary_change_type']}")
            print(f"  Scope: {analysis['scope'] or 'none'}")
            print(f"  Breaking change: {analysis['breaking_change']}")
            print(f"  Files changed: {len(analysis['changes'])}")
            print(f"  Suggested message: {analysis['suggested_message']}")

        print(f"\n🔄 Status:")
        print(f"  Ready to commit: {suggestions['ready_to_commit']}")

        if suggestions.get('recommendations'):
            print(f"  Recommendations:")
            for rec in suggestions['recommendations']:
                print(f"    - {rec}")

    except Exception as e:
        print(f"❌ Error testing assistant: {e}")

    print("\n✅ Intelligent git assistant test complete")


if __name__ == "__main__":
    asyncio.run(test_intelligent_git_assistant())