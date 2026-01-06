#!/usr/bin/env python3
"""
Auto-documentation system for Echo Brain.
Generates comprehensive documentation from code changes, decisions, and operations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import ast
import re
import asyncio
import httpx

logger = logging.getLogger(__name__)

@dataclass
class CodeChange:
    """Represents a code change for documentation."""
    file_path: str
    change_type: str  # added, modified, deleted
    functions_changed: List[str]
    classes_changed: List[str]
    lines_added: int
    lines_removed: int
    complexity_change: float
    timestamp: datetime

@dataclass
class Decision:
    """Represents a technical decision made."""
    decision_id: str
    title: str
    context: str
    options_considered: List[str]
    choice_made: str
    rationale: str
    impact: str
    timestamp: datetime
    related_files: List[str]

@dataclass
class SystemMetrics:
    """Current system metrics for documentation."""
    total_files: int
    total_lines: int
    test_coverage: float
    documentation_coverage: float
    complexity_score: float
    dependencies: List[str]
    api_endpoints: int
    database_tables: int

class AutoDocumenter:
    """
    Automatically generates and maintains documentation for Echo Brain.

    Features:
    - Code change summaries
    - Decision records (ADRs)
    - API documentation
    - System architecture diagrams (Mermaid)
    - Performance metrics
    - Knowledge base integration
    """

    def __init__(
        self,
        project_root: Path = Path("/opt/tower-echo-brain"),
        kb_url: str = "https://***REMOVED***/api/kb"
    ):
        self.project_root = project_root
        self.docs_dir = project_root / "docs"
        self.docs_dir.mkdir(exist_ok=True)
        self.kb_url = kb_url
        self.http_client = httpx.AsyncClient(verify=False, timeout=30.0)

    async def document_code_changes(
        self,
        since: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """
        Document all code changes since a given time.

        Returns:
            Tuple of (success, documentation_path)
        """
        if not since:
            since = datetime.now() - timedelta(days=7)

        changes = await self._analyze_code_changes(since)

        # Generate documentation
        doc_content = self._generate_change_documentation(changes)

        # Save to file
        doc_path = self.docs_dir / f"CHANGES_{datetime.now().strftime('%Y%m%d')}.md"
        doc_path.write_text(doc_content)

        # Also save to knowledge base
        await self._save_to_kb(
            title=f"Code Changes - {datetime.now().strftime('%Y-%m-%d')}",
            content=doc_content,
            category="code_changes",
            tags=["automated", "changes", "code"]
        )

        return True, str(doc_path)

    async def _analyze_code_changes(self, since: datetime) -> List[CodeChange]:
        """Analyze code changes using git history."""
        changes = []

        # Get git diff for each changed file
        import subprocess
        result = subprocess.run(
            ["git", "-C", str(self.project_root), "diff", "--stat", "--since", since.isoformat()],
            capture_output=True,
            text=True
        )

        if result.stdout:
            for line in result.stdout.splitlines()[:-1]:  # Skip summary line
                if "|" in line:
                    parts = line.split("|")
                    file_path = parts[0].strip()
                    stats = parts[1].strip()

                    # Parse insertions/deletions
                    added = stats.count("+")
                    removed = stats.count("-")

                    # Analyze file for functions/classes
                    full_path = self.project_root / file_path
                    functions, classes = [], []

                    if full_path.suffix == ".py" and full_path.exists():
                        try:
                            functions, classes = self._analyze_python_file(full_path)
                        except:
                            pass

                    changes.append(CodeChange(
                        file_path=file_path,
                        change_type="modified",
                        functions_changed=functions,
                        classes_changed=classes,
                        lines_added=added,
                        lines_removed=removed,
                        complexity_change=0.0,  # Would need deeper analysis
                        timestamp=datetime.now()
                    ))

        return changes

    def _analyze_python_file(self, file_path: Path) -> Tuple[List[str], List[str]]:
        """Extract functions and classes from Python file."""
        try:
            tree = ast.parse(file_path.read_text())
            functions = []
            classes = []

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)

            return functions, classes
        except:
            return [], []

    def _generate_change_documentation(self, changes: List[CodeChange]) -> str:
        """Generate markdown documentation from changes."""
        doc = f"""# Code Changes Report
Generated: {datetime.now().isoformat()}

## Summary
- **Files Changed**: {len(changes)}
- **Lines Added**: {sum(c.lines_added for c in changes)}
- **Lines Removed**: {sum(c.lines_removed for c in changes)}

## Changes by File

"""
        for change in changes:
            doc += f"### {change.file_path}\n"
            doc += f"- **Type**: {change.change_type}\n"
            doc += f"- **Lines**: +{change.lines_added} -{change.lines_removed}\n"

            if change.functions_changed:
                doc += f"- **Functions**: {', '.join(change.functions_changed)}\n"
            if change.classes_changed:
                doc += f"- **Classes**: {', '.join(change.classes_changed)}\n"

            doc += "\n"

        return doc

    async def create_architecture_diagram(self) -> Tuple[bool, str]:
        """
        Generate architecture diagram in Mermaid format.

        Returns:
            Tuple of (success, diagram_path)
        """
        diagram = """```mermaid
graph TB
    subgraph "Echo Brain Core"
        INTEL[Intelligence Layer]
        EXEC[Execution Layer]
        OBS[Observability Layer]
    end

    subgraph "Tower Infrastructure"
        OLLAMA[Ollama LLMs]
        GPU1[AMD GPU]
        GPU2[NVIDIA GPU]
        DB[(PostgreSQL)]
        REDIS[(Redis)]
    end

    subgraph "External Services"
        GH[GitHub]
        KB[Knowledge Base]
        VAULT[HashiCorp Vault]
    end

    INTEL --> OLLAMA
    EXEC --> GPU1
    EXEC --> GPU2
    INTEL --> DB
    EXEC --> REDIS
    INTEL --> KB
    EXEC --> GH
    INTEL --> VAULT

    subgraph "Components"
        INTEL --> MR[Model Router]
        INTEL --> CC[Conversation Context]
        EXEC --> VE[Verified Executor]
        EXEC --> IA[Incremental Analyzer]
        EXEC --> SR[Safe Refactor]
        EXEC --> GO[Git Operations]
        OBS --> ET[Execution Traces]
        OBS --> PM[Performance Metrics]
    end
```"""

        doc_path = self.docs_dir / "ARCHITECTURE.md"
        content = f"""# Echo Brain Architecture
Generated: {datetime.now().isoformat()}

## System Overview

{diagram}

## Component Descriptions

### Intelligence Layer
- **Model Router**: Selects appropriate LLM based on task type and urgency
- **Conversation Context**: Maintains multi-turn conversation state
- **Query Handler**: Processes user queries and routes to appropriate handlers

### Execution Layer
- **Verified Executor**: Ensures actions actually succeed with verification
- **Incremental Analyzer**: Processes large codebases without timeout
- **Safe Refactor**: Git-integrated code changes with rollback capability
- **Git Operations**: Version control and GitHub integration

### Observability Layer
- **Execution Traces**: Complete audit trail of all operations
- **Performance Metrics**: Latency, success rates, resource usage
- **Alert Manager**: Proactive issue detection and notification
"""

        doc_path.write_text(content)
        return True, str(doc_path)

    async def document_api_endpoints(self) -> Tuple[bool, str]:
        """
        Auto-generate API documentation from code.

        Returns:
            Tuple of (success, api_doc_path)
        """
        endpoints = []

        # Scan for FastAPI/Flask endpoints
        for py_file in self.project_root.rglob("*.py"):
            try:
                content = py_file.read_text()

                # FastAPI patterns
                fastapi_patterns = [
                    r'@app\.(get|post|put|delete|patch)\("([^"]+)"\)',
                    r'@router\.(get|post|put|delete|patch)\("([^"]+)"\)'
                ]

                for pattern in fastapi_patterns:
                    matches = re.findall(pattern, content)
                    for method, path in matches:
                        endpoints.append({
                            "method": method.upper(),
                            "path": path,
                            "file": str(py_file.relative_to(self.project_root))
                        })

            except Exception as e:
                logger.warning(f"Failed to parse {py_file}: {e}")

        # Generate documentation
        doc = f"""# API Documentation
Generated: {datetime.now().isoformat()}

## Endpoints

| Method | Path | Source |
|--------|------|--------|
"""
        for ep in sorted(endpoints, key=lambda x: (x["path"], x["method"])):
            doc += f"| {ep['method']} | {ep['path']} | {ep['file']} |\n"

        doc += f"\n## Total Endpoints: {len(endpoints)}\n"

        doc_path = self.docs_dir / "API.md"
        doc_path.write_text(doc)

        return True, str(doc_path)

    async def record_decision(
        self,
        title: str,
        context: str,
        options: List[str],
        choice: str,
        rationale: str,
        impact: str,
        related_files: Optional[List[str]] = None
    ) -> Tuple[bool, str]:
        """
        Record an architectural decision (ADR).

        Returns:
            Tuple of (success, adr_path)
        """
        decision = Decision(
            decision_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            title=title,
            context=context,
            options_considered=options,
            choice_made=choice,
            rationale=rationale,
            impact=impact,
            timestamp=datetime.now(),
            related_files=related_files or []
        )

        # Generate ADR document
        adr_content = f"""# {title}

## Status
**Decided** - {decision.timestamp.isoformat()}

## Context
{context}

## Options Considered
"""
        for i, option in enumerate(options, 1):
            adr_content += f"{i}. {option}\n"

        adr_content += f"""
## Decision
**Choice**: {choice}

## Rationale
{rationale}

## Impact
{impact}

## Related Files
"""
        for file in related_files or []:
            adr_content += f"- {file}\n"

        adr_content += f"""
---
*Decision ID: {decision.decision_id}*
*Generated by Echo Brain*
"""

        # Save ADR
        adr_path = self.docs_dir / "decisions" / f"ADR_{decision.decision_id}_{title.replace(' ', '_')}.md"
        adr_path.parent.mkdir(exist_ok=True)
        adr_path.write_text(adr_content)

        # Save to knowledge base
        await self._save_to_kb(
            title=f"ADR: {title}",
            content=adr_content,
            category="decisions",
            tags=["adr", "architecture", "decision"]
        )

        return True, str(adr_path)

    async def generate_readme(self) -> Tuple[bool, str]:
        """
        Auto-generate or update README.md.

        Returns:
            Tuple of (success, readme_path)
        """
        # Collect metrics
        metrics = await self._collect_system_metrics()

        readme_content = f"""# Echo Brain - Autonomous AI System

## Overview
Echo Brain is an autonomous AI orchestration system that provides intelligent task execution, code refactoring, and system management capabilities.

## System Metrics
- **Total Files**: {metrics.total_files}
- **Total Lines of Code**: {metrics.total_lines:,}
- **API Endpoints**: {metrics.api_endpoints}
- **Database Tables**: {metrics.database_tables}

## Features

### Intelligence Layer
- Multi-model routing with urgency awareness
- Conversation context management
- Smart model selection based on task requirements

### Execution Layer
- Verified task execution with rollback capability
- Incremental code analysis for large codebases
- Git-integrated safe refactoring
- Automated documentation generation

### Observability Layer
- Comprehensive execution tracing
- Performance monitoring and metrics
- Automated alerting and reporting

## Installation
```bash
cd /opt/tower-echo-brain
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration
Edit `config/settings.yaml` to configure:
- Model endpoints
- Database connections
- API keys
- Execution parameters

## Usage
```python
from src.core.echo.echo_brain import EchoBrain

echo = EchoBrain()
result = await echo.execute_task("refactor code for better performance")
```

## Documentation
- [Architecture](docs/ARCHITECTURE.md)
- [API Reference](docs/API.md)
- [Decision Records](docs/decisions/)
- [Change Logs](docs/CHANGES_*.md)

## Dependencies
"""
        for dep in metrics.dependencies[:10]:  # Top 10 dependencies
            readme_content += f"- {dep}\n"

        readme_content += f"""
---
*Last Updated: {datetime.now().isoformat()}*
*Auto-generated by Echo Brain Documentation System*
"""

        readme_path = self.project_root / "README.md"
        readme_path.write_text(readme_content)

        return True, str(readme_path)

    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # Count files and lines
        total_files = len(list(self.project_root.rglob("*.py")))
        total_lines = 0
        for f in self.project_root.rglob("*.py"):
            if f.is_file():
                try:
                    total_lines += len(f.read_text().splitlines())
                except UnicodeDecodeError:
                    # Skip files with encoding issues
                    pass

        # Count API endpoints (simplified)
        api_endpoints = 0
        for f in self.project_root.rglob("*.py"):
            try:
                content = f.read_text()
                api_endpoints += content.count("@app.")
                api_endpoints += content.count("@router.")
            except:
                pass

        # Get dependencies from requirements.txt if exists
        deps = []
        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            deps = [
                line.strip()
                for line in req_file.read_text().splitlines()
                if line.strip() and not line.startswith("#")
            ]

        return SystemMetrics(
            total_files=total_files,
            total_lines=total_lines,
            test_coverage=0.0,  # Would need pytest-cov
            documentation_coverage=0.0,  # Would need analysis
            complexity_score=0.0,  # Would need radon
            dependencies=deps,
            api_endpoints=api_endpoints,
            database_tables=27  # From echo_brain schema
        )

    async def _save_to_kb(
        self,
        title: str,
        content: str,
        category: str,
        tags: List[str]
    ) -> bool:
        """Save documentation to knowledge base."""
        try:
            response = await self.http_client.post(
                f"{self.kb_url}/articles",
                json={
                    "title": title,
                    "content": content,
                    "category": category,
                    "tags": tags
                }
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Failed to save to KB: {e}")
            return False


async def test_auto_documenter():
    """Test auto-documentation features."""
    doc = AutoDocumenter()

    print("Testing Auto-Documenter...")

    # Generate architecture diagram
    success, path = await doc.create_architecture_diagram()
    print(f"  Architecture Diagram: {'✅' if success else '❌'} {path}")

    # Document API endpoints
    success, path = await doc.document_api_endpoints()
    print(f"  API Documentation: {'✅' if success else '❌'} {path}")

    # Record a decision
    success, path = await doc.record_decision(
        title="Use Smart Model Manager",
        context="Need to handle models that don't fit in VRAM",
        options=["Always use largest model", "Use API fallback", "Smart selection based on availability"],
        choice="Smart selection based on availability",
        rationale="Balances quality with response time and hardware constraints",
        impact="Better user experience with faster responses",
        related_files=["src/intelligence/smart_model_manager.py"]
    )
    print(f"  Decision Record: {'✅' if success else '❌'} {path}")

    # Generate README
    success, path = await doc.generate_readme()
    print(f"  README Generation: {'✅' if success else '❌'} {path}")

    print("\n✅ Auto-documenter test complete")


if __name__ == "__main__":
    asyncio.run(test_auto_documenter())