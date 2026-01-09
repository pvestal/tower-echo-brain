#!/usr/bin/env python3
"""
Incremental code analysis system that processes files in manageable batches.
Separation of concerns: This handles WHAT to analyze, not HOW to analyze.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional
import asyncio
import hashlib
import json

@dataclass
class AnalysisTarget:
    """Single file targeted for analysis with metadata."""
    path: Path
    content_hash: str
    last_analyzed: Optional[float] = None
    priority: int = 0  # Higher = analyze sooner

@dataclass
class AnalysisBatch:
    """Batch of files to analyze together."""
    targets: list[AnalysisTarget]
    batch_id: str
    max_tokens: int = 50000  # Approximate token budget per batch

class IncrementalAnalyzer:
    """
    Manages incremental analysis of large codebases.

    Design principles:
    - Never analyze unchanged files
    - Process in batches that won't timeout
    - Track progress persistently
    - Prioritize recently modified files
    """

    def __init__(self, project_root: Path, state_file: Path):
        self.project_root = project_root
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> dict:
        """Load analysis state from persistent storage."""
        if self.state_file.exists():
            return json.loads(self.state_file.read_text())
        return {"file_hashes": {}, "last_full_scan": None}

    def _save_state(self) -> None:
        """Persist analysis state."""
        self.state_file.write_text(json.dumps(self.state, indent=2))

    def _hash_file(self, path: Path) -> str:
        """Generate content hash for change detection."""
        content = path.read_bytes()
        return hashlib.sha256(content).hexdigest()[:16]

    def get_changed_files(self, extensions: list[str] = None) -> Iterator[AnalysisTarget]:
        """
        Yield files that have changed since last analysis.

        Args:
            extensions: File extensions to include (e.g., ['.py', '.ts'])
        """
        extensions = extensions or ['.py']

        for path in self.project_root.rglob('*'):
            # Skip non-files and excluded directories
            if not path.is_file():
                continue
            if any(part.startswith('.') for part in path.parts):
                continue
            if 'node_modules' in path.parts or '__pycache__' in path.parts:
                continue
            if path.suffix not in extensions:
                continue

            current_hash = self._hash_file(path)
            stored_hash = self.state["file_hashes"].get(str(path))

            if current_hash != stored_hash:
                yield AnalysisTarget(
                    path=path,
                    content_hash=current_hash,
                    last_analyzed=self.state.get("timestamps", {}).get(str(path))
                )

    def create_batches(self, targets: list[AnalysisTarget], max_files: int = 20) -> Iterator[AnalysisBatch]:
        """
        Group targets into analyzable batches.

        Args:
            targets: Files to batch
            max_files: Maximum files per batch (prevents timeout)
        """
        batch = []
        batch_num = 0

        for target in targets:
            batch.append(target)
            if len(batch) >= max_files:
                yield AnalysisBatch(
                    targets=batch,
                    batch_id=f"batch_{batch_num}"
                )
                batch = []
                batch_num += 1

        if batch:
            yield AnalysisBatch(targets=batch, batch_id=f"batch_{batch_num}")

    def mark_analyzed(self, target: AnalysisTarget) -> None:
        """Record that a file has been analyzed."""
        self.state["file_hashes"][str(target.path)] = target.content_hash
        self._save_state()


def test_incremental_analyzer():
    """Verify analyzer only returns changed files."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        state_file = root / ".analysis_state.json"

        # Create test file
        test_file = root / "test.py"
        test_file.write_text("x = 1")

        analyzer = IncrementalAnalyzer(root, state_file)

        # First run: should find the file
        changed = list(analyzer.get_changed_files())
        assert len(changed) == 1, f"Expected 1 changed file, got {len(changed)}"

        # Mark as analyzed
        analyzer.mark_analyzed(changed[0])

        # Second run: should find nothing
        changed = list(analyzer.get_changed_files())
        assert len(changed) == 0, f"Expected 0 changed files, got {len(changed)}"

        # Modify file
        test_file.write_text("x = 2")

        # Third run: should find it again
        changed = list(analyzer.get_changed_files())
        assert len(changed) == 1, f"Expected 1 changed file after modification"

        print("✅ Incremental analyzer working correctly")


if __name__ == "__main__":
    test_incremental_analyzer()