"""
Self-Modification System for Echo Brain
Enables autonomous code changes with safety controls
"""
import os
import ast
import json
import shutil
import hashlib
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import asyncpg

logger = logging.getLogger(__name__)


@dataclass
class CodeChange:
    """Represents a code change to be applied"""
    file_path: str
    change_type: str  # 'create', 'modify', 'delete'
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    description: str = ""
    risk_level: str = "low"  # low, medium, high

    def to_dict(self) -> Dict:
        return {
            'file_path': self.file_path,
            'change_type': self.change_type,
            'old_content': self.old_content,
            'new_content': self.new_content,
            'line_start': self.line_start,
            'line_end': self.line_end,
            'description': self.description,
            'risk_level': self.risk_level
        }


@dataclass
class BackupEntry:
    """Tracks file backups for rollback"""
    original_path: str
    backup_path: str
    timestamp: datetime
    checksum: str
    change_id: str


class CodeSafetyController:
    """Determines what code changes can be safely applied"""

    def __init__(self):
        # Define safe paths for modification
        self.allowed_paths = [
            '/opt/tower-echo-brain/src/',
            '/opt/tower-echo-brain/tests/',
            '/opt/tower-echo-brain/config/'
        ]

        # Paths that should never be modified
        self.forbidden_paths = [
            '/etc/',
            '/usr/',
            '/bin/',
            '/sbin/',
            '/boot/',
            '/lib/',
            '/dev/',
            '/proc/',
            '/sys/',
            '.git/',
            '__pycache__/',
            '.env',
            'credentials',
            'secrets'
        ]

        # File extensions allowed for modification
        self.allowed_extensions = ['.py', '.json', '.yaml', '.yml', '.txt', '.md', '.sql']

        # Maximum file size for modification (10MB)
        self.max_file_size = 10 * 1024 * 1024

        # Maximum number of files per change
        self.max_files_per_change = 10

    async def evaluate_change(self, change: CodeChange) -> Tuple[bool, str, str]:
        """
        Evaluate if a code change is safe to apply

        Returns:
            Tuple of (is_safe, risk_level, reason)
        """
        try:
            path = Path(change.file_path)

            # Check if path is absolute
            if not path.is_absolute():
                return False, "high", "Path must be absolute"

            # Check forbidden paths
            for forbidden in self.forbidden_paths:
                if forbidden in str(path):
                    return False, "forbidden", f"Path contains forbidden directory: {forbidden}"

            # Check allowed paths
            path_allowed = False
            for allowed in self.allowed_paths:
                if str(path).startswith(allowed):
                    path_allowed = True
                    break

            if not path_allowed:
                return False, "high", f"Path not in allowed directories"

            # Check file extension
            if path.suffix not in self.allowed_extensions:
                return False, "medium", f"File extension {path.suffix} not allowed"

            # Check file size for existing files
            if change.change_type == 'modify' and path.exists():
                file_size = path.stat().st_size
                if file_size > self.max_file_size:
                    return False, "medium", f"File too large: {file_size} bytes"

            # Evaluate risk based on change type and location
            risk_level = self._calculate_risk_level(change)

            # Check for dangerous patterns in the new content
            if change.new_content:
                dangerous_patterns = [
                    'exec(', 'eval(', '__import__',
                    'subprocess.', 'os.system',
                    'rm -rf', 'DROP TABLE', 'DELETE FROM',
                    'password=', 'api_key=', 'secret='
                ]

                for pattern in dangerous_patterns:
                    if pattern in change.new_content:
                        return False, "high", f"Dangerous pattern detected: {pattern}"

            return True, risk_level, "Change approved"

        except Exception as e:
            logger.error(f"Failed to evaluate change safety: {e}")
            return False, "high", f"Evaluation error: {e}"

    def _calculate_risk_level(self, change: CodeChange) -> str:
        """Calculate the risk level of a change"""
        path = Path(change.file_path)

        # Core system files are higher risk
        if 'core.py' in path.name or 'safety.py' in path.name:
            return "high"

        # Test files are lower risk
        if '/tests/' in str(path):
            return "low"

        # Config files are medium risk
        if '/config/' in str(path):
            return "medium"

        # Delete operations are higher risk
        if change.change_type == 'delete':
            return "high"

        # New files are lower risk
        if change.change_type == 'create':
            return "low"

        return "medium"


class CodeChangeValidator:
    """Validates code changes before and after application"""

    def __init__(self):
        self.python_version = "3.8"  # Minimum Python version

    async def validate_syntax(self, code: str, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python syntax

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not file_path.endswith('.py'):
            return True, None  # Skip validation for non-Python files

        try:
            # Try to parse the Python code
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"Validation error: {e}"

    async def validate_imports(self, code: str) -> Tuple[bool, List[str]]:
        """
        Check if all imports are available

        Returns:
            Tuple of (all_available, missing_imports)
        """
        try:
            tree = ast.parse(code)
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module.split('.')[0])

            missing = []
            for imp in set(imports):
                try:
                    __import__(imp)
                except ImportError:
                    missing.append(imp)

            return len(missing) == 0, missing

        except Exception as e:
            logger.error(f"Failed to validate imports: {e}")
            return False, [str(e)]

    async def run_tests(self, test_path: Optional[str] = None) -> Tuple[bool, str]:
        """
        Run tests to ensure changes don't break functionality

        Returns:
            Tuple of (tests_passed, output)
        """
        try:
            cmd = "python -m pytest"
            if test_path:
                cmd += f" {test_path}"
            else:
                cmd += " /opt/tower-echo-brain/tests/"

            # Add minimal test run flags
            cmd += " -q --tb=short --maxfail=3"

            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()
            output = f"{stdout.decode()}\n{stderr.decode()}"

            return process.returncode == 0, output

        except Exception as e:
            return False, f"Test execution failed: {e}"


class CodeApplicator:
    """Applies code changes with backup and rollback capabilities"""

    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.backup_dir = Path('/opt/tower-echo-brain/.backups')
        self.backup_dir.mkdir(exist_ok=True)
        self.safety_controller = CodeSafetyController()
        self.validator = CodeChangeValidator()
        self._pool = None

    async def get_connection(self):
        """Get database connection"""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(**self.db_config, min_size=2, max_size=10)
        return self._pool

    async def apply_change(self, change: CodeChange, task_id: int) -> Tuple[bool, str, Optional[str]]:
        """
        Apply a code change with full safety checks and validation

        Returns:
            Tuple of (success, message, backup_id)
        """
        backup_id = None

        try:
            # 1. Safety check
            is_safe, risk_level, reason = await self.safety_controller.evaluate_change(change)
            if not is_safe:
                return False, f"Change rejected by safety controller: {reason}", None

            # 2. Create backup if modifying existing file
            if change.change_type in ['modify', 'delete']:
                backup_id = await self._create_backup(change.file_path, task_id)

            # 3. Validate new content syntax
            if change.new_content:
                is_valid, error = await self.validator.validate_syntax(
                    change.new_content,
                    change.file_path
                )
                if not is_valid:
                    return False, f"Syntax validation failed: {error}", backup_id

            # 4. Apply the change
            success, apply_msg = await self._apply_file_change(change)
            if not success:
                if backup_id:
                    await self._rollback(backup_id)
                return False, f"Failed to apply change: {apply_msg}", None

            # 5. Post-application validation
            if change.change_type != 'delete' and change.file_path.endswith('.py'):
                # Validate the file can still be imported
                with open(change.file_path, 'r') as f:
                    content = f.read()

                is_valid, error = await self.validator.validate_syntax(content, change.file_path)
                if not is_valid:
                    await self._rollback(backup_id)
                    return False, f"Post-application validation failed: {error}", None

            # 6. Log successful change
            await self._log_change(task_id, change, backup_id, "success", risk_level)

            return True, f"Successfully applied {change.change_type} to {change.file_path}", backup_id

        except Exception as e:
            logger.error(f"Failed to apply change: {e}")
            if backup_id:
                await self._rollback(backup_id)
            return False, f"Error applying change: {e}", None

    async def _create_backup(self, file_path: str, task_id: int) -> str:
        """Create a backup of a file before modification"""
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                return None

            # Generate backup ID
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_id = f"{task_id}_{timestamp}_{source_path.name}"

            # Create backup path
            backup_path = self.backup_dir / backup_id

            # Calculate checksum
            with open(source_path, 'rb') as f:
                checksum = hashlib.sha256(f.read()).hexdigest()

            # Copy file to backup location
            shutil.copy2(source_path, backup_path)

            # Store backup record in database
            pool = await self.get_connection()
            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO code_change_backups
                    (backup_id, original_path, backup_path, checksum, task_id)
                    VALUES ($1, $2, $3, $4, $5)
                """, backup_id, str(source_path), str(backup_path), checksum, task_id)

            logger.info(f"Created backup {backup_id} for {file_path}")
            return backup_id

        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise

    async def _apply_file_change(self, change: CodeChange) -> Tuple[bool, str]:
        """Apply the actual file change"""
        try:
            path = Path(change.file_path)

            if change.change_type == 'create':
                # Ensure parent directory exists
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(change.new_content)

            elif change.change_type == 'modify':
                if change.line_start is not None and change.line_end is not None:
                    # Partial file modification
                    lines = path.read_text().split('\n')
                    new_lines = change.new_content.split('\n')
                    lines[change.line_start:change.line_end] = new_lines
                    path.write_text('\n'.join(lines))
                else:
                    # Full file replacement
                    path.write_text(change.new_content)

            elif change.change_type == 'delete':
                path.unlink()

            return True, "Change applied successfully"

        except Exception as e:
            return False, str(e)

    async def _rollback(self, backup_id: str) -> bool:
        """Rollback a change using backup"""
        try:
            pool = await self.get_connection()
            async with pool.acquire() as conn:
                backup = await conn.fetchrow("""
                    SELECT original_path, backup_path, checksum
                    FROM code_change_backups
                    WHERE backup_id = $1
                """, backup_id)

                if not backup:
                    logger.error(f"Backup {backup_id} not found")
                    return False

                # Restore the file
                backup_path = Path(backup['backup_path'])
                original_path = Path(backup['original_path'])

                if backup_path.exists():
                    shutil.copy2(backup_path, original_path)
                    logger.info(f"Rolled back {original_path} from backup {backup_id}")
                    return True
                else:
                    logger.error(f"Backup file {backup_path} not found")
                    return False

        except Exception as e:
            logger.error(f"Failed to rollback: {e}")
            return False

    async def _log_change(self, task_id: int, change: CodeChange,
                         backup_id: Optional[str], status: str, risk_level: str):
        """Log the code change to database"""
        try:
            pool = await self.get_connection()
            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO code_changes
                    (task_id, file_path, change_type, description, risk_level,
                     backup_id, status, applied_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, task_id, change.file_path, change.change_type,
                    change.description, risk_level, backup_id, status, datetime.now())

        except Exception as e:
            logger.error(f"Failed to log code change: {e}")


class SelfImprovementAgent:
    """Agent that generates and applies self-improvement changes"""

    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.applicator = CodeApplicator(db_config)
        self.validator = CodeChangeValidator()

    async def execute_improvement(self, task_id: int, improvement_type: str,
                                 target_file: str, description: str) -> Dict[str, Any]:
        """
        Execute a self-improvement task

        Args:
            task_id: Task ID from autonomous_tasks
            improvement_type: Type of improvement (optimize, refactor, fix_bug, enhance)
            target_file: File to improve
            description: What to improve

        Returns:
            Execution result dictionary
        """
        try:
            # 1. Analyze the current code
            analysis = await self._analyze_code(target_file, improvement_type)

            # 2. Generate improvement suggestions
            changes = await self._generate_changes(
                target_file,
                improvement_type,
                description,
                analysis
            )

            # 3. Apply changes one by one
            results = []
            for change in changes:
                success, message, backup_id = await self.applicator.apply_change(
                    change,
                    task_id
                )

                results.append({
                    'file': change.file_path,
                    'type': change.change_type,
                    'success': success,
                    'message': message,
                    'backup_id': backup_id
                })

                if not success:
                    # Stop on first failure
                    break

            # 4. Run tests to verify improvements
            if all(r['success'] for r in results):
                tests_passed, test_output = await self.validator.run_tests()

                if not tests_passed:
                    # Rollback all changes if tests fail
                    for result in results:
                        if result['backup_id']:
                            await self.applicator._rollback(result['backup_id'])

                    return {
                        'success': False,
                        'message': 'Tests failed after applying changes',
                        'test_output': test_output,
                        'changes_rolled_back': True
                    }

            return {
                'success': all(r['success'] for r in results),
                'changes_applied': len([r for r in results if r['success']]),
                'results': results
            }

        except Exception as e:
            logger.error(f"Failed to execute improvement: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def _analyze_code(self, file_path: str, improvement_type: str) -> Dict[str, Any]:
        """Analyze code for potential improvements"""
        # This would integrate with the reasoning agent
        # For now, return basic analysis
        return {
            'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            'improvement_type': improvement_type,
            'timestamp': datetime.now().isoformat()
        }

    async def _generate_changes(self, file_path: str, improvement_type: str,
                               description: str, analysis: Dict) -> List[CodeChange]:
        """Generate code changes based on improvement type"""
        # This would integrate with the coding agent to generate actual improvements
        # For now, return empty list as placeholder
        return []


# Database schema for tracking changes
SCHEMA = """
CREATE TABLE IF NOT EXISTS code_changes (
    id SERIAL PRIMARY KEY,
    task_id INTEGER REFERENCES autonomous_tasks(id),
    file_path TEXT NOT NULL,
    change_type VARCHAR(20) NOT NULL,
    description TEXT,
    risk_level VARCHAR(20),
    backup_id VARCHAR(100),
    status VARCHAR(20) NOT NULL,
    applied_at TIMESTAMP DEFAULT NOW(),
    rolled_back_at TIMESTAMP,
    INDEX idx_task_id (task_id),
    INDEX idx_backup_id (backup_id)
);

CREATE TABLE IF NOT EXISTS code_change_backups (
    backup_id VARCHAR(100) PRIMARY KEY,
    original_path TEXT NOT NULL,
    backup_path TEXT NOT NULL,
    checksum VARCHAR(64) NOT NULL,
    task_id INTEGER REFERENCES autonomous_tasks(id),
    created_at TIMESTAMP DEFAULT NOW()
);
"""