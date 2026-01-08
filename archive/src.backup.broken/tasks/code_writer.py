#!/usr/bin/env python3
"""
Code Writer Executor - Safe file operations with validation
Autonomous code writing capability for Echo Brain
"""

import asyncio
import ast
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json

logger = logging.getLogger(__name__)

class CodeWriter:
    """Safe file operations with validation and backups"""
    
    def __init__(self, backup_dir: str = "/opt/tower-echo-brain/backups/code"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"CodeWriter initialized with backup dir: {self.backup_dir}")
    
    async def read_file(self, path: str) -> str:
        """Safely read file contents"""
        try:
            file_path = Path(path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            
            if not file_path.is_file():
                raise ValueError(f"Path is not a file: {path}")
            
            # Security check: ensure path is within /opt/
            if not str(file_path.resolve()).startswith("/opt/"):
                raise PermissionError(f"Access denied: {path} (must be in /opt/)")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"Read {len(content)} bytes from {path}")
            return content
        
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
            raise
    
    async def write_file(self, path: str, content: str, backup: bool = True) -> Dict[str, Any]:
        """
        Safely write file with backup and validation
        
        Args:
            path: Target file path
            content: Content to write
            backup: Create backup if file exists
        
        Returns:
            Dict with success, backup_path, validation results
        """
        result = {
            'success': False,
            'path': path,
            'backup_path': None,
            'validation': None,
            'error': None,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            file_path = Path(path)
            
            # Security check
            if not str(file_path.resolve()).startswith("/opt/"):
                raise PermissionError(f"Access denied: {path} (must be in /opt/)")
            
            # Validate Python syntax if .py file
            if file_path.suffix == '.py':
                is_valid, error_msg = await self.validate_python_syntax(content)
                result['validation'] = {
                    'valid': is_valid,
                    'error': error_msg
                }
                if not is_valid:
                    raise SyntaxError(f"Invalid Python syntax: {error_msg}")
            
            # Create backup if requested and file exists
            if backup and file_path.exists():
                backup_path = await self._create_backup(file_path)
                result['backup_path'] = str(backup_path)
                logger.info(f"Created backup: {backup_path}")
            
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            result['success'] = True
            result['bytes_written'] = len(content)
            logger.info(f"Wrote {len(content)} bytes to {path}")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error writing {path}: {e}")
        
        return result
    
    async def edit_line(self, path: str, line_num: int, new_content: str) -> Dict[str, Any]:
        """
        Edit a specific line in a file
        
        Args:
            path: File path
            line_num: Line number (1-indexed)
            new_content: New line content
        """
        result = {
            'success': False,
            'path': path,
            'line_num': line_num,
            'old_content': None,
            'new_content': new_content,
            'error': None
        }
        
        try:
            # Read file
            content = await self.read_file(path)
            lines = content.split('\n')
            
            # Validate line number
            if line_num < 1 or line_num > len(lines):
                raise ValueError(f"Line {line_num} out of range (1-{len(lines)})")
            
            # Store old content
            result['old_content'] = lines[line_num - 1]
            
            # Update line
            lines[line_num - 1] = new_content
            
            # Write back
            write_result = await self.write_file(path, '\n'.join(lines), backup=True)
            result['success'] = write_result['success']
            result['backup_path'] = write_result['backup_path']
            
            logger.info(f"Updated line {line_num} in {path}")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error editing line {line_num} in {path}: {e}")
        
        return result
    
    async def replace_pattern(self, path: str, pattern: str, replacement: str, 
                            regex: bool = False, max_count: int = 0) -> Dict[str, Any]:
        """
        Replace pattern in file
        
        Args:
            path: File path
            pattern: Pattern to find
            replacement: Replacement text
            regex: Use regex if True, literal string if False
            max_count: Maximum replacements (0 = all)
        """
        result = {
            'success': False,
            'path': path,
            'pattern': pattern,
            'replacement': replacement,
            'replacements_made': 0,
            'error': None
        }
        
        try:
            # Read file
            content = await self.read_file(path)
            
            # Perform replacement
            if regex:
                new_content, count = re.subn(pattern, replacement, content, count=max_count)
            else:
                if max_count > 0:
                    new_content = content.replace(pattern, replacement, max_count)
                    count = min(content.count(pattern), max_count)
                else:
                    new_content = content.replace(pattern, replacement)
                    count = content.count(pattern)
            
            result['replacements_made'] = count
            
            if count > 0:
                # Write back
                write_result = await self.write_file(path, new_content, backup=True)
                result['success'] = write_result['success']
                result['backup_path'] = write_result['backup_path']
                logger.info(f"Made {count} replacements in {path}")
            else:
                logger.warning(f"Pattern '{pattern}' not found in {path}")
                result['success'] = True  # Not an error, just no matches
                
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error replacing pattern in {path}: {e}")
        
        return result
    
    async def validate_python_syntax(self, content: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python syntax
        
        Returns:
            (is_valid, error_message)
        """
        try:
            ast.parse(content)
            return (True, None)
        except SyntaxError as e:
            error_msg = f"Line {e.lineno}: {e.msg}"
            return (False, error_msg)
        except Exception as e:
            return (False, str(e))
    
    async def validate_json(self, content: str) -> Tuple[bool, Optional[str]]:
        """Validate JSON syntax"""
        try:
            json.loads(content)
            return (True, None)
        except json.JSONDecodeError as e:
            return (False, str(e))
    
    async def _create_backup(self, file_path: Path) -> Path:
        """Create timestamped backup of file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{file_path.name}.backup_{timestamp}"
        backup_path = self.backup_dir / backup_filename
        
        # Copy file
        shutil.copy2(file_path, backup_path)
        
        return backup_path
    
    async def rollback(self, backup_path: str, target_path: str) -> Dict[str, Any]:
        """Rollback file from backup"""
        result = {
            'success': False,
            'backup_path': backup_path,
            'target_path': target_path,
            'error': None
        }
        
        try:
            backup = Path(backup_path)
            target = Path(target_path)
            
            if not backup.exists():
                raise FileNotFoundError(f"Backup not found: {backup_path}")
            
            # Copy backup to target
            shutil.copy2(backup, target)
            
            result['success'] = True
            logger.info(f"Rolled back {target_path} from {backup_path}")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error rolling back: {e}")
        
        return result
    
    async def list_backups(self, filename: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all backups, optionally filtered by filename"""
        backups = []
        
        for backup_file in sorted(self.backup_dir.glob("*.backup_*"), reverse=True):
            if filename and not backup_file.name.startswith(filename):
                continue
            
            backups.append({
                'path': str(backup_file),
                'filename': backup_file.name,
                'size': backup_file.stat().st_size,
                'created': datetime.fromtimestamp(backup_file.stat().st_mtime).isoformat()
            })
        
        return backups

# Singleton instance
_code_writer = None

def get_code_writer() -> CodeWriter:
    """Get singleton CodeWriter instance"""
    global _code_writer
    if _code_writer is None:
        _code_writer = CodeWriter()
    return _code_writer
