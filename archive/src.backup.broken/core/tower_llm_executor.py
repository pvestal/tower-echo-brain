#!/usr/bin/env python3
"""
Tower LLM Executor for Echo Brain
Allows Echo to delegate heavy tasks to local Tower LLMs with execution capabilities
"""

import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


class TowerLLMExecutor:
    """Executor that gives Tower LLMs actual capabilities through Echo Brain"""

    def __init__(self, model: str = "qwen2.5-coder:7b"):
        self.model = model
        self.ollama_api = "http://localhost:11434/api/generate"
        self.execution_history = []
        self.max_commands_per_task = 50  # Safety limit

        # Define safe operations Tower LLM can perform
        self.capabilities = {
            "file_operations": {
                "move": self._move_file,
                "copy": self._copy_file,
                "delete": self._delete_file,
                "create_dir": self._create_directory,
                "list": self._list_files,
                "read": self._read_file,
                "write": self._write_file,
            },
            "code_analysis": {
                "find_imports": self._find_imports,
                "find_functions": self._find_functions,
                "find_classes": self._find_classes,
                "analyze_dependencies": self._analyze_dependencies,
            },
            "system_operations": {
                "run_pytest": self._run_pytest,
                "check_syntax": self._check_syntax,
                "format_code": self._format_code,
            }
        }

    async def delegate_task(self, task: str, context: Dict = None) -> Dict:
        """
        Delegate a task to Tower LLM with execution capabilities
        This is the main entry point Echo Brain will use
        """
        logger.info(f"🤖 Delegating to Tower LLM ({self.model}): {task[:100]}...")

        # Build the prompt with available capabilities
        prompt = self._build_prompt(task, context)

        # Get LLM's analysis and command plan
        llm_response = await self._query_llm(prompt)

        if not llm_response:
            return {
                "success": False,
                "error": "Tower LLM did not respond",
                "model": self.model
            }

        # Parse and execute commands
        commands = self._parse_commands(llm_response)
        results = await self._execute_commands(commands)

        # Log to database for Echo's learning
        await self._log_execution(task, commands, results)

        return {
            "success": True,
            "task": task,
            "model": self.model,
            "commands_executed": len(commands),
            "results": results,
            "execution_history": self.execution_history[-10:],  # Last 10 actions
            "timestamp": datetime.now().isoformat()
        }

    def _build_prompt(self, task: str, context: Dict) -> str:
        """Build a structured prompt for Tower LLM"""
        capabilities_doc = json.dumps(list(self.capabilities.keys()), indent=2)

        prompt = f"""You are a Tower LLM with execution capabilities integrated into Echo Brain.

TASK: {task}

CONTEXT: {json.dumps(context, indent=2) if context else 'None'}

AVAILABLE CAPABILITIES:
{capabilities_doc}

You can execute commands by responding with JSON blocks:
```json
{{
  "action": "file_operations.move",
  "params": {{"src": "/path/from", "dst": "/path/to"}}
}}
```

Analyze the task and provide a sequence of commands to execute.
Be specific and include all necessary parameters.
IMPORTANT: You can execute real actions, so be careful and precise.

Response format:
1. Brief analysis of what needs to be done
2. JSON command blocks for execution
3. Expected outcome

Begin your analysis:
"""
        return prompt

    async def _query_llm(self, prompt: str) -> str:
        """Query Tower LLM asynchronously"""
        try:
            # Run in executor to not block
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    self.ollama_api,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0.1,  # Low temp for consistent execution
                    },
                    timeout=60
                )
            )

            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logger.error(f"Tower LLM error: {response.status_code}")
                return ""

        except Exception as e:
            logger.error(f"Failed to query Tower LLM: {e}")
            return ""

    def _parse_commands(self, llm_response: str) -> List[Dict]:
        """Parse LLM response into executable commands"""
        import re
        commands = []

        # Find all JSON blocks in the response
        json_pattern = r'```json\s*(.*?)\s*```'
        json_blocks = re.findall(json_pattern, llm_response, re.DOTALL)

        for block in json_blocks:
            try:
                cmd = json.loads(block)
                if isinstance(cmd, dict) and "action" in cmd:
                    commands.append(cmd)
                elif isinstance(cmd, list):
                    commands.extend([c for c in cmd if isinstance(c, dict) and "action" in c])
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON block: {block[:100]}...")
                continue

        # Safety limit
        return commands[:self.max_commands_per_task]

    async def _execute_commands(self, commands: List[Dict]) -> List[Dict]:
        """Execute parsed commands safely"""
        results = []

        for i, cmd in enumerate(commands):
            try:
                action = cmd.get("action", "")
                params = cmd.get("params", {})

                # Parse action category and method
                if "." in action:
                    category, method = action.split(".", 1)
                    if category in self.capabilities and method in self.capabilities[category]:
                        # Execute the capability
                        func = self.capabilities[category][method]
                        result = await self._execute_safely(func, params)
                        results.append(result)

                        # Log execution
                        self.execution_history.append({
                            "action": action,
                            "params": params,
                            "result": result,
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        results.append({"success": False, "error": f"Unknown action: {action}"})
                else:
                    results.append({"success": False, "error": "Invalid action format"})

            except Exception as e:
                logger.error(f"Command execution failed: {e}")
                results.append({"success": False, "error": str(e)})

        return results

    async def _execute_safely(self, func, params: Dict) -> Dict:
        """Execute a function safely with error handling"""
        try:
            # Run in executor if not async
            if asyncio.iscoroutinefunction(func):
                return await func(**params)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: func(**params))
        except Exception as e:
            return {"success": False, "error": str(e)}

    # File operation implementations
    def _move_file(self, src: str, dst: str) -> Dict:
        """Move a file safely"""
        try:
            src_path = Path(src)
            dst_path = Path(dst)

            if not src_path.exists():
                return {"success": False, "error": f"Source doesn't exist: {src}"}

            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dst_path))

            return {"success": True, "moved": src, "to": dst}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _copy_file(self, src: str, dst: str) -> Dict:
        """Copy a file"""
        try:
            shutil.copy2(src, dst)
            return {"success": True, "copied": src, "to": dst}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _delete_file(self, path: str) -> Dict:
        """Delete a file or directory"""
        try:
            p = Path(path)
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            return {"success": True, "deleted": path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _create_directory(self, path: str) -> Dict:
        """Create a directory"""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return {"success": True, "created": path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _list_files(self, path: str, pattern: str = "*") -> Dict:
        """List files in a directory"""
        try:
            files = list(Path(path).glob(pattern))[:100]  # Limit to 100
            return {"success": True, "files": [str(f) for f in files], "count": len(files)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _read_file(self, path: str, lines: int = 50) -> Dict:
        """Read a file's contents"""
        try:
            with open(path, 'r') as f:
                content = f.read().splitlines()[:lines]
            return {"success": True, "path": path, "content": content}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _write_file(self, path: str, content: str) -> Dict:
        """Write content to a file"""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                f.write(content)
            return {"success": True, "wrote": path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # Code analysis implementations
    def _find_imports(self, path: str) -> Dict:
        """Find all imports in a Python file"""
        try:
            with open(path, 'r') as f:
                lines = f.readlines()

            imports = [line.strip() for line in lines
                      if line.strip().startswith(('import ', 'from '))]

            return {"success": True, "path": path, "imports": imports}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _find_functions(self, path: str) -> Dict:
        """Find all function definitions"""
        try:
            with open(path, 'r') as f:
                lines = f.readlines()

            functions = [line.strip() for line in lines
                        if line.strip().startswith('def ')]

            return {"success": True, "path": path, "functions": functions}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _find_classes(self, path: str) -> Dict:
        """Find all class definitions"""
        try:
            with open(path, 'r') as f:
                lines = f.readlines()

            classes = [line.strip() for line in lines
                      if line.strip().startswith('class ')]

            return {"success": True, "path": path, "classes": classes}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _analyze_dependencies(self, path: str) -> Dict:
        """Analyze file dependencies"""
        imports = self._find_imports(path)
        if not imports["success"]:
            return imports

        return {
            "success": True,
            "path": path,
            "imports": imports["imports"],
            "external": [i for i in imports["imports"] if not i.startswith("from .")],
            "internal": [i for i in imports["imports"] if i.startswith("from .")]
        }

    # System operations
    def _run_pytest(self, path: str) -> Dict:
        """Run pytest on a file or directory"""
        try:
            result = subprocess.run(
                ["pytest", path, "-v"],
                capture_output=True,
                text=True,
                timeout=30
            )
            return {
                "success": result.returncode == 0,
                "output": result.stdout[:1000],
                "errors": result.stderr[:1000] if result.stderr else None
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _check_syntax(self, path: str) -> Dict:
        """Check Python syntax"""
        try:
            result = subprocess.run(
                ["python3", "-m", "py_compile", path],
                capture_output=True,
                text=True,
                timeout=10
            )
            return {
                "success": result.returncode == 0,
                "path": path,
                "valid": result.returncode == 0,
                "errors": result.stderr if result.stderr else None
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _format_code(self, path: str) -> Dict:
        """Format Python code with black"""
        try:
            result = subprocess.run(
                ["black", path],
                capture_output=True,
                text=True,
                timeout=10
            )
            return {
                "success": result.returncode == 0,
                "path": path,
                "formatted": result.returncode == 0
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _log_execution(self, task: str, commands: List[Dict], results: List[Dict]):
        """Log execution to Echo Brain's database for learning"""
        try:
            # This would integrate with Echo's database
            # For now, just log to file
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "task": task,
                "model": self.model,
                "commands": commands,
                "results": results,
                "success_rate": sum(1 for r in results if r.get("success")) / len(results) if results else 0
            }

            log_path = Path("/opt/tower-echo-brain/tower_llm_execution.log")
            with open(log_path, 'a') as f:
                f.write(json.dumps(log_entry) + "\n")

        except Exception as e:
            logger.error(f"Failed to log execution: {e}")


# Global instance for Echo Brain to use
tower_executor = TowerLLMExecutor()