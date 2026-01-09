#!/usr/bin/env python3
"""
Tower LLM Agent - Gives execution capabilities to Tower's local LLMs
This allows Claude to oversee while Tower LLMs do the actual work
"""

import os
import json
import subprocess
import requests
from pathlib import Path
import shutil
import ast
import re

class TowerLLMAgent:
    """Agent that can execute commands based on LLM decisions"""

    def __init__(self, model="qwen2.5-coder:7b"):
        self.model = model
        self.ollama_api = "http://localhost:11434/api/generate"
        self.allowed_commands = {
            "move_file": self.move_file,
            "delete_file": self.delete_file,
            "create_directory": self.create_directory,
            "read_file": self.read_file,
            "list_files": self.list_files,
            "run_grep": self.run_grep,
            "write_file": self.write_file,
        }
        self.execution_log = []

    def move_file(self, src: str, dst: str) -> dict:
        """Safely move a file"""
        try:
            src_path = Path(src)
            dst_path = Path(dst)

            if not src_path.exists():
                return {"success": False, "error": f"Source {src} doesn't exist"}

            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dst_path))

            self.execution_log.append(f"MOVED: {src} -> {dst}")
            return {"success": True, "action": "moved", "from": src, "to": dst}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete_file(self, path: str) -> dict:
        """Safely delete a file"""
        try:
            file_path = Path(path)
            if file_path.exists():
                if file_path.is_dir():
                    shutil.rmtree(file_path)
                else:
                    file_path.unlink()
                self.execution_log.append(f"DELETED: {path}")
                return {"success": True, "action": "deleted", "path": path}
            return {"success": False, "error": "File doesn't exist"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def create_directory(self, path: str) -> dict:
        """Create a directory"""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.execution_log.append(f"CREATED DIR: {path}")
            return {"success": True, "action": "created_dir", "path": path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def read_file(self, path: str, lines: int = 50) -> dict:
        """Read a file's contents"""
        try:
            with open(path, 'r') as f:
                content = f.read().splitlines()[:lines]
            return {"success": True, "content": content, "path": path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_files(self, path: str, pattern: str = "*") -> dict:
        """List files in directory"""
        try:
            files = list(Path(path).glob(pattern))
            return {"success": True, "files": [str(f) for f in files[:100]]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def run_grep(self, pattern: str, path: str) -> dict:
        """Run grep to find patterns"""
        try:
            result = subprocess.run(
                ["grep", "-r", pattern, path],
                capture_output=True,
                text=True,
                timeout=10
            )
            matches = result.stdout.splitlines()[:20]
            return {"success": True, "matches": matches}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def write_file(self, path: str, content: str) -> dict:
        """Write content to a file"""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                f.write(content)
            self.execution_log.append(f"WROTE: {path}")
            return {"success": True, "action": "wrote", "path": path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def ask_llm(self, prompt: str) -> str:
        """Query the Tower LLM for decisions"""
        try:
            response = requests.post(self.ollama_api, json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.1
            }, timeout=60)

            if response.status_code == 200:
                return response.json().get("response", "")
            return ""
        except Exception as e:
            print(f"LLM Error: {e}")
            return ""

    def parse_llm_commands(self, llm_response: str) -> list:
        """Parse LLM response into executable commands"""
        commands = []

        # Look for JSON command blocks
        json_pattern = r'```json\s*(.*?)\s*```'
        json_matches = re.findall(json_pattern, llm_response, re.DOTALL)

        for match in json_matches:
            try:
                cmd = json.loads(match)
                if isinstance(cmd, dict) and "command" in cmd:
                    commands.append(cmd)
                elif isinstance(cmd, list):
                    commands.extend([c for c in cmd if "command" in c])
            except:
                pass

        # Also look for simple command patterns
        if not commands:
            if "move_file" in llm_response:
                # Extract move commands
                move_pattern = r'move_file\("([^"]+)",\s*"([^"]+)"\)'
                for src, dst in re.findall(move_pattern, llm_response):
                    commands.append({"command": "move_file", "args": {"src": src, "dst": dst}})

        return commands

    def execute_command(self, command: dict) -> dict:
        """Execute a single command safely"""
        cmd_name = command.get("command")
        args = command.get("args", {})

        if cmd_name in self.allowed_commands:
            func = self.allowed_commands[cmd_name]
            if isinstance(args, dict):
                return func(**args)
            else:
                return {"success": False, "error": "Invalid arguments"}
        else:
            return {"success": False, "error": f"Command {cmd_name} not allowed"}

    def run_task(self, task_description: str, context: dict = None) -> dict:
        """Run a complete task with LLM decision-making and execution"""
        print(f"🤖 Tower LLM Agent ({self.model}) starting task...")

        # Build prompt with context
        prompt = f"""You are a Tower LLM Agent with execution capabilities.

TASK: {task_description}

CONTEXT:
{json.dumps(context, indent=2) if context else 'No additional context'}

You can execute these commands:
- move_file(src, dst) - Move a file
- delete_file(path) - Delete a file
- create_directory(path) - Create a directory
- read_file(path, lines) - Read file contents
- list_files(path, pattern) - List files
- run_grep(pattern, path) - Search for patterns
- write_file(path, content) - Write to a file

Respond with JSON command blocks like:
```json
{{"command": "move_file", "args": {{"src": "/path/from", "dst": "/path/to"}}}}
```

Analyze the task and provide commands to execute:
"""

        # Get LLM decision
        print("  📝 Analyzing task...")
        llm_response = self.ask_llm(prompt)

        if not llm_response:
            return {"success": False, "error": "No LLM response"}

        # Parse commands
        commands = self.parse_llm_commands(llm_response)
        print(f"  📋 LLM suggested {len(commands)} commands")

        # Execute commands
        results = []
        for i, cmd in enumerate(commands):
            print(f"  ⚡ Executing command {i+1}/{len(commands)}: {cmd.get('command')}")
            result = self.execute_command(cmd)
            results.append(result)

            if not result.get("success"):
                print(f"    ❌ Failed: {result.get('error')}")

        # Summary
        successful = sum(1 for r in results if r.get("success"))
        print(f"\n✅ Completed: {successful}/{len(commands)} commands successful")

        return {
            "success": True,
            "llm_analysis": llm_response[:500],
            "commands_executed": len(commands),
            "successful_commands": successful,
            "execution_log": self.execution_log,
            "results": results
        }


# Example usage for Echo Brain reorganization
if __name__ == "__main__":
    agent = TowerLLMAgent(model="qwen2.5-coder:7b")

    # Task for the LLM to execute
    result = agent.run_task(
        task_description="Find all Python files in /opt/tower-echo-brain/src/core/echo/ that start with 'echo_' and organize them into subcategories based on their functionality",
        context={
            "base_path": "/opt/tower-echo-brain",
            "target_categories": ["api", "service", "integration", "brain", "auth"],
            "max_files_to_process": 10
        }
    )

    # Save results
    with open("/opt/tower-echo-brain/agent_execution_report.json", 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n📊 Report saved to agent_execution_report.json")
    print(f"Execution log: {', '.join(agent.execution_log[:5])}...")