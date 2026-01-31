import os
import subprocess
from pathlib import Path

class CodebaseSearcher:
    def __init__(self, codebase_path="/home/patrick/code"):
        self.codebase_path = Path(codebase_path)
    
    def search_code(self, query, file_extensions=[".py", ".js", ".md"]):
        """Grep through codebase for relevant code"""
        results = []
        for ext in file_extensions:
            cmd = ["grep", "-r", "-n", query, str(self.codebase_path), "--include", f"*{ext}"]
            try:
                output = subprocess.check_output(cmd, text=True)
                if output:
                    results.append(f"Found in {ext} files:\n{output[:1000]}...")
            except:
                pass
        return "\n".join(results) if results else "No code found"
