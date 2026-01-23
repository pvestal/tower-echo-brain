#!/usr/bin/env python3
"""
Execute code tasks using local Ollama models with actual code execution.
"""
import sys
import httpx
import json
import subprocess
import tempfile
import os

OLLAMA_URL = "http://localhost:11434"
MODEL = "deepseek-coder-v2:16b"

def generate_and_execute(task: str) -> dict:
    """Generate code and actually execute it."""
    
    # Step 1: Generate code
    prompt = f"""Write Python code to solve this task. Return ONLY executable Python code, no explanations.
Task: {task}

Return complete, runnable Python code:"""

    response = httpx.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=60.0
    )
    
    generated_code = response.json().get("response", "")
    
    # Extract code from markdown if wrapped
    if "```python" in generated_code:
        start = generated_code.find("```python") + 9
        end = generated_code.find("```", start)
        code = generated_code[start:end].strip()
    elif "```" in generated_code:
        start = generated_code.find("```") + 3
        end = generated_code.find("```", start)
        code = generated_code[start:end].strip()
    else:
        code = generated_code.strip()
    
    # Step 2: Execute the code
    execution_result = {}
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        execution_result = {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "executed": True
        }
        
        os.unlink(temp_path)
        
    except Exception as e:
        execution_result = {
            "error": str(e),
            "executed": False
        }
    
    return {
        "task": task,
        "generated_code": code,
        "execution": execution_result,
        "model": MODEL
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: execute_code.py 'task description'")
        sys.exit(1)
    
    task = ' '.join(sys.argv[1:])
    result = generate_and_execute(task)
    
    print("\n=== GENERATED CODE ===")
    print(result["generated_code"])
    
    print("\n=== EXECUTION RESULT ===")
    if result["execution"].get("executed"):
        print("STDOUT:", result["execution"]["stdout"])
        if result["execution"]["stderr"]:
            print("STDERR:", result["execution"]["stderr"])
        print("Return Code:", result["execution"]["returncode"])
    else:
        print("Failed to execute:", result["execution"].get("error"))
    
    # Save full result
    with open("/tmp/code_execution_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nFull result saved to /tmp/code_execution_result.json")

if __name__ == "__main__":
    main()
