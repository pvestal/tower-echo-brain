#!/usr/bin/env python3
"""Fix model listing timeout and add comprehensive error handling."""

import os
import logging
import asyncio
from pathlib import Path

def fix_model_listing():
    """Fix model listing timeout issue."""

    # Find the model management file
    model_files = [
        "/opt/tower-echo-brain/src/misc/model_manager.py",
        "/opt/tower-echo-brain/src/core/model_management.py",
        "/opt/tower-echo-brain/src/api/models.py"
    ]

    for file_path in model_files:
        if not os.path.exists(file_path):
            continue

        print(f"🔧 Fixing {file_path}...")

        with open(file_path, 'r') as f:
            content = f.read()

        original = content

        # Add timeout handling
        improvements = [
            # Add async timeout
            ("async def list_models(", """@asyncio.timeout(30)
async def list_models("""),

            # Add try-catch for Ollama calls
            ("response = await ollama.", """try:
            response = await asyncio.wait_for(ollama."""),

            # Close the try block
            ("return response", """    return response
        except asyncio.TimeoutError:
            logger.error("Model listing timed out after 30 seconds")
            return {"error": "Model listing timeout"}
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return {"error": str(e)}"""),
        ]

        # Apply improvements carefully
        if "asyncio.timeout" not in content:
            content = "import asyncio\n" + content

        # Fix the actual timeout issue in Ollama calls
        if "ollama.list()" in content:
            content = content.replace(
                "ollama.list()",
                "asyncio.wait_for(ollama.list(), timeout=10.0)"
            )

        if "requests.get" in content and "timeout=" not in content:
            content = content.replace(
                "requests.get(url)",
                "requests.get(url, timeout=10)"
            )

        if content != original:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"  ✅ Fixed {file_path}")

def add_error_handling():
    """Add comprehensive error handling to main API files."""

    api_file = "/opt/tower-echo-brain/src/api/echo.py"

    if not os.path.exists(api_file):
        print(f"⚠️  {api_file} not found")
        return

    with open(api_file, 'r') as f:
        lines = f.readlines()

    # Add error handling wrapper
    error_handler = '''
def handle_errors(func):
    """Decorator to handle errors gracefully."""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except asyncio.TimeoutError:
            logger.error(f"Timeout in {func.__name__}")
            return {"error": "Request timeout", "status": "timeout"}
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            return {"error": str(e), "status": "error"}
    return wrapper

'''

    # Find where to insert the error handler
    import_index = 0
    for i, line in enumerate(lines):
        if line.startswith("import") or line.startswith("from"):
            import_index = i + 1

    # Insert error handler after imports
    if "handle_errors" not in "".join(lines):
        lines.insert(import_index + 1, error_handler)

    # Add decorator to all route functions
    for i, line in enumerate(lines):
        if line.strip().startswith("@router.") or line.strip().startswith("@app."):
            # Check if next line is not already a decorator
            if i + 1 < len(lines) and not lines[i + 1].strip().startswith("@"):
                # Check if the function is async
                for j in range(i + 1, min(i + 5, len(lines))):
                    if lines[j].strip().startswith("async def"):
                        lines.insert(i + 1, "@handle_errors\n")
                        break

    # Write back
    with open(api_file, 'w') as f:
        f.writelines(lines)

    print(f"✅ Added error handling to {api_file}")

def fix_ollama_timeout():
    """Specifically fix Ollama timeout issues."""

    ollama_fix = '''#!/usr/bin/env python3
"""Fix Ollama timeout issues in Echo Brain."""

import asyncio
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class OllamaManager:
    """Manage Ollama operations with proper timeout handling."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.base_url = "http://localhost:11434"

    async def list_models(self) -> Dict:
        """List available models with timeout."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.json()
        except asyncio.TimeoutError:
            logger.error(f"Model listing timed out after {self.timeout} seconds")
            return {"models": [], "error": "timeout"}
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return {"models": [], "error": str(e)}

    async def pull_model(self, model_name: str) -> Dict:
        """Pull a model with progress tracking."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=None) as client:
                # Use streaming for long operations
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/pull",
                    json={"name": model_name}
                ) as response:
                    progress = []
                    async for line in response.aiter_lines():
                        if line:
                            progress.append(line)
                            # Could emit progress updates here
                    return {"status": "success", "model": model_name}
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return {"status": "error", "error": str(e)}

    async def generate(self, model: str, prompt: str, timeout: int = 60) -> Dict:
        """Generate response with configurable timeout."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                return response.json()
        except asyncio.TimeoutError:
            logger.error(f"Generation timed out after {timeout} seconds")
            return {"error": "timeout", "response": ""}
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {"error": str(e), "response": ""}

# Global instance
ollama_manager = OllamaManager()
'''

    ollama_file = "/opt/tower-echo-brain/src/core/ollama_manager.py"
    with open(ollama_file, 'w') as f:
        f.write(ollama_fix)

    print(f"✅ Created Ollama manager with timeout handling at {ollama_file}")

if __name__ == "__main__":
    print("🔧 Fixing model timeout and error handling...")
    fix_model_listing()
    add_error_handling()
    fix_ollama_timeout()
    print("✅ All fixes applied!")