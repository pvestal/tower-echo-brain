#!/usr/bin/env python3
"""
Deep codebase review using local Ollama models.
Run: python scripts/deep_review.py /path/to/review
"""
import sys
import os
import httpx
import json
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
MODEL = "deepseek-coder-v2:16b"

def review_file(filepath: str) -> dict:
    """Review a single file using DeepSeek."""
    with open(filepath) as f:
        content = f.read()
    
    prompt = f"""You are reviewing Python code. Be critical and specific.

File: {filepath}
```python
{content[:8000]}  # Truncate for context window
```

Analyze for:
1. Dead code (unused functions, unreachable branches)
2. Security issues (hardcoded secrets, SQL injection, etc)
3. Performance problems (N+1 queries, blocking calls, memory leaks)
4. Missing error handling
5. Code that claims to work but doesn't (theater code)

For each issue found, provide:
- Line number
- Problem description
- Suggested fix

Be direct. No praise. Only problems."""

    response = httpx.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=300.0
    )
    return {
        "file": filepath,
        "review": response.json().get("response", ""),
        "model": MODEL
    }

def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "/opt/tower-echo-brain/src"
    target_path = Path(target)
    
    if target_path.is_file():
        files = [target_path]
    else:
        files = list(target_path.rglob("*.py"))
    
    results = []
    for f in files[:20]:  # Limit to avoid timeout
        print(f"Reviewing: {f}")
        result = review_file(str(f))
        results.append(result)
        print(result["review"][:500] + "...\n")
    
    # Save full results
    with open("/tmp/deep_review_results.json", "w") as out:
        json.dump(results, out, indent=2)
    print(f"\nFull results saved to /tmp/deep_review_results.json")

if __name__ == "__main__":
    main()
