"""Coding Agent - Autonomous code generation and debugging"""
import logging
import subprocess
import tempfile
import os
import psycopg2
from psycopg2.extras import DictCursor
from typing import Dict, List, Optional
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

DB_CONFIG = {
    'host': 'localhost',
    'database': 'tower_consolidated',
    'user': 'patrick',
    'password': 'RP78eIrW7cI2jYvL5akt1yurE'
}

class CodingAgent(BaseAgent):
    """Agent specialized for code generation and debugging"""
    
    def __init__(self):
        super().__init__(
            name="CodingAgent",
            model="deepseek-coder-v2:16b"  # Best code model available
        )
        self.system_prompt = """You are an expert coding agent. You MUST:
1. READ THE PRIMARY TASK CAREFULLY and solve EXACTLY what is asked
2. Write clean, working code that directly addresses the task
3. Include error handling where appropriate
4. Add minimal comments only where necessary
5. Return ONLY code unless explanation requested

CRITICAL RULES:
- ALWAYS solve the exact task given, not a similar one
- If the task says "return True", return True, not a complex function
- If the task asks for specific functionality, implement exactly that
- Reference material is for context ONLY - do not copy unrelated solutions

When writing code:
- Use proper indentation
- Handle edge cases appropriately
- Include type hints for Python
- Keep functions focused on the specific task

When debugging:
- Identify the root cause
- Explain the issue briefly
- Provide the corrected code"""

    async def process(self, task: str, context: Dict = None) -> Dict:
        """Process a coding task"""
        context = context or {}
        
        # 1. Search for relevant past solutions
        past_solutions = self._search_past_solutions(task)
        
        # 2. Search codebase for context
        codebase_context = self._search_codebase(task)
        
        # 3. Build enhanced prompt
        enhanced_prompt = self._build_prompt(task, past_solutions, codebase_context, context)
        
        # 4. Generate code
        logger.info(f"CodingAgent processing: {task[:50]}...")
        response = await self.call_model(enhanced_prompt, self.system_prompt)
        
        # 5. Extract and validate code
        code = self._extract_code(response)
        validation = None
        if code and context.get("validate", True):
            language = context.get("language", "python")
            validation = self._validate_code(code, language)
        
        result = {
            "task": task,
            "response": response,
            "code": code,
            "validation": validation,
            "model": self.model,
            "context_used": {
                "past_solutions": len(past_solutions),
                "codebase_refs": len(codebase_context)
            }
        }
        
        # Save successful solutions
        if validation and validation.get("valid"):
            self._save_solution(task, code, context)
        
        self.add_to_history(task, {"code_length": len(code) if code else 0, "valid": validation.get("valid") if validation else None})
        return result
    
    def _search_past_solutions(self, query: str) -> List[Dict]:
        """Search past solutions for similar problems"""
        try:
            with psycopg2.connect(**DB_CONFIG) as conn:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    # Extract keywords
                    keywords = [w for w in query.lower().split() if len(w) > 3][:5]
                    if not keywords:
                        return []
                    
                    conditions = " OR ".join([
                        f"problem_description ILIKE '%{kw}%'" for kw in keywords
                    ])
                    cur.execute(f"""
                        SELECT problem_description, solution_applied, files_modified
                        FROM past_solutions
                        WHERE verified_working = TRUE AND ({conditions})
                        LIMIT 3
                    """)
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.warning(f"Past solutions search failed: {e}")
            return []
    
    def _search_codebase(self, query: str) -> List[Dict]:
        """Search codebase index for relevant code"""
        try:
            with psycopg2.connect(**DB_CONFIG) as conn:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    # Extract keywords
                    keywords = [w for w in query.lower().split() if len(w) > 3][:5]
                    if not keywords:
                        return []
                    
                    conditions = " OR ".join([
                        f"entity_name ILIKE '%{kw}%'" for kw in keywords
                    ])
                    cur.execute(f"""
                        SELECT entity_type, entity_name, file_path, signature
                        FROM codebase_index
                        WHERE {conditions}
                        LIMIT 5
                    """)
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.warning(f"Codebase search failed: {e}")
            return []
    
    def _build_prompt(self, task: str, solutions: List, codebase: List, context: Dict) -> str:
        """Build enhanced prompt with context"""
        parts = []

        # CRITICAL: Task comes FIRST to ensure it's the primary focus
        parts.append(f"## PRIMARY TASK (MUST FOLLOW EXACTLY)\n{task}")
        parts.append("\nIMPORTANT: Generate code that directly solves the above task. Reference material below is for context only.\n")

        if context.get("requirements"):
            parts.append(f"Requirements: {context['requirements']}\n")

        if context.get("language"):
            parts.append(f"Language: {context['language']}\n")

        # Only include past solutions if they closely match the task
        if solutions:
            # Filter solutions to only those that seem relevant
            task_lower = task.lower()
            relevant_solutions = []
            for s in solutions[:2]:
                desc = s.get('problem_description', '').lower()
                # Only include if there's significant overlap
                if any(word in desc for word in task_lower.split() if len(word) > 5):
                    relevant_solutions.append(s)

            if relevant_solutions:
                parts.append("\n## Reference: Similar Past Solutions")
                for s in relevant_solutions[:1]:  # Only show 1 most relevant
                    parts.append(f"Previous: {s['problem_description'][:100]}")
                    parts.append(f"Approach: {s['solution_applied'][:200]}")
                parts.append("")

        if codebase and len(codebase) > 0:
            parts.append("## Reference: Codebase Context")
            for c in codebase[:2]:  # Limit to 2 references
                parts.append(f"- {c['entity_type']}: {c['entity_name']} in {c['file_path']}")
            parts.append("")

        if context.get("files"):
            parts.append("## Files to Consider")
            for f in context["files"][:2]:
                parts.append(f"- {f}")
            parts.append("")

        parts.append("\nRemember: Focus on the PRIMARY TASK above. Use reference material only if directly relevant.")

        return "\n".join(parts)
    
    def _extract_code(self, response: str) -> Optional[str]:
        """Extract code block from response"""
        if "```" in response:
            blocks = response.split("```")
            for i, block in enumerate(blocks):
                if i % 2 == 1:  # Odd indices are code blocks
                    lines = block.strip().split("\n")
                    # Remove language identifier if present
                    if lines and lines[0].lower() in ["python", "javascript", "typescript", "bash", "sql", "js", "ts", "py"]:
                        return "\n".join(lines[1:])
                    return block.strip()
        # If no code blocks, check if entire response looks like code
        if response.strip().startswith(("def ", "class ", "import ", "from ", "async def")):
            return response.strip()
        return None
    
    def _validate_code(self, code: str, language: str) -> Dict:
        """Validate code by attempting to parse/compile it"""
        if language.lower() in ["python", "py"]:
            return self._validate_python(code)
        elif language.lower() in ["javascript", "js"]:
            return self._validate_javascript(code)
        return {"valid": True, "method": "no_validation", "message": f"No validator for {language}"}
    
    def _validate_python(self, code: str) -> Dict:
        """Validate Python code syntax"""
        try:
            compile(code, "<string>", "exec")
            return {"valid": True, "method": "compile", "message": "Syntax OK"}
        except SyntaxError as e:
            return {"valid": False, "method": "compile", "message": f"Line {e.lineno}: {e.msg}"}
        except Exception as e:
            return {"valid": False, "method": "error", "message": str(e)}
    
    def _validate_javascript(self, code: str) -> Dict:
        """Basic JavaScript validation using node"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(code)
                temp_path = f.name
            
            result = subprocess.run(
                ["node", "--check", temp_path],
                capture_output=True,
                text=True,
                timeout=5
            )
            os.unlink(temp_path)
            
            if result.returncode == 0:
                return {"valid": True, "method": "node_check", "message": "Syntax OK"}
            return {"valid": False, "method": "node_check", "message": result.stderr[:200]}
        except FileNotFoundError:
            return {"valid": True, "method": "skipped", "message": "Node not available"}
        except Exception as e:
            return {"valid": False, "method": "error", "message": str(e)}
    
    def _save_solution(self, task: str, code: str, context: Dict):
        """Save successful solution to past_solutions"""
        try:
            with psycopg2.connect(**DB_CONFIG) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO past_solutions (problem_description, solution_applied, verified_working, tags)
                        VALUES (%s, %s, TRUE, %s)
                        ON CONFLICT DO NOTHING
                    """, (task[:500], code[:2000], [context.get("language", "python")]))
                    conn.commit()
                    logger.info("Saved new solution to past_solutions")
        except Exception as e:
            logger.warning(f"Failed to save solution: {e}")

# Singleton instance
coding_agent = CodingAgent()
