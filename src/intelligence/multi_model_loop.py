"""
Multi-Model Loop for Echo Brain Improvement Tasks

Orchestrates Planner (deepseek-r1) -> Coder (qwen2.5-coder) -> Tests -> Critic (deepseek-r1)
in a synchronous loop with a hard 3-iteration cap.

Option B layout: deepseek-r1:8b handles both Planner and Critic roles (different prompts),
qwen2.5-coder:7b handles Coder role. Models are preloaded at loop entry and restored after.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Awaitable

import httpx

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"
PLANNER_MODEL = "deepseek-r1:8b"
CODER_MODEL = "qwen2.5-coder:7b"
CRITIC_MODEL = "deepseek-r1:8b"  # Same model, different prompt
MAX_ITERATIONS = 3
LLM_TIMEOUT = 300.0  # deepseek-r1 <think> phase can be slow after model swap


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ImprovementPlan:
    """Planner output: what to fix and how."""
    root_cause: str = ""
    steps: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)


@dataclass
class PatchProposal:
    """Coder output: the actual code change."""
    current_code: str = ""
    proposed_code: str = ""
    target_file: str = ""
    reasoning: str = ""
    risk: str = "medium"


@dataclass
class ReviewReport:
    """Critic output: structured review of a patch."""
    score: int = 0
    verdict: str = "reject"  # approve | revise | reject
    risks: List[str] = field(default_factory=list)
    required_changes: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class PriorProposal:
    """Summary of a previous proposal for the same issue."""
    title: str = ""
    status: str = ""
    critic_score: int = 0
    critic_verdict: str = ""
    reasoning: str = ""


@dataclass
class LoopResult:
    """Final output of the multi-model loop."""
    plan: Optional[ImprovementPlan] = None
    proposal: Optional[PatchProposal] = None
    review: Optional[ReviewReport] = None
    test_output: str = ""
    iterations: int = 0
    approved: bool = False
    needs_human_review: bool = False
    history: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TestRunner callback type
# ---------------------------------------------------------------------------
# The caller provides a callback: async (PatchProposal) -> str
# It should run relevant tests/linters and return output as a string.
# If no callback is provided, tests are skipped.
TestRunner = Callable[[PatchProposal], Awaitable[str]]


# ---------------------------------------------------------------------------
# Multi-Model Loop
# ---------------------------------------------------------------------------

class MultiModelLoop:
    """
    Synchronous improvement loop: Plan -> Code -> Test -> Critique -> Iterate.

    Usage:
        loop = MultiModelLoop()
        result = await loop.run(issue, code_context, graph_context)
    """

    def __init__(
        self,
        test_runner: Optional[TestRunner] = None,
        prior_proposals: Optional[List[PriorProposal]] = None,
        actual_file_content: Optional[str] = None,
    ):
        self.test_runner = test_runner
        self.prior_proposals = prior_proposals or []
        self.actual_file_content = actual_file_content
        self._prior_models: List[str] = []

    # -- public entry point -------------------------------------------------

    async def run(
        self,
        issue: Dict[str, Any],
        code_context: Dict[str, Any],
        graph_context: List[Dict] = None,
    ) -> LoopResult:
        """
        Run the multi-model improvement loop for a single issue.

        Returns a LoopResult with the final proposal and review.
        """
        result = LoopResult()
        graph_context = graph_context or []

        try:
            # Swap models for the loop
            await self._preload_loop_models()

            # Step 1: Plan
            plan = await self._plan(issue, code_context, graph_context)
            result.plan = plan

            if not plan.steps:
                logger.warning("Planner produced no steps, aborting loop")
                result.needs_human_review = True
                return result

            # Iterate: Code -> Test -> Critique -> maybe revise plan
            feedback = ""
            for iteration in range(1, MAX_ITERATIONS + 1):
                result.iterations = iteration
                logger.info(f"Loop iteration {iteration}/{MAX_ITERATIONS}")

                # Step 2: Code
                proposal = await self._code(issue, plan, code_context, feedback)
                result.proposal = proposal

                if not proposal.proposed_code:
                    logger.warning(f"Coder produced no code on iteration {iteration}")
                    result.history.append({
                        "iteration": iteration, "phase": "code", "outcome": "empty"
                    })
                    continue

                # Step 3: Test
                test_output = ""
                if self.test_runner:
                    try:
                        test_output = await self.test_runner(proposal)
                    except Exception as e:
                        test_output = f"Test runner error: {e}"
                        logger.error(f"Test runner failed: {e}")
                result.test_output = test_output

                # Step 4: Critique
                review = await self._critique(issue, plan, proposal, test_output)
                result.review = review

                result.history.append({
                    "iteration": iteration,
                    "score": review.score,
                    "verdict": review.verdict,
                    "risks": review.risks,
                    "required_changes": review.required_changes,
                })

                if review.verdict == "approve":
                    logger.info(f"Critic approved on iteration {iteration} (score={review.score})")
                    result.approved = True
                    return result

                if review.verdict == "reject":
                    logger.info(f"Critic rejected on iteration {iteration}: {review.notes}")
                    result.needs_human_review = True
                    return result

                # verdict == "revise": feed critic notes back as constraints
                feedback = self._build_feedback(review)
                logger.info(f"Critic requested revision: {review.notes}")

            # Exhausted iterations without approval
            logger.info(f"Loop exhausted {MAX_ITERATIONS} iterations without approval")
            result.needs_human_review = True
            return result

        except Exception as e:
            logger.error(f"Multi-model loop error: {e}", exc_info=True)
            result.needs_human_review = True
            return result

        finally:
            await self._restore_models()

    # -- Planner ------------------------------------------------------------

    async def _plan(
        self,
        issue: Dict[str, Any],
        code_context: Dict[str, Any],
        graph_context: List[Dict],
    ) -> ImprovementPlan:
        """Ask the Planner model to produce a structured improvement plan."""
        code_chunks = "\n\n".join(code_context.get("chunks", ["No code found"]))
        related_file = code_context.get("related_file", "Unknown")

        graph_section = ""
        if graph_context:
            lines = ["RELATED COMPONENTS (from knowledge graph):"]
            for item in graph_context:
                direction = "->" if item.get("direction") == "out" else "<-"
                lines.append(f"  {direction} {item.get('entity', '?')} ({item.get('predicate', '?')})")
            graph_section = "\n".join(lines) + "\n"

        prompt = f"""You are Echo Brain's planning engine. Analyze this issue and produce a structured improvement plan.

ISSUE:
Type: {issue.get('issue_type', 'unknown')}
Severity: {issue.get('severity', 'unknown')}
Title: {issue.get('title', 'No title')}
Description: {issue.get('description', 'No description')}
Stack trace: {issue.get('stack_trace') or 'None'}

RELATED CODE:
File: {related_file}
```python
{code_chunks}
```

{graph_section}
Respond with valid JSON only, no other text:
{{
  "root_cause": "<one paragraph explaining the root cause>",
  "steps": ["<step 1>", "<step 2>"],
  "constraints": ["<constraint: e.g. must not break existing callers>"],
  "acceptance_criteria": ["<criterion: e.g. tests pass>", "<criterion: e.g. no new imports>"]
}}"""

        raw = await self._call_llm(PLANNER_MODEL, prompt, temperature=0.4)
        return self._parse_plan(raw)

    # -- Coder --------------------------------------------------------------

    async def _code(
        self,
        issue: Dict[str, Any],
        plan: ImprovementPlan,
        code_context: Dict[str, Any],
        feedback: str = "",
    ) -> PatchProposal:
        """Ask the Coder model to produce a patch based on the plan."""
        code_chunks = "\n\n".join(code_context.get("chunks", ["No code found"]))
        related_file = code_context.get("related_file", "Unknown")

        feedback_section = ""
        if feedback:
            feedback_section = f"\nPREVIOUS REVIEW FEEDBACK (you must address these):\n{feedback}\n"

        prompt = f"""You are a code generation engine. Produce a minimal, correct patch for this issue.

ISSUE: {issue.get('title', 'No title')}
ROOT CAUSE: {plan.root_cause}

PLAN:
Steps: {json.dumps(plan.steps)}
Constraints: {json.dumps(plan.constraints)}
Acceptance criteria: {json.dumps(plan.acceptance_criteria)}

CURRENT CODE ({related_file}):
```python
{code_chunks}
```
{feedback_section}
Respond with valid JSON only, no other text:
{{
  "current_code": "<exact code to replace>",
  "proposed_code": "<replacement code>",
  "target_file": "{related_file}",
  "reasoning": "<why this fixes the issue>",
  "risk": "<low|medium|high>"
}}"""

        raw = await self._call_llm(CODER_MODEL, prompt, temperature=0.2)
        return self._parse_proposal(raw)

    # -- Critic -------------------------------------------------------------

    async def _critique(
        self,
        issue: Dict[str, Any],
        plan: ImprovementPlan,
        proposal: PatchProposal,
        test_output: str,
    ) -> ReviewReport:
        """Ask the Critic model to review the proposed patch."""

        # Build prior proposals section
        prior_section = ""
        if self.prior_proposals:
            lines = ["PRIOR PROPOSALS FOR THIS ISSUE (do not repeat failed approaches):"]
            for pp in self.prior_proposals:
                lines.append(
                    f"  - [{pp.status}] score={pp.critic_score}/10 verdict={pp.critic_verdict}"
                    f" | {pp.reasoning[:200]}"
                )
            prior_section = "\n".join(lines) + "\n\n"

        # Build actual file section for verification
        file_section = ""
        if self.actual_file_content:
            # Truncate to ~3000 chars to stay within context budget
            content = self.actual_file_content
            if len(content) > 3000:
                content = content[:3000] + "\n... (truncated)"
            file_section = f"""ACTUAL FILE ON DISK ({proposal.target_file}):
```python
{content}
```
IMPORTANT: Verify that "Current code" above actually exists in this file. If it does not match, the patch cannot be applied — reject it.

"""

        prompt = f"""You are Echo Brain's code critic. Review this proposed patch.

ORIGINAL ISSUE:
Title: {issue.get('title', 'No title')}
Severity: {issue.get('severity', 'unknown')}
Description: {issue.get('description', 'No description')}

{prior_section}IMPROVEMENT PLAN:
Root cause: {plan.root_cause}
Steps: {json.dumps(plan.steps)}
Acceptance criteria: {json.dumps(plan.acceptance_criteria)}

PROPOSED PATCH:
Target file: {proposal.target_file}
Risk self-assessment: {proposal.risk}
Reasoning: {proposal.reasoning}

Current code:
```python
{proposal.current_code}
```

Proposed replacement:
```python
{proposal.proposed_code}
```

{file_section}TEST RESULTS:
{test_output or 'No tests were run.'}

Review process:
1. Does the patch actually address the root cause?
2. Could it introduce regressions or break existing callers?
3. Does the "Current code" actually exist in the target file? (If actual file provided above, verify this.)
4. Are edge cases handled appropriately?
5. Is the change minimal (no unnecessary refactoring)?
6. Do test results (if any) confirm the fix?

Respond with valid JSON only, no other text:
{{
  "score": <1-10>,
  "verdict": "<approve|revise|reject>",
  "risks": ["<risk 1>"],
  "required_changes": ["<change 1>"],
  "notes": "<summary of assessment>"
}}"""

        raw = await self._call_llm(CRITIC_MODEL, prompt, temperature=0.3)
        return self._parse_review(raw)

    # -- LLM call -----------------------------------------------------------

    async def _call_llm(self, model: str, prompt: str, temperature: float = 0.3) -> str:
        """Call Ollama and return the raw response text."""
        try:
            async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": temperature},
                    },
                )

                if response.status_code != 200:
                    logger.error(f"LLM call to {model} failed: {response.status_code}")
                    return ""

                data = response.json()
                return data.get("response", "")

        except Exception as e:
            logger.error(f"LLM call to {model} error: {e}")
            return ""

    # -- Model management ---------------------------------------------------

    async def _preload_loop_models(self):
        """Pre-warm loop models on CPU. No GPU pinning — all inference on CPU.

        All generative models run on CPU (num_gpu=0) to preserve AMD VRAM
        for ComfyUI-ROCm video generation. 96GB system RAM is sufficient.
        """
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                # Snapshot current models for restore
                resp = await client.get(f"{OLLAMA_URL}/api/ps")
                if resp.status_code == 200:
                    data = resp.json()
                    loop_models = {PLANNER_MODEL, CODER_MODEL}
                    keep_models = loop_models | {"nomic-embed-text:latest", "nomic-embed-text"}
                    self._prior_models = [
                        m["name"] for m in data.get("models", [])
                        if m["name"] not in keep_models
                    ]

                # Pre-warm loop models on CPU (num_gpu=0), short keep_alive
                for model in [PLANNER_MODEL, CODER_MODEL]:
                    logger.info(f"Pre-warming {model} on CPU (num_gpu=0)")
                    await client.post(
                        f"{OLLAMA_URL}/api/generate",
                        json={"model": model, "prompt": "ready", "stream": False,
                              "options": {"num_predict": 1, "num_gpu": 0},
                              "keep_alive": "10m"},
                        timeout=60.0,
                    )

                logger.info("Loop models pre-warmed on CPU")

        except Exception as e:
            logger.warning(f"Model preload failed (will proceed anyway): {e}")

    async def _restore_models(self):
        """Set short keep_alive on loop models so they auto-evict."""
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                # Set loop models to expire after 5 minutes
                for model in [PLANNER_MODEL, CODER_MODEL]:
                    logger.info(f"Setting {model} keep_alive to 5m")
                    try:
                        await client.post(
                            f"{OLLAMA_URL}/api/generate",
                            json={"model": model, "prompt": "", "stream": False,
                                  "keep_alive": "5m", "options": {"num_gpu": 0}},
                            timeout=30.0,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to set keep_alive for {model}: {e}")

                # 2. Restore previously loaded models
                for model_name in self._prior_models:
                    logger.info(f"Restoring {model_name}")
                    try:
                        await client.post(
                            f"{OLLAMA_URL}/api/generate",
                            json={"model": model_name, "prompt": "ready",
                                  "stream": False, "options": {"num_predict": 1}},
                            timeout=60.0,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to restore {model_name}: {e}")

        except Exception as e:
            logger.warning(f"Model restore failed: {e}")

    # -- Parsing helpers ----------------------------------------------------

    def _extract_json(self, raw: str) -> Optional[Dict]:
        """Extract JSON from LLM response, handling markdown fences and thinking tags."""
        if not raw:
            return None

        import re

        # Strip <think>...</think> blocks (deepseek-r1 chain-of-thought)
        # Handle both complete and unclosed think tags
        cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        # If <think> was opened but never closed, strip everything from <think> onward
        if '<think>' in cleaned:
            cleaned = cleaned[:cleaned.index('<think>')].strip()
        # Also strip any remaining </think> orphans
        cleaned = cleaned.replace('</think>', '').strip()

        # Try direct parse first
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code fence
        fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try finding first { ... } block (greedy to get outermost braces)
        brace_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        # Last resort: find any { ... } with greedy match
        brace_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not extract JSON from LLM response ({len(raw)} chars)")
        logger.debug(f"Unparseable response preview: {cleaned[:300]}")
        return None

    def _parse_plan(self, raw: str) -> ImprovementPlan:
        data = self._extract_json(raw)
        if not data:
            return ImprovementPlan(root_cause="Failed to parse planner response")

        return ImprovementPlan(
            root_cause=data.get("root_cause", ""),
            steps=data.get("steps", []),
            constraints=data.get("constraints", []),
            acceptance_criteria=data.get("acceptance_criteria", []),
        )

    def _parse_proposal(self, raw: str) -> PatchProposal:
        data = self._extract_json(raw)
        if not data:
            return PatchProposal()

        return PatchProposal(
            current_code=data.get("current_code", ""),
            proposed_code=data.get("proposed_code", ""),
            target_file=data.get("target_file", ""),
            reasoning=data.get("reasoning", ""),
            risk=data.get("risk", "medium"),
        )

    def _parse_review(self, raw: str) -> ReviewReport:
        data = self._extract_json(raw)
        if not data:
            return ReviewReport(
                score=3, verdict="revise",
                notes="Failed to parse critic response — treating as revise, not reject",
                required_changes=["Critic output was unparseable; re-evaluate on next iteration"],
            )

        verdict = data.get("verdict", "reject")
        if verdict not in ("approve", "revise", "reject"):
            verdict = "reject"

        score = data.get("score", 0)
        if not isinstance(score, int) or score < 1 or score > 10:
            score = max(1, min(10, int(score) if isinstance(score, (int, float)) else 3))

        return ReviewReport(
            score=score,
            verdict=verdict,
            risks=data.get("risks", []),
            required_changes=data.get("required_changes", []),
            notes=data.get("notes", ""),
        )

    def _build_feedback(self, review: ReviewReport) -> str:
        """Format critic feedback as constraints for the next coder iteration."""
        lines = []
        if review.required_changes:
            lines.append("REQUIRED CHANGES:")
            for i, change in enumerate(review.required_changes, 1):
                lines.append(f"  {i}. {change}")
        if review.risks:
            lines.append("IDENTIFIED RISKS:")
            for risk in review.risks:
                lines.append(f"  - {risk}")
        if review.notes:
            lines.append(f"REVIEWER NOTES: {review.notes}")
        return "\n".join(lines)
