"""
Improvement Engine Worker - Echo Brain Phase 2a

Analyzes detected issues and proposes code fixes (REVIEW gated).
Part of the IMPROVE stage in the INGEST→THINK→IMPROVE loop.
"""

import os
import logging
from typing import Optional, Dict, List
import asyncpg
import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)


class ImprovementEngine:
    """Reasons about detected issues and proposes code fixes (REVIEW gated)"""

    def __init__(self):
        self.db_url = os.environ.get("DATABASE_URL",
            f"postgresql://patrick:{os.environ.get('DB_PASSWORD', '')}@localhost/echo_brain")
        self.qdrant_url = "http://localhost:6333"
        self.ollama_url = "http://localhost:11434"
        self.collection = "echo_memory"
        self.model = "mistral:7b"
        self.max_issues_per_cycle = 5

        # Initialize clients
        self.qdrant_client = QdrantClient(url=self.qdrant_url)

    async def run_cycle(self):
        """Main worker cycle — called by scheduler"""
        try:
            logger.info("🔧 Improvement Engine starting cycle")

            # Connect to database
            conn = await asyncpg.connect(self.db_url)

            try:
                # 1. Query open issues (critical + warning)
                issues = await self._get_open_issues(conn)

                if not issues:
                    logger.info("No open issues to analyze")
                    return

                logger.info(f"Found {len(issues)} open issues to analyze")

                analyzed = 0
                proposals_generated = 0

                for issue in issues[:self.max_issues_per_cycle]:
                    # 2. Skip issues that already have proposals
                    if await self._has_existing_proposal(conn, issue['id']):
                        logger.debug(f"Issue {issue['id']} already has proposal, skipping")
                        continue

                    analyzed += 1

                    # 3. Find related code in Qdrant
                    code_context = await self._find_related_code(issue)

                    if not code_context:
                        logger.warning(f"No related code found for issue {issue['id']}")
                        continue

                    # 4. Enrich with graph context (impact analysis)
                    graph_context = await self._get_graph_context(issue, code_context)

                    # 5-6. Run multi-model loop (Planner -> Coder -> Critic)
                    analysis, loop_meta = await self._run_multi_model_loop(
                        issue, code_context, graph_context, conn=conn
                    )

                    if not analysis:
                        logger.error(f"Failed to analyze issue {issue['id']}")
                        continue

                    # 7. Store proposal in self_improvement_proposals
                    proposal_id = await self._store_proposal(
                        conn, issue, analysis, loop_meta=loop_meta
                    )

                    if proposal_id:
                        proposals_generated += 1

                        # 8. Create notification for Patrick
                        await self._create_notification(conn, issue, analysis, proposal_id)

                # 9. Log summary
                logger.info(f"✅ Improvement Engine cycle complete: "
                          f"Analyzed {analyzed} issues, generated {proposals_generated} proposals")

            finally:
                await conn.close()

        except Exception as e:
            logger.error(f"❌ Improvement Engine error: {e}", exc_info=True)
            raise

    async def _build_test_runner(self, target_file: str):
        """Build a test runner callback that validates proposed patches.

        Checks:
        1. Proposed code parses (ast.parse) — catches syntax errors
        2. current_code exists in the target file — catches phantom patches
        """
        import ast

        async def test_runner(proposal) -> str:
            results = []

            # 1. Syntax check on proposed code
            if proposal.proposed_code:
                try:
                    ast.parse(proposal.proposed_code)
                    results.append("PASS: proposed_code parses as valid Python")
                except SyntaxError as e:
                    results.append(f"FAIL: proposed_code has syntax error: {e}")

            # 2. Verify current_code exists in the actual file
            if proposal.current_code and target_file and os.path.isfile(target_file):
                try:
                    with open(target_file, 'r') as f:
                        file_content = f.read()
                    if proposal.current_code.strip() in file_content:
                        results.append("PASS: current_code found in target file")
                    else:
                        results.append(
                            "FAIL: current_code NOT found in target file — "
                            "patch cannot be applied"
                        )
                except Exception as e:
                    results.append(f"WARN: could not read target file: {e}")

            # 3. If we can, simulate the replacement and parse the full file
            if (proposal.current_code and proposal.proposed_code
                    and target_file and os.path.isfile(target_file)):
                try:
                    with open(target_file, 'r') as f:
                        original = f.read()
                    if proposal.current_code.strip() in original:
                        patched = original.replace(
                            proposal.current_code.strip(),
                            proposal.proposed_code.strip(),
                            1,
                        )
                        ast.parse(patched)
                        results.append("PASS: full file parses after applying patch")
                except SyntaxError as e:
                    results.append(f"FAIL: full file has syntax error after patch: {e}")
                except Exception:
                    pass

            return "\n".join(results) if results else "No tests were run."

        return test_runner

    async def _get_prior_proposals(self, conn, issue_id) -> list:
        """Fetch prior proposals for the same issue to prevent repeated failures."""
        try:
            from src.intelligence.multi_model_loop import PriorProposal

            rows = await conn.fetch("""
                SELECT title, status, critic_score, critic_verdict, reasoning
                FROM self_improvement_proposals
                WHERE issue_id = $1
                ORDER BY created_at DESC
                LIMIT 5
            """, issue_id)

            return [
                PriorProposal(
                    title=row['title'],
                    status=row['status'],
                    critic_score=row['critic_score'] or 0,
                    critic_verdict=row['critic_verdict'] or '',
                    reasoning=(row['reasoning'] or '')[:300],
                )
                for row in rows
            ]
        except Exception as e:
            logger.debug(f"Could not fetch prior proposals: {e}")
            return []

    async def _read_actual_file(self, code_context: Dict) -> Optional[str]:
        """Read the actual target file from disk for critic verification."""
        target_file = code_context.get('related_file')
        if not target_file or not os.path.isfile(target_file):
            return None
        try:
            with open(target_file, 'r') as f:
                return f.read()
        except Exception as e:
            logger.debug(f"Could not read target file {target_file}: {e}")
            return None

    async def _run_multi_model_loop(
        self, issue: Dict, code_context: Dict, graph_context: List[Dict],
        conn=None,
    ) -> tuple:
        """Run the multi-model Planner->Coder->Critic loop for an issue.

        Returns (analysis_dict, loop_meta_dict) compatible with _store_proposal.
        Falls back to the legacy single-LLM path if the loop fails entirely.
        """
        try:
            from src.intelligence.multi_model_loop import MultiModelLoop

            # Build enriched context for the critic
            target_file = code_context.get('related_file', '')
            test_runner = await self._build_test_runner(target_file)
            prior_proposals = await self._get_prior_proposals(conn, issue['id']) if conn else []
            actual_file_content = await self._read_actual_file(code_context)

            loop = MultiModelLoop(
                test_runner=test_runner,
                prior_proposals=prior_proposals,
                actual_file_content=actual_file_content,
            )
            result = await loop.run(issue, code_context, graph_context)

            loop_meta = {
                "iterations": result.iterations,
                "critic_score": result.review.score if result.review else 0,
                "critic_verdict": result.review.verdict if result.review else "",
                "approved": result.approved,
                "needs_human_review": result.needs_human_review,
            }

            if result.proposal and (result.proposal.proposed_code or result.plan):
                analysis = {
                    "root_cause": result.plan.root_cause if result.plan else "",
                    "current_code": result.proposal.current_code if result.proposal else "",
                    "proposed_code": result.proposal.proposed_code if result.proposal else "",
                    "reasoning": result.proposal.reasoning if result.proposal else "",
                    "risk": result.proposal.risk if result.proposal else "medium",
                }

                # Append critic notes to reasoning for visibility
                if result.review and result.review.notes:
                    analysis["reasoning"] += f"\n\nCritic ({result.review.score}/10): {result.review.notes}"
                if result.review and result.review.risks:
                    analysis["reasoning"] += f"\nRisks: {', '.join(result.review.risks)}"

                logger.info(
                    f"Multi-model loop completed: {result.iterations} iterations, "
                    f"score={loop_meta['critic_score']}, verdict={loop_meta['critic_verdict']}"
                )

                # Write result to knowledge graph
                await self._store_loop_graph_edge(issue, loop_meta)

                return analysis, loop_meta

            logger.warning("Multi-model loop produced no usable proposal, falling back to single-LLM")

        except Exception as e:
            logger.error(f"Multi-model loop failed, falling back to single-LLM: {e}", exc_info=True)

        # Fallback: legacy single-LLM analysis
        prompt = self._build_analysis_prompt(issue, code_context, graph_context)
        analysis = await self._analyze_with_llm(prompt)
        loop_meta = {"iterations": 0, "critic_score": 0, "critic_verdict": "legacy"}
        return analysis, loop_meta

    async def _store_loop_graph_edge(self, issue: Dict, loop_meta: Dict):
        """Write multi-model loop result as facts for the knowledge graph.

        Inserts into the `facts` table which the GraphEngine reads on refresh.
        """
        try:
            conn = await asyncpg.connect(self.db_url)
            try:
                subject = issue.get("related_file") or issue.get("related_worker") or "unknown_target"
                verdict = loop_meta.get("critic_verdict", "unknown")
                score = loop_meta.get("critic_score", 0)
                iterations = loop_meta.get("iterations", 0)

                await conn.execute(
                    "INSERT INTO facts (subject, predicate, object, confidence) VALUES ($1, $2, $3, $4)",
                    subject,
                    f"reviewed_by_critic_{verdict}",
                    f"multi_model_loop (score={score}/10, iters={iterations})",
                    min(score / 10.0, 1.0),
                )

                await conn.execute(
                    "INSERT INTO facts (subject, predicate, object, confidence) VALUES ($1, $2, $3, $4)",
                    issue.get("title", "unknown_issue"),
                    "produced_proposal",
                    f"verdict={verdict}, score={score}/10",
                    min(score / 10.0, 1.0),
                )
            finally:
                await conn.close()

        except Exception as e:
            logger.debug(f"Failed to write loop result to graph: {e}")

    async def _get_open_issues(self, conn) -> List[Dict]:
        """Get open critical and warning issues"""
        query = """
            SELECT id, issue_type, severity, source, title, description,
                   related_file, related_worker, stack_trace
            FROM self_detected_issues
            WHERE status = 'open'
            AND severity IN ('critical', 'warning')
            ORDER BY
                CASE severity
                    WHEN 'critical' THEN 1
                    WHEN 'warning' THEN 2
                    ELSE 3
                END,
                created_at DESC
            LIMIT $1
        """

        rows = await conn.fetch(query, self.max_issues_per_cycle * 2)  # Get extra to account for existing proposals
        return [dict(row) for row in rows]

    async def _has_existing_proposal(self, conn, issue_id: str) -> bool:
        """Check if a proposal already exists for this issue"""
        query = """
            SELECT EXISTS(
                SELECT 1 FROM self_improvement_proposals
                WHERE issue_id = $1
            )
        """
        return await conn.fetchval(query, issue_id)

    async def _find_related_code(self, issue: Dict) -> Optional[Dict]:
        """Find code related to the issue using Qdrant"""
        try:
            # If we have a specific file, search for it
            if issue.get('related_file'):
                search_query = f"{issue['related_file']} {issue['title']}"
                filter_condition = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value="self_codebase")
                        ),
                        models.FieldCondition(
                            key="file_path",
                            match=models.MatchValue(value=issue['related_file'])
                        )
                    ]
                )
            else:
                # Search broadly using issue title and description
                search_query = f"{issue['title']} {issue.get('description', '')}"
                filter_condition = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value="self_codebase")
                        )
                    ]
                )

            # Get embeddings for the search query
            embedding = await self._get_embedding(search_query)

            if not embedding:
                return None

            # Search Qdrant (query_points replaces deprecated .search)
            results = self.qdrant_client.query_points(
                collection_name=self.collection,
                query=embedding,
                query_filter=filter_condition,
                limit=3,
                with_payload=True,
            )

            if not results.points:
                return None

            # Extract code chunks and metadata
            code_chunks = []
            file_paths = set()

            for hit in results.points:
                payload = hit.payload or {}
                code_chunks.append(payload.get('text', ''))
                if 'file_path' in payload:
                    file_paths.add(payload['file_path'])

            return {
                'chunks': code_chunks,
                'file_paths': list(file_paths),
                'related_file': issue.get('related_file') or (list(file_paths)[0] if file_paths else None)
            }

        except Exception as e:
            logger.error(f"Error finding related code: {e}")
            return None

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding vector for text using Ollama"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={
                        "model": "nomic-embed-text",
                        "prompt": text
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    embedding = data.get("embedding", [])
                    return embedding if embedding else None
                else:
                    logger.error(f"Embedding failed: {response.status_code}")
                    return None

        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None

    async def _get_graph_context(self, issue: Dict, code_context: Dict) -> List[Dict]:
        """Query the knowledge graph for entities related to the issue.

        Extracts entity names from the issue title, related file, and code
        context, then traverses 1-hop in the graph to find connected components.
        This powers impact analysis in the LLM prompt.
        """
        try:
            from src.core.graph_engine import get_graph_engine
            engine = get_graph_engine()
            await engine._ensure_loaded()
            if not engine._graph:
                return []

            # Extract candidate entities from issue + code
            candidates = set()
            title = issue.get("title", "")
            for word in title.split():
                clean = word.strip(".:;,()").lower()
                if len(clean) > 3 and clean not in ("error", "failed", "issue", "warning"):
                    candidates.add(clean)
            if issue.get("related_file"):
                # Use filename stem as entity
                fname = issue["related_file"].rsplit("/", 1)[-1].replace(".py", "")
                candidates.add(fname.lower())
            if issue.get("related_worker"):
                candidates.add(issue["related_worker"].lower())

            # Query graph for each candidate
            all_connected = []
            seen = set()
            for entity in list(candidates)[:5]:
                connected = engine.get_connected_entities(entity, max_hops=1)
                for item in connected:
                    key = (item["entity"], item["predicate"])
                    if key not in seen:
                        seen.add(key)
                        all_connected.append(item)

            return all_connected[:15]
        except Exception as e:
            logger.debug(f"Graph context unavailable: {e}")
            return []

    def _format_graph_section(self, graph_context: List[Dict] = None) -> str:
        """Format graph context into a prompt section."""
        if not graph_context:
            return ""
        lines = ["RELATED COMPONENTS (from knowledge graph):"]
        for item in graph_context:
            direction = "->" if item["direction"] == "out" else "<-"
            lines.append(f"  {direction} {item['entity']} ({item['predicate']})")
        lines.append("")
        return "\n".join(lines) + "\n"

    def _build_analysis_prompt(self, issue: Dict, code_context: Dict, graph_context: List[Dict] = None) -> str:
        """Build the analysis prompt for LLM"""
        code_chunks = "\n\n".join(code_context.get('chunks', ['No code found']))
        related_file = code_context.get('related_file', 'Unknown')

        prompt = f"""You are Echo Brain's self-improvement system.

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

{self._format_graph_section(graph_context)}TASK:
1. What is the root cause of this issue?
2. What specific code change would fix it?
3. Show the EXACT current code that needs changing and the EXACT replacement code.
4. What is the risk of this change? (low/medium/high)
5. Could this change break anything else?

Respond in this exact format:
ROOT_CAUSE: <one paragraph explanation>
RISK: <low|medium|high>
CURRENT_CODE:
```python
<exact code to replace>
```
PROPOSED_CODE:
```python
<replacement code>
```
REASONING: <why this fix works and what to watch for>
"""
        return prompt

    async def _analyze_with_llm(self, prompt: str) -> Optional[Dict]:
        """Send prompt to LLM and parse response"""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0.7
                    }
                )

                if response.status_code != 200:
                    logger.error(f"LLM request failed: {response.status_code}")
                    return None

                data = response.json()
                llm_response = data.get('response', '')

                # Parse the response
                analysis = self._parse_llm_response(llm_response)
                return analysis

        except Exception as e:
            logger.error(f"Error analyzing with LLM: {e}")
            return None

    def _parse_llm_response(self, response: str) -> Dict:
        """Parse structured response from LLM"""
        try:
            analysis = {
                'root_cause': '',
                'risk': 'medium',
                'current_code': '',
                'proposed_code': '',
                'reasoning': ''
            }

            # Extract ROOT_CAUSE
            if 'ROOT_CAUSE:' in response:
                start = response.index('ROOT_CAUSE:') + len('ROOT_CAUSE:')
                end = response.index('RISK:') if 'RISK:' in response else len(response)
                analysis['root_cause'] = response[start:end].strip()

            # Extract RISK
            if 'RISK:' in response:
                start = response.index('RISK:') + len('RISK:')
                end = response.index('CURRENT_CODE:') if 'CURRENT_CODE:' in response else len(response)
                risk_text = response[start:end].strip().lower()
                if 'low' in risk_text:
                    analysis['risk'] = 'low'
                elif 'high' in risk_text:
                    analysis['risk'] = 'high'
                else:
                    analysis['risk'] = 'medium'

            # Extract CURRENT_CODE
            if 'CURRENT_CODE:' in response and '```' in response[response.index('CURRENT_CODE:'):]:
                start = response.index('CURRENT_CODE:')
                code_start = response.index('```', start) + 3
                if 'python' in response[code_start:code_start+10]:
                    code_start = response.index('\n', code_start) + 1
                code_end = response.index('```', code_start)
                analysis['current_code'] = response[code_start:code_end].strip()

            # Extract PROPOSED_CODE
            if 'PROPOSED_CODE:' in response and '```' in response[response.index('PROPOSED_CODE:'):]:
                start = response.index('PROPOSED_CODE:')
                code_start = response.index('```', start) + 3
                if 'python' in response[code_start:code_start+10]:
                    code_start = response.index('\n', code_start) + 1
                code_end = response.index('```', code_start)
                analysis['proposed_code'] = response[code_start:code_end].strip()

            # Extract REASONING
            if 'REASONING:' in response:
                start = response.index('REASONING:') + len('REASONING:')
                analysis['reasoning'] = response[start:].strip()

            return analysis

        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return {
                'root_cause': 'Failed to parse LLM response',
                'risk': 'high',
                'current_code': '',
                'proposed_code': '',
                'reasoning': str(e)
            }

    async def _store_proposal(self, conn, issue: Dict, analysis: Dict,
                              loop_meta: Dict = None) -> Optional[str]:
        """Store improvement proposal in database and vectorize to Qdrant."""
        try:
            loop_meta = loop_meta or {}
            query = """
                INSERT INTO self_improvement_proposals (
                    issue_id, title, description, target_file,
                    current_code, proposed_code, reasoning,
                    risk_assessment, status,
                    loop_iterations, critic_score, critic_verdict
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                RETURNING id
            """

            title = f"Fix: {issue['title']}"
            target_file = issue.get('related_file', 'Unknown')

            # Set status based on critic verdict
            status = 'pending'
            if loop_meta.get('approved'):
                status = 'critic_approved'
            elif loop_meta.get('needs_human_review'):
                status = 'needs_review'

            proposal_id = await conn.fetchval(
                query,
                issue['id'],
                title,
                analysis['root_cause'],
                target_file,
                analysis['current_code'],
                analysis['proposed_code'],
                analysis['reasoning'],
                analysis['risk'],
                status,
                loop_meta.get('iterations', 0),
                loop_meta.get('critic_score', 0),
                loop_meta.get('critic_verdict', ''),
            )

            logger.info(f"Created proposal {proposal_id} for issue {issue['id']}")

            # Vectorize proposal to Qdrant for future semantic retrieval
            await self._vectorize_proposal(str(proposal_id), title, issue, analysis, target_file)

            return str(proposal_id)

        except Exception as e:
            logger.error(f"Error storing proposal: {e}")
            return None

    async def _vectorize_proposal(self, proposal_id: str, title: str,
                                  issue: Dict, analysis: Dict, target_file: str):
        """Embed and store proposal in Qdrant so it's discoverable via semantic search."""
        try:
            # Build a rich text representation for embedding
            text = (
                f"Improvement Proposal: {title}\n"
                f"Issue: {issue.get('title', '')}\n"
                f"Severity: {issue.get('severity', 'unknown')}\n"
                f"Target: {target_file}\n"
                f"Root cause: {analysis.get('root_cause', '')}\n"
                f"Risk: {analysis.get('risk', 'medium')}\n"
                f"Fix: {analysis.get('reasoning', '')}"
            )

            embedding = await self._get_embedding(text)
            if not embedding:
                return

            import uuid
            from datetime import datetime
            point_id = str(uuid.uuid4())
            payload = {
                "type": "improvement_proposal",
                "source": "improvement_engine",
                "proposal_id": proposal_id,
                "issue_type": issue.get("issue_type", ""),
                "severity": issue.get("severity", ""),
                "risk": analysis.get("risk", "medium"),
                "target_file": target_file,
                "text": text[:5000],
                "created_at": datetime.now().isoformat(),
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.put(
                    f"{self.qdrant_url}/collections/{self.collection}/points",
                    json={"points": [{"id": point_id, "vector": embedding, "payload": payload}]},
                )
                if resp.status_code in (200, 201):
                    logger.info(f"Vectorized proposal {proposal_id} to Qdrant")
                else:
                    logger.warning(f"Failed to vectorize proposal: {resp.status_code}")

        except Exception as e:
            logger.warning(f"Error vectorizing proposal {proposal_id}: {e}")

    async def _create_notification(self, conn, issue: Dict, analysis: Dict, proposal_id: str):
        """Create notification for Patrick about new proposal"""
        try:
            query = """
                INSERT INTO autonomous_notifications (
                    message, priority, source
                ) VALUES ($1, $2, $3)
            """

            message = (
                f"New improvement proposal: Fix for '{issue['title']}' "
                f"(risk: {analysis['risk']}). "
                f"Review at: /api/echo/proposals/{proposal_id}"
            )

            priority = 'high' if analysis['risk'] == 'high' else 'medium'

            await conn.execute(query, message, priority, 'improvement_engine')
            logger.debug(f"Created notification for proposal {proposal_id}")

        except Exception as e:
            logger.error(f"Error creating notification: {e}")