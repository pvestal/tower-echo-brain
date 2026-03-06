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

                    # 5. Build analysis prompt with issue + code + graph context
                    prompt = self._build_analysis_prompt(issue, code_context, graph_context)

                    # 6. Send to Ollama, parse response
                    analysis = await self._analyze_with_llm(prompt)

                    if not analysis:
                        logger.error(f"Failed to analyze issue {issue['id']}")
                        continue

                    # 7. Store proposal in self_improvement_proposals
                    proposal_id = await self._store_proposal(conn, issue, analysis)

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

    async def _store_proposal(self, conn, issue: Dict, analysis: Dict) -> Optional[str]:
        """Store improvement proposal in database and vectorize to Qdrant."""
        try:
            query = """
                INSERT INTO self_improvement_proposals (
                    issue_id, title, description, target_file,
                    current_code, proposed_code, reasoning,
                    risk_assessment, status
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
            """

            title = f"Fix: {issue['title']}"
            target_file = issue.get('related_file', 'Unknown')

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
                'pending'
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