"""
Fact extraction from documents and conversations
Uses local LLM to extract structured facts
"""
import asyncio
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

import sys
sys.path.insert(0, '/opt/tower-echo-brain/src')

from services.llm_service import get_llm_service

@dataclass
class ExtractedFact:
    subject: str
    predicate: str
    object: str
    confidence: float
    source_type: str
    source_id: str
    extracted_at: datetime

EXTRACTION_PROMPT = """Extract factual information from this text. Focus on:
- Personal facts about Patrick (preferences, ownership, locations, relationships)
- Technical facts (systems, configurations, projects)
- Temporal facts (events, dates, schedules)

Text:
{text}

Return JSON array of facts. Each fact must have:
- subject: who/what the fact is about
- predicate: the relationship/property
- object: the value/target
- confidence: 0.0-1.0 how certain this fact is

Example output:
[
  {"subject": "Patrick", "predicate": "drives", "object": "2022 Toyota Tundra", "confidence": 0.95},
  {"subject": "Echo Brain", "predicate": "runs_on_port", "object": "8309", "confidence": 1.0}
]

Only return the JSON array, nothing else."""

class FactExtractor:
    def __init__(self):
        self.llm_service = get_llm_service()
        self.model = "llama3.1:8b"  # Good balance of speed and quality

    async def extract_facts(
        self,
        text: str,
        source_type: str,
        source_id: str,
        min_confidence: float = 0.7
    ) -> List[ExtractedFact]:
        """Extract facts from text using LLM"""

        if not text or len(text.strip()) < 50:
            return []

        # Truncate very long texts
        if len(text) > 4000:
            text = text[:4000] + "..."

        prompt = EXTRACTION_PROMPT.format(text=text)

        try:
            response = await self.llm_service.generate(
                prompt=prompt,
                model=self.model,
                temperature=0.1  # Low temp for consistent extraction
            )

            # Parse JSON from response
            content = response.content.strip()

            # Handle markdown code blocks
            if "```" in content:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
                if match:
                    content = match.group(1)

            facts_data = json.loads(content)

            extracted = []
            for fact in facts_data:
                if fact.get("confidence", 0) >= min_confidence:
                    extracted.append(ExtractedFact(
                        subject=fact.get("subject", "Unknown"),
                        predicate=fact.get("predicate", "unknown"),
                        object=fact.get("object", ""),
                        confidence=fact.get("confidence", 0.5),
                        source_type=source_type,
                        source_id=source_id,
                        extracted_at=datetime.utcnow()
                    ))

            return extracted

        except json.JSONDecodeError as e:
            print(f"Failed to parse facts JSON: {e}")
            return []
        except Exception as e:
            print(f"Fact extraction failed: {e}")
            return []

def get_fact_extractor() -> FactExtractor:
    return FactExtractor()