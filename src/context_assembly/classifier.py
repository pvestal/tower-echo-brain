"""
Domain Classifier - Routes queries to appropriate knowledge domains
Prevents cross-domain contamination (e.g., anime content in math queries)
"""
import re
import logging
from enum import Enum
from typing import List, Tuple, Dict

logger = logging.getLogger("echo.context_assembly.classifier")


class Domain(str, Enum):
    """Knowledge domains with strict boundaries"""
    TECHNICAL = "technical"      # Code, APIs, debugging, architecture
    ANIME = "anime"              # Anime projects, characters, ComfyUI generation
    PERSONAL = "personal"        # Patrick's info, preferences, equipment
    SYSTEM = "system"            # Tower services, health, monitoring
    GENERAL = "general"          # Math, facts, general knowledge
    CREATIVE = "creative"        # Story generation, narrative
    FINANCIAL = "financial"      # Plaid, banking, transactions


class DomainClassifier:
    """
    Classifies queries into domains to prevent contamination.
    Each domain has:
    - Allowed Qdrant collections
    - Allowed PostgreSQL tables
    - Confidence threshold for inclusion
    """

    def __init__(self):
        self.domain_rules = self._initialize_rules()
        self.domain_sources = self._initialize_sources()

    def _initialize_rules(self) -> Dict[Domain, Dict]:
        """Define classification rules for each domain"""
        return {
            Domain.TECHNICAL: {
                "signals": [
                    r"\b(code|function|class|def|async|api|endpoint|bug|error|debug)\b",
                    r"\b(python|javascript|typescript|sql|git|docker|kubernetes)\b",
                    r"\b(implement|refactor|optimize|test|deploy)\b",
                    r"\b(pipeline|context|layer|module|import|package)\b"
                ],
                "negative_signals": [r"\b(anime|character|scene|mei|goblin)\b"],
                "weight": 1.0
            },
            Domain.ANIME: {
                "signals": [
                    r"\b(anime|manga|episode|scene|character)\b",
                    r"\b(tokyo[\s_]debt|goblin[\s_]slayer|mei|cyberpunk[\s_]goblins)\b",
                    r"\b(comfyui|lora|generation|storyboard|frame)\b",
                    r"\b(narrat|story|plot|protagonist)\b"
                ],
                "negative_signals": [r"\b(code|function|api|debug|sql)\b"],
                "weight": 0.9
            },
            Domain.PERSONAL: {
                "signals": [
                    r"\b(patrick|my|i|me)\b",
                    r"\b(truck|tundra|toyota|rv|vehicle)\b",
                    r"\b(tower[\s_]server|echo[\s_]brain)\b",
                    r"\b(preference|remember|know about me)\b"
                ],
                "negative_signals": [],
                "weight": 0.95
            },
            Domain.SYSTEM: {
                "signals": [
                    r"\b(service|health|status|monitor|running)\b",
                    r"\b(systemctl|restart|logs|nginx|port)\b",
                    r"\b(ollama|qdrant|postgresql|comfyui)\b",
                    r"\b(tower-.*|localhost:\d+)\b"
                ],
                "negative_signals": [r"\b(anime|character)\b"],
                "weight": 0.85
            },
            Domain.GENERAL: {
                "signals": [
                    r"\b(what is|how many|when|where|define)\b",
                    r"\b(\d+\s*[\+\-\*\/]\s*\d+)\b",  # Math operations
                    r"\b(calculate|compute|solve)\b",
                    r"\b(fact|information|explain)\b"
                ],
                "negative_signals": [r"\b(anime|patrick|tower|code)\b"],
                "weight": 0.7
            },
            Domain.CREATIVE: {
                "signals": [
                    r"\b(create|generate|write|imagine|describe)\b",
                    r"\b(story|narrative|scene|description)\b",
                    r"\b(creative|artistic|design)\b"
                ],
                "negative_signals": [r"\b(code|debug|api|sql)\b"],
                "weight": 0.8
            },
            Domain.FINANCIAL: {
                "signals": [
                    r"\b(bank|money|balance|transaction|payment)\b",
                    r"\b(plaid|stripe|paypal|credit|debit)\b",
                    r"\b(expense|income|budget|financial)\b"
                ],
                "negative_signals": [],
                "weight": 0.9
            }
        }

    def _initialize_sources(self) -> Dict[Domain, Dict]:
        """Map domains to allowed data sources"""
        return {
            Domain.TECHNICAL: {
                "qdrant_collections": ["code", "documents"],
                "pg_tables": ["claude_conversations"],  # Tech discussions only
                "facts_filter": lambda f: "code" in f or "api" in f,
                "max_sources": 10,
                "min_score": 0.4
            },
            Domain.ANIME: {
                "qdrant_collections": ["anime_characters", "scene_embeddings"],
                "pg_tables": ["anime_production.projects", "anime_production.characters"],
                "facts_filter": lambda f: "anime" in f or "character" in f,
                "max_sources": 15,
                "min_score": 0.3
            },
            Domain.PERSONAL: {
                "qdrant_collections": ["echo_memory", "conversations"],
                "pg_tables": ["claude_conversations", "facts"],
                "facts_filter": lambda f: "patrick" in f.lower(),
                "max_sources": 10,
                "min_score": 0.35
            },
            Domain.SYSTEM: {
                "qdrant_collections": ["documents"],
                "pg_tables": ["service_health", "system_logs"],
                "facts_filter": lambda f: "service" in f or "tower" in f,
                "max_sources": 5,
                "min_score": 0.4
            },
            Domain.GENERAL: {
                "qdrant_collections": [],  # No vector search for general
                "pg_tables": [],  # No conversation search
                "facts_filter": lambda f: True,  # Allow all facts
                "max_sources": 3,
                "min_score": 0.5
            },
            Domain.CREATIVE: {
                "qdrant_collections": ["echo_memory", "scene_embeddings"],
                "pg_tables": ["claude_conversations"],
                "facts_filter": lambda f: "story" in f or "narrative" in f,
                "max_sources": 8,
                "min_score": 0.35
            },
            Domain.FINANCIAL: {
                "qdrant_collections": ["documents"],
                "pg_tables": ["transactions", "financial_data"],
                "facts_filter": lambda f: "financial" in f or "money" in f,
                "max_sources": 5,
                "min_score": 0.45
            }
        }

    def classify(self, query: str) -> List[Tuple[Domain, float]]:
        """
        Classify query into domains with confidence scores.
        Returns list of (domain, confidence) tuples sorted by confidence.

        This is the KEY to preventing contamination - by strictly
        classifying queries, we only search relevant sources.
        """
        query_lower = query.lower()
        domain_scores = {}

        for domain, rules in self.domain_rules.items():
            score = 0.0

            # Check positive signals
            for signal_pattern in rules["signals"]:
                if re.search(signal_pattern, query_lower):
                    score += rules["weight"]

            # Check negative signals (reduce score)
            for neg_pattern in rules["negative_signals"]:
                if re.search(neg_pattern, query_lower):
                    score -= rules["weight"] * 0.5

            # Normalize score
            if score > 0:
                domain_scores[domain] = min(1.0, score)

        # If no domain matched strongly, default to GENERAL
        if not domain_scores or max(domain_scores.values()) < 0.3:
            domain_scores[Domain.GENERAL] = 0.5

        # Sort by confidence
        results = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)

        # Log classification for debugging
        logger.info(f"Query: '{query[:50]}...' classified as: {results[:2]}")

        return results

    def get_allowed_sources(self, domain: Domain) -> Dict:
        """Get allowed data sources for a domain"""
        return self.domain_sources.get(domain, self.domain_sources[Domain.GENERAL])

    def should_search_source(self, domain: Domain, source_type: str, source_name: str) -> bool:
        """
        Determine if a specific source should be searched for this domain.
        This is the enforcement point for preventing contamination.
        """
        sources = self.get_allowed_sources(domain)

        if source_type == "qdrant":
            return source_name in sources.get("qdrant_collections", [])
        elif source_type == "postgresql":
            return source_name in sources.get("pg_tables", [])
        elif source_type == "facts":
            return True  # Facts are filtered by content, not source

        return False

    def filter_results_by_domain(self, results: List[Dict], domain: Domain) -> List[Dict]:
        """
        Post-filter results to ensure they match the domain.
        Last line of defense against contamination.
        """
        sources = self.get_allowed_sources(domain)
        min_score = sources.get("min_score", 0.3)
        max_sources = sources.get("max_sources", 10)

        filtered = []
        for result in results:
            # Check minimum score
            if result.get("score", 0) < min_score:
                continue

            # For facts, apply content filter
            if result.get("type") == "fact":
                fact_filter = sources.get("facts_filter", lambda f: True)
                content = result.get("content", "").lower()
                if not fact_filter(content):
                    continue

            filtered.append(result)

            if len(filtered) >= max_sources:
                break

        if len(filtered) < len(results):
            logger.info(f"Filtered {len(results)} -> {len(filtered)} results for domain {domain}")

        return filtered