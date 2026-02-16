"""
Domain Classifier - Routes queries to appropriate knowledge domains
Prevents cross-domain contamination (e.g., anime content in math queries)
Hybrid: fast regex path + embedding cosine similarity for ambiguous queries
"""
import math
import re
import logging
from enum import Enum
from typing import List, Tuple, Dict, Optional

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


# Representative phrases per domain. When regex is ambiguous, the query embedding
# is compared against embeddings of these exemplars via cosine similarity.
# Cheap (~0ms once cached) and doesn't require loading a separate LLM.
DOMAIN_EXEMPLARS: Dict[Domain, List[str]] = {
    Domain.TECHNICAL: [
        "How does the Python async function work",
        "Debug the API endpoint error in the code",
        "Refactor the class to use dependency injection",
        "What does this TypeScript import do",
    ],
    Domain.ANIME: [
        "What checkpoint does the Mario Galaxy anime project use",
        "Generate training images for the character LoRA",
        "ComfyUI sampler settings for anime generation",
        "Which characters are in the Tokyo Debt project",
    ],
    Domain.PERSONAL: [
        "What is Patrick's truck model",
        "Tell me about Patrick's preferences",
        "What equipment does Patrick have",
        "Who is Patrick and what does he work on",
    ],
    Domain.SYSTEM: [
        "What services are running on the Tower server",
        "Check the Echo Brain health status",
        "What port does Qdrant listen on",
        "Restart the nginx service on Tower",
    ],
    Domain.GENERAL: [
        "What is the capital of France",
        "Calculate 15 times 23",
        "Define the word entropy",
        "How many planets are in the solar system",
    ],
    Domain.CREATIVE: [
        "Write a short story about a robot",
        "Generate a creative narrative description",
        "Imagine a scene in a fantasy world",
    ],
    Domain.FINANCIAL: [
        "Check my bank account balance via Plaid",
        "Show recent credit card transactions",
        "What are my monthly expenses",
    ],
}


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
        # Lazy-loaded: domain exemplar embeddings (computed once, cached forever)
        self._exemplar_embeddings: Optional[Dict[Domain, List[List[float]]]] = None
        self._exemplar_loading = False

    def _initialize_rules(self) -> Dict[Domain, Dict]:
        """Define classification rules for each domain"""
        return {
            Domain.TECHNICAL: {
                "signals": [
                    r"\b(code|function|class|def|async|api|endpoint|bug|error|debug)\b",
                    r"\b(python|javascript|typescript|sql|git|docker|kubernetes)\b",
                    r"\b(implement|refactor|optimize|test|deploy|how|work|works)\b",
                    r"\b(pipeline|context|assembly|classifier|layer|module|import|package)\b",
                    r"\b(domain|enum|regex|signals|retriever)\b"
                ],
                "negative_signals": [r"\b(anime|character|scene|mei|goblin)\b"],
                "weight": 1.5  # Increase weight for technical queries
            },
            Domain.ANIME: {
                "signals": [
                    r"\b(anime|manga|episode|scene|character)\b",
                    r"\b(tokyo[\s_]debt|goblin[\s_]slayer|mei|cyberpunk[\s_]goblins|mario[\s_]galaxy)\b",
                    r"\b(comfyui|lora|generation|storyboard|frame)\b",
                    r"\b(checkpoint|safetensors|cfg[\s_]scale|sampler|negative[\s_]prompt)\b",
                    r"\b(cyberrealistic|realistic[\s_]vision|realcartoon|chilloutmix)\b",
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
                    r"\b(service|services|health|status|monitor|running|run|runs)\b",
                    r"\b(systemctl|restart|logs|nginx|port|ports)\b",
                    r"\b(ollama|qdrant|postgresql|comfyui)\b",
                    r"\b(tower-.*|localhost:\d+|Tower)\b",
                    # Echo Brain specific keywords
                    r"\b(echo[\s_]brain|embedding[\s_]model|agent[\s_]types?|frontend[\s_]stack)\b",
                    r"\b(coding[\s_]agent|reasoning[\s_]agent|narration[\s_]agent)\b",
                    r"\b(nomic[\s-]embed[\s-]text|deepseek|gemma|mistral)\b",
                    r"\b(architecture|modules|directories|retriever|ingestion)\b",
                    r"\b(context[\s_]assembly|model[\s_]routing|authoritative)\b",
                    r"\b(vue|typescript|tailwind|react)\b",
                    r"\b(database|databases|postgres|echo_memory)\b",
                    r"\b(voice|tts|stt|whisper|piper|speech|audio|microphone|synthesize)\b"
                ],
                "negative_signals": [r"\b(anime|character|scene|mei|goblin)\b"],
                "weight": 1.5  # Higher weight for Echo Brain self-knowledge
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
        # Only echo_memory collection exists in Qdrant
        # Use different score thresholds per domain to filter results
        return {
            Domain.TECHNICAL: {
                "qdrant_collections": ["echo_memory"],
                "pg_tables": ["claude_conversations"],
                "facts_filter": lambda f: any(kw in f for kw in [
                    "code", "api", "function", "class", "python", "javascript",
                    "endpoint", "database", "query", "module", "package", "import",
                    "async", "error", "bug", "deploy", "docker", "git",
                ]),
                "max_sources": 15,
                "min_score": 0.3
            },
            Domain.ANIME: {
                "qdrant_collections": ["story_bible", "echo_memory"],
                "pg_tables": [],
                "facts_filter": lambda f: any(kw in f for kw in [
                    "anime", "character", "checkpoint", "safetensors", "comfyui",
                    "lora", "generation", "sampler", "mario", "tokyo", "cyberpunk",
                    "realcartoon", "chilloutmix", "cfg", "training", "scene",
                    "framepack", "video", "image", "prompt", "negative",
                    "illumination", "pixar", "episode", "story",
                ]),
                "max_sources": 20,
                "min_score": 0.25
            },
            Domain.PERSONAL: {
                "qdrant_collections": ["echo_memory"],
                "pg_tables": ["claude_conversations"],
                "facts_filter": lambda f: any(kw in f.lower() for kw in [
                    "patrick", "tower", "preference", "vehicle", "truck",
                    "tundra", "equipment", "home", "personal",
                ]),
                "max_sources": 15,
                "min_score": 0.3
            },
            Domain.SYSTEM: {
                "qdrant_collections": ["echo_memory"],
                "pg_tables": [],
                "facts_filter": lambda f: any(kw in f for kw in [
                    "service", "tower", "gpu", "ram", "cpu", "port", "nginx",
                    "systemd", "ollama", "qdrant", "postgres", "health",
                    "echo", "brain", "model", "embedding", "architecture",
                ]),
                "max_sources": 20,
                "min_score": 0.25
            },
            Domain.GENERAL: {
                "qdrant_collections": ["echo_memory"],  # Hybrid search for unclassified queries too
                "pg_tables": [],  # No conversation search
                "facts_filter": lambda f: True,  # Allow all facts
                "max_sources": 20,  # Increased to allow more facts
                "min_score": 0.25  # Lower threshold — general queries need recall
            },
            Domain.CREATIVE: {
                "qdrant_collections": ["echo_memory"],  # Use echo_memory for all
                "pg_tables": ["claude_conversations"],
                "facts_filter": lambda f: "story" in f or "narrative" in f,
                "max_sources": 8,
                "min_score": 0.35
            },
            Domain.FINANCIAL: {
                "qdrant_collections": ["echo_memory"],  # Use echo_memory for all
                "pg_tables": [],  # transactions, financial_data don't exist
                "facts_filter": lambda f: "financial" in f or "money" in f,
                "max_sources": 5,
                "min_score": 0.45
            }
        }

    async def classify(self, query: str, query_embedding: Optional[List[float]] = None) -> List[Tuple[Domain, float]]:
        """
        Classify query into domains with confidence scores.
        Returns list of (domain, confidence) tuples sorted by confidence.

        Hybrid approach:
        1. Run regex first (fast, ~0ms)
        2. If clear winner (score >= 0.6 AND gap >= 0.15), return immediately
        3. Otherwise, use embedding cosine similarity against domain exemplars (~0ms)
           and merge scores: 0.4 * regex + 0.6 * embedding
        """
        # Step 1: Fast regex classification
        regex_results = self._regex_classify(query)

        # Step 2: Check if regex result is confident enough
        if len(regex_results) >= 1:
            top_score = regex_results[0][1]
            second_score = regex_results[1][1] if len(regex_results) >= 2 else 0.0
            gap = top_score - second_score

            if top_score >= 0.6 and gap >= 0.15:
                logger.info(f"Regex confident: {regex_results[0][0].value} ({top_score:.2f}), gap={gap:.2f}")
                return regex_results

        # Step 3: Ambiguous — use embedding similarity if we have the query embedding
        if query_embedding:
            embedding_result = await self._embedding_classify(query_embedding)

            if embedding_result:
                # Merge: 0.4 * regex + 0.6 * embedding
                merged_scores: Dict[Domain, float] = {}
                for domain, score in regex_results:
                    merged_scores[domain] = 0.4 * score

                for domain, emb_score in embedding_result:
                    merged_scores[domain] = merged_scores.get(domain, 0.0) + 0.6 * emb_score

                results = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
                logger.info(f"Embedding+regex merged: {[(d.value, f'{s:.2f}') for d, s in results[:3]]}")
                return results

        # No embedding available or embedding classify failed — return regex as-is
        return regex_results

    def _regex_classify(self, query: str) -> List[Tuple[Domain, float]]:
        """Fast regex-based classification (original logic)."""
        query_lower = query.lower()
        domain_scores = {}

        for domain, rules in self.domain_rules.items():
            matching_signals = 0
            total_signals = len(rules["signals"])

            # Check positive signals
            for signal_pattern in rules["signals"]:
                if re.search(signal_pattern, query_lower):
                    matching_signals += 1

            # Check negative signals (reduce matches)
            for neg_pattern in rules["negative_signals"]:
                if re.search(neg_pattern, query_lower):
                    matching_signals -= 0.5

            # Normalize score: matching_signals / total_signals * weight
            if matching_signals > 0 and total_signals > 0:
                score = (matching_signals / total_signals) * rules["weight"]
                domain_scores[domain] = min(1.0, score)

        # If no domain matched strongly, default to GENERAL
        if not domain_scores or max(domain_scores.values()) < 0.3:
            domain_scores[Domain.GENERAL] = 0.5

        # Sort by confidence
        results = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)

        # Log classification for debugging
        logger.info(f"Query: '{query[:50]}...' regex classified as: {results[:2]}")

        return results

    async def _embedding_classify(self, query_embedding: List[float]) -> Optional[List[Tuple[Domain, float]]]:
        """Classify by cosine similarity between query embedding and domain exemplar embeddings.
        ~0ms once exemplars are cached. No LLM call, no model swap."""
        try:
            if self._exemplar_embeddings is None:
                await self._load_exemplar_embeddings()

            if not self._exemplar_embeddings:
                return None

            # Compute average cosine similarity to each domain's exemplars
            domain_scores: Dict[Domain, float] = {}
            for domain, exemplar_vecs in self._exemplar_embeddings.items():
                if not exemplar_vecs:
                    continue
                similarities = [self._cosine_similarity(query_embedding, ev) for ev in exemplar_vecs]
                # Use max similarity (best matching exemplar) rather than average
                domain_scores[domain] = max(similarities)

            if not domain_scores:
                return None

            # Normalize to 0-1 range
            max_score = max(domain_scores.values())
            min_score = min(domain_scores.values())
            score_range = max_score - min_score if max_score > min_score else 1.0

            results = []
            for domain, score in domain_scores.items():
                normalized = (score - min_score) / score_range if score_range > 0 else 0.5
                results.append((domain, normalized))

            results.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"Embedding classify: {[(d.value, f'{s:.2f}') for d, s in results[:3]]}")
            return results

        except Exception as e:
            logger.warning(f"Embedding classify failed: {e}")
            return None

    async def _load_exemplar_embeddings(self):
        """Load and cache embeddings for domain exemplar phrases. Called once."""
        if self._exemplar_loading:
            return  # Another call is already loading
        self._exemplar_loading = True

        try:
            import httpx

            self._exemplar_embeddings = {}

            async with httpx.AsyncClient(timeout=30.0) as client:
                for domain, phrases in DOMAIN_EXEMPLARS.items():
                    vecs = []
                    for phrase in phrases:
                        try:
                            resp = await client.post(
                                "http://localhost:11434/api/embed",
                                json={"model": "nomic-embed-text", "input": phrase}
                            )
                            if resp.status_code == 200:
                                embeddings = resp.json().get("embeddings", [])
                                if embeddings:
                                    vecs.append(embeddings[0])
                        except Exception:
                            continue
                    self._exemplar_embeddings[domain] = vecs

            total = sum(len(v) for v in self._exemplar_embeddings.values())
            logger.info(f"Loaded {total} domain exemplar embeddings across {len(self._exemplar_embeddings)} domains")

        except Exception as e:
            logger.error(f"Failed to load exemplar embeddings: {e}")
            self._exemplar_embeddings = {}
        finally:
            self._exemplar_loading = False

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

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
