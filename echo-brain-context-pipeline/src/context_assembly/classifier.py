"""
Query Classifier

Classifies incoming queries into domains to prevent context contamination.
Uses a hybrid approach: keyword matching + LLM fallback for ambiguous cases.
"""

import re
from dataclasses import dataclass

from .models import ClassificationResult, Domain


@dataclass
class DomainKeywords:
    """Keywords and patterns that indicate a specific domain."""
    primary: set[str]      # Strong indicators
    secondary: set[str]    # Weaker indicators  
    negative: set[str]     # Keywords that suggest this is NOT the domain
    patterns: list[re.Pattern]  # Regex patterns


# Domain keyword definitions
DOMAIN_KEYWORDS: dict[Domain, DomainKeywords] = {
    Domain.TECHNICAL: DomainKeywords(
        primary={
            # Infrastructure
            "postgresql", "postgres", "qdrant", "ollama", "fastapi", "docker",
            "redis", "vault", "hashicorp", "nginx", "systemd", "ubuntu",
            # Programming
            "python", "typescript", "javascript", "rust", "api", "endpoint",
            "function", "class", "module", "import", "async", "await",
            "exception", "error", "debug", "logging", "test", "pytest",
            # Echo Brain specific
            "echo brain", "echo-brain", "tower server", "microservice",
            "vector", "embedding", "rag", "retrieval", "inference",
            # Database
            "sql", "query", "table", "index", "migration", "schema",
            "connection pool", "transaction",
            # DevOps
            "deploy", "ci/cd", "pipeline", "container", "kubernetes", "k8s",
            # General tech
            "server", "network", "cpu", "ram", "gpu", "rtx", "cuda", "vram"
        },
        secondary={
            "code", "script", "config", "configuration", "setup", "install",
            "build", "compile", "run", "execute", "performance", "optimize",
            "memory", "storage", "disk", "file", "directory", "path"
        },
        negative={
            "anime", "lora", "checkpoint", "comfyui", "character", "episode",
            "story", "plot", "scene"
        },
        patterns=[
            re.compile(r'\b(fix|debug|error|exception|traceback)\b', re.I),
            re.compile(r'\b(how (do|can|to)|why (is|does|doesn\'t))\b.*\b(code|api|server|database)\b', re.I),
            re.compile(r'\.(py|ts|js|sql|yaml|json|toml|sh)\b', re.I),
        ]
    ),
    
    Domain.ANIME: DomainKeywords(
        primary={
            # Projects
            "tokyo debt desire", "cyberpunk goblin slayer", "tdd", "cgs",
            # AI Art/Video
            "lora", "checkpoint", "comfyui", "stable diffusion", "sdxl",
            "ltx", "animatediff", "controlnet", "img2img", "txt2img",
            "character consistency", "style transfer",
            # Content
            "episode", "scene", "storyboard", "keyframe", "animation",
            "character design", "background", "render"
        },
        secondary={
            "anime", "manga", "art", "visual", "frame", "shot", "clip",
            "video generation", "image generation", "prompt", "negative prompt",
            "cfg scale", "steps", "sampler", "seed"
        },
        negative={
            "postgresql", "fastapi", "docker", "kubernetes", "pytest",
            "migration", "deployment", "ci/cd"
        },
        patterns=[
            re.compile(r'\b(lora|checkpoint|controlnet)\s*(training|settings|config)', re.I),
            re.compile(r'\b(character|style|scene)\s*(consistency|transfer|generation)', re.I),
            re.compile(r'\bcomfyui\b', re.I),
        ]
    ),
    
    Domain.PERSONAL: DomainKeywords(
        primary={
            # Personal items
            "tundra", "sundowner", "trailblazer", "rv", "toy hauler",
            "victron", "lifepo4", "solar", "inverter", "shunt",
            # Personal context
            "my schedule", "my preference", "remind me", "remember that",
            "i mentioned", "we discussed", "last time"
        },
        secondary={
            "utv", "camping", "travel", "road trip", "battery", "electrical",
            "shore power", "boondocking", "kitchen", "knife", "cooking"
        },
        negative=set(),  # Personal can blend with anything
        patterns=[
            re.compile(r'\b(my|our)\s+(schedule|appointment|preference|setting)', re.I),
            re.compile(r'\b(remind|remember|recall)\s+(me|that|when)', re.I),
        ]
    ),
    
    Domain.GENERAL: DomainKeywords(
        primary=set(),
        secondary=set(),
        negative=set(),
        patterns=[]
    )
}


class QueryClassifier:
    """
    Classifies queries into domains using keyword matching and pattern recognition.
    
    The classifier uses a scoring system:
    - Primary keyword match: +3 points
    - Secondary keyword match: +1 point
    - Negative keyword match: -2 points
    - Pattern match: +2 points
    
    The domain with the highest score wins. If no domain scores above threshold,
    falls back to GENERAL.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.6,
        min_score_threshold: int = 3,
        ollama_client = None  # Optional: for LLM fallback on ambiguous cases
    ):
        self.confidence_threshold = confidence_threshold
        self.min_score_threshold = min_score_threshold
        self.ollama_client = ollama_client
    
    def classify(self, query: str) -> ClassificationResult:
        """
        Classify a query into a domain.
        
        Args:
            query: The user's query text
            
        Returns:
            ClassificationResult with domain, confidence, and matched keywords
        """
        query_lower = query.lower()
        scores: dict[Domain, int] = {domain: 0 for domain in Domain}
        matched_keywords: dict[Domain, list[str]] = {domain: [] for domain in Domain}
        
        # Score each domain
        for domain, keywords in DOMAIN_KEYWORDS.items():
            if domain == Domain.GENERAL:
                continue  # General is fallback only
            
            # Check primary keywords
            for kw in keywords.primary:
                if kw in query_lower:
                    scores[domain] += 3
                    matched_keywords[domain].append(kw)
            
            # Check secondary keywords
            for kw in keywords.secondary:
                if kw in query_lower:
                    scores[domain] += 1
                    matched_keywords[domain].append(kw)
            
            # Check negative keywords
            for kw in keywords.negative:
                if kw in query_lower:
                    scores[domain] -= 2
            
            # Check patterns
            for pattern in keywords.patterns:
                if pattern.search(query):
                    scores[domain] += 2
        
        # Find winning domain
        max_score = max(scores.values())
        
        if max_score < self.min_score_threshold:
            # No clear winner, use GENERAL
            return ClassificationResult(
                query=query,
                domain=Domain.GENERAL,
                confidence=0.5,
                reasoning="No domain-specific keywords detected",
                matched_keywords=[]
            )
        
        # Get the winning domain(s)
        winners = [d for d, s in scores.items() if s == max_score]
        
        if len(winners) == 1:
            winning_domain = winners[0]
            # Calculate confidence based on score magnitude and margin
            second_highest = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0
            margin = max_score - second_highest
            confidence = min(0.95, 0.6 + (margin * 0.05) + (max_score * 0.02))
            
            return ClassificationResult(
                query=query,
                domain=winning_domain,
                confidence=confidence,
                reasoning=f"Matched {len(matched_keywords[winning_domain])} keywords with score {max_score}",
                matched_keywords=matched_keywords[winning_domain]
            )
        else:
            # Tie between domains - need to resolve
            # Priority: TECHNICAL > ANIME > PERSONAL > GENERAL
            priority_order = [Domain.TECHNICAL, Domain.ANIME, Domain.PERSONAL]
            for domain in priority_order:
                if domain in winners:
                    return ClassificationResult(
                        query=query,
                        domain=domain,
                        confidence=0.6,  # Lower confidence due to tie
                        reasoning=f"Tie between {winners}, resolved by priority",
                        matched_keywords=matched_keywords[domain]
                    )
            
            return ClassificationResult(
                query=query,
                domain=Domain.GENERAL,
                confidence=0.5,
                reasoning="Tie could not be resolved",
                matched_keywords=[]
            )
    
    async def classify_with_llm_fallback(self, query: str) -> ClassificationResult:
        """
        Classify with LLM fallback for low-confidence cases.
        
        Use this when you need higher accuracy and have compute budget.
        """
        result = self.classify(query)
        
        if result.confidence >= self.confidence_threshold:
            return result
        
        if self.ollama_client is None:
            return result
        
        # Use LLM for ambiguous cases
        llm_result = await self._classify_with_llm(query)
        
        # Merge results - LLM breaks the tie
        if llm_result.confidence > result.confidence:
            return llm_result
        return result
    
    async def _classify_with_llm(self, query: str) -> ClassificationResult:
        """Use Ollama to classify an ambiguous query."""
        prompt = f"""Classify this query into exactly one domain. Respond with only the domain name.

Domains:
- technical: Programming, servers, databases, DevOps, Echo Brain infrastructure
- anime: Tokyo Debt Desire, Cyberpunk Goblin Slayer, LoRA training, ComfyUI, AI art/video generation
- personal: RV, Tundra truck, Victron electrical, personal schedule, preferences
- general: Everything else

Query: {query}

Domain:"""
        
        try:
            response = await self.ollama_client.generate(
                model="qwen2.5:7b",  # Fast model for classification
                prompt=prompt,
                options={"temperature": 0.1, "num_predict": 20}
            )
            
            domain_text = response["response"].strip().lower()
            
            # Parse response
            domain_map = {
                "technical": Domain.TECHNICAL,
                "anime": Domain.ANIME,
                "personal": Domain.PERSONAL,
                "general": Domain.GENERAL
            }
            
            domain = domain_map.get(domain_text, Domain.GENERAL)
            
            return ClassificationResult(
                query=query,
                domain=domain,
                confidence=0.75,  # LLM gets moderate confidence
                reasoning=f"LLM classified as {domain_text}",
                matched_keywords=[]
            )
        except Exception as e:
            # Fallback to GENERAL on error
            return ClassificationResult(
                query=query,
                domain=Domain.GENERAL,
                confidence=0.5,
                reasoning=f"LLM classification failed: {e}",
                matched_keywords=[]
            )


# Convenience function for simple usage
def classify_query(query: str) -> ClassificationResult:
    """Quick classification without LLM fallback."""
    classifier = QueryClassifier()
    return classifier.classify(query)


# ============================================================================
# Tests
# ============================================================================

def test_classifier():
    """Basic tests for the classifier."""
    classifier = QueryClassifier()
    
    test_cases = [
        # Technical queries
        ("How do I fix the PostgreSQL connection pooling issue?", Domain.TECHNICAL),
        ("The FastAPI endpoint is returning 500 errors", Domain.TECHNICAL),
        ("Debug the async function in the retriever module", Domain.TECHNICAL),
        ("Echo brain memory vectors not updating", Domain.TECHNICAL),
        
        # Anime queries
        ("What LoRA settings work best for character consistency?", Domain.ANIME),
        ("ComfyUI workflow for Tokyo Debt Desire scene generation", Domain.ANIME),
        ("The checkpoint isn't loading in stable diffusion", Domain.ANIME),
        ("Train a LoRA for Cyberpunk Goblin Slayer characters", Domain.ANIME),
        
        # Personal queries
        ("What's the Victron MultiPlus settings for shore power?", Domain.PERSONAL),
        ("Remind me about the Tundra oil change schedule", Domain.PERSONAL),
        ("We discussed my RV battery setup last time", Domain.PERSONAL),
        
        # General queries
        ("What's the weather like today?", Domain.GENERAL),
        ("Explain how neural networks work", Domain.GENERAL),
        ("Hello, how are you?", Domain.GENERAL),
    ]
    
    passed = 0
    failed = 0
    
    for query, expected_domain in test_cases:
        result = classifier.classify(query)
        if result.domain == expected_domain:
            passed += 1
            print(f"✓ '{query[:50]}...' -> {result.domain.value} (confidence: {result.confidence:.2f})")
        else:
            failed += 1
            print(f"✗ '{query[:50]}...' -> Expected {expected_domain.value}, got {result.domain.value}")
            print(f"  Reasoning: {result.reasoning}")
            print(f"  Matched: {result.matched_keywords}")
    
    print(f"\nResults: {passed}/{passed + failed} passed")
    return passed, failed


if __name__ == "__main__":
    test_classifier()
