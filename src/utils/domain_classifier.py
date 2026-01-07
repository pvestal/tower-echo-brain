"""Domain classifier to prevent cross-domain contamination in Echo Brain memory system."""

class QueryDomainClassifier:
    """Classifies queries to prevent cross-domain contamination between anime and technical content."""

    ANIME_KEYWORDS = [
        'anime', 'goblin', 'cyber', 'scene', 'character', 'mei',
        'tokyo debt', 'slayer', 'lora', 'comfyui', 'kobayashi',
        'generation', 'workflow', 'sdxl', 'checkpoint', 'prompt'
    ]

    TECH_KEYWORDS = [
        'code', 'function', 'debug', 'system', 'architecture',
        'database', 'api', 'docker', 'python', 'sql', 'error',
        'bug', 'deploy', 'git', 'server', 'service', 'endpoint'
    ]

    def classify(self, query: str) -> str:
        """
        Classifies a query into domains.

        Returns:
            'anime', 'technical', or 'general'
        """
        query_lower = query.lower()

        # Score each domain
        anime_score = sum(1 for kw in self.ANIME_KEYWORDS if kw in query_lower)
        tech_score = sum(1 for kw in self.TECH_KEYWORDS if kw in query_lower)

        # Prioritize technical classification to keep tech queries clean
        if tech_score > 0:
            return 'technical'
        elif anime_score > 0:
            return 'anime'

        # Default to general for non-specific queries
        return 'general'

    def filter_content(self, content: str, query_domain: str) -> bool:
        """
        Determines if content is safe for a given query domain.

        Args:
            content: The content to check
            query_domain: The classified domain of the query

        Returns:
            True if content is safe for the query domain, False otherwise
        """
        if not content:
            return True

        content_lower = content.lower()

        # Check for anime content
        has_anime = any(kw in content_lower for kw in self.ANIME_KEYWORDS)

        # Block anime content for technical queries
        if query_domain == 'technical' and has_anime:
            return False

        # Allow all content for anime queries
        if query_domain == 'anime':
            return True

        # For general queries, allow non-anime content
        if query_domain == 'general' and has_anime:
            return False

        return True

    def filter_facts(self, facts: list, query_domain: str) -> list:
        """
        Filters a list of facts based on query domain.

        Args:
            facts: List of fact strings
            query_domain: The classified domain of the query

        Returns:
            Filtered list of facts safe for the query domain
        """
        return [f for f in facts if self.filter_content(f, query_domain)]