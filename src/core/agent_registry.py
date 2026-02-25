"""
Agent Registry for Echo Brain

Loads agent definitions from markdown files in /opt/tower-echo-brain/agents/,
provides intent-based routing, and handles model availability checks with fallback.

Agent files use YAML frontmatter + markdown body for system prompts.
Hot-reloads when file mtimes change.
"""
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import yaml

logger = logging.getLogger("echo.core.agent_registry")

AGENTS_DIR = Path("/opt/tower-echo-brain/agents")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_CHECK_TTL = 300  # 5 minutes


@dataclass
class AgentConfig:
    """Configuration for a single agent loaded from a markdown file."""
    name: str
    description: str = ""
    model: str = "mistral:7b"
    fallback_model: str = "mistral:7b"
    intents: List[str] = field(default_factory=list)
    system_prompt: str = ""
    token_budget_model: str = "mistral:7b"
    options: Dict = field(default_factory=dict)
    compaction: Dict = field(default_factory=dict)
    source_file: str = ""


class AgentRegistry:
    """
    Registry of agent definitions loaded from markdown files.

    Features:
    - Intent-based agent selection
    - mtime-based hot-reload
    - Ollama model availability check with caching
    - Graceful fallback chain
    """

    def __init__(self):
        self._agents: Dict[str, AgentConfig] = {}
        self._intent_map: Dict[str, str] = {}  # intent -> agent name
        self._file_mtimes: Dict[str, float] = {}
        self._available_models: set = set()
        self._last_model_check: float = 0
        self._loaded = False

    def load(self) -> None:
        """Load all agent definitions from the agents directory."""
        if not AGENTS_DIR.exists():
            logger.warning(f"Agents directory not found: {AGENTS_DIR}")
            return

        agents = {}
        intent_map = {}
        file_mtimes = {}

        for md_file in sorted(AGENTS_DIR.glob("*.md")):
            try:
                agent = self._parse_agent_file(md_file)
                agents[agent.name] = agent
                file_mtimes[str(md_file)] = md_file.stat().st_mtime

                for intent in agent.intents:
                    intent_map[intent] = agent.name

                logger.info(f"Loaded agent: {agent.name} (model={agent.model}, intents={agent.intents})")
            except Exception as e:
                logger.error(f"Failed to parse agent file {md_file}: {e}")

        self._agents = agents
        self._intent_map = intent_map
        self._file_mtimes = file_mtimes
        self._loaded = True

        logger.info(f"Agent registry loaded: {len(agents)} agents, {len(intent_map)} intent mappings")

    def _parse_agent_file(self, path: Path) -> AgentConfig:
        """Parse a markdown file with YAML frontmatter into an AgentConfig."""
        content = path.read_text(encoding="utf-8")

        # Split YAML frontmatter from body
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = yaml.safe_load(parts[1]) or {}
                body = parts[2].strip()
            else:
                frontmatter = {}
                body = content
        else:
            frontmatter = {}
            body = content

        return AgentConfig(
            name=frontmatter.get("name", path.stem),
            description=frontmatter.get("description", ""),
            model=frontmatter.get("model", "mistral:7b"),
            fallback_model=frontmatter.get("fallback_model", "mistral:7b"),
            intents=frontmatter.get("intents", []),
            system_prompt=body,
            token_budget_model=frontmatter.get("token_budget_model", frontmatter.get("model", "mistral:7b")),
            options=frontmatter.get("options", {}),
            compaction=frontmatter.get("compaction", {}),
            source_file=str(path),
        )

    def _check_reload(self) -> None:
        """Hot-reload if any agent file has been modified."""
        if not AGENTS_DIR.exists():
            return

        needs_reload = False

        # Check existing files for mtime changes
        for md_file in AGENTS_DIR.glob("*.md"):
            path_str = str(md_file)
            current_mtime = md_file.stat().st_mtime
            if path_str not in self._file_mtimes or self._file_mtimes[path_str] != current_mtime:
                needs_reload = True
                break

        # Check for deleted files
        if not needs_reload:
            existing_files = {str(f) for f in AGENTS_DIR.glob("*.md")}
            if set(self._file_mtimes.keys()) != existing_files:
                needs_reload = True

        if needs_reload:
            logger.info("Agent files changed, reloading registry")
            self.load()

    def select(self, intent: str) -> AgentConfig:
        """Select an agent by intent string.

        Args:
            intent: A QueryType.value string (e.g. 'code_query', 'general_knowledge')

        Returns:
            The matching AgentConfig, or the 'general' agent, or a default config.
        """
        self._check_reload()

        agent_name = self._intent_map.get(intent)
        if agent_name and agent_name in self._agents:
            return self._agents[agent_name]

        # Fallback to 'general' agent
        if "general" in self._agents:
            return self._agents["general"]

        # Last resort: return a default config
        from src.model_config import get_model
        return AgentConfig(
            name="default",
            model=get_model("general"),
            fallback_model="mistral:7b",
            intents=[],
            system_prompt="You are a helpful AI assistant. Answer questions directly and accurately.",
            token_budget_model=get_model("general"),
        )

    async def resolve_model(self, agent: AgentConfig) -> str:
        """Resolve which model to actually use, checking Ollama availability.

        Fallback chain: agent.model -> agent.fallback_model -> model_config default
        """
        await self._refresh_models_if_stale()

        if self._is_available(agent.model):
            return agent.model

        logger.warning(f"Agent {agent.name}: model {agent.model} unavailable, trying fallback {agent.fallback_model}")

        if self._is_available(agent.fallback_model):
            return agent.fallback_model

        # Last resort
        from src.model_config import get_model
        default = get_model("general")
        logger.warning(f"Agent {agent.name}: fallback {agent.fallback_model} also unavailable, using {default}")
        return default

    async def _refresh_models_if_stale(self) -> None:
        """Refresh the set of available Ollama models if cache is stale."""
        if time.time() - self._last_model_check < MODEL_CHECK_TTL:
            return

        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{OLLAMA_URL}/api/tags")
                models = resp.json().get("models", [])
                self._available_models = set()
                for m in models:
                    name = m["name"]
                    self._available_models.add(name)
                    if name.endswith(":latest"):
                        self._available_models.add(name.replace(":latest", ""))
                self._last_model_check = time.time()
        except Exception as e:
            logger.warning(f"Failed to refresh Ollama model list: {e}")

    def _is_available(self, model: str) -> bool:
        """Check if a model is in the cached available set."""
        if not self._available_models:
            return True  # If we can't check, assume available

        return (
            model in self._available_models
            or f"{model}:latest" in self._available_models
            or any(model in m for m in self._available_models)
        )

    def force_reload(self) -> None:
        """Force reload all agent definitions."""
        self._last_model_check = 0  # Also refresh model list on next resolve
        self.load()

    def get_all(self) -> List[AgentConfig]:
        """Get all loaded agents."""
        self._check_reload()
        return list(self._agents.values())


# Singleton
_registry: Optional[AgentRegistry] = None


def get_agent_registry() -> AgentRegistry:
    """Get the singleton AgentRegistry, loading on first access."""
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
        _registry.load()
    return _registry
