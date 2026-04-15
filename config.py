"""
Configuration loader for the Jira RAG Agent.

Reads credentials and tunable knobs from environment variables (with a
.env fallback). Env vars are read lazily on every attribute access so
UIs (e.g. the Streamlit app) can inject credentials at runtime by
setting os.environ before constructing the Jira / Anthropic clients.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: str) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return int(default)


class _Config:
    """Lazy config: every attribute access re-reads os.environ."""

    # --- Jira ---
    @property
    def JIRA_HOST(self) -> str:
        return os.getenv("JIRA_HOST", "https://your-domain.atlassian.net")

    @property
    def JIRA_EMAIL(self) -> str:
        return os.getenv("JIRA_EMAIL", "")

    @property
    def JIRA_API_TOKEN(self) -> str:
        return os.getenv("JIRA_API_TOKEN", "")

    # --- Anthropic ---
    @property
    def ANTHROPIC_API_KEY(self) -> str:
        return os.getenv("ANTHROPIC_API_KEY", "")

    @property
    def CLAUDE_MODEL(self) -> str:
        return os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

    @property
    def CLAUDE_AUTH_MODE(self) -> str:
        """'api' (pay-per-token, needs ANTHROPIC_API_KEY) or
        'subscription' (uses Claude Code OAuth tied to a Claude.ai plan)."""
        return os.getenv("CLAUDE_AUTH_MODE", "api").lower()

    # --- Vector store ---
    @property
    def VECTOR_DB(self) -> str:
        return os.getenv("VECTOR_DB", "chroma")

    @property
    def PINECONE_API_KEY(self) -> str:
        return os.getenv("PINECONE_API_KEY", "")

    @property
    def PINECONE_ENVIRONMENT(self) -> str:
        return os.getenv("PINECONE_ENVIRONMENT", "")

    # --- Optional Redis cache ---
    @property
    def REDIS_HOST(self) -> str:
        return os.getenv("REDIS_HOST", "localhost")

    @property
    def REDIS_PORT(self) -> int:
        return _int("REDIS_PORT", "6379")

    # --- RAG tunables ---
    @property
    def CHUNK_SIZE(self) -> int:
        return _int("CHUNK_SIZE", "1000")

    @property
    def CHUNK_OVERLAP(self) -> int:
        return _int("CHUNK_OVERLAP", "200")

    @property
    def TOP_K_RETRIEVAL(self) -> int:
        return _int("TOP_K_RETRIEVAL", "5")

    # --- Test case generation bounds ---
    @property
    def MIN_TEST_CASES_PER_TICKET(self) -> int:
        return _int("MIN_TEST_CASES_PER_TICKET", "5")

    @property
    def MAX_TEST_CASES_PER_TICKET(self) -> int:
        return _int("MAX_TEST_CASES_PER_TICKET", "15")

    # --- Extended thinking ---
    @property
    def CLAUDE_EXTENDED_THINKING(self) -> bool:
        return os.getenv(
            "CLAUDE_EXTENDED_THINKING", "true"
        ).lower() in ("1", "true", "yes", "on")

    @property
    def CLAUDE_THINKING_BUDGET(self) -> int:
        return _int("CLAUDE_THINKING_BUDGET", "4096")

    # --- Connection timeouts ---
    @property
    def JIRA_TIMEOUT(self) -> int:
        return _int("JIRA_TIMEOUT", "30")

    @property
    def API_TIMEOUT(self) -> int:
        return _int("API_TIMEOUT", "60")


config = _Config()
Config = _Config  # backwards-compatible alias
