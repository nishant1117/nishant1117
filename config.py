"""
Configuration loader for the Jira RAG Agent.

Reads credentials and tunable knobs from environment variables (with a
.env fallback). Keep secrets out of version control — use the provided
.env.example as a template.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Base configuration"""

    # Jira connection
    JIRA_HOST = os.getenv("JIRA_HOST", "https://your-domain.atlassian.net")
    JIRA_EMAIL = os.getenv("JIRA_EMAIL")
    JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")

    # Anthropic / Claude
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")

    # Vector DB selection
    VECTOR_DB = os.getenv("VECTOR_DB", "chroma")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

    # Optional Redis cache
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

    # RAG chunking / retrieval
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))

    # Test case generation bounds
    MIN_TEST_CASES_PER_TICKET = int(os.getenv("MIN_TEST_CASES_PER_TICKET", "5"))
    MAX_TEST_CASES_PER_TICKET = int(os.getenv("MAX_TEST_CASES_PER_TICKET", "15"))


config = Config()
