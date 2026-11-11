import os
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def _db_url() -> str:
    url = os.getenv("DATABASE_URL", f"sqlite:///{(BASE / 'gateway.db').as_posix()}")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url


DATABASE_URL = _db_url()
REDIS_URL = os.getenv("REDIS_URL", "")  # empty -> in-memory store (fine for one instance / local dev)
ADMIN_KEY = os.getenv("ADMIN_KEY", "")

if not ADMIN_KEY:
    import warnings
    warnings.warn(
        "ADMIN_KEY is not set. Admin endpoints are not safe for deployment.",
        RuntimeWarning,
    )
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.88"))
DEFAULT_RATE_LIMIT = int(os.getenv("DEFAULT_RATE_LIMIT", "60"))  # requests per minute per key
PROVIDER_TIMEOUT = float(os.getenv("PROVIDER_TIMEOUT", "30"))
# Preferred order for model="auto" and for fallbacks. Models whose provider has no API key are skipped.
AUTO_CHAIN = [m.strip() for m in os.getenv(
    "AUTO_CHAIN", "claude-sonnet-4-6,gpt-4o-mini,llama-3.1-8b-instant,mock-smart").split(",") if m.strip()]
