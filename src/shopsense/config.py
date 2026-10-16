"""Shared paths and environment loading.

Paths resolve from this file, not the working directory, because the MCP server runs as a
subprocess with its own cwd.
"""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "shopsense"
CORPUS_DIR = DATA_DIR / "corpus"
MOCK_API_DIR = DATA_DIR / "mock_api"
EVAL_DIR = DATA_DIR / "eval"
EMBED_CACHE_DIR = PROJECT_ROOT / ".embcache"
# Runtime output goes here; data/ is read-only input.
RUNTIME_DIR = Path(os.getenv("SHOPSENSE_RUNTIME_DIR") or PROJECT_ROOT / "runtime")

load_dotenv(PROJECT_ROOT / ".env", override=True)

# Silence gRPC fork-handler log noise from the Google client.
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("LITELLM_LOG", "ERROR")
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
try:
    import litellm
    litellm.suppress_debug_info = True          # no "Give Feedback / Get Help" banner on handled errors
except ImportError:
    pass

# LiteLLM reads GEMINI_API_KEY and langchain-google-genai reads GOOGLE_API_KEY; mirror one into the other.
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
if os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
# Langfuse v4 reads LANGFUSE_BASE_URL; accept LANGFUSE_HOST as well.
if os.getenv("LANGFUSE_HOST") and not os.getenv("LANGFUSE_BASE_URL"):
    os.environ["LANGFUSE_BASE_URL"] = os.environ["LANGFUSE_HOST"]

def automation_paused() -> bool:
    """Operations switch (SHOPSENSE_AUTOMATION=paused): nothing is auto-approved; every action waits for a person.
    Read on every call, so it takes effect without a code change."""
    return os.getenv("SHOPSENSE_AUTOMATION", "on").strip().lower() == "paused"


AUTOMATION_PAUSED = "automation paused by operations: every action needs a person"

# SHOPSENSE_LLM_MODEL overrides LLM_MODEL from .env (Gemini free-tier quotas are per model).
LLM_MODEL = os.getenv("SHOPSENSE_LLM_MODEL") or os.getenv("LLM_MODEL", "gemini/gemini-flash-latest")
# Models that grade or check the system's output should come from a different provider (a different model
# family shares the generator's blind spots less). Both default to LLM_MODEL; reports say when they are the same.
JUDGE_MODEL = os.getenv("SHOPSENSE_JUDGE_MODEL") or os.getenv("JUDGE_MODEL") or LLM_MODEL
REVIEWER_MODEL = os.getenv("SHOPSENSE_REVIEWER_MODEL") or os.getenv("REVIEWER_MODEL") or JUDGE_MODEL


def model_provider(model: str) -> str:
    import litellm
    try:
        return litellm.get_llm_provider(model)[1]
    except Exception:
        return model.split("/", 1)[0]


def independent_of_system(model: str) -> bool:
    """True when the grading model is from a different provider than the system's model."""
    return model_provider(model) != model_provider(LLM_MODEL)
# Per-call model timeout. LiteLLM's default is 6000 s, so one stalled response could block a ticket for hours;
# a timed-out call is retried by num_retries like any other transport error.
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
