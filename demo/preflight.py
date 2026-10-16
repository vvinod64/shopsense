"""Pre-flight check before recording the demo. Prints one line per check; nothing secret is printed.

    .venv/bin/python demo/preflight.py           # check everything
    .venv/bin/python demo/preflight.py --reset   # also move earlier demo state aside (runtime/demo -> archive)

The demo and the API server for Postman both use runtime/demo, so a reset gives every demo order a clean
history (an earlier refund on the same order would otherwise make the order system hold the request).
"""
import os
import shutil
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "runtime" / "demo"
os.environ.setdefault("SHOPSENSE_RUNTIME_DIR", str(DEMO_DIR))
results = []


def check(name: str, fn) -> None:
    try:
        detail = fn()
        results.append(True)
        print(f"  \033[32mOK  \033[0m {name:<34}{detail or ''}")
    except Exception as e:
        results.append(False)
        print(f"  \033[31mFAIL\033[0m {name:<34}{type(e).__name__}: {str(e)[:110]}")


def env_keys():
    import shopsense.config  # noqa: F401  (loads .env)
    needed = ["GEMINI_API_KEY", "SUPERMEMORY_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "LANGFUSE_PUBLIC_KEY",
              "LANGFUSE_SECRET_KEY"]
    missing = [k for k in needed if not os.getenv(k)]
    assert not missing, f"missing in .env: {missing}"
    return f"all {len(needed)} set; staff sign in on /ui as lead, ops or finance"


def gemini():
    import litellm
    from shopsense.config import LLM_MODEL
    try:
        litellm.completion(model=LLM_MODEL, messages=[{"role": "user", "content": "Reply OK"}], num_retries=0,
                           max_tokens=5, timeout=30)
    except Exception as e:
        text = str(e)
        if "PerDay" in text:
            raise RuntimeError(f"{LLM_MODEL}: daily quota used up (resets 12:30 IST); "
                               "try SHOPSENSE_LLM_MODEL=gemini/gemini-3.1-flash-lite") from None
        if "429" in text:
            raise RuntimeError(f"{LLM_MODEL}: per-minute limit; wait a minute and re-run") from None
        if "503" in text or "UNAVAILABLE" in text:
            raise RuntimeError(f"{LLM_MODEL}: Google is overloaded right now (503); retry in a few minutes "
                               "or set SHOPSENSE_LLM_MODEL=gemini/gemini-3.1-flash-lite") from None
        raise
    return f"{LLM_MODEL} answers (override with SHOPSENSE_LLM_MODEL)"


def qdrant():
    from shopsense.m3.rag import COLLECTION, all_chunks, client
    n = client().count(COLLECTION, exact=True).count
    assert n == len(all_chunks()), f"{n} points, expected {len(all_chunks())}: run python -m shopsense.m3.rag"
    return f"{n} chunks indexed (cluster awake)"


def supermemory():
    from shopsense.m3.memory import customer_tag, recall
    recall("preferred language", customer_tag("KW-C-00298"), k=1)
    return "reachable"


def langfuse():
    from langfuse import get_client
    assert get_client().auth_check(), "auth failed: check keys and LANGFUSE_BASE_URL region"
    return os.getenv("LANGFUSE_BASE_URL", "")


def port_free():
    with socket.socket() as s:
        assert s.connect_ex(("127.0.0.1", 8000)) != 0, "something already listens on :8000 (old uvicorn?)"
    return "8000 free for uvicorn"


def demo_state():
    dirs = [DEMO_DIR, ROOT / "runtime" / "demo-cli"]          # API server state, demo.py state
    used = [d for d in dirs if d.exists() and any(d.iterdir())]
    if "--reset" in sys.argv and used:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for d in used:
            shutil.move(str(d), ROOT / "runtime" / f"{d.name}_archive_{stamp}")
        return f"earlier demo state moved to runtime/*_archive_{stamp}"
    if used:
        return f"{', '.join('runtime/' + d.name for d in used)} has earlier state (use --reset for a clean run)"
    return "runtime/demo and runtime/demo-cli are clean"


if __name__ == "__main__":
    print("\nShopSense pre-flight\n")
    check("demo state", demo_state)
    check(".env keys", env_keys)
    check("Gemini quota", gemini)
    check("Qdrant Cloud", qdrant)
    check("Supermemory", supermemory)
    check("Langfuse", langfuse)
    check("API port", port_free)
    ok = all(results)
    print(f"\n{'Ready to record.' if ok else 'Fix the FAIL lines first.'}")
    print("Start ShopSense with:\n  uv run shopsense\n")
    sys.exit(0 if ok else 1)
