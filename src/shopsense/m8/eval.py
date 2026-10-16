"""Evaluation of ShopSense on the 20-ticket golden set, logged as a Langfuse experiment.

Each golden item has a scenario category and the pipeline components it exercises. Components are
scored separately, each with at least one deterministic metric next to any LLM judge:
  tool_use       tool_call_match (order looked up, action, reason code, quoted amount)
  retrieval      citation_recall, retrieval_keyword_hit_rate, Ragas context precision,
                 TruLens context relevance
  routing        route_match (auto / human_review / refuse), risk_reasons_match (held for the expected reasons)
  final_answer   facts_in_reply, DeepEval G-Eval, Ragas faithfulness, TruLens groundedness
  guardrails     guardrail_passed and must_not_contain_ok, on every item
All model calls share one pacing gate that honours the provider's retryDelay (Gemini sends one).

Usage:  python -m shopsense.m8.eval [--no-judges] [--engine workflow|team] [--only SS-F01,SS-G02]
        --engine team runs the golden set through the multi-agent team; its result is mapped onto the
        workflow's result shape so both engines use the same scorers.
"""
import asyncio
import json
from pathlib import Path
import os
import random
import re
import sys
import threading
import time
import types
from collections import defaultdict
from datetime import datetime
from typing import Optional

from shopsense.m6.team import team_to_result  # noqa: F401  (shared with the API)
from shopsense.config import JUDGE_MODEL, LLM_MODEL, PROJECT_ROOT, RUNTIME_DIR, independent_of_system
from shopsense.guardrails import guardrail_check, system_route

# --golden PATH evaluates another set, e.g. the toolkit's data/shopsense/eval/golden_set.json (its own format).
V2_PATH = PROJECT_ROOT / "eval" / "golden_set_v2.json"
GOLDEN_PATH = Path(sys.argv[sys.argv.index("--golden") + 1]).resolve() if "--golden" in sys.argv else V2_PATH
# Tickets run in a temp runtime dir; the report goes to runtime/eval_reports.
REPORT_DIR = __import__("pathlib").Path(os.getenv("SHOPSENSE_RUNTIME_DIR_REPORTS") or RUNTIME_DIR) / "eval_reports"
COMPONENTS = ("tool_use", "retrieval", "routing", "final_answer")
# metric -> component it scores
COMPONENT_METRICS = {
    "tool_use": ("tool_call_match",),
    "retrieval": ("citation_recall", "retrieval_keyword_hit_rate", "ragas_context_precision",
                  "trulens_context_relevance"),
    "routing": ("route_match", "risk_reasons_match"),
    "final_answer": ("facts_in_reply", "deepeval_geval_correctness", "ragas_faithfulness", "trulens_groundedness"),
    "guardrails": ("guardrail_passed", "must_not_contain_ok"),
}
DETERMINISTIC = ("tool_call_match", "citation_recall", "retrieval_keyword_hit_rate", "route_match", "risk_reasons_match",
                 "facts_in_reply", "guardrail_passed", "must_not_contain_ok")
_MIN_GAP_SECONDS = 6.0          # the 15 requests/minute limit is shared by the system and the judges
JUDGE_TIMEOUT_SECONDS = 240.0   # one judge call (G-Eval and Ragas make several model calls inside it)
# LiteLLM's default request timeout is 6000 s and the system's calls set none, so one stalled response could
# block a ticket for over an hour. The eval bounds each model call; LiteLLM's own retries still apply.
MODEL_CALL_TIMEOUT_SECONDS = 90.0
_last_call = [0.0]
_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?:\s*['\"](\d+(?:\.\d+)?)s")


def from_toolkit(case: dict) -> dict:
    """A toolkit golden case (question, expected, category, must_cite, must_not_contain, expected_route) in the
    shape the scorers read. It carries no tool-use expectations or fact patterns, so it is scored on routing,
    guardrails, retrieval when it lists citations, and the reference answer (judges) when it expects an answer."""
    components = ["routing"] + (["retrieval"] if case.get("must_cite") else []) + \
        (["final_answer"] if case.get("expected_route") == "auto" else [])
    return {"id": case["id"], "category": case["category"], "ticket": case["question"],
            "expected_route": case["expected_route"], "must_cite": case.get("must_cite") or [],
            "must_not_contain": case.get("must_not_contain") or [], "reference_answer": case.get("expected", ""),
            "expected_facts": [], "components": components}


def load_golden(path: Path = None) -> list:
    raw = json.loads((path or GOLDEN_PATH).read_text(encoding="utf-8"))
    if isinstance(raw, list):                # the toolkit's format
        return [from_toolkit(c) for c in raw]
    items = raw["items"]
    assert len(items) == 20 and len({i["id"] for i in items}) == 20
    for c in COMPONENTS:     # at least five items per component
        assert sum(c in i["components"] for i in items) >= 5, f"fewer than 5 golden items score {c}"
    return items


@__import__("functools").lru_cache(maxsize=1)
def _titles() -> dict:
    from shopsense.config import CORPUS_DIR
    return {f"{d['slug']}.md": d["title"] for d in json.loads((CORPUS_DIR / "index.json").read_text())}


# Shared rate-limit gate

def _pace():
    elapsed = time.time() - _last_call[0]
    if elapsed < _MIN_GAP_SECONDS:
        time.sleep(_MIN_GAP_SECONDS - elapsed)
    _last_call[0] = time.time()


def _off_loop(fn, *args, timeout: float, **kwargs):
    """Run one judge call on a fresh daemon thread, with a hard timeout. The Langfuse task is async, so the
    evaluator runs on Langfuse's event-loop thread, where judges that call asyncio.run() fail; off that thread
    every judge gets its own loop, and a stuck call is abandoned and retried instead of freezing the run.
    (Timeouts use a monotonic clock that stops while the machine sleeps - run long evals under caffeinate.)"""
    box: dict = {}

    def target():
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as e:
            box["error"] = e
    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(f"judge call exceeded {timeout:.0f}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def with_backoff(fn, *args, max_retries: int = 4, **kwargs):
    for attempt in range(max_retries):
        _pace()
        try:
            return _off_loop(fn, *args, timeout=JUDGE_TIMEOUT_SECONDS, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            m = _RETRY_DELAY_RE.search(str(e))
            wait = float(m.group(1)) + 1.0 if m else (2 ** attempt) + random.random()
            print(f"  [judge retry {attempt + 1}/{max_retries}] waiting {wait:.1f}s: {type(e).__name__}")
            time.sleep(wait)


# Deterministic scorers

def route_match(item, trace) -> float:
    return 1.0 if trace["route"] == item["expected_route"] else 0.0


def risk_reasons_missing(item, trace) -> list:
    """Expected escalation reasons (regex each) that the ticket's risk list does not carry."""
    reasons = " | ".join(trace.get("risk") or [])
    return [p for p in item.get("expected_risk") or [] if not re.search(p, reasons, re.IGNORECASE)]


def must_not_contain_ok(item, trace) -> float:
    reply = (trace["reply"] or "").lower()
    return 0.0 if any(s.lower() in reply for s in item["must_not_contain"]) else 1.0


def citation_recall(item, trace) -> Optional[float]:
    if not item["must_cite"]:
        return None
    got = set(trace["retrieved_titles"])
    return sum(1 for t in item["must_cite"] if t in got) / len(item["must_cite"])


def facts_in_reply(item, trace) -> Optional[float]:
    if not item["expected_facts"]:
        return None
    reply = trace["reply"] or ""
    return sum(1 for p in item["expected_facts"] if re.search(p, reply, re.IGNORECASE)) / len(item["expected_facts"])


def tool_call_mismatches(item, trace) -> list:
    """Differences from the golden expectation (empty when correct): order looked up, proposed
    action, reason code, and the quoted amount when the item gives one.
    """
    tc = trace.get("tool_calls") or {}
    checks = [("order_ref", item["expected_order_ref"], tc.get("order_lookup")),
              ("action", item["expected_action"], tc.get("action")),
              ("reason_code", item["expected_reason_code"], tc.get("reason_code"))]
    wrong = [f"{k}: expected {e!r}, got {g!r}" for k, e, g in checks if e != g]
    want, got = item.get("expected_amount_inr"), tc.get("amount_inr")
    if want is not None and (got is None or abs(float(got) - want) > 0.01):
        wrong.append(f"amount_inr: expected {want!r}, got {got!r}")
    return wrong


def tool_call_match(item, trace) -> Optional[float]:
    """1.0 only when every expected tool argument matches."""
    if "expected_action" not in item:
        return None
    return 0.0 if tool_call_mismatches(item, trace) else 1.0


def retrieval_keyword_hit_rate(item, trace) -> Optional[float]:
    """Fraction of the expected keywords found in the retrieved text."""
    keywords = item.get("expected_keywords_in_context") or []
    if not keywords:
        return None
    combined = " ".join(trace["retrieved_texts"]).lower()
    return sum(1 for kw in keywords if kw.lower() in combined) / len(keywords)


# LLM judges

# OpenAI-compatible endpoints Ragas can reach for each LiteLLM provider prefix: (base_url, API-key variable).
JUDGE_ENDPOINTS = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "openai": (None, "OPENAI_API_KEY"),
    "anthropic": ("https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    # a local open-weight model: no key, no quota (Ollama serves an OpenAI-compatible API)
    "ollama": ((os.getenv("OLLAMA_API_BASE") or "http://localhost:11434").rstrip("/") + "/v1", None),
    "ollama_chat": ((os.getenv("OLLAMA_API_BASE") or "http://localhost:11434").rstrip("/") + "/v1", None),
}


def judge_endpoint(model: str) -> tuple:
    """(provider, model name without prefix, base_url, key variable) for the judge model."""
    import litellm
    name, provider, _, _ = litellm.get_llm_provider(model)
    if provider not in JUDGE_ENDPOINTS:
        raise ValueError(f"no OpenAI-compatible judge endpoint configured for provider {provider!r}; "
                         f"add it to JUDGE_ENDPOINTS or run with --no-judges")
    return (provider, name, *JUDGE_ENDPOINTS[provider])


@__import__("functools").lru_cache(maxsize=1)
def _judges():
    # Ragas 0.4.3 imports a symbol langchain_community removed; register a stand-in.
    stub = types.ModuleType("langchain_community.chat_models.vertexai")
    stub.ChatVertexAI = type("_UnusedChatVertexAI", (), {})
    sys.modules.setdefault("langchain_community.chat_models.vertexai", stub)
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    from deepeval.models import GeminiModel, LiteLLMModel
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import ContextPrecisionWithReference, Faithfulness
    from trulens.providers.litellm import LiteLLM as TruLiteLLM
    provider, name, base_url, key_var = judge_endpoint(JUDGE_MODEL)
    key = os.environ[key_var] if key_var else "ollama"        # local models need no key
    if not independent_of_system(JUDGE_MODEL):
        print(f"  note: the judges use {JUDGE_MODEL}, the same provider as the system ({LLM_MODEL}); set "
              f"SHOPSENSE_JUDGE_MODEL to a model from another provider for an independent judge", file=sys.stderr)
    ragas_llm = llm_factory(name, provider="openai", client=AsyncOpenAI(api_key=key, base_url=base_url))
    class NoLogprobsLiteLLM(LiteLLMModel):
        """G-Eval weights its score with token log-probabilities when the model offers them; Groq, Anthropic
        and Mistral do not, and raising AttributeError here makes G-Eval use its plain scored output."""
        def generate_raw_response(self, *a, **k):
            raise AttributeError("log_probs unsupported")

        async def a_generate_raw_response(self, *a, **k):
            raise AttributeError("log_probs unsupported")

    # the key is passed explicitly: without it DeepEval's LiteLLM wrapper falls back to GOOGLE_API_KEY for any provider
    judge_cls = LiteLLMModel if provider == "openai" else NoLogprobsLiteLLM
    deepeval_llm = GeminiModel(model=name, api_key=key) if provider == "gemini" else judge_cls(model=JUDGE_MODEL, api_key=key)
    return {"faithfulness": Faithfulness(llm=ragas_llm), "context_precision": ContextPrecisionWithReference(llm=ragas_llm),
            "deepeval": deepeval_llm, "trulens": TruLiteLLM(model_engine=JUDGE_MODEL)}


def deepeval_correctness(item, trace):
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, SingleTurnParams
    metric = GEval(name="Correctness", model=_judges()["deepeval"], threshold=0.5,
                   criteria=("Determine whether the actual output correctly answers the customer's ticket, matching "
                             "the facts (numbers, windows, fees, approval routes) in the expected output. Minor "
                             "rewording is fine; wrong or missing facts are not. A correct refusal matches a refusal."),
                   evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                                      SingleTurnParams.EXPECTED_OUTPUT])
    case = LLMTestCase(input=item["ticket"], actual_output=trace["reply"], expected_output=item["reference_answer"])
    score = with_backoff(metric.measure, case)
    return float(score), (metric.reason or "")[:500]


def _run_coro(make_coro):
    """Run a coroutine, including from inside a running event loop (the Langfuse task is async)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(make_coro())
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(make_coro())).result()


def ragas_faithfulness(trace) -> float:
    contexts = trace["retrieved_texts"] or ["(no policy retrieved)"]
    res = with_backoff(_run_coro, lambda: _judges()["faithfulness"].ascore(
        user_input=trace["ticket"], response=trace["reply"], retrieved_contexts=contexts))
    return float(res.value)


def trulens_groundedness(trace):
    """Returns (score clamped to [0, 1], comment). The raw value is kept in the comment when TruLens
    returns more than 1.
    """
    score, _ = with_backoff(_judges()["trulens"].groundedness_measure_with_cot_reasons,
                            source="\n\n".join(trace["retrieved_texts"]), statement=trace["reply"])
    return _clamped(score)


def _clamped(score):
    raw = float(score)
    clamped = min(1.0, max(0.0, raw))
    return clamped, (None if clamped == raw else f"raw TruLens score {raw:.3f} out of [0,1]; clamped")


def ragas_context_precision(item, trace) -> float:
    """Whether the chunks that matter for the reference answer are ranked first."""
    contexts = trace["retrieved_texts"] or ["(no policy retrieved)"]
    res = with_backoff(_run_coro, lambda: _judges()["context_precision"].ascore(
        user_input=trace["ticket"], reference=item["reference_answer"], retrieved_contexts=contexts))
    return float(res.value)


def trulens_context_relevance(trace):
    """Relevance of the retrieved policy text to the ticket. Returns (score, comment)."""
    if not trace["retrieved_texts"]:
        return 0.0, "no policy retrieved"      # an empty retrieval scores 0
    score, _ = with_backoff(_judges()["trulens"].context_relevance_with_cot_reasons,
                            question=trace["ticket"], context="\n\n".join(trace["retrieved_texts"]))
    return _clamped(score)


# System under test

TRACES: dict = {}
ONLY = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else set()
ENGINE = sys.argv[sys.argv.index("--engine") + 1] if "--engine" in sys.argv else "workflow"   # workflow | team
RUN_TAG = datetime.now().strftime("%H%M%S")
RUN_PREFIX = "" if ENGINE == "workflow" else f"{ENGINE}-"
SYSTEM_UNDER_TEST = {"workflow": "M5 workflow", "team": "M6 multi-agent team"}
assert ENGINE in SYSTEM_UNDER_TEST, f"--engine must be one of {sorted(SYSTEM_UNDER_TEST)}"


def _is_rate_limit(e: Exception) -> bool:
    return "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "RateLimit" in type(e).__name__


def error_trace(item: dict, error: Optional[str]) -> dict:
    """Trace for an item that failed to run, so every scorer fails it instead of dropping it."""
    return {"id": item["id"], "ticket": item["ticket"], "status": "error", "route": "error", "reply": "",
            "risk": [], "proposal": None, "tool_calls": {}, "retrieved_titles": [], "retrieved_texts": [], "error": error,
            "guardrail": {"passed": False, "flags": [f"task_error: {error}"], "action": "block", "route": "error"}}


def tool_calls(result: dict) -> dict:
    """Order-system calls behind a result: the order looked up, the quote reason and amount, and the
    proposed action. Anything other than a refund or replacement counts as 'none'.
    """
    proposal = result.get("proposal_full") or {}
    quote = proposal.get("quote") or {}
    action = proposal.get("action")
    return {"order_lookup": (result.get("order") or {}).get("order_ref"),
            "quote_reason": quote.get("reason"), "quote_amount_inr": quote.get("refund_amount_inr"),
            "action": action if action in ("refund", "replacement") else "none", "raw_action": action,
            "reason_code": proposal.get("reason_code") if action in ("refund", "replacement") else None,
            "amount_inr": quote.get("refund_amount_inr") if action == "refund" else None}


def make_trace(item: dict, result: dict) -> dict:
    """Flatten a workflow-shaped result into the trace the scorers read."""
    from shopsense.m3.rag import chunk
    hits = result.get("policy_hits") or []
    return {
        "id": item["id"], "ticket": item["ticket"], "status": result["status"], "route": system_route(result),
        "reply": result.get("draft") or "", "risk": result.get("risk") or [], "proposal": result.get("proposal"),
        "tool_calls": tool_calls(result), "retrieved_titles": [_titles().get(h["source"], h["source"]) for h in hits],
        "retrieved_texts": [h.get("text") or chunk(h["cid"])["text"] for h in hits],
        "guardrail": guardrail_check(item["ticket"], result),
    }


def run_item(item: dict) -> dict:
    """Run one golden ticket through the workflow and return its trace."""
    from shopsense.m5.workflow import start_ticket
    result, error = None, None
    for attempt in range(3):
        try:   # a fresh ticket id per attempt
            result = start_ticket(item["ticket"], f"EVAL-{RUN_TAG}-{item['id']}-a{attempt}", item.get("customer_ref"))
            break
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)[:200]}"
            if _is_rate_limit(e):
                print(f"  [{item['id']}] rate limited - waiting 65s for the quota window (attempt {attempt + 1}/3)")
                time.sleep(65)
                continue
            break
    trace = error_trace(item, error) if result is None else make_trace(item, result)
    TRACES[item["id"]] = trace
    return trace


async def run_item_team(item: dict) -> dict:
    """Run one golden ticket through the multi-agent team (one MCP session)."""
    from shopsense.m6.team import run_ticket
    final, error = None, None
    for attempt in range(3):
        try:
            final = await run_ticket(item["ticket"], f"EVAL-TEAM-{RUN_TAG}-{item['id']}-a{attempt}",
                                     item.get("customer_ref"))
            break
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)[:200]}"
            if _is_rate_limit(e):
                print(f"  [{item['id']}] rate limited - waiting 65s for the quota window (attempt {attempt + 1}/3)")
                await asyncio.sleep(65)
                continue
            break
    if final is None:
        trace = error_trace(item, error)
    else:
        result = team_to_result(final)
        trace = {**make_trace(item, result), "revision_count": result["revision_count"],
                 "llm_fallbacks": result["llm_fallbacks"], "team_log": result["log"]}
    TRACES[item["id"]] = trace
    return trace


def _judge(scores: list, name: str, fn) -> None:
    """One judge call; if it fails, only that metric is skipped."""
    try:
        value, comment = fn()
        scores.append((name, value, comment))
    except Exception as e:
        print(f"  [judge failed, skipping] {name}: {type(e).__name__}: {str(e)[:160]}")


def component_scorer(item: dict, trace: dict, use_judges: bool) -> list:
    """Score each component the item covers, plus guardrails and routing on every item.
    Returns [(name, value, comment)]. Judges skip errored items; deterministic metrics fail them.
    """
    comps = set(item["components"])
    judged = use_judges and trace["status"] != "error"
    scores = [("route_match", route_match(item, trace), f"expected {item['expected_route']}, got {trace['route']}"),
              ("must_not_contain_ok", must_not_contain_ok(item, trace), None)]
    if item.get("expected_risk"):      # held for the right reasons, not just held
        missing = risk_reasons_missing(item, trace)
        scores.append(("risk_reasons_match", 1 - len(missing) / len(item["expected_risk"]),
                       ("missing: " + ", ".join(missing)) if missing else None))
    g = trace["guardrail"]
    scores.append(("guardrail_passed", 1.0 if g["passed"] else 0.0, "; ".join(g["flags"]) or "clean"))
    if "tool_use" in comps:
        wrong = tool_call_mismatches(item, trace)
        scores.append(("tool_call_match", 0.0 if wrong else 1.0, "; ".join(wrong) or None))
    if "retrieval" in comps:
        for name, fn in (("citation_recall", citation_recall), ("retrieval_keyword_hit_rate", retrieval_keyword_hit_rate)):
            v = fn(item, trace)
            if v is not None:
                scores.append((name, v, None))
        if judged:
            _judge(scores, "ragas_context_precision", lambda: (ragas_context_precision(item, trace), None))
            _judge(scores, "trulens_context_relevance", lambda: trulens_context_relevance(trace))
    if "final_answer" in comps:
        v = facts_in_reply(item, trace)
        if v is not None:
            scores.append(("facts_in_reply", v, None))
        if judged:
            _judge(scores, "deepeval_geval_correctness", lambda: deepeval_correctness(item, trace))
            if item["expected_route"] != "refuse":     # a correct refusal has no policy to be grounded in
                _judge(scores, "ragas_faithfulness", lambda: (ragas_faithfulness(trace), None))
                _judge(scores, "trulens_groundedness", lambda: trulens_groundedness(trace))
    return scores


def component_table(golden: list, scores_by_id: dict) -> dict:
    """{component: {"items": n, "metrics": {metric: {"n", "mean"}}}} over the items scoring that
    component (guardrails: every item).
    """
    table = {}
    for comp, metrics in COMPONENT_METRICS.items():
        ids = [i["id"] for i in golden if i["id"] in scores_by_id and (comp == "guardrails" or comp in i["components"])]
        vals = defaultdict(list)
        for iid in ids:
            for n, v, _ in scores_by_id[iid]:
                if n in metrics:
                    vals[n].append(v)
        table[comp] = {"items": len(ids), "metrics": {m: {"n": len(vals[m]), "mean": round(sum(vals[m]) / len(vals[m]), 3)}
                                                      for m in metrics if vals[m]}}
    return table


def run(use_judges: bool = True) -> dict:
    from shopsense.tracing import langfuse as langfuse_client
    from langfuse.experiment import Evaluation
    import litellm
    litellm.request_timeout = MODEL_CALL_TIMEOUT_SECONDS
    golden = load_golden()
    by_id = {i["id"]: i for i in golden}
    if ONLY:
        golden = [i for i in golden if i["id"] in ONLY]
    langfuse = langfuse_client()
    assert langfuse.auth_check(), "Langfuse auth failed - check keys and LANGFUSE_BASE_URL region"
    data = [{"input": i, "expected_output": i["reference_answer"],
             "metadata": {"id": i["id"], "category": i["category"], "components": i["components"],
                          "expected_route": i["expected_route"]}}
            for i in golden]
    SCORES: dict = {}

    async def task(*, item, **kwargs):     # the team runs inside Langfuse's event loop
        golden_item = item["input"] if isinstance(item, dict) else item.input
        print(f"- {golden_item['id']} ({golden_item['category']})", flush=True)
        return await run_item_team(golden_item) if ENGINE == "team" else run_item(golden_item)

    def evaluator(*, input, output, expected_output, metadata, **kwargs):
        scores = component_scorer(input, output, use_judges)
        SCORES[input["id"]] = scores
        return [Evaluation(name=n, value=float(v), comment=(c or None)) for n, v, c in scores]

    result = langfuse.run_experiment(name="shopsense-m8-golden-eval",
                                     run_name=f"{RUN_PREFIX}run-{RUN_TAG}" + (f"-only-{'-'.join(sorted(ONLY))}" if ONLY else ""), data=data,
                                     task=task, evaluators=[evaluator], max_concurrency=1,
                                     description=f"ShopSense {SYSTEM_UNDER_TEST[ENGINE]} on golden_set_v2 (20 tickets)")
    langfuse.flush()

    # Report
    per_metric = defaultdict(list)
    per_cat_route = defaultdict(list)
    for iid, scores in SCORES.items():
        for n, v, _ in scores:
            per_metric[n].append(v)
        per_cat_route[by_id[iid]["category"]].append(next(v for n, v, _ in scores if n == "route_match"))
    summary = {n: {"n": len(v), "mean": round(sum(v) / len(v), 3)} for n, v in sorted(per_metric.items())}
    route_by_cat = {c: f"{int(sum(v))}/{len(v)}" for c, v in sorted(per_cat_route.items())}
    by_component = component_table(golden, SCORES)
    failures = []
    for iid, scores in sorted(SCORES.items()):
        for n, v, c in scores:
            if n in DETERMINISTIC and v < 1:
                failures.append({"id": iid, "metric": n, "value": v, "detail": c,
                                 "reply": TRACES[iid]["reply"][:240], "risk": TRACES[iid]["risk"],
                                 "tool_calls": TRACES[iid].get("tool_calls")})
    disagreements = []
    for iid, scores in SCORES.items():
        d = {n: v for n, v, _ in scores}
        det = d.get("facts_in_reply")
        judge = d.get("deepeval_geval_correctness")
        if det is not None and judge is not None and abs(det - judge) > 0.4:
            disagreements.append({"id": iid, "facts_in_reply": det, "geval": round(judge, 2)})

    report = {"run_at": datetime.now().isoformat(timespec="seconds"), "model": LLM_MODEL, "engine": ENGINE,
              "judge_model": JUDGE_MODEL if use_judges else None,
              "judge_independent": independent_of_system(JUDGE_MODEL) if use_judges else None,
              "system_under_test": SYSTEM_UNDER_TEST[ENGINE], "judges": use_judges, "items_scored": len(SCORES),
              "errored": [i for i, t in TRACES.items() if t["status"] == "error"],
              "golden_set": str(GOLDEN_PATH.relative_to(PROJECT_ROOT)) if GOLDEN_PATH.is_relative_to(PROJECT_ROOT) else str(GOLDEN_PATH), "metrics_by_component": by_component,
              "metrics": summary, "route_accuracy_by_category": route_by_cat, "failures": failures,
              "judge_vs_deterministic_disagreements": disagreements,
              "judge_audit": audit_judge({iid: {"scores": [{"name": n, "value": v} for n, v, _ in SCORES[iid]]}
                                          for iid in SCORES}) if use_judges else None,
              "items": {iid: {"trace": {k: v for k, v in TRACES[iid].items() if k != "retrieved_texts"},
                              "scores": [{"name": n, "value": v, "comment": c} for n, v, c in SCORES[iid]]}
                        for iid in SCORES}}
    out = REPORT_DIR / f"m8_eval_{RUN_PREFIX}{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))

    print(f"\n{'component':14s} {'items':>5s}  {'metric':30s} {'n':>4s} {'mean':>7s}")
    for comp, row in by_component.items():
        for k, (n, s) in enumerate(row["metrics"].items()):
            print(f"{comp if k == 0 else '':14s} {row['items'] if k == 0 else '':>5}  {n:30s} {s['n']:>4d} {s['mean']:>7.3f}")
    errored = [i for i, t in TRACES.items() if t["status"] == "error"]
    print(f"\nitems scored: {len(SCORES)}/{len(golden)}   errored: {errored or 'none'}")
    print(f"route accuracy by category: {route_by_cat}")
    print(f"\nfailures ({len(failures)}):")
    for f in failures:
        print(f"  {f['id']:7s} {f['metric']:26s} {f['value']:.2f}  {str(f['detail'])[:150]}")
    if ENGINE == "team":
        fallbacks = {i: t["llm_fallbacks"] for i, t in TRACES.items() if t.get("llm_fallbacks")}
        print(f"team: revisions {sum(t.get('revision_count', 0) for t in TRACES.values())} total; "
              f"model calls that fell back to code: {fallbacks or 'none'}")
    print(f"\njudge vs deterministic disagreements (>0.4): {disagreements}")
    print(f"\nLangfuse experiment: shopsense-m8-golden-eval / {RUN_PREFIX}run-{RUN_TAG}   report: {out.relative_to(PROJECT_ROOT)}")
    return report


def self_check():
    golden = load_golden(V2_PATH)           # the self-check always validates our own set
    item = next(i for i in golden if i["id"] == "SS-F02")
    good = {"route": "auto", "reply": "Electronics have a 15-day return window.",
            "retrieved_titles": ["Category Policy Addendum: Electronics"]}
    bad = {"route": "human_review", "reply": "You have 14 days.", "retrieved_titles": []}
    assert route_match(item, good) == 1 and route_match(item, bad) == 0
    assert must_not_contain_ok(item, good) == 1 and must_not_contain_ok(item, bad) == 0
    assert citation_recall(item, good) == 1 and citation_recall(item, bad) == 0
    assert facts_in_reply(item, good) == 1 and facts_in_reply(item, bad) == 0
    team = {"ticket_id": "T", "status": "escalated_to_human", "draft": "A specialist will review this.",
            "findings": [{"cid": 0, "source": "returns-policy.md", "text": "..."}],
            "order_facts": {"order_ref": "KW-O-1", "order_value_inr": 12999, "sla": {}},
            "resolution": {"action": "refund", "reason_code": "damaged", "quote": {"refund_amount_inr": 12999.0}},
            "review": {"risk": ["refund above the ₹2,000 automated cap"]}, "log": ["supervisor[policy] -> triage"]}
    r = team_to_result(team)
    assert system_route(r) == "human_review" and r["proposal"]["amount_inr"] == 12999.0 and r["llm_fallbacks"] == 1
    assert r["policy_hits"] == [{"cid": 0, "source": "returns-policy.md"}] and "sla" not in r["order"]
    assert tool_calls(r) == {"order_lookup": "KW-O-1", "quote_reason": None, "quote_amount_inr": 12999.0,
                             "action": "refund", "raw_action": "refund", "reason_code": "damaged", "amount_inr": 12999.0}
    # SS-I02: the real quote (7 days late, 25% of ₹799) vs the injected ₹5,000
    i02 = next(i for i in golden if i["id"] == "SS-I02")
    m5_result = {"order": {"order_ref": "KW-O-000046"}, "proposal_full": {
        "action": "refund", "reason_code": "late_delivery", "quote": {"reason": "late_delivery", "refund_amount_inr": 199.75}}}
    assert tool_call_match(i02, {"tool_calls": tool_calls(m5_result)}) == 1
    m5_result["proposal_full"]["quote"]["refund_amount_inr"] = 5000.0
    assert tool_call_match(i02, {"tool_calls": tool_calls(m5_result)}) == 0
    assert retrieval_keyword_hit_rate(item, {"retrieved_texts": ["return window is strictly **15 calendar days**"]}) == 0.5
    from collections import Counter
    print(f"M8 eval self-check: PASS - {len(golden)} items by scenario {dict(Counter(i['category'] for i in golden))}, "
          f"by component {dict(Counter(c for i in golden for c in i['components']))}; deterministic scorers incl. "
          "tool_call_match and retrieval_keyword_hit_rate verified on synthetic traces; M6 team state maps onto the "
          "M5 result shape.")


REFERENCE_METRICS = ("route_match", "facts_in_reply", "must_not_contain_ok")   # scored against the golden answers


def audit_judge(items: dict, judge: str = "deepeval_geval_correctness", threshold: float = 0.5) -> dict:
    """Agreement between the LLM judge and the golden set's human-written answers.

    Reference verdict per ticket: the reply passes every check written from the golden answer (route,
    expected facts, forbidden phrases). Judge verdict: score >= threshold. The expensive disagreement
    is the judge passing a reply the reference fails.
    """
    cells = {"pass_pass": [], "fail_fail": [], "judge_pass_ref_fail": [], "judge_fail_ref_pass": []}
    for iid, item in sorted(items.items()):
        scores = {s["name"]: s["value"] for s in item["scores"]}
        ref = [scores[m] for m in REFERENCE_METRICS if m in scores]
        if judge not in scores or not ref:
            continue
        ref_pass, judge_pass = all(v >= 1 for v in ref), scores[judge] >= threshold
        key = ("pass_pass" if judge_pass else "judge_fail_ref_pass") if ref_pass else \
              ("judge_pass_ref_fail" if judge_pass else "fail_fail")
        cells[key].append(iid)
    n = sum(len(v) for v in cells.values())
    agree = len(cells["pass_pass"]) + len(cells["fail_fail"])
    return {"judge": judge, "threshold": threshold, "reference": list(REFERENCE_METRICS), "n": n,
            "agreement": round(agree / n, 3) if n else None, **{k: v for k, v in cells.items()}}


def audit_judge_live(path=None) -> dict:
    """Run the G-Eval judge on human-labelled correct and wrong replies (eval/judge_audit_cases.json)."""
    spec = json.loads((path or PROJECT_ROOT / "eval" / "judge_audit_cases.json").read_text(encoding="utf-8"))
    golden = {i["id"]: i for i in load_golden(V2_PATH)}
    rows = []
    for case in spec["cases"]:
        score, reason = deepeval_correctness(golden[case["golden_id"]], {"reply": case["reply"]})
        rows.append({"id": case["golden_id"], "label": case["label"], "score": round(score, 3),
                     "judge": "pass" if score >= 0.5 else "fail", "reason": reason[:300]})
        print(f"  {case['golden_id']} human={case['label']:<4} judge={rows[-1]['judge']:<4} score={score:.2f}", flush=True)
        time.sleep(float(os.getenv("SHOPSENSE_JUDGE_PAUSE_SECONDS", "0")))     # e.g. 8 on a free Groq tier
    agree = sum(r["label"] == r["judge"] for r in rows)
    return {"judge": "deepeval_geval_correctness", "judge_model": JUDGE_MODEL, "system_model": LLM_MODEL,
            "judge_independent": independent_of_system(JUDGE_MODEL), "n": len(rows),
            "agreement": round(agree / len(rows), 3),
            "judge_pass_human_fail": [r["id"] for r in rows if r["judge"] == "pass" and r["label"] == "fail"],
            "judge_fail_human_pass": [r["id"] for r in rows if r["judge"] == "fail" and r["label"] == "pass"],
            "rows": rows}


def main():
    if "--audit-judge-live" in sys.argv:       # the judge on hand-labelled replies (about 20 model calls)
        report = audit_judge_live()
        from shopsense.tracing import log_eval_scores
        log_eval_scores("eval:judge-audit", {"judge_human_agreement": report["agreement"],
                                             "judge_passed_wrong_replies": len(report["judge_pass_human_fail"]),
                                             "judge_failed_correct_replies": len(report["judge_fail_human_pass"])},
                        {"judge_model": report["judge_model"], "system_model": report["system_model"],
                         "independent": report["judge_independent"]})
        out = REPORT_DIR / f"judge_audit_{datetime.now():%Y%m%d_%H%M%S}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Judge audit: agreement with human labels {report['agreement']:.1%} (n={report['n']}); "
              f"judge passed a wrong reply: {report['judge_pass_human_fail'] or 'none'}; "
              f"judge failed a correct reply: {report['judge_fail_human_pass'] or 'none'}\nreport: {out}")
        return
    if "--audit-judge" in sys.argv:            # python -m shopsense.m8.eval --audit-judge <report.json>
        path = Path(sys.argv[sys.argv.index("--audit-judge") + 1])
        audit = audit_judge(json.loads(path.read_text(encoding="utf-8"))["items"])
        print(f"Judge audit on {path.name}: {audit['judge']} vs the golden answers, n={audit['n']}, "
              f"agreement {audit['agreement']:.1%}")
        for k in ("pass_pass", "fail_fail", "judge_pass_ref_fail", "judge_fail_ref_pass"):
            print(f"  {k:<22}{len(audit[k]):>3}  {', '.join(audit[k])}")
        return
    self_check()
    if "--offline" in sys.argv:
        return
    import subprocess
    import tempfile
    if "--phase" not in sys.argv:          # tickets and actions go to a temp runtime dir
        with tempfile.TemporaryDirectory() as d:
            args = [sys.executable, "-m", "shopsense.m8.eval", "--phase", *[a for a in sys.argv[1:]]]
            env = {**os.environ, "SHOPSENSE_RUNTIME_DIR_REPORTS": str(RUNTIME_DIR), "SHOPSENSE_RUNTIME_DIR": d,
                   "LLM_CALL_PAUSE_SECONDS": os.getenv("LLM_CALL_PAUSE_SECONDS", "6")}
            raise SystemExit(subprocess.run(args, env=env, timeout=5400).returncode)
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)   # `kill -USR1 <pid>` prints every thread's stack
    run(use_judges="--no-judges" not in sys.argv)


if __name__ == "__main__":
    main()
