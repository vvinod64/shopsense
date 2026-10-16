"""Langfuse tracing helpers.

traced() wraps workflow nodes and agents in spans, span() adds nested spans, ticket_trace() opens the
root span for a ticket, and llm_generation() records model calls with token usage. RUN_EVENTS keeps
local timing records for the per-agent table. SHOPSENSE_TRACING=0 disables tracing.

Everything sent to Langfuse passes through redact() first (emails, phone, card, Aadhaar and PAN numbers),
and every ticket trace is tagged with the model and the prompt version, so one run can be found later.
"""
import contextvars
import functools
import hashlib
import inspect
import os
import re
import time
from contextlib import contextmanager
from functools import lru_cache
from typing import Optional

from shopsense import config  # noqa: F401  (loads .env)

RUN_EVENTS: list[dict] = []   # {"role", "ms", "ok", "tokens", "cost_usd"} per traced call
_CURRENT_EVENT: contextvars.ContextVar = contextvars.ContextVar("shopsense_current_event", default=None)


def enabled() -> bool:
    return os.getenv("SHOPSENSE_TRACING", "1") != "0" and bool(os.getenv("LANGFUSE_PUBLIC_KEY"))


# Customer data is redacted before it leaves the process, not in the dashboard. Order and customer
# references (KW-O-..., KW-C-...) and rupee amounts are kept: they are what a trace is searched by.
# Pattern-based: names are caught after introductions, titles and sign-offs, addresses by unit numbers,
# street words and PIN codes; a bare name in mid-sentence is not (that would need an entity recogniser).
_DATE = (r"(?:\d{1,4}[-/. ](?:\d{1,2}|[A-Za-z]{3,9})[-/. ]\d{2,4}|\d{1,2}(?:st|nd|rd|th)?(?: of)? [A-Za-z]{3,9},? \d{2,4}|"
         r"[A-Za-z]{3,9} \d{1,2}(?:st|nd|rd|th)?,? \d{4})")
# "DOB 12-Mar-88", "my DOB is 12-Mar-1988", "date of birth: 12/03/1988", "born on the 12th of March 1988"
_DOB_LABEL = (r"(?i)\b(dob|d\.o\.b\.?|date of birth|birth ?date|birthday|born(?: on)?)"
              r"(\W{0,3}(?:(?:is|was|=|on|:)\W{0,3})?(?:the\s+)?)")
REDACTIONS = [
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "[EMAIL]"),
    (re.compile(r"\b[\w.-]{2,}@[A-Za-z]{2,}\b"), "[UPI]"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[PAN]"),
    (re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"), "[IFSC]"),
    (re.compile(_DOB_LABEL + _DATE), r"\1\2[DOB]"),
    (re.compile(r"\b(?i:pin ?code|pincode|postal code|zip(?: code)?|pin)\W{0,3}[1-9]\d{2}\s?\d{3}\b"), "[PIN]"),
    (re.compile(r"(?i)(?<!\[)\b(password|passwd|pwd|passcode|pass code|pin|otp|cvv|cvc)(\s*(?:is|was|:|=|-)?\s*)\S+"),
     r"\1\2[SECRET]"),
    (re.compile(r"(?i)\b(ending(?: in| with)?|ends(?: in| with)|last (?:4|four) digits(?: are| is)?)(\s*:?\s*)\d{4}\b"),
     r"\1\2[LAST4]"),
    (re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}(?:[ -]?\d{1,7})?\b"), "[CARD/AADHAAR]"),
    (re.compile(r"(?<![\w-])(?:\+91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}\b"), "[PHONE]"),
    (re.compile(r"(?<![\w-])\d{9,18}\b"), "[ACCOUNT]"),
    # Addresses: unit numbers, street lines and PIN codes next to an address word
    (re.compile(r"\b(?i:flat|house|h\.? ?no\.?|door no\.?|plot|apartment|apt\.?)\s*(?i:no\.?)?\s*#?\s*\w{1,6}\b"), "[ADDRESS]"),
    (re.compile(r"\b\d{1,4}(?:st|nd|rd|th)?[A-Za-z]?,?\s+(?:[A-Z][\w.]*\s+){0,3}(?i:road|rd\.?|street|st\.|lane|nagar|"
                r"colony|layout|marg|avenue|cross|main road|block|sector)\b"), "[ADDRESS]"),
    (re.compile(r"\b(?i:sector|block|phase|ward)\s*-?\s*\d{1,3}[A-Za-z]?\b"), "[ADDRESS]"),
    (re.compile(r"([A-Z][a-z]+\s*[-,]\s*)[1-9]\d{5}\b"), r"\1[PIN]"),
    # Names: after self-introductions, titles and sign-offs
    (re.compile(r"\b((?i:my name is|i am|i'm|name\s*:)\s+)(?!(?:Platinum|Gold|Silver|Bronze|Premium|Prime|VIP|Very|Really|So|"
                r"Not|Extremely|Still|Writing|Contacting|Sorry|Happy|Unhappy|Disappointed|Frustrated|Angry|Furious|Upset|"
                r"Done|Fed|Tired|Waiting|Unable|The|A|An)\b)[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}"), r"\1[NAME]"),
    (re.compile(r"\b((?:Mr|Mrs|Ms|Miss|Dr|Shri|Smt)\.?\s+)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"), r"\1[NAME]"),
    (re.compile(r"((?i:regards|thanks|thank you|cheers|sincerely|warm regards|best)\s*,?\s*\n?\s*)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s*$"),
     r"\1[NAME]"),
    (re.compile(r"(\s[-–—]\s*)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s*$"), r"\1[NAME]"),
]


def redact(text: str) -> str:
    for pattern, label in REDACTIONS:
        text = pattern.sub(label, text)
    return text


def _mask(data, **_):
    """Langfuse mask hook: applied to every input, output and metadata payload before export."""
    if isinstance(data, str):
        return redact(data)
    if isinstance(data, dict):
        return {k: _mask(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_mask(v) for v in data]
    return data


PROMPTS: dict = {}   # name -> text of every system prompt in use; their hash is the prompt version


def register_prompt(name: str, text: str) -> str:
    PROMPTS[name] = text
    return text


PROMPT_MODULES = ("shopsense.m1.parser", "shopsense.m2.agent", "shopsense.m5.workflow")


def prompt_version() -> str:
    """Hash of every system prompt, the same in every process whichever modules it happened to import."""
    import importlib
    for name in PROMPT_MODULES:
        importlib.import_module(name)
    joined = "\n".join(f"{k}:{PROMPTS[k]}" for k in sorted(PROMPTS))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:8]


class TailSamplingExporter:
    """Keep every trace that went wrong and a fixed share of healthy ones (SHOPSENSE_TRACE_SAMPLE_RATE).

    A trace is decided when its root span ends, so the whole ticket is kept or dropped together. Kept always:
    any span with an error status or at WARNING/ERROR level (fallbacks, held tickets, failed checks). Healthy
    traces are sampled by trace id, so the same trace gets the same decision everywhere.
    """
    LEVEL = "langfuse.observation.level"

    def __init__(self, inner, rate: float, max_buffered: int = 20000):
        self.inner, self.rate, self.max_buffered = inner, rate, max_buffered
        self._buffer: dict = {}
        self._decided: dict = {}
        self._lock = __import__("threading").Lock()
        self.kept = self.dropped = 0

    @staticmethod
    def _bad(span) -> bool:
        from opentelemetry.trace import StatusCode
        return (span.status.status_code == StatusCode.ERROR
                or (span.attributes or {}).get(TailSamplingExporter.LEVEL) in ("WARNING", "ERROR"))

    def _keep(self, trace_id: int, spans: list) -> bool:
        return any(self._bad(sp) for sp in spans) or (trace_id % 10_000) < self.rate * 10_000

    def export(self, spans):
        from opentelemetry.sdk.trace.export import SpanExportResult
        out = []
        with self._lock:
            for sp in spans:
                tid = sp.context.trace_id
                if tid in self._decided:
                    if self._decided[tid]:
                        out.append(sp)
                    continue
                self._buffer.setdefault(tid, []).append(sp)
                if sp.parent is None:                      # the root span ended: decide the whole trace
                    group = self._buffer.pop(tid)
                    keep = self._keep(tid, group)
                    self._decided[tid] = keep
                    if len(self._decided) > 50_000:
                        self._decided.pop(next(iter(self._decided)))
                    self.kept += keep
                    self.dropped += not keep
                    out += group if keep else []
            if sum(len(v) for v in self._buffer.values()) > self.max_buffered:   # never hold memory unbounded
                for group in self._buffer.values():
                    out += group
                self._buffer.clear()
        return self.inner.export(out) if out else SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        with self._lock:                                   # undecided traces are kept, never lost
            pending = [sp for group in self._buffer.values() for sp in group]
            self._buffer.clear()
        if pending:
            self.inner.export(pending)
        return self.inner.force_flush(timeout_millis) if hasattr(self.inner, "force_flush") else True

    def shutdown(self) -> None:
        self.force_flush()
        self.inner.shutdown()


def _langfuse_exporter():
    """The exporter Langfuse would build itself (OTLP to the project's endpoint)."""
    import base64
    from langfuse._version import __version__
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    pk, sk = os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"]
    base = (os.getenv("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com").rstrip("/")
    return OTLPSpanExporter(endpoint=f"{base}/api/public/otel/v1/traces", headers={
        "Authorization": "Basic " + base64.b64encode(f"{pk}:{sk}".encode()).decode("ascii"),
        "x-langfuse-sdk-name": "python", "x-langfuse-sdk-version": __version__, "x-langfuse-public-key": pk})


@lru_cache(maxsize=1)
def langfuse():
    """The one Langfuse client, created with the redaction mask (other modules must use this one). With
    SHOPSENSE_TRACE_SAMPLE_RATE below 1, healthy traces are sampled and every failure is kept."""
    from shopsense import config  # noqa: F401  (loads .env)
    from langfuse import Langfuse
    rate = float(os.getenv("SHOPSENSE_TRACE_SAMPLE_RATE", "1"))
    if rate >= 1:
        return Langfuse(mask=_mask)
    return Langfuse(mask=_mask, span_exporter=TailSamplingExporter(_langfuse_exporter(), rate))


class _NullSpan:
    trace_id = None
    id = None

    def update(self, **_):
        return self

    def score_trace(self, **_):
        pass

    def score(self, **_):
        pass


@contextmanager
def span(name: str, as_type: str = "span", **attrs):
    """Nested observation under the active span."""
    if not enabled():
        yield _NullSpan()
        return
    with langfuse().start_as_current_observation(as_type=as_type, name=name, **attrs) as s:
        yield s


def _summarise(value, limit: int = 600):
    """Keep span payloads small and JSON-safe."""
    if isinstance(value, dict):
        return {k: _summarise(v, limit) for k, v in list(value.items())[:25]}
    if isinstance(value, (list, tuple)):
        return [_summarise(v, limit) for v in list(value)[:10]]
    if isinstance(value, str):
        return value[:limit]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:limit]


PAUSE_EXCEPTIONS = {"GraphInterrupt", "NodeInterrupt"}   # LangGraph human-in-the-loop pause


def _mark(event: dict, exc: Exception) -> None:
    """A LangGraph interrupt is a pause for a human, not a failure."""
    if type(exc).__name__ in PAUSE_EXCEPTIONS:
        event["paused"] = True
    else:
        event["ok"] = False


def traced(role: str, input_keys: tuple = ()):
    """Wrap a sync or async node in an agent:<role> span and append a timing record to RUN_EVENTS,
    including on failure.
    """
    def decorator(fn):
        def _open(state):
            event = {"role": role, "ms": 0.0, "ok": True, "tokens": 0, "input_tokens": 0, "cached_tokens": 0,
                     "cost_usd": 0.0}
            token = _CURRENT_EVENT.set(event)
            inp = {k: _summarise(state.get(k)) for k in input_keys if isinstance(state, dict) and k in state}
            return event, token, inp

        def _close(event, token, t0):
            event["ms"] = (time.perf_counter() - t0) * 1000
            RUN_EVENTS.append(event)
            _CURRENT_EVENT.reset(token)

        def _output(result):
            return {"wrote": sorted(set(result or {}) - {"log"}), "log": _summarise((result or {}).get("log", []))}

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def awrapper(state):
                t0 = time.perf_counter()
                event, token, inp = _open(state)
                try:
                    with span(f"agent:{role}", input=inp) as s:
                        result = await fn(state)
                        s.update(output=_output(result))
                        return result
                except Exception as e:
                    _mark(event, e)
                    raise
                finally:
                    _close(event, token, t0)
            return awrapper

        @functools.wraps(fn)
        def wrapper(state):
            t0 = time.perf_counter()
            event, token, inp = _open(state)
            try:
                with span(f"agent:{role}", input=inp) as s:
                    result = fn(state)
                    s.update(output=_output(result))
                    return result
            except Exception as e:
                _mark(event, e)
                raise
            finally:
                _close(event, token, t0)
        return wrapper
    return decorator


def usage_from_ai_message(msg) -> Optional[dict]:
    u = getattr(msg, "usage_metadata", None)
    if not u:
        return None
    usage = {"input": u.get("input_tokens", 0), "output": u.get("output_tokens", 0), "total": u.get("total_tokens", 0)}
    cached = (u.get("input_token_details") or {}).get("cache_read")
    return usage | ({"input_cached": cached} if cached else {})


def usage_from_litellm(resp) -> Optional[dict]:
    u = getattr(resp, "usage", None)
    if not u:
        return None
    usage = {"input": getattr(u, "prompt_tokens", 0) or 0, "output": getattr(u, "completion_tokens", 0) or 0,
             "total": getattr(u, "total_tokens", 0) or 0}
    cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", None)
    return usage | ({"input_cached": cached} if cached else {})


def estimate_cost(model: str, usage: Optional[dict]) -> Optional[dict]:
    """USD cost of one call at the provider's list price (LiteLLM price map); None if the model is not priced.
    On the Gemini free tier nothing is billed, so this is what the same traffic would cost on a paid key.
    """
    if not usage:
        return None
    try:
        import litellm
        inp, out = litellm.cost_per_token(model=model, prompt_tokens=int(usage.get("input", 0) or 0),
                                          completion_tokens=int(usage.get("output", 0) or 0))
    except Exception:
        return None
    return {"input": inp, "output": out, "total": inp + out}


@contextmanager
def llm_generation(purpose: str, model: str, input=None):
    """Generation span for one LLM call. Call .record(output, usage) on the handle."""
    class _Handle:
        def __init__(self, s):
            self._s = s

        def record(self, output=None, usage: Optional[dict] = None):
            kwargs = {"output": _summarise(output)}
            if usage:
                kwargs["usage_details"] = usage
                cost = estimate_cost(model, usage)
                if cost:
                    kwargs["cost_details"] = cost
                event = _CURRENT_EVENT.get()
                if event is not None:
                    event["tokens"] += int(usage.get("total", 0) or 0)
                    event["input_tokens"] = event.get("input_tokens", 0) + int(usage.get("input", 0) or 0)
                    event["cached_tokens"] = event.get("cached_tokens", 0) + int(usage.get("input_cached", 0) or 0)
                    event["cost_usd"] = event.get("cost_usd", 0.0) + (cost["total"] if cost else 0.0)
            self._s.update(**kwargs)

    with span(f"llm:{purpose}", as_type="generation", model=model, input=_summarise(input)) as s:
        yield _Handle(s)


@contextmanager
def ticket_trace(workflow: str, ticket_id: str, customer_ref: Optional[str] = None, tags: tuple = (), input=None):
    """Root span for a ticket; session, user and tags propagate to child spans."""
    if not enabled():
        yield _NullSpan()
        return
    from langfuse import propagate_attributes
    with langfuse().start_as_current_observation(as_type="span", name=f"ticket:{workflow}",
                                                 input=_summarise(input)) as root:
        from shopsense.config import LLM_MODEL
        with propagate_attributes(session_id=ticket_id, user_id=customer_ref or "anonymous",
                                  tags=["shopsense", workflow, *tags], version=prompt_version(),
                                  metadata={"model": LLM_MODEL, "prompt_version": prompt_version()}):
            yield root


def log_eval_scores(name: str, scores: dict, details: Optional[dict] = None) -> Optional[str]:
    """One evaluator trace in Langfuse with a numeric score per metric (guardrail, injection and judge audits)."""
    if not enabled():
        return None
    try:
        with langfuse().start_as_current_observation(as_type="evaluator", name=name) as s:
            s.update(input=_summarise(details or {}), output=scores)
            for metric, value in scores.items():
                if isinstance(value, (int, float)):
                    s.score_trace(name=metric, value=float(value), data_type="NUMERIC")
            trace_id = s.trace_id
        langfuse().flush()
        return trace_id
    except Exception as e:
        print(f"  Langfuse logging failed: {type(e).__name__}")
        return None


def flush():
    if enabled():
        langfuse().flush()


def dashboard(events: Optional[list] = None):
    """Per-agent latency, tokens, prompt-cache hit rate and estimated cost from RUN_EVENTS."""
    import pandas as pd
    df = pd.DataFrame(events if events is not None else RUN_EVENTS)
    if df.empty:
        return df
    for col in ("cost_usd", "input_tokens", "cached_tokens"):
        if col not in df:
            df[col] = 0.0
    agg = df.groupby("role").agg(calls=("ms", "count"), total_ms=("ms", "sum"), avg_ms=("ms", "mean"),
                                 tokens=("tokens", "sum"), input_tokens=("input_tokens", "sum"),
                                 cached_tokens=("cached_tokens", "sum"), cost_usd=("cost_usd", "sum"),
                                 failures=("ok", lambda s: int((~s).sum())))
    agg["cache_hit_rate"] = (agg["cached_tokens"] / agg["input_tokens"].where(agg["input_tokens"] > 0)).fillna(0).round(3)
    return agg.reset_index().sort_values("total_ms", ascending=False).reset_index(drop=True)
