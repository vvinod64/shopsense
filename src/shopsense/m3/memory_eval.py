"""Memory layer (M3) evaluation, one score per sub-part, on eval/memory_cases.json.

  summary      WorkingMemory keeps order references, amounts and the stated language across
               budget-triggered summaries (the only part that calls a model: the rolling summariser)
  preferences  a language the customer states is saved and returned by customer_context (the ledger)
  long_term_copy  the preference's copy in Supermemory, recalled directly (retrieval and indexing lag)
  contacts     the ledger's contact count and the third-contact rule
  lessons      reviewer lessons: duplicates merged, newest three, per issue type

Usage:
  python -m shopsense.m3.memory_eval             # live: the configured LLM summarises, real Supermemory
  python -m shopsense.m3.memory_eval --offline   # extractive stub and an in-process store, no network

Live, the preference cases' long-term copies go to throwaway Supermemory containers (shopsense-eval-<run>-
<customer>, customer refs suffixed with the run id); memory.recall is polled for up to POLL_TIMEOUT_S per
write to allow for indexing lag (the latency is reported), and every container is deleted afterwards. Offline, an in-process
store stands in for Supermemory. The SQLite ledger is a temporary file per case either way, so nothing is
written to runtime/ except the report (runtime/eval_reports/).
"""
import json
import re
import tempfile
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from shopsense.config import PROJECT_ROOT, RUNTIME_DIR
from shopsense.m3 import memory

CASES_PATH = PROJECT_ROOT / "eval" / "memory_cases.json"
LANGUAGES = ("Hindi", "English", "Tamil", "Telugu", "Kannada", "Bengali", "Marathi", "Malayalam", "Gujarati", "Punjabi")
_FACTY = re.compile(r"KW-[A-Z]+-\d+|₹|\d|\b(?:" + "|".join(LANGUAGES) + r")\b", re.IGNORECASE)


def load_cases(path: Optional[Path] = None) -> dict:
    return json.loads((path or CASES_PATH).read_text(encoding="utf-8"))


# (a) Rolling summary

def extractive_summarizer(turns: list, prev_summary: str = "") -> str:
    """Offline stand-in for summarize_turns: the previous summary plus every sentence that carries an
    order ref, an amount or number, or a language name. Deterministic and model-free."""
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(t["content"] for t in turns))
    kept = [s.strip() for s in sentences if _FACTY.search(s)]
    return " ".join(([prev_summary] if prev_summary else []) + kept)


def failing_summarizer(turns: list, prev_summary: str = "") -> str:
    raise RuntimeError("summariser unavailable")


def _numbers(text: str) -> set:
    return {float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def fact_kept(fact: dict, text: str) -> bool:
    if fact["kind"] == "order_ref":
        return fact["value"].upper() in text.upper()
    if fact["kind"] == "amount":
        return any(abs(n - float(fact["value"])) < 0.005 for n in _numbers(text))
    if fact["kind"] == "language":
        return re.search(rf"\b{re.escape(fact['value'])}\b", text, re.IGNORECASE) is not None
    raise ValueError(f"unknown fact kind {fact['kind']!r}")


def score_summary_case(case: dict, summarizer: Callable) -> dict:
    wm = memory.WorkingMemory(recent_keep=case["recent_keep"], token_budget=case["token_budget"],
                              summarizer=failing_summarizer if case.get("summarizer") == "failing" else summarizer)
    for role, content in case["turns"]:
        wm.add(role, content)
    context = "\n".join(m["content"] for m in wm.messages())
    checks = {f"keeps {f['kind']} {f['value']}": fact_kept(f, context) for f in case["must_keep"]}
    if "expect_compressions_min" in case:
        checks[f"summarised >= {case['expect_compressions_min']}x"] = wm.compressions >= case["expect_compressions_min"]
        tail = [{"role": r, "content": c} for r, c in case["turns"][-case["recent_keep"]:]]
        checks["recent turns verbatim"] = wm.turns[-len(tail):] == tail
    if "expect_compressions_max" in case:
        checks[f"summarised <= {case['expect_compressions_max']}x"] = wm.compressions <= case["expect_compressions_max"]
        checks["every turn kept"] = len(wm.turns) == len(case["turns"])
    if "expect_failures_min" in case:
        checks["summariser failure recorded"] = wm.summary_failures >= case["expect_failures_min"]
    facts = case["must_keep"]
    return {"id": case["id"], "passed": all(checks.values()), "checks": checks,
            "facts_kept": sum(fact_kept(f, context) for f in facts), "facts": len(facts),
            "compressions": wm.compressions, "summary_failures": wm.summary_failures, "summary": wm.summary}


# Isolation: fresh ledger, in-process long-term store

@contextmanager
def fresh_ledger():
    saved = memory.DB_PATH
    with tempfile.TemporaryDirectory(prefix="shopsense-memory-eval-") as d:
        memory.DB_PATH = Path(d) / "memory.db"
        try:
            yield memory.DB_PATH
        finally:
            memory.DB_PATH = saved


class LocalStore:
    """Stand-in for Supermemory's remember/recall: one record per (container, custom_id), the same
    custom_id overwrites, recall returns that container's records of the requested kind."""

    def __init__(self):
        self.records = {}

    def remember(self, text, kind, container_tag, custom_id=None, **extra):
        key = (container_tag, custom_id or f"auto-{len(self.records)}")
        self.records[key] = {"kind": kind, "text": text}
        return key[1]

    def recall(self, query, container_tag, k=3, kind=None):
        hits = [{"kind": r["kind"], "text": r["text"], "score": 1.0}
                for (tag, _), r in self.records.items() if tag == container_tag and (kind is None or r["kind"] == kind)]
        return hits[:k]


@contextmanager
def local_long_term_memory():
    store, saved = LocalStore(), (memory.remember, memory.recall)
    memory.remember, memory.recall = store.remember, store.recall
    try:
        yield store
    finally:
        memory.remember, memory.recall = saved


class LiveSupermemory:
    """Real Supermemory through memory.py's own functions, with customer_tag redirected to throwaway
    containers for this run; cleanup() deletes everything written to them."""

    def __init__(self, run_id: str):
        self.run_id, self.tags = run_id, set()

    def customer_ref(self, ref: str) -> str:
        return f"{ref}-R{self.run_id}"

    def tag(self, customer_ref: str) -> str:
        t = f"shopsense-eval-{self.run_id}-{customer_ref.strip().lower()}"
        self.tags.add(t)
        return t

    def cleanup(self) -> dict:
        if not self.tags:
            return {"containers": 0}
        try:
            resp = memory._mem().documents.delete_bulk(container_tags=sorted(self.tags))
            return {"containers": len(self.tags), "response": getattr(resp, "model_dump", lambda: str(resp))()}
        except Exception as e:
            return {"containers": len(self.tags), "error": f"{type(e).__name__}: {e}"}


@contextmanager
def live_long_term_memory(run_id: str):
    live, saved = LiveSupermemory(run_id), memory.customer_tag
    memory.customer_tag = live.tag
    try:
        yield live
    finally:
        memory.customer_tag = saved
        live.cleanup_result = live.cleanup()


# (b) Preferences (ledger-backed) and (b') the long-term copy in Supermemory

POLL_TIMEOUT_S, POLL_EVERY_S = 60, 3
# the query get_customer_preferences falls back to; the copy is recalled with it directly
PREFERENCE_QUERY = "customer's preferred language, contact channel and tone"


def languages_in(preferences) -> list:
    """Languages named in returned preferences: a JSON object from the ledger ({"preferred_language": ...}),
    a JSON list of recalled memories, or a 'No preferences ...' string."""
    if not isinstance(preferences, str) or preferences.startswith(("No preferences", "unavailable")):
        return []
    data = json.loads(preferences)
    text = " ".join(map(str, data.values())) if isinstance(data, dict) else " ".join(m.get("text", "") for m in data)
    return [lang for lang in LANGUAGES if re.search(rf"\b{lang}\b", text, re.IGNORECASE)]


def copy_languages(customer_ref: str) -> list:
    """Languages in the customer's long-term copy, recalled from Supermemory (or the stand-in) directly."""
    try:
        hits = memory.recall(PREFERENCE_QUERY, memory.customer_tag(customer_ref.strip().upper()), k=3, kind="semantic")
    except Exception:
        return []
    return languages_in(json.dumps(hits))


def poll_copy(customer_ref: str, done: Callable[[list], bool], timeout_s: float) -> tuple:
    """Recall the copy until done(languages) or the timeout: (languages, seconds waited, done?)."""
    start = time.monotonic()
    while True:
        langs = copy_languages(customer_ref)
        waited = round(time.monotonic() - start, 1)
        if done(langs) or waited >= timeout_s:
            return langs, waited, done(langs)
        time.sleep(POLL_EVERY_S)


def _final_checks(case: dict, returned: list) -> dict:
    want = case["expect_language"]
    checks = {f"returns {want}": (want in returned) if want else returned == []}
    for lang in case.get("expect_not", []):
        checks[f"does not return {lang}"] = lang not in returned
    return checks


def score_preference_case(case: dict, live: Optional[LiveSupermemory] = None) -> tuple:
    """Runs the case's messages through customer_context and scores two things separately:
    the preferences it returns (ledger-backed, the source of truth), and the long-term copy, recalled
    from Supermemory directly. After each stated language the copy is polled until it returns that
    language (and none of expect_not, after the last message): immediate offline, up to POLL_TIMEOUT_S
    live. Returns (ledger_row, copy_row)."""
    ref = live.customer_ref(case["customer_ref"]) if live else case["customer_ref"]
    timeout = POLL_TIMEOUT_S if live else 0
    checks, copy_checks, latencies, returned, copy, wrote = {}, {}, [], [], [], False
    with fresh_ledger(), (nullcontext() if live else local_long_term_memory()):
        if case.get("other_customer"):
            other = case["other_customer"]
            other_ref = live.customer_ref(other["customer_ref"]) if live else other["customer_ref"]
            stated = (memory.customer_context(other_ref, other["text"]).get("stated_now") or {}).get("preferred_language")
            _, waited, ok = poll_copy(other_ref, lambda langs: stated in langs, timeout)
            latencies.append(waited)
            copy_checks["other customer's copy stored and recallable"] = ok
        for i, msg in enumerate(case["messages"], start=1):
            out = memory.customer_context(ref, msg["text"])
            stated = (out.get("stated_now") or {}).get("preferred_language")
            checks[f"message {i} stated = {msg['expect_stated']}"] = stated == msg["expect_stated"]
            returned = languages_in(out.get("preferences"))
            if stated:
                wrote = True
                avoid = case.get("expect_not", []) if i == len(case["messages"]) else []
                copy, waited, _ = poll_copy(ref, lambda langs: stated in langs and not any(a in langs for a in avoid), timeout)
                latencies.append(waited)
        if not wrote:
            copy = copy_languages(ref)
    checks.update(_final_checks(case, returned))
    copy_checks.update(_final_checks(case, copy))
    return ({"id": case["id"], "passed": all(checks.values()), "checks": checks, "returned": returned},
            {"id": case["id"], "passed": all(copy_checks.values()), "checks": copy_checks, "returned": copy,
             "recall_latency_s": latencies})


# (c) Contact ledger and the third-contact rule

def third_contact(prior_contacts: int) -> bool:
    """The workflow's own rule (m5.workflow.assess_risk) applied to the ledger count."""
    from shopsense.m5.workflow import assess_risk
    risk = assess_risk({"issue_type": "shipping_delay", "sentiment": "neutral"}, {}, None, prior_contacts, "")
    return any("third contact" in r for r in risk)


def score_contact_case(case: dict) -> dict:
    with fresh_ledger():
        for ticket_id, customer_ref, order_ref, created_at in case["tickets"]:
            memory.log_ticket(ticket_id, customer_ref, "eval", {"issue_type": "shipping_delay", "order_id": order_ref},
                              "r", created_at=created_at, write_episode=False)
        q = case["query"]
        n = memory.contact_count(q["customer_ref"], q["order_ref"], now=datetime.fromisoformat(case["now"]),
                                 exclude_ticket=q.get("exclude_ticket"))
    checks = {f"count = {case['expect_count']}": n == case["expect_count"],
              f"third contact = {case['expect_third_contact']}": third_contact(n) == case["expect_third_contact"]}
    return {"id": case["id"], "passed": all(checks.values()), "checks": checks, "count": n}


# (d) Reviewer lessons

def score_lesson_case(case: dict) -> dict:
    checks = {}
    with fresh_ledger():
        for i, r in enumerate(case["records"], start=1):
            ok = memory.record_lesson(r.get("issue_type", case["issue_type"]), r["note"], f"EVAL-{case['id']}-{i}", "rejected")
            checks[f"record {i} stored = {r['expect_recorded']}"] = ok == r["expect_recorded"]
            time.sleep(0.002)                     # distinct created_at, so "newest" is well defined
        got = memory.lessons_for(case["issue_type"])
    checks["lessons_for (newest three, merged)"] = got == case["expect"]
    return {"id": case["id"], "passed": all(checks.values()), "checks": checks, "lessons": got}


# Runner

def _part(rows: list) -> dict:
    return {"n": len(rows), "passed": sum(r["passed"] for r in rows),
            "accuracy": round(sum(r["passed"] for r in rows) / len(rows), 4) if rows else None, "per_case": rows}


def evaluate_memory(offline: bool = True, path: Optional[Path] = None, write_report: bool = True) -> dict:
    spec = load_cases(path)
    summarizer = extractive_summarizer if offline else memory.summarize_turns
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    live_ctx = nullcontext() if offline else live_long_term_memory(run_id)
    with live_ctx as live:
        rows = [score_preference_case(c, live) for c in spec["preferences"]]
    preferences, copy = _part([r[0] for r in rows]), _part([r[1] for r in rows])
    lat = [t for r in copy["per_case"] for t in r["recall_latency_s"]]
    copy["recall_latency_s"] = ({"writes": len(lat), "max": max(lat), "mean": round(sum(lat) / len(lat), 1)}
                                if lat else None)
    copy["store"] = ({"backend": "supermemory", "containers": f"shopsense-eval-{run_id}-*", "cleanup": live.cleanup_result}
                     if live else {"backend": "in-process stand-in"})
    parts = {
        "summary": _part([score_summary_case(c, summarizer) for c in spec["summary"]]),
        "preferences": preferences,
        "long_term_copy": copy,
        "contacts": _part([score_contact_case(c) for c in spec["contacts"]]),
        "lessons": _part([score_lesson_case(c) for c in spec["lessons"]]),
    }
    s = parts["summary"]["per_case"]
    parts["summary"]["fact_retention"] = round(sum(r["facts_kept"] for r in s) / max(1, sum(r["facts"] for r in s)), 4)
    report = {"run_at": datetime.now().isoformat(timespec="seconds"),
              "summarizer": "extractive stub (offline)" if offline else f"summarize_turns ({memory.LLM_MODEL})",
              "long_term_store": "in-process stand-in (offline)" if offline else "Supermemory (throwaway containers)",
              "cases_file": str((path or CASES_PATH).relative_to(PROJECT_ROOT)), "parts": parts}
    if write_report:
        out = RUNTIME_DIR / "eval_reports" / f"memory_eval_{'offline_' if offline else ''}{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        report["report_path"] = str(out.relative_to(PROJECT_ROOT)) if out.is_relative_to(PROJECT_ROOT) else str(out)
    return report


def print_report(report: dict) -> None:
    print(f"\nM3 memory eval (summariser: {report['summarizer']}; long-term store: {report['long_term_store']})")
    print(f"{'sub-part':<16}{'passed':>8}{'accuracy':>10}")
    for name, p in report["parts"].items():
        print(f"{name:<16}{p['passed']:>5}/{p['n']:<2}{p['accuracy']:>10.3f}")
    print(f"summary fact retention: {report['parts']['summary']['fact_retention']:.3f}")
    prefs = report["parts"]["long_term_copy"]
    if prefs.get("recall_latency_s"):
        print(f"long-term copy recall latency: {prefs['recall_latency_s']}")
    if prefs["store"].get("cleanup"):
        print(f"cleanup: {prefs['store']['cleanup']}")
    for name, p in report["parts"].items():
        for r in p["per_case"]:
            if not r["passed"]:
                print(f"  FAIL {name} {r['id']}: " + ", ".join(k for k, ok in r["checks"].items() if not ok))
    if report.get("report_path"):
        print(f"Report: {report['report_path']}")


if __name__ == "__main__":
    import sys
    print_report(evaluate_memory(offline="--offline" in sys.argv))
