"""Observability and reliability checks.

Offline: the traced() contract, the fault experiment against the order API, and tracing overhead.
Live: tickets through the workflow with a normal, flaky and unavailable order system plus one team
ticket, a per-agent latency and token table, and a read-back of the spans from Langfuse.
"""
import json
import os
import statistics
import sys
import time

from shopsense import mock_api, tracing
from shopsense.config import LLM_MODEL
from shopsense.reliability import (BASE_CONFIG, FaultConfig, TOOL_CALL_LOG, make_robust, reset_breakers,
                                   run_fault_experiment)

REAL_ORDERS = ["KW-O-000006", "KW-O-000030", "KW-O-000031", "KW-O-000046"]
ORDER_KEYS = {"order_ref", "order_value_inr", "customer_ref"}


def _valid_order(v) -> bool:
    return isinstance(v, dict) and ORDER_KEYS <= set(v)


def traced_contract_check():
    """A traced node records timing; a failing node records ok=False and re-raises."""
    tracing.RUN_EVENTS.clear()

    @tracing.traced("dummy")
    def ok_node(state):
        return {"log": ["ran"]}

    @tracing.traced("boom")
    def bad_node(state):
        raise ValueError("simulated node failure")

    with tracing.span("m7-selfcheck"):
        ok_node({})
        try:
            bad_node({})
            raise AssertionError("the exception must propagate")
        except ValueError:
            pass
    assert [(e["role"], e["ok"]) for e in tracing.RUN_EVENTS] == [("dummy", True), ("boom", False)]
    tracing.RUN_EVENTS.clear()


def fault_experiment_on_order_api() -> dict:
    """Fault experiment against the mock order API."""
    reset_breakers()
    return run_fault_experiment(mock_api.get_order, REAL_ORDERS, _valid_order, n=200, config=BASE_CONFIG)


def overhead_benchmark(n: int = 200) -> dict:
    """Robust order lookups with tracing off and on."""
    lookup = make_robust(lambda r: mock_api.get_order(r), "bench.get_order", validate=_valid_order)

    def run():
        times = []
        for i in range(n):
            t0 = time.perf_counter()
            lookup(REAL_ORDERS[i % len(REAL_ORDERS)])
            times.append((time.perf_counter() - t0) * 1000)
        return statistics.median(times)

    prev = os.environ.get("SHOPSENSE_TRACING")
    os.environ["SHOPSENSE_TRACING"] = "0"
    untraced = run()
    os.environ["SHOPSENSE_TRACING"] = prev or "1"
    with tracing.span("overhead-benchmark"):
        traced_ms = run()
    tracing.flush()
    return {"untraced_median_ms": round(untraced, 4), "traced_median_ms": round(traced_ms, 4),
            "overhead_ms_per_call": round(traced_ms - untraced, 4)}


def self_check():
    traced_contract_check()
    r = fault_experiment_on_order_api()
    assert r["baseline"]["success_rate"] == 0.57 and r["retries"]["success_rate"] == 0.895, r
    assert r["fallback"] == {"ok": 200, "fallback_used": 21, "success_rate": 1.0}, r
    assert r["breaker"]["calls_that_hit_the_dependency"] == 3, r
    cost = tracing.estimate_cost(LLM_MODEL, {"input": 1_000_000, "output": 1_000_000, "total": 2_000_000})
    assert cost and cost["total"] > 0, f"{LLM_MODEL} has no list price, so traces would show no cost"
    assert tracing.estimate_cost("no-such-provider/model", {"input": 1, "output": 1}) is None
    print("M7 self-check: PASS")
    print("  A1 traced(): success and failure both recorded; failures re-raised")
    print(f"  B1 baseline : {r['baseline']}")
    print(f"  B2 retries  : {r['retries']}")
    print(f"  B3 fallback : {r['fallback']}   (fallback = human queue)")
    print(f"  B4 breaker  : {r['breaker']}")
    print(f"  A5 cost     : {LLM_MODEL} at list price, USD per 1M tokens in/out: "
          f"{cost['input']:.2f} / {cost['output']:.2f}")
    o = overhead_benchmark()
    print(f"  A4 overhead : {o}  -> report the UNTRACED number as real latency")
    return r, o


# Live

TICKETS = {
    "normal": ("KW-T-M7-N", "KW-C-00298", "My furniture order KW-O-000006 came a whole week late. I want compensation."),
    "flaky":  ("KW-T-M7-F", "KW-C-00229", "Order KW-O-000031 (a toy) was delivered 4 days late. Anything you can do?"),
    "down":   ("KW-T-M7-D", "KW-C-00142", "My apparel order KW-O-000030 arrived 4 days late, I'd like compensation."),
    "team":   ("KW-T-M7-T", "KW-C-00003", "My home & kitchen order KW-O-000046 arrived a week late. Please compensate me."),
}


def langfuse_readback(trace_ids: list, since_s: float, attempts: int = 8) -> dict:
    """Read spans back through the Langfuse v2 observations API, polling while ingestion catches up."""
    import httpx
    from datetime import datetime, timedelta, timezone
    base = os.environ["LANGFUSE_BASE_URL"].rstrip("/")
    auth = (os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"])
    frm = (datetime.now(timezone.utc) - timedelta(seconds=since_s + 120)).isoformat()
    obs = []
    for _ in range(attempts):
        to = datetime.now(timezone.utc).isoformat()
        rows, cursor = [], None
        while True:   # request the field groups needed; results are paged by cursor
            params = {"fromStartTime": frm, "toStartTime": to, "limit": 100, "fields": "core,basic,usage,metadata"}
            if cursor:
                params["cursor"] = cursor
            r = httpx.get(f"{base}/api/public/v2/observations", auth=auth, timeout=30, params=params)
            r.raise_for_status()
            body = r.json()
            data = body.get("data", [])
            rows += data
            cursor = (body.get("meta") or {}).get("cursor")
            if not data or not cursor:
                break
        obs = [o for o in rows if o.get("traceId") in set(trace_ids)]
        if len({o["traceId"] for o in obs}) == len(set(trace_ids)) and any(o.get("type") == "GENERATION" for o in obs):
            break
        time.sleep(10)
    by_type = {}
    for o in obs:
        by_type.setdefault(o.get("type"), set()).add(o.get("name"))
    gens = [o for o in obs if o.get("type") == "GENERATION"]
    tokens = sum(int((o.get("usageDetails") or {}).get("total") or o.get("totalUsage") or 0) for o in gens)
    cost = sum(float((o.get("costDetails") or {}).get("total") or o.get("totalCost") or 0) for o in gens)
    nested = sum(1 for o in obs if o.get("parentObservationId"))
    fallbacks = [f"{o['name']} ({(o.get('metadata') or {}).get('reason')})" for o in obs
                 if isinstance(o.get("metadata"), dict) and o["metadata"].get("status") == "fallback"]
    sessions = sorted({o.get("sessionId") for o in obs if o.get("sessionId")})
    return {"observations": len(obs), "traces_found": len({o["traceId"] for o in obs}),
            "by_type": {k: sorted(v)[:12] for k, v in by_type.items()}, "generations": len(gens),
            "tokens_recorded": tokens, "cost_recorded_usd": round(cost, 6), "nested_observations": nested, "fallback_spans": fallbacks,
            "session_ids": sessions}


def _live():
    import asyncio
    from shopsense.m5.workflow import start_ticket
    from shopsense.m6.team import run_ticket
    t_start = time.time()
    tracing.RUN_EVENTS.clear()
    results, trace_ids = {}, []

    tid, cref, text = TICKETS["normal"]
    results["normal"] = start_ticket(text, tid, cref)

    mock_api.install_faults(BASE_CONFIG)                       # flaky: 20% fail, 10% timeout, 10% malformed
    TOOL_CALL_LOG.clear()
    tid, cref, text = TICKETS["flaky"]
    results["flaky"] = start_ticket(text, tid, cref)
    results["flaky"]["tool_log"] = [{k: e.get(k) for k in ("tool", "status", "attempts", "reason")}
                                    for e in TOOL_CALL_LOG if e["tool"].startswith("order_api")]
    mock_api.remove_faults()

    reset_breakers()
    mock_api.install_faults(FaultConfig(seed=7, fail_rate=1.0))  # order system down
    TOOL_CALL_LOG.clear()
    tid, cref, text = TICKETS["down"]
    results["down"] = start_ticket(text, tid, cref)
    results["down"]["tool_log"] = [{k: e.get(k) for k in ("tool", "status", "attempts", "reason")}
                                   for e in TOOL_CALL_LOG if e["tool"].startswith("order_api")]
    mock_api.remove_faults()
    reset_breakers()

    tid, cref, text = TICKETS["team"]
    team = asyncio.run(run_ticket(text, tid, cref))
    results["team"] = {"status": team["status"], "trace_id": team.get("trace_id"), "log": team["log"]}

    for k in ("normal", "flaky", "down"):
        r = results[k]
        print(f"\n[{k}] status={r['status']} risk={r['risk']} proposal={r['proposal']}")
        if r.get("tool_log"):
            print(f"   order-system calls: {r['tool_log']}")
        trace_ids.append(r["trace_id"])
    print(f"\n[team] status={results['team']['status']}  ({len(results['team']['log'])} trajectory lines)")
    trace_ids.append(results["team"]["trace_id"])

    dash = tracing.dashboard()
    print("\nA3 per-agent dashboard (this run, from RUN_EVENTS; cost in USD at list price):")
    print(dash.to_string(index=False))
    print(f"   estimated cost of this run: ${dash['cost_usd'].sum():.5f} for {int(dash['tokens'].sum())} tokens")

    tracing.flush()
    print("\nReading the spans back from Langfuse ...")
    rb = langfuse_readback(trace_ids, time.time() - t_start)
    print(json.dumps(rb, indent=2))

    # Behaviour under faults and trace contents
    assert results["normal"]["status"] == "auto_resolved"
    flaky_log = results["flaky"]["tool_log"]
    assert results["flaky"]["status"] in ("auto_resolved", "awaiting_approval"), "flaky system must not crash a ticket"
    assert any((e["attempts"] or 0) > 1 for e in flaky_log) or any(e["status"] == "fallback" for e in flaky_log), \
        "with 40% injected faults some order call must have needed a retry or a fallback"
    assert results["down"]["status"] == "awaiting_approval", "a down order system must route to the human queue"
    assert any("human queue" in r for r in results["down"]["risk"])
    assert all(e["status"] == "fallback" for e in results["down"]["tool_log"])
    assert results["team"]["status"] in ("resolved", "escalated_to_human")
    roles = set(dash["role"])
    assert {"triage", "policy_check", "draft", "checks", "finalize"} <= roles and {"supervisor", "order_actions"} <= roles
    assert dash["tokens"].sum() > 0, "generation spans must have recorded token usage"
    assert dash["cost_usd"].sum() > 0, "generation spans must carry an estimated cost"
    assert rb["traces_found"] == len(trace_ids), f"every ticket trace must reach Langfuse: {rb}"
    assert rb["generations"] > 0 and rb["nested_observations"] > 0
    assert rb["fallback_spans"], "fallback tool spans must be visible in the trace"
    assert rb["tokens_recorded"] > 0, "token usage must be recorded on generation spans in Langfuse"
    assert set(rb["session_ids"]) >= {t[0] for t in TICKETS.values()}, "every ticket tagged with its session id"
    assert "approval" not in set(dash[dash["failures"] > 0]["role"]), "an interrupt is a pause, not a failure"
    flaky_outcome = ("absorbed by retries" if results["flaky"]["status"] == "auto_resolved"
                     else "retried, then degraded to the human queue")
    print("\nM7 live: PASS - every ticket traced end to end (root -> agents -> LLM generations with tokens and cost, "
          f"retriever and tool spans); flaky order system: {flaky_outcome}; a down order system routed the "
          "ticket to the human queue with fallback spans visible in Langfuse.")


def smoke_test():
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = subprocess.run([sys.executable, "-m", "shopsense.m7.observability", "--phase", "live"],
                           env={**os.environ, "SHOPSENSE_RUNTIME_DIR": d}, timeout=2400)
        if p.returncode:
            raise SystemExit(p.returncode)


if __name__ == "__main__":
    if "--phase" in sys.argv:
        _live()
    else:
        self_check()
        if "--offline" not in sys.argv:
            smoke_test()
