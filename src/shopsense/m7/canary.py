"""Canary tickets with known answers, run on a schedule and checked against SLOs, with alerts.

The canaries are golden-set tickets scored by the same deterministic checks as the M8 evaluation (route,
expected facts, forbidden phrases, guardrails, tool use). Each run also measures latency, cost and errors.

    python -m shopsense.m7.canary                 # one run
    python -m shopsense.m7.canary --every 60      # every 60 minutes
    cron:  0 * * * *  cd <repo> && .venv/bin/python -m shopsense.m7.canary

A breached SLO writes runtime/alerts/alert_*.json, posts {"text": ...} to SHOPSENSE_ALERT_WEBHOOK when set
(Slack-compatible), and exits with code 2 so a scheduler sees the failure. Scores go to Langfuse as a
`canary` trace. Tickets run in a throwaway runtime folder, so canaries never touch real ledgers.
"""
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime

CANARY_IDS = [i.strip() for i in os.getenv("SHOPSENSE_CANARY_IDS", "SS-F02,SS-G04,SS-I01").split(",") if i.strip()]
SLOS = {   # name: (limit, "min" or "max")
    "canary_pass_rate": (float(os.getenv("SHOPSENSE_SLO_PASS_RATE", "1.0")), "min"),
    "p95_latency_s": (float(os.getenv("SHOPSENSE_SLO_P95_LATENCY_S", "120")), "max"),
    "cost_per_ticket_usd": (float(os.getenv("SHOPSENSE_SLO_COST_PER_TICKET_USD", "0.01")), "max"),
    "error_rate": (float(os.getenv("SHOPSENSE_SLO_ERROR_RATE", "0")), "max"),
}
DETERMINISTIC = ("route_match", "must_not_contain_ok", "guardrail_passed", "tool_call_match", "facts_in_reply",
                 "citation_recall", "retrieval_keyword_hit_rate")


def breaches(measured: dict) -> list:
    out = []
    for name, (limit, kind) in SLOS.items():
        v = measured.get(name)
        if v is not None and ((kind == "min" and v < limit) or (kind == "max" and v > limit)):
            out.append(f"{name} = {v:g} (SLO {'>=' if kind == 'min' else '<='} {limit:g})")
    return out


def alert(summary: str, report: dict, alerts_dir) -> str:
    alerts_dir.mkdir(parents=True, exist_ok=True)
    path = alerts_dir / f"alert_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps({"summary": summary, "report": report}, indent=2, default=str), encoding="utf-8")
    hook = os.getenv("SHOPSENSE_ALERT_WEBHOOK")
    if hook:
        try:
            import httpx
            httpx.post(hook, json={"text": f"ShopSense canary: {summary}"}, timeout=15)
        except Exception as e:
            print(f"  alert webhook failed: {type(e).__name__}", file=sys.stderr)
    return str(path)


def run_canaries() -> dict:
    """Run the canary tickets (inside the throwaway runtime) and return the measurements."""
    from shopsense import tracing
    from shopsense.m8 import eval as m8
    golden = {i["id"]: i for i in m8.load_golden(m8.V2_PATH)}
    rows = []
    for cid in CANARY_IDS:
        item = golden[cid]
        before = len(tracing.RUN_EVENTS)
        t0 = time.perf_counter()
        trace = m8.run_item(item)
        seconds = time.perf_counter() - t0
        events = tracing.RUN_EVENTS[before:]
        scores = {n: v for n, v, _ in m8.component_scorer(item, trace, use_judges=False) if n in DETERMINISTIC}
        rows.append({"id": cid, "status": trace["status"], "route": trace["route"], "seconds": round(seconds, 1),
                     "cost_usd": round(sum(e.get("cost_usd", 0.0) for e in events), 6),
                     "tokens": sum(e.get("tokens", 0) for e in events),
                     "passed": trace["status"] != "error" and all(v >= 1 for v in scores.values()),
                     "failed_checks": [n for n, v in scores.items() if v < 1]})
    secs = sorted(r["seconds"] for r in rows)
    measured = {"canary_pass_rate": round(sum(r["passed"] for r in rows) / len(rows), 3),
                "p95_latency_s": secs[min(len(secs) - 1, max(0, round(0.95 * len(secs)) - 1))],
                "cost_per_ticket_usd": round(statistics.mean(r["cost_usd"] for r in rows), 6),
                "error_rate": round(sum(r["status"] == "error" for r in rows) / len(rows), 3)}
    return {"run_at": datetime.now().isoformat(timespec="seconds"), "canaries": rows, "measured": measured,
            "slos": {k: {"limit": v[0], "kind": v[1]} for k, v in SLOS.items()}}


def _log_to_langfuse(report: dict) -> None:
    from shopsense import tracing
    if not tracing.enabled():
        return
    try:
        with tracing.langfuse().start_as_current_observation(as_type="span", name="canary") as s:
            s.update(output=report["measured"], level="WARNING" if report["breaches"] else "DEFAULT")
            for name, value in report["measured"].items():
                s.score_trace(name=f"canary_{name}" if not name.startswith("canary") else name, value=float(value),
                              data_type="NUMERIC")
        tracing.flush()
    except Exception as e:
        print(f"  Langfuse logging failed: {type(e).__name__}", file=sys.stderr)


def _phase() -> int:
    from shopsense.config import RUNTIME_DIR
    from pathlib import Path
    reports_root = Path(os.environ.get("SHOPSENSE_RUNTIME_DIR_REPORTS") or RUNTIME_DIR)
    report = run_canaries()
    report["breaches"] = breaches(report["measured"])
    out = reports_root / "eval_reports" / f"canary_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _log_to_langfuse(report)
    for r in report["canaries"]:
        print(f"  {r['id']}: {'PASS' if r['passed'] else 'FAIL ' + str(r['failed_checks'])}  {r['seconds']}s  "
              f"${r['cost_usd']:.5f}  route={r['route']}")
    print(f"canary: {report['measured']}  report: {out}")
    if report["breaches"]:
        path = alert("; ".join(report["breaches"]), report, reports_root / "alerts")
        print(f"ALERT: {'; '.join(report['breaches'])}  ({path})")
        return 2
    print("all SLOs met")
    return 0


def self_check():
    assert breaches({"canary_pass_rate": 1.0, "p95_latency_s": 30, "cost_per_ticket_usd": 0.001, "error_rate": 0}) == []
    found = breaches({"canary_pass_rate": 0.67, "p95_latency_s": 300, "cost_per_ticket_usd": 0.001, "error_rate": 0})
    assert len(found) == 2 and found[0].startswith("canary_pass_rate") and "p95_latency_s" in found[1]
    with tempfile.TemporaryDirectory() as d:
        from pathlib import Path
        os.environ.pop("SHOPSENSE_ALERT_WEBHOOK", None)
        p = Path(alert("test", {"x": 1}, Path(d)))
        assert p.exists() and json.loads(p.read_text())["summary"] == "test"
    print(f"M7 canary self-check: PASS - SLO breaches detected in both directions; alerts written. "
          f"Canaries: {CANARY_IDS}; SLOs: { {k: v[0] for k, v in SLOS.items()} }")


def main():
    if "--offline" in sys.argv:
        self_check()
        return 0
    if "--phase" in sys.argv:
        return _phase()
    every = float(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else None
    from shopsense.config import RUNTIME_DIR
    while True:
        with tempfile.TemporaryDirectory() as d:            # canary tickets never touch the real ledgers
            env = {**os.environ, "SHOPSENSE_RUNTIME_DIR": d, "SHOPSENSE_RUNTIME_DIR_REPORTS": str(RUNTIME_DIR)}
            code = subprocess.run([sys.executable, "-m", "shopsense.m7.canary", "--phase"], env=env).returncode
        if every is None:
            return code
        time.sleep(every * 60)


if __name__ == "__main__":
    raise SystemExit(main())
