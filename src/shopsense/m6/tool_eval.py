"""MCP tool layer evaluation: labelled calls to every tool of the kartway-orders server, scored exactly.

Each case in eval/tool_cases.json calls one tool through the MCP protocol (FastMCP in-memory client, the
same tool schemas and error handling a stdio client gets) with inputs from the mock tables and checks the
result against hand-derived values. No model calls. Writes go to a temporary runtime directory, so the
request logs never touch runtime/; the report goes to runtime/eval_reports/.

Usage:
  python -m shopsense.m6.tool_eval          (or python -m shopsense.m6.mcp_server --eval)
"""
import asyncio
import json
import logging
import os
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from shopsense import mock_api
from shopsense.config import PROJECT_ROOT, RUNTIME_DIR

CASES_PATH = PROJECT_ROOT / "eval" / "tool_cases.json"


def load_cases(path: Optional[Path] = None) -> dict:
    return json.loads((path or CASES_PATH).read_text(encoding="utf-8"))


def ownership(order: dict, signed_in: str) -> str:
    """The callers' rule (m6/team.py order_actions, m5/workflow.py): a signed-in customer sees only their own order."""
    return "allowed" if (order.get("customer_ref") or "").upper() == signed_in.strip().upper() else "refused"


def _same(got, want) -> bool:
    if isinstance(want, (int, float)) and not isinstance(want, bool) and isinstance(got, (int, float)):
        return abs(float(got) - float(want)) < 0.005
    return got == want


def _log_count(spec: dict) -> int:
    path = mock_api.ACTIONS_FILE if spec["file"] == "actions" else mock_api.ESCALATIONS_FILE
    if not path.exists():
        return 0
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = {k: v for k, v in spec.items() if k not in ("file", "count")}
    return sum(1 for r in rows if all(r.get(k) == v for k, v in keys.items()))


def score_case(case: dict, result: Optional[dict], error: Optional[str]) -> dict:
    """Every labelled field must match; returns the failed checks (an empty list is a pass)."""
    failed = []
    if "expect_error" in case:
        if error is None or case["expect_error"].lower() not in error.lower():
            failed.append(f"expected an error containing {case['expect_error']!r}, got {error or result!r}")
        return {"id": case["id"], "tool": case["tool"], "passed": not failed, "failed": failed}
    if error is not None:
        return {"id": case["id"], "tool": case["tool"], "passed": False, "failed": [f"unexpected error: {error}"]}
    result = dict(result or {})
    if case.get("signed_in_customer"):
        result["ownership"] = ownership(result, case["signed_in_customer"])
    for field, want in case.get("expect", {}).items():
        if not _same(result.get(field), want):
            failed.append(f"{field}: expected {want!r}, got {result.get(field)!r}")
    for field, needle in case.get("expect_contains", {}).items():
        value = result.get(field)
        text = " | ".join(map(str, value)) if isinstance(value, list) else str(value)
        if needle.lower() not in text.lower():
            failed.append(f"{field}: expected to contain {needle!r}, got {value!r}")
    if case.get("expect_log"):
        n = _log_count(case["expect_log"])
        if n != case["expect_log"]["count"]:
            failed.append(f"{case['expect_log']['file']} log: expected {case['expect_log']['count']} record(s), got {n}")
    return {"id": case["id"], "tool": case["tool"], "passed": not failed, "failed": failed}


@contextmanager
def isolated_runtime(as_of: Optional[str]):
    """A temporary runtime dir for the action/escalation logs, a pinned date and automation switched on."""
    saved_files = (mock_api.ACTIONS_FILE, mock_api.ESCALATIONS_FILE)
    saved_env = {k: os.environ.get(k) for k in ("SHOPSENSE_RUNTIME_DIR", "SHOPSENSE_AS_OF", "SHOPSENSE_AUTOMATION")}
    with tempfile.TemporaryDirectory(prefix="shopsense-tool-eval-") as d:
        os.environ["SHOPSENSE_RUNTIME_DIR"] = d
        os.environ.pop("SHOPSENSE_AUTOMATION", None)
        if as_of:
            os.environ["SHOPSENSE_AS_OF"] = as_of
        mock_api.ACTIONS_FILE, mock_api.ESCALATIONS_FILE = Path(d) / "actions.jsonl", Path(d) / "escalations.jsonl"
        try:
            yield Path(d)
        finally:
            mock_api.ACTIONS_FILE, mock_api.ESCALATIONS_FILE = saved_files
            for k, v in saved_env.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


async def _run_cases(cases: list) -> list:
    from fastmcp import Client
    from shopsense.m6.mcp_server import mcp
    rows = []
    async with Client(mcp) as client:
        for case in cases:
            r = await client.call_tool(case["tool"], case["args"], raise_on_error=False)
            error = (r.content[0].text if r.content else "tool error") if r.is_error else None
            rows.append(score_case(case, None if error else r.structured_content, error))
    return rows


def summarize(rows: list) -> dict:
    by_tool = defaultdict(lambda: [0, 0])
    for r in rows:
        by_tool[r["tool"]][0] += r["passed"]
        by_tool[r["tool"]][1] += 1
    return {"n": len(rows), "passed": sum(r["passed"] for r in rows),
            "accuracy": round(sum(r["passed"] for r in rows) / len(rows), 4) if rows else None,
            "per_tool": {t: {"passed": p, "n": n, "accuracy": round(p / n, 4)} for t, (p, n) in sorted(by_tool.items())}}


def evaluate_tools(path: Optional[Path] = None, write_report: bool = True) -> dict:
    spec = load_cases(path)
    logging.getLogger("fastmcp").setLevel(logging.CRITICAL)      # expected tool errors are scored, not printed
    with isolated_runtime(spec.get("as_of")):
        rows = asyncio.run(_run_cases(spec["cases"]))
    report = {"run_at": datetime.now().isoformat(timespec="seconds"), "as_of": spec.get("as_of"),
              "cases_file": str((path or CASES_PATH).relative_to(PROJECT_ROOT)), **summarize(rows), "per_case": rows}
    if write_report:
        out = RUNTIME_DIR / "eval_reports" / f"tool_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        report["report_path"] = str(out.relative_to(PROJECT_ROOT)) if out.is_relative_to(PROJECT_ROOT) else str(out)
    return report


def print_report(report: dict) -> None:
    print(f"\nMCP tool eval ({report['n']} cases, as_of {report['as_of']}, no model calls)")
    print(f"{'tool':<24}{'passed':>8}{'accuracy':>10}")
    for tool, s in report["per_tool"].items():
        print(f"{tool:<24}{s['passed']:>5}/{s['n']:<2}{s['accuracy']:>10.3f}")
    print(f"{'overall':<24}{report['passed']:>5}/{report['n']:<2}{report['accuracy']:>10.3f}")
    for r in report["per_case"]:
        if not r["passed"]:
            print(f"  FAIL {r['id']} {r['tool']}: " + "; ".join(r["failed"]))
    if report.get("report_path"):
        print(f"Report: {report['report_path']}")


if __name__ == "__main__":
    print_report(evaluate_tools())
