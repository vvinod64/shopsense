"""Check a golden set against the policy corpus: an expected answer is only as good as the policy behind it.

For every item, each figure the expected answer states (days, hours, months, years, percentages, amounts) and
each expected-fact pattern must appear in the policy documents the item cites (must_cite titles and the
documents named in its evidence). Figures the customer's own message contains are skipped. No model calls.

Usage:
  python -m shopsense.m8.golden_audit                                   # eval/golden_set_v2.json
  python -m shopsense.m8.golden_audit data/shopsense/eval/golden_set.json   # the toolkit's set (read only)
"""
import json
import re
import sys
from pathlib import Path
from typing import Optional

from shopsense.config import CORPUS_DIR, PROJECT_ROOT

V2_PATH = PROJECT_ROOT / "eval" / "golden_set_v2.json"
UNITS = r"(%|percent|(?:calendar |business |consecutive )?(?:days?|hours?|months?|years?))"
FIGURE = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)[\s-]*" + UNITS + r"(?![a-z])", re.IGNORECASE)
MONEY = re.compile(r"([$₹])\s?(\d[\d,]*(?:\.\d+)?)")
WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
         "nine": "9", "ten": "10", "twelve": "12", "fourteen": "14", "fifteen": "15", "thirty": "30",
         "forty-two": "42", "forty two": "42", "forty-five": "45", "forty five": "45", "sixty": "60", "ninety": "90"}


def _corpus() -> dict:
    """title -> text and file name -> text, for the Markdown corpus."""
    out = {}
    for d in json.loads((CORPUS_DIR / "index.json").read_text(encoding="utf-8")):
        text = (CORPUS_DIR / "markdown" / f"{d['slug']}.md").read_text(encoding="utf-8")
        out[d["title"]] = out[f"{d['slug']}.md"] = text
    return out


def _norm(text: str) -> str:
    """Markdown emphasis off, hyphens and non-breaking spaces as spaces, spelled-out numbers as digits."""
    text = text.replace("*", "").replace(" ", " ").lower()
    for w, n in WORDS.items():
        text = re.sub(rf"\b{w}\b", n, text)
    return re.sub(r"(?<=\d)-|-(?=\d)|(?<=[a-z])-(?=[a-z])", " ", text)


def _unit_stem(unit: str) -> str:
    unit = unit.lower().replace("calendar ", "").replace("business ", "").replace("consecutive ", "")
    return "%" if unit in ("%", "percent") else unit.rstrip("s")


def figure_in(number: str, unit: str, text: str) -> bool:
    """'15 calendar days', '**15 calendar days**', '15-day', 'fifteen (15) days' all count as 15 days."""
    num, stem = re.escape(number.replace(",", "")), _unit_stem(unit)
    flat = _norm(text).replace(",", "")
    if stem == "%":
        return bool(re.search(rf"(?<![\d.]){num}\s*(%|percent)", flat))
    return bool(re.search(rf"(?<![\d.]){num}\)?\s+(?:\(\d+\)\s+)?(?:calendar |business |consecutive )?{stem}", flat))


def cited_documents(item: dict, corpus: dict) -> tuple:
    names = list(item.get("must_cite") or []) + re.findall(r"[\w-]+\.md", item.get("evidence") or "")
    found = [n for n in dict.fromkeys(names) if n in corpus]
    return found, "\n".join(corpus[n] for n in found)


def audit_item(item: dict, corpus: dict) -> list:
    """Problems with one item. The expected answer goes through the same figure check as a ShopSense reply
    (guardrails.unsupported_claims): each figure must come from a policy sentence on the same topic, and a
    named category's addendum overrides the general figure. The evidence is the cited documents plus the
    general returns policy. Dollar amounts must appear in the evidence, and expected-fact patterns too."""
    from shopsense.guardrails import unsupported_claims
    from shopsense.m3.rag import load_chunks
    question = item.get("ticket") or item.get("question") or ""
    answer = item.get("reference_answer") or item.get("expected") or ""
    docs, text = cited_documents(item, corpus)
    if not docs:
        return []                                  # nothing cited: not a policy-fact item
    files = {n if n.endswith(".md") else _slug(n, corpus) for n in docs} | {"returns-policy.md"}
    findings = [{"source": c["source"], "text": c["text"]} for c in load_chunks() if c["source"] in files]
    problems = [p for p in unsupported_claims(answer, findings, {}, {}, ticket_text=question)
                if p.startswith(("policy figure", "policy range"))]     # amounts: checked below
    asked = _norm(question).replace(",", "")
    for m in MONEY.finditer(answer):
        amount = m.group(2).replace(",", "")
        if amount not in asked and amount not in _norm(text).replace(",", ""):
            problems.append(f"amount '{m.group(0)}' is not in {', '.join(docs)}")
    for pattern in item.get("expected_facts") or []:
        if not re.search(pattern, text, re.IGNORECASE) and not re.search(pattern, _norm(text), re.IGNORECASE):
            problems.append(f"expected fact /{pattern}/ is not in {', '.join(docs)}")
    return problems


def _slug(title: str, corpus: dict) -> str:
    return next(name for name, body in corpus.items() if name.endswith(".md") and body == corpus[title])


def audit(path: Optional[Path] = None) -> dict:
    raw = json.loads(Path(path or V2_PATH).read_text(encoding="utf-8"))
    items = raw if isinstance(raw, list) else raw["items"]
    corpus = _corpus()
    report = {i["id"]: audit_item(i, corpus) for i in items}
    return {"golden_set": str(path or V2_PATH), "items": len(items),
            "items_with_problems": {k: v for k, v in report.items() if v}}


def self_check() -> None:
    corpus = _corpus()
    ok = {"id": "x", "question": "Electronics window?", "expected": "Electronics: 15 calendar days.",
          "must_cite": ["Category Policy Addendum: Electronics"]}
    wrong = {**ok, "expected": "Electronics have a 14-day window."}
    assert audit_item(ok, corpus) == [] and audit_item(wrong, corpus), audit_item(wrong, corpus)
    print("golden audit self-check: PASS - a figure is checked against the documents the item cites")


if __name__ == "__main__":
    self_check()
    target = next((a for a in sys.argv[1:] if not a.startswith("-")), None)
    result = audit(Path(target) if target else None)
    print(f"{result['golden_set']}: {result['items']} items, {len(result['items_with_problems'])} with problems")
    for iid, problems in result["items_with_problems"].items():
        for p in problems:
            print(f"  {iid}: {p}")
