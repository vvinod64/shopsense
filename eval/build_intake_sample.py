"""Build an intake sample in the Data Sourcing Guide's ratios (4.4.2) from the provided intake records.

The provided 200 records are harsher than the guide asks (1 in 3.6 angry, 1 in 3.5 without an order
reference, 82 claiming more than ₹2,000). This picks a subset, without changing any record, that matches
the guide: 1 in 8 angry, 1 in 6 missing the order reference, two or three claiming refunds above the
automated cap, at least two legal threats, at least two prompt injections, at least one safety report.

    .venv/bin/python eval/build_intake_sample.py        # writes eval/intake_guide_ratio.jsonl
    .venv/bin/python -m shopsense.m1.parser --eval --limit 48 --records eval/intake_guide_ratio.jsonl
"""
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "shopsense" / "intake" / "records.jsonl"
OUT = ROOT / "eval" / "intake_guide_ratio.jsonl"
SIZE, ANGRY, MISSING, ABOVE_CAP = 48, 6, 8, 3          # 1 in 8, 1 in 6, "two or three"
MIN_LEGAL, MIN_INJECTION, MIN_SAFETY = 2, 2, 1
CAP = 2000

LEGAL = re.compile(r"consumer (forum|court)|\blegal\b|lawyer|lawsuit|\bsue\b|\bcourt\b|notice", re.IGNORECASE)
SAFETY = re.compile(r"fire|smoke|spark|burn|shock|overheat|explod|melt", re.IGNORECASE)


def traits(rec: dict) -> dict:
    from shopsense.guardrails import detect_prompt_injection
    gt, text = rec["ground_truth"], rec["raw_text"]
    return {"angry": gt.get("sentiment") == "angry",
            "missing": "order_ref" in (gt.get("missing_fields") or []),
            "above": (gt.get("claimed_amount_inr") or 0) > CAP,
            "legal": bool(LEGAL.search(text)), "injection": detect_prompt_injection(text),
            "safety": bool(SAFETY.search(text))}


def counts(sample: list) -> dict:
    return {k: sum(t[k] for _, t in sample) for k in ("angry", "missing", "above", "legal", "injection", "safety")}


def ok(c: dict, size: int) -> bool:
    return (size == SIZE and c["angry"] == ANGRY and c["missing"] == MISSING and c["above"] == ABOVE_CAP
            and c["legal"] >= MIN_LEGAL and c["injection"] >= MIN_INJECTION and c["safety"] >= MIN_SAFETY)


def build(seed: int = 42, tries: int = 20000) -> list:
    pool = [(rec, traits(rec)) for rec in map(json.loads, SOURCE.read_text(encoding="utf-8").splitlines())]
    rng = random.Random(seed)
    for _ in range(tries):
        rng.shuffle(pool)
        sample, c = [], dict.fromkeys(("angry", "missing", "above", "legal", "injection", "safety"), 0)
        # the special cases first, then fill without overshooting the three exact quotas
        ordered = sorted(pool, key=lambda p: -(p[1]["legal"] + p[1]["injection"] + p[1]["safety"]))
        for rec, t in ordered:
            if len(sample) == SIZE:
                break
            if (c["angry"] + t["angry"] > ANGRY or c["missing"] + t["missing"] > MISSING
                    or c["above"] + t["above"] > ABOVE_CAP):
                continue
            needs_special = (c["legal"] < MIN_LEGAL and t["legal"]) or (c["injection"] < MIN_INJECTION and t["injection"]) \
                or (c["safety"] < MIN_SAFETY and t["safety"])
            specials_done = c["legal"] >= MIN_LEGAL and c["injection"] >= MIN_INJECTION and c["safety"] >= MIN_SAFETY
            if needs_special or specials_done:
                sample.append((rec, t))
                c = counts(sample)
        if ok(c, len(sample)):
            return sorted(sample, key=lambda p: p[0]["record_id"])
    raise RuntimeError("no sample matches the guide's ratios")


if __name__ == "__main__":
    sample = build()
    OUT.write_text("".join(json.dumps(rec, ensure_ascii=False) + "\n" for rec, _ in sample), encoding="utf-8")
    c = counts(sample)
    print(f"{OUT.relative_to(ROOT)}: {len(sample)} records drawn unchanged from {SOURCE.relative_to(ROOT)}")
    print(f"  angry {c['angry']} (1 in {len(sample) / c['angry']:.0f}) · missing order ref {c['missing']} "
          f"(1 in {len(sample) / c['missing']:.0f}) · above ₹{CAP:,} {c['above']} · legal {c['legal']} · "
          f"injection {c['injection']} · safety {c['safety']}")
