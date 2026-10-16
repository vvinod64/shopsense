"""Mock order-system API: orders, customers, shipments, refunds and escalations.

Plain functions shared by the agent tools and the MCP server. Refund rules follow the policy
corpus. Unknown records raise LookupError and invalid input raises ValueError.
"""
import json
import math
import os
from datetime import date, datetime, timedelta
from functools import lru_cache

import pandas as pd

from shopsense.config import AUTOMATION_PAUSED, MOCK_API_DIR, RUNTIME_DIR, automation_paused
from shopsense.m1.parser import REFUND_AUTO_CAP_INR

ACTIONS_FILE = RUNTIME_DIR / "actions.jsonl"
ESCALATIONS_FILE = RUNTIME_DIR / "escalations.jsonl"

REASON_CODES = ("not_delivered", "late_delivery", "damaged", "quality", "wrong_item", "changed_mind")
# Kartway-fault reasons: full refund, restocking fees waived.
FULL_REFUND_REASONS = {"not_delivered", "damaged", "quality", "wrong_item"}

# Change-of-mind restocking fees by category (category addenda override returns-policy 2.16-2.17).
# opened_only: the fee applies only to opened items.
CHANGE_OF_MIND_FEES = {
    "Electronics":       {"fee": 0.085, "opened_only": True,  "basis": "Electronics addendum §4.1 (8.5% on opened, non-defective returns)"},
    "Beauty":            {"fee": 0.175, "opened_only": False, "basis": "Beauty addendum §44.4 4.1 (17.5% on non-defective returns)"},
    "Home & Kitchen":    {"fee": 0.12,  "opened_only": True,  "basis": "Home & Kitchen addendum §5.1-5.2 (12% opened; sealed exempt)"},
    "Sports & Outdoors": {"fee": 0.12,  "opened_only": True,  "basis": "Sports & Outdoors addendum §4.1 (12% on opened, non-defective)"},
    "Toys":              {"fee": 0.12,  "opened_only": True,  "basis": "Toys addendum §4.4 (12% on open-box change of mind)"},
    "Furniture":         {"fee": 0.15,  "opened_only": False, "basis": "Furniture addendum §4.8.4 (15% change of mind, large-format items)"},
    "Apparel":           {"fee": 0.0,   "opened_only": False, "basis": "Apparel addendum §4.8.4 (no restocking fee within the 45-day window)"},
}
NON_RETURNABLE = {"Groceries": "Returns policy §2.4 and Groceries addendum: groceries are non-returnable unless spoiled, damaged or expired"}
# Change-of-mind return windows in calendar days from delivery (returns-policy 2.1 default and category addenda).
RETURN_WINDOWS = {"Beauty": 14, "Electronics": 15, "Home & Kitchen": 30, "Furniture": 42, "Apparel": 45,
                  "Sports & Outdoors": 45, "Toys": 45}
DEFAULT_RETURN_WINDOW = 30
# Shipping policy 4.3.1: days without a tracking scan before a parcel can be declared lost (metro minimum).
LOST_IN_TRANSIT_DAYS = 7


def as_of() -> date:
    """Reference date for return windows; SHOPSENSE_AS_OF (YYYY-MM-DD) pins it for reproducible runs."""
    pinned = os.getenv("SHOPSENSE_AS_OF")
    return date.fromisoformat(pinned) if pinned else date.today()


# Tables

@lru_cache(maxsize=None)
def _table(name: str) -> pd.DataFrame:
    return pd.read_csv(MOCK_API_DIR / f"{name}.csv")


def _row(table: str, key: str, value: str) -> dict:
    df = _table(table)
    hit = df[df[key] == value.strip().upper()]
    if hit.empty:
        raise LookupError(f"{key} {value!r} not found")
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in hit.iloc[0].to_dict().items()}


def _jsonable(d: dict) -> dict:
    return json.loads(json.dumps(d, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


# Read APIs

def get_order(order_ref: str) -> dict:
    order = _row("orders", "order_ref", order_ref)
    product = _row("products", "sku", order["sku"])
    return _jsonable({**order, "product_title": product["title"], "category": product["category"],
                      "warranty_months": product["warranty_months"]})


def get_customer(customer_ref: str) -> dict:
    return _jsonable(_row("customers", "customer_ref", customer_ref))


def get_shipment(order_ref: str) -> dict:
    s = _row("shipments", "order_ref", order_ref)
    return _jsonable({k: s[k] for k in ("tracking_id", "carrier", "last_scan", "last_scan_at", "scan_location")})


def delivery_sla(order_ref: str) -> dict:
    o = _row("orders", "order_ref", order_ref)
    promised, actual = int(o["delivery_promise_days"]), o["actual_delivery_days"]
    if actual is None or o["status"] in ("placed", "shipped"):
        return {"order_ref": o["order_ref"], "status": o["status"], "promised_days": promised,
                "actual_days": None, "days_late": None, "note": "Not delivered yet - check shipment tracking."}
    days_late = max(0, int(actual) - promised)
    return {"order_ref": o["order_ref"], "status": o["status"], "promised_days": promised,
            "actual_days": int(actual), "days_late": days_late, "sla_breach": days_late > 0}


def eligibility_blockers(order_ref: str, reason_code: str) -> list[str]:
    """Policy conditions an automated refund cannot verify or that rule it out. Any blocker means the
    request waits for a human; the calculator still quotes the amount the policy would pay."""
    order = _row("orders", "order_ref", order_ref)
    category = _row("products", "sku", order["sku"])["category"]
    reason = reason_code.strip().lower()
    blockers = []
    if reason == "not_delivered":
        shipment = _row("shipments", "order_ref", order_ref)
        if shipment["last_scan"] == "delivered" or order["status"] == "delivered":
            blockers.append("carrier shows the parcel as delivered: dispute process (Shipping policy 4.4)")
        else:
            last = datetime.fromisoformat(str(shipment["last_scan_at"])).date()
            if (as_of() - last).days < LOST_IN_TRANSIT_DAYS:
                blockers.append(f"last tracking scan under {LOST_IN_TRANSIT_DAYS} days ago: not yet lost in transit "
                                "(Shipping policy 4.3.1)")
            blockers.append("lost-in-transit refund needs a courier tracer first (Shipping policy 4.3.2)")
    elif reason == "changed_mind":
        if order["actual_delivery_days"] is None or order["status"] != "delivered":
            blockers.append("order not delivered: a change-of-mind return starts after delivery")
        else:
            delivered = (datetime.fromisoformat(str(order["placed_at"]))
                         + timedelta(days=float(order["actual_delivery_days"]))).date()
            window = RETURN_WINDOWS.get(category, DEFAULT_RETURN_WINDOW)
            age = (as_of() - delivered).days
            if age > window:
                blockers.append(f"return window closed: delivered {age} days ago, {category} window is {window} days")
    return blockers


def existing_refunds(order_ref: str) -> list[dict]:
    df = _table("refunds")
    return [_jsonable(r) for r in df[df["order_ref"] == order_ref.strip().upper()].to_dict("records")]


# Refund calculator

def quote_refund(order_ref: str, reason_code: str, item_condition: str = "opened") -> dict:
    """Refund the policy allows for an order and reason. item_condition is 'opened' or 'unopened'."""
    reason = reason_code.strip().lower()
    if reason not in REASON_CODES:
        raise ValueError(f"reason_code must be one of {list(REASON_CODES)}")
    if item_condition not in ("opened", "unopened"):
        raise ValueError("item_condition must be 'opened' or 'unopened'")
    order = get_order(order_ref)
    value = float(order["order_value_inr"])
    category = order["category"]
    pct, basis, notes, credits = 1.0, [], [], 0

    if reason in FULL_REFUND_REASONS:
        basis.append("Restocking fees are waived for defective, damaged, wrong-item or undelivered orders "
                     "(category addenda waiver clauses)")
    elif reason == "late_delivery":
        sla = delivery_sla(order_ref)
        if sla["days_late"] is None:
            raise ValueError("order has not been delivered yet; late-delivery compensation applies after delivery")
        d = sla["days_late"]
        basis.append(f"Shipping policy §4.2 delay compensation (order was {d} day(s) past SLA)")
        if d == 0:
            pct = 0.0
            notes.append("Delivered within SLA - no delay compensation applies.")
        elif d <= 3:
            pct, credits = 0.0, 150
            notes.append("1-3 days late: refund of the shipping fee only; the fee is not in the order data, "
                         "so a human must confirm the amount.")
        elif d <= 7:
            pct, credits = 0.25, 300
        else:
            credits = 500
            notes.append("8+ days late: full refund under the lost-in-transit protocol.")
        if d:
            notes.append(f"Goodwill: {credits} Kartway Credits per Shipping policy §4.2 - there is no credits API "
                         "in the order system, so the loyalty team applies them.")
        notes.append("SLA counts business days; the order data records calendar days, so treat this as an estimate.")
    else:
        if category in NON_RETURNABLE:
            pct = 0.0
            basis.append(NON_RETURNABLE[category])
        else:
            rule = CHANGE_OF_MIND_FEES[category]
            fee = 0.0 if (rule["opened_only"] and item_condition == "unopened") else rule["fee"]
            pct = 1.0 - fee
            basis.append(rule["basis"])
            notes.append("Fixed fees stated in USD in the corpus (e.g. return shipping §2.11) are not applied.")

    amount = round(value * pct, 2)
    return {"order_ref": order["order_ref"], "category": category, "order_value_inr": value,
            "reason": reason, "item_condition": item_condition, "refund_pct": pct,
            "refund_amount_inr": amount, "goodwill_credits": credits,
            "eligibility_blockers": (blockers := eligibility_blockers(order_ref, reason)),
            "auto_approvable": 0 < amount <= REFUND_AUTO_CAP_INR and not blockers,
            "cap_inr": REFUND_AUTO_CAP_INR, "policy_basis": basis, "notes": notes}


# Refund Authorisation Matrix: lowest human role allowed to approve an amount.
APPROVAL_TIERS = [(10_000, "team_lead"), (50_000, "operations_manager"), (float("inf"), "finance")]
ROLE_LABELS = {"team_lead": "Team Lead", "operations_manager": "Operations Manager", "finance": "Finance Department"}


def approval_role(amount_inr: float) -> str:
    return next(role for limit, role in APPROVAL_TIERS if float(amount_inr or 0) <= limit)


# Write APIs (append-only action log)

def _append(path, record: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def _actions(order_ref: str, kind: str) -> list[dict]:
    if not ACTIONS_FILE.exists():
        return []
    rows = [json.loads(line) for line in ACTIONS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r["order_ref"] == order_ref and r["kind"] == kind]


def _hold_blockers(hold_reasons) -> list:
    """Reasons the caller or operations require a person: these hold a request whatever its amount."""
    return ([AUTOMATION_PAUSED] if automation_paused() else []) + [f"needs a person: {r}" for r in hold_reasons or []]


def create_refund_request(order_ref: str, reason_code: str, ticket_id: str, item_condition: str = "opened",
                          hold_reasons: list = None) -> dict:
    """Create a refund request. The amount always comes from quote_refund. Auto-approved only when the
    amount is within the cap, the account is not flagged, the order has no earlier refund, automation is
    not paused and the caller gave no reason to hold it.
    """
    quote = quote_refund(order_ref, reason_code, item_condition)
    request_id = f"RR-{ticket_id}-{quote['order_ref']}"
    earlier = [a for a in _actions(quote["order_ref"], "refund") if a["request_id"] == request_id]
    if earlier:
        return earlier[0]          # idempotent replay
    order = get_order(order_ref)
    customer = get_customer(order["customer_ref"])
    prior = [r for r in existing_refunds(order_ref) if r["status"] != "rejected"] + _actions(quote["order_ref"], "refund")
    blockers = list(quote["eligibility_blockers"])
    if quote["refund_amount_inr"] <= 0:
        blockers.append("no refundable amount under policy")
    if quote["refund_amount_inr"] > REFUND_AUTO_CAP_INR:
        blockers.append(f"amount above the ₹{REFUND_AUTO_CAP_INR:,} automated cap")
    if customer.get("flagged_for_abuse"):
        blockers.append("account requires manual review")
    if prior:
        blockers.append("order already has a refund on record")
    blockers += _hold_blockers(hold_reasons)
    status = "auto_approved" if not blockers else "pending_human_approval"
    record = {"kind": "refund", "request_id": request_id,
              "order_ref": quote["order_ref"], "ticket_id": ticket_id, "reason": quote["reason"],
              "amount_inr": quote["refund_amount_inr"], "status": status, "blockers": blockers,
              "policy_basis": quote["policy_basis"], "created_at": datetime.now().isoformat(timespec="seconds")}
    return _append(ACTIONS_FILE, record)


def create_replacement_request(order_ref: str, reason_code: str, ticket_id: str, hold_reasons: list = None) -> dict:
    """Replacement for a damaged, defective or wrong item."""
    reason = reason_code.strip().lower()
    if reason not in FULL_REFUND_REASONS - {"not_delivered"}:
        raise ValueError("replacements are only for damaged, quality or wrong_item")
    order = get_order(order_ref)
    request_id = f"RP-{ticket_id}-{order['order_ref']}"
    earlier = [a for a in _actions(order["order_ref"], "replacement") if a["request_id"] == request_id]
    if earlier:
        return earlier[0]
    product = _row("products", "sku", order["sku"])
    customer = get_customer(order["customer_ref"])
    blockers = []
    if not product["in_stock"]:
        blockers.append("item out of stock")
    if float(order["order_value_inr"]) > REFUND_AUTO_CAP_INR:
        blockers.append(f"order value above the ₹{REFUND_AUTO_CAP_INR:,} automated cap")
    if customer.get("flagged_for_abuse"):
        blockers.append("account requires manual review")
    blockers += _hold_blockers(hold_reasons)
    record = {"kind": "replacement", "request_id": request_id,
              "order_ref": order["order_ref"], "ticket_id": ticket_id, "reason": reason,
              "status": "auto_approved" if not blockers else "pending_human_approval", "blockers": blockers,
              "created_at": datetime.now().isoformat(timespec="seconds")}
    return _append(ACTIONS_FILE, record)


def actions_for_ticket(ticket_id: str) -> list[dict]:
    if not ACTIONS_FILE.exists():
        return []
    rows = [json.loads(line) for line in ACTIONS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r.get("ticket_id") == ticket_id]


def record_human_decision(request_id: str, ticket_id: str, order_ref: str, decision: str,
                          approver: str, note: str = "") -> dict:
    """Record who decided on a pending request, when and why. Idempotent per request_id."""
    if decision not in ("approved", "rejected"):
        raise ValueError("decision must be 'approved' or 'rejected'")
    if not approver:
        raise ValueError("an approver identity is required")
    earlier = [a for a in _actions(order_ref, "human_decision") if a["request_id"] == request_id]
    if earlier:
        return earlier[0]
    return _append(ACTIONS_FILE, {"kind": "human_decision", "request_id": request_id, "order_ref": order_ref,
                                  "ticket_id": ticket_id, "decision": decision, "approver": approver,
                                  "note": note, "decided_at": datetime.now().isoformat(timespec="seconds")})


def record_escalation(ticket_id: str, reason: str, suggested_action: str = "", source: str = "agent") -> dict:
    return _append(ESCALATIONS_FILE, {"ticket_id": ticket_id, "reason": reason, "suggested_action": suggested_action,
                                      "source": source, "escalated_at": datetime.now().isoformat(timespec="seconds")})


# Fault injection

_FAULTY_APIS = ("get_order", "get_customer", "get_shipment", "delivery_sla")
_ORIGINALS: dict = {}


def install_faults(config) -> None:
    """Wrap the read APIs with the seeded fault injector; remove_faults() restores them."""
    from shopsense.reliability import make_flaky
    g = globals()
    for name in _FAULTY_APIS:
        _ORIGINALS.setdefault(name, g[name])
        g[name] = make_flaky(_ORIGINALS[name], config, malformed=lambda *a, **k: {"order_ref": a[0] if a else None})


def remove_faults() -> None:
    globals().update(_ORIGINALS)
    _ORIGINALS.clear()


if os.getenv("SHOPSENSE_FAULTS"):
    from shopsense.reliability import FaultConfig
    install_faults(FaultConfig.parse(os.environ["SHOPSENSE_FAULTS"]))


def self_check():
    """Offline check of the refund calculator against the policy rules."""
    orders = _table("orders")
    late = orders[(orders.actual_delivery_days - orders.delivery_promise_days).between(4, 7)].iloc[0]
    q = quote_refund(late.order_ref, "late_delivery")
    assert q["refund_pct"] == 0.25 and q["goodwill_credits"] == 300, q
    beauty = orders.merge(_table("products"), on="sku")
    b = beauty[beauty.category == "Beauty"].iloc[0]
    assert abs(quote_refund(b.order_ref, "changed_mind")["refund_pct"] - 0.825) < 1e-9
    g = beauty[beauty.category == "Groceries"].iloc[0]
    assert quote_refund(g.order_ref, "changed_mind")["refund_amount_inr"] == 0
    e = beauty[beauty.category == "Electronics"].iloc[0]
    assert quote_refund(e.order_ref, "changed_mind", "unopened")["refund_pct"] == 1.0
    assert quote_refund(e.order_ref, "damaged")["refund_pct"] == 1.0
    os.environ["SHOPSENSE_AS_OF"] = "2027-01-01"          # long after every delivery: windows closed
    expired = quote_refund(e.order_ref, "changed_mind", "unopened")
    assert expired["eligibility_blockers"] and not expired["auto_approvable"], expired
    shipped = orders[orders.status == "shipped"].iloc[0]
    assert any("tracer" in b for b in quote_refund(shipped.order_ref, "not_delivered")["eligibility_blockers"])
    assert not quote_refund(late.order_ref, "late_delivery")["eligibility_blockers"]
    os.environ.pop("SHOPSENSE_AS_OF")
    for bad in [lambda: get_order("KW-O-999999"), lambda: quote_refund(late.order_ref, "because")]:
        try:
            bad(); raise AssertionError("expected an error")
        except (LookupError, ValueError):
            pass
    print("mock_api self-check: PASS - late 4-7 days = 25%, beauty change-of-mind = 82.5%, groceries = 0, "
          "unopened electronics = 100%, closed return windows and unconfirmed lost parcels never auto-approve, "
          "unknown order/reason rejected.")


if __name__ == "__main__":
    self_check()
