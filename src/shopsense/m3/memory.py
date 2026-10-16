"""Per-customer memory.

Working memory with a rolling summary for the current conversation; Supermemory for episodic,
semantic and procedural long-term memory; and a SQLite ledger for counts that must be exact, such as
contacts about an order in the last 14 days.
"""
import json
import re
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Callable, Optional

import litellm
from langchain_core.tools import tool

from shopsense.config import LLM_MODEL, LLM_TIMEOUT_SECONDS, RUNTIME_DIR
from shopsense.m1.parser import REFUND_AUTO_CAP_INR

DB_PATH = RUNTIME_DIR / "memory.db"
PROCEDURES_TAG = "shopsense-procedures"      # shared by all customers
SENTIMENT_RANK = {"positive": 0, "neutral": 1, "frustrated": 2, "angry": 3}
MEMORY_KINDS = ("episodic", "semantic", "procedural")


def customer_tag(customer_ref: str) -> str:
    """Supermemory container tag for one customer."""
    return f"shopsense-{customer_ref.strip().lower()}"


# Working memory and rolling summary

def token_count(text: str) -> int:
    """Token count via tiktoken, with a 4-characters-per-token fallback."""
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)


def summarize_turns(turns: list[dict], prev_summary: str = "") -> str:
    """Summarise turns, merged with the previous summary so earlier facts are kept."""
    convo = "\n".join(f'{t["role"]}: {t["content"]}' for t in turns)
    system = ("You maintain a running summary of a customer-support conversation. Update the existing "
              "summary with the new turns, preserving concrete facts (order references, amounts, dates, "
              "product names, what the customer asked for, what was promised) and the customer's stated "
              "preferences as facts (for example 'Customer asked for replies in Hindi'). Write the summary in "
              "English whatever language the conversation is in. Return only the updated summary, under 100 words.")
    user = f"Existing summary:\n{prev_summary or '(none)'}\n\nNew turns:\n{convo}\n\nUpdated summary:"
    resp = litellm.completion(model=LLM_MODEL, temperature=0, num_retries=3, timeout=LLM_TIMEOUT_SECONDS, messages=[
        {"role": "system", "content": system}, {"role": "user", "content": user}])
    return resp.choices[0].message.content.strip()


class WorkingMemory:
    """Conversation buffer. Past the token budget, all but the last recent_keep turns are folded into
    a rolling summary. Turns are dropped only after the summary succeeds: if the summariser fails, every
    turn is kept and summarising is tried again on the next turn (up to MAX_TURNS, then the oldest go).
    """
    MAX_TURNS = 40

    def __init__(self, recent_keep: int = 4, token_budget: int = 1500,
                 summarizer: Callable[[list, str], str] = summarize_turns):
        self.recent_keep = recent_keep
        self.token_budget = token_budget
        self.summarizer = summarizer
        self.turns: list[dict] = []
        self.summary = ""
        self.compressions = 0
        self.summary_failures = 0

    def tokens(self) -> int:
        return token_count(self.summary) + sum(token_count(t["content"]) for t in self.turns)

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})
        if self.tokens() > self.token_budget and len(self.turns) > self.recent_keep:
            old = self.turns[:-self.recent_keep]
            try:
                summary = self.summarizer(old, self.summary)
            except Exception:
                self.summary_failures += 1
                self.turns = self.turns[-self.MAX_TURNS:]
                return
            self.summary, self.turns = summary, self.turns[-self.recent_keep:]
            self.compressions += 1

    def messages(self) -> list[dict]:
        """Summary (if any) followed by the recent turns."""
        head = ([{"role": "system", "content": f"Summary of earlier conversation: {self.summary}"}]
                if self.summary else [])
        return head + list(self.turns)


# Long-term memory (Supermemory)

@lru_cache(maxsize=1)
def _mem():
    import supermemory
    key = os.getenv("SUPERMEMORY_API_KEY")
    if not key:
        raise RuntimeError("SUPERMEMORY_API_KEY must be set in .env")
    return supermemory.Supermemory(api_key=key)


def remember(text: str, kind: str, container_tag: str, custom_id: Optional[str] = None, **extra) -> str:
    """Write one memory. custom_id makes the write idempotent. Indexing is asynchronous."""
    assert kind in MEMORY_KINDS, f"kind must be one of {MEMORY_KINDS}"
    kwargs = {"content": text, "container_tag": container_tag, "metadata": {"type": kind, **extra}}
    if custom_id:
        kwargs["custom_id"] = custom_id
    return _mem().add(**kwargs).id


def recall(query: str, container_tag: str, k: int = 3, kind: Optional[str] = None) -> list[dict]:
    """Semantic recall from extracted memories, falling back to the document layer when empty."""
    out = []
    for r in _mem().search.memories(q=query, container_tag=container_tag, limit=k * 2).results:
        if getattr(r, "memory", None):
            out.append({"kind": (r.metadata or {}).get("type", "?"), "text": r.memory,
                        "score": round(float(r.similarity or 0.0), 3)})
    if not out:
        for r in _mem().search.documents(q=query, container_tags=[container_tag], limit=k * 2).results:
            # Use the best chunk only; joining chunks repeats older versions of an updated document.
            chunks = [c for c in (r.chunks or []) if c and c.content]
            text = max(chunks, key=lambda c: c.score).content.strip() if chunks else ""
            if text:
                out.append({"kind": (r.metadata or {}).get("type", "?"), "text": text,
                            "score": round(float(r.score or 0.0), 3)})
    if kind:
        out = [m for m in out if m["kind"] == kind]
    return out[:k]


def wait_until_searchable(query: str, container_tag: str, timeout_s: int = 90) -> bool:
    """Poll recall() until the query returns something or the timeout passes."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if recall(query, container_tag, k=1):
            return True
        time.sleep(3)
    return False


# Procedures, each restated from the policy document named in brackets.
PROCEDURES = {
    "proc-refund": (
        "Refund procedure: (1) look up the order before promising anything; (2) for a late-delivery "
        "complaint check the delivery SLA first; (3) compute the refund amount with the refund "
        f"calculator; (4) a refund above ₹{REFUND_AUTO_CAP_INR:,} must never be auto-approved, whatever "
        "the customer's tier, loyalty or sentiment - route it to the Team Lead (Tier 2) queue. "
        "[Refund Authorisation and Escalation Matrix §4.2.1]"),
    "proc-escalation": (
        "Escalation procedure: hand the ticket to a human within one message turn when the customer "
        "asks for a human, uses abusive language (after one warning), contacts us for the third time "
        "about the same order within 14 days, mentions legal action or a consumer forum, reports "
        "physical harm or a product-safety hazard, or requests a refund above the automated cap. "
        "[Escalation Triggers and Customer Communication Standard §2]"),
    "proc-abuse-flag": (
        "Flagged-account procedure: never tell a customer, directly or indirectly, that their account "
        "is flagged, under review or monitored. Use neutral wording such as 'your request is "
        "undergoing standard processing' and escalate to a human. "
        "[Return Fraud and Abuse Prevention Standard §4.4]"),
    "proc-tone": (
        "Communication procedure: never apologise in a way that admits legal fault or negligence on "
        "behalf of Kartway (for example never say 'our warehouse made a mistake'); express regret "
        "for the inconvenience instead. [Escalation Triggers and Customer Communication Standard §4]"),
}


def seed_procedures() -> int:
    """Store the procedures; idempotent via custom_id."""
    for cid, text in PROCEDURES.items():
        remember(text, "procedural", PROCEDURES_TAG, custom_id=cid)
    return len(PROCEDURES)


# Ticket ledger (SQLite)

def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _conn() as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            ticket_id TEXT PRIMARY KEY,
            customer_ref TEXT,
            raw_text TEXT,
            issue_type TEXT,
            sentiment TEXT,
            urgency TEXT,
            order_ref TEXT,
            refund_amount_requested REAL,
            mentions_legal_action INTEGER DEFAULT 0,
            mentions_safety_concern INTEGER DEFAULT 0,
            response TEXT,
            tools_called TEXT,
            escalated INTEGER DEFAULT 0,
            resolution TEXT,
            created_at TEXT
        )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tickets_customer ON tickets(customer_ref)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tickets_order ON tickets(order_ref)")
        # Stated preferences are one current value per key: a newer statement replaces the older one. Semantic
        # recall cannot guarantee that (an overwritten document's extracted memories stay searchable), so the
        # ledger is the source of truth and Supermemory keeps a copy for the agents' free-text recall.
        c.execute("""
        CREATE TABLE IF NOT EXISTS preferences (
            customer_ref TEXT,
            key TEXT,
            value TEXT,
            updated_at TEXT,
            PRIMARY KEY (customer_ref, key)
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS lessons (
            issue_type TEXT,
            lesson TEXT,
            ticket_id TEXT,
            decision TEXT,
            created_at TEXT,
            PRIMARY KEY (issue_type, lesson)
        )""")


# Lessons from human reviewers (experience memory): a reviewer's note on a rejected reply, or on a
# request for changes, is kept for that issue type and given to the drafter on later tickets of the same
# kind. The same lesson twice is stored once (its date refreshed); only the newest few are used.
MAX_LESSONS = 3


def record_lesson(issue_type: Optional[str], note: str, ticket_id: str, decision: str) -> bool:
    lesson = " ".join((note or "").split())[:300]
    if not issue_type or len(lesson) < 8:
        return False
    init_db()
    with _conn() as c:
        c.execute("INSERT INTO lessons (issue_type, lesson, ticket_id, decision, created_at) VALUES (?, ?, ?, ?, ?) "
                  "ON CONFLICT(issue_type, lesson) DO UPDATE SET ticket_id = excluded.ticket_id, "
                  "decision = excluded.decision, created_at = excluded.created_at",
                  (issue_type, lesson, ticket_id, decision, datetime.now(timezone.utc).isoformat()))
    return True


def lessons_for(issue_type: Optional[str], k: int = MAX_LESSONS) -> list:
    if not issue_type:
        return []
    init_db()
    with _conn() as c:
        rows = c.execute("SELECT lesson FROM lessons WHERE issue_type = ? ORDER BY created_at DESC LIMIT ?",
                         (issue_type, k)).fetchall()
    return [r["lesson"] for r in rows]


def log_ticket(
    ticket_id: str,
    customer_ref: Optional[str],
    raw_text: str,
    parsed: dict,
    response: str,
    tools_called: Optional[list] = None,
    escalated: bool = False,
    resolution: str = "auto_resolved",
    created_at: Optional[str] = None,
    write_episode: bool = True,
) -> dict:
    """Record a handled ticket in the ledger and as an episodic memory. A Supermemory failure is
    reported, not raised.
    """
    init_db()
    created_at = created_at or datetime.now().isoformat(timespec="seconds")
    order_ref = parsed.get("order_id")
    with _conn() as c:
        c.execute("""
            INSERT OR REPLACE INTO tickets
            (ticket_id, customer_ref, raw_text, issue_type, sentiment, urgency, order_ref,
             refund_amount_requested, mentions_legal_action, mentions_safety_concern,
             response, tools_called, escalated, resolution, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            ticket_id, customer_ref, raw_text[:2000], parsed.get("issue_type"), parsed.get("sentiment"),
            parsed.get("urgency"), order_ref, parsed.get("refund_amount_requested"),
            int(bool(parsed.get("mentions_legal_action"))), int(bool(parsed.get("mentions_safety_concern"))),
            response[:3000], json.dumps(tools_called or []), int(escalated), resolution, created_at))

    status = {"ledger": "ok", "episodic": "skipped"}
    if customer_ref and write_episode:
        episode = (f"On {created_at[:10]}, customer {customer_ref} raised ticket {ticket_id} about "
                   f"{parsed.get('issue_type', 'an issue')}"
                   + (f" for order {order_ref}" if order_ref else "")
                   + f". Sentiment: {parsed.get('sentiment', 'unknown')}. "
                   + f"Summary: {parsed.get('summary', raw_text[:160])} Outcome: {resolution}.")
        try:
            remember(episode, "episodic", customer_tag(customer_ref), custom_id=f"episode-{ticket_id}",
                     ticket_id=ticket_id, order_ref=order_ref or "", date=created_at[:10])
            status["episodic"] = "ok"
        except Exception as e:
            status["episodic"] = f"failed: {type(e).__name__}"
    return status


def contact_count(customer_ref: str, order_ref: str, days: int = 14, now: Optional[datetime] = None,
                  exclude_ticket: Optional[str] = None) -> int:
    """Distinct tickets from this customer about this order within `days`, excluding `exclude_ticket`.
    Three or more contacts require a human (Escalation Triggers standard 2.3).
    """
    init_db()
    since = ((now or datetime.now()) - timedelta(days=days)).isoformat(timespec="seconds")
    with _conn() as c:
        row = c.execute("""SELECT COUNT(DISTINCT ticket_id) AS n FROM tickets
                           WHERE customer_ref=? AND order_ref=? AND created_at >= ? AND ticket_id != ?""",
                        (customer_ref, order_ref.strip().upper(), since, exclude_ticket or "")).fetchone()
    return int(row["n"])


def get_history(customer_ref: str, limit: int = 3) -> list:
    init_db()
    with _conn() as c:
        rows = c.execute("""
            SELECT ticket_id, issue_type, sentiment, urgency, order_ref, escalated, resolution,
                   created_at, substr(raw_text, 1, 200) AS text_preview
            FROM tickets WHERE customer_ref=? ORDER BY created_at DESC LIMIT ?""",
                         (customer_ref, limit)).fetchall()
    return [dict(r) for r in rows]


def get_sentiment_trend(customer_ref: str) -> dict:
    """Compare the mean sentiment of the last three tickets with the three before them."""
    init_db()
    with _conn() as c:
        rows = c.execute("SELECT sentiment FROM tickets WHERE customer_ref=? ORDER BY created_at DESC LIMIT 10",
                         (customer_ref,)).fetchall()
    sentiments = [r["sentiment"] for r in rows]
    if not sentiments:
        return {"trend": "no_history", "recent_sentiments": []}
    recent = [SENTIMENT_RANK.get(s, 1) for s in sentiments[:3]]
    prior = [SENTIMENT_RANK.get(s, 1) for s in sentiments[3:6]]
    trend = "stable"
    if recent and prior:
        delta = sum(recent) / len(recent) - sum(prior) / len(prior)
        trend = "deteriorating" if delta > 0.5 else "improving" if delta < -0.5 else "stable"
    return {"trend": trend, "recent_sentiments": sentiments[:5], "sample_size": len(sentiments)}


def ledger_stats(customer_ref: str) -> Optional[dict]:
    init_db()
    with _conn() as c:
        row = c.execute("""
            SELECT COUNT(*) AS total_tickets, COALESCE(SUM(escalated),0) AS total_escalated,
                   COALESCE(SUM(refund_amount_requested),0) AS total_refund_requested_inr,
                   MAX(mentions_legal_action) AS legal_history
            FROM tickets WHERE customer_ref=?""", (customer_ref,)).fetchone()
    if not row["total_tickets"]:
        return None
    stats = dict(row)
    stats["sentiment_trend"] = get_sentiment_trend(customer_ref)["trend"]
    flags = []
    if stats["total_escalated"] >= 3:
        flags.append("repeat_escalator")
    if stats["legal_history"]:
        flags.append("legal_history")
    if stats["sentiment_trend"] == "deteriorating":
        flags.append("sentiment_deteriorating")
    stats["risk_flags"] = flags
    stats["legal_history"] = bool(stats["legal_history"])
    return stats


# Agent tools

@tool
def get_customer_history(customer_ref: str, limit: int = 3) -> str:
    """Fetch this customer's most recent support tickets (issue type, sentiment, order, outcome).
    Use for any repeat customer or when the customer refers to a previous issue."""
    ref = customer_ref.strip().upper()
    history = get_history(ref, limit=limit)
    return json.dumps(history, default=str) if history else f"No prior tickets for {ref}."


@tool
def get_customer_summary(customer_ref: str) -> str:
    """Fetch the relationship view for a customer: ticket/escalation counts, refund totals,
    sentiment trend and risk flags (computed), plus relevant past episodes and known facts
    (recalled from long-term memory). Use at the start of a ticket for a returning customer."""
    ref = customer_ref.strip().upper()
    out = {"customer_ref": ref, "stats": ledger_stats(ref)}
    try:
        out["memories"] = recall("customer's past support issues, outcomes and preferences",
                                 customer_tag(ref), k=5)
    except Exception as e:
        out["memories"] = f"unavailable ({type(e).__name__})"
    if not out["stats"] and not out["memories"]:
        return f"No prior history for {ref}."
    return json.dumps(out, default=str, ensure_ascii=False)


@tool
def check_sentiment_trend(customer_ref: str) -> str:
    """Check whether this customer's sentiment is deteriorating across their recent tickets."""
    return json.dumps(get_sentiment_trend(customer_ref.strip().upper()))


@tool
def count_recent_contacts(customer_ref: str, order_ref: str) -> str:
    """Count this customer's tickets about this order in the last 14 days, including the current one
    if it was already logged. Three or more means automatic resolution is locked and the ticket must
    go to a human (third-contact rule)."""
    n = contact_count(customer_ref.strip().upper(), order_ref)
    return json.dumps({"contacts_last_14_days": n, "third_contact_rule_triggered": n >= 3})


@tool
def get_customer_preferences(customer_ref: str) -> str:
    """Recall this customer's known preferences (language, channel, tone) from long-term memory.
    Use to tailor the reply."""
    ref = customer_ref.strip().upper()
    init_db()
    with _conn() as c:
        rows = c.execute("SELECT key, value FROM preferences WHERE customer_ref=?", (ref,)).fetchall()
    if rows:
        return json.dumps({r["key"]: r["value"] for r in rows}, ensure_ascii=False)
    prefs = recall("customer's preferred language, contact channel and tone", customer_tag(ref),
                   k=3, kind="semantic")         # stated before the ledger kept preferences
    return json.dumps(prefs, ensure_ascii=False) if prefs else f"No preferences recorded for {ref}."


@tool
def update_customer_preference(customer_ref: str, key: str, value: str) -> str:
    """Remember a preference the customer explicitly stated. key is one of: preferred_language,
    preferred_channel, communication_tone, notes. Example: 'please reply in Hindi' ->
    key='preferred_language', value='Hindi'."""
    valid = {"preferred_language", "preferred_channel", "communication_tone", "notes"}
    if key not in valid:
        return f"ERROR: key must be one of {sorted(valid)}"
    ref = customer_ref.strip().upper()
    init_db()
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO preferences VALUES (?, ?, ?, ?)",
                  (ref, key, value, datetime.now(timezone.utc).isoformat()))
    label = key.replace("preferred_", "preferred ").replace("_", " ")
    try:                                     # the ledger already holds it; the copy is for free-text recall
        remember(f"Customer {ref} {label}: {value}.", "semantic", customer_tag(ref),
                 custom_id=f"pref-{ref}-{key}", preference_key=key)
    except Exception as e:
        print(f"  [preference copy to long-term memory failed ({type(e).__name__}); kept in the ledger]")
    return f"Preference saved: {ref} {key} = {value}"


@tool
def get_procedure(task: str) -> str:
    """Recall the standard procedure for a task, e.g. 'handle a refund request', 'when to escalate',
    'customer account is flagged', 'how to apologise'. Follow the returned steps."""
    hits = recall(task, PROCEDURES_TAG, k=2, kind="procedural")
    return json.dumps(hits, ensure_ascii=False) if hits else "No stored procedure found."


STATED_LANGUAGE = re.compile(r"\b(?:reply|respond|answer|write|talk|speak)\s+(?:to me\s+)?in\s+"
                             r"(Hindi|English|Tamil|Telugu|Kannada|Bengali|Marathi|Malayalam|Gujarati|Punjabi)\b",
                             re.IGNORECASE)


def customer_context(customer_ref: Optional[str], ticket_text: str = "") -> dict:
    """What the drafter should know about a returning customer: stored preferences and recent tickets.
    A language preference stated in this message is saved first. Memory failures never block a ticket."""
    if not customer_ref:
        return {}
    ref, out = customer_ref.strip().upper(), {}
    stated = STATED_LANGUAGE.search(ticket_text or "")
    try:
        if stated:
            update_customer_preference.invoke({"customer_ref": ref, "key": "preferred_language",
                                               "value": stated.group(1).title()})
            out["stated_now"] = {"preferred_language": stated.group(1).title()}
        out["preferences"] = get_customer_preferences.invoke({"customer_ref": ref})
    except Exception as e:
        out["preferences"] = f"unavailable ({type(e).__name__})"
    out["recent_tickets"] = get_history(ref, limit=3)
    return out


# Self-check and smoke test

def self_check():
    """Offline check of the rolling summary and the ledger counts."""
    import tempfile
    from pathlib import Path
    global DB_PATH

    # Rolling summary triggers past the budget and keeps the recent turns verbatim.
    calls = []
    def stub(turns, prev):
        calls.append((len(turns), prev))
        return (prev + " | " if prev else "") + f"{len(turns)} turns summarised"
    wm = WorkingMemory(recent_keep=4, token_budget=60, summarizer=stub)
    wm.add("user", "My order KW-O-777001 arrived with a cracked screen.")
    for i in range(8):
        wm.add("user", f"Follow-up question number {i} about delivery timelines and refunds.")
    assert calls, "summarisation must fire once the budget is exceeded"
    assert len(wm.turns) <= 4 + 1 and wm.summary, "older turns fold into the summary"
    assert len(calls) > 1 and calls[1][1], "later compressions receive the PREVIOUS summary to merge"
    assert wm.messages()[0]["role"] == "system", "summary leads the next model call"

    # Third-contact rule counts distinct tickets per order within 14 days.
    saved = DB_PATH
    with tempfile.TemporaryDirectory() as d:
        DB_PATH = Path(d) / "test.db"
        now = datetime(2026, 10, 12, 12, 0, 0)
        p = {"issue_type": "shipping_delay", "sentiment": "frustrated", "order_id": "KW-O-1"}
        log_ticket("T1", "KW-C-1", "x", p, "r", created_at="2026-09-20T10:00:00", write_episode=False)
        log_ticket("T2", "KW-C-1", "x", p, "r", created_at="2026-10-01T10:00:00", write_episode=False)
        log_ticket("T3", "KW-C-1", "x", {**p, "sentiment": "angry"}, "r", created_at="2026-10-11T10:00:00",
                   write_episode=False)
        assert contact_count("KW-C-1", "kw-o-1", now=now) == 2, "T1 is outside the 14-day window"
        log_ticket("T4", "KW-C-1", "x", {**p, "sentiment": "angry"}, "r", created_at="2026-10-12T09:00:00",
                   write_episode=False)
        assert contact_count("KW-C-1", "KW-O-1", now=now) == 3
        assert ledger_stats("KW-C-1")["total_tickets"] == 4
    DB_PATH = saved
    print("M3 memory self-check: PASS - rolling summary merges and keeps recent turns; "
          "third-contact count respects the 14-day window.")


def smoke_test():
    ref = "KW-C-SMOKE1"
    tag = customer_tag(ref)

    print(f"Seeding {seed_procedures()} procedural memories ...")
    log_ticket("KW-T-SMOKE-1", ref, "My blender order KW-O-500100 arrived broken.",
               {"issue_type": "product_defect", "sentiment": "frustrated", "order_id": "KW-O-500100",
                "summary": "Blender arrived broken; customer wants a replacement."},
               "Replacement arranged.", resolution="replacement_issued", created_at="2026-10-01T10:00:00")
    print(update_customer_preference.invoke({"customer_ref": ref, "key": "preferred_language", "value": "Hindi"}))

    print("Waiting for Supermemory to extract the memories ...")
    ok = wait_until_searchable("blender", tag) and wait_until_searchable("refund", PROCEDURES_TAG)
    assert ok, "memories were not searchable within 60s"

    print("\nEpisodic - what happened:")
    for m in recall("what went wrong with my kitchen appliance order?", tag, k=2):
        print(f"  {m['score']:.3f} [{m['kind']}] {m['text']}")
    print("\nSemantic - stable facts:")
    for m in recall("which language does the customer prefer?", tag, k=2):
        print(f"  {m['score']:.3f} [{m['kind']}] {m['text']}")
    print("\nProcedural - how to do the task:")
    for m in recall("a customer wants ₹15,000 back, what do I do?", PROCEDURES_TAG, k=1):
        print(f"  {m['score']:.3f} [{m['kind']}] {m['text'][:220]}")

    # Plant a fact, compress past it, and check it survives.
    wm = WorkingMemory(recent_keep=4, token_budget=220)
    wm.add("user", "Hi, my order KW-O-777001 arrived with a cracked phone screen.")
    wm.add("assistant", "I'm sorry to hear that. Let me look into it.")
    for i in range(8):
        wm.add("user", f"Also, part {i}: how do standard and express delivery timelines compare?")
        wm.add("assistant", "Detailed explanation of delivery timelines. " * 8)
    assert wm.compressions >= 1, "budget was not exceeded"
    assert "KW-O-777001" in wm.summary, "planted order reference was lost in summarisation"
    print(f"\nPASS - rolling summary fired {wm.compressions}x and the planted order ref survived.")
    print("Summary:", wm.summary)


if __name__ == "__main__":
    import sys
    self_check()
    if "--offline" not in sys.argv:
        smoke_test()
