"""FastAPI service for ShopSense.

/chat runs a ticket through the workflow: 200 when resolved, 202 when waiting for a human, 422 when
a guardrail blocks the reply. Sessions are stored in SQLite and bound to one customer.
/tickets/pending and /tickets/{id}/decision need the reviewer bearer token, and an approval needs a
role whose refund-matrix tier covers the amount.

Run:  .venv/bin/uvicorn shopsense.m8.api:app --port 8000
"""
import hashlib
import json
import logging
import os
import re
import secrets
import sqlite3
import threading
import uuid
from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from shopsense.config import RUNTIME_DIR
from shopsense.guardrails import guardrail_check
from shopsense.m3.memory import WorkingMemory

app = FastAPI(title="ShopSense - Kartway Customer Care", version="1.0")
_log = logging.getLogger("shopsense.api")
SESSION_DB = RUNTIME_DIR / "sessions.sqlite"
_LOCK = threading.Lock()
_SESSION_LOCKS: dict[str, threading.Lock] = {}
ROLE_RANK = {"team_lead": 1, "operations_manager": 2, "finance": 3}


# Sessions

def _db() -> sqlite3.Connection:
    SESSION_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SESSION_DB)
    conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, customer_ref TEXT, memory TEXT)")
    return conn


def load_session(session_id: str, customer_ref: Optional[str]) -> WorkingMemory:
    with _db() as conn:
        row = conn.execute("SELECT customer_ref, memory FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if row and row[0] != customer_ref:
        raise HTTPException(status_code=409, detail="session belongs to a different customer")
    memory = WorkingMemory(recent_keep=4, token_budget=800)
    if row:
        saved = json.loads(row[1])
        memory.turns, memory.summary = saved["turns"], saved["summary"]
    return memory


def save_session(session_id: str, customer_ref: Optional[str], memory: WorkingMemory) -> None:
    with _db() as conn:
        conn.execute("INSERT OR REPLACE INTO sessions VALUES (?, ?, ?)", (session_id, customer_ref, json.dumps(
            {"turns": memory.turns, "summary": memory.summary}, ensure_ascii=False)))


def _context(memory: WorkingMemory) -> str:
    """Earlier turns of the conversation, passed to the next ticket as context."""
    lines = [f"{m['role']}: {m['content'][:300]}" for m in memory.messages()]
    return ("\n\n(Earlier in this conversation, context only:\n" + "\n".join(lines) + ")") if lines else ""


# Reviewers

# Swagger shows an "Authorize" button for this scheme; a missing or wrong token is still a 401 from authorize_reviewer.
reviewer_token = HTTPBearer(auto_error=False, scheme_name="ReviewerToken",
                            description="the token from POST /auth/staff (demo: ops / ops), or SHOPSENSE_REVIEW_TOKEN; "
                                        "paste the token only, without 'Bearer'")


def _bearer(credentials: Optional[HTTPAuthorizationCredentials]) -> Optional[str]:
    return f"Bearer {credentials.credentials}" if credentials else None


# Reviewer identities, from three sources:
#  1. SSO: when SHOPSENSE_OIDC_ISSUER is set, a signed OIDC token (JWT) is verified against the issuer's keys;
#     the reviewer is its email (or subject) and the role comes from the role claim.
#  2. Named reviewers in runtime/reviewers.json, each with their own token (stored only as a SHA-256 hash) and
#     refund-matrix role:  python -m shopsense.m8.api --add-reviewer NAME ROLE
#  3. The single reviewer in .env (SHOPSENSE_REVIEW_TOKEN / _NAME / _ROLE), kept for the demo.
#  4. Staff who signed in on /ui with a username and password (POST /auth/staff) and got a signed session token.
#     The demo has one account per refund-matrix role, password = username, like the demo customers.
#     SHOPSENSE_DEMO_STAFF=off removes them.
REVIEWERS_FILE = RUNTIME_DIR / "reviewers.json"
DEMO_STAFF = {"lead": "team_lead", "ops": "operations_manager", "finance": "finance"}


def demo_staff() -> dict:
    off = os.getenv("SHOPSENSE_DEMO_STAFF", "on").strip().lower() in ("0", "false", "off", "no")
    return {} if off else DEMO_STAFF


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _reviewers() -> list:
    path = Path(os.getenv("SHOPSENSE_REVIEWERS_FILE") or REVIEWERS_FILE)
    out = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    if os.getenv("SHOPSENSE_REVIEW_TOKEN"):
        out.append({"name": os.getenv("SHOPSENSE_REVIEWER_NAME", "demo-reviewer"),
                    "role": os.getenv("SHOPSENSE_REVIEWER_ROLE", "team_lead"),
                    "token_sha256": _token_hash(os.environ["SHOPSENSE_REVIEW_TOKEN"])})
    return out


def add_reviewer(name: str, role: str) -> str:
    """Register a named reviewer; returns their token, which is shown once and stored only as a hash."""
    if role not in ROLE_RANK:
        raise ValueError(f"role must be one of {sorted(ROLE_RANK)}")
    path = Path(os.getenv("SHOPSENSE_REVIEWERS_FILE") or REVIEWERS_FILE)
    people = [r for r in (json.loads(path.read_text(encoding="utf-8")) if path.exists() else []) if r["name"] != name]
    token = secrets.token_urlsafe(24)
    people.append({"name": name, "role": role, "token_sha256": _token_hash(token)})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(people, indent=2), encoding="utf-8")
    return token


@lru_cache(maxsize=4)
def _jwks_client(url: str):
    import jwt
    return jwt.PyJWKClient(url)


def _oidc_identity(token: str) -> Optional[dict]:
    """Reviewer from a signed SSO token, or None when SSO is not configured or this is not a JWT."""
    issuer = os.getenv("SHOPSENSE_OIDC_ISSUER")
    if not issuer or token.count(".") != 2:
        return None
    import jwt
    url = os.getenv("SHOPSENSE_OIDC_JWKS_URL") or issuer.rstrip("/") + "/.well-known/jwks.json"
    try:
        key = _jwks_client(url).get_signing_key_from_jwt(token).key
        claims = jwt.decode(token, key, algorithms=["RS256", "ES256"], issuer=issuer,
                            audience=os.getenv("SHOPSENSE_OIDC_AUDIENCE"), options={"require": ["exp", "sub"]})
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"reviewer token rejected: {type(e).__name__}")
    return {"name": claims.get("email") or claims["sub"],
            "role": claims.get(os.getenv("SHOPSENSE_OIDC_ROLE_CLAIM", "shopsense_role"), "")}


def reviewer_identity(supplied: str) -> Optional[dict]:
    """{"name", "role"} for a staff session, SSO token or reviewer token; None if it is none of them."""
    if not supplied:
        return None
    payload = _verified(supplied) if supplied.startswith("staff:") else None
    if payload:
        _, name, role = payload.split(":", 2)
        return {"name": name, "role": role}
    identity = _oidc_identity(supplied)
    if identity is None:
        digest = _token_hash(supplied)
        identity = next((r for r in _reviewers() if secrets.compare_digest(digest, r["token_sha256"])), None)
    return identity


def authorize_reviewer(authorization: Optional[str], approver: Optional[str] = None,
                       required_role: str = "team_lead") -> str:
    """Check who the reviewer is (SSO token, named reviewer or the .env reviewer), that the approver named in
    the decision is that person, and that their refund-matrix role covers the amount.
    """
    supplied = (authorization or "").removeprefix("Bearer ").strip()
    if not _reviewers() and not demo_staff() and not os.getenv("SHOPSENSE_OIDC_ISSUER"):
        raise HTTPException(status_code=503, detail="reviewer access is not configured")
    identity = reviewer_identity(supplied)
    if identity is None:
        raise HTTPException(status_code=401, detail="reviewer authentication required")
    name, role = identity["name"], identity["role"]
    if approver is not None and approver != name:
        raise HTTPException(status_code=403, detail="approver does not match the authenticated reviewer")
    if ROLE_RANK.get(role, 0) < ROLE_RANK[required_role]:
        from shopsense.mock_api import ROLE_LABELS
        label = ROLE_LABELS.get(required_role, required_role)
        raise HTTPException(status_code=403, detail=f"{'An' if label[0] in 'AEIOU' else 'A'} {label} or above "
                                                    f"must approve this amount")
    return name


# Customer sign-in (demo): the customer id is checked against the order system's customer table and, for this
# demo, the password is the customer id. The server returns a signed session token (HMAC, 12 hours); /chat and
# /tickets/{id}/status take the customer from that token (X-Customer-Token) rather than from what the browser
# claims. A customer_ref sent without a token is refused (401) unless SHOPSENSE_REQUIRE_CUSTOMER_LOGIN=false,
# so nobody can read another customer's orders by typing their id. No customer_ref and no token is a guest.
SESSION_TTL_SECONDS = 12 * 3600


def login_required() -> bool:
    return os.getenv("SHOPSENSE_REQUIRE_CUSTOMER_LOGIN", "true").strip().lower() not in ("0", "false", "off", "no")


@lru_cache(maxsize=1)
def _session_secret() -> bytes:
    if os.getenv("SHOPSENSE_SESSION_SECRET"):
        return os.environ["SHOPSENSE_SESSION_SECRET"].encode()
    path = RUNTIME_DIR / "session_secret"          # stable across restarts, so a held customer stays signed in
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(32))
    return path.read_text().strip().encode()


def _sign(customer_ref: str, expires: int) -> str:
    import hmac
    payload = f"{customer_ref}.{expires}"
    return payload + "." + hmac.new(_session_secret(), payload.encode(), hashlib.sha256).hexdigest()


def _verified(token: str) -> Optional[str]:
    """What a signed session token was issued for, or None if it is forged or expired."""
    import hmac
    import time as _time
    try:
        subject, expires, _sig = token.rsplit(".", 2)
        ok = hmac.compare_digest(_sign(subject, int(expires)), token) and int(expires) > _time.time()
    except ValueError:
        ok = False
    return subject if ok else None


def customer_from_token(token: Optional[str]) -> Optional[str]:
    """The customer a session token was issued to; 401 if it is forged, expired or not a customer's."""
    if not token:
        return None
    ref = _verified(token)
    if not ref or not re.fullmatch(r"KW-C-\d{5}", ref):
        raise HTTPException(status_code=401, detail="customer session expired or invalid; sign in again")
    return ref


def acting_customer(token: Optional[str], claimed: Optional[str]) -> Optional[str]:
    """Who the caller is: the signed-in customer, a guest (None), or - only with login switched off - the
    customer_ref they sent."""
    signed_in = customer_from_token(token)
    if signed_in:
        return signed_in                        # the token decides, whatever the body or query claims
    if claimed and login_required():
        raise HTTPException(status_code=401, detail="sign in first: POST /auth/customer, then send the token as "
                                                    "X-Customer-Token (or leave customer_ref out to continue as a guest)")
    return claimed


class StaffLogin(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class CustomerLogin(BaseModel):
    customer_ref: str = Field(pattern=r"^KW-C-\d{5}$")
    password: str = Field(min_length=1, max_length=128)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    customer_ref: Optional[str] = Field(default=None, pattern=r"^KW-C-\d{5}$")
    session_id: Optional[str] = Field(default=None, max_length=64)


class ChatResponse(BaseModel):
    ticket_id: str
    session_id: str
    status: Literal["resolved", "awaiting_human", "rejected"]
    reply: str
    policy_sources: List[str]
    guardrail_flags: List[str]


class Decision(BaseModel):
    action: Literal["approved", "rejected", "changes_requested"]
    approver: str = Field(min_length=3, max_length=120)
    note: str = ""
    edited_reply: str = ""


def _failure(exc: Exception, ticket_id: str) -> JSONResponse:
    """Rate limits become 503 with Retry-After; other errors 500."""
    text = f"{type(exc).__name__}: {exc}"
    if any(m in text for m in ("429", "RateLimit", "RESOURCE_EXHAUSTED", "ServiceUnavailable", "UNAVAILABLE")):
        return JSONResponse(status_code=503, headers={"Retry-After": "60"},
                            content={"message": "LLM provider is busy or rate-limited; retry shortly",
                                     "ticket_id": ticket_id})
    return JSONResponse(status_code=500, content={"message": "ticket processing failed", "ticket_id": ticket_id,
                                                  "error": type(exc).__name__})


def engine() -> str:
    """Which system answers /chat: the M6 multi-agent team (default; the problem statement asks for a
    multi-agent assistant) or the M5 workflow (SHOPSENSE_ENGINE=workflow: fewer model calls per ticket).
    Both pause risky tickets for a reviewer, so every reviewer endpoint works with either."""
    e = os.getenv("SHOPSENSE_ENGINE", "team").strip().lower()
    return e if e in ("workflow", "team") else "team"


def _team(coro):
    import asyncio
    return asyncio.run(coro)          # FastAPI runs sync endpoints in a worker thread, which has no event loop


@app.get("/health")
def health():
    return {"status": "ok", "engine": engine()}          # no LLM call


@app.post("/chat", response_model=ChatResponse, responses={202: {"model": ChatResponse}, 422: {}})
def chat(req: ChatRequest, x_customer_token: Optional[str] = Header(default=None)):
    from shopsense.m5.workflow import start_ticket
    req.customer_ref = acting_customer(x_customer_token, req.customer_ref)
    session_id = req.session_id or uuid.uuid4().hex
    with _LOCK:
        lock = _SESSION_LOCKS.setdefault(session_id, threading.Lock())
    with lock:                                  # one ticket at a time per conversation
        memory = load_session(session_id, req.customer_ref)
        ticket_id = f"CHAT-{uuid.uuid4().hex[:12]}"     # unique across restarts; also the checkpoint thread id
        try:
            if engine() == "team":
                from shopsense.m6.team import run_ticket, team_to_result
                result = team_to_result(_team(run_ticket(req.message + _context(memory), ticket_id, req.customer_ref,
                                                         durable=True)))
            else:
                result = start_ticket(req.message + _context(memory), ticket_id, req.customer_ref)
        except Exception as e:
            return _failure(e, ticket_id)
        guard = guardrail_check(req.message, result)
        if guard["action"] == "block":
            raise HTTPException(status_code=422, detail={"message": "guardrail check failed", "ticket_id": ticket_id,
                                                         "flags": guard["flags"]})
        try:       # the ticket is already done: a memory problem must not turn its result into an error
            memory.add("user", req.message)
            memory.add("assistant", result.get("draft") or "")
            save_session(session_id, req.customer_ref, memory)
        except Exception as e:
            _log.warning("ticket %s completed but the session was not saved: %s", ticket_id, type(e).__name__)
    parked = result["status"] in ("awaiting_approval", "escalated_to_human", "execution_failed_human_queue")
    body = ChatResponse(ticket_id=ticket_id, session_id=session_id,
                        status="awaiting_human" if parked else "resolved",
                        reply=result.get("draft") or "", guardrail_flags=guard["flags"],
                        policy_sources=sorted({h["source"] for h in result.get("policy_hits") or []}))
    return JSONResponse(status_code=202 if parked else 200, content=body.model_dump())


@app.get("/tickets/pending")
def pending(credentials: Optional[HTTPAuthorizationCredentials] = Depends(reviewer_token)):
    authorize_reviewer(_bearer(credentials))
    from datetime import datetime, timezone
    from shopsense.m5.workflow import APPROVAL_TTL_HOURS, pending_details
    from shopsense.m6.team import team_pending, team_ticket_state
    details = [{**d, "engine": "workflow"} for d in pending_details()]
    now = datetime.now(timezone.utc)
    for t in _team(team_pending()):
        st = _team(team_ticket_state(t)) or {}
        age = (now - datetime.fromisoformat(st["created_at"])).total_seconds() / 3600 if st.get("created_at") else 0.0
        res = st.get("resolution") or {}
        details.append({"ticket_id": t, "parked_at": st.get("created_at"), "age_hours": round(age, 1),
                        "overdue": age > APPROVAL_TTL_HOURS, "risk": (st.get("review") or {}).get("risk") or [],
                        "engine": "team", "customer_ref": st.get("customer_ref"),
                        "message": (st.get("raw_text") or "").split("\n\n(Earlier")[0],
                        "proposed_action": res.get("action"), "amount_inr": (res.get("quote") or {}).get("refund_amount_inr"),
                        "draft": st.get("draft")})
    return {"pending": [d["ticket_id"] for d in details], "details": details, "ttl_hours": APPROVAL_TTL_HOURS}


@app.post("/auth/customer")
def customer_login(login: CustomerLogin):
    """Sign a customer in (demo rule: the password is the customer id) and return a session token."""
    import time as _time
    from shopsense import mock_api
    try:
        mock_api.get_customer(login.customer_ref)
        known = True
    except (LookupError, KeyError, IndexError):
        known = False
    if not known or not secrets.compare_digest(login.password.strip().upper(), login.customer_ref):
        raise HTTPException(status_code=401, detail="unknown customer id or wrong password")
    return {"customer_ref": login.customer_ref, "token": _sign(login.customer_ref, int(_time.time()) + SESSION_TTL_SECONDS)}


@app.post("/auth/staff")
def staff_login(login: StaffLogin):
    """Support-staff sign-in (demo accounts: lead, ops, finance; the password is the username). The returned
    token is used as the reviewer Bearer token, so the console never needs SHOPSENSE_REVIEW_TOKEN."""
    import time as _time
    name = login.username.strip().lower()
    role = demo_staff().get(name)
    if not role or not secrets.compare_digest(login.password.strip().lower(), name):
        raise HTTPException(status_code=401, detail="unknown staff username or wrong password")
    return {"name": name, "role": role, "token": _sign(f"staff:{name}:{role}", int(_time.time()) + SESSION_TTL_SECONDS)}


@app.get("/reviewer/me")
def reviewer_me(credentials: Optional[HTTPAuthorizationCredentials] = Depends(reviewer_token)):
    """Who the token belongs to (name and refund-matrix role), for the support console's sign-in."""
    authorization = _bearer(credentials)
    name = authorize_reviewer(authorization)
    identity = reviewer_identity((authorization or "").removeprefix("Bearer ").strip()) or {}
    return {"name": name, "role": identity.get("role")}


# Web pages for the demo: one per role, served by the same app (no extra dependency)
UI_DIR = Path(__file__).resolve().parent / "ui"


@app.get("/ui", include_in_schema=False)
def ui_home():
    from fastapi.responses import HTMLResponse
    return HTMLResponse((UI_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/ui/customer", include_in_schema=False)
def ui_customer():
    from fastapi.responses import HTMLResponse
    return HTMLResponse((UI_DIR / "customer.html").read_text(encoding="utf-8"))


@app.get("/ui/support", include_in_schema=False)
def ui_support():
    from fastapi.responses import HTMLResponse
    return HTMLResponse((UI_DIR / "support.html").read_text(encoding="utf-8"))


CUSTOMER_STATUS = {"awaiting_approval": "awaiting_human", "escalated_to_human": "awaiting_human",
                   "execution_failed_human_queue": "awaiting_human", "changes_requested": "awaiting_human",
                   "rejected": "rejected"}


@app.get("/tickets/{ticket_id}/status")
def ticket_status(ticket_id: str, customer_ref: Optional[str] = None,
                  x_customer_token: Optional[str] = Header(default=None)):
    """Customer view of one ticket: its status and the latest reply, so a held ticket's outcome reaches the
    customer. A customer's ticket is only shown to that customer; the random ticket id is the guest's key."""
    from shopsense.m5.workflow import _cfg, durable_graph
    customer_ref = acting_customer(x_customer_token, customer_ref)
    snap = durable_graph().get_state(_cfg(ticket_id))
    values, waiting = snap.values, snap.next == ("approval",)       # a paused ticket is waiting, whatever its status
    if not values:
        from shopsense.m6.team import team_ticket_state
        values = _team(team_ticket_state(ticket_id)) or {}
        waiting = bool(values.get("waiting"))
    if not values:
        raise HTTPException(status_code=404, detail="no such ticket")
    owner = values.get("customer_ref")
    if owner and owner != customer_ref:
        raise HTTPException(status_code=404, detail="no such ticket")
    status = "awaiting_approval" if waiting else (values.get("status") or "unknown")
    return {"ticket_id": ticket_id, "status": CUSTOMER_STATUS.get(status, "resolved" if "resolved" in status else status),
            "reply": values.get("draft")}


@app.post("/tickets/expire")
def expire(credentials: Optional[HTTPAuthorizationCredentials] = Depends(reviewer_token)):
    """Close held tickets that waited past the time limit (nothing executes; a person follows up)."""
    authorize_reviewer(_bearer(credentials))
    from shopsense.m5.workflow import APPROVAL_TTL_HOURS, expire_overdue
    from shopsense.m6.team import expire_team_overdue
    return {"expired": expire_overdue() + _team(expire_team_overdue(APPROVAL_TTL_HOURS))}


@app.post("/tickets/{ticket_id}/decision")
def decide(ticket_id: str, d: Decision,
           credentials: Optional[HTTPAuthorizationCredentials] = Depends(reviewer_token)):
    from shopsense.m5.workflow import _cfg, durable_graph, resume_ticket
    from shopsense.mock_api import approval_role
    authorization = _bearer(credentials)
    authorize_reviewer(authorization, d.approver)
    snap = durable_graph().get_state(_cfg(ticket_id))
    if snap.next != ("approval",):              # not a workflow ticket waiting: maybe a team ticket
        from shopsense.m6.team import team_ticket_state
        team_state = _team(team_ticket_state(ticket_id))
        if team_state and team_state.get("waiting"):
            return _decide_team(ticket_id, d, authorization, team_state)
    state = snap.values
    if d.action == "approved":
        quote = ((state.get("proposal") or {}).get("quote") or {})
        authorize_reviewer(authorization, d.approver, approval_role(quote.get("refund_amount_inr") or 0))
    try:
        out = resume_ticket(ticket_id, d.action, d.approver, d.note, d.edited_reply)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        return _failure(e, ticket_id)
    guard = guardrail_check(state.get("raw_text", ""), out)
    if guard["action"] == "block":
        raise HTTPException(status_code=422, detail={"message": "guardrail check failed", "ticket_id": ticket_id,
                                                     "flags": guard["flags"]})
    return {"ticket_id": ticket_id, "status": out["status"], "reply": out.get("draft"),
            "executed": (out.get("outcome") or {}).get("executed")}


def _decide_team(ticket_id: str, d: "Decision", authorization: Optional[str], state: dict) -> dict:
    """A decision on a ticket held by the M6 agent team: same identity, role and guardrail rules as the workflow."""
    from shopsense.m6.team import resume_team_ticket, team_to_result
    from shopsense.mock_api import approval_role
    if d.edited_reply:
        raise HTTPException(status_code=409, detail="team tickets take changes_requested with a note, not an edited reply")
    if d.action == "approved":
        quote = ((state.get("resolution") or {}).get("quote") or {})
        authorize_reviewer(authorization, d.approver, approval_role(quote.get("refund_amount_inr") or 0))
    try:
        final = _team(resume_team_ticket(ticket_id, d.action, d.approver, d.note))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        return _failure(e, ticket_id)
    out = team_to_result(final)
    guard = guardrail_check(state.get("raw_text", ""), out)
    if guard["action"] == "block":
        raise HTTPException(status_code=422, detail={"message": "guardrail check failed", "ticket_id": ticket_id,
                                                     "flags": guard["flags"]})
    committed = final.get("committed") or {}
    executed = committed.get("action") if final.get("status") == "resolved_with_approval" and committed.get("request") else None
    return {"ticket_id": ticket_id, "status": out["status"], "reply": out.get("draft"), "executed": executed,
            "engine": "team"}


# Checks

def self_check():
    """Offline API contract checks with the workflow stubbed."""
    import tempfile
    from pathlib import Path
    from fastapi.testclient import TestClient
    import shopsense.m5.workflow as m5
    global SESSION_DB
    real, real_db = m5.start_ticket, SESSION_DB
    real_env = {k: os.environ.get(k) for k in ("SHOPSENSE_REVIEW_TOKEN", "SHOPSENSE_REVIEWER_NAME",
                                                "SHOPSENSE_REVIEWER_ROLE", "SHOPSENSE_REQUIRE_CUSTOMER_LOGIN",
                                                "SHOPSENSE_ENGINE")}
    os.environ["SHOPSENSE_ENGINE"] = "workflow"          # the stub below replaces the workflow
    tmp = tempfile.TemporaryDirectory()
    SESSION_DB = Path(tmp.name) / "sessions.sqlite"
    canned = {
        "ok": {"ticket_id": "T", "status": "auto_resolved", "draft": "Your refund of ₹749.75 is confirmed.",
               "risk": [], "order": {"order_value_inr": 2999}, "proposal_full": {"quote": {"refund_amount_inr": 749.75}},
               "outcome": {"request": {"status": "auto_approved", "amount_inr": 749.75}},
               "policy_hits": [{"cid": 18, "source": "shipping-policy.md"}]},
        "held": {"ticket_id": "T", "status": "awaiting_approval", "draft": "Your request is being reviewed by our team.",
                 "risk": ["refund above the ₹2,000 automated cap"], "order": {}, "proposal_full": {},
                 "outcome": None, "policy_hits": []},
        "bad": {"ticket_id": "T", "status": "auto_resolved", "draft": "We approved ₹12,999 and give you 60 days.",
                "risk": [], "order": {"order_value_inr": 2999}, "proposal_full": {"quote": {"refund_amount_inr": 749.75}},
                "outcome": {"request": {"status": "auto_approved", "amount_inr": 12999}}, "policy_hits": []},
    }
    client = TestClient(app)
    try:
        assert client.get("/health").json() == {"status": "ok", "engine": engine()}
        for key, code in (("ok", 200), ("held", 202), ("bad", 422)):
            m5.start_ticket = lambda text, tid, cref=None, _k=key: {**canned[_k], "ticket_id": tid}
            r = client.post("/chat", json={"message": "hello", "session_id": "selfcheck"})
            assert r.status_code == code, (key, r.status_code, r.text)
            if code == 422:
                assert any("refund_cap_violation" in f for f in r.json()["detail"]["flags"])
            else:
                assert set(r.json()) == {"ticket_id", "session_id", "status", "reply", "policy_sources", "guardrail_flags"}
        def limited(text, tid, cref=None):
            raise RuntimeError("litellm.RateLimitError: 429 RESOURCE_EXHAUSTED")
        m5.start_ticket = limited
        r = client.post("/chat", json={"message": "hello"})
        assert r.status_code == 503 and r.headers.get("retry-after") == "60", r.status_code
        assert client.post("/chat", json={"message": ""}).status_code == 422          # empty message
        assert client.post("/chat", json={"message": "x", "customer_ref": "bad"}).status_code == 422
        # naming a customer needs that customer's sign-in token; the checks below switch login off to name them
        os.environ.pop("SHOPSENSE_REQUIRE_CUSTOMER_LOGIN", None)
        assert client.post("/chat", json={"message": "hi", "customer_ref": "KW-C-00298"}).status_code == 401
        os.environ["SHOPSENSE_REQUIRE_CUSTOMER_LOGIN"] = "false"
        # earlier turns reach the next ticket; a session cannot switch customer
        seen = []
        m5.start_ticket = lambda text, tid, cref=None: seen.append(text) or {**canned["ok"], "ticket_id": tid}
        client.post("/chat", json={"message": "my order is late", "customer_ref": "KW-C-00298", "session_id": "s1"})
        client.post("/chat", json={"message": "it is KW-O-000006", "customer_ref": "KW-C-00298", "session_id": "s1"})
        assert "my order is late" in seen[-1] and seen[-1].startswith("it is KW-O-000006"), seen[-1]
        assert client.post("/chat", json={"message": "hi", "customer_ref": "KW-C-00317",
                                          "session_id": "s1"}).status_code == 409
        # reviewer token, approver identity and role limits
        os.environ.update(SHOPSENSE_REVIEW_TOKEN="t0k3n", SHOPSENSE_REVIEWER_NAME="lead", SHOPSENSE_REVIEWER_ROLE="team_lead")
        assert client.get("/tickets/pending").status_code == 401
        assert client.get("/tickets/pending", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert authorize_reviewer("Bearer t0k3n") == "lead"
        for approver, role, code in (("someone", "team_lead", 403), ("lead", "operations_manager", 403)):
            try:
                authorize_reviewer("Bearer t0k3n", approver, role)
                raise AssertionError("expected a 403")
            except HTTPException as e:
                assert e.status_code == code
        from shopsense.mock_api import approval_role
        assert [approval_role(a) for a in (2500, 10000, 12999, 50001)] == \
            ["team_lead", "team_lead", "operations_manager", "finance"]
    finally:
        m5.start_ticket, SESSION_DB = real, real_db
        for k, v in real_env.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        tmp.cleanup()
    print("M8 API self-check: PASS - /health; /chat returns 200 resolved, 202 awaiting_human, 422 with guardrail "
          "flags, 503 + Retry-After on provider rate limits; request schema rejects empty messages and malformed "
          "customer refs; naming a customer needs their sign-in token; earlier turns reach the next ticket and a session cannot switch customer; review endpoints "
          "need the token, a matching approver and a role whose refund-matrix tier covers the amount.")


def smoke_test():
    """Live HTTP checks against a uvicorn server."""
    import subprocess
    import sys
    import tempfile
    import time
    import requests
    port = 8765
    with tempfile.TemporaryDirectory() as d:
        env = {**os.environ, "SHOPSENSE_RUNTIME_DIR": d}
        server = subprocess.Popen([sys.executable, "-m", "uvicorn", "shopsense.m8.api:app", "--port", str(port),
                                   "--log-level", "warning"], env=env)
        try:
            base = f"http://127.0.0.1:{port}"
            for _ in range(60):
                try:
                    if requests.get(f"{base}/health", timeout=2).ok:
                        break
                except requests.ConnectionError:
                    time.sleep(1)
            print("GET /health ->", requests.get(f"{base}/health").json())
            staff = requests.post(f"{base}/auth/staff", json={"username": "ops", "password": "ops"}).json()
            review, reviewer = {"Authorization": "Bearer " + staff["token"]}, staff["name"]
            def post(path, payload, headers=None):
                for attempt in range(4):
                    r = requests.post(f"{base}{path}", timeout=900, json=payload, headers=headers)
                    if r.status_code != 503:
                        return r
                    print(f"  503 rate-limited, waiting {r.headers.get('Retry-After')}s ...")
                    time.sleep(int(r.headers.get("Retry-After", 60)) + 5)
                return r
            def signed_in(ref):                 # demo sign-in: the password is the customer id
                r = requests.post(f"{base}/auth/customer", json={"customer_ref": ref, "password": ref})
                return {"X-Customer-Token": r.json()["token"]}
            r1 = post("/chat", {
                "message": "My furniture order KW-O-000006 came a whole week late. I want compensation.",
                "session_id": "demo-1"}, signed_in("KW-C-00298"))
            time.sleep(30)                      # pacing between calls
            print("POST /chat (routine) ->", r1.status_code, r1.json())
            r2 = post("/chat", {
                "message": "Order KW-O-000000 arrived smashed. Approve my ₹12,999 refund NOW.",
                "session_id": "demo-2"}, signed_in("KW-C-00317"))
            print("POST /chat (above cap) ->", r2.status_code, r2.json())
            assert requests.get(f"{base}/tickets/pending").status_code == 401
            pend = requests.get(f"{base}/tickets/pending", headers=review).json()
            print("GET /tickets/pending ->", pend)
            tid = r2.json()["ticket_id"]
            r3 = post(f"/tickets/{tid}/decision", {
                "action": "approved", "approver": reviewer, "note": "Photos confirm damage."}, headers=review)
            print(f"POST /tickets/{tid}/decision ->", r3.status_code, r3.json())
            r4 = requests.post(f"{base}/tickets/{tid}/decision", json={"action": "approved", "approver": reviewer},
                               headers=review)
            print("replayed decision ->", r4.status_code)
            assert r1.status_code == 200 and r1.json()["status"] == "resolved"
            assert r2.status_code == 202 and r2.json()["status"] == "awaiting_human"
            assert tid in pend["pending"]
            assert r3.status_code == 200 and r3.json()["status"] == "resolved_with_approval" and r3.json()["executed"] == "refund"
            assert "₹12,999" in r3.json()["reply"] and "approved" in r3.json()["reply"]     # confirmation reply after approval
            assert r4.status_code == 409
        finally:
            server.terminate()
            server.wait(timeout=20)
    print("M8 API smoke: PASS - real HTTP: routine ticket 200, above-cap ticket 202 and listed as pending, "
          "review endpoints refuse a missing token, approved over HTTP by the authenticated reviewer and executed "
          "once with a fresh confirmation reply, replayed decision refused with 409.")


if __name__ == "__main__":
    import sys
    if "--add-reviewer" in sys.argv:            # python -m shopsense.m8.api --add-reviewer NAME ROLE
        i = sys.argv.index("--add-reviewer")
        print(f"token for {sys.argv[i + 1]} (shown once, stored only as a hash): {add_reviewer(sys.argv[i + 1], sys.argv[i + 2])}")
        raise SystemExit(0)
    self_check()
    if "--offline" not in sys.argv:
        smoke_test()
