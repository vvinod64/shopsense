## Setup

```bash
uv sync
cp .env.example .env    # fill in Gemini, Supermemory, Qdrant Cloud and Langfuse keys
.venv/bin/python -m shopsense.m3.rag   # builds the Qdrant Cloud index (embeddings are disk-cached)
```

`data/` is read-only input. Everything the system writes goes to `runtime/` (override with `SHOPSENSE_RUNTIME_DIR`).

## Milestones → modules

| Milestone | Module(s) | Check (offline) | Live demo / evidence |
|---|---|---|---|
| M1 Structured intake + repair loop | `m1/parser.py` | `python -m shopsense.m1.parser --offline` | `python -m shopsense.m1.parser`, `… m1.parser --eval` (field accuracy → Langfuse) |
| M2 Tool-enabled agent | `m2/agent.py`, `mock_api`, `reliability` | `… m2.agent --offline` | `… m2.agent` |
| M3 Memory + semantic index | `m3/memory.py`, `m3/rag.py` | `… m3.memory --offline` | `… m3.memory`, `… m3.rag` |
| M4 Production RAG + eval | `m4/hybrid.py`, `m4/rerank.py`, `m4/eval.py` | `… m4.eval --offline` | `… m4.eval` (scores → Langfuse) |
| M5 LangGraph workflow + checkpoint + HITL | `m5/workflow.py` | `… m5.workflow --offline` | `… m5.workflow` (resume in a new process) |
| M6 Multi-agent team + MCP | `m6/team.py`, `m6/mcp_server.py` | `… m6.team --offline` | `… m6.team`, `… m8.eval --engine team` (golden set) |
| M7 Observability + hardening | `tracing`, `reliability`, `m7/observability.py` | `… m7.observability --offline` | `… m7.observability` (spans read back from Langfuse) |
| M8 Eval + guardrails + API | `guardrails`, `m8/eval.py`, `m8/api.py` | `… guardrails`, `… m8.api --offline` | `… m8.eval`, `… m8.api` |

(`…` = `.venv/bin/python -m shopsense.`)

## Run it

```bash
uv run shopsense
```

Then open http://localhost:8000/ui and sign in as a **customer** (customer ID and password; in this demo the password
is the customer ID, e.g. KW-C-00298 / KW-C-00298; leave both empty for a guest) or as **support staff** (`lead`,
`ops` or `finance`, password = username; they may approve refunds up to ₹10,000, ₹50,000 and above). Use one browser
window per person. The same API serves Postman (`demo/ShopSense.postman_collection.json`), Swagger at `/docs`, and
the terminal clients `demo/chat.py` and `demo/review.py`.

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | liveness, no LLM call |
| POST | `/auth/customer` | `{"customer_ref", "password"}` → `{"customer_ref", "token"}` (demo: the password is the customer id) · 401 unknown customer or wrong password |
| POST | `/chat` | `{"message", "session_id"?}` plus header `X-Customer-Token` for a signed-in customer (no token: a guest) → 200 resolved · 202 awaiting human · 422 guardrail block (flags in body) · 409 if the session belongs to another customer · 401 if `customer_ref` is sent without that customer's token · 503 + `Retry-After` when the LLM is rate-limited |
| GET | `/tickets/{ticket_id}/status` | the customer's view of one ticket (status and latest reply); a customer's ticket needs their `X-Customer-Token` |
| GET | `/reviewer/me` | reviewer only: the signed-in reviewer's name and role |
| POST | `/auth/staff` | `{"username", "password"}` → `{"name", "role", "token"}`; the token is the reviewer Bearer token |
| GET | `/ui`, `/ui/customer`, `/ui/support` | sign-in page (Customer or Support staff), customer chat, support console |
| GET | `/tickets/pending` | reviewer only: tickets parked at the human-approval gate |
| POST | `/tickets/{ticket_id}/decision` | reviewer only: `{"action": "approved" \| "rejected" \| "changes_requested", "approver", "note"?, "edited_reply"?}` · 401 no/wrong token · 403 approver mismatch or role below the amount's tier · 409 if not pending or the edited reply fails the reply checks |

Reviewer endpoints need `Authorization: Bearer <token>`: the token from `POST /auth/staff` (`{"username", "password"}`), a named reviewer's token, an SSO token, or `SHOPSENSE_REVIEW_TOKEN`. `approver` must be that reviewer, and an approval needs a role whose refund-matrix tier covers the amount (`team_lead` ≤ ₹10,000, `operations_manager` ≤ ₹50,000, `finance` above). Conversations (`session_id`) are stored in `runtime/sessions.sqlite`; earlier turns are passed to the next ticket as context.

Customers sign in first; the token says who the customer is, so nobody can read another customer's orders by
sending their id. `SHOPSENSE_REQUIRE_CUSTOMER_LOGIN=false` lets a trusted client send `customer_ref` without a token
(development only). Example:

```bash
TOKEN=$(curl -s localhost:8000/auth/customer -H 'Content-Type: application/json' \
  -d '{"customer_ref": "KW-C-00298", "password": "KW-C-00298"}' | python3 -c 'import sys, json; print(json.load(sys.stdin)["token"])')
curl -s localhost:8000/chat -H 'Content-Type: application/json' -H "X-Customer-Token: $TOKEN" \
  -d '{"message": "My furniture order KW-O-000006 came a week late. I want compensation."}'
```

Public demo: `ngrok http 8000` (requires an ngrok authtoken), then point Postman at `<public-url>/chat`.


## Demo

Web pages served by the API: `/ui` is the sign-in page with two choices, Customer (customer ID and password, the ID itself in this demo; leave both empty for a guest) and Support staff (username and password: lead, ops or finance in this demo). Customers get their chat (`/ui/customer`); staff get the console (`/ui/support`), where escalations appear and update on their own. In terminals, `demo/chat.py` (one window per customer; it asks for the customer ID and password) and `demo/review.py` (the support-staff console; it asks for the staff login) are interactive clients for the running API: a customer's held ticket appears in the console by itself, the reviewer decides it there, and the outcome appears in the customer's window by itself (`GET /tickets/{id}/status`). `demo/` also holds the video kit: `preflight.py` (checks keys, Gemini, Qdrant, Supermemory, Langfuse and the port; `--reset` archives earlier demo state), `demo.py m1` … `m8` (one short segment per milestone, in slide order), `ShopSense.postman_collection.json` (the API walkthrough, including the human-approval flow) and `DEMO_SCRIPT.md` (the minute-by-minute script). Demo state goes to `runtime/demo` (API) and `runtime/demo-cli` (demo.py), never `data/`.
