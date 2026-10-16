# ShopSense demo video (under 10 minutes)

Milestone order, matching the deck. Each segment: show the slide for a few seconds, then the live part. The
slide's speaker notes are the narration. Free-tier tickets take 15-60 s in the workflow and 1-2 minutes in the agent team that serves the browser: record each segment separately and
cut the waiting in editing.

## Before recording (10 minutes)

1. Lid open, charger in, notifications off, terminal font large (about 18 pt, 110 columns).
2. Clean state and checks:
   ```bash
   .venv/bin/python demo/preflight.py --reset
   ```
   Every line must say OK. If Gemini says 503 or quota: `export SHOPSENSE_LLM_MODEL=gemini/gemini-3.1-flash-lite`
   and run preflight again.
3. Start ShopSense in its own terminal (it prints the address and the demo logins):
   ```bash
   uv run shopsense
   ```
4. Postman: import `demo/ShopSense.postman_collection.json`. Nothing to fill in: its sign-in requests store the
   tokens themselves.
5. Open in browser tabs: the deck, Langfuse (Traces, plus Datasets/Experiments for `shopsense-m8-golden-eval`),
   Supermemory dashboard, Qdrant Cloud collection `shopsense_policies`.

## Timeline

| Time | Slide | Live part | What to point at |
|---|---|---|---|
| 0:00-0:40 | 1 Cover, 2 Architecture | (slides only) | Code decides the action and amount; the model writes the words |
| 0:40-1:30 | 3 M1 | `.venv/bin/python demo/demo.py m1` | The schema error sent back, the repaired ticket; the Hinglish message parsed |
| 1:30-2:30 | 4 M2 | `.venv/bin/python demo/demo.py m2` | Tools chosen by the model; ₹749.75 `auto_approved`; ₹12,999 `pending_human_approval` |
| 2:30-3:20 | 5 M3 | `.venv/bin/python demo/demo.py m3` (+ Supermemory tab) | Rolling summary; "Hindi" saved and recalled; 3rd contact adds a risk |
| 3:20-4:00 | 6 M4 | `.venv/bin/python demo/demo.py m4` | Electronics 15-day clause ranked first in Markdown and PDF; the metrics table |
| 4:00-4:40 | 7 M5 | `uv run python demo/demo.py m5` | Routine resolved; ₹6,499 held, nothing executed; approved by a separate process; replay refused |
| 4:40-7:30 | 8 M6 | `uv run python demo/demo.py m6`, then the browser (customers and staff, see below) | 8 MCP tools, 2 resources; the invented ₹500 / 60 days caught by the fact-checker |
| 6:40-7:40 | 9 M7 | `.venv/bin/python demo/demo.py m7` + Langfuse trace | 57% -> 89.5% -> 100%; the down-system ticket; its trace tree in Langfuse |
| 7:40-9:00 | 10 M8 | `.venv/bin/python demo/demo.py m8` + Langfuse experiment | Guardrails allow / block / hold; per-component table, LLM judges; team results; the toolkit's own golden set (routing 18/20) |
| 9:00-9:30 | 10 M8 | (slide) | Honest limits: pattern-based injection detection, free-tier latency |

### Live in the browser: one window per role (you type the messages)

The browser is served by the M6 multi-agent team (the default engine), so show it during the M6 segment; M5 is shown with `demo.py m5`.

Start ShopSense (`uv run shopsense`). Every window opens the same sign-in page, http://localhost:8000/ui, which has
two choices:
- **Customer**: type a customer ID and password (in this demo the password is the customer ID, for example
  KW-C-00298 / KW-C-00298). Leave both empty to continue as a guest. A wrong password is refused.
- **Support staff**: username and password, the same word: `lead` (approves refunds up to ₹10,000), `ops` (up to
  ₹50,000) or `finance` (above). Use `ops` for the ₹12,999 refund; `lead` is refused, which shows the refund matrix.

Use one browser window per person, side by side: for example customer KW-C-00298, customer KW-C-00317 (or a
guest), and support staff. Use separate windows (or a private window), not tabs of one window, so each keeps its own
sign-in.

Type a message in a customer window. A routine one is answered there (Resolved). A risky one shows "With our
support team" and appears on its own in the support window as a card: customer, message, why it needs a person,
proposed action and amount, draft reply, with Approve / Send back / Reject. After the decision the outcome
appears on its own in the customer's window ("Update: approved").

### The same in terminals (alternative)

Terminal 1: `uv run shopsense`. Terminals 2 and 3, one per customer: `uv run python demo/chat.py` (it asks for the
customer ID and password; Enter for a guest; `/examples` lists messages worth trying). Terminal 4, support staff:
`uv run python demo/review.py` (it asks for the staff username and password).

A routine message resolves in the customer's window (200). A risky one is held (202): it appears on its own in the
support console with the customer, message, reasons, proposed action, amount and draft; `a 1` approves it (`r 1`
rejects, `c 1 <note>` sends it back), and the outcome appears on its own in that customer's window. Restart the API
while a ticket is held to show it survives.

### M5 in Postman (the strongest moment)

1. 02 and 03 sign in the two customers: 200; the collection stores each session token.
2. 04 names a customer without signing in: 401. Say: nobody can read another customer's orders by typing their ID.
3. 05 routine late delivery: 200, `resolved`, refund ₹749.75.
4. 06 follow-up in the same session: it remembers the order; replies in Hindi.
5. 07 angry ₹12,999: 202, `awaiting_human`, "being reviewed by an Operations Manager". Nothing executed.
6. 08 injection and 09 someone else's order: both 202, held.
7. **Restart the server** (Ctrl+C in the uvicorn terminal, start it again). Say: the ticket waits in a SQLite checkpoint.
8. 10 pending without a token: 401. 11 signs in as staff `ops`; 12 with that token: the held tickets are listed.
9. 13 approve: 200, `resolved_with_approval`, executed `refund`, a fresh confirmation reply.
10. 14 replay the approval: 409, nothing executed twice.

## If something goes wrong while recording

- 503 from `/chat`: the model is busy; wait for `Retry-After` and send again (cut it in editing).
- A segment fails on quota: switch `SHOPSENSE_LLM_MODEL`, re-run `preflight.py`, re-record that segment.
- Results already recorded, if a live run is impossible: M8 shows the saved evaluation reports, and Langfuse
  holds every run (`run-104032` judged workflow, `run-141913` latest workflow and the toolkit's golden set, `team-run-102134`
  full team run, `team-run-133630` team retrieval after its fix).
