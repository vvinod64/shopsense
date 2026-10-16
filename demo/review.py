"""Support-staff console: escalated tickets appear as they happen; approve, reject or send back.

Start ShopSense first (uv run shopsense), then in its own terminal:
    uv run python demo/review.py
It asks for a staff username and password (demo: lead, ops or finance; the password is the username).

Commands:
    list                 show every ticket waiting for a decision (also shown whenever a new one arrives)
    a N [note]           approve ticket number N
    r N [note]           reject ticket N
    c N note             send ticket N back with a note (the reply is revised and comes back here)
    quit
"""
import getpass
import sys
import threading
import time

import httpx

B, D, G, Y, R, M, X = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[35m", "\033[0m"
HEADERS = {}


def sign_in(client: httpx.Client) -> dict:
    """Ask for a staff username and password until the API accepts them."""
    while True:
        username = input("Staff username: ").strip()
        r = client.post("/auth/staff", json={"username": username, "password": getpass.getpass("Password: ")})
        if r.status_code == 200:
            return r.json()
        print(f"  {R}unknown username or wrong password{X}")


def fetch(client: httpx.Client) -> list:
    r = client.get("/tickets/pending", headers=HEADERS)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    return r.json().get("details") or []


def show(items: list, numbers: dict) -> None:
    if not items:
        print(f"  {D}no tickets waiting{X}")
        return
    for d in items:
        n = numbers[d["ticket_id"]]
        amount = f"₹{d['amount_inr']:,.2f}".replace(".00", "") if d.get("amount_inr") else "-"
        print(f"\n  {B}[{n}] {d['ticket_id']}{X}  customer {d.get('customer_ref') or 'guest'}  "
              f"{D}waiting {d.get('age_hours', 0)}h{X}{f'  {R}OVERDUE{X}' if d.get('overdue') else ''}")
        print(f"      {B}message{X}   {(d.get('message') or '')[:220]}")
        print(f"      {B}why held{X}  {Y}{'; '.join(d.get('risk') or ['-'])}{X}")
        print(f"      {B}proposed{X}  {d.get('proposed_action') or 'no action'}  {amount}")
        print(f"      {B}draft{X}     {(d.get('draft') or '')[:300]}")


def main() -> None:
    args = sys.argv[1:]
    url = args[args.index("--url") + 1] if "--url" in args else "http://localhost:8000"
    client = httpx.Client(base_url=url, timeout=900)
    try:
        client.get("/health")
    except httpx.HTTPError:
        sys.exit(f"ShopSense is not running at {url}. Start it first: uv run shopsense")
    me = sign_in(client)
    HEADERS["Authorization"] = f"Bearer {me['token']}"
    reviewer, prompt = me["name"], f"{M}{B}{me['name']} ({me['role'].replace('_', ' ')})>{X} "
    try:
        items = fetch(client)
    except (httpx.HTTPError, RuntimeError) as e:
        sys.exit(f"Cannot reach the reviewer endpoints at {url}: {e}")
    numbers, seen, lock = {}, set(), threading.Lock()

    def number(items_):
        for d in items_:
            numbers.setdefault(d["ticket_id"], len(numbers) + 1)

    number(items)
    seen.update(d["ticket_id"] for d in items)
    print(f"{B}ShopSense support console{X}  {D}{url}  (new escalations appear on their own; 'list', 'a N', 'r N', 'c N note'){X}")
    show(items, numbers)

    def poll():
        while True:
            time.sleep(5)
            try:
                current = fetch(client)
            except (httpx.HTTPError, RuntimeError):
                continue
            with lock:
                fresh = [d for d in current if d["ticket_id"] not in seen]
                if fresh:
                    number(fresh)
                    seen.update(d["ticket_id"] for d in fresh)
                    print(f"\n\n{Y}{B}New escalation{'s' if len(fresh) > 1 else ''}{X}", end="")
                    show(fresh, numbers)
                    print(f"\n{prompt}", end="", flush=True)

    threading.Thread(target=poll, daemon=True).start()
    by_number = lambda n: next((t for t, k in numbers.items() if k == n), None)   # noqa: E731
    while True:
        try:
            line = input(f"\n{prompt}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        if cmd in ("quit", "exit", "q"):
            break
        if cmd == "list":
            with lock:
                items = fetch(client)
                number(items)
                show(items, numbers)
            continue
        if cmd not in ("a", "r", "c"):
            print("  commands: list | a N [note] | r N [note] | c N note | quit")
            continue
        num, _, note = rest.partition(" ")
        ticket = by_number(int(num)) if num.isdigit() else None
        if not ticket:
            print("  give the ticket number shown in brackets, e.g. a 1")
            continue
        if cmd == "c" and not note.strip():
            print("  say what to change, e.g. c 1 mention the photo upload")
            continue
        action = {"a": "approved", "r": "rejected", "c": "changes_requested"}[cmd]
        print(f"  {D}sending decision...{X}")
        t0 = time.time()
        r = client.post(f"/tickets/{ticket}/decision", headers=HEADERS,
                        json={"action": action, "approver": reviewer, "note": note.strip()})
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"detail": r.text}
        colour = G if r.status_code == 200 else R
        print(f"  {colour}{B}HTTP {r.status_code}{X} {D}({time.time() - t0:.0f}s){X}")
        if r.status_code == 200:
            print(f"  {B}status{X}    {body.get('status')}   executed: {body.get('executed')}")
            print(f"  {B}reply{X}     {body.get('reply')}")
            if action == "changes_requested":       # the revised ticket comes back to the queue
                seen.discard(ticket)
        else:
            print(f"  {body.get('detail', body)}")


if __name__ == "__main__":
    main()
