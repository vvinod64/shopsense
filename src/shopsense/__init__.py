def main() -> None:
    """Start ShopSense: `uv run shopsense`, then open http://localhost:8000/ui in a browser.

    Demo state goes to runtime/demo (set SHOPSENSE_RUNTIME_DIR to change it); PORT changes the port.
    """
    import os
    from pathlib import Path
    port = int(os.getenv("PORT", "8000"))
    os.environ.setdefault("SHOPSENSE_RUNTIME_DIR", str(Path(__file__).resolve().parents[2] / "runtime" / "demo"))
    import uvicorn
    print(f"""
  ShopSense is starting. Open http://localhost:{port}/ui

  Customer:       customer ID and password (demo: the password is the customer ID),
                  or leave both empty to continue as a guest
  Support staff:  lead (approves refunds up to Rs 10,000), ops (up to Rs 50,000) or finance;
                  the password is the username

  Stop with Ctrl+C.
""", flush=True)
    uvicorn.run("shopsense.m8.api:app", host="127.0.0.1", port=port, log_level="warning")
