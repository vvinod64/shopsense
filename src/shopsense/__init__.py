def main() -> None:
    """Start ShopSense: `uv run shopsense`, then open http://localhost:8000/ui in a browser.

    Starts the MCP order server once (MCP_PORT, default 8001) and the API (PORT, default 8000); the agent team
    reaches the order system through that one server. Demo state goes to runtime/demo (set
    SHOPSENSE_RUNTIME_DIR to change it).
    """
    import os
    import socket
    import subprocess
    import sys
    import time
    from pathlib import Path
    port, mcp_port = int(os.getenv("PORT", "8000")), int(os.getenv("MCP_PORT", "8001"))
    os.environ.setdefault("SHOPSENSE_RUNTIME_DIR", str(Path(__file__).resolve().parents[2] / "runtime" / "demo"))

    server = None
    if not os.getenv("SHOPSENSE_MCP_URL"):          # an MCP server someone already runs is used as it is
        server = subprocess.Popen([sys.executable, "-m", "shopsense.m6.mcp_server", "--http", "--port", str(mcp_port)])
        for _ in range(100):                        # ready when it accepts connections (about 2 s)
            if server.poll() is not None:
                sys.exit(f"The MCP order server did not start (is port {mcp_port} in use? set MCP_PORT).")
            try:
                socket.create_connection(("127.0.0.1", mcp_port), timeout=0.5).close()
                break
            except OSError:
                time.sleep(0.2)
        os.environ["SHOPSENSE_MCP_URL"] = f"http://127.0.0.1:{mcp_port}/mcp"

    import uvicorn
    print(f"""
  ShopSense is starting. Open http://localhost:{port}/ui
  Order system (MCP server): {os.environ["SHOPSENSE_MCP_URL"]}

  Customer:       customer ID and password (demo: the password is the customer ID),
                  or leave both empty to continue as a guest
  Support staff:  lead (approves refunds up to Rs 10,000), ops (up to Rs 50,000) or finance;
                  the password is the username

  Stop with Ctrl+C.
""", flush=True)
    try:
        uvicorn.run("shopsense.m8.api:app", host="127.0.0.1", port=port, log_level="warning")
    finally:
        if server is not None:
            server.terminate()
            server.wait(timeout=10)
