"""MCP server for the Kartway order system (orders, refunds, shipping) over stdio.

Approval rules and the refund cap are enforced in mock_api, so no client can approve above the cap.
Logs go to stderr because stdout carries the JSON-RPC stream.

Run standalone:  python -m shopsense.m6.mcp_server
Tool eval:       python -m shopsense.m6.mcp_server --eval   (labelled calls, see m6/tool_eval.py)
"""
import sys

from fastmcp import FastMCP

from shopsense import mock_api
from shopsense.config import CORPUS_DIR
from shopsense.m1.parser import REFUND_AUTO_CAP_INR

mcp = FastMCP("kartway-orders")


@mcp.tool()
def lookup_order(order_ref: str) -> dict:
    """Look up an order.

    Args:
        order_ref: order reference in the form KW-O-NNNNNN.
    Returns:
        Order status, value in INR, customer_ref, product title and category, delivery promise and
        actual delivery days. Raises an error if the order does not exist.
    """
    return mock_api.get_order(order_ref)


@mcp.tool()
def get_customer_account(customer_ref: str) -> dict:
    """Look up a customer's account: tier, lifetime orders, return rate and whether the account needs
    manual review. Account status is INTERNAL and must never be disclosed to the customer.

    Args:
        customer_ref: customer reference in the form KW-C-NNNNN.
    """
    return mock_api.get_customer(customer_ref)


@mcp.tool()
def track_shipment(order_ref: str) -> dict:
    """Current shipment status for an order: carrier, last scan status, scan time and location.

    Args:
        order_ref: order reference in the form KW-O-NNNNNN.
    """
    return mock_api.get_shipment(order_ref)


@mcp.tool()
def check_delivery_sla(order_ref: str) -> dict:
    """Promised vs actual delivery days for an order and how many days late it was (null if not delivered).

    Args:
        order_ref: order reference in the form KW-O-NNNNNN.
    """
    return mock_api.delivery_sla(order_ref)


@mcp.tool()
def calculate_refund(order_ref: str, reason_code: str, item_condition: str = "opened") -> dict:
    """Quote the refund the Kartway policy allows. Read-only: creates nothing.

    Args:
        order_ref: order reference in the form KW-O-NNNNNN.
        reason_code: one of not_delivered, late_delivery, damaged, quality, wrong_item, changed_mind.
        item_condition: 'opened' or 'unopened' (affects category restocking fees).
    Returns:
        refund_amount_inr, refund_pct, whether it is within the automated cap, and the policy basis.
    """
    return mock_api.quote_refund(order_ref, reason_code, item_condition)


@mcp.tool()
def request_refund(order_ref: str, reason_code: str, ticket_id: str, item_condition: str = "opened") -> dict:
    """Create a refund request. The server computes the amount and decides approval: auto_approved only
    within the automated cap for an account in good standing with no prior refund; otherwise
    pending_human_approval. Idempotent per (ticket_id, order_ref).

    Args:
        order_ref: order reference in the form KW-O-NNNNNN.
        reason_code: one of not_delivered, late_delivery, damaged, quality, wrong_item, changed_mind.
        ticket_id: the support ticket this refund belongs to.
        item_condition: 'opened' or 'unopened'.
    """
    return mock_api.create_refund_request(order_ref, reason_code, ticket_id, item_condition)


@mcp.tool()
def request_replacement(order_ref: str, reason_code: str, ticket_id: str) -> dict:
    """Create a replacement request for a damaged, quality-defective or wrong item. Idempotent per ticket.

    Args:
        order_ref: order reference in the form KW-O-NNNNNN.
        reason_code: one of damaged, quality, wrong_item.
        ticket_id: the support ticket this replacement belongs to.
    """
    return mock_api.create_replacement_request(order_ref, reason_code, ticket_id)


@mcp.tool()
def escalate_ticket(ticket_id: str, reason: str, suggested_action: str = "") -> dict:
    """Hand a ticket to the human support queue.

    Args:
        ticket_id: the support ticket to escalate.
        reason: why a human is needed (e.g. refund above cap, legal threat, product safety).
        suggested_action: optional recommendation for the human agent.
    """
    return mock_api.record_escalation(ticket_id, reason, suggested_action, source="mcp")


# No delete tool, no tool that sets an amount, and no tool that approves a pending request:
# approval is a human decision recorded by the workflow.


@mcp.resource("policy://refund-authority")
def refund_authority() -> str:
    """The refund authorisation matrix (who may approve how much). Application-loaded context."""
    return (CORPUS_DIR / "markdown" / "refund-authority.md").read_text(encoding="utf-8")


@mcp.resource("policy://automation-limits")
def automation_limits() -> str:
    """Hard limits the automated system operates under - loaded by the app, never chosen by a model."""
    return (f"AUTOMATION LIMITS\n"
            f"1. Refunds above ₹{REFUND_AUTO_CAP_INR:,} are never auto-approved (Refund matrix §4.2.1).\n"
            f"2. Account review status is never disclosed to the customer (Fraud standard §4.4).\n"
            f"3. Legal threats, safety reports and third contacts within 14 days go to a human "
            f"(Escalation standard §2).\n")


if __name__ == "__main__" and "--eval" in sys.argv:
    from shopsense.m6.tool_eval import evaluate_tools, print_report
    print_report(evaluate_tools())
elif __name__ == "__main__" and "--http" in sys.argv:
    # One long-running server for the API (uv run shopsense starts it): clients connect over HTTP instead of
    # each ticket starting its own server process. Bound to this machine only; the tools have no login.
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8001
    print(f"[kartway-orders MCP] serving http://127.0.0.1:{port}/mcp", file=sys.stderr)
    mcp.run(transport="http", host="127.0.0.1", port=port, show_banner=False)
elif __name__ == "__main__":
    print("[kartway-orders MCP] starting on stdio", file=sys.stderr)
    mcp.run(transport="stdio", show_banner=False)
