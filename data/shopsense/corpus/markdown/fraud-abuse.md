# Return Fraud and Abuse Prevention Standard

# KARTWAY OPERATIONS POLICY MANUAL
## SECTION 4: RETURN ABUSE, FRAUD PREVENTION, AND ACCOUNT MONITORING
**Document ID:** POL-RET-402  
**Effective Date:** 14 January 2025  
**Applies To:** Customer Support Tier 1, Tier 2, Trust & Safety Specialists  

---

### 4.1 Purpose and Scope
This policy establishes the operational thresholds, investigative protocols, communication restrictions, and escalation pathways governing customer return behavior on the Kartway platform. All support personnel are strictly bound by the procedures detailed herein.

---

### 4.2 Account Review Thresholds
1. **Return-Rate Flag Trigger:** An account is automatically flagged for review by the automated risk engine when it meets or exceeds either of the following criteria within a rolling 90-day window:
   * A return frequency exceeding **25%** of total completed order volume, provided the total order count is **5 or more**.
   * A net monetary refund value exceeding **$1,500.00 USD**, regardless of the overall percentage.

2. **Automated Restriction:** Upon reaching the threshold defined in Clause 4.2.1, the account status changes internally to `REV-PENDING`. Standard automated label generation is suspended for this account pending manual review by Tier 2 support.

---

### 4.3 Serial Refund Requesters
1. **Definition:** A customer is classified as a "Serial Refund Requester" if they submit **3 or more** distinct refund claims without a corresponding physical item return ("Refund Without Return"), or if they report "Item Not Received" (INR) on more than **15%** of delivered packages within a 180-day period.

2. **Actionable Restrictions:**
   * Support agents are prohibited from issuing instant refunds via peer-to-peer or manual ledger overrides for accounts carrying the Serial Refund tag.
   * All subsequent return or refund requests from a Serial Refund Requester must be processed under **Mandatory Inspection Protocol (MIP)**, requiring the returned merchandise to be received, scanned, and inspected at the Kartway Returns Processing Center in Reno, NV, before any credit is authorized.

---

### 4.4 Prohibition of Flag Disclosure
1. **Absolute Confidentiality:** Under no circumstances shall a customer support agent disclose to a customer, directly or indirectly, that their account is "flagged," "under review," "monitored," or subject to "fraud prevention protocols."

2. **Permissible Phrasing:** When a customer inquires about delayed return processing, restricted label generation, or denied refund requests, agents must use strictly neutral operational language. Permissible phrasing is limited to:
   * *"Your return is currently undergoing standard administrative processing."*
   * *"Our system requires additional verification for this specific transaction before we can issue a replacement or refund."*

3. **Violation Consequence:** Disclosing account flag status or internal risk scores constitutes a Tier 3 disciplinary violation under Kartway employment policy.

---

### 4.5 Fraud Escalation Route
1. **Suspicion Triggers:** Agents must immediately halt standard support workflows and escalate to the Trust & Safety (T&S) department if any of the following indicators are observed:
   * Photographic evidence of returned goods shows clear signs of digital alteration, stock imagery, or substitution with non-Kartway items.
   * The customer uses a known commercial freight forwarder combined with a disposable payment instrument and a high-value electronics order.
   * Direct threats of chargeback abuse or extortion directed at the support agent.

2. **Escalation SLA and Pathway:**
   * **Step 1:** The agent must tag the support ticket with the internal macro `#FRD-ESC-IMMED`.
   * **Step 2:** Populate the mandatory fraud intake form in the internal CRM with tracking numbers, order IDs, and chat transcripts.
   * **Step 3:** Route the ticket to the **Trust & Safety Fraud Queue (Level 2)**.
   * **SLA:** T&S must acknowledge and lock the account within **2 business hours** of escalation. During the interim, the support agent must place the customer ticket on a 24-hour hold using canned response `HOLD-VERIF-04`.
