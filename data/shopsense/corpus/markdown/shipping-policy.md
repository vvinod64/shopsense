# Shipping, Delivery and Lost-Parcel Policy

# KARTWAY OPERATIONS POLICY MANUAL
## SECTION 4: SHIPPING, TRANSIT DISRUPTIONS, AND FULFILMENT EXCEPTIONS
**Document ID:** POL-OPS-402  
**Effective Date:** 14 January 2025  
**Applies To:** All Tier-1, Tier-2, and Escalations Support Agents  

---

### 4.1 Delivery Service Level Agreements (SLAs)

**4.1.1** Kartway fulfillment operations classify delivery speeds into three distinct service tiers. The applicable Service Level Agreement (SLA) is determined by the geographic classification of the destination postal code, categorized strictly as either Metro or Non-Metro per the Kartway Geographic Routing Directory (GRD).

**4.1.2** Service Tier SLAs:
1. **Standard Delivery:**
   - Metro Destination: 3 business days from carrier pickup.
   - Non-Metro Destination: 6 business days from carrier pickup.
2. **Expedited Delivery:**
   - Metro Destination: 2 business days from carrier pickup.
   - Non-Metro Destination: 4 business days from carrier pickup.
3. **Same-Day Delivery:**
   - Metro Destination: End of local business day (20:00 local time) for orders placed by 11:00 local time.
   - Non-Metro Destination: Not available. Orders placed with Same-Day selected for non-metro addresses automatically default to Expedited delivery terms.

**4.1.3** Business days exclude Sundays and public holidays recognized in the destination jurisdiction. The day of carrier pickup is designated as Day 0.

---

### 4.2 Definition of Delayed Shipment and Compensation Ladder

**4.2.1** A shipment is formally classified as "Delayed" when tracking data confirms that the delivery has not occurred by 23:59 local time on the final day of the applicable SLA window specified in Section 4.1, and no delivery attempt has been logged within the preceding 24 hours.

**4.2.2** The standard compensation ladder for verified delayed shipments is strictly enforced as follows:
1. **Delay of 1 to 3 business days past SLA:**
   - Action: Issue a shipping fee refund to the original payment method within 3 business days.
   - Customer Goodwill: 150 Kartway Credits ($15.00 nominal value) added to the user account.
2. **Delay of 4 to 7 business days past SLA:**
   - Action: Issue a 25% refund of the total merchandise value (excluding taxes).
   - Customer Goodwill: 300 Kartway Credits ($30.00 nominal value) added to the user account.
3. **Delay of 8 or more business days past SLA:**
   - Action: Mandatory transition to Lost-in-Transit protocol under Section 4.3. Full refund of order value plus 500 Kartway Credits ($50.00 nominal value).

**4.2.3** Exceptions to the Compensation Ladder:
- Delays caused by force majeure events (as certified by Logistics Operations Director Marcus Vance) exempt Kartway from financial compensation, though goodwill credits of 100 Kartway Credits may be issued at agent discretion.
- Orders containing hazardous materials or oversized freight (exceeding 30kg) governed by special freight carriers operate on an extended delay threshold of 5 business days before Tier-1 compensation applies.

---

### 4.3 Lost-In-Transit Declaration Thresholds

**4.3.1** A shipment cannot be declared "Lost-in-Transit" until a mandatory waiting period has elapsed without a definitive tracking scan update:
- **Domestic Metro Shipments:** 7 consecutive calendar days without a tracking scan.
- **Domestic Non-Metro Shipments:** 10 consecutive calendar days without a tracking scan.
- **International / Cross-Border Shipments:** 21 consecutive calendar days without a tracking scan.

**4.3.2** Upon reaching the threshold specified in 4.3.1, the support agent must initiate an immediate tracer ticket with the designated courier partner. The courier is allotted exactly 48 hours to locate the parcel.

**4.3.3** If the courier fails to locate the parcel within the 48-hour tracer window, the agent must process a full refund to the customer's original payment method within 24 hours and issue a replacement order at zero cost if stock permits.

---

### 4.4 Courier-Marked "Delivered" But Customer Disputes Process

**4.4.1** When a tracking scan records a status of "Delivered" but the customer asserts non-receipt, agents must adhere strictly to the Non-Receipt Dispute (NRD) workflow. Agents are prohibited from issuing immediate refunds on the first contact for NRD cases where the order value exceeds $150.00.

**4.4.2** Mandatory verification steps for NRD cases:
1. **Geo-Location Audit:** Review the courier's delivery GPS pin. If the delivery pin is greater than 50 meters from the registered shipping address coordinates, the agent must treat the scan as invalid, issue an immediate replacement, and file an operational fault claim against the courier.
2. **Proof of Delivery (POD) Retrieval:** For orders over $50.00, request the signed receipt or photo proof from the carrier API.
3. **Customer Declaration:** The customer must complete and submit the Kartway Non-Receipt Statutory Declaration Form (Form NRD-99) within 7 calendar days of the disputed delivery timestamp.

**4.4.3** Resolution Pathways:
- **First Offense (Order Value $\le$ $200.00):** Agent may issue a one-time courtesy refund or replacement without waiting for Form NRD-99 completion.
- **Repeat Offenses (Customer has 2 or more prior NRD claims within a rolling 365-day window):** The account is automatically flagged for Fraud Investigation Unit (FIU) review led by Senior Investigator Elena Rostova. No refund or replacement shall be issued until the FIU concludes its investigation, which carries a SLA of 5 business days.

---

### 4.5 Address-Change Cutoffs and Interception Protocols

**4.5.1** Customers may request a delivery address modification or redirection via the Kartway Customer Support portal or self-service dashboard. 

**4.5.2** The absolute operational cutoffs for address changes are defined by fulfillment status:
- **Pre-Fulfillment (Status: Order Processing / Unfulfilled):** Address changes are permitted instantly via the customer dashboard up until the warehouse prints the pick list, which occurs on a rolling schedule every 2 hours.
- **Post-Fulfillment (Status: Picked, Packed, or Handed to Carrier):** Address changes are only permitted if the tracking status has not yet reached "Out for Delivery" or "Linehaul Departure." 

**4.5.3** Processing Interceptions:
- If the parcel is in transit with a participating integrated carrier (e.g., SwiftPost, Apex Logistics), the agent may trigger an automated carrier divert command through the internal Logistics Control Panel.
- A flat administrative interception fee of $5.00 is charged to the customer unless the original address error was caused by a verified Kartway system glitch.
- If the parcel has reached the final-mile local depot and bears an "Out for Delivery" status, address interception is strictly impossible. The agent must advise the customer that the package will be returned to sender after 3 failed delivery attempts, at which point a refund will be processed minus the original shipping fee.
