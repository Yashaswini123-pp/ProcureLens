"""
ProcureLens Celonis Process Query Language (PQL) Repository.

Contains production-grade PQL query definitions for enterprise Purchase-to-Pay (P2P) analysis:
1. Average purchase-to-pay throughput time
2. Number of purchase orders by process variant
3. Repeated approval activities (Approval Rework)
4. Missing invoice validation (Compliance Deviation)
5. Supplier-wise processing performance
6. Payment delays (Due Date vs Payment Execution)
7. Process rework rate
8. Process happy-path compliance rate

All queries follow official Celonis PQL syntax specifications using PU-functions (Pull-Up),
CALC_THROUGHPUT, VARIANT, PROCESS EQUALS, and FILTER constructs.
"""

from typing import Dict, List


def throughput_time_query() -> str:
    """
    Query 1: Average purchase-to-pay throughput time (Requisition to Payment).
    Calculates case duration in days between first requisition and final payment.
    """
    return """-- Query 1: Average Purchase-to-Pay Throughput Time (Days)
AVG(
    CALC_THROUGHPUT(
        FIRST_OCCURRENCE['Purchase Requisition Created'] TO LAST_OCCURRENCE['Payment Executed'],
        REMAP_TIMESTAMPS("Event_Log"."Timestamp", DAYS)
    )
)"""


def po_by_variant_query() -> str:
    """
    Query 2: Number of purchase orders grouped by process variant.
    Returns variant string, case count, and percentage share.
    """
    return """-- Query 2: Purchase Order Volume by Process Variant
SELECT
    VARIANT("Event_Log"."Activity") AS "Process_Variant",
    COUNT_TABLE("Purchase_Orders") AS "PO_Count",
    (COUNT_TABLE("Purchase_Orders") / GLOBAL(COUNT_TABLE("Purchase_Orders"))) * 100.0 AS "Variant_Percentage"
FROM "Purchase_Orders"
ORDER BY COUNT_TABLE("Purchase_Orders") DESC"""


def repeated_approvals_query() -> str:
    """
    Query 3: Repeated approval activities (Rework indicator).
    Finds purchase orders where 'Purchase Order Approved' occurred more than once.
    """
    return """-- Query 3: Repeated Approval Activities (Rework Cases)
SELECT
    "Purchase_Orders"."PO_ID",
    "Purchase_Orders"."Department",
    "Suppliers"."Supplier_Name",
    PU_COUNT("Purchase_Orders", "Event_Log"."Event_ID", "Event_Log"."Activity" = 'Purchase Order Approved') AS "Approval_Count"
FROM "Purchase_Orders"
WHERE PU_COUNT("Purchase_Orders", "Event_Log"."Event_ID", "Event_Log"."Activity" = 'Purchase Order Approved') > 1
ORDER BY "Approval_Count" DESC"""


def missing_validation_query() -> str:
    """
    Query 4: Missing invoice validation.
    Detects cases where 'Invoice Approved' occurred without prior 'Invoice Validated'.
    """
    return """-- Query 4: Invoices Approved Without Prior Validation (Segregation of Duties / Audit Risk)
SELECT
    "Purchase_Orders"."PO_ID",
    "Invoices"."Invoice_ID",
    "Invoices"."Invoice_Amount",
    "Suppliers"."Supplier_Name"
FROM "Purchase_Orders"
WHERE PROCESS NOT EQUALS 'Purchase Requisition Created' -> ... -> 'Invoice Validated' -> ... -> 'Invoice Approved'
  AND PU_COUNT("Purchase_Orders", "Event_Log"."Event_ID", "Event_Log"."Activity" = 'Invoice Approved') > 0
  AND PU_COUNT("Purchase_Orders", "Event_Log"."Event_ID", "Event_Log"."Activity" = 'Invoice Validated') = 0"""


def supplier_performance_query() -> str:
    """
    Query 5: Supplier-wise processing performance.
    Computes average throughput time and total procurement spend per vendor.
    """
    return """-- Query 5: Supplier-Wise Throughput Time & Procurement Volume
SELECT
    "Suppliers"."Supplier_ID",
    "Suppliers"."Supplier_Name",
    "Suppliers"."Supplier_Category",
    "Suppliers"."Supplier_Rating",
    COUNT_TABLE("Purchase_Orders") AS "Total_POs",
    SUM("Purchase_Orders"."Purchase_Amount") AS "Total_Spend",
    AVG(
        CALC_THROUGHPUT(
            FIRST_OCCURRENCE['Purchase Order Generated'] TO LAST_OCCURRENCE['Goods Received'],
            REMAP_TIMESTAMPS("Event_Log"."Timestamp", DAYS)
        )
    ) AS "Avg_Delivery_Days"
FROM "Suppliers"
GROUP BY
    "Suppliers"."Supplier_ID",
    "Suppliers"."Supplier_Name",
    "Suppliers"."Supplier_Category",
    "Suppliers"."Supplier_Rating"
ORDER BY "Total_Spend" DESC"""


def payment_delay_query() -> str:
    """
    Query 6: Payment delays.
    Compares payment execution date with invoice due date to detect late supplier payments.
    """
    return """-- Query 6: Late Supplier Payments (Payment Date > Due Date)
SELECT
    "Invoices"."Invoice_ID",
    "Purchase_Orders"."PO_ID",
    "Suppliers"."Supplier_Name",
    "Invoices"."Due_Date",
    "Payments"."Payment_Date",
    DATEDIFF(dd, "Invoices"."Due_Date", "Payments"."Payment_Date") AS "Days_Past_Due",
    "Payments"."Payment_Amount"
FROM "Invoices"
JOIN "Payments" ON "Payments"."Invoice_ID" = "Invoices"."Invoice_ID"
JOIN "Purchase_Orders" ON "Invoices"."PO_ID" = "Purchase_Orders"."PO_ID"
JOIN "Suppliers" ON "Purchase_Orders"."Supplier_ID" = "Suppliers"."Supplier_ID"
WHERE "Payments"."Payment_Date" > "Invoices"."Due_Date"
ORDER BY "Days_Past_Due" DESC"""


def process_rework_query() -> str:
    """
    Query 7: Process rework rate across activities.
    Computes the percentage of cases suffering from any repeated activity.
    """
    return """-- Query 7: Overall Case Rework Rate
100.0 * (
    COUNT(DISTINCT CASE WHEN PU_MAX("Purchase_Orders", PU_COUNT("Purchase_Orders", "Event_Log"."Event_ID", "Event_Log"."Activity")) > 1 THEN "Purchase_Orders"."PO_ID" ELSE NULL END)
    /
    COUNT_TABLE("Purchase_Orders")
)"""


def process_compliance_query() -> str:
    """
    Query 8: Process happy-path compliance rate.
    Computes the ratio of cases strictly matching the standard 8-step happy path.
    """
    return """-- Query 8: Standard Purchase-to-Pay Happy Path Compliance Rate (%)
100.0 * (
    COUNT(
        CASE WHEN PROCESS EQUALS
            'Purchase Requisition Created' ->
            'Purchase Order Generated' ->
            'Purchase Order Approved' ->
            'Goods Received' ->
            'Invoice Received' ->
            'Invoice Validated' ->
            'Invoice Approved' ->
            'Payment Executed'
        THEN "Purchase_Orders"."PO_ID"
        ELSE NULL END
    )
    /
    COUNT_TABLE("Purchase_Orders")
)"""


def get_all_pql_queries() -> Dict[str, Dict]:
    """Returns catalog of all production PQL queries with metadata."""
    return {
        "p2p_throughput_time": {
            "title": "Average P2P Throughput Time",
            "category": "Performance",
            "description": "Calculates overall duration from Requisition Creation to Payment Execution.",
            "query": throughput_time_query()
        },
        "variant_distribution": {
            "title": "Volume by Process Variant",
            "category": "Process Discovery",
            "description": "Groups all cases into distinct end-to-end trace variants.",
            "query": po_by_variant_query()
        },
        "repeated_approvals": {
            "title": "Repeated Approval Cycles (Rework)",
            "category": "Inefficiency",
            "description": "Flags cases where PO approval required multiple attempts or loopbacks.",
            "query": repeated_approvals_query()
        },
        "missing_validation": {
            "title": "Missing Invoice Validation",
            "category": "Compliance & Audit",
            "description": "Identifies invoices approved without documented operational validation.",
            "query": missing_validation_query()
        },
        "supplier_performance": {
            "title": "Supplier Delivery & Lead Time",
            "category": "Vendor Management",
            "description": "Measures vendor fulfillment cycle times and total procurement spend.",
            "query": supplier_performance_query()
        },
        "payment_delays": {
            "title": "Payment Delays (Late Payment Penalty Risk)",
            "category": "Working Capital",
            "description": "Identifies invoices settled past their contractual due date.",
            "query": payment_delay_query()
        },
        "process_rework": {
            "title": "Process Rework Percentage",
            "category": "Quality",
            "description": "Proportion of total procurement volume containing repeated activities.",
            "query": process_rework_query()
        },
        "process_compliance": {
            "title": "Happy-Path Conformance Rate",
            "category": "Conformance",
            "description": "Percentage of purchase orders executing the standard 8-step P2P lifecycle.",
            "query": process_compliance_query()
        }
    }
