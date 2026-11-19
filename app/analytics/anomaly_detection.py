"""
ProcureLens Anomaly & Risk Detection Engine.

Detects operational deviations, potential procurement fraud, compliance violations, and bottlenecks:
- Duplicate invoice attempts
- Missing mandatory approval steps
- Out-of-order sequence deviations (e.g., Payment executed before Goods Received)
- Late supplier payments past due date
- Excessive activity rework loops
- Transaction amount discrepancies (Invoice Amount > PO Amount)
"""

import logging
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger("procurelens.analytics.anomaly_detection")


class AnomalyDetector:
    """
    Scans relational procurement records and event logs to flag business anomalies and fraud risks.
    """

    def detect_duplicate_invoices(self, invoices_df: pd.DataFrame) -> List[Dict]:
        """
        Flags duplicate invoice submissions (same PO_ID and exact same Invoice_Amount).
        """
        if invoices_df.empty or "PO_ID" not in invoices_df.columns or "Invoice_Amount" not in invoices_df.columns:
            return []

        anomalies = []
        # Find duplicates on PO_ID and Invoice_Amount
        dups = invoices_df[invoices_df.duplicated(subset=["PO_ID", "Invoice_Amount"], keep=False)].copy()

        for _, row in dups.iterrows():
            anomalies.append({
                "type": "DUPLICATE_INVOICE",
                "record_id": f"INV-{row['Invoice_ID']}",
                "case_id": f"PO-{row['PO_ID']}",
                "description": (
                    f"Duplicate invoice detected: Invoice #{row['Invoice_ID']} submitted with amount "
                    f"${row['Invoice_Amount']:,.2f} matching another invoice for PO #{row['PO_ID']}."
                ),
                "severity": "HIGH",
                "details": f"PO_ID={row['PO_ID']}, Amount=${row['Invoice_Amount']:,.2f}, Status={row.get('Invoice_Status', 'N/A')}"
            })

        return anomalies

    def detect_missing_approvals(self, event_log_df: pd.DataFrame) -> List[Dict]:
        """
        Identifies cases that progressed to invoice/payment without formal 'Purchase Order Approved' or 'Invoice Approved'.
        """
        if event_log_df.empty:
            return []

        anomalies = []
        grouped = event_log_df.groupby("Case_ID")["Activity"].apply(set)

        for case_id, activities in grouped.items():
            has_po_approval = "Purchase Order Approved" in activities
            has_inv_approval = "Invoice Approved" in activities
            has_payment = "Payment Executed" in activities

            if has_payment and not has_po_approval:
                anomalies.append({
                    "type": "MISSING_PO_APPROVAL",
                    "record_id": str(case_id),
                    "case_id": str(case_id),
                    "description": (
                        f"Compliance violation: Case {case_id} reached payment without formal "
                        f"'Purchase Order Approved' activity in the audit trail."
                    ),
                    "severity": "HIGH",
                    "details": "Payment executed with missing Purchase Order Approval"
                })

            if has_payment and not has_inv_approval:
                anomalies.append({
                    "type": "MISSING_INVOICE_APPROVAL",
                    "record_id": str(case_id),
                    "case_id": str(case_id),
                    "description": (
                        f"Segregation-of-duties risk: Case {case_id} completed payment without formal "
                        f"'Invoice Approved' activity."
                    ),
                    "severity": "HIGH",
                    "details": "Payment executed with missing Invoice Approval"
                })

        return anomalies

    def detect_unusual_sequences(self, event_log_df: pd.DataFrame) -> List[Dict]:
        """
        Flags out-of-order process milestones (e.g. Invoice Received before Goods Received, or Payment before Invoice).
        """
        if event_log_df.empty:
            return []

        anomalies = []
        df_sorted = event_log_df.sort_values(by=["Case_ID", "Timestamp"])

        for case_id, group in df_sorted.groupby("Case_ID"):
            activities = group["Activity"].tolist()

            # Sequence Check 1: Payment executed before Goods Received
            if "Payment Executed" in activities and "Goods Received" in activities:
                pay_idx = activities.index("Payment Executed")
                gr_idx = activities.index("Goods Received")
                if pay_idx < gr_idx:
                    anomalies.append({
                        "type": "PREMATURE_PAYMENT",
                        "record_id": str(case_id),
                        "case_id": str(case_id),
                        "description": (
                            f"Control failure in Case {case_id}: Payment was executed prior to Goods Receipt confirmation."
                        ),
                        "severity": "HIGH",
                        "details": f"Payment at position {pay_idx+1}, Goods Receipt at position {gr_idx+1}"
                    })

            # Sequence Check 2: Invoice Approved before Invoice Validated
            if "Invoice Approved" in activities and "Invoice Validated" not in activities:
                anomalies.append({
                    "type": "UNVALIDATED_INVOICE_APPROVAL",
                    "record_id": str(case_id),
                    "case_id": str(case_id),
                    "description": (
                        f"Operational skip in Case {case_id}: Invoice was approved without 3-way matching or validation."
                    ),
                    "severity": "MEDIUM",
                    "details": "Missing 'Invoice Validated' step before approval"
                })

        return anomalies

    def detect_late_payments(
        self,
        payments_df: pd.DataFrame,
        invoices_df: pd.DataFrame
    ) -> List[Dict]:
        """
        Detects payments executed past invoice contractual due dates.
        """
        if payments_df.empty or invoices_df.empty:
            return []

        anomalies = []
        merged = pd.merge(
            payments_df,
            invoices_df[["Invoice_ID", "PO_ID", "Due_Date", "Invoice_Amount"]],
            on="Invoice_ID",
            how="inner"
        )

        merged["Payment_Date"] = pd.to_datetime(merged["Payment_Date"])
        merged["Due_Date"] = pd.to_datetime(merged["Due_Date"])

        late_mask = merged["Payment_Date"] > merged["Due_Date"]
        late_payments = merged[late_mask].copy()
        late_payments["Days_Late"] = (late_payments["Payment_Date"] - late_payments["Due_Date"]).dt.days

        for _, row in late_payments.iterrows():
            days = int(row["Days_Late"])
            severity = "HIGH" if days > 14 else ("MEDIUM" if days > 5 else "LOW")
            anomalies.append({
                "type": "LATE_PAYMENT",
                "record_id": f"PAY-{row['Payment_ID']}",
                "case_id": f"PO-{row['PO_ID']}",
                "description": (
                    f"Late supplier payment: Payment #{row['Payment_ID']} executed {days} days past due date "
                    f"(Due: {row['Due_Date'].strftime('%Y-%m-%d')}, Paid: {row['Payment_Date'].strftime('%Y-%m-%d')})."
                ),
                "severity": severity,
                "details": f"Amount=${row['Payment_Amount']:,.2f}, Days_Past_Due={days}"
            })

        return anomalies

    def detect_repeated_activities(
        self,
        event_log_df: pd.DataFrame,
        threshold: int = 2
    ) -> List[Dict]:
        """
        Flags excessive rework loops where an activity repeats multiple times.
        """
        if event_log_df.empty:
            return []

        anomalies = []
        counts = event_log_df.groupby(["Case_ID", "Activity"]).size().reset_index(name="Count")
        rework = counts[counts["Count"] >= threshold]

        for _, row in rework.iterrows():
            count = int(row["Count"])
            severity = "HIGH" if count >= 3 else "MEDIUM"
            anomalies.append({
                "type": "EXCESSIVE_REWORK",
                "record_id": f"{row['Case_ID']}-{row['Activity']}",
                "case_id": str(row["Case_ID"]),
                "description": (
                    f"Rework bottleneck in Case {row['Case_ID']}: Activity '{row['Activity']}' "
                    f"was executed {count} times (threshold={threshold})."
                ),
                "severity": severity,
                "details": f"Activity={row['Activity']}, Execution_Count={count}"
            })

        return anomalies

    def detect_invalid_transactions(
        self,
        purchase_orders_df: pd.DataFrame,
        invoices_df: pd.DataFrame
    ) -> List[Dict]:
        """
        Detects price variance where cumulative invoice amounts exceed PO approved amount by > 5%.
        """
        if purchase_orders_df.empty or invoices_df.empty:
            return []

        anomalies = []
        inv_totals = invoices_df.groupby("PO_ID")["Invoice_Amount"].sum().reset_index(name="Total_Invoiced")
        merged = pd.merge(purchase_orders_df, inv_totals, on="PO_ID", how="inner")

        # Flag where invoiced > 1.05 * PO amount
        variance_mask = merged["Total_Invoiced"] > (merged["Purchase_Amount"] * 1.05)
        discrepancies = merged[variance_mask]

        for _, row in discrepancies.iterrows():
            diff = row["Total_Invoiced"] - row["Purchase_Amount"]
            pct_over = round((diff / row["Purchase_Amount"]) * 100.0, 1)
            anomalies.append({
                "type": "PRICE_VARIANCE_OVERBILL",
                "record_id": f"PO-{row['PO_ID']}",
                "case_id": f"PO-{row['PO_ID']}",
                "description": (
                    f"Overbilling discrepancy: Invoiced total (${row['Total_Invoiced']:,.2f}) exceeds "
                    f"authorized PO #{row['PO_ID']} amount (${row['Purchase_Amount']:,.2f}) by ${diff:,.2f} (+{pct_over}%)."
                ),
                "severity": "HIGH",
                "details": f"PO_Amount=${row['Purchase_Amount']:,.2f}, Invoiced=${row['Total_Invoiced']:,.2f}, Diff=+${diff:,.2f}"
            })

        return anomalies

    def run_all_checks(self, datasets: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        Executes all anomaly scans across datasets and consolidates prioritized findings.
        """
        all_anomalies = []

        # 1. Duplicate Invoices
        if "invoices" in datasets:
            all_anomalies.extend(self.detect_duplicate_invoices(datasets["invoices"]))

        # 2. Event Log Checks (Missing Approvals, Sequences, Repeated Activities)
        if "event_log" in datasets:
            evt = datasets["event_log"]
            all_anomalies.extend(self.detect_missing_approvals(evt))
            all_anomalies.extend(self.detect_unusual_sequences(evt))
            all_anomalies.extend(self.detect_repeated_activities(evt, threshold=2))

        # 3. Late Payments
        if "payments" in datasets and "invoices" in datasets:
            all_anomalies.extend(self.detect_late_payments(datasets["payments"], datasets["invoices"]))

        # 4. Transaction Price Variances
        if "purchase_orders" in datasets and "invoices" in datasets:
            all_anomalies.extend(self.detect_invalid_transactions(datasets["purchase_orders"], datasets["invoices"]))

        # Rank by severity: HIGH -> MEDIUM -> LOW
        severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        all_anomalies.sort(key=lambda x: severity_order.get(x["severity"], 3))

        logger.info(f"Anomaly detection complete. Identified {len(all_anomalies)} total operational anomalies.")
        return all_anomalies
