"""
ProcureLens Executive KPI & Metrics Calculator.

Computes live enterprise procurement performance indicators:
- Total PO volume & procurement spend
- End-to-end process throughput time
- Approval cycle duration and bottlenecks
- Supplier payment delay rates and aging
- Process compliance rate (P2P happy-path)
- Departmental spend and cycle time comparisons
- Monthly trend aggregations
"""

import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger("procurelens.analytics.kpi_calculator")


class KPICalculator:
    """Computes executive and operational KPIs directly from relational database tables."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def total_purchase_orders(self) -> int:
        """Total number of purchase orders in the system."""
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM Purchase_Orders")).scalar()
            return int(result or 0)

    def total_procurement_value(self) -> float:
        """Total monetary value of all purchase orders."""
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT SUM(Purchase_Amount) FROM Purchase_Orders")).scalar()
            return round(float(result or 0.0), 2)

    def average_processing_time(self) -> float:
        """Average duration (in days) from case start to completion."""
        query = """
        SELECT AVG(duration_days) FROM (
            SELECT Case_ID,
                   (JULIANDAY(MAX(Timestamp)) - JULIANDAY(MIN(Timestamp))) AS duration_days
            FROM Event_Log
            GROUP BY Case_ID
        )
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query)).scalar()
            return round(float(result or 0.0), 1)

    def delayed_payment_count(self) -> int:
        """Number of payments made past the invoice due date."""
        query = """
        SELECT COUNT(*)
        FROM Payments p
        JOIN Invoices i ON p.Invoice_ID = i.Invoice_ID
        WHERE p.Payment_Date > i.Due_Date
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query)).scalar()
            return int(result or 0)

    def delayed_payment_percentage(self) -> float:
        """Percentage of total payments that were delayed."""
        query = """
        SELECT
            100.0 * COUNT(CASE WHEN p.Payment_Date > i.Due_Date THEN 1 END) / NULLIF(COUNT(*), 0)
        FROM Payments p
        JOIN Invoices i ON p.Invoice_ID = i.Invoice_ID
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query)).scalar()
            return round(float(result or 0.0), 1)

    def approval_delay_average(self) -> float:
        """Average time in days from PO creation to approval."""
        query = """
        SELECT AVG(JULIANDAY(Approval_Date) - JULIANDAY(Creation_Date))
        FROM Purchase_Orders
        WHERE Approval_Date IS NOT NULL AND Approval_Date >= Creation_Date
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query)).scalar()
            return round(float(result or 0.0), 1)

    def process_compliance_rate(self) -> float:
        """
        Percentage of purchase orders that strictly followed the standard 8-step process.
        """
        query = """
        SELECT Case_ID, Activity, Timestamp
        FROM Event_Log
        ORDER BY Case_ID, Timestamp
        """
        df = pd.read_sql_query(query, self.engine)
        if df.empty:
            return 0.0

        standard_seq = (
            "Purchase Requisition Created",
            "Purchase Order Generated",
            "Purchase Order Approved",
            "Goods Received",
            "Invoice Received",
            "Invoice Validated",
            "Invoice Approved",
            "Payment Executed"
        )

        grouped = df.groupby("Case_ID")["Activity"].apply(tuple)
        total = len(grouped)
        conforming = sum(1 for seq in grouped.values if seq == standard_seq)
        return round((conforming / total) * 100.0, 1) if total > 0 else 0.0

    def monthly_trends(self) -> List[Dict]:
        """Aggregated monthly purchase order volume, total spend, and average amount."""
        query = """
        SELECT
            STRFTIME('%Y-%m', Creation_Date) AS Month,
            COUNT(*) AS PO_Count,
            SUM(Purchase_Amount) AS Total_Spend,
            AVG(Purchase_Amount) AS Avg_Amount
        FROM Purchase_Orders
        GROUP BY Month
        ORDER BY Month ASC
        """
        df = pd.read_sql_query(query, self.engine)
        return df.to_dict(orient="records")

    def department_comparison(self) -> List[Dict]:
        """Departmental breakdown of PO volume, spend, and approval duration."""
        query = """
        SELECT
            Department,
            COUNT(*) AS PO_Count,
            SUM(Purchase_Amount) AS Total_Spend,
            AVG(Purchase_Amount) AS Avg_Amount,
            AVG(JULIANDAY(Approval_Date) - JULIANDAY(Creation_Date)) AS Avg_Approval_Days
        FROM Purchase_Orders
        GROUP BY Department
        ORDER BY Total_Spend DESC
        """
        df = pd.read_sql_query(query, self.engine)
        if "Avg_Approval_Days" in df.columns:
            df["Avg_Approval_Days"] = df["Avg_Approval_Days"].round(1)
        if "Total_Spend" in df.columns:
            df["Total_Spend"] = df["Total_Spend"].round(2)
        return df.to_dict(orient="records")

    def procurement_status_distribution(self) -> Dict[str, int]:
        """Count of purchase orders by PO_Status."""
        query = """
        SELECT PO_Status, COUNT(*) AS Count
        FROM Purchase_Orders
        GROUP BY PO_Status
        """
        df = pd.read_sql_query(query, self.engine)
        return dict(zip(df["PO_Status"], df["Count"]))

    def supplier_performance_scores(self, top_n: int = 15) -> List[Dict]:
        """Top suppliers ranked by spend with rating and delivery delay metrics."""
        query = """
        SELECT
            s.Supplier_ID,
            s.Supplier_Name,
            s.Supplier_Category,
            s.Supplier_Rating,
            COUNT(po.PO_ID) AS Total_Orders,
            SUM(po.Purchase_Amount) AS Total_Spend,
            COUNT(CASE WHEN p.Payment_Date > i.Due_Date THEN 1 END) AS Delayed_Payments
        FROM Suppliers s
        JOIN Purchase_Orders po ON s.Supplier_ID = po.Supplier_ID
        LEFT JOIN Invoices i ON po.PO_ID = i.PO_ID
        LEFT JOIN Payments p ON i.Invoice_ID = p.Invoice_ID
        GROUP BY s.Supplier_ID, s.Supplier_Name, s.Supplier_Category, s.Supplier_Rating
        ORDER BY Total_Spend DESC
        LIMIT :limit
        """
        df = pd.read_sql_query(text(query), self.engine, params={"limit": top_n})
        if "Total_Spend" in df.columns:
            df["Total_Spend"] = df["Total_Spend"].round(2)
        return df.to_dict(orient="records")

    def get_full_summary(self) -> Dict:
        """Consolidated executive dashboard summary payload."""
        total_pos = self.total_purchase_orders()
        total_val = self.total_procurement_value()
        avg_time = self.average_processing_time()
        delayed_cnt = self.delayed_payment_count()
        delayed_pct = self.delayed_payment_percentage()
        approval_delay = self.approval_delay_average()
        compliance = self.process_compliance_rate()

        return {
            "total_purchase_orders": total_pos,
            "total_procurement_value": total_val,
            "avg_processing_time_days": avg_time,
            "delayed_payments": delayed_cnt,
            "delayed_payment_pct": delayed_pct,
            "approval_delay_avg_days": approval_delay,
            "compliance_rate": compliance,
            "monthly_trends": self.monthly_trends(),
            "department_comparison": self.department_comparison(),
            "status_distribution": self.procurement_status_distribution()
        }
