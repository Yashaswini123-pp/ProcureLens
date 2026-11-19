"""
Common database query functions for ProcureLens.

Every function accepts a SQLAlchemy Engine and returns either a
pandas DataFrame or a plain dict, making them suitable for use in
dashboards, reports, and analytics pipelines.
"""

from typing import Dict, Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


# ---------------------------------------------------------------------------
# Raw table readers
# ---------------------------------------------------------------------------

def get_all_purchase_orders(engine: Engine) -> pd.DataFrame:
    """Return all purchase orders as a DataFrame."""
    query = text("""
        SELECT
            po.PO_ID,
            po.Supplier_ID,
            s.Supplier_Name,
            s.Supplier_Category,
            po.Department,
            po.Purchase_Amount,
            po.Creation_Date,
            po.Approval_Date,
            po.PO_Status
        FROM Purchase_Orders po
        LEFT JOIN Suppliers s ON po.Supplier_ID = s.Supplier_ID
        ORDER BY po.PO_ID
    """)
    return pd.read_sql(query, engine)


def get_all_suppliers(engine: Engine) -> pd.DataFrame:
    """Return all suppliers as a DataFrame."""
    query = text("SELECT * FROM Suppliers ORDER BY Supplier_ID")
    return pd.read_sql(query, engine)


def get_all_invoices(engine: Engine) -> pd.DataFrame:
    """Return all invoices joined with PO info."""
    query = text("""
        SELECT
            i.Invoice_ID,
            i.PO_ID,
            po.Department,
            po.Supplier_ID,
            s.Supplier_Name,
            i.Invoice_Amount,
            i.Invoice_Date,
            i.Due_Date,
            i.Invoice_Status
        FROM Invoices i
        LEFT JOIN Purchase_Orders po ON i.PO_ID = po.PO_ID
        LEFT JOIN Suppliers s ON po.Supplier_ID = s.Supplier_ID
        ORDER BY i.Invoice_ID
    """)
    return pd.read_sql(query, engine)


def get_all_payments(engine: Engine) -> pd.DataFrame:
    """Return all payments joined with invoice and PO info."""
    query = text("""
        SELECT
            p.Payment_ID,
            p.Invoice_ID,
            i.PO_ID,
            p.Payment_Date,
            p.Payment_Amount,
            p.Payment_Status,
            i.Invoice_Amount,
            i.Due_Date,
            JULIANDAY(p.Payment_Date) - JULIANDAY(i.Due_Date) AS Days_Late
        FROM Payments p
        LEFT JOIN Invoices i ON p.Invoice_ID = i.Invoice_ID
        ORDER BY p.Payment_ID
    """)
    return pd.read_sql(query, engine)


def get_all_goods_receipts(engine: Engine) -> pd.DataFrame:
    """Return all goods receipts joined with PO info."""
    query = text("""
        SELECT
            gr.Receipt_ID,
            gr.PO_ID,
            po.Supplier_ID,
            s.Supplier_Name,
            po.Department,
            gr.Receipt_Date,
            gr.Received_Quantity
        FROM Goods_Receipts gr
        LEFT JOIN Purchase_Orders po ON gr.PO_ID = po.PO_ID
        LEFT JOIN Suppliers s ON po.Supplier_ID = s.Supplier_ID
        ORDER BY gr.Receipt_ID
    """)
    return pd.read_sql(query, engine)


def get_all_event_logs(engine: Engine) -> pd.DataFrame:
    """Return full event log ordered by Case_ID and Timestamp."""
    query = text("""
        SELECT *
        FROM Event_Log
        ORDER BY Case_ID, Timestamp
    """)
    return pd.read_sql(query, engine)


# ---------------------------------------------------------------------------
# Analytical queries
# ---------------------------------------------------------------------------

def get_purchase_order_summary(engine: Engine) -> Dict[str, Any]:
    """Return high-level PO statistics as a dict.

    Keys: total_pos, total_amount, avg_amount, by_status (dict),
    by_department (dict), date_range (tuple).
    """
    base = pd.read_sql(
        text("SELECT * FROM Purchase_Orders"), engine
    )
    summary: Dict[str, Any] = {
        "total_pos": len(base),
        "total_amount": float(base["Purchase_Amount"].sum()),
        "avg_amount": float(base["Purchase_Amount"].mean()),
        "by_status": base["PO_Status"].value_counts().to_dict(),
        "by_department": (
            base.groupby("Department")["Purchase_Amount"]
            .sum()
            .to_dict()
        ),
        "date_range": (
            str(base["Creation_Date"].min()),
            str(base["Creation_Date"].max()),
        ),
    }
    return summary


def get_supplier_performance(engine: Engine) -> pd.DataFrame:
    """Aggregate supplier performance metrics.

    Columns: Supplier_ID, Supplier_Name, Supplier_Category, Location,
    Supplier_Rating, Total_POs, Total_PO_Amount, Avg_PO_Amount,
    Total_Invoices, Total_Invoice_Amount, Avg_Days_Late.
    """
    query = text("""
        SELECT
            s.Supplier_ID,
            s.Supplier_Name,
            s.Supplier_Category,
            s.Location,
            s.Supplier_Rating,
            COUNT(DISTINCT po.PO_ID) AS Total_POs,
            COALESCE(SUM(po.Purchase_Amount), 0) AS Total_PO_Amount,
            COALESCE(AVG(po.Purchase_Amount), 0) AS Avg_PO_Amount,
            COUNT(DISTINCT i.Invoice_ID) AS Total_Invoices,
            COALESCE(SUM(i.Invoice_Amount), 0) AS Total_Invoice_Amount,
            AVG(
                CASE
                    WHEN p.Payment_Date IS NOT NULL AND i.Due_Date IS NOT NULL
                    THEN JULIANDAY(p.Payment_Date) - JULIANDAY(i.Due_Date)
                    ELSE NULL
                END
            ) AS Avg_Days_Late
        FROM Suppliers s
        LEFT JOIN Purchase_Orders po ON s.Supplier_ID = po.Supplier_ID
        LEFT JOIN Invoices i ON po.PO_ID = i.PO_ID
        LEFT JOIN Payments p ON i.Invoice_ID = p.Invoice_ID
        GROUP BY
            s.Supplier_ID, s.Supplier_Name, s.Supplier_Category,
            s.Location, s.Supplier_Rating
        ORDER BY Total_PO_Amount DESC
    """)
    return pd.read_sql(query, engine)


def get_payment_delays(engine: Engine) -> pd.DataFrame:
    """Return payment delay analysis per invoice.

    Columns: Payment_ID, Invoice_ID, PO_ID, Department, Supplier_Name,
    Invoice_Amount, Payment_Amount, Due_Date, Payment_Date, Days_Late,
    Is_Late.
    """
    query = text("""
        SELECT
            p.Payment_ID,
            p.Invoice_ID,
            i.PO_ID,
            po.Department,
            s.Supplier_Name,
            i.Invoice_Amount,
            p.Payment_Amount,
            i.Due_Date,
            p.Payment_Date,
            JULIANDAY(p.Payment_Date) - JULIANDAY(i.Due_Date) AS Days_Late,
            CASE
                WHEN JULIANDAY(p.Payment_Date) > JULIANDAY(i.Due_Date)
                THEN 1 ELSE 0
            END AS Is_Late
        FROM Payments p
        JOIN Invoices i ON p.Invoice_ID = i.Invoice_ID
        JOIN Purchase_Orders po ON i.PO_ID = po.PO_ID
        JOIN Suppliers s ON po.Supplier_ID = s.Supplier_ID
        ORDER BY Days_Late DESC
    """)
    return pd.read_sql(query, engine)


def get_department_stats(engine: Engine) -> pd.DataFrame:
    """Aggregate procurement statistics per department.

    Columns: Department, Total_POs, Total_PO_Amount, Avg_PO_Amount,
    Total_Invoices, Total_Payments, Avg_Days_Late, Unique_Suppliers.
    """
    query = text("""
        SELECT
            po.Department,
            COUNT(DISTINCT po.PO_ID)        AS Total_POs,
            SUM(po.Purchase_Amount)          AS Total_PO_Amount,
            AVG(po.Purchase_Amount)          AS Avg_PO_Amount,
            COUNT(DISTINCT i.Invoice_ID)     AS Total_Invoices,
            COUNT(DISTINCT p.Payment_ID)     AS Total_Payments,
            AVG(
                CASE
                    WHEN p.Payment_Date IS NOT NULL AND i.Due_Date IS NOT NULL
                    THEN JULIANDAY(p.Payment_Date) - JULIANDAY(i.Due_Date)
                    ELSE NULL
                END
            ) AS Avg_Days_Late,
            COUNT(DISTINCT po.Supplier_ID)   AS Unique_Suppliers
        FROM Purchase_Orders po
        LEFT JOIN Invoices i ON po.PO_ID = i.PO_ID
        LEFT JOIN Payments p ON i.Invoice_ID = p.Invoice_ID
        GROUP BY po.Department
        ORDER BY Total_PO_Amount DESC
    """)
    return pd.read_sql(query, engine)


def get_event_log_for_case(engine: Engine, case_id: str) -> pd.DataFrame:
    """Return all events for a specific Case_ID, ordered by Timestamp."""
    query = text("""
        SELECT *
        FROM Event_Log
        WHERE Case_ID = :case_id
        ORDER BY Timestamp
    """)
    return pd.read_sql(query, engine, params={"case_id": case_id})
