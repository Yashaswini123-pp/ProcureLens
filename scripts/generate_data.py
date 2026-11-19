#!/usr/bin/env python3
"""
Synthetic data generator for ProcureLens.

Creates realistic procurement data — suppliers, purchase orders, invoices,
payments, goods receipts, and a process‑mining event log — then persists
everything to CSV files (data/raw/) and an SQLite database (data/procurelens.db).

Usage:
    python scripts/generate_data.py
"""

import os
import sys

# ---------------------------------------------------------------------------
# Path setup so ``app.*`` imports resolve when running from the repo root.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from app.database.models import Base
from app.database.connection import init_db

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
DB_PATH = os.path.join(DATA_DIR, "procurelens.db")
DB_URL = f"sqlite:///{DB_PATH}"

os.makedirs(RAW_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants / Look‑ups
# ---------------------------------------------------------------------------
SUPPLIER_CATEGORIES: List[str] = [
    "Raw Materials",
    "IT Services",
    "Office Supplies",
    "Logistics",
    "Professional Services",
    "Manufacturing",
    "MRO Supplies",
    "Packaging",
]

DEPARTMENTS: List[str] = [
    "Procurement",
    "Finance",
    "Operations",
    "IT",
    "HR",
    "Marketing",
    "R&D",
    "Legal",
]

PO_STATUSES: List[str] = [
    "Approved",
    "Pending",
    "Closed",
    "Rejected",
    "Cancelled",
]

INVOICE_STATUSES: List[str] = [
    "Paid",
    "Pending",
    "Overdue",
    "Disputed",
    "Cancelled",
]

PAYMENT_STATUSES: List[str] = [
    "Completed",
    "Pending",
    "Failed",
    "Reversed",
]

LOCATIONS: List[str] = [
    "New York, USA",
    "Los Angeles, USA",
    "Chicago, USA",
    "Houston, USA",
    "Dallas, USA",
    "San Francisco, USA",
    "Seattle, USA",
    "Boston, USA",
    "Atlanta, USA",
    "Denver, USA",
    "London, UK",
    "Berlin, Germany",
    "Paris, France",
    "Toronto, Canada",
    "Mumbai, India",
    "Shanghai, China",
    "Tokyo, Japan",
    "Sydney, Australia",
    "São Paulo, Brazil",
    "Singapore",
]

COMPANY_PREFIXES: List[str] = [
    "Global", "United", "Pacific", "Atlantic", "Premier", "Elite",
    "Apex", "Summit", "Pinnacle", "Nexus", "Prime", "Core",
    "Alpha", "Omega", "Delta", "Sigma", "Vanguard", "Titan",
    "Sterling", "Meridian", "Vertex", "Catalyst", "Horizon",
    "Evergreen", "Ironclad", "Cobalt", "Crimson", "Azure",
    "Sapphire", "Emerald",
]

COMPANY_SUFFIXES: List[str] = [
    "Industries", "Solutions", "Corp", "Enterprises", "Group",
    "Technologies", "Services", "International", "Partners", "Co",
    "Holdings", "Systems", "Supply Co", "Materials", "Dynamics",
    "Logistics", "Manufacturing", "Associates", "Consulting", "Ltd",
]

COMPANY_MIDDLES: List[str] = [
    "Tech", "Supply", "Resource", "Trade", "Freight", "Data",
    "Build", "Craft", "Link", "Net", "Star", "Wave", "Flow",
    "Edge", "Path", "Bridge", "Arc", "Line", "Point", "Mark",
]

# Resources / people who perform activities
RESOURCES: Dict[str, List[str]] = {
    "Purchase Requisition Created": [
        "John Smith", "Maria Garcia", "David Kim", "Sarah Johnson",
        "James Wilson", "Emily Chen", "Robert Brown", "Lisa Anderson",
    ],
    "Purchase Order Generated": [
        "Michael Taylor", "Jennifer Lee", "Chris Martinez",
        "Amanda White", "Daniel Harris",
    ],
    "Purchase Order Approved": [
        "Patricia Moore", "Thomas Jackson", "Nancy Thompson",
        "Steven Clark", "Karen Lewis",
    ],
    "Goods Received": [
        "Kevin Robinson", "Sandra Hall", "Brian Young",
        "Donna Allen", "Paul King",
    ],
    "Invoice Received": [
        "Michelle Wright", "George Lopez", "Betty Hill",
        "Edward Scott", "Dorothy Green",
    ],
    "Invoice Validated": [
        "Ronald Adams", "Helen Baker", "Kenneth Nelson",
        "Carol Carter", "Jason Mitchell",
    ],
    "Invoice Approved": [
        "Patricia Moore", "Thomas Jackson", "Nancy Thompson",
        "Steven Clark", "Karen Lewis",
    ],
    "Invoice Rejected": [
        "Patricia Moore", "Thomas Jackson", "Nancy Thompson",
    ],
    "Payment Executed": [
        "Sharon Perez", "Timothy Roberts", "Deborah Turner",
        "Jeffrey Phillips", "Cynthia Campbell",
    ],
}

# Typical processing‑time ranges between activities (min_days, max_days)
STEP_DURATIONS: Dict[str, Tuple[int, int]] = {
    "Purchase Requisition Created": (0, 0),       # starting event
    "Purchase Order Generated":    (1, 5),
    "Purchase Order Approved":     (1, 7),
    "Goods Received":              (3, 20),
    "Invoice Received":            (1, 10),
    "Invoice Validated":           (1, 5),
    "Invoice Approved":            (1, 3),
    "Invoice Rejected":            (1, 3),
    "Payment Executed":            (1, 30),
}

DATE_START = datetime(2024, 1, 1)
DATE_END = datetime(2024, 6, 30)
TOTAL_DAYS = (DATE_END - DATE_START).days


# ===================================================================
# Helper utilities
# ===================================================================

def random_datetime(start: datetime, end: datetime) -> datetime:
    """Return a random datetime between *start* and *end*."""
    delta = end - start
    seconds = int(delta.total_seconds())
    offset = random.randint(0, max(seconds, 1))
    return start + timedelta(seconds=offset)


def add_days(dt: datetime, min_d: int, max_d: int) -> datetime:
    """Add a random number of days (+ random hours) to *dt*."""
    days = random.randint(min_d, max_d)
    hours = random.randint(0, 23)
    minutes = random.randint(0, 59)
    return dt + timedelta(days=days, hours=hours, minutes=minutes)


def generate_supplier_name() -> str:
    """Create a plausible company name."""
    pattern = random.choice(["pm_s", "p_ms", "p_s", "pm"])
    prefix = random.choice(COMPANY_PREFIXES)
    middle = random.choice(COMPANY_MIDDLES)
    suffix = random.choice(COMPANY_SUFFIXES)
    if pattern == "pm_s":
        return f"{prefix} {middle} {suffix}"
    elif pattern == "p_ms":
        return f"{prefix} {middle}{suffix}"
    elif pattern == "p_s":
        return f"{prefix} {suffix}"
    else:
        return f"{prefix} {middle}"


# ===================================================================
# Data generation functions
# ===================================================================

def generate_suppliers(n: int = 500) -> pd.DataFrame:
    """Generate *n* supplier records."""
    names_seen: set = set()
    rows = []
    while len(rows) < n:
        name = generate_supplier_name()
        if name in names_seen:
            continue
        names_seen.add(name)
        rows.append(
            {
                "Supplier_ID": len(rows) + 1,
                "Supplier_Name": name,
                "Supplier_Category": random.choice(SUPPLIER_CATEGORIES),
                "Location": random.choice(LOCATIONS),
                "Supplier_Rating": round(random.uniform(1.0, 5.0), 2),
            }
        )
    return pd.DataFrame(rows)


def generate_purchase_orders(
    n: int = 5_000,
    supplier_ids: List[int] = None,
) -> pd.DataFrame:
    """Generate *n* purchase order records."""
    if supplier_ids is None:
        supplier_ids = list(range(1, 501))

    rows = []
    for po_id in range(1, n + 1):
        creation = random_datetime(DATE_START, DATE_END)
        status = random.choices(
            PO_STATUSES,
            weights=[50, 20, 20, 7, 3],
            k=1,
        )[0]
        approval = None
        if status in ("Approved", "Closed"):
            approval = add_days(creation, 1, 7)

        rows.append(
            {
                "PO_ID": po_id,
                "Supplier_ID": random.choice(supplier_ids),
                "Department": random.choice(DEPARTMENTS),
                "Purchase_Amount": round(random.uniform(500, 500_000), 2),
                "Creation_Date": creation,
                "Approval_Date": approval,
                "PO_Status": status,
            }
        )
    return pd.DataFrame(rows)


def generate_invoices(po_df: pd.DataFrame) -> pd.DataFrame:
    """Generate invoices for purchase orders.

    ~20 % of POs get 2–3 invoices (split billing), the rest get 1.
    Only POs with status Approved or Closed receive invoices.
    """
    eligible = po_df[po_df["PO_Status"].isin(["Approved", "Closed"])]
    rows = []
    invoice_id = 1

    for _, po in eligible.iterrows():
        n_invoices = (
            random.choice([2, 3])
            if random.random() < 0.20
            else 1
        )
        total = po["Purchase_Amount"]
        # Split amounts
        if n_invoices == 1:
            amounts = [total]
        else:
            splits = sorted(
                [random.uniform(0.1, 0.9) for _ in range(n_invoices - 1)]
            )
            splits = [0.0] + splits + [1.0]
            amounts = [
                round(total * (splits[i + 1] - splits[i]), 2)
                for i in range(n_invoices)
            ]

        base_date = po["Approval_Date"] or po["Creation_Date"]
        for amount in amounts:
            inv_date = add_days(base_date, 1, 15)
            due_date = add_days(inv_date, 15, 45)
            status = random.choices(
                INVOICE_STATUSES,
                weights=[45, 25, 15, 10, 5],
                k=1,
            )[0]
            rows.append(
                {
                    "Invoice_ID": invoice_id,
                    "PO_ID": int(po["PO_ID"]),
                    "Invoice_Amount": amount,
                    "Invoice_Date": inv_date,
                    "Due_Date": due_date,
                    "Invoice_Status": status,
                }
            )
            invoice_id += 1

    return pd.DataFrame(rows)


def generate_goods_receipts(po_df: pd.DataFrame) -> pd.DataFrame:
    """Generate goods receipt records for approved / closed POs."""
    eligible = po_df[po_df["PO_Status"].isin(["Approved", "Closed"])]
    rows = []
    receipt_id = 1

    for _, po in eligible.iterrows():
        base_date = po["Approval_Date"] or po["Creation_Date"]
        receipt_date = add_days(base_date, 3, 20)
        quantity = random.randint(1, 5000)
        rows.append(
            {
                "Receipt_ID": receipt_id,
                "PO_ID": int(po["PO_ID"]),
                "Receipt_Date": receipt_date,
                "Received_Quantity": quantity,
            }
        )
        receipt_id += 1

    return pd.DataFrame(rows)


def generate_payments(inv_df: pd.DataFrame) -> pd.DataFrame:
    """Generate payments for invoices.

    ~12 % of payments are late (Payment_Date > Due_Date).
    ~5 % of invoices get a duplicate payment attempt.
    """
    eligible = inv_df[inv_df["Invoice_Status"].isin(["Paid", "Pending", "Overdue"])]
    rows = []
    payment_id = 1

    for _, inv in eligible.iterrows():
        due = inv["Due_Date"]
        inv_date = inv["Invoice_Date"]

        # Decide if late
        is_late = random.random() < 0.12
        if is_late:
            pay_date = add_days(due, 1, 30)
            status = random.choices(
                ["Completed", "Pending"],
                weights=[70, 30],
                k=1,
            )[0]
        else:
            days_before = max(1, (due - inv_date).days - 1) if due > inv_date else 1
            pay_date = add_days(inv_date, 1, min(days_before, 30))
            status = "Completed"

        rows.append(
            {
                "Payment_ID": payment_id,
                "Invoice_ID": int(inv["Invoice_ID"]),
                "Payment_Date": pay_date,
                "Payment_Amount": inv["Invoice_Amount"],
                "Payment_Status": status,
            }
        )
        payment_id += 1

        # ~5 % duplicate payment attempt
        if random.random() < 0.05:
            dup_date = add_days(pay_date, 0, 2)
            rows.append(
                {
                    "Payment_ID": payment_id,
                    "Invoice_ID": int(inv["Invoice_ID"]),
                    "Payment_Date": dup_date,
                    "Payment_Amount": inv["Invoice_Amount"],
                    "Payment_Status": random.choice(["Failed", "Reversed"]),
                }
            )
            payment_id += 1

    return pd.DataFrame(rows)


# ===================================================================
# Event log generation
# ===================================================================

STANDARD_FLOW: List[str] = [
    "Purchase Requisition Created",
    "Purchase Order Generated",
    "Purchase Order Approved",
    "Goods Received",
    "Invoice Received",
    "Invoice Validated",
    "Invoice Approved",
    "Payment Executed",
]


def build_event_trace(
    case_id: str,
    department: str,
    start_time: datetime,
    variation: str,
) -> List[Dict]:
    """Build a single process trace with the requested variation.

    Parameters
    ----------
    variation : str
        One of "normal", "repeated_approval", "skipped_step",
        "invoice_rework", "duplicate_invoice".
    """
    events: List[Dict] = []
    flow = list(STANDARD_FLOW)

    # ----- Apply variation -----
    if variation == "repeated_approval":
        idx = flow.index("Purchase Order Approved")
        repeats = random.choice([2, 3])
        for _ in range(repeats - 1):
            flow.insert(idx + 1, "Purchase Order Approved")

    elif variation == "skipped_step":
        skip = random.choice(["Goods Received", "Invoice Validated"])
        if skip in flow:
            flow.remove(skip)

    elif variation == "invoice_rework":
        idx = flow.index("Invoice Validated")
        # Insert rejection + re‑receive + re‑validate after the first validation
        flow.insert(idx + 1, "Invoice Rejected")
        flow.insert(idx + 2, "Invoice Received")
        flow.insert(idx + 3, "Invoice Validated")

    elif variation == "duplicate_invoice":
        idx = flow.index("Invoice Received")
        flow.insert(idx + 1, "Invoice Received")  # duplicate

    # ----- Generate timestamps along the trace -----
    current_time = start_time
    for activity in flow:
        dur = STEP_DURATIONS.get(activity, (1, 5))
        if activity != flow[0]:
            current_time = add_days(current_time, dur[0], dur[1])

        resource = random.choice(
            RESOURCES.get(activity, ["System"])
        )
        events.append(
            {
                "Case_ID": case_id,
                "Activity": activity,
                "Timestamp": current_time,
                "Resource": resource,
                "Department": department,
            }
        )

    return events


def generate_event_log(po_df: pd.DataFrame) -> pd.DataFrame:
    """Create the process mining event log.

    Guarantees >= 30 000 events by assigning a variation mix that
    inflates the total above 30k (each trace has 8+ events).
    """
    all_events: List[Dict] = []

    n_cases = len(po_df)
    # Assign variation per case
    variation_pool: List[str] = []
    n_repeated = int(n_cases * 0.15)
    n_skipped = int(n_cases * 0.10)
    n_rework = int(n_cases * 0.08)
    n_duplicate = int(n_cases * 0.05)
    n_normal = n_cases - n_repeated - n_skipped - n_rework - n_duplicate

    variation_pool += ["normal"] * n_normal
    variation_pool += ["repeated_approval"] * n_repeated
    variation_pool += ["skipped_step"] * n_skipped
    variation_pool += ["invoice_rework"] * n_rework
    variation_pool += ["duplicate_invoice"] * n_duplicate
    random.shuffle(variation_pool)

    for idx, (_, po) in enumerate(po_df.iterrows()):
        case_id = f"CASE-{po['PO_ID']:05d}"
        variation = variation_pool[idx]
        trace = build_event_trace(
            case_id=case_id,
            department=po["Department"],
            start_time=po["Creation_Date"],
            variation=variation,
        )
        all_events.extend(trace)

    df = pd.DataFrame(all_events)
    df = df.sort_values(["Case_ID", "Timestamp"]).reset_index(drop=True)
    df["Event_ID"] = range(1, len(df) + 1)
    df = df[["Event_ID", "Case_ID", "Activity", "Timestamp", "Resource", "Department"]]
    return df


# ===================================================================
# Persistence helpers
# ===================================================================

def save_csv(df: pd.DataFrame, name: str) -> str:
    path = os.path.join(RAW_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    return path


def save_to_db(
    engine,
    suppliers_df: pd.DataFrame,
    po_df: pd.DataFrame,
    invoices_df: pd.DataFrame,
    goods_df: pd.DataFrame,
    payments_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> None:
    """Write all DataFrames to the SQLite database."""
    suppliers_df.to_sql("Suppliers", engine, if_exists="replace", index=False)
    po_df.to_sql("Purchase_Orders", engine, if_exists="replace", index=False)
    invoices_df.to_sql("Invoices", engine, if_exists="replace", index=False)
    goods_df.to_sql("Goods_Receipts", engine, if_exists="replace", index=False)
    payments_df.to_sql("Payments", engine, if_exists="replace", index=False)
    events_df.to_sql("Event_Log", engine, if_exists="replace", index=False)


# ===================================================================
# Summary statistics
# ===================================================================

def print_summary(
    suppliers_df: pd.DataFrame,
    po_df: pd.DataFrame,
    invoices_df: pd.DataFrame,
    goods_df: pd.DataFrame,
    payments_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("  ProcureLens — Synthetic Data Generation Summary")
    print(sep)

    print(f"\n{'Table':<25} {'Rows':>10}")
    print("-" * 37)
    print(f"{'Suppliers':<25} {len(suppliers_df):>10,}")
    print(f"{'Purchase_Orders':<25} {len(po_df):>10,}")
    print(f"{'Invoices':<25} {len(invoices_df):>10,}")
    print(f"{'Goods_Receipts':<25} {len(goods_df):>10,}")
    print(f"{'Payments':<25} {len(payments_df):>10,}")
    print(f"{'Event_Log':<25} {len(events_df):>10,}")

    print(f"\n--- Purchase Orders ---")
    print(f"  Total amount:  ${po_df['Purchase_Amount'].sum():>15,.2f}")
    print(f"  Avg amount:    ${po_df['Purchase_Amount'].mean():>15,.2f}")
    print(f"  Status distribution:")
    for status, count in po_df["PO_Status"].value_counts().items():
        print(f"    {status:<15} {count:>6,}  ({100*count/len(po_df):.1f}%)")

    print(f"\n--- Departments ---")
    dept_sums = po_df.groupby("Department")["Purchase_Amount"].sum().sort_values(ascending=False)
    for dept, total in dept_sums.items():
        n = (po_df["Department"] == dept).sum()
        print(f"    {dept:<15} {n:>5} POs   ${total:>15,.2f}")

    print(f"\n--- Suppliers ---")
    print(f"  Categories: {', '.join(suppliers_df['Supplier_Category'].unique())}")
    print(f"  Rating range: {suppliers_df['Supplier_Rating'].min():.2f} – "
          f"{suppliers_df['Supplier_Rating'].max():.2f}")

    print(f"\n--- Invoices ---")
    print(f"  Total invoice amount: ${invoices_df['Invoice_Amount'].sum():>15,.2f}")
    print(f"  Status distribution:")
    for status, count in invoices_df["Invoice_Status"].value_counts().items():
        print(f"    {status:<15} {count:>6,}  ({100*count/len(invoices_df):.1f}%)")

    print(f"\n--- Payments ---")
    print(f"  Total payment amount: ${payments_df['Payment_Amount'].sum():>15,.2f}")
    print(f"  Status distribution:")
    for status, count in payments_df["Payment_Status"].value_counts().items():
        print(f"    {status:<15} {count:>6,}  ({100*count/len(payments_df):.1f}%)")

    # Late payments
    if "Due_Date" not in payments_df.columns:
        # Merge to get due dates
        merged = payments_df.merge(
            invoices_df[["Invoice_ID", "Due_Date"]],
            on="Invoice_ID",
            how="left",
        )
    else:
        merged = payments_df

    if "Due_Date" in merged.columns:
        late = merged[merged["Payment_Date"] > merged["Due_Date"]]
        pct = 100 * len(late) / len(merged) if len(merged) > 0 else 0
        print(f"  Late payments: {len(late):,} ({pct:.1f}%)")

    print(f"\n--- Event Log ---")
    unique_cases = events_df["Case_ID"].nunique()
    print(f"  Unique cases:   {unique_cases:>8,}")
    print(f"  Total events:   {len(events_df):>8,}")
    print(f"  Avg events/case: {len(events_df)/unique_cases:>7.1f}")
    print(f"  Activities: {events_df['Activity'].nunique()}")
    for act, cnt in events_df["Activity"].value_counts().sort_index().items():
        print(f"    {act:<35} {cnt:>6,}")

    print(f"\n  CSV files saved to: {RAW_DIR}")
    print(f"  SQLite database:    {DB_PATH}")
    print(sep + "\n")


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    print("Generating ProcureLens synthetic data (seed=42) ...")

    # 1. Suppliers
    print("  [1/6] Suppliers …")
    suppliers_df = generate_suppliers(500)

    # 2. Purchase Orders
    print("  [2/6] Purchase Orders …")
    po_df = generate_purchase_orders(
        5_000,
        supplier_ids=suppliers_df["Supplier_ID"].tolist(),
    )

    # 3. Invoices
    print("  [3/6] Invoices …")
    invoices_df = generate_invoices(po_df)

    # 4. Goods Receipts
    print("  [4/6] Goods Receipts …")
    goods_df = generate_goods_receipts(po_df)

    # 5. Payments
    print("  [5/6] Payments …")
    payments_df = generate_payments(invoices_df)

    # 6. Event Log
    print("  [6/6] Event Log …")
    events_df = generate_event_log(po_df)

    # Save CSVs
    print("  Saving CSV files …")
    save_csv(suppliers_df, "suppliers")
    save_csv(po_df, "purchase_orders")
    save_csv(invoices_df, "invoices")
    save_csv(goods_df, "goods_receipts")
    save_csv(payments_df, "payments")
    save_csv(events_df, "event_log")

    # Create DB and save
    print("  Writing to SQLite database …")
    engine = create_engine(DB_URL, echo=False)
    Base.metadata.create_all(engine)
    save_to_db(engine, suppliers_df, po_df, invoices_df, goods_df, payments_df, events_df)

    # Summary
    print_summary(suppliers_df, po_df, invoices_df, goods_df, payments_df, events_df)


if __name__ == "__main__":
    main()
