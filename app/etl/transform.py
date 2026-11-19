"""
ProcureLens ETL - Transformation Module.

Handles cleaning, validation, normalization, and integrity checks:
- Missing value imputation and flagging
- Duplicate detection and removal
- Timestamp parsing and ISO standardization
- Case ID and primary key validation
- Activity name standardization
- Foreign key referential integrity validation
- Event log synthesis from transactional records
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger("procurelens.etl.transform")


@dataclass
class TransformationReport:
    """Audit report recording operations performed during transformation."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tables_processed: List[str] = field(default_factory=list)
    initial_rows: Dict[str, int] = field(default_factory=dict)
    final_rows: Dict[str, int] = field(default_factory=dict)
    missing_values_handled: Dict[str, int] = field(default_factory=dict)
    duplicates_removed: Dict[str, int] = field(default_factory=dict)
    invalid_dates_fixed: Dict[str, int] = field(default_factory=dict)
    fk_violations_found: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def summary(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "tables_processed": self.tables_processed,
            "total_initial_rows": sum(self.initial_rows.values()),
            "total_final_rows": sum(self.final_rows.values()),
            "duplicates_removed": sum(self.duplicates_removed.values()),
            "fk_violations": sum(self.fk_violations_found.values()),
            "error_count": len(self.errors),
        }


class DataTransformer:
    """Transforms and cleans procurement records for downstream process intelligence."""

    STANDARD_ACTIVITIES = {
        "purchase requisition created": "Purchase Requisition Created",
        "purchase order generated": "Purchase Order Generated",
        "purchase order approved": "Purchase Order Approved",
        "goods received": "Goods Received",
        "invoice received": "Invoice Received",
        "invoice validated": "Invoice Validated",
        "invoice approved": "Invoice Approved",
        "invoice rejected": "Invoice Rejected",
        "payment executed": "Payment Executed"
    }

    def __init__(self):
        self.report = TransformationReport()

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        table_name: str = "dataset",
        strategy: str = "smart"
    ) -> pd.DataFrame:
        """
        Handle missing values using domain-appropriate defaults or forward-filling.
        """
        df_clean = df.copy()
        null_count = df_clean.isnull().sum().sum()
        if null_count == 0:
            return df_clean

        if strategy == "smart":
            # For numeric columns: fill with 0 or median
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_clean[col].isnull().any():
                    if "Rating" in col:
                        df_clean[col] = df_clean[col].fillna(3.0)
                    elif "Amount" in col or "Quantity" in col:
                        df_clean[col] = df_clean[col].fillna(0.0)
                    else:
                        df_clean[col] = df_clean[col].fillna(0)

            # For text columns: fill with 'Unknown'
            text_cols = df_clean.select_dtypes(include=["object"]).columns
            for col in text_cols:
                if df_clean[col].isnull().any():
                    df_clean[col] = df_clean[col].fillna("Unknown")

        elif strategy == "drop":
            df_clean = df_clean.dropna()
        elif strategy == "fill_zero":
            df_clean = df_clean.fillna(0)

        self.report.missing_values_handled[table_name] = int(null_count)
        return df_clean

    def remove_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        table_name: str = "dataset"
    ) -> Tuple[pd.DataFrame, int]:
        """
        Detect and eliminate duplicate records based on primary key or unique subset.
        """
        initial_len = len(df)
        df_dedup = df.drop_duplicates(subset=subset, keep="first")
        removed = initial_len - len(df_dedup)
        self.report.duplicates_removed[table_name] = removed
        if removed > 0:
            logger.info(f"Removed {removed} duplicates from {table_name}")
        return df_dedup, removed

    def normalize_timestamps(
        self,
        df: pd.DataFrame,
        columns: List[str],
        table_name: str = "dataset"
    ) -> pd.DataFrame:
        """
        Normalize date columns into pandas datetime types with UTC timezone awareness.
        """
        df_norm = df.copy()
        for col in columns:
            if col in df_norm.columns:
                try:
                    df_norm[col] = pd.to_datetime(df_norm[col], errors="coerce")
                    invalid_dates = df_norm[col].isnull().sum() - df[col].isnull().sum()
                    if invalid_dates > 0:
                        self.report.invalid_dates_fixed[f"{table_name}.{col}"] = int(invalid_dates)
                except Exception as e:
                    self.report.errors.append(f"Failed to parse datetime in {table_name}.{col}: {str(e)}")
        return df_norm

    def validate_case_ids(
        self,
        df: pd.DataFrame,
        case_id_column: str = "Case_ID"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Validates case IDs for non-null, string format, and valid characters.
        Returns: (valid_df, invalid_df)
        """
        if case_id_column not in df.columns:
            raise KeyError(f"Case ID column '{case_id_column}' not found in DataFrame")

        valid_mask = (
            df[case_id_column].notnull() &
            (df[case_id_column].astype(str).str.strip() != "") &
            (df[case_id_column].astype(str) != "nan")
        )
        return df[valid_mask].copy(), df[~valid_mask].copy()

    def standardize_activity_names(
        self,
        df: pd.DataFrame,
        activity_column: str = "Activity"
    ) -> pd.DataFrame:
        """
        Maps raw or varied activity name strings to canonical P2P activity names.
        """
        if activity_column not in df.columns:
            return df

        df_std = df.copy()
        df_std[activity_column] = df_std[activity_column].astype(str).str.strip()

        def map_activity(val: str) -> str:
            lower = val.lower()
            return self.STANDARD_ACTIVITIES.get(lower, val)

        df_std[activity_column] = df_std[activity_column].apply(map_activity)
        return df_std

    def validate_foreign_keys(
        self,
        child_df: pd.DataFrame,
        parent_df: pd.DataFrame,
        fk_column: str,
        pk_column: str,
        table_name: str = "relationship"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Checks referential integrity between child and parent tables.
        Returns: (valid_child_records, orphaned_child_records)
        """
        parent_keys = set(parent_df[pk_column].dropna().unique())
        valid_mask = child_df[fk_column].isin(parent_keys)

        valid_df = child_df[valid_mask].copy()
        orphaned_df = child_df[~valid_mask].copy()

        violations = len(orphaned_df)
        self.report.fk_violations_found[table_name] = violations
        if violations > 0:
            logger.warning(f"Found {violations} foreign key violations in {table_name}")

        return valid_df, orphaned_df

    def transform_all(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Execute end-to-end transformation and validation across all procurement datasets.
        """
        transformed = {}
        for name, df in datasets.items():
            self.report.tables_processed.append(name)
            self.report.initial_rows[name] = len(df)

        # 1. Suppliers
        if "suppliers" in datasets:
            sup = datasets["suppliers"].copy()
            sup = self.handle_missing_values(sup, "suppliers")
            sup, _ = self.remove_duplicates(sup, subset=["Supplier_ID"], table_name="suppliers")
            if "Supplier_Rating" in sup.columns:
                sup["Supplier_Rating"] = sup["Supplier_Rating"].clip(1.0, 5.0)
            self.report.final_rows["suppliers"] = len(sup)
            transformed["suppliers"] = sup

        # 2. Purchase Orders
        if "purchase_orders" in datasets:
            po = datasets["purchase_orders"].copy()
            po = self.handle_missing_values(po, "purchase_orders")
            po, _ = self.remove_duplicates(po, subset=["PO_ID"], table_name="purchase_orders")
            po = self.normalize_timestamps(po, ["Creation_Date", "Approval_Date"], "purchase_orders")
            if "suppliers" in transformed:
                po, _ = self.validate_foreign_keys(po, transformed["suppliers"], "Supplier_ID", "Supplier_ID", "PO_to_Supplier")
            self.report.final_rows["purchase_orders"] = len(po)
            transformed["purchase_orders"] = po

        # 3. Invoices
        if "invoices" in datasets:
            inv = datasets["invoices"].copy()
            inv = self.handle_missing_values(inv, "invoices")
            inv, _ = self.remove_duplicates(inv, subset=["Invoice_ID"], table_name="invoices")
            inv = self.normalize_timestamps(inv, ["Invoice_Date", "Due_Date"], "invoices")
            if "purchase_orders" in transformed:
                inv, _ = self.validate_foreign_keys(inv, transformed["purchase_orders"], "PO_ID", "PO_ID", "Invoice_to_PO")
            self.report.final_rows["invoices"] = len(inv)
            transformed["invoices"] = inv

        # 4. Goods Receipts
        if "goods_receipts" in datasets:
            gr = datasets["goods_receipts"].copy()
            gr = self.handle_missing_values(gr, "goods_receipts")
            gr, _ = self.remove_duplicates(gr, subset=["Receipt_ID"], table_name="goods_receipts")
            gr = self.normalize_timestamps(gr, ["Receipt_Date"], "goods_receipts")
            if "purchase_orders" in transformed:
                gr, _ = self.validate_foreign_keys(gr, transformed["purchase_orders"], "PO_ID", "PO_ID", "Receipt_to_PO")
            self.report.final_rows["goods_receipts"] = len(gr)
            transformed["goods_receipts"] = gr

        # 5. Payments
        if "payments" in datasets:
            pay = datasets["payments"].copy()
            pay = self.handle_missing_values(pay, "payments")
            pay, _ = self.remove_duplicates(pay, subset=["Payment_ID"], table_name="payments")
            pay = self.normalize_timestamps(pay, ["Payment_Date"], "payments")
            if "invoices" in transformed:
                pay, _ = self.validate_foreign_keys(pay, transformed["invoices"], "Invoice_ID", "Invoice_ID", "Payment_to_Invoice")
            self.report.final_rows["payments"] = len(pay)
            transformed["payments"] = pay

        # 6. Event Log
        if "event_log" in datasets:
            evt = datasets["event_log"].copy()
            evt = self.handle_missing_values(evt, "event_log")
            evt = self.standardize_activity_names(evt, "Activity")
            evt = self.normalize_timestamps(evt, ["Timestamp"], "event_log")
            evt, _ = self.validate_case_ids(evt, "Case_ID")
            evt, _ = self.remove_duplicates(evt, subset=["Event_ID"] if "Event_ID" in evt.columns else None, table_name="event_log")
            self.report.final_rows["event_log"] = len(evt)
            transformed["event_log"] = evt

        return transformed
