"""
ProcureLens Feature Engineering Pipeline for Payment Delay Prediction.

Engineers leakage-safe features to predict whether an invoice will experience late payment.
CRITICAL INTEGRITY RULE: No future payment details (Payment_Date, Payment_Amount, Payment_Status)
are ever used as input features. Only data available before payment execution is leveraged.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger("procurelens.ml.feature_engineering")


class FeatureEngineer:
    """Constructs leakage-safe training and inference datasets from procurement tables."""

    def __init__(self):
        self.dept_encoder = LabelEncoder()
        self.cat_encoder = LabelEncoder()
        self.is_fitted = False
        self.feature_names = [
            "purchase_amount",
            "supplier_rating",
            "approval_duration_days",
            "invoice_processing_days",
            "department_encoded",
            "supplier_category_encoded",
            "num_activities",
            "has_rework",
            "month_of_year"
        ]

    def build_features(
        self,
        po_df: pd.DataFrame,
        invoices_df: pd.DataFrame,
        payments_df: pd.DataFrame,
        event_log_df: pd.DataFrame,
        suppliers_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merges procurement tables and derives predictive features.
        
        Target column: 'is_delayed' (1 if Payment_Date > Due_Date, else 0).
        """
        # 1. Merge PO with Suppliers if not already joined
        merged = po_df.copy()
        if "Supplier_Rating" not in merged.columns or "Supplier_Category" not in merged.columns:
            cols_to_add = [c for c in ["Supplier_ID", "Supplier_Category", "Supplier_Rating"] if c in suppliers_df.columns]
            merged = pd.merge(
                merged,
                suppliers_df[cols_to_add].drop_duplicates(subset=["Supplier_ID"]),
                on="Supplier_ID",
                how="left",
                suffixes=("", "_sup")
            )
        # If suffixes were created, coalesce
        if "Supplier_Category_sup" in merged.columns and "Supplier_Category" not in merged.columns:
            merged["Supplier_Category"] = merged["Supplier_Category_sup"]
        elif "Supplier_Category_x" in merged.columns:
            merged["Supplier_Category"] = merged["Supplier_Category_x"]

        if "Supplier_Rating_sup" in merged.columns and "Supplier_Rating" not in merged.columns:
            merged["Supplier_Rating"] = merged["Supplier_Rating_sup"]
        elif "Supplier_Rating_x" in merged.columns:
            merged["Supplier_Rating"] = merged["Supplier_Rating_x"]

        # 2. Merge with Invoices (one invoice per PO for feature modeling, or explode)
        merged = pd.merge(
            merged,
            invoices_df[["Invoice_ID", "PO_ID", "Invoice_Amount", "Invoice_Date", "Due_Date"]],
            on="PO_ID",
            how="inner"
        )

        # 3. Derive Target: 'is_delayed'
        # Join payments to compute label
        if not payments_df.empty:
            merged = pd.merge(
                merged,
                payments_df[["Invoice_ID", "Payment_Date", "Payment_Status"]],
                on="Invoice_ID",
                how="left"
            )
            merged["Payment_Date"] = pd.to_datetime(merged["Payment_Date"])
            merged["Due_Date"] = pd.to_datetime(merged["Due_Date"])
            # Target is 1 if paid past due date or still overdue
            merged["is_delayed"] = (
                (merged["Payment_Date"] > merged["Due_Date"]) |
                (merged["Payment_Date"].isna() & (merged["Due_Date"] < pd.Timestamp.now()))
            ).astype(int)
        else:
            merged["is_delayed"] = 0

        # 4. Feature: Approval Duration in days
        merged["Creation_Date"] = pd.to_datetime(merged["Creation_Date"])
        merged["Approval_Date"] = pd.to_datetime(merged["Approval_Date"])
        merged["Invoice_Date"] = pd.to_datetime(merged["Invoice_Date"])

        approval_delta = (merged["Approval_Date"] - merged["Creation_Date"]).dt.total_seconds() / 86400.0
        merged["approval_duration_days"] = approval_delta.fillna(1.0).clip(lower=0.0)

        # 5. Feature: Invoice Processing Duration (Invoice Date minus PO Creation Date)
        inv_delta = (merged["Invoice_Date"] - merged["Creation_Date"]).dt.total_seconds() / 86400.0
        merged["invoice_processing_days"] = inv_delta.fillna(3.0).clip(lower=0.0)

        # 6. Feature: Activity Counts & Rework from Event Log
        if not event_log_df.empty:
            evt_copy = event_log_df.copy()
            evt_copy["Case_ID"] = evt_copy["Case_ID"].astype(str)
            merged["Case_ID_str"] = "PO-" + merged["PO_ID"].astype(str)

            # Total events per case
            counts = evt_copy.groupby("Case_ID").size().reset_index(name="num_activities")
            merged = pd.merge(merged, counts, left_on="Case_ID_str", right_on="Case_ID", how="left")
            merged["num_activities"] = merged["num_activities"].fillna(8)

            # Rework flag (any activity repeated)
            act_rework = evt_copy.groupby(["Case_ID", "Activity"]).size().reset_index(name="reps")
            rework_cases = set(act_rework[act_rework["reps"] > 1]["Case_ID"])
            merged["has_rework"] = merged["Case_ID_str"].isin(rework_cases).astype(int)
        else:
            merged["num_activities"] = 8
            merged["has_rework"] = 0

        # 7. Feature: Month of year
        merged["month_of_year"] = merged["Creation_Date"].dt.month.fillna(1).astype(int)

        # 8. Categorical Encodings
        departments = merged["Department"].fillna("Operations").astype(str)
        categories = merged["Supplier_Category"].fillna("General").astype(str)

        if not self.is_fitted:
            self.dept_encoder.fit(departments)
            self.cat_encoder.fit(categories)
            self.is_fitted = True

        # Transform with fallback for unseen classes
        def safe_transform(encoder, series):
            known = set(encoder.classes_)
            safe_series = series.apply(lambda x: x if x in known else encoder.classes_[0])
            return encoder.transform(safe_series)

        merged["department_encoded"] = safe_transform(self.dept_encoder, departments)
        merged["supplier_category_encoded"] = safe_transform(self.cat_encoder, categories)

        # Rename / map feature columns
        merged["purchase_amount"] = merged["Purchase_Amount"].astype(float)
        merged["supplier_rating"] = merged["Supplier_Rating"].astype(float).fillna(3.0)

        # Clean nulls
        for f in self.feature_names:
            if f in merged.columns:
                merged[f] = merged[f].fillna(0)

        cols_to_keep = ["PO_ID", "Invoice_ID", "Creation_Date"] + self.feature_names + ["is_delayed"]
        return merged[[c for c in cols_to_keep if c in merged.columns]].copy()

    def get_time_split(
        self,
        features_df: pd.DataFrame,
        split_month: int = 4
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Time-aware train/test separation to prevent temporal data leakage:
        Trains on transactions up to split_month, tests on transactions thereafter.
        """
        df = features_df.sort_values(by="Creation_Date").reset_index(drop=True)

        train_mask = df["Creation_Date"].dt.month <= split_month
        test_mask = df["Creation_Date"].dt.month > split_month

        train_df = df[train_mask]
        test_df = df[test_mask]

        # Fallback if split yields empty set
        if len(test_df) == 0:
            split_idx = int(len(df) * 0.75)
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]

        X_train = train_df[self.feature_names]
        y_train = train_df["is_delayed"]
        X_test = test_df[self.feature_names]
        y_test = test_df["is_delayed"]

        logger.info(
            f"Time split complete: Train={len(X_train)} samples (delay rate: {y_train.mean():.1%}), "
            f"Test={len(X_test)} samples (delay rate: {y_test.mean():.1%})"
        )
        return X_train, X_test, y_train, y_test
