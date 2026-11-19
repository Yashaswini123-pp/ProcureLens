"""
Unit tests for ETL pipeline modules: extract, transform, load.
"""

import os
import pandas as pd
import pytest
from app.etl.extract import DataExtractor
from app.etl.transform import DataTransformer
from app.etl.load import DataLoader


def test_missing_values_handling():
    """Verify missing numeric and text values are imputed cleanly."""
    transformer = DataTransformer()
    raw_df = pd.DataFrame([
        {"PO_ID": 1, "Purchase_Amount": None, "Department": None},
        {"PO_ID": 2, "Purchase_Amount": 500.0, "Department": "Finance"}
    ])
    clean_df = transformer.handle_missing_values(raw_df, "test_table")
    assert clean_df["Purchase_Amount"].iloc[0] == 0.0
    assert clean_df["Department"].iloc[0] == "Unknown"
    assert clean_df["Purchase_Amount"].iloc[1] == 500.0


def test_remove_duplicates():
    """Verify duplicate records are detected and pruned."""
    transformer = DataTransformer()
    raw_df = pd.DataFrame([
        {"Supplier_ID": 1, "Supplier_Name": "Vendor A"},
        {"Supplier_ID": 1, "Supplier_Name": "Vendor A"},
        {"Supplier_ID": 2, "Supplier_Name": "Vendor B"}
    ])
    clean_df, removed = transformer.remove_duplicates(raw_df, subset=["Supplier_ID"])
    assert len(clean_df) == 2
    assert removed == 1


def test_timestamp_normalization():
    """Verify varied date strings are parsed into valid datetime objects."""
    transformer = DataTransformer()
    raw_df = pd.DataFrame([
        {"Date_Str": "2024-01-15 10:30:00"},
        {"Date_Str": "2024/02/20"},
        {"Date_Str": "invalid_date"}
    ])
    norm_df = transformer.normalize_timestamps(raw_df, ["Date_Str"])
    assert pd.api.types.is_datetime64_any_dtype(norm_df["Date_Str"])
    assert pd.notnull(norm_df["Date_Str"].iloc[0])
    assert pd.isna(norm_df["Date_Str"].iloc[2])  # Coerced to NaT


def test_standardize_activity_names():
    """Verify non-standard case activity strings are canonicalized."""
    transformer = DataTransformer()
    df = pd.DataFrame([
        {"Activity": "purchase order approved"},
        {"Activity": "GOODS RECEIVED"},
        {"Activity": "Custom Step"}
    ])
    std_df = transformer.standardize_activity_names(df)
    assert std_df["Activity"].iloc[0] == "Purchase Order Approved"
    assert std_df["Activity"].iloc[1] == "Goods Received"
    assert std_df["Activity"].iloc[2] == "Custom Step"


def test_validate_foreign_keys():
    """Verify child records with orphaned foreign keys are flagged."""
    transformer = DataTransformer()
    parent = pd.DataFrame([{"Supplier_ID": 10}, {"Supplier_ID": 20}])
    child = pd.DataFrame([
        {"PO_ID": 1, "Supplier_ID": 10},
        {"PO_ID": 2, "Supplier_ID": 99}  # Orphaned
    ])
    valid, orphaned = transformer.validate_foreign_keys(child, parent, "Supplier_ID", "Supplier_ID")
    assert len(valid) == 1
    assert len(orphaned) == 1
    assert orphaned.iloc[0]["PO_ID"] == 2


def test_data_loader_to_sqlite(test_engine):
    """Verify DataLoader inserts records properly into SQLite."""
    loader = DataLoader(test_engine)
    suppliers = pd.DataFrame([
        {"Supplier_ID": 901, "Supplier_Name": "ETL Supplier 1", "Supplier_Category": "IT", "Location": "Boston, USA", "Supplier_Rating": 4.0},
        {"Supplier_ID": 902, "Supplier_Name": "ETL Supplier 2", "Supplier_Category": "HR", "Location": "Austin, USA", "Supplier_Rating": 4.5}
    ])
    count = loader.load_to_database(suppliers, "Suppliers", if_exists="append")
    assert count == 2
