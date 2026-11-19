"""
Shared test fixtures and in-memory mock datasets for ProcureLens test suite.
"""

import os
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Setup path so tests can import from app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.models import Base, Supplier, PurchaseOrder, Invoice, Payment, GoodsReceipt, EventLog
from app.backend.api import app
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def test_engine():
    """In-memory SQLite database engine for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def test_session(test_engine):
    """Database session for test execution."""
    SessionLocal = sessionmaker(bind=test_engine)
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def sample_suppliers_df():
    """Mock suppliers DataFrame."""
    return pd.DataFrame([
        {"Supplier_ID": 1, "Supplier_Name": "Acme Industrial", "Supplier_Category": "Raw Materials", "Location": "New York, USA", "Supplier_Rating": 4.8},
        {"Supplier_ID": 2, "Supplier_Name": "Apex Technologies", "Supplier_Category": "IT Services", "Location": "San Francisco, USA", "Supplier_Rating": 4.2},
        {"Supplier_ID": 3, "Supplier_Name": "Pinnacle Logistics", "Supplier_Category": "Logistics", "Location": "Chicago, USA", "Supplier_Rating": 3.1},
        {"Supplier_ID": 4, "Supplier_Name": "Vanguard MRO", "Supplier_Category": "MRO Supplies", "Location": "Dallas, USA", "Supplier_Rating": 2.2},
        {"Supplier_ID": 5, "Supplier_Name": "Sterling Pack", "Supplier_Category": "Packaging", "Location": "Atlanta, USA", "Supplier_Rating": 3.9},
    ])


@pytest.fixture
def sample_purchase_orders_df():
    """Mock purchase orders DataFrame."""
    now = datetime(2024, 1, 15)
    return pd.DataFrame([
        {"PO_ID": 101, "Supplier_ID": 1, "Department": "Operations", "Purchase_Amount": 15000.0, "Creation_Date": now, "Approval_Date": now + timedelta(days=2), "PO_Status": "Approved"},
        {"PO_ID": 102, "Supplier_ID": 2, "Department": "IT", "Purchase_Amount": 45000.0, "Creation_Date": now + timedelta(days=1), "Approval_Date": now + timedelta(days=4), "PO_Status": "Approved"},
        {"PO_ID": 103, "Supplier_ID": 3, "Department": "Procurement", "Purchase_Amount": 8500.0, "Creation_Date": now + timedelta(days=2), "Approval_Date": now + timedelta(days=3), "PO_Status": "Approved"},
        {"PO_ID": 104, "Supplier_ID": 4, "Department": "Operations", "Purchase_Amount": 120000.0, "Creation_Date": now + timedelta(days=3), "Approval_Date": now + timedelta(days=12), "PO_Status": "Approved"},
        {"PO_ID": 105, "Supplier_ID": 5, "Department": "Marketing", "Purchase_Amount": 6200.0, "Creation_Date": now + timedelta(days=4), "Approval_Date": None, "PO_Status": "Pending"},
    ])


@pytest.fixture
def sample_invoices_df():
    """Mock invoices DataFrame."""
    now = datetime(2024, 1, 20)
    return pd.DataFrame([
        {"Invoice_ID": 201, "PO_ID": 101, "Invoice_Amount": 15000.0, "Invoice_Date": now, "Due_Date": now + timedelta(days=30), "Invoice_Status": "Paid"},
        {"Invoice_ID": 202, "PO_ID": 102, "Invoice_Amount": 45000.0, "Invoice_Date": now + timedelta(days=2), "Due_Date": now + timedelta(days=32), "Invoice_Status": "Paid"},
        {"Invoice_ID": 203, "PO_ID": 103, "Invoice_Amount": 8500.0, "Invoice_Date": now + timedelta(days=3), "Due_Date": now + timedelta(days=33), "Invoice_Status": "Overdue"},
        {"Invoice_ID": 204, "PO_ID": 104, "Invoice_Amount": 120000.0, "Invoice_Date": now + timedelta(days=4), "Due_Date": now + timedelta(days=34), "Invoice_Status": "Paid"},
        {"Invoice_ID": 205, "PO_ID": 104, "Invoice_Amount": 120000.0, "Invoice_Date": now + timedelta(days=6), "Due_Date": now + timedelta(days=36), "Invoice_Status": "Disputed"},  # Duplicate!
    ])


@pytest.fixture
def sample_payments_df():
    """Mock payments DataFrame."""
    due = datetime(2024, 2, 20)
    return pd.DataFrame([
        {"Payment_ID": 301, "Invoice_ID": 201, "Payment_Date": due - timedelta(days=2), "Payment_Amount": 15000.0, "Payment_Status": "Completed"},
        {"Payment_ID": 302, "Invoice_ID": 202, "Payment_Date": due + timedelta(days=5), "Payment_Amount": 45000.0, "Payment_Status": "Completed"},  # Late!
        {"Payment_ID": 303, "Invoice_ID": 204, "Payment_Date": due + timedelta(days=15), "Payment_Amount": 120000.0, "Payment_Status": "Completed"},  # Very Late!
    ])


@pytest.fixture
def sample_event_log_df():
    """Mock 8-step P2P process mining event log DataFrame with variations."""
    t0 = datetime(2024, 1, 1, 10, 0, 0)
    events = []
    # Case 1: Standard conforming case
    std_steps = [
        "Purchase Requisition Created", "Purchase Order Generated", "Purchase Order Approved",
        "Goods Received", "Invoice Received", "Invoice Validated", "Invoice Approved", "Payment Executed"
    ]
    for i, act in enumerate(std_steps):
        events.append({
            "Event_ID": len(events) + 1,
            "Case_ID": "CASE-001",
            "Activity": act,
            "Timestamp": t0 + timedelta(days=i*2),
            "Resource": "User-A",
            "Department": "Operations"
        })

    # Case 2: Conforming case
    for i, act in enumerate(std_steps):
        events.append({
            "Event_ID": len(events) + 1,
            "Case_ID": "CASE-002",
            "Activity": act,
            "Timestamp": t0 + timedelta(days=i*3),
            "Resource": "User-B",
            "Department": "IT"
        })

    # Case 3: Approval rework loop (Purchase Order Approved twice)
    rework_steps = [
        "Purchase Requisition Created", "Purchase Order Generated", "Purchase Order Approved",
        "Purchase Order Approved",  # Repeated
        "Goods Received", "Invoice Received", "Invoice Validated", "Invoice Approved", "Payment Executed"
    ]
    for i, act in enumerate(rework_steps):
        events.append({
            "Event_ID": len(events) + 1,
            "Case_ID": "CASE-003",
            "Activity": act,
            "Timestamp": t0 + timedelta(days=i*2),
            "Resource": "User-C",
            "Department": "Procurement"
        })

    # Case 4: Skipped step (Missing Goods Received)
    skipped_steps = [
        "Purchase Requisition Created", "Purchase Order Generated", "Purchase Order Approved",
        "Invoice Received", "Invoice Validated", "Invoice Approved", "Payment Executed"
    ]
    for i, act in enumerate(skipped_steps):
        events.append({
            "Event_ID": len(events) + 1,
            "Case_ID": "CASE-004",
            "Activity": act,
            "Timestamp": t0 + timedelta(days=i*2),
            "Resource": "User-D",
            "Department": "Finance"
        })

    return pd.DataFrame(events)


@pytest.fixture
def api_client():
    """FastAPI TestClient fixture."""
    return TestClient(app)
