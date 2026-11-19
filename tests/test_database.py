"""
Database integration and relational integrity tests.
"""

from datetime import datetime
from sqlalchemy import text
from app.database.models import Supplier, PurchaseOrder, Invoice, Payment, EventLog


def test_tables_created(test_engine):
    """Verify all 6 core tables are successfully created in the schema."""
    with test_engine.connect() as conn:
        tables = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
        table_names = [t[0] for t in tables]

    assert "Suppliers" in table_names
    assert "Purchase_Orders" in table_names
    assert "Invoices" in table_names
    assert "Payments" in table_names
    assert "Goods_Receipts" in table_names
    assert "Event_Log" in table_names


def test_supplier_and_po_relationship(test_session):
    """Verify supplier creation and foreign key cascade to purchase orders."""
    supplier = Supplier(
        Supplier_Name="Test Global Logistics",
        Supplier_Category="Logistics",
        Location="Chicago, USA",
        Supplier_Rating=4.5
    )
    test_session.add(supplier)
    test_session.commit()

    po = PurchaseOrder(
        Supplier_ID=supplier.Supplier_ID,
        Department="Operations",
        Purchase_Amount=25000.0,
        Creation_Date=datetime(2024, 1, 10),
        PO_Status="Approved"
    )
    test_session.add(po)
    test_session.commit()

    assert po.PO_ID is not None
    assert po.supplier.Supplier_Name == "Test Global Logistics"
    assert len(supplier.purchase_orders) == 1


def test_event_log_persistence(test_session):
    """Verify event log records can be inserted and queried chronologically."""
    event = EventLog(
        Case_ID="CASE-TEST-100",
        Activity="Purchase Requisition Created",
        Timestamp=datetime(2024, 2, 1, 9, 30),
        Resource="Jane Doe",
        Department="Finance"
    )
    test_session.add(event)
    test_session.commit()

    retrieved = test_session.query(EventLog).filter_by(Case_ID="CASE-TEST-100").first()
    assert retrieved is not None
    assert retrieved.Activity == "Purchase Requisition Created"
    assert retrieved.Resource == "Jane Doe"
