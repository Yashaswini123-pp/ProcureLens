"""
SQLAlchemy ORM models for ProcureLens enterprise procurement analytics.

Defines all tables: Suppliers, Purchase_Orders, Invoices, Payments,
Goods_Receipts, and Event_Log with proper relationships, indexes,
and foreign key constraints.
"""

from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    DateTime,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Supplier(Base):
    """Supplier master data."""

    __tablename__ = "Suppliers"

    Supplier_ID = Column(Integer, primary_key=True, autoincrement=True)
    Supplier_Name = Column(String(200), nullable=False)
    Supplier_Category = Column(String(100), nullable=False)
    Location = Column(String(200), nullable=False)
    Supplier_Rating = Column(Float, nullable=False)

    # Relationships
    purchase_orders = relationship(
        "PurchaseOrder", back_populates="supplier", lazy="select"
    )

    def __repr__(self) -> str:
        return (
            f"<Supplier(Supplier_ID={self.Supplier_ID}, "
            f"Supplier_Name='{self.Supplier_Name}')>"
        )


class PurchaseOrder(Base):
    """Purchase order header records."""

    __tablename__ = "Purchase_Orders"

    PO_ID = Column(Integer, primary_key=True, autoincrement=True)
    Supplier_ID = Column(
        Integer,
        ForeignKey("Suppliers.Supplier_ID", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    Department = Column(String(100), nullable=False, index=True)
    Purchase_Amount = Column(Float, nullable=False)
    Creation_Date = Column(DateTime, nullable=False)
    Approval_Date = Column(DateTime, nullable=True)
    PO_Status = Column(String(50), nullable=False, index=True)

    # Relationships
    supplier = relationship("Supplier", back_populates="purchase_orders")
    invoices = relationship(
        "Invoice", back_populates="purchase_order", lazy="select"
    )
    goods_receipts = relationship(
        "GoodsReceipt", back_populates="purchase_order", lazy="select"
    )

    __table_args__ = (
        Index("ix_po_creation_date", "Creation_Date"),
        Index("ix_po_status_dept", "PO_Status", "Department"),
    )

    def __repr__(self) -> str:
        return (
            f"<PurchaseOrder(PO_ID={self.PO_ID}, "
            f"Department='{self.Department}', "
            f"Purchase_Amount={self.Purchase_Amount})>"
        )


class Invoice(Base):
    """Invoice records linked to purchase orders."""

    __tablename__ = "Invoices"

    Invoice_ID = Column(Integer, primary_key=True, autoincrement=True)
    PO_ID = Column(
        Integer,
        ForeignKey("Purchase_Orders.PO_ID", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    Invoice_Amount = Column(Float, nullable=False)
    Invoice_Date = Column(DateTime, nullable=False)
    Due_Date = Column(DateTime, nullable=False)
    Invoice_Status = Column(String(50), nullable=False, index=True)

    # Relationships
    purchase_order = relationship("PurchaseOrder", back_populates="invoices")
    payments = relationship(
        "Payment", back_populates="invoice", lazy="select"
    )

    __table_args__ = (
        Index("ix_invoice_dates", "Invoice_Date", "Due_Date"),
    )

    def __repr__(self) -> str:
        return (
            f"<Invoice(Invoice_ID={self.Invoice_ID}, "
            f"PO_ID={self.PO_ID}, "
            f"Invoice_Amount={self.Invoice_Amount})>"
        )


class Payment(Base):
    """Payment records linked to invoices."""

    __tablename__ = "Payments"

    Payment_ID = Column(Integer, primary_key=True, autoincrement=True)
    Invoice_ID = Column(
        Integer,
        ForeignKey("Invoices.Invoice_ID", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    Payment_Date = Column(DateTime, nullable=False)
    Payment_Amount = Column(Float, nullable=False)
    Payment_Status = Column(String(50), nullable=False, index=True)

    # Relationships
    invoice = relationship("Invoice", back_populates="payments")

    def __repr__(self) -> str:
        return (
            f"<Payment(Payment_ID={self.Payment_ID}, "
            f"Invoice_ID={self.Invoice_ID}, "
            f"Payment_Amount={self.Payment_Amount})>"
        )


class GoodsReceipt(Base):
    """Goods receipt records linked to purchase orders."""

    __tablename__ = "Goods_Receipts"

    Receipt_ID = Column(Integer, primary_key=True, autoincrement=True)
    PO_ID = Column(
        Integer,
        ForeignKey("Purchase_Orders.PO_ID", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    Receipt_Date = Column(DateTime, nullable=False)
    Received_Quantity = Column(Integer, nullable=False)

    # Relationships
    purchase_order = relationship(
        "PurchaseOrder", back_populates="goods_receipts"
    )

    def __repr__(self) -> str:
        return (
            f"<GoodsReceipt(Receipt_ID={self.Receipt_ID}, "
            f"PO_ID={self.PO_ID}, "
            f"Received_Quantity={self.Received_Quantity})>"
        )


class EventLog(Base):
    """Process mining event log following XES conventions."""

    __tablename__ = "Event_Log"

    Event_ID = Column(Integer, primary_key=True, autoincrement=True)
    Case_ID = Column(String(50), nullable=False, index=True)
    Activity = Column(String(200), nullable=False, index=True)
    Timestamp = Column(DateTime, nullable=False, index=True)
    Resource = Column(String(100), nullable=False)
    Department = Column(String(100), nullable=False)

    __table_args__ = (
        Index("ix_event_case_timestamp", "Case_ID", "Timestamp"),
        Index("ix_event_activity_dept", "Activity", "Department"),
    )

    def __repr__(self) -> str:
        return (
            f"<EventLog(Event_ID={self.Event_ID}, "
            f"Case_ID='{self.Case_ID}', "
            f"Activity='{self.Activity}')>"
        )
