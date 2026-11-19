"""
ProcureLens API Pydantic Schemas.

Defines typed request and response contracts for all REST endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    database_connected: bool
    version: str
    celonis_configured: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MonthlyTrend(BaseModel):
    Month: str
    PO_Count: int
    Total_Spend: float
    Avg_Amount: float


class DepartmentStat(BaseModel):
    Department: str
    PO_Count: int
    Total_Spend: float
    Avg_Amount: float
    Avg_Approval_Days: Optional[float] = None


class KPISummaryResponse(BaseModel):
    total_purchase_orders: int
    total_procurement_value: float
    avg_processing_time_days: float
    delayed_payments: int
    delayed_payment_pct: float
    approval_delay_avg_days: float
    compliance_rate: float
    monthly_trends: List[Dict[str, Any]]
    department_comparison: List[Dict[str, Any]]
    status_distribution: Dict[str, int]


class PurchaseOrderSchema(BaseModel):
    PO_ID: int
    Supplier_ID: int
    Department: str
    Purchase_Amount: float
    Creation_Date: datetime
    Approval_Date: Optional[datetime] = None
    PO_Status: str


class PaginatedPurchaseOrders(BaseModel):
    items: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int
    total_pages: int


class SupplierSchema(BaseModel):
    Supplier_ID: int
    Supplier_Name: str
    Supplier_Category: str
    Location: str
    Supplier_Rating: float
    Total_Orders: Optional[int] = 0
    Total_Spend: Optional[float] = 0.0


class ProcessVariantSchema(BaseModel):
    variant_id: int
    variant: List[str]
    variant_str: str
    count: int
    percentage: float
    is_standard: bool


class BottleneckSchema(BaseModel):
    from_activity: str
    to_activity: str
    transition: str
    case_count: int
    avg_duration_days: float
    median_duration_days: float
    max_duration_days: float


class ComplianceResponse(BaseModel):
    conformance_rate: float
    total_cases: int
    normal_cases_count: int
    abnormal_cases_count: int
    standard_process: List[str]
    normal_examples: List[str]
    abnormal_examples: List[str]


class AnomalySchema(BaseModel):
    type: str
    record_id: str
    case_id: str
    description: str
    severity: str
    details: str


class PredictionItem(BaseModel):
    po_id: int
    purchase_amount: float
    department: str
    delay_probability: float
    risk_category: str
    recommendation: str


class PredictionResponse(BaseModel):
    total_scored: int
    risk_distribution: Dict[str, int]
    high_risk_count: int
    predictions: List[Dict[str, Any]]
    model_notes: List[str]


class ETLRunResponse(BaseModel):
    status: str
    duration_seconds: float
    total_records_loaded: int
    table_breakdown: Dict[str, int]
    transformation_audit: Dict[str, Any]
