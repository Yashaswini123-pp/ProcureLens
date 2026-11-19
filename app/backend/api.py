"""
ProcureLens Enterprise REST API.

FastAPI application providing process intelligence, bottleneck analysis,
anomaly detection, and machine learning risk scoring for procurement operations.
"""

import io
import logging
import os
import sys
from typing import Dict, List, Optional
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.analytics.anomaly_detection import AnomalyDetector
from app.analytics.kpi_calculator import KPICalculator
from app.backend.schemas import (
    AnomalySchema,
    BottleneckSchema,
    ComplianceResponse,
    ETLRunResponse,
    HealthResponse,
    KPISummaryResponse,
    PaginatedPurchaseOrders,
    PredictionResponse,
    ProcessVariantSchema,
    SupplierSchema,
)
from app.celonis.config import CelonisConfig
from app.database.connection import get_engine, init_db
from app.database.queries import (
    get_all_event_logs,
    get_all_invoices,
    get_all_payments,
    get_all_purchase_orders,
    get_all_suppliers,
)
from app.etl.pipeline import ETLPipeline
from app.ml.feature_engineering import FeatureEngineer
from app.ml.predictor import PaymentDelayPredictor
from app.process_mining.engine import ProcessMiningEngine

logger = logging.getLogger("procurelens.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Initialize FastAPI App
app = FastAPI(
    title="ProcureLens API",
    description="Enterprise Purchase-to-Pay Process Intelligence & Bottleneck Detection Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend dashboard and external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared singletons
_engine = None
_pm_engine: Optional[ProcessMiningEngine] = None
_anomaly_detector = AnomalyDetector()
_predictor: Optional[PaymentDelayPredictor] = None


def get_db_engine():
    global _engine
    if _engine is None:
        _engine = get_engine()
    return _engine


def get_pm_engine_instance() -> ProcessMiningEngine:
    global _pm_engine
    if _pm_engine is None or _pm_engine.df.empty:
        engine = get_db_engine()
        try:
            df_events = get_all_event_logs(engine)
            _pm_engine = ProcessMiningEngine(df_events)
        except Exception as e:
            logger.warning(f"Could not load event log into PM engine: {str(e)}")
            _pm_engine = ProcessMiningEngine()
    return _pm_engine


def get_predictor_instance() -> PaymentDelayPredictor:
    global _predictor
    if _predictor is None:
        _predictor = PaymentDelayPredictor()
    return _predictor


@app.on_event("startup")
def startup_event():
    """Verify database existence and initialize connections on startup."""
    try:
        engine = get_db_engine()
        # Verify tables exist
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection successfully initialized.")
    except Exception as e:
        logger.warning(f"Database check failed during startup: {str(e)}")


# -----------------------------------------------------------------------------
# 1. Health Endpoint
# -----------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """System health check and connection diagnostic probe."""
    db_ok = False
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    celonis_cfg = CelonisConfig()

    return HealthResponse(
        status="healthy" if db_ok else "degraded",
        database_connected=db_ok,
        version="1.0.0",
        celonis_configured=celonis_cfg.is_configured()
    )


# -----------------------------------------------------------------------------
# 2. Executive Dashboard Summary
# -----------------------------------------------------------------------------
@app.get("/dashboard/summary", response_model=KPISummaryResponse, tags=["Dashboard"])
def get_dashboard_summary():
    """Returns high-level procurement KPIs, monthly trends, and departmental metrics."""
    try:
        engine = get_db_engine()
        calc = KPICalculator(engine)
        return calc.get_full_summary()
    except Exception as e:
        logger.error(f"Failed to calculate dashboard summary: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error computing KPI summary: {str(e)}"
        )


# -----------------------------------------------------------------------------
# 3. Purchase Orders (Paginated & Filterable)
# -----------------------------------------------------------------------------
@app.get("/purchase-orders", response_model=PaginatedPurchaseOrders, tags=["Procurement"])
def get_purchase_orders(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(25, ge=1, le=200, description="Items per page"),
    department: Optional[str] = Query(None, description="Filter by department"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by PO status"),
    supplier_id: Optional[int] = Query(None, description="Filter by supplier ID")
):
    """Retrieves paginated purchase orders with optional departmental and status filters."""
    engine = get_db_engine()
    where_clauses = ["1=1"]
    params = {}

    if department:
        where_clauses.append("Department = :dept")
        params["dept"] = department
    if status_filter:
        where_clauses.append("PO_Status = :stat")
        params["stat"] = status_filter
    if supplier_id:
        where_clauses.append("Supplier_ID = :sup")
        params["sup"] = supplier_id

    where_sql = " AND ".join(where_clauses)

    try:
        with engine.connect() as conn:
            # Count total matching
            count_sql = f"SELECT COUNT(*) FROM Purchase_Orders WHERE {where_sql}"
            total_items = conn.execute(text(count_sql), params).scalar() or 0

            # Fetch page
            offset = (page - 1) * page_size
            data_sql = f"""
            SELECT PO_ID, Supplier_ID, Department, Purchase_Amount,
                   Creation_Date, Approval_Date, PO_Status
            FROM Purchase_Orders
            WHERE {where_sql}
            ORDER BY Creation_Date DESC
            LIMIT :limit OFFSET :offset
            """
            params["limit"] = page_size
            params["offset"] = offset

            df = pd.read_sql_query(text(data_sql), conn, params=params)
            # Format datetime
            items = df.to_dict(orient="records")
            for item in items:
                for k, v in item.items():
                    if isinstance(v, pd.Timestamp):
                        item[k] = v.isoformat()

            total_pages = (total_items + page_size - 1) // page_size if total_items > 0 else 1

            return PaginatedPurchaseOrders(
                items=items,
                total=total_items,
                page=page,
                page_size=page_size,
                total_pages=total_pages
            )
    except Exception as e:
        logger.error(f"Error retrieving purchase orders: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# 4. Suppliers
# -----------------------------------------------------------------------------
@app.get("/suppliers", response_model=List[SupplierSchema], tags=["Procurement"])
def get_suppliers(limit: int = Query(50, ge=1, le=500)):
    """Retrieves list of vendors with spend and performance metrics."""
    try:
        engine = get_db_engine()
        calc = KPICalculator(engine)
        scores = calc.supplier_performance_scores(top_n=limit)
        return scores
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# 5. Process Variants
# -----------------------------------------------------------------------------
@app.get("/process/variants", response_model=List[ProcessVariantSchema], tags=["Process Mining"])
def get_process_variants():
    """Discovers and ranks all distinct Purchase-to-Pay execution variants."""
    try:
        pm = get_pm_engine_instance()
        variants = pm.discover_process_variants()
        return variants[:20]  # Return top 20 variants
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# 6. Process Bottlenecks
# -----------------------------------------------------------------------------
@app.get("/process/bottlenecks", response_model=List[BottleneckSchema], tags=["Process Mining"])
def get_process_bottlenecks():
    """Calculates average waiting time between transitions to identify operational bottlenecks."""
    try:
        pm = get_pm_engine_instance()
        bottlenecks = pm.identify_bottlenecks()
        return bottlenecks
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# 7. Process Compliance
# -----------------------------------------------------------------------------
@app.get("/process/compliance", response_model=ComplianceResponse, tags=["Process Mining"])
def get_process_compliance():
    """Measures conformance against the standard 8-step P2P happy path."""
    try:
        pm = get_pm_engine_instance()
        comp = pm.compare_normal_abnormal()
        return ComplianceResponse(
            conformance_rate=comp["conformance_rate"],
            total_cases=comp["total_cases"],
            normal_cases_count=comp["normal_cases_count"],
            abnormal_cases_count=comp["abnormal_cases_count"],
            standard_process=pm.STANDARD_P2P_PROCESS,
            normal_examples=[str(x) for x in comp["normal_examples"]],
            abnormal_examples=[str(x) for x in comp["abnormal_examples"]]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# 8. Anomaly Detection
# -----------------------------------------------------------------------------
@app.get("/anomalies", response_model=List[AnomalySchema], tags=["Risk & Audit"])
def get_anomalies(severity: Optional[str] = Query(None, description="Filter: HIGH, MEDIUM, LOW")):
    """Scans all procurement records to flag duplicate invoices, missing steps, and price variances."""
    try:
        engine = get_db_engine()
        datasets = {
            "purchase_orders": get_all_purchase_orders(engine),
            "invoices": get_all_invoices(engine),
            "payments": get_all_payments(engine),
            "event_log": get_all_event_logs(engine),
            "suppliers": get_all_suppliers(engine)
        }
        detector = AnomalyDetector()
        findings = detector.run_all_checks(datasets)

        if severity:
            findings = [f for f in findings if f["severity"] == severity.upper()]

        return findings
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# 9. Predictive Machine Learning Insights
# -----------------------------------------------------------------------------
@app.get("/predictions", response_model=PredictionResponse, tags=["Predictive ML"])
def get_predictions(limit: int = Query(50, ge=1, le=500)):
    """Computes late payment probability scores using trained ML model."""
    try:
        engine = get_db_engine()
        po_df = get_all_purchase_orders(engine)
        inv_df = get_all_invoices(engine)
        pay_df = get_all_payments(engine)
        evt_df = get_all_event_logs(engine)
        sup_df = get_all_suppliers(engine)

        fe = FeatureEngineer()
        features_df = fe.build_features(po_df, inv_df, pay_df, evt_df, sup_df)

        predictor = get_predictor_instance()
        scored_df = predictor.predict_batch(features_df.head(limit))

        # Risk breakdown
        risk_dist = scored_df["risk_category"].value_counts().to_dict()
        high_risk_cnt = int(risk_dist.get("HIGH", 0))

        predictions = []
        for _, row in scored_df.iterrows():
            prob = float(row.get("delay_probability", 0.0))
            cat = str(row.get("risk_category", "LOW"))
            amt = float(row.get("purchase_amount", 0.0))
            po_id = int(row.get("PO_ID", 0))

            rec = (
                "Expedite approval and notify Accounts Payable for early disbursement."
                if cat == "HIGH"
                else ("Standard working capital processing." if cat == "LOW" else "Monitor invoice validation lead time.")
            )

            predictions.append({
                "po_id": po_id,
                "purchase_amount": amt,
                "department": str(row.get("Department", "General")),
                "delay_probability": prob,
                "risk_category": cat,
                "recommendation": rec
            })

        return PredictionResponse(
            total_scored=len(predictions),
            risk_distribution={k: int(v) for k, v in risk_dist.items()},
            high_risk_count=high_risk_cnt,
            predictions=predictions,
            model_notes=predictor.get_model_limitations()
        )
    except Exception as e:
        logger.error(f"Prediction calculation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# 10. Data Ingestion & ETL Endpoints
# -----------------------------------------------------------------------------
@app.post("/upload", response_model=Dict, tags=["Data Pipeline"])
async def upload_file(
    file: UploadFile = File(...),
    target_table: str = Query("purchase_orders", description="Target database table")
):
    """Uploads a CSV or Excel procurement dataset and streams it into the ETL pipeline."""
    contents = await file.read()
    pipeline = ETLPipeline(get_db_engine())
    result = pipeline.run_from_upload(contents, file.filename, target_table)

    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("error") or result.get("message"))

    return result


@app.post("/etl/run", response_model=ETLRunResponse, tags=["Data Pipeline"])
def trigger_etl():
    """Executes the complete raw dataset ETL pipeline."""
    try:
        pipeline = ETLPipeline(get_db_engine())
        report = pipeline.run()
        if report.get("status") == "error":
            raise HTTPException(status_code=500, detail=report.get("message"))
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
