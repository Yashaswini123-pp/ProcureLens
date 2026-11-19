"""
ProcureLens Enterprise Process Intelligence Dashboard.

A high-performance executive dashboard for Purchase-to-Pay (P2P) process mining,
bottleneck detection, anomaly scanning, and predictive delay forecasting.

Run via:
    streamlit run app/frontend/dashboard.py
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup sys.path for direct module imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.analytics.anomaly_detection import AnomalyDetector
from app.analytics.kpi_calculator import KPICalculator
from app.celonis.config import CelonisConfig
from app.celonis.pql_queries import get_all_pql_queries
from app.database.connection import get_engine
from app.database.queries import (
    get_all_event_logs,
    get_all_invoices,
    get_all_payments,
    get_all_purchase_orders,
    get_all_suppliers,
)
from app.ml.feature_engineering import FeatureEngineer
from app.ml.model_training import ModelTrainer
from app.ml.predictor import PaymentDelayPredictor
from app.process_mining.engine import ProcessMiningEngine
from app.process_mining.visualizations import (
    create_activity_frequency_chart,
    create_bottleneck_chart,
    create_process_flow_diagram,
    create_throughput_distribution,
    create_transition_heatmap,
    create_variant_chart,
)

# -----------------------------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ProcureLens | Process Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Enterprise SaaS CSS Palette
CUSTOM_CSS = """
<style>
    /* Global Styles */
    .stApp {
        background-color: #0a1628;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #162544 0%, #0f1d38 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        color: #94a3b8;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 4px;
    }
    .metric-subtitle {
        font-size: 0.78rem;
        color: #64748b;
    }
    
    /* Badges & Tags */
    .badge-high {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .badge-medium {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .badge-low {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
    }

    /* Synthetic Data Disclaimer Banner */
    .synthetic-banner {
        background-color: rgba(30, 58, 138, 0.35);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 0.8rem;
        color: #93c5fd;
        margin-bottom: 20px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Data Loaders
# -----------------------------------------------------------------------------
@st.cache_resource
def get_db():
    return get_engine()


@st.cache_data(ttl=300)
def load_procurement_data():
    engine = get_db()
    pos = get_all_purchase_orders(engine)
    sups = get_all_suppliers(engine)
    invs = get_all_invoices(engine)
    pays = get_all_payments(engine)
    evts = get_all_event_logs(engine)
    return {
        "purchase_orders": pos,
        "suppliers": sups,
        "invoices": invs,
        "payments": pays,
        "event_log": evts
    }


# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="padding: 10px 0 20px 0;">
        <h2 style="margin: 0; color: #ffffff; font-size: 1.5rem; display: flex; align-items: center; gap: 8px;">
            <span>🔍</span> ProcureLens
        </h2>
        <div style="font-size: 0.75rem; color: #38bdf8; font-weight: 500; letter-spacing: 0.04em;">
            PROCESS INTELLIGENCE PLATFORM
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

PAGE_OPTIONS = [
    "📊 Executive Overview",
    "📈 Process Explorer",
    "🔍 Bottleneck Analysis",
    "📦 Supplier Analytics",
    "⚠️ Risk & Anomaly Detection",
    "🤖 Predictive Insights",
    "📑 Celonis & PQL Reference"
]

selected_page = st.sidebar.radio("Navigation", PAGE_OPTIONS, index=0)

st.sidebar.markdown("---")

# Environment & Celonis Diagnostic in Sidebar
celonis_cfg = CelonisConfig()
status_label = "Connected" if celonis_cfg.is_configured() else "Local Mode (PM4Py)"
status_color = "#34d399" if celonis_cfg.is_configured() else "#38bdf8"

st.sidebar.markdown(
    f"""
    <div style="font-size: 0.78rem; color: #94a3b8;">
        <b>Engine:</b> <span style="color: {status_color}; font-weight: 600;">{status_label}</span><br>
        <b>Database:</b> SQLite (Local)<br>
        <b>Analytics:</b> PM4Py 2.7+ & Scikit-learn
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div style="margin-top: 30px; font-size: 0.72rem; color: #64748b; line-height: 1.4;">
        ProcureLens v1.0.0<br>
        Designed for Celonis Process Intelligence Rising Star Technical Evaluation.
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# Main Application Header Banner
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="synthetic-banner">
        <b>ℹ️ SIMULATION NOTICE:</b> Displaying high-fidelity synthetic procurement data (5,000 purchase orders, 500 suppliers, 30,000+ event-log entries). All calculations, process variants, and machine learning models run live on this simulated enterprise dataset.
    </div>
    """,
    unsafe_allow_html=True
)

# Load data safely
try:
    data = load_procurement_data()
    engine = get_db()
    kpi_calc = KPICalculator(engine)
    pm_engine = ProcessMiningEngine(data["event_log"])
except Exception as e:
    st.error(f"Error loading procurement data: {str(e)}. Please run `python scripts/generate_data.py` to initialize.")
    st.stop()


# =============================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# =============================================================================
if selected_page == "📊 Executive Overview":
    st.title("Executive Overview")
    st.markdown("#### Turning Procurement Data into Process Intelligence")

    # High Level KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)

    total_pos = kpi_calc.total_purchase_orders()
    total_val = kpi_calc.total_procurement_value()
    avg_days = kpi_calc.average_processing_time()
    delayed_pct = kpi_calc.delayed_payment_percentage()
    compliance_pct = kpi_calc.process_compliance_rate()

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total PO Volume</div>
                <div class="metric-value">{total_pos:,}</div>
                <div class="metric-subtitle">Across 8 departments</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Procurement Spend</div>
                <div class="metric-value">${total_val/1e6:,.1f}M</div>
                <div class="metric-subtitle">Total gross spend</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Avg P2P Duration</div>
                <div class="metric-value">{avg_days:.1f} <span style="font-size: 1rem; color:#94a3b8;">Days</span></div>
                <div class="metric-subtitle">Requisition to Payment</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Delayed Payments</div>
                <div class="metric-value" style="color: {'#f87171' if delayed_pct > 10 else '#ffffff'};">{delayed_pct:.1f}%</div>
                <div class="metric-subtitle">Settled after due date</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Happy Path Rate</div>
                <div class="metric-value" style="color: {'#34d399' if compliance_pct > 50 else '#fbbf24'};">{compliance_pct:.1f}%</div>
                <div class="metric-subtitle">Conforming cases</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Monthly Trends and Department Breakdown
    row2_col1, row2_col2 = st.columns([6, 4])

    with row2_col1:
        trends = kpi_calc.monthly_trends()
        if trends:
            trends_df = pd.DataFrame(trends)
            fig_trend = px.line(
                trends_df,
                x="Month",
                y="Total_Spend",
                title="Monthly Procurement Spend ($)",
                markers=True,
                line_shape="spline"
            )
            fig_trend.update_traces(line_color="#38bdf8", line_width=3)
            fig_trend.update_layout(
                plot_bgcolor="#162544",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                height=350,
                yaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickformat="$,.0f"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.08)")
            )
            st.plotly_chart(fig_trend, use_container_width=True)

    with row2_col2:
        status_dist = kpi_calc.procurement_status_distribution()
        if status_dist:
            fig_status = px.pie(
                names=list(status_dist.keys()),
                values=list(status_dist.values()),
                title="Purchase Order Status Distribution",
                hole=0.55,
                color_discrete_sequence=["#38bdf8", "#34d399", "#f59e0b", "#f87171", "#818cf8"]
            )
            fig_status.update_layout(
                plot_bgcolor="#162544",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                height=350,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2)
            )
            st.plotly_chart(fig_status, use_container_width=True)

    # Department Comparison
    st.subheader("Departmental Procurement Performance")
    dept_stats = kpi_calc.department_comparison()
    if dept_stats:
        dept_df = pd.DataFrame(dept_stats)
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            fig_dept_spend = px.bar(
                dept_df,
                x="Total_Spend",
                y="Department",
                orientation="h",
                title="Total Spend by Business Unit",
                color="Total_Spend",
                color_continuous_scale="Blues"
            )
            fig_dept_spend.update_layout(
                plot_bgcolor="#162544",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                height=320,
                yaxis=dict(autorange="reversed"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickformat="$,.0f"),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_dept_spend, use_container_width=True)

        with col_d2:
            fig_dept_days = px.bar(
                dept_df,
                x="Avg_Approval_Days",
                y="Department",
                orientation="h",
                title="Average Approval Duration (Days)",
                color="Avg_Approval_Days",
                color_continuous_scale="Tealgrn"
            )
            fig_dept_days.update_layout(
                plot_bgcolor="#162544",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                height=320,
                yaxis=dict(autorange="reversed"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_dept_days, use_container_width=True)


# =============================================================================
# PAGE 2: PROCESS EXPLORER
# =============================================================================
elif selected_page == "📈 Process Explorer":
    st.title("Process Explorer")
    st.markdown("#### Reconstruct End-to-End Workflow Paths & Analyze Variant Frequencies")

    # Sankey Process Flow
    st.subheader("Process Flow Visualization")
    transition_mat = pm_engine.get_transition_matrix()
    if not transition_mat.empty:
        fig_flow = create_process_flow_diagram(transition_mat)
        st.plotly_chart(fig_flow, use_container_width=True)

    col_v1, col_v2 = st.columns([6, 4])

    with col_v1:
        variants = pm_engine.discover_process_variants()
        st.subheader("Process Execution Variants")
        fig_var = create_variant_chart(variants, top_n=10)
        st.plotly_chart(fig_var, use_container_width=True)

    with col_v2:
        st.subheader("Variant Details")
        if variants:
            var_table_data = [
                {
                    "Variant": f"#{v['variant_id']}",
                    "Case Count": f"{v['count']:,}",
                    "Share (%)": f"{v['percentage']}%",
                    "Path": v["variant_str"]
                }
                for v in variants[:10]
            ]
            st.dataframe(pd.DataFrame(var_table_data), use_container_width=True, height=450)

    # Activity Frequencies & Throughput
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        freqs = pm_engine.get_activity_frequencies()
        fig_freq = create_activity_frequency_chart(freqs)
        st.plotly_chart(fig_freq, use_container_width=True)

    with col_f2:
        throughput_stats = pm_engine.calculate_throughput_times()
        fig_dist = create_throughput_distribution(throughput_stats)
        st.plotly_chart(fig_dist, use_container_width=True)

    # Interactive Case Inspector
    st.markdown("---")
    st.subheader("Interactive Case Journey Inspector")
    sample_cases = data["event_log"]["Case_ID"].unique()[:50]
    selected_case = st.selectbox("Select Case ID to inspect event trail:", sample_cases)

    if selected_case:
        case_events = pm_engine.get_case_details(selected_case)
        st.dataframe(
            case_events[["Activity", "Timestamp", "Resource", "Department"]],
            use_container_width=True
        )


# =============================================================================
# PAGE 3: BOTTLENECK ANALYSIS
# =============================================================================
elif selected_page == "🔍 Bottleneck Analysis":
    st.title("Bottleneck Analysis")
    st.markdown("#### Discover Execution Delays, Waiting Times & Repeated Cycles")

    bottlenecks = pm_engine.identify_bottlenecks()

    if bottlenecks:
        top_bottleneck = bottlenecks[0]
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Primary Operational Bottleneck</div>
                    <div class="metric-value" style="font-size: 1.25rem; color: #f87171;">{top_bottleneck['transition']}</div>
                    <div class="metric-subtitle">Avg delay: {top_bottleneck['avg_duration_days']:.1f} days ({top_bottleneck['case_count']:,} cases)</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col_b2:
            second_b = bottlenecks[1] if len(bottlenecks) > 1 else top_bottleneck
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Secondary Bottleneck</div>
                    <div class="metric-value" style="font-size: 1.25rem; color: #fbbf24;">{second_b['transition']}</div>
                    <div class="metric-subtitle">Avg delay: {second_b['avg_duration_days']:.1f} days ({second_b['case_count']:,} cases)</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col_b3:
            rework_df = pm_engine.detect_repeated_activities()
            rework_cases = rework_df["Case_ID"].nunique()
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Cases with Process Rework</div>
                    <div class="metric-value" style="color: #60a5fa;">{rework_cases:,}</div>
                    <div class="metric-subtitle">Contains repeated activities</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        fig_bn = create_bottleneck_chart(bottlenecks, top_n=10)
        st.plotly_chart(fig_bn, use_container_width=True)

    # Transition Heatmap
    st.subheader("Transition Density Heatmap")
    matrix = pm_engine.get_transition_matrix()
    if not matrix.empty:
        fig_heat = create_transition_heatmap(matrix)
        st.plotly_chart(fig_heat, use_container_width=True)

    # Repeated Approvals Analysis
    st.subheader("Repeated Approval Cycles (Rework Inefficiencies)")
    rework_df = pm_engine.detect_repeated_activities()
    if not rework_df.empty:
        approval_rework = rework_df[rework_df["Activity"] == "Purchase Order Approved"]
        st.write(f"Identified **{len(approval_rework):,} cases** where purchase order approval required multiple iterative submissions.")
        st.dataframe(approval_rework.head(20), use_container_width=True)


# =============================================================================
# PAGE 4: SUPPLIER ANALYTICS
# =============================================================================
elif selected_page == "📦 Supplier Analytics":
    st.title("Supplier Analytics")
    st.markdown("#### Vendor Delivery Performance, Procurement Spend & Payment Reliability")

    sups_df = data["suppliers"]
    pos_df = data["purchase_orders"]
    pays_df = data["payments"]
    invs_df = data["invoices"]

    scores = kpi_calc.supplier_performance_scores(top_n=50)
    scores_df = pd.DataFrame(scores)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        # Spend by Category
        cat_spend = pos_df.merge(sups_df, on="Supplier_ID").groupby("Supplier_Category")["Purchase_Amount"].sum().reset_index()
        fig_cat = px.pie(
            cat_spend,
            names="Supplier_Category",
            values="Purchase_Amount",
            title="Procurement Spend by Vendor Category",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_cat.update_layout(
            plot_bgcolor="#162544",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            height=360
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_s2:
        # Top 10 Suppliers by Spend
        if not scores_df.empty:
            fig_sup_spend = px.bar(
                scores_df.head(10),
                x="Total_Spend",
                y="Supplier_Name",
                orientation="h",
                title="Top 10 Suppliers by Procurement Spend ($)",
                color="Supplier_Rating",
                color_continuous_scale="Viridis"
            )
            fig_sup_spend.update_layout(
                plot_bgcolor="#162544",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                height=360,
                yaxis=dict(autorange="reversed"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickformat="$,.0f")
            )
            st.plotly_chart(fig_sup_spend, use_container_width=True)

    st.subheader("Supplier Scorecard & Risk Profile")
    if not scores_df.empty:
        st.dataframe(
            scores_df[["Supplier_ID", "Supplier_Name", "Supplier_Category", "Supplier_Rating", "Total_Orders", "Total_Spend", "Delayed_Payments"]],
            use_container_width=True,
            height=400
        )


# =============================================================================
# PAGE 5: RISK & ANOMALY DETECTION
# =============================================================================
elif selected_page == "⚠️ Risk & Anomaly Detection":
    st.title("Risk & Anomaly Detection")
    st.markdown("#### Automated Controls: Duplicate Invoices, Missing Approvals & Process Deviations")

    detector = AnomalyDetector()
    anomalies = detector.run_all_checks(data)

    high_count = sum(1 for a in anomalies if a["severity"] == "HIGH")
    med_count = sum(1 for a in anomalies if a["severity"] == "MEDIUM")
    low_count = sum(1 for a in anomalies if a["severity"] == "LOW")

    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
    with col_a1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Anomalies</div>
                <div class="metric-value">{len(anomalies):,}</div>
                <div class="metric-subtitle">Across all control checks</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_a2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">High Severity Risks</div>
                <div class="metric-value" style="color: #f87171;">{high_count:,}</div>
                <div class="metric-subtitle">Immediate audit triage</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_a3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Medium Severity Risks</div>
                <div class="metric-value" style="color: #fbbf24;">{med_count:,}</div>
                <div class="metric-subtitle">Operational friction</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_a4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Low Severity Risks</div>
                <div class="metric-value" style="color: #34d399;">{low_count:,}</div>
                <div class="metric-subtitle">Minor timing variances</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Anomaly Types Chart
    df_anom = pd.DataFrame(anomalies)
    if not df_anom.empty:
        type_counts = df_anom["type"].value_counts().reset_index()
        type_counts.columns = ["Anomaly_Type", "Count"]

        fig_anom_bar = px.bar(
            type_counts,
            x="Count",
            y="Anomaly_Type",
            orientation="h",
            title="Operational Control Violations by Type",
            color="Count",
            color_continuous_scale="Reds"
        )
        fig_anom_bar.update_layout(
            plot_bgcolor="#162544",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            height=320,
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_anom_bar, use_container_width=True)

        # Filters
        st.subheader("Detected Anomalies Log")
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            chosen_sev = st.multiselect("Filter Severity", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM"])
        with filter_col2:
            all_types = sorted(df_anom["type"].unique())
            chosen_types = st.multiselect("Filter Type", all_types, default=all_types)

        filtered = df_anom[
            df_anom["severity"].isin(chosen_sev) &
            df_anom["type"].isin(chosen_types)
        ]

        st.dataframe(
            filtered[["severity", "type", "record_id", "case_id", "description", "details"]],
            use_container_width=True,
            height=450
        )


# =============================================================================
# PAGE 6: PREDICTIVE INSIGHTS
# =============================================================================
elif selected_page == "🤖 Predictive Insights":
    st.title("Predictive Insights")
    st.markdown("#### Machine Learning: Payment Delay Risk Modeling & Working Capital Optimization")

    fe = FeatureEngineer()
    features_df = fe.build_features(
        data["purchase_orders"],
        data["invoices"],
        data["payments"],
        data["event_log"],
        data["suppliers"]
    )

    predictor = PaymentDelayPredictor()

    col_btn, col_info = st.columns([3, 7])
    with col_btn:
        if st.button("🚀 Train & Benchmark Models", type="primary"):
            with st.spinner("Training Logistic Regression vs Random Forest on historical procurement dataset..."):
                trainer = ModelTrainer()
                X_train, X_test, y_train, y_test = fe.get_time_split(features_df, split_month=4)
                results = trainer.train_and_compare(X_train, X_test, y_train, y_test)
                saved_path = trainer.save_artifacts()
                st.session_state["ml_results"] = results
                predictor = PaymentDelayPredictor(saved_path)
                st.success(f"Models evaluated and serialized to {saved_path}!")

    # Display Model Evaluation if available
    ml_res = st.session_state.get("ml_results", None)
    if ml_res:
        st.subheader(f"Model Benchmark Summary (Selected: {ml_res['best_model']})")

        eval_lr = ml_res["models"]["Logistic Regression"]
        eval_rf = ml_res["models"]["Random Forest"]

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric("ROC-AUC Score", f"{eval_rf['roc_auc']:.3f}", delta=f"{eval_rf['roc_auc'] - eval_lr['roc_auc']:+.3f} vs LR")
        with metric_col2:
            st.metric("Precision", f"{eval_rf['precision']:.3f}", delta=f"{eval_rf['precision'] - eval_lr['precision']:+.3f}")
        with metric_col3:
            st.metric("Recall", f"{eval_rf['recall']:.3f}", delta=f"{eval_rf['recall'] - eval_lr['recall']:+.3f}")
        with metric_col4:
            st.metric("F1-Score", f"{eval_rf['f1_score']:.3f}", delta=f"{eval_rf['f1_score'] - eval_lr['f1_score']:+.3f}")

        # Feature Importance Chart
        if "feature_importances" in ml_res:
            imp_df = pd.DataFrame(list(ml_res["feature_importances"].items()), columns=["Feature", "Importance"])
            fig_imp = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Random Forest Feature Importance (Delay Drivers)",
                color="Importance",
                color_continuous_scale="Teal"
            )
            fig_imp.update_layout(
                plot_bgcolor="#162544",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                height=320,
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    # Scored Transactions Table
    st.subheader("Transaction Risk Scoring & Prioritization")
    scored_sample = predictor.predict_batch(features_df.head(100))

    col_r1, col_r2 = st.columns([4, 6])
    with col_r1:
        risk_dist = scored_sample["risk_category"].value_counts().reset_index()
        risk_dist.columns = ["Risk_Tier", "Count"]
        fig_risk = px.pie(
            risk_dist,
            names="Risk_Tier",
            values="Count",
            title="Risk Tier Distribution (Sample)",
            hole=0.45,
            color="Risk_Tier",
            color_discrete_map={"HIGH": "#f87171", "MEDIUM": "#fbbf24", "LOW": "#34d399"}
        )
        fig_risk.update_layout(
            plot_bgcolor="#162544",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            height=300
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_r2:
        st.markdown(
            """
            <div style="background-color: #162544; border-radius: 8px; padding: 15px; margin-top: 25px;">
                <h4 style="margin: 0 0 10px 0;">Risk Tier Policy Thresholds</h4>
                <ul style="font-size: 0.85rem; line-height: 1.6; color: #cbd5e1;">
                    <li><span class="badge-high">HIGH (≥ 70%)</span> Priority intervention required: Notify AP lead, verify invoice approval and terms.</li>
                    <li><span class="badge-medium">MEDIUM (30% - 69%)</span> Elevated risk: Track against goods receipt milestone.</li>
                    <li><span class="badge-low">LOW (&lt; 30%)</span> Standard automated payment pipeline.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.dataframe(
        scored_sample[["PO_ID", "Invoice_ID", "purchase_amount", "supplier_rating", "approval_duration_days", "delay_probability", "risk_category"]],
        use_container_width=True,
        height=350
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.78rem; color: #94a3b8;">
            <b>Governance & Limits:</b> This model was trained with zero leakage of future payment metrics (Payment_Date, Payment_Amount, or Payment_Status). Features are restricted strictly to attributes observed prior to payment execution.
        </div>
        """,
        unsafe_allow_html=True
    )


# =============================================================================
# PAGE 7: CELONIS & PQL REFERENCE
# =============================================================================
elif selected_page == "📑 Celonis & PQL Reference":
    st.title("Celonis & PQL Integration Reference")
    st.markdown("#### Production Process Query Language (PQL) Repository & Cloud Integration Architecture")

    st.markdown(
        """
        ProcureLens is architected to seamlessly interface with the **Celonis Process Intelligence Graph (EMS)**.
        Below is the verified repository of Celonis PQL queries engineered to run against the ProcureLens data model.
        """
    )

    pql_catalog = get_all_pql_queries()

    for key, item in pql_catalog.items():
        with st.expander(f"🔹 {item['title']} [{item['category']}]", expanded=(key == "p2p_throughput_time")):
            st.markdown(f"**Objective:** {item['description']}")
            st.code(item["query"], language="sql")
