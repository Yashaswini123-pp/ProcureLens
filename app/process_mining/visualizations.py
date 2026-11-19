"""
ProcureLens Process Mining Visualizations.

Generates high-fidelity Plotly charts styled with an executive enterprise dark navy aesthetic:
- Sankey diagram for end-to-end process flows
- Top process variants horizontal bar chart
- Activity frequency distribution
- Throughput time distribution histogram
- Bottleneck ranking bar chart
- Transition matrix heatmap
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Enterprise Dark Theme constants
BG_COLOR = "#0a1628"
PAPER_COLOR = "#0f1e36"
CARD_BG = "#162544"
TEXT_COLOR = "#e2e8f0"
MUTED_TEXT = "#94a3b8"
GRID_COLOR = "rgba(255, 255, 255, 0.08)"
ACCENT_BLUE = "#3b82f6"
ACCENT_CYAN = "#06b6d4"
ACCENT_AMBER = "#f59e0b"
ACCENT_RED = "#ef4444"
ACCENT_GREEN = "#10b981"

DARK_TEMPLATE = dict(
    layout=go.Layout(
        plot_bgcolor=CARD_BG,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR, family="Inter, system-ui, sans-serif"),
        title=dict(font=dict(color="#ffffff", size=16)),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    )
)


def create_process_flow_diagram(
    matrix_or_dfg: pd.DataFrame,
    max_nodes: int = 12
) -> go.Figure:
    """
    Constructs a Sankey diagram illustrating activity transitions and volume flows.
    """
    if matrix_or_dfg.empty:
        fig = go.Figure()
        fig.update_layout(title="No transition data available", template=DARK_TEMPLATE)
        return fig

    # If DataFrame matrix
    activities = list(set(matrix_or_dfg.index).union(set(matrix_or_dfg.columns)))
    act_to_idx = {act: i for i, act in enumerate(activities)}

    sources = []
    targets = []
    values = []

    for src in matrix_or_dfg.index:
        for tgt in matrix_or_dfg.columns:
            count = matrix_or_dfg.loc[src, tgt]
            if count > 0:
                sources.append(act_to_idx[src])
                targets.append(act_to_idx[tgt])
                values.append(int(count))

    node_colors = [
        ACCENT_BLUE if "Approved" in act else (
            ACCENT_CYAN if "Created" in act or "Generated" in act else (
                ACCENT_GREEN if "Payment" in act else (
                    ACCENT_RED if "Rejected" in act else "#64748b"
                )
            )
        )
        for act in activities
    ]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=18,
            thickness=18,
            line=dict(color="#1e293b", width=1),
            label=activities,
            color=node_colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(59, 130, 246, 0.25)"
        )
    )])

    fig.update_layout(
        title="Purchase-to-Pay End-to-End Process Flow (Sankey)",
        font=dict(color=TEXT_COLOR, size=11),
        template=DARK_TEMPLATE,
        height=500,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def create_variant_chart(variants: List[Dict], top_n: int = 10) -> go.Figure:
    """
    Horizontal bar chart showing the frequency and percentage of top process variants.
    """
    if not variants:
        fig = go.Figure()
        fig.update_layout(title="No variants discovered", template=DARK_TEMPLATE)
        return fig

    top_variants = variants[:top_n]
    labels = [f"Variant {v['variant_id']}" + (" (Standard)" if v.get("is_standard") else "") for v in top_variants]
    counts = [v["count"] for v in top_variants]
    percentages = [v["percentage"] for v in top_variants]
    colors = [ACCENT_GREEN if v.get("is_standard") else ACCENT_BLUE for v in top_variants]

    hover_texts = ["<br>".join(v["variant"]) for v in top_variants]

    fig = go.Figure(go.Bar(
        x=counts,
        y=labels,
        orientation="h",
        marker=dict(color=colors, line=dict(color="#1e293b", width=1)),
        text=[f"{cnt:,} ({pct}%)" for cnt, pct in zip(counts, percentages)],
        textposition="outside",
        hovertext=hover_texts,
        hoverinfo="text+x"
    ))

    fig.update_layout(
        title=f"Top {len(top_variants)} Process Execution Variants",
        xaxis_title="Case Volume",
        yaxis=dict(autorange="reversed"),
        template=DARK_TEMPLATE,
        height=450,
        margin=dict(l=150, r=60, t=50, b=40)
    )
    return fig


def create_activity_frequency_chart(frequencies: Dict[str, int]) -> go.Figure:
    """Bar chart displaying the frequency of each activity."""
    if not frequencies:
        fig = go.Figure()
        fig.update_layout(title="No activity data", template=DARK_TEMPLATE)
        return fig

    sorted_freq = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
    activities = [x[0] for x in sorted_freq]
    counts = [x[1] for x in sorted_freq]

    colors = [
        ACCENT_RED if "Rejected" in a else (
            ACCENT_GREEN if "Payment" in a else ACCENT_BLUE
        )
        for a in activities
    ]

    fig = go.Figure(go.Bar(
        x=counts,
        y=activities,
        orientation="h",
        marker=dict(color=colors),
        text=[f"{c:,}" for c in counts],
        textposition="outside"
    ))

    fig.update_layout(
        title="Activity Execution Frequencies Across All Cases",
        xaxis_title="Execution Count",
        yaxis=dict(autorange="reversed"),
        template=DARK_TEMPLATE,
        height=400,
        margin=dict(l=180, r=40, t=50, b=40)
    )
    return fig


def create_throughput_distribution(throughput_data: Dict) -> go.Figure:
    """Histogram with KDE overlay depicting case completion durations."""
    durations = throughput_data.get("case_durations", [])
    if not durations:
        fig = go.Figure()
        fig.update_layout(title="No throughput data available", template=DARK_TEMPLATE)
        return fig

    mean_val = throughput_data.get("mean_days", 0)
    median_val = throughput_data.get("median_days", 0)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=durations,
        nbinsx=40,
        name="Case Count",
        marker=dict(color="rgba(59, 130, 246, 0.7)", line=dict(color=ACCENT_BLUE, width=1))
    ))

    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color=ACCENT_AMBER,
        annotation_text=f"Mean: {mean_val:.1f}d",
        annotation_position="top right"
    )
    fig.add_vline(
        x=median_val,
        line_dash="dot",
        line_color=ACCENT_GREEN,
        annotation_text=f"Median: {median_val:.1f}d",
        annotation_position="top left"
    )

    fig.update_layout(
        title="Throughput Time Distribution (Days from Requisition to Payment)",
        xaxis_title="Duration (Days)",
        yaxis_title="Number of Cases",
        template=DARK_TEMPLATE,
        height=380,
        bargap=0.1
    )
    return fig


def create_bottleneck_chart(bottlenecks: List[Dict], top_n: int = 8) -> go.Figure:
    """Horizontal bar chart ranking the longest transition delays."""
    if not bottlenecks:
        fig = go.Figure()
        fig.update_layout(title="No bottleneck data", template=DARK_TEMPLATE)
        return fig

    top_b = bottlenecks[:top_n]
    transitions = [b["transition"] for b in top_b]
    avg_days = [b["avg_duration_days"] for b in top_b]
    case_counts = [b["case_count"] for b in top_b]

    # Color scale by severity
    max_d = max(avg_days) if avg_days else 1
    colors = [
        ACCENT_RED if d > 12 else (ACCENT_AMBER if d > 6 else ACCENT_BLUE)
        for d in avg_days
    ]

    fig = go.Figure(go.Bar(
        x=avg_days,
        y=transitions,
        orientation="h",
        marker=dict(color=colors),
        text=[f"{d:.1f} days ({cnt:,} cases)" for d, cnt in zip(avg_days, case_counts)],
        textposition="outside"
    ))

    fig.update_layout(
        title=f"Top {len(top_b)} Activity Transition Delays (Bottlenecks)",
        xaxis_title="Average Waiting Time (Days)",
        yaxis=dict(autorange="reversed"),
        template=DARK_TEMPLATE,
        height=420,
        margin=dict(l=260, r=80, t=50, b=40)
    )
    return fig


def create_transition_heatmap(matrix: pd.DataFrame) -> go.Figure:
    """Heatmap displaying activity-to-activity transition frequencies."""
    if matrix.empty:
        fig = go.Figure()
        fig.update_layout(title="No transition matrix data", template=DARK_TEMPLATE)
        return fig

    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=matrix.columns.tolist(),
        y=matrix.index.tolist(),
        colorscale=[
            [0.0, CARD_BG],
            [0.2, "#1e3a8a"],
            [0.5, "#2563eb"],
            [0.8, "#60a5fa"],
            [1.0, "#93c5fd"]
        ],
        text=matrix.values,
        texttemplate="%{text}",
        hoverongaps=False
    ))

    fig.update_layout(
        title="Activity Transition Frequency Matrix",
        xaxis_title="Target Activity",
        yaxis_title="Source Activity",
        template=DARK_TEMPLATE,
        height=520,
        margin=dict(l=180, r=40, t=50, b=120)
    )
    fig.update_xaxes(tickangle=45)
    return fig
