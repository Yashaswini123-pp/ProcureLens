"""
ProcureLens Process Mining Engine.

Provides enterprise-grade process intelligence:
- Trace and variant discovery
- Activity frequency and throughput time statistics
- Bottleneck identification and transition duration analysis
- Conformance checking against configurable standard P2P model
- Rework and skipped activity detection
- Transition probability and frequency matrices
- Integration with PM4Py for DFG and process model discovery
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger("procurelens.process_mining.engine")

# Check for PM4Py
try:
    import pm4py
    from pm4py.objects.conversion.log import converter as log_converter
    PM4PY_AVAILABLE = True
except ImportError:
    pm4py = None
    PM4PY_AVAILABLE = False


class ProcessMiningEngine:
    """
    Enterprise Process Mining Engine for Purchase-to-Pay (P2P) Workflows.
    """

    # Configurable standard Purchase-to-Pay happy path
    STANDARD_P2P_PROCESS: List[str] = [
        "Purchase Requisition Created",
        "Purchase Order Generated",
        "Purchase Order Approved",
        "Goods Received",
        "Invoice Received",
        "Invoice Validated",
        "Invoice Approved",
        "Payment Executed"
    ]

    def __init__(self, event_log_df: Optional[pd.DataFrame] = None):
        self.df: pd.DataFrame = pd.DataFrame()
        if event_log_df is not None and not event_log_df.empty:
            self.load_event_log(event_log_df)

    def load_event_log(self, df: pd.DataFrame) -> None:
        """
        Load and prepare the event log DataFrame.
        Ensures Case_ID, Activity, and Timestamp are properly typed and ordered.
        """
        required_cols = {"Case_ID", "Activity", "Timestamp"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Event log is missing mandatory columns: {missing}")

        clean_df = df.copy()
        clean_df["Case_ID"] = clean_df["Case_ID"].astype(str)
        clean_df["Activity"] = clean_df["Activity"].astype(str)
        clean_df["Timestamp"] = pd.to_datetime(clean_df["Timestamp"])

        # Sort strictly by Case_ID then Timestamp
        clean_df = clean_df.sort_values(by=["Case_ID", "Timestamp"]).reset_index(drop=True)
        self.df = clean_df
        logger.info(f"Loaded event log with {len(self.df)} events across {self.df['Case_ID'].nunique()} cases.")

    # -------------------------------------------------------------------------
    # Process Variants Discovery
    # -------------------------------------------------------------------------
    def discover_process_variants(self) -> List[Dict]:
        """
        Group cases into unique execution variants and calculate frequencies and percentages.
        
        Returns:
            List of dicts:
            [{"variant_id": 1, "variant": [...], "count": 1200, "percentage": 24.0, "is_standard": True}, ...]
        """
        if self.df.empty:
            return []

        # Aggregate activities per case into a tuple
        grouped = self.df.groupby("Case_ID")["Activity"].apply(tuple)
        variant_counts = grouped.value_counts()
        total_cases = len(grouped)

        variants_list = []
        std_tuple = tuple(self.STANDARD_P2P_PROCESS)

        for idx, (variant_tuple, count) in enumerate(variant_counts.items(), start=1):
            pct = round((count / total_cases) * 100.0, 2)
            is_standard = (variant_tuple == std_tuple)
            variants_list.append({
                "variant_id": idx,
                "variant": list(variant_tuple),
                "variant_str": " -> ".join(variant_tuple),
                "count": int(count),
                "percentage": pct,
                "is_standard": is_standard
            })

        return variants_list

    # -------------------------------------------------------------------------
    # Activity Frequencies
    # -------------------------------------------------------------------------
    def get_activity_frequencies(self) -> Dict[str, int]:
        """Return occurrence count of every activity across the entire event log."""
        if self.df.empty:
            return {}
        counts = self.df["Activity"].value_counts().to_dict()
        return {str(k): int(v) for k, v in counts.items()}

    # -------------------------------------------------------------------------
    # Throughput Times Calculation
    # -------------------------------------------------------------------------
    def calculate_throughput_times(self) -> Dict:
        """
        Calculate total case execution duration (first event to last event).
        
        Returns:
            Dict containing mean, median, min, max, std (in days), and per-case durations.
        """
        if self.df.empty:
            return {
                "mean_days": 0.0,
                "median_days": 0.0,
                "min_days": 0.0,
                "max_days": 0.0,
                "std_days": 0.0,
                "case_durations": []
            }

        case_times = self.df.groupby("Case_ID")["Timestamp"].agg(["min", "max", "count"])
        case_times["duration_days"] = (case_times["max"] - case_times["min"]).dt.total_seconds() / 86400.0

        durations = case_times["duration_days"].values

        return {
            "mean_days": round(float(np.mean(durations)), 2),
            "median_days": round(float(np.median(durations)), 2),
            "min_days": round(float(np.min(durations)), 2),
            "max_days": round(float(np.max(durations)), 2),
            "std_days": round(float(np.std(durations)), 2),
            "case_durations": [round(float(d), 2) for d in durations[:5000]]
        }

    # -------------------------------------------------------------------------
    # Bottleneck Analysis & Transition Times
    # -------------------------------------------------------------------------
    def identify_bottlenecks(self) -> List[Dict]:
        """
        Calculate average waiting and processing time between subsequent activities.
        
        Returns:
            Ranked list of transitions with highest average duration in days.
        """
        if self.df.empty:
            return []

        df_sorted = self.df.sort_values(by=["Case_ID", "Timestamp"]).copy()
        df_sorted["Next_Activity"] = df_sorted.groupby("Case_ID")["Activity"].shift(-1)
        df_sorted["Next_Timestamp"] = df_sorted.groupby("Case_ID")["Timestamp"].shift(-1)

        # Drop terminal events
        transitions = df_sorted.dropna(subset=["Next_Activity", "Next_Timestamp"]).copy()
        transitions["Duration_Days"] = (
            (transitions["Next_Timestamp"] - transitions["Timestamp"]).dt.total_seconds() / 86400.0
        )

        bottlenecks = []
        grouped = transitions.groupby(["Activity", "Next_Activity"])
        for (source, target), group in grouped:
            durations = group["Duration_Days"].values
            bottlenecks.append({
                "from_activity": source,
                "to_activity": target,
                "transition": f"{source} -> {target}",
                "case_count": int(len(group)),
                "avg_duration_days": round(float(np.mean(durations)), 2),
                "median_duration_days": round(float(np.median(durations)), 2),
                "max_duration_days": round(float(np.max(durations)), 2),
            })

        # Rank by longest average duration
        bottlenecks.sort(key=lambda x: x["avg_duration_days"], reverse=True)
        return bottlenecks

    # -------------------------------------------------------------------------
    # Repeated Activities (Rework)
    # -------------------------------------------------------------------------
    def detect_repeated_activities(self) -> pd.DataFrame:
        """
        Find cases where an activity occurs more than once (process rework loop).
        """
        if self.df.empty:
            return pd.DataFrame(columns=["Case_ID", "Activity", "Repeat_Count"])

        activity_counts = self.df.groupby(["Case_ID", "Activity"]).size().reset_index(name="Repeat_Count")
        rework = activity_counts[activity_counts["Repeat_Count"] > 1].copy()
        rework = rework.sort_values(by="Repeat_Count", ascending=False)
        return rework

    # -------------------------------------------------------------------------
    # Skipped Activities (Process Omissions)
    # -------------------------------------------------------------------------
    def detect_skipped_activities(self) -> pd.DataFrame:
        """
        Identify cases where critical standard activities are omitted.
        """
        if self.df.empty:
            return pd.DataFrame(columns=["Case_ID", "Skipped_Activities", "Skipped_Count"])

        grouped = self.df.groupby("Case_ID")["Activity"].apply(set)
        std_set = set(self.STANDARD_P2P_PROCESS)

        records = []
        for case_id, executed_activities in grouped.items():
            skipped = std_set - executed_activities
            if skipped:
                records.append({
                    "Case_ID": case_id,
                    "Skipped_Activities": list(skipped),
                    "Skipped_Count": len(skipped)
                })

        return pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # Conformance & Standard vs Deviating Comparison
    # -------------------------------------------------------------------------
    def compare_normal_abnormal(self) -> Dict:
        """
        Partition cases into conforming (exact standard sequence) vs non-conforming.
        """
        if self.df.empty:
            return {
                "conformance_rate": 0.0,
                "total_cases": 0,
                "normal_cases_count": 0,
                "abnormal_cases_count": 0,
                "normal_examples": [],
                "abnormal_examples": []
            }

        grouped = self.df.groupby("Case_ID")["Activity"].apply(list)
        total = len(grouped)
        std_seq = self.STANDARD_P2P_PROCESS

        normal_cases = []
        abnormal_cases = []

        for case_id, seq in grouped.items():
            if seq == std_seq:
                normal_cases.append(case_id)
            else:
                abnormal_cases.append(case_id)

        rate = round((len(normal_cases) / total) * 100.0, 2) if total > 0 else 0.0

        return {
            "conformance_rate": rate,
            "total_cases": total,
            "normal_cases_count": len(normal_cases),
            "abnormal_cases_count": len(abnormal_cases),
            "normal_examples": normal_cases[:10],
            "abnormal_examples": abnormal_cases[:10]
        }

    # -------------------------------------------------------------------------
    # Activity Transition Matrix
    # -------------------------------------------------------------------------
    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Construct a cross-tabulation matrix of activity-to-activity transition counts.
        """
        if self.df.empty:
            return pd.DataFrame()

        df_sorted = self.df.sort_values(by=["Case_ID", "Timestamp"]).copy()
        df_sorted["Next_Activity"] = df_sorted.groupby("Case_ID")["Activity"].shift(-1)
        transitions = df_sorted.dropna(subset=["Next_Activity"])

        matrix = pd.crosstab(
            transitions["Activity"],
            transitions["Next_Activity"],
            rownames=["Source Activity"],
            colnames=["Target Activity"]
        )
        return matrix

    # -------------------------------------------------------------------------
    # PM4Py Integration
    # -------------------------------------------------------------------------
    def get_pm4py_dfg(self) -> Tuple[Dict, Dict, Dict]:
        """
        Computes Directly-Follows Graph (DFG) using PM4Py if installed.
        Returns: (dfg, start_activities, end_activities)
        """
        if not PM4PY_AVAILABLE or self.df.empty:
            return {}, {}, {}

        try:
            # Format dataframe for PM4Py
            formatted_df = pm4py.format_dataframe(
                self.df,
                case_id="Case_ID",
                activity_key="Activity",
                timestamp_key="Timestamp"
            )
            dfg, start_activities, end_activities = pm4py.discover_dfg(formatted_df)
            # Convert tuple keys to string representation for JSON serialization
            dfg_serializable = {f"{k[0]} -> {k[1]}": int(v) for k, v in dfg.items()}
            start_serializable = {str(k): int(v) for k, v in start_activities.items()}
            end_serializable = {str(k): int(v) for k, v in end_activities.items()}

            return dfg_serializable, start_serializable, end_serializable
        except Exception as e:
            logger.error(f"PM4Py DFG calculation failed: {str(e)}")
            return {}, {}, {}

    def get_case_details(self, case_id: str) -> pd.DataFrame:
        """Retrieve full ordered event sequence and metadata for a specific case."""
        case_events = self.df[self.df["Case_ID"] == str(case_id)].copy()
        return case_events.sort_values(by="Timestamp").reset_index(drop=True)
