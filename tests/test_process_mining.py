"""
Unit tests for Process Mining Engine.
"""

from app.process_mining.engine import ProcessMiningEngine


def test_discover_process_variants(sample_event_log_df):
    """Verify distinct process traces and percentages are calculated."""
    pm = ProcessMiningEngine(sample_event_log_df)
    variants = pm.discover_process_variants()

    assert len(variants) > 0
    # Total percentage should sum to approximately 100%
    total_pct = sum(v["percentage"] for v in variants)
    assert abs(total_pct - 100.0) < 0.5
    assert any(v["is_standard"] for v in variants)


def test_activity_frequencies(sample_event_log_df):
    """Verify activity occurrence counts match event log records."""
    pm = ProcessMiningEngine(sample_event_log_df)
    freqs = pm.get_activity_frequencies()

    assert "Purchase Requisition Created" in freqs
    assert freqs["Purchase Requisition Created"] == 4  # 4 mock cases
    assert "Payment Executed" in freqs


def test_throughput_times_calculation(sample_event_log_df):
    """Verify mean, median, min, max throughput times are non-negative."""
    pm = ProcessMiningEngine(sample_event_log_df)
    stats = pm.calculate_throughput_times()

    assert stats["mean_days"] > 0
    assert stats["median_days"] > 0
    assert stats["max_days"] >= stats["min_days"]
    assert len(stats["case_durations"]) == 4


def test_identify_bottlenecks(sample_event_log_df):
    """Verify transition wait times and bottleneck identification."""
    pm = ProcessMiningEngine(sample_event_log_df)
    bottlenecks = pm.identify_bottlenecks()

    assert len(bottlenecks) > 0
    # First bottleneck should have highest average duration
    if len(bottlenecks) > 1:
        assert bottlenecks[0]["avg_duration_days"] >= bottlenecks[1]["avg_duration_days"]


def test_detect_repeated_activities(sample_event_log_df):
    """Verify detection of repeated activities (CASE-003 has 2 approvals)."""
    pm = ProcessMiningEngine(sample_event_log_df)
    rework = pm.detect_repeated_activities()

    assert not rework.empty
    case_rework = rework[rework["Case_ID"] == "CASE-003"]
    assert len(case_rework) > 0
    assert case_rework.iloc[0]["Activity"] == "Purchase Order Approved"
    assert case_rework.iloc[0]["Repeat_Count"] == 2


def test_detect_skipped_activities(sample_event_log_df):
    """Verify detection of missing activities (CASE-004 misses Goods Received)."""
    pm = ProcessMiningEngine(sample_event_log_df)
    skipped = pm.detect_skipped_activities()

    assert not skipped.empty
    case_skip = skipped[skipped["Case_ID"] == "CASE-004"]
    assert len(case_skip) > 0
    skipped_list = case_skip.iloc[0]["Skipped_Activities"]
    assert "Goods Received" in skipped_list


def test_conformance_comparison(sample_event_log_df):
    """Verify conformance partitioning into standard vs deviating."""
    pm = ProcessMiningEngine(sample_event_log_df)
    comp = pm.compare_normal_abnormal()

    assert comp["total_cases"] == 4
    assert comp["normal_cases_count"] == 2  # CASE-001 and CASE-002
    assert comp["abnormal_cases_count"] == 2  # CASE-003 and CASE-004
    assert comp["conformance_rate"] == 50.0
