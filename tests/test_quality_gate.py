"""Tests for the deterministic quality gate."""

from governance.quality_gate import cost_metrics, evaluate_quality_gate, gate_rules

THRESHOLDS = {"map_50_95": 0.75, "recall": 0.85, "max_latency_ms": 50}


def test_all_pass():
    metrics = {"map_50_95": 0.78, "recall": 0.87, "latency_ms": 29}
    result = evaluate_quality_gate(metrics, THRESHOLDS)
    assert result.passed is True
    assert all(c["ok"] for c in result.checks)


def test_fails_on_low_map():
    metrics = {"map_50_95": 0.70, "recall": 0.90, "latency_ms": 20}
    result = evaluate_quality_gate(metrics, THRESHOLDS)
    assert result.passed is False


def test_fails_on_high_latency():
    metrics = {"map_50_95": 0.80, "recall": 0.90, "latency_ms": 60}
    result = evaluate_quality_gate(metrics, THRESHOLDS)
    assert result.passed is False


def test_missing_metric_fails_closed():
    metrics = {"map_50_95": 0.80, "recall": 0.90}  # no latency
    result = evaluate_quality_gate(metrics, THRESHOLDS)
    assert result.passed is False


def test_empty_thresholds_never_passes():
    result = evaluate_quality_gate({"map_50_95": 0.9}, {})
    assert result.passed is False


def test_rules_are_derived_from_keys_not_a_fixed_list():
    assert gate_rules(THRESHOLDS) == [("map_50_95", ">=", 0.75), ("recall", ">=", 0.85), ("latency_ms", "<=", 50)]
    # Any task's metrics can be gated: >= by default, <= with a max_ prefix.
    thresholds = {"top1_accuracy": 0.9, "max_model_size_mb": 20}
    assert evaluate_quality_gate({"top1_accuracy": 0.95, "model_size_mb": 5.4}, thresholds).passed is True
    assert evaluate_quality_gate({"top1_accuracy": 0.95, "model_size_mb": 66}, thresholds).passed is False
    assert cost_metrics(thresholds, THRESHOLDS) == {"model_size_mb", "latency_ms"}
