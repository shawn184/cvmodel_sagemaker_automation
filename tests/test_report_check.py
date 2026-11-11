"""Tests for the deterministic checks of the agent's claims against recorded facts."""

import cli
from agents.tools import _backend
from governance import report_check as rc

ARN = "arn:aws:sagemaker:ap-southeast-1:000000000000:pipeline/ObjectDetectionAmgTrainingPipeline/execution/{n}"
GATE = {"map_50_95": 0.45, "recall": 0.70, "max_latency_ms": 100}
PASSING_STEPS = [{"name": "QualityGate", "status": "Succeeded"}, {"name": "RegisterModel-RegisterModel", "status": "Succeeded"}]
FAILING_STEPS = [{"name": "QualityGate", "status": "Succeeded"}, {"name": "ModelBelowThreshold", "status": "Failed"}]


def _record(n="a", status="registered", **metrics):
    base = {"map_50": 0.77, "map_50_95": 0.5794, "precision": 0.7424, "recall": 0.7845, "f1": 0.7629,
            "latency_ms": 57.64, "model_size_mb": 66.29}
    return {"experiment_id": f"exp-{n}", "execution_arn": ARN.format(n=n), "model": "rt_detr", "status": status,
            "hyperparameters": {"lr0": 0.0001, "batch": 16, "optimizer": "AdamW"}, "metrics": {**base, **metrics}}


def test_pipeline_gate_outcome_from_steps():
    assert rc.pipeline_gate_passed(PASSING_STEPS) is True
    assert rc.pipeline_gate_passed(FAILING_STEPS) is False
    assert rc.pipeline_gate_passed([{"name": "TrainModel", "status": "Failed"}]) is None


def test_report_matching_facts_has_no_issues():
    claims = {"gate_passed": True, "registered": True, "metrics": {"map_50_95": 0.58, "latency_ms": 57.6}}
    assert rc.verify_report(claims, _record(), True) == []


def test_report_contradicting_facts_is_flagged():
    claims = {"gate_passed": False, "registered": False, "metrics": {"recall": 0.5705, "auc": 0.9}}
    issues = rc.verify_report(claims, _record(), True)
    assert len(issues) == 4
    assert any("QualityGate step decided PASS" in i for i in issues)
    assert any("not registered" in i and "'registered'" in i for i in issues)
    assert any("recall=0.5705" in i for i in issues)
    assert any("no metric 'auc'" in i for i in issues)


def test_start_claims_checked_against_training_job():
    ok = {"model": "rt_detr", "hyperparameters": {"lr0": 0.0001, "optimizer": "AdamW"}}
    assert rc.verify_start(ok, _record()) == []
    bad = {"model": "yolo", "hyperparameters": {"lr0": 0.003, "momentum": 0.9}}
    issues = rc.verify_start(bad, _record())
    assert any("ran 'rt_detr'" in i for i in issues)
    assert any("lr0=0.0001" in i for i in issues)
    assert any("no hyperparameter 'momentum'" in i for i in issues)


def test_identical_metrics_to_previous_run_is_flagged():
    previous = _record("a")
    previous["hyperparameters"]["lr0"] = 0.0005
    issues = rc.run_invariants(_record("b"), [previous], GATE, True)
    assert len(issues) == 1
    assert "identical to exp-a" in issues[0] and "lr0" in issues[0]
    # Latency is a max_ (lower-is-better) gate metric, so its jitter does not hide a repeat.
    assert len(rc.run_invariants(_record("b", latency_ms=61.0), [previous], GATE, True)) == 1
    # Different numbers, or a different model, are not a repeat.
    assert rc.run_invariants(_record("b", recall=0.7), [previous], GATE, True) == []
    assert rc.run_invariants(_record("b", model_size_mb=1.0), [previous], GATE, True) == []
    previous["model"] = "yolo"
    assert rc.run_invariants(_record("b"), [previous], GATE, True) == []


def test_gate_config_drift_is_flagged():
    issues = rc.run_invariants(_record(recall=0.5, status="registered"), [], GATE, True)
    assert len(issues) == 1 and "config changed" in issues[0]
    assert rc.run_invariants(_record(), [], GATE, True) == []
    assert rc.run_invariants(_record(), [], GATE, None) == []  # gate never ran: nothing to compare


def test_record_run_prints_check_and_saves(monkeypatch, capsys):
    project = "object_detection_amg"
    monkeypatch.setattr(_backend, "collect_run_record", lambda p, arn: _record("x"))
    claims = {"gate_passed": True, "registered": False, "metrics": {"map_50_95": 0.5794}}
    cli._record_run(project, ARN.format(n="x"), PASSING_STEPS, claims, {"model": "rt_detr", "hyperparameters": {}})
    out = capsys.readouterr().out
    assert "=== Report check ===" in out
    assert "WARNING: agent said the model was not registered" in out
    assert "Recorded exp-x (registered)" in out
    assert _backend.experiment_for_execution(project, ARN.format(n="x")) is not None


def test_record_run_reports_ok_when_everything_agrees(monkeypatch, capsys):
    monkeypatch.setattr(_backend, "collect_run_record", lambda p, arn: _record("y"))
    claims = {"gate_passed": True, "registered": True, "metrics": {"recall": 0.78}}
    cli._record_run("object_detection_amg", ARN.format(n="y"), PASSING_STEPS, claims)
    out = capsys.readouterr().out
    assert "OK: the agent's report and run agree" in out
    assert "WARNING" not in out


def test_claims_extraction_failure_does_not_abort(capsys):
    class NoStructuredOutput:
        pass

    assert cli._claims(NoStructuredOutput(), rc.ReportClaims, "x") is None
    assert "Could not extract the agent's claims" in capsys.readouterr().out
