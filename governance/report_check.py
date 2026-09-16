"""Deterministic checks of what the agent *said* against what actually happened.

The agent's narrative is a claim, not a record. After each run the CLI asks the
agent to restate its report as structured data (``StartClaims`` /
``ReportClaims``) and this module compares those claims with the facts read
from SageMaker and S3 (the experiment record). It also checks run-level
invariants that do not involve the LLM at all, e.g. metrics identical to an
earlier run, which means a hyperparameter change had no effect.

Everything here is plain program logic; nothing calls a model.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from governance.quality_gate import cost_metrics, evaluate_quality_gate


class StartClaims(BaseModel):
    """What the agent reported when it started the pipeline."""

    model: str | None = Field(None, description="Approved model id passed to start_training_pipeline, e.g. 'yolo'.")
    hyperparameters: dict[str, float | int | str] = Field(
        default_factory=dict,
        description="Hyperparameters the tool result reported as applied (resolved values, e.g. {'lr0': 0.003}).",
    )
    execution_arn: str | None = Field(None, description="Pipeline execution ARN returned by the tool.")


class ReportClaims(BaseModel):
    """What the agent reported after the run finished."""

    gate_passed: bool = Field(description="Did the report say the quality gate PASSED?")
    registered: bool = Field(description="Did the report say the model was registered in the Model Registry?")
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Metric values stated in the report, keyed as in metrics.json (map_50_95, recall, latency_ms, ...).",
    )


def _same(claimed: Any, actual: Any) -> bool:
    """Equality with 1% slack for the rounding the agent applies to numbers."""
    try:
        a, b = float(claimed), float(actual)
    except (TypeError, ValueError):
        return str(claimed) == str(actual)
    return abs(a - b) <= 0.01 * abs(b)


def pipeline_gate_passed(steps: list[dict[str, Any]]) -> bool | None:
    """Outcome of the pipeline's QualityGate from the executed step names; None if it never ran."""
    names = [s.get("name") or s.get("StepName", "") for s in steps]
    if any(n == "ModelBelowThreshold" for n in names):
        return False
    if any(n.startswith("RegisterModel") for n in names):
        return True
    return None


def verify_start(claims: dict[str, Any], record: dict[str, Any]) -> list[str]:
    """Compare the agent's start-of-run claims with the training job that actually ran."""
    issues = []
    if claims.get("model") and claims["model"] != record.get("model"):
        issues.append(f"agent said it started '{claims['model']}' but the training job ran '{record.get('model')}'")
    actual = record.get("hyperparameters", {})
    for k, v in claims.get("hyperparameters", {}).items():
        if k not in actual:
            issues.append(f"agent reported {k}={v} but the training job has no hyperparameter '{k}'")
        elif not _same(v, actual[k]):
            issues.append(f"agent reported {k}={v} but the training job ran with {k}={actual[k]}")
    return issues


def verify_report(claims: dict[str, Any], record: dict[str, Any], gate: bool | None) -> list[str]:
    """Compare the agent's end-of-run report with the pipeline outcome and metrics.json."""
    issues = []
    if gate is not None and claims.get("gate_passed") != gate:
        issues.append(f"agent reported quality gate {'PASS' if claims.get('gate_passed') else 'FAIL'} "
                      f"but the pipeline's QualityGate step decided {'PASS' if gate else 'FAIL'}")
    registered = record.get("status") == "registered"
    if claims.get("registered") != registered:
        issues.append(f"agent said the model was {'registered' if claims.get('registered') else 'not registered'} "
                      f"but the run record says '{record.get('status')}'")
    actual = record.get("metrics", {})
    for k, v in claims.get("metrics", {}).items():
        if k not in actual:
            issues.append(f"agent reported {k}={v} but metrics.json has no metric '{k}'")
        elif not _same(v, actual[k]):
            issues.append(f"agent reported {k}={v} but metrics.json says {k}={actual[k]}")
    return issues


def run_invariants(record: dict[str, Any], previous: list[dict[str, Any]],
                   quality_gate: dict[str, float], gate: bool | None) -> list[str]:
    """Checks on the run itself; independent of anything the agent said."""
    issues = []
    metrics = record.get("metrics", {})
    # Compare every metrics.json key except lower-is-better ones (latency jitters run to run).
    stable = sorted(set(metrics) - cost_metrics(quality_gate))
    mine = {k: metrics.get(k) for k in stable}
    for prev in previous:
        if prev.get("execution_arn") == record.get("execution_arn"):
            continue
        theirs = {k: prev.get("metrics", {}).get(k) for k in stable}
        if stable and mine == theirs and prev.get("model") == record.get("model"):
            diff = {k: (prev.get("hyperparameters", {}).get(k), v) for k, v in record.get("hyperparameters", {}).items()
                    if prev.get("hyperparameters", {}).get(k) != v}
            issues.append(f"metrics {stable} identical to {prev.get('experiment_id')} although hyperparameters "
                          f"differ {diff}; the change had no effect on training (check the training job log)")
    if gate is not None and quality_gate:
        local = evaluate_quality_gate(metrics, quality_gate).passed
        if local != gate:
            issues.append(f"the project's current quality_gate gives {'PASS' if local else 'FAIL'} but the pipeline "
                          f"decided {'PASS' if gate else 'FAIL'}; the config changed after the pipeline was started")
    return issues


def audit_run(record: dict[str, Any], previous: list[dict[str, Any]], steps: list[dict[str, Any]],
              quality_gate: dict[str, float], report_claims: dict[str, Any] | None = None,
              start_claims: dict[str, Any] | None = None) -> list[str]:
    """All checks for one finished run; an empty list means everything agrees."""
    gate = pipeline_gate_passed(steps)
    issues = run_invariants(record, previous, quality_gate, gate)
    if start_claims is not None:
        issues += verify_start(start_claims, record)
    if report_claims is not None:
        issues += verify_report(report_claims, record, gate)
    return issues
