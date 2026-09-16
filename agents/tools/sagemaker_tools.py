"""SageMaker-facing controlled tools.

Exposes the *only* infrastructure actions the agent can take:
  * start_training_pipeline(project_name, model_id, hyperparameters_json)
  * get_pipeline_status(execution_arn)
  * get_training_metrics(project_name, execution_arn)
  * evaluate_model_quality_gate(project_name, metrics_json)
  * register_candidate_model(project_name, model_id, metrics_json, ...)

GOVERNANCE
----------
Every tool validates its inputs against approved configuration *before*
touching AWS. The agent receives capabilities, not unrestricted infrastructure
access. The agent cannot create EC2, modify IAM, delete S3, etc. — those calls
simply do not exist as tools.

Each tool degrades gracefully: if the caller's IAM identity lacks the required
SageMaker permissions, the tool returns a clear "blocked" status naming the
problem rather than crashing. This lets the whole orchestration flow be
demonstrated end-to-end, with the real SageMaker calls activating the moment a
suitable execution role and permissions are granted.
"""

from __future__ import annotations

import json

from strands import tool

import config
import settings
from agents.tools import _backend
from governance.quality_gate import evaluate_quality_gate


def _validate_choice(cfg, model_id: str, hyperparameters: dict) -> tuple[list[str], dict]:
    """Return (governance violations, resolved hyperparameters).

    Resolved = the model's best run so far (defaults before its first run)
    overridden by the proposal. The first run of a model in a project must use
    the defaults exactly; later runs may change exactly one hyperparameter from
    the best run and may not repeat a recorded run.
    """
    errors: list[str] = []
    catalog = _backend.catalog_by_id()
    allowed = cfg.approved_models or list(catalog)
    if model_id not in catalog:
        errors.append(f"model '{model_id}' is not in the approved catalogue {sorted(catalog)}")
        return errors, hyperparameters
    if model_id not in allowed:
        errors.append(f"model '{model_id}' is not approved for project '{cfg.project_name}' ({allowed})")

    space = _backend.search_space_for(cfg, model_id)
    for key, val in hyperparameters.items():
        if key not in space:
            errors.append(f"hyperparameter '{key}' is not tunable for '{model_id}' (allowed: {sorted(space)})")
        elif val not in space[key]:
            errors.append(f"hyperparameter {key}={val} not in allowed set {space[key]}")

    baseline = _backend.baseline_hyperparameters_for(cfg, model_id)
    resolved = {**baseline, **hyperparameters}
    changed = {k: v for k, v in resolved.items() if baseline.get(k) != v}
    best = _backend.best_experiment_for(cfg, model_id)
    if changed and not best:
        errors.append(
            f"first run of '{model_id}' in '{cfg.project_name}' must use the default "
            f"hyperparameters {baseline}; got overrides {changed}"
        )
    elif len(changed) > 1:
        tunable = {k: baseline[k] for k in space}
        errors.append(
            f"only one hyperparameter may differ from the model's best run so far "
            f"({best['experiment_id']}: {tunable}) per run; got {changed}"
        )
    if not errors and _backend.is_tried(cfg, model_id, resolved):
        errors.append(
            f"configuration {resolved} was already trained for '{model_id}' in "
            f"'{cfg.project_name}'; choose an untried single change (see tried_configurations)"
        )
    return errors, resolved


@tool
def start_training_pipeline(project_name: str, model_id: str, hyperparameters_json: str = "{}") -> str:
    """Start the SageMaker training/evaluation pipeline for a project.

    This is the single controlled entry point that lets the agent request
    execution. The agent supplies an *approved* model and hyperparameters within
    the *approved* ranges; the tool validates both before calling SageMaker
    ``StartPipelineExecution``.

    Args:
        project_name: Short project name.
        model_id: Approved model id (e.g. "yolo", "rt_detr").
        hyperparameters_json: JSON object of hyperparameters within the model's
            search space (see ``get_available_models``). Omitted keys take the
            value of the model's best run so far (the defaults before its first
            run). The first run of a model must use the defaults ("{}"); later
            runs may change exactly one hyperparameter from that best run and
            may not repeat a trained configuration.

    Returns:
        JSON with the pipeline execution ARN, or a ``blocked``/``rejected`` status
        with the reason.
    """
    try:
        cfg = config.load_project_config(project_name)
    except config.ConfigError as e:
        return json.dumps({"status": "rejected", "reason": str(e)})

    try:
        hyperparameters = json.loads(hyperparameters_json or "{}")
    except json.JSONDecodeError as e:
        return json.dumps({"status": "rejected", "reason": f"invalid hyperparameters_json: {e}"})

    violations, hyperparameters = _validate_choice(cfg, model_id, hyperparameters)
    if violations:
        return json.dumps({"status": "rejected", "governance_violations": violations})

    if not settings.SAGEMAKER_EXECUTION_ROLE_ARN:
        return json.dumps({
            "status": "blocked",
            "reason": "No SageMaker execution role configured (env CV_MLOPS_SAGEMAKER_ROLE).",
            "validated_request": {"project": project_name, "model": model_id, "hyperparameters": hyperparameters},
            "next_step": "Create the execution role (infrastructure/iam) and set CV_MLOPS_SAGEMAKER_ROLE.",
        })

    try:
        import boto3

        from pipeline.pipeline import build_pipeline

        pipeline = build_pipeline(cfg, model_id, hyperparameters)
        pipeline.upsert(role_arn=settings.SAGEMAKER_EXECUTION_ROLE_ARN)
        execution = pipeline.start()
        return json.dumps({
            "status": "started",
            "project": project_name,
            "model": model_id,
            "hyperparameters": hyperparameters,
            "execution_arn": execution.arn,
        })
    except Exception as e:  # noqa: BLE001 - surface AWS/permission errors to the agent
        return json.dumps({
            "status": "blocked",
            "reason": f"{type(e).__name__}: {e}",
            "validated_request": {"project": project_name, "model": model_id, "hyperparameters": hyperparameters},
        })


@tool
def get_pipeline_status(execution_arn: str) -> str:
    """Return the status of a running SageMaker pipeline execution.

    Args:
        execution_arn: ARN returned by ``start_training_pipeline``.

    Returns:
        JSON with the execution status, or a ``blocked`` status on access errors.
    """
    try:
        import boto3

        sm = boto3.client("sagemaker", region_name=settings.AWS_REGION)
        resp = sm.describe_pipeline_execution(PipelineExecutionArn=execution_arn)
        # Per-step view so a "Step failure" reason can be traced to the actual step.
        steps = sm.list_pipeline_execution_steps(PipelineExecutionArn=execution_arn).get("PipelineExecutionSteps", [])
        return json.dumps({
            "status": resp.get("PipelineExecutionStatus"),
            "execution_arn": execution_arn,
            "failure_reason": resp.get("FailureReason"),
            "steps": [
                {"name": s.get("StepName"), "status": s.get("StepStatus"), "failure_reason": s.get("FailureReason")}
                for s in reversed(steps)
            ],
        })
    except Exception as e:  # noqa: BLE001
        return json.dumps({"status": "blocked", "reason": f"{type(e).__name__}: {e}"})


@tool
def get_training_metrics(project_name: str, execution_arn: str = "") -> str:
    """Retrieve evaluation metrics produced by a pipeline run.

    In the POC these are read from the project's evaluation-results S3 prefix
    (``metrics.json`` written by the evaluation step). Falls back to the most
    recent recorded experiment if S3 is unreachable, so the agent flow can be
    demonstrated locally.

    Args:
        project_name: Short project name.
        execution_arn: Optional pipeline execution ARN to locate the exact run.

    Returns:
        JSON metrics object.
    """
    try:
        cfg = config.load_project_config(project_name)
    except config.ConfigError as e:
        return json.dumps({"error": str(e)})

    key = f"{settings.s3_prefix(cfg.project_name, 'evaluation-results')}latest/metrics.json"
    try:
        import boto3

        s3 = boto3.client("s3", region_name=settings.AWS_REGION)
        obj = s3.get_object(Bucket=settings.S3_BUCKET, Key=key)
        return obj["Body"].read().decode("utf-8")
    except Exception as e:  # noqa: BLE001
        exps = _backend.load_experiments(project_name)
        if exps:
            return json.dumps({
                "source": "local_experiment_history (S3 unavailable)",
                "reason": f"{type(e).__name__}: {e}",
                "metrics": exps[-1]["metrics"],
            })
        return json.dumps({"status": "blocked", "reason": f"{type(e).__name__}: {e}"})


@tool
def evaluate_model_quality_gate(project_name: str, metrics_json: str) -> str:
    """Apply the DETERMINISTIC quality gate to candidate metrics.

    This is the authoritative PASS/FAIL. It is pure program logic — the LLM's
    opinion does not enter into it.

    Args:
        project_name: Short project name.
        metrics_json: JSON metrics object for the candidate model.

    Returns:
        JSON with ``passed`` (bool) and a per-threshold breakdown.
    """
    try:
        cfg = config.load_project_config(project_name)
        metrics = json.loads(metrics_json)
    except (config.ConfigError, json.JSONDecodeError) as e:
        return json.dumps({"passed": False, "error": str(e)})

    result = evaluate_quality_gate(metrics, cfg.quality_gate)
    return json.dumps({
        "project": project_name,
        "passed": result.passed,
        "thresholds": cfg.quality_gate,
        "checks": result.checks,
        "summary": result.summary,
    }, indent=2)


@tool
def register_candidate_model(
    project_name: str,
    model_id: str,
    metrics_json: str,
    model_artifact_s3_uri: str = "",
) -> str:
    """Register a candidate model in the SageMaker Model Registry.

    Only permitted when the deterministic quality gate PASSES — the gate is
    re-checked here so registration cannot be talked into by the LLM. Registered
    models are created with ``PendingManualApproval`` so human approval remains
    in the loop.

    Args:
        project_name: Short project name.
        model_id: Approved model id that produced the artifact.
        metrics_json: JSON metrics object for the candidate.
        model_artifact_s3_uri: s3:// URI of the model.tar.gz artifact.

    Returns:
        JSON with the model package ARN, or a ``rejected``/``blocked`` status.
    """
    try:
        cfg = config.load_project_config(project_name)
        metrics = json.loads(metrics_json)
    except (config.ConfigError, json.JSONDecodeError) as e:
        return json.dumps({"status": "rejected", "reason": str(e)})

    gate = evaluate_quality_gate(metrics, cfg.quality_gate)
    if not gate.passed:
        return json.dumps({
            "status": "rejected",
            "reason": "quality gate did not pass; registration refused",
            "gate": gate.summary,
        })

    if not settings.SAGEMAKER_EXECUTION_ROLE_ARN:
        return json.dumps({
            "status": "blocked",
            "reason": "No SageMaker execution role configured; cannot register.",
            "would_register": {
                "model_package_group": cfg.model_registry_group,
                "model": model_id,
                "metrics": metrics,
                "approval_status": "PendingManualApproval",
            },
        })

    try:
        from pipeline.registry import register_model_package

        arn = register_model_package(cfg, model_id, metrics, model_artifact_s3_uri)
        return json.dumps({
            "status": "registered",
            "model_package_group": cfg.model_registry_group,
            "model_package_arn": arn,
            "approval_status": "PendingManualApproval",
        })
    except Exception as e:  # noqa: BLE001
        return json.dumps({"status": "blocked", "reason": f"{type(e).__name__}: {e}"})
