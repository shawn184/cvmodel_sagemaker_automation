"""Shared, non-agent-facing helpers used by the controlled tools.

Keeping this logic out of the ``@tool`` functions means the tools stay thin,
auditable wrappers ("narrowly defined tools").
"""

from __future__ import annotations

import json
from datetime import timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

import settings
from governance.quality_gate import evaluate_quality_gate


# ---------------------------------------------------------------------------
# Approved model catalogue
# ---------------------------------------------------------------------------
def load_catalog() -> list[dict[str, Any]]:
    path = settings.CATALOG_DIR / "approved_models.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(data.get("models", []))


def catalog_by_id() -> dict[str, dict[str, Any]]:
    return {m["id"]: m for m in load_catalog()}


# ---------------------------------------------------------------------------
# Hyperparameter search space / defaults per model
# ---------------------------------------------------------------------------
def search_space_for(cfg, model_id: str) -> dict[str, list[Any]]:
    """Project search space for a model (by id, else by its architecture)."""
    arch = catalog_by_id().get(model_id, {}).get("architecture", "")
    return cfg.search_space_for(model_id, arch)


def default_hyperparameters_for(cfg, model_id: str) -> dict[str, Any]:
    """Catalog defaults overridden by the first value of each search-space list."""
    arch = catalog_by_id().get(model_id, {}).get("architecture", "")
    catalog_defaults = catalog_by_id().get(model_id, {}).get("default_hyperparameters", {})
    return {**catalog_defaults, **cfg.default_hyperparameters_for(model_id, arch)}


def has_prior_experiments(project_name: str, model_id: str) -> bool:
    return any(e.get("model") == model_id for e in load_experiments(project_name))


def tried_configurations(cfg, model_id: str) -> list[dict[str, Any]]:
    """Hyperparameter sets already trained for a model, restricted to its search-space keys."""
    keys = search_space_for(cfg, model_id)
    return [
        {k: e.get("hyperparameters", {}).get(k) for k in keys}
        for e in load_experiments(cfg.project_name) if e.get("model") == model_id
    ]


def is_tried(cfg, model_id: str, resolved: dict[str, Any]) -> bool:
    keys = search_space_for(cfg, model_id)
    return {k: resolved.get(k) for k in keys} in tried_configurations(cfg, model_id)


def best_experiment_for(cfg, model_id: str) -> dict[str, Any] | None:
    """The model's recorded run with the highest primary metric; None before its first run."""
    primary = primary_metric(cfg)
    runs = [e for e in load_experiments(cfg.project_name) if e.get("model") == model_id]
    return max(runs, key=lambda e: e.get("metrics", {}).get(primary, float("-inf")), default=None)


def baseline_hyperparameters_for(cfg, model_id: str) -> dict[str, Any]:
    """Hyperparameters the next run is tuned from: the best run's, else the defaults."""
    defaults = default_hyperparameters_for(cfg, model_id)
    best = best_experiment_for(cfg, model_id)
    if not best:
        return defaults
    keys = search_space_for(cfg, model_id)
    return {**defaults, **{k: v for k, v in best.get("hyperparameters", {}).items() if k in keys}}


def untried_single_changes(cfg, model_id: str) -> list[dict[str, Any]]:
    """Configurations the next run may use: one value changed from the baseline, not yet trained."""
    if not has_prior_experiments(cfg.project_name, model_id):
        return [default_hyperparameters_for(cfg, model_id)]  # the first run must be the defaults
    baseline = baseline_hyperparameters_for(cfg, model_id)
    options = [{**baseline, key: v} for key, values in search_space_for(cfg, model_id).items() for v in values
               if v != baseline.get(key)]
    return [o for o in options if not is_tried(cfg, model_id, o)]


def primary_metric(cfg) -> str:
    """Metric used to rank models: the first target metric, else mAP50-95."""
    return next(iter(cfg.target_metrics), "map_50_95")


def target_met(metrics: dict[str, Any], target_metrics: dict[str, float]) -> bool:
    """True when every project target_metrics threshold is satisfied (same rules as the gate)."""
    return bool(target_metrics) and evaluate_quality_gate(metrics, target_metrics).passed


# ---------------------------------------------------------------------------
# Experiment history (local mock store for the POC).
# ---------------------------------------------------------------------------
def _experiments_path() -> Path:
    return settings.EXPERIMENTS_DIR / "experiments.json"


def load_experiments(project_name: str) -> list[dict[str, Any]]:
    path = _experiments_path()
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get(project_name, []))


def get_champion(project_name: str) -> dict[str, Any] | None:
    for exp in load_experiments(project_name):
        if exp.get("is_champion"):
            return exp
    return None


def experiment_for_execution(project_name: str, execution_arn: str) -> dict[str, Any] | None:
    return next((e for e in load_experiments(project_name) if e.get("execution_arn") == execution_arn), None)


def save_experiment(project_name: str, record: dict[str, Any]) -> bool:
    """Append a run record; returns False if that execution is already recorded."""
    path = _experiments_path()
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    exps = data.setdefault(project_name, [])
    if any(e.get("execution_arn") == record["execution_arn"] for e in exps):
        return False
    if record.get("is_champion"):  # a new champion demotes the previous one
        for e in exps:
            e["is_champion"] = False
    exps.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return True


# Hyperparameters the estimator adds for the container; not part of the experiment.
_RUN_ONLY_KEYS = {"model_id", "model_family", "base_weights", "task_type", "class_names"}


def _coerce(v: Any) -> Any:
    """SageMaker stores hyperparameters as JSON-encoded strings."""
    try:
        return json.loads(v) if isinstance(v, str) else v
    except ValueError:
        return v


def model_id_from_hyperparameters(hp: dict[str, Any]) -> str | None:
    """model_id is recorded explicitly; older runs only carry the catalog family."""
    if hp.get("model_id"):
        return hp["model_id"]
    return next((mid for mid, m in catalog_by_id().items() if m.get("family") == hp.get("model_family")), None)


def _step_metadata(steps: list[dict[str, Any]], name: str) -> dict[str, Any]:
    # ModelStep names its sub-step "RegisterModel-RegisterModel"; match on the prefix.
    return next((s.get("Metadata", {}) for s in steps if s.get("StepName", "").split("-")[0] == name), {})


def make_experiment_record(project_name: str, execution_arn: str, steps: list[dict[str, Any]],
                           training_job: dict[str, Any], metrics: dict[str, Any],
                           primary: str = "map_50_95") -> dict[str, Any] | None:
    """Build an experiments.json entry from a finished execution; None if it failed before the gate.

    ``steps`` is the ListPipelineExecutionSteps output, ``training_job`` the
    DescribeTrainingJob output of the TrainModel step. A registered run becomes
    champion when it beats the current champion on the ``primary`` metric.
    """
    gate = _step_metadata(steps, "QualityGate").get("Condition", {}).get("Outcome")
    if gate not in ("True", "False"):
        return None
    hp = {k: _coerce(v) for k, v in training_job.get("HyperParameters", {}).items() if not k.startswith("sagemaker_")}
    register_arn = _step_metadata(steps, "RegisterModel").get("RegisterModel", {}).get("Arn", "")
    registered = gate == "True" and bool(register_arn)
    end = training_job.get("TrainingEndTime")
    champion = get_champion(project_name)
    best_so_far = champion.get("metrics", {}).get(primary, float("-inf")) if champion else float("-inf")
    return {
        "experiment_id": f"exp-{execution_arn.rsplit('/', 1)[1]}",
        "execution_arn": execution_arn,
        "model": model_id_from_hyperparameters(hp),
        "model_version": int(register_arn.rsplit("/", 1)[1]) if registered else None,
        "hyperparameters": {k: v for k, v in hp.items() if k not in _RUN_ONLY_KEYS},
        "metrics": metrics,
        "model_artifact": training_job.get("ModelArtifacts", {}).get("S3ModelArtifacts"),
        "status": "registered" if registered else "rejected",
        # Keep the best: a registered run that beats the champion on the primary metric takes over.
        "is_champion": registered and metrics.get(primary, float("-inf")) > best_so_far,
        "training_date": end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if end else None,
    }


def collect_run_record(project_name: str, execution_arn: str) -> dict[str, Any] | None:
    """Read a finished execution from SageMaker/S3 and build its experiment record."""
    import boto3

    import config

    cfg = config.load_project_config(project_name)
    sm = boto3.client("sagemaker", region_name=settings.AWS_REGION)
    steps = sm.list_pipeline_execution_steps(PipelineExecutionArn=execution_arn)["PipelineExecutionSteps"]
    train_arn = _step_metadata(steps, "TrainModel").get("TrainingJob", {}).get("Arn")
    if not train_arn or _step_metadata(steps, "QualityGate").get("Condition", {}).get("Outcome") is None:
        return None
    job = sm.describe_training_job(TrainingJobName=train_arn.rsplit("/", 1)[1])

    s3 = boto3.client("s3", region_name=settings.AWS_REGION)
    key = f"{settings.s3_prefix(project_name, 'evaluation-results')}latest/metrics.json"
    metrics = json.loads(s3.get_object(Bucket=settings.S3_BUCKET, Key=key)["Body"].read()).get("metrics", {})
    return make_experiment_record(project_name, execution_arn, steps, job, metrics, primary_metric(cfg))


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------
def parse_s3_uri(uri: str) -> tuple[str, str]:
    p = urlparse(uri)
    return p.netloc, p.path.lstrip("/")


def s3_prefix_exists(uri: str) -> tuple[bool, int]:
    """Return (exists, object_count) for an s3:// prefix. Best-effort.

    Returns (False, -1) if S3 cannot be reached / access denied, so callers can
    distinguish "empty" from "could not check".
    """
    try:
        import boto3

        bucket, prefix = parse_s3_uri(uri)
        s3 = boto3.client("s3", region_name=settings.AWS_REGION)
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        return resp.get("KeyCount", 0) > 0, resp.get("KeyCount", 0)
    except Exception:
        return False, -1
