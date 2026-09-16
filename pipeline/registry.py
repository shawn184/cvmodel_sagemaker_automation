"""Model registration + the deterministic quality-gate condition step.

Two entry points:
  * ``build_register_step`` / ``build_quality_gate_condition`` — used when
    assembling the SageMaker Pipeline DAG.
  * ``register_model_package`` — a direct registration call used by the
    ``register_candidate_model`` tool once a candidate has passed the gate.

Registered models are created with ``PendingManualApproval`` so a human remains
in the approval loop.
"""

from __future__ import annotations

from typing import Any

import settings
from governance.quality_gate import gate_rules


def build_quality_gate_condition(cfg, evaluation_step, evaluation_report, register_step, fail_step):
    """Build the ConditionStep enforcing the deterministic quality gate.

    Mirrors ``governance/quality_gate.py`` as pipeline conditions so the gate is
    enforced identically inside the managed pipeline.
    """
    from sagemaker.workflow.condition_step import ConditionStep
    from sagemaker.workflow.conditions import (
        ConditionGreaterThanOrEqualTo,
        ConditionLessThanOrEqualTo,
    )
    from sagemaker.workflow.functions import JsonGet

    def metric(name: str):
        return JsonGet(
            step_name=evaluation_step.name,
            property_file=evaluation_report,
            json_path=f"metrics.{name}",
        )

    # Same rules as the local gate: `<metric>` -> >=, `max_<metric>` -> <=.
    conditions = []
    for metric_key, op, threshold in gate_rules(cfg.quality_gate):
        condition = ConditionGreaterThanOrEqualTo if op == ">=" else ConditionLessThanOrEqualTo
        conditions.append(condition(left=metric(metric_key), right=threshold))

    return ConditionStep(
        name="QualityGate",
        conditions=conditions,
        if_steps=[register_step],
        else_steps=[fail_step],
    )


def build_register_step(cfg, model_artifacts, session):
    """Build the ModelStep that registers the artifact in the Model Registry.

    A plain ``Model`` on the prebuilt PyTorch image is registered (no inference
    code, no repack step); serving is out of scope for this framework.
    """
    from sagemaker.model import Model
    from sagemaker.model_metrics import MetricsSource, ModelMetrics
    from sagemaker.workflow.model_step import ModelStep

    from pipeline.processing import prebuilt_image_uri

    metrics_uri = f"{settings.s3_uri(cfg.project_name, 'evaluation-results')}latest/metrics.json"
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(content_type="application/json", s3_uri=metrics_uri)
    )
    model = Model(
        image_uri=prebuilt_image_uri(cfg.training.instance_type),
        model_data=model_artifacts,
        role=settings.SAGEMAKER_EXECUTION_ROLE_ARN,
        sagemaker_session=session,
    )
    register_args = model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large", cfg.training.instance_type],
        transform_instances=["ml.m5.large"],
        model_package_group_name=cfg.model_registry_group,
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
    )
    return ModelStep(name="RegisterModel", step_args=register_args)


def register_model_package(cfg, model_id: str, metrics: dict[str, Any], model_artifact_s3_uri: str) -> str:
    """Directly register a model package (used by the controlled tool).

    Returns the created model package ARN.
    """
    import boto3

    sm = boto3.client("sagemaker", region_name=settings.AWS_REGION)

    # Ensure the model package group exists (idempotent).
    try:
        sm.create_model_package_group(
            ModelPackageGroupName=cfg.model_registry_group,
            ModelPackageGroupDescription=f"Candidate models for {cfg.project_name}",
        )
    except sm.exceptions.ClientError:
        pass  # already exists

    resp = sm.create_model_package(
        ModelPackageGroupName=cfg.model_registry_group,
        ModelPackageDescription=f"{model_id} candidate for {cfg.project_name}",
        ModelApprovalStatus="PendingManualApproval",
        CustomerMetadataProperties={
            "model_id": model_id,
            "project": cfg.project_name,
            **{k: str(v) for k, v in metrics.items()},
        },
    )
    return resp["ModelPackageArn"]
