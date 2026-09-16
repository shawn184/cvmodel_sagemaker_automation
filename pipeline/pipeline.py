"""Assembles the SageMaker Pipeline DAG.

    Validate -> Train -> Evaluate -> QualityGate(Condition) -> Register | Fail

The pipeline is the ML *process control* layer: it defines what sequence must
execute. The orchestrator agent only starts it (via a controlled tool) and reads
its status/metrics — it never edits the DAG at runtime.

All jobs run on AWS's prebuilt PyTorch container (see settings.py); the code in
``training/`` is uploaded to S3 by the SDK when the pipeline is upserted.
"""

from __future__ import annotations

from typing import Any

import settings
from pipeline.evaluation import build_evaluation_step
from pipeline.processing import build_validation_step
from pipeline.registry import (
    build_quality_gate_condition,
    build_register_step,
)
from pipeline.training import build_estimator


def build_pipeline(cfg, model_id: str, hyperparameters: dict[str, Any]):
    """Build (but do not start) the SageMaker Pipeline for a training request.

    Args:
        cfg: ProjectConfig.
        model_id: Approved model id.
        hyperparameters: Validated hyperparameters.

    Returns:
        A ``sagemaker.workflow.pipeline.Pipeline``.
    """
    from sagemaker.inputs import TrainingInput
    from sagemaker.workflow.fail_step import FailStep
    from sagemaker.workflow.pipeline import Pipeline
    from sagemaker.workflow.pipeline_context import PipelineSession
    from sagemaker.workflow.steps import TrainingStep

    # Everything the SDK uploads (code, reports) lands under our bucket/prefix.
    session = PipelineSession(
        default_bucket=settings.S3_BUCKET,
        default_bucket_prefix=settings.s3_prefix(cfg.project_name, "pipeline-artifacts").rstrip("/"),
    )

    # 1. Validate dataset
    validate_step = build_validation_step(cfg, session)

    # 2. Train
    estimator = build_estimator(cfg, model_id, hyperparameters, session)
    train_step = TrainingStep(
        name="TrainModel",
        step_args=estimator.fit(inputs={
            "train": TrainingInput(s3_data=cfg.dataset.train),
            "validation": TrainingInput(s3_data=cfg.dataset.validation),
        }),
        depends_on=[validate_step],
    )
    model_artifacts = train_step.properties.ModelArtifacts.S3ModelArtifacts

    # 3. Evaluate
    eval_step, eval_report = build_evaluation_step(cfg, session, model_artifacts)

    # 4/5. Register or Fail, gated by the deterministic quality gate
    register_step = build_register_step(cfg, model_artifacts, session)
    fail_step = FailStep(
        name="ModelBelowThreshold",
        error_message="Candidate model did not satisfy the project quality gate.",
    )
    gate_step = build_quality_gate_condition(cfg, eval_step, eval_report, register_step, fail_step)

    return Pipeline(
        name=f"{cfg.model_registry_group}TrainingPipeline",
        steps=[validate_step, train_step, eval_step, gate_step],
        sagemaker_session=session,
    )
