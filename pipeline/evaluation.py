"""Evaluation step: runs the trained model against the held-out test set and
emits a ``metrics.json`` used by the quality gate.

Uses AWS's prebuilt PyTorch image via ``PyTorchProcessor``; ``training/`` is
uploaded as the source dir so ``requirements.txt`` (Ultralytics) is installed.
"""

from __future__ import annotations

import json

import settings


def build_evaluation_step(cfg, session, model_artifacts):
    """Build a ProcessingStep that evaluates a trained model.

    Args:
        cfg: ProjectConfig.
        session: ``PipelineSession`` used to assemble the pipeline.
        model_artifacts: S3 URI (or step property) of the trained model.tar.gz.

    Returns:
        (ProcessingStep, PropertyFile) — the step and the property file that
        exposes ``metrics.json`` to downstream condition steps.
    """
    from sagemaker.processing import ProcessingInput, ProcessingOutput
    from sagemaker.pytorch.processing import PyTorchProcessor
    from sagemaker.workflow.properties import PropertyFile
    from sagemaker.workflow.steps import ProcessingStep

    processor = PyTorchProcessor(
        framework_version=settings.PYTORCH_FRAMEWORK_VERSION,
        py_version=settings.PYTORCH_PY_VERSION,
        role=settings.SAGEMAKER_EXECUTION_ROLE_ARN,
        instance_count=1,
        instance_type=cfg.training.instance_type,
        sagemaker_session=session,
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="metrics.json",
    )

    step_args = processor.run(
        code="evaluate.py",
        source_dir=str(settings.TRAINING_SOURCE_DIR),
        inputs=[
            ProcessingInput(source=model_artifacts, destination="/opt/ml/processing/model"),
            ProcessingInput(source=cfg.dataset.test, destination="/opt/ml/processing/test"),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"{settings.s3_uri(cfg.project_name, 'evaluation-results')}latest/",
            )
        ],
        # Class names let the script generate data.yaml for the test split.
        arguments=["--class-names", json.dumps(cfg.dataset.classes)],
    )
    step = ProcessingStep(name="EvaluateModel", step_args=step_args, property_files=[evaluation_report])
    return step, evaluation_report
