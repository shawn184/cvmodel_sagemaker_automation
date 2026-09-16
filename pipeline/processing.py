"""Dataset-validation processing step.

Runs the same basic dataset checks as the local ``validate_dataset`` tool, but
inside a managed SageMaker Processing job so the pipeline can fail fast before
provisioning GPU training compute. Uses AWS's prebuilt PyTorch CPU image; the
script only needs the standard library, so no dependency install happens here.
"""

from __future__ import annotations

import settings

VALIDATION_INSTANCE = "ml.m5.large"


def prebuilt_image_uri(instance_type: str, scope: str = "training") -> str:
    """AWS PyTorch container matching the framework version and instance type."""
    from sagemaker import image_uris

    return image_uris.retrieve(
        "pytorch",
        settings.AWS_REGION,
        version=settings.PYTORCH_FRAMEWORK_VERSION,
        py_version=settings.PYTORCH_PY_VERSION,
        instance_type=instance_type,
        image_scope=scope,
    )


def build_validation_step(cfg, session):
    """Build a ProcessingStep that validates the dataset in S3.

    Args:
        cfg: ProjectConfig.
        session: ``PipelineSession`` used to assemble the pipeline.

    Returns:
        A ``sagemaker.workflow.steps.ProcessingStep``.
    """
    from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
    from sagemaker.workflow.steps import ProcessingStep

    processor = ScriptProcessor(
        image_uri=prebuilt_image_uri(VALIDATION_INSTANCE),
        command=["python3"],
        role=settings.SAGEMAKER_EXECUTION_ROLE_ARN,
        instance_count=1,
        instance_type=VALIDATION_INSTANCE,
        sagemaker_session=session,
    )
    step_args = processor.run(
        code=str(settings.TRAINING_SOURCE_DIR / "validate.py"),
        inputs=[
            ProcessingInput(source=cfg.dataset.train, destination="/opt/ml/processing/train"),
            ProcessingInput(source=cfg.dataset.validation, destination="/opt/ml/processing/validation"),
        ],
        outputs=[ProcessingOutput(output_name="validation_report", source="/opt/ml/processing/report")],
    )
    return ProcessingStep(name="ValidateDataset", step_args=step_args)
