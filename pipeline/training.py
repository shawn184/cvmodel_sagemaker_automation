"""Training step: builds the SageMaker PyTorch estimator for a chosen model.

Uses AWS's prebuilt PyTorch training container; ``training/`` is uploaded as the
source dir and its ``requirements.txt`` (Ultralytics) is installed at job start.
Compute is provisioned only while the training job runs and is torn down when it
completes — no permanently running GPU server.
"""

from __future__ import annotations

from typing import Any

import settings
from agents.tools._backend import catalog_by_id


def build_estimator(cfg, model_id: str, hyperparameters: dict[str, Any], session):
    """Create a PyTorch estimator for the given approved model + hyperparameters.

    Args:
        cfg: ProjectConfig.
        model_id: Approved model id.
        hyperparameters: Validated hyperparameters (within the project's space).
        session: ``PipelineSession`` used to assemble the pipeline.

    Returns:
        A ``sagemaker.pytorch.PyTorch`` estimator.
    """
    from sagemaker.pytorch import PyTorch  # lazy import

    catalog = catalog_by_id()
    model_meta = catalog.get(model_id, {})
    merged_hp = {**model_meta.get("default_hyperparameters", {}), **hyperparameters}
    # The container needs the architecture, base weights and class names
    # (used to generate data.yaml; label ids follow the list order).
    # model_id is only read back when the finished run is recorded.
    merged_hp.update({
        "model_id": model_id,
        "model_family": model_meta.get("family", model_id),
        "base_weights": model_meta.get("base_weights", ""),
        "task_type": cfg.task_type,
        "class_names": cfg.dataset.classes,  # SDK JSON-encodes the list
    })

    max_run = min(cfg.training.max_runtime_seconds, int(cfg.training.max_gpu_hours * 3600))

    return PyTorch(
        entry_point="train.py",
        source_dir=str(settings.TRAINING_SOURCE_DIR),
        framework_version=settings.PYTORCH_FRAMEWORK_VERSION,
        py_version=settings.PYTORCH_PY_VERSION,
        role=settings.SAGEMAKER_EXECUTION_ROLE_ARN,
        instance_count=1,
        instance_type=cfg.training.instance_type,
        max_run=max_run,
        output_path=settings.s3_uri(cfg.project_name, "models"),
        # Keep the uploaded source tarball with the other pipeline artifacts.
        code_location=settings.s3_uri(cfg.project_name, "pipeline-artifacts").rstrip("/"),
        hyperparameters=merged_hp,
        enable_network_isolation=False,  # jobs download pip deps + base weights
        sagemaker_session=session,
        disable_profiler=True
    )
