"""Project configuration loading + validation.

The whole framework is *configuration-driven*: a new computer-vision project is
onboarded by dropping a YAML file into ``projects/`` rather than editing code.
This module loads those files into a typed :class:`ProjectConfig` and validates
them.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

import settings


class ConfigError(ValueError):
    """Raised when a project configuration file is missing or malformed."""


@dataclass
class DatasetConfig:
    train: str
    validation: str
    test: str
    annotation_format: str = "yolo"  # yolo | coco | pascal_voc
    classes: list[str] = field(default_factory=list)


@dataclass
class TrainingConfig:
    instance_type: str = "ml.g5.xlarge"
    max_gpu_hours: float = 5.0
    max_runtime_seconds: int = 18000


@dataclass
class ProjectConfig:
    project_name: str
    task_type: str  # object_detection | classification | ...
    dataset: DatasetConfig
    target_metrics: dict[str, float]
    training: TrainingConfig
    # Which entries of the approved model catalogue this project may use.
    approved_models: list[str] = field(default_factory=list)
    # Deterministic quality-gate thresholds.
    quality_gate: dict[str, float] = field(default_factory=dict)
    # Search space keyed by architecture (cnn/transformer) or model id.
    # Each value is a list of allowed values; the first one is the default.
    hyperparameter_space: dict[str, dict[str, list[Any]]] = field(default_factory=dict)

    def search_space_for(self, model_id: str, architecture: str = "") -> dict[str, list[Any]]:
        """Allowed values per hyperparameter; model id entry wins over architecture."""
        for key in (model_id, architecture):
            if key and key in self.hyperparameter_space:
                return dict(self.hyperparameter_space[key])
        return {}

    def default_hyperparameters_for(self, model_id: str, architecture: str = "") -> dict[str, Any]:
        """Default config: the first listed value of every hyperparameter."""
        return {k: v[0] for k, v in self.search_space_for(model_id, architecture).items()}

    # ---- convenience -----------------------------------------------------
    @property
    def model_registry_group(self) -> str:
        """SageMaker Model Package Group name for this project."""
        return "".join(w.capitalize() for w in self.project_name.split("_"))

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_REQUIRED_TOP_LEVEL = ("project_name", "task_type", "dataset", "target_metrics")


def _validate_raw(raw: dict[str, Any], source: str) -> None:
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: top-level YAML must be a mapping.")
    missing = [k for k in _REQUIRED_TOP_LEVEL if k not in raw]
    if missing:
        raise ConfigError(f"{source}: missing required keys: {', '.join(missing)}")
    for split in ("train", "validation", "test"):
        if split not in raw["dataset"]:
            raise ConfigError(f"{source}: dataset.{split} is required.")
    for group, space in (raw.get("hyperparameter_space") or {}).items():
        if not isinstance(space, dict):
            raise ConfigError(f"{source}: hyperparameter_space.{group} must be a mapping.")
        for hp, values in space.items():
            if not isinstance(values, list) or not values:
                raise ConfigError(
                    f"{source}: hyperparameter_space.{group}.{hp} must be a non-empty list "
                    "(first value = default)."
                )


def load_project_config(name_or_path: str | Path) -> ProjectConfig:
    """Load a project config by short name (``vehicle_detection``) or by path."""
    path = Path(name_or_path)
    if not path.exists():
        candidate = settings.PROJECTS_DIR / f"{name_or_path}.yaml"
        if not candidate.exists():
            raise ConfigError(
                f"No project config found for '{name_or_path}'. "
                f"Looked at {path} and {candidate}."
            )
        path = candidate

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    _validate_raw(raw, path.name)

    ds = raw["dataset"]
    tr = raw.get("training", {}) or {}
    return ProjectConfig(
        project_name=raw["project_name"],
        task_type=raw["task_type"],
        dataset=DatasetConfig(
            train=ds["train"],
            validation=ds["validation"],
            test=ds["test"],
            annotation_format=ds.get("annotation_format", "yolo"),
            classes=list(ds.get("classes", [])),
        ),
        target_metrics=dict(raw.get("target_metrics", {})),
        training=TrainingConfig(
            instance_type=tr.get("instance_type", "ml.g5.xlarge"),
            max_gpu_hours=float(tr.get("max_gpu_hours", 5.0)),
            max_runtime_seconds=int(tr.get("max_runtime_seconds", 18000)),
        ),
        approved_models=list(raw.get("approved_models", [])),
        quality_gate=dict(raw.get("quality_gate", {})),
        hyperparameter_space=dict(raw.get("hyperparameter_space", {})),
    )


def list_projects() -> list[str]:
    """Names of all configured projects."""
    return sorted(p.stem for p in settings.PROJECTS_DIR.glob("*.yaml"))
