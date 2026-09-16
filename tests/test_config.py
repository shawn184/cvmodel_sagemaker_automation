"""Tests for project configuration loading + validation."""

import config


def test_reference_project_loads():
    cfg = config.load_project_config("object_detection_reference")
    assert cfg.project_name == "object_detection_reference"
    assert cfg.task_type == "object_detection"
    assert cfg.dataset.train.startswith("s3://")
    assert "yolo" in cfg.approved_models
    assert cfg.quality_gate["map_50_95"] == 0.75


def test_model_registry_group_is_camelcase():
    cfg = config.load_project_config("object_detection_reference")
    assert cfg.model_registry_group == "ObjectDetectionReference"


def test_missing_project_raises():
    try:
        config.load_project_config("does_not_exist")
    except config.ConfigError as e:
        assert "No project config" in str(e)
    else:
        raise AssertionError("expected ConfigError")


def test_list_projects_includes_samples():
    projects = config.list_projects()
    assert "object_detection_reference" in projects
    assert "object_detection_amg" in projects


def test_search_space_resolves_by_architecture_then_model_id():
    cfg = config.load_project_config("object_detection_reference")
    assert cfg.search_space_for("yolo", "cnn")["patience"] == [20]
    assert cfg.default_hyperparameters_for("rt_detr", "transformer")["lr0"] == 0.0001
    assert cfg.search_space_for("unknown", "unknown") == {}
    # A model-id entry takes precedence over the architecture entry.
    cfg.hyperparameter_space["yolo"] = {"lr0": [0.005]}
    assert cfg.default_hyperparameters_for("yolo", "cnn") == {"lr0": 0.005}
