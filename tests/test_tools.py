"""Tests for the controlled tools' governance behaviour.

These verify the guardrails that matter most: the agent cannot start training
with an unapproved model or out-of-range hyperparameters, and registration is
refused unless the deterministic gate passes. No AWS access required — the
governance checks run before any AWS call.

Strands ``@tool`` objects are invoked directly here via their underlying
function so we test the logic, not the agent loop.
"""

import json

from agents.tools import model_tools, sagemaker_tools


def _call(tool, **kwargs):
    """Call a Strands tool's underlying python function directly."""
    fn = getattr(tool, "__wrapped__", None) or getattr(tool, "func", None) or tool
    return fn(**kwargs)


def test_available_models_are_restricted_to_project():
    out = json.loads(_call(model_tools.get_available_models, project_name="object_detection_reference"))
    ids = {m["id"] for m in out["approved_models"]}
    assert ids == {"yolo", "rt_detr"}  # faster_rcnn excluded for this project


def test_start_pipeline_rejects_unapproved_model():
    out = json.loads(_call(
        sagemaker_tools.start_training_pipeline,
        project_name="object_detection_reference",
        model_id="faster_rcnn",
        hyperparameters_json="{}",
    ))
    assert out["status"] == "rejected"
    assert any("not approved" in v for v in out["governance_violations"])


def test_available_models_expose_search_space_and_defaults():
    out = json.loads(_call(model_tools.get_available_models, project_name="object_detection_reference"))
    by_id = {m["id"]: m for m in out["approved_models"]}
    # yolo -> cnn space (Ultralytics names); rt_detr -> transformer space.
    assert by_id["yolo"]["hyperparameter_space"]["lr0"] == [0.01, 0.003, 0.005]
    assert by_id["yolo"]["default_hyperparameters"]["lr0"] == 0.01
    assert by_id["rt_detr"]["hyperparameter_space"]["batch"] == [16]
    assert by_id["rt_detr"]["default_hyperparameters"]["lr0"] == 0.0001
    # A fixed optimizer reaches the container but is not tunable (not in the space).
    assert by_id["yolo"]["default_hyperparameters"]["optimizer"] == "SGD"
    assert by_id["rt_detr"]["default_hyperparameters"]["optimizer"] == "AdamW"
    assert "optimizer" not in by_id["yolo"]["hyperparameter_space"]
    # Both models already have a recorded experiment in the reference project.
    assert by_id["yolo"]["must_use_defaults"] is False
    assert by_id["yolo"]["best_experiment_id"] == "exp-001"
    # Next run: one of 2 other lr0 or 2 other weight_decay values, changed from the best run.
    assert by_id["yolo"]["baseline_hyperparameters"]["lr0"] == 0.01
    assert len(by_id["yolo"]["allowed_next_configurations"]) == 4


def test_start_pipeline_rejects_value_outside_search_space():
    out = json.loads(_call(
        sagemaker_tools.start_training_pipeline,
        project_name="object_detection_reference",
        model_id="yolo",
        hyperparameters_json=json.dumps({"lr0": 5.0}),  # not in [0.01, 0.003, 0.005]
    ))
    assert out["status"] == "rejected"
    assert any("lr0" in v for v in out["governance_violations"])


def test_start_pipeline_rejects_untunable_key():
    out = json.loads(_call(
        sagemaker_tools.start_training_pipeline,
        project_name="object_detection_reference",
        model_id="yolo",
        hyperparameters_json=json.dumps({"epochs": 300}),  # not in the cnn space
    ))
    assert out["status"] == "rejected"
    assert any("epochs" in v and "not tunable" in v for v in out["governance_violations"])


def test_first_run_must_use_defaults():
    # object_detection_amg has no experiment history -> only defaults allowed.
    out = json.loads(_call(
        sagemaker_tools.start_training_pipeline,
        project_name="object_detection_amg",
        model_id="yolo",
        hyperparameters_json=json.dumps({"lr0": 0.003}),
    ))
    assert out["status"] == "rejected"
    assert any("first run" in v for v in out["governance_violations"])


def test_first_run_with_defaults_passes_governance():
    out = json.loads(_call(
        sagemaker_tools.start_training_pipeline,
        project_name="object_detection_amg",
        model_id="rt_detr",
        hyperparameters_json="{}",
    ))
    assert out["status"] == "blocked"  # governance ok, blocked only on missing role
    hp = out["validated_request"]["hyperparameters"]
    assert hp["lr0"] == 0.0001 and hp["batch"] == 16 and hp["epochs"] == 100
    assert hp["optimizer"] == "AdamW"  # catalog default reaches the training job


def test_only_one_hyperparameter_may_change_per_run():
    out = json.loads(_call(
        sagemaker_tools.start_training_pipeline,
        project_name="object_detection_reference",
        model_id="yolo",
        hyperparameters_json=json.dumps({"lr0": 0.003, "weight_decay": 0.001}),
    ))
    assert out["status"] == "rejected"
    assert any("only one hyperparameter" in v for v in out["governance_violations"])


def test_start_pipeline_valid_request_is_blocked_not_rejected():
    # Approved model with history + value from the search space: passes
    # governance, then blocks on the missing execution role.
    out = json.loads(_call(
        sagemaker_tools.start_training_pipeline,
        project_name="object_detection_reference",
        model_id="yolo",
        hyperparameters_json=json.dumps({"lr0": 0.003}),
    ))
    assert out["status"] == "blocked"
    assert out["validated_request"]["model"] == "yolo"
    # Unspecified keys are filled with the defaults.
    assert out["validated_request"]["hyperparameters"]["batch"] == 32


def _record_amg_yolo_run(n: int, lr0: float, map_50_95: float):
    from agents.tools import _backend

    _backend.save_experiment("object_detection_amg", {
        "experiment_id": f"exp-{n}", "execution_arn": f"arn:x/{n}", "model": "yolo", "status": "rejected",
        "hyperparameters": {"imgsz": 640, "batch": 32, "lr0": lr0, "weight_decay": 0.0005, "patience": 20},
        "metrics": {"map_50_95": map_50_95}, "is_champion": False,
    })


def _record_amg_default_yolo_run():
    _record_amg_yolo_run(1, 0.01, 0.45)


def _start_amg_yolo(**hyperparameters):
    return json.loads(_call(
        sagemaker_tools.start_training_pipeline,
        project_name="object_detection_amg", model_id="yolo", hyperparameters_json=json.dumps(hyperparameters),
    ))


def test_next_run_is_one_change_from_the_best_run_so_far():
    _record_amg_default_yolo_run()  # map 0.45 at lr0=0.01
    _record_amg_yolo_run(2, 0.003, 0.50)  # best so far, regardless of order
    _record_amg_yolo_run(3, 0.005, 0.40)
    # Two changes from the defaults, but only one from the best run: allowed.
    out = _start_amg_yolo(lr0=0.003, weight_decay=0.001)
    assert out["status"] == "blocked"
    # Omitted keys take the best run's values, so this is the same request.
    out = _start_amg_yolo(weight_decay=0.001)
    assert out["validated_request"]["hyperparameters"]["lr0"] == 0.003
    # Back to the default lr0 plus a weight_decay change = two changes from the best run.
    out = _start_amg_yolo(lr0=0.01, weight_decay=0.001)
    assert out["status"] == "rejected"
    assert any("best run so far (exp-2" in v for v in out["governance_violations"])
    models = json.loads(_call(model_tools.get_available_models, project_name="object_detection_amg"))
    yolo = next(m for m in models["approved_models"] if m["id"] == "yolo")
    assert yolo["best_experiment_id"] == "exp-2" and yolo["baseline_hyperparameters"]["lr0"] == 0.003
    # Every lr0 has been tried, so only the two other weight_decay values remain.
    assert [c["weight_decay"] for c in yolo["allowed_next_configurations"]] == [0.0001, 0.001]


def test_start_pipeline_rejects_already_trained_configuration():
    _record_amg_default_yolo_run()
    out = json.loads(_call(
        sagemaker_tools.start_training_pipeline,
        project_name="object_detection_amg", model_id="yolo", hyperparameters_json="{}",
    ))
    assert out["status"] == "rejected"
    assert any("already trained" in v for v in out["governance_violations"])
    # An untried single change passes governance (then blocks on the missing role).
    out = json.loads(_call(
        sagemaker_tools.start_training_pipeline,
        project_name="object_detection_amg", model_id="yolo", hyperparameters_json=json.dumps({"lr0": 0.003}),
    ))
    assert out["status"] == "blocked"


def test_available_models_report_tried_and_allowed_next_configurations():
    _record_amg_default_yolo_run()
    out = json.loads(_call(model_tools.get_available_models, project_name="object_detection_amg"))
    by_id = {m["id"]: m for m in out["approved_models"]}
    assert len(by_id["yolo"]["allowed_next_configurations"]) == 4 and len(by_id["yolo"]["tried_configurations"]) == 1
    # Before its first run a model has exactly one allowed configuration: the defaults.
    assert by_id["rt_detr"]["must_use_defaults"] is True
    assert by_id["rt_detr"]["allowed_next_configurations"] == [by_id["rt_detr"]["default_hyperparameters"]]


def test_register_refused_when_gate_fails():
    metrics = json.dumps({"map_50_95": 0.50, "recall": 0.50, "latency_ms": 30})
    out = json.loads(_call(
        sagemaker_tools.register_candidate_model,
        project_name="object_detection_reference",
        model_id="yolo",
        metrics_json=metrics,
    ))
    assert out["status"] == "rejected"


def test_quality_gate_tool_matches_thresholds():
    metrics = json.dumps({"map_50_95": 0.78, "recall": 0.87, "latency_ms": 29})
    out = json.loads(_call(
        sagemaker_tools.evaluate_model_quality_gate,
        project_name="object_detection_reference",
        metrics_json=metrics,
    ))
    assert out["passed"] is True
