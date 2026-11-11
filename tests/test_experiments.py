"""Tests for automatic experiment recording after a pipeline run (no AWS)."""

import json
from datetime import datetime, timezone

import settings
from agents.tools import _backend

PROJECT = "object_detection_amg"
ARN = f"arn:aws:sagemaker:ap-southeast-1:000000000000:pipeline/ObjectDetectionAmgTrainingPipeline/execution/ipx68r86ftpy"
METRICS = {"map_50": 0.61, "map_50_95": 0.42, "precision": 0.7, "recall": 0.58, "f1": 0.63, "latency_ms": 12.3, "model_size_mb": 5.4}
JOB = {
    "HyperParameters": {
        "batch": "32", "epochs": "100", "imgsz": "640", "lr0": "0.01", "weight_decay": "0.0005", "patience": "20",
        "model_family": '"yolo"', "base_weights": '"yolo11n.pt"', "task_type": '"object_detection"',
        "class_names": '["person"]', "sagemaker_program": '"train.py"',
    },
    "ModelArtifacts": {"S3ModelArtifacts": "s3://bucket/models/x/output/model.tar.gz"},
    "TrainingEndTime": datetime(2026, 9, 8, 8, 31, 17, tzinfo=timezone.utc),
}


def _steps(gate="True", registered=True):
    steps = [
        {"StepName": "ValidateDataset", "Metadata": {"ProcessingJob": {"Arn": "arn:x:processing-job/v"}}},
        {"StepName": "TrainModel", "Metadata": {"TrainingJob": {"Arn": "arn:x:training-job/pipelines-x-TrainModel-abc"}}},
        {"StepName": "EvaluateModel", "Metadata": {"ProcessingJob": {"Arn": "arn:x:processing-job/e"}}},
    ]
    if gate is not None:
        steps.append({"StepName": "QualityGate", "Metadata": {"Condition": {"Outcome": gate}}})
    if registered:
        steps.append({"StepName": "RegisterModel-RegisterModel",
                      "Metadata": {"RegisterModel": {"Arn": "arn:x:model-package/ObjectDetectionAmg/3"}}})
    return steps


def test_registered_run_record():
    rec = _backend.make_experiment_record(PROJECT, ARN, _steps(), JOB, METRICS)
    assert rec["experiment_id"] == "exp-ipx68r86ftpy"
    assert rec["model"] == "yolo"  # inferred from model_family for runs without model_id
    assert rec["model_version"] == 3
    assert rec["status"] == "registered" and rec["is_champion"] is True
    assert rec["hyperparameters"] == {"batch": 32, "epochs": 100, "imgsz": 640, "lr0": 0.01,
                                      "weight_decay": 0.0005, "patience": 20}
    assert rec["metrics"] == METRICS
    assert rec["model_artifact"] == "s3://bucket/models/x/output/model.tar.gz"
    assert rec["training_date"] == "2026-09-08T08:31:17Z"


def test_gate_failed_run_is_recorded_as_rejected():
    rec = _backend.make_experiment_record(PROJECT, ARN, _steps(gate="False", registered=False), JOB, METRICS)
    assert rec["status"] == "rejected"
    assert rec["model_version"] is None and rec["is_champion"] is False


def test_run_failed_before_gate_is_not_recorded():
    assert _backend.make_experiment_record(PROJECT, ARN, _steps(gate=None, registered=False), JOB, {}) is None


def test_explicit_model_id_wins_over_family():
    job = {**JOB, "HyperParameters": {**JOB["HyperParameters"], "model_id": '"rt_detr"'}}
    rec = _backend.make_experiment_record(PROJECT, ARN, _steps(), job, METRICS)
    assert rec["model"] == "rt_detr"
    assert "model_id" not in rec["hyperparameters"]


def test_save_experiment_appends_once_and_unlocks_next_run():
    rec = _backend.make_experiment_record(PROJECT, ARN, _steps(), JOB, METRICS)
    assert _backend.save_experiment(PROJECT, rec) is True
    assert _backend.save_experiment(PROJECT, rec) is False  # same execution: no duplicate
    data = json.loads((settings.EXPERIMENTS_DIR / "experiments.json").read_text())
    assert len(data[PROJECT]) == 1
    assert len(data["object_detection_reference"]) == 2  # other projects untouched
    assert _backend.has_prior_experiments(PROJECT, "yolo") is True
    assert _backend.get_champion(PROJECT)["experiment_id"] == "exp-ipx68r86ftpy"


def test_better_registered_run_replaces_champion():
    _backend.save_experiment(PROJECT, _backend.make_experiment_record(PROJECT, ARN, _steps(), JOB, METRICS))
    better = {**METRICS, "map_50_95": 0.50}
    second = _backend.make_experiment_record(PROJECT, ARN.replace("ipx68r86ftpy", "abc123def456"), _steps(), JOB, better)
    assert second["is_champion"] is True
    _backend.save_experiment(PROJECT, second)
    exps = _backend.load_experiments(PROJECT)
    assert [e["is_champion"] for e in exps] == [False, True]  # old champion demoted


def test_worse_or_equal_registered_run_keeps_champion():
    _backend.save_experiment(PROJECT, _backend.make_experiment_record(PROJECT, ARN, _steps(), JOB, METRICS))
    same = _backend.make_experiment_record(PROJECT, ARN.replace("ipx68r86ftpy", "abc123def456"), _steps(), JOB, METRICS)
    worse = _backend.make_experiment_record(PROJECT, ARN.replace("ipx68r86ftpy", "zzz999"), _steps(), JOB,
                                            {**METRICS, "map_50_95": 0.10})
    assert same["is_champion"] is False and worse["is_champion"] is False


def test_champion_ranked_on_projects_primary_metric():
    import config

    cfg = config.load_project_config(PROJECT)
    assert _backend.primary_metric(cfg) == "map_50_95"
    _backend.save_experiment(PROJECT, _backend.make_experiment_record(PROJECT, ARN, _steps(), JOB, METRICS))
    higher_recall = {**METRICS, "recall": 0.9}
    rec = _backend.make_experiment_record(PROJECT, ARN.replace("ipx68r86ftpy", "abc123def456"), _steps(), JOB,
                                          higher_recall, primary="recall")
    assert rec["is_champion"] is True


def test_target_met_applies_gate_rules_to_target_metrics():
    targets = {"map_50_95": 0.30, "recall": 0.50}
    assert _backend.target_met(METRICS, targets) is True
    assert _backend.target_met({**METRICS, "recall": 0.4}, targets) is False
    assert _backend.target_met(METRICS, {}) is False


def test_tried_configurations_and_lookup_by_execution():
    import config

    cfg = config.load_project_config(PROJECT)
    rec = _backend.make_experiment_record(PROJECT, ARN, _steps(), JOB, METRICS)
    _backend.save_experiment(PROJECT, rec)
    assert _backend.tried_configurations(cfg, "yolo") == [
        {"imgsz": 640, "batch": 32, "lr0": 0.01, "weight_decay": 0.0005, "patience": 20}
    ]
    assert _backend.is_tried(cfg, "yolo", {**rec["hyperparameters"]}) is True
    assert _backend.is_tried(cfg, "yolo", {**rec["hyperparameters"], "lr0": 0.003}) is False
    assert _backend.experiment_for_execution(PROJECT, ARN)["experiment_id"] == "exp-ipx68r86ftpy"
    assert _backend.experiment_for_execution(PROJECT, "arn:none") is None
