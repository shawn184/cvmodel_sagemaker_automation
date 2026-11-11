"""Tests for the `run` series loop (agent and SageMaker replaced by fakes)."""

import pytest

import agents.orchestrator
import cli
import config
from agents.tools import _backend

PROJECT = "object_detection_amg"
TARGETS = {"map_50_95": 0.30, "recall": 0.50}  # pinned so edits to the project yaml don't break the tests
ARN = "arn:aws:sagemaker:ap-southeast-1:000000000000:pipeline/ObjectDetectionAmgTrainingPipeline/execution/run{n}"


@pytest.fixture
def series(monkeypatch):
    """Fake agent starts run n; fake watch records the n-th (status, map_50_95) result."""
    real_load = config.load_project_config

    def load_with_pinned_targets(name):
        cfg = real_load(name)
        if name == PROJECT:
            cfg.target_metrics = TARGETS
        return cfg

    def make(results_per_run):
        started = []
        monkeypatch.setattr(config, "load_project_config", load_with_pinned_targets)

        class FakeAgent:
            def __call__(self, instruction):
                started.append(instruction)
                return f"Started execution {ARN.format(n=len(started))}"

        def fake_watch(project, execution_arn, poll, start_claims=None):
            status, map_50_95 = results_per_run[len(started) - 1]
            _backend.save_experiment(project, {
                "experiment_id": f"exp-run{len(started)}", "execution_arn": execution_arn, "model": "yolo",
                "status": status, "hyperparameters": {"lr0": 0.01},
                "metrics": {"map_50_95": map_50_95, "recall": 0.6},
                "is_champion": status == "registered" and _backend.get_champion(project) is None,
            })
            return 0 if status == "registered" else 1

        monkeypatch.setattr(agents.orchestrator, "build_orchestrator", lambda: FakeAgent())
        monkeypatch.setattr(cli, "_watch_and_report", fake_watch)
        return started
    return make


def test_run_stops_when_target_met(series):
    started = series([("rejected", 0.20), ("registered", 0.35), ("registered", 0.9)])
    assert cli.main(["run", PROJECT]) == 0
    assert len(started) == 2  # third run never started
    assert "run 2 of at most 5" in started[1]


def test_run_exhausts_budget_and_reports_best(series, capsys):
    started = series([("registered", 0.20), ("registered", 0.25), ("registered", 0.22)])
    assert cli.main(["run", PROJECT, "--max-runs", "3"]) == 1
    assert len(started) == 3
    assert "without meeting target_metrics" in capsys.readouterr().out


def test_run_does_nothing_when_champion_already_meets_target(series, capsys):
    _backend.save_experiment(PROJECT, {
        "experiment_id": "exp-done", "execution_arn": "arn:x/done", "model": "yolo", "status": "registered",
        "hyperparameters": {}, "metrics": {"map_50_95": 0.45, "recall": 0.57}, "is_champion": True,
    })
    started = series([("registered", 0.9)])
    assert cli.main(["run", PROJECT]) == 0
    assert started == []
    assert "already meets target_metrics" in capsys.readouterr().out


def test_custom_instruction_is_manual_first_run(series):
    _backend.save_experiment(PROJECT, {
        "experiment_id": "exp-done", "execution_arn": "arn:x/done", "model": "yolo", "status": "registered",
        "hyperparameters": {}, "metrics": {"map_50_95": 0.45, "recall": 0.57}, "is_champion": True,
    })
    started = series([("registered", 0.9)])
    assert cli.main(["run", PROJECT, "train rt_detr with defaults"]) == 0
    assert started == ["train rt_detr with defaults"]  # shortcut skipped, instruction used verbatim


def test_custom_instruction_that_starts_nothing_exits_cleanly(monkeypatch, capsys):
    class Advisor:
        def __call__(self, instruction):
            return "Recommendation: yolo with lr0 0.003. Pipeline not started, as requested."

    monkeypatch.setattr(agents.orchestrator, "build_orchestrator", lambda: Advisor())
    assert cli.main(["run", PROJECT, "recommend only, do NOT start the pipeline"]) == 0
    assert "stopping the series" not in capsys.readouterr().out


def test_series_stops_when_agent_starts_nothing(monkeypatch, capsys):
    class Rejected:
        def __call__(self, instruction):
            return '{"status": "rejected", "governance_violations": ["already trained"]}'

    monkeypatch.setattr(agents.orchestrator, "build_orchestrator", lambda: Rejected())
    assert cli.main(["run", PROJECT, "--max-runs", "2"]) == 1
    assert "did not start a pipeline" in capsys.readouterr().out
