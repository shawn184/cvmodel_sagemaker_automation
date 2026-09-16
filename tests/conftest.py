"""Shared test fixtures.

The suite must never reach AWS: blank the execution role so pipeline tools stop
at the governance/"blocked" boundary even when a developer's .env sets it.

Experiment history is redirected to a temp copy (reference project seeded, amg
empty) so tests neither depend on nor modify real recorded runs.
"""

import json

import pytest

import settings


@pytest.fixture(autouse=True)
def _no_aws(monkeypatch):
    monkeypatch.setattr(settings, "SAGEMAKER_EXECUTION_ROLE_ARN", "")


@pytest.fixture(autouse=True)
def _isolated_experiments(tmp_path, monkeypatch):
    real = json.loads((settings.EXPERIMENTS_DIR / "experiments.json").read_text(encoding="utf-8"))
    seed = {"object_detection_reference": real["object_detection_reference"], "object_detection_amg": []}
    (tmp_path / "experiments.json").write_text(json.dumps(seed), encoding="utf-8")
    monkeypatch.setattr(settings, "EXPERIMENTS_DIR", tmp_path)
    monkeypatch.setattr(settings, "LOGS_DIR", tmp_path / "logs")  # run/watch transcripts stay out of the repo
