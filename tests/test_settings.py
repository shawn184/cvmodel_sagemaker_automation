"""Tests for the S3 output layout: <base>/<project>/<kind>/."""

import pytest

import settings


def test_s3_prefix_is_project_then_kind(monkeypatch):
    monkeypatch.setattr(settings, "S3_BASE_PREFIX", "agent_automation/")
    monkeypatch.setattr(settings, "S3_BUCKET", "bucket")
    assert settings.s3_prefix("object_detection_amg", "evaluation-results") == \
        "agent_automation/object_detection_amg/evaluation-results/"
    assert settings.s3_uri("object_detection_amg", "models") == "s3://bucket/agent_automation/object_detection_amg/models/"


def test_s3_prefix_without_base_prefix(monkeypatch):
    monkeypatch.setattr(settings, "S3_BASE_PREFIX", "")
    assert settings.s3_prefix("p", "pipeline-artifacts") == "p/pipeline-artifacts/"


def test_s3_prefix_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown S3 output kind"):
        settings.s3_prefix("p", "datasets")
