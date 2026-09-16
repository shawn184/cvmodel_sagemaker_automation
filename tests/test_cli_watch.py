"""Tests for the CLI's wait-and-report logic (no AWS, no Bedrock)."""

import json

import pytest

import cli
import settings

ARN = "arn:aws:sagemaker:ap-southeast-1:123456789012:pipeline/DemoTrainingPipeline/execution/4h6faf2xzam5"


def test_extract_execution_arn_from_agent_text():
    text = f"Started the pipeline.\n\nExecution ARN: `{ARN}`\nFollow it in the console."
    assert cli._extract_execution_arn(text) == ARN


def test_extract_execution_arn_absent_when_blocked():
    assert cli._extract_execution_arn('{"status": "blocked", "reason": "no role"}') is None


def test_wait_for_execution_polls_until_terminal(monkeypatch):
    responses = iter([
        {"status": "Executing", "steps": [{"name": "TrainModel", "status": "Executing"}]},
        {"status": "Executing", "steps": [{"name": "EvaluateModel", "status": "Executing"}]},
        {"status": "Succeeded", "steps": [{"name": "RegisterModel", "status": "Succeeded"}]},
    ])
    sleeps: list[int] = []
    monkeypatch.setattr(cli.time, "sleep", sleeps.append)

    final = cli._wait_for_execution(ARN, 300, status_fn=lambda arn: json.dumps(next(responses)))

    assert final["status"] == "Succeeded"
    assert sleeps == [300, 300]


def test_wait_for_execution_raises_when_status_blocked(monkeypatch):
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    blocked = json.dumps({"status": "blocked", "reason": "AccessDenied"})
    with pytest.raises(RuntimeError, match="AccessDenied"):
        cli._wait_for_execution(ARN, 300, status_fn=lambda arn: blocked)


def test_terminal_output_is_saved_to_log_file(capsys):
    cli.main(["watch", "object_detection_reference", "not-an-arn"])
    out = capsys.readouterr().out
    log = next(settings.LOGS_DIR.glob("object_detection_reference_*.txt"))  # conftest points LOGS_DIR at a temp dir
    saved = log.read_text()
    assert "Not a pipeline execution ARN" in out and "Not a pipeline execution ARN" in saved
    assert saved.startswith("#") and "watch object_detection_reference" in saved.splitlines()[0]


def test_watch_rejects_non_arn(capsys):
    assert cli.main(["watch", "object_detection_reference", "not-an-arn"]) == 1
    assert "Not a pipeline execution ARN" in capsys.readouterr().out
