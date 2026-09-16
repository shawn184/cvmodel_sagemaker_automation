"""Command-line entry point for the CV agentic MLOps framework.

This is the permanent way to drive the orchestrator agent against a project.
Installed as the ``cv-mlops`` console script (see pyproject.toml).

The orchestrator (backed by Amazon Bedrock) reads configuration, validates the
dataset, reviews approved models and prior experiments, recommends an
experiment, and starts the SageMaker pipeline and
reports status/metrics.

The SageMaker-executing tools degrade gracefully: without an execution role /
SageMaker permissions they return a clear "blocked" status, so the full
orchestration and reasoning flow is usable immediately. The Bedrock reasoning
and S3/config inspection run for real.

Examples:
    cv-mlops run                                  # default project, up to 5 runs
    cv-mlops run object_detection_reference --max-runs 1
    cv-mlops run object_detection_reference "recommend the next experiment"
    cv-mlops watch object_detection_reference EXECUTION_ARN   # re-attach to a running execution
    cv-mlops list

``run`` and ``watch`` mirror everything printed (including the agent's streamed
answers) into ``logs/<project>_<timestamp>.txt`` while it happens.

``run`` starts a pipeline execution, polls it every 5 minutes, has the agent
report the metrics, quality gate and champion comparison, checks that report
against the recorded facts (governance/report_check.py), records the run in
experiments/experiments.json, then starts the next experiment. It stops as soon
as a registered model meets the project's target_metrics, after --max-runs
executions (default 5), or when the agent cannot start a pipeline. The best
registered model is kept as champion. A custom instruction applies to the
first run only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import config
import settings

DEFAULT_INSTRUCTION = (
    "Coordinate a training experiment for project '{project}'. "
    "First read the project configuration and validate the dataset. Then review "
    "the approved models and previous experiments, and recommend which approved "
    "model to train next with hyperparameters inside the allowed ranges, giving a "
    "brief justification. Then attempt to start the training pipeline and report "
    "exactly what happened (including any blocked/rejected status verbatim)."
)

REPORT_INSTRUCTION = (
    "The SageMaker pipeline execution for project '{project}' has finished. "
    "Final status from get_pipeline_status: {status_json}\n\n"
    "Do NOT start a new pipeline and do NOT call register_candidate_model (the "
    "pipeline's RegisterModel step already handles registration). "
    "If the execution succeeded or failed at the QualityGate/ModelBelowThreshold step: "
    "read the metrics with get_training_metrics, run evaluate_model_quality_gate, "
    "compare against the current champion with compare_model_versions, and report the "
    "numbers, the gate result and the champion-vs-challenger comparison. "
    "If it failed at an earlier step, explain which step failed and why, using the "
    "step failure reasons above."
)

SERIES_INSTRUCTION = DEFAULT_INSTRUCTION + (
    " This is run {run} of at most {max_runs} in an automated series whose goal is to "
    "reach the project's target_metrics (primary metric: {primary}); the {max_runs}-run budget "
    "is for the whole series across all models, not per model. Never repeat a "
    "configuration listed in tried_configurations; pick the approved model and, from its "
    "allowed_next_configurations (exactly one hyperparameter changed from that model's best run "
    "so far), the change most likely to improve {primary} given the previous experiments. "
    "Start the pipeline and stop; do not wait for it."
)

START_CLAIMS_INSTRUCTION = (
    "Restate, as structured data, what start_training_pipeline returned: the model id, the "
    "hyperparameters it reported as applied and the execution ARN. Use only values from the tool result."
)

REPORT_CLAIMS_INSTRUCTION = (
    "Restate the report you just gave as structured data: whether the quality gate passed, whether the "
    "model was registered, and the metric values you reported. Use only values from the tool results."
)

TERMINAL_STATUSES = {"Succeeded", "Failed", "Stopped"}
DEFAULT_POLL_SECONDS = 300
DEFAULT_MAX_RUNS = 5
ARN_RE = re.compile(r"arn:aws:sagemaker:[a-z0-9-]+:\d+:pipeline/[^/\s]+/execution/[A-Za-z0-9]+")


class _Tee:
    """Stream wrapper that writes to the terminal and to a log file as output happens."""

    def __init__(self, stream, log):
        self._stream, self._log = stream, log

    def write(self, text: str) -> int:
        self._stream.write(text)
        self._log.write(text)
        self._log.flush()  # keep the file current while the run is in progress
        return len(text)

    def flush(self) -> None:
        self._stream.flush()
        self._log.flush()

    def __getattr__(self, name):  # isatty, encoding, fileno, ... come from the terminal stream
        return getattr(self._stream, name)


@contextmanager
def _log_to_file(path: Path, command: str):
    """Mirror stdout and stderr (the agent streams to stdout) into ``path`` for the duration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log:
        log.write(f"# {datetime.now():%Y-%m-%d %H:%M:%S}  cv-mlops {command}\n")
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(stdout, log), _Tee(stderr, log)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = stdout, stderr


def _default_log_path(project: str) -> Path:
    return settings.LOGS_DIR / f"{project}_{datetime.now():%Y%m%d-%H%M%S}.txt"


def _extract_execution_arn(text: str) -> str | None:
    m = ARN_RE.search(text)
    return m.group(0) if m else None


def _wait_for_execution(execution_arn: str, poll_seconds: int, status_fn=None) -> dict:
    """Poll get_pipeline_status until the execution reaches a terminal status."""
    if status_fn is None:
        from agents.tools.sagemaker_tools import get_pipeline_status
        status_fn = get_pipeline_status

    while True:
        status = json.loads(status_fn(execution_arn))
        if status.get("status") == "blocked":
            raise RuntimeError(f"Cannot read pipeline status: {status.get('reason')}")
        steps = ", ".join(f"{s['name']}={s['status']}" for s in status.get("steps", []))
        print(f"[{datetime.now():%H:%M:%S}] {status.get('status')}  {steps}", flush=True)
        if status.get("status") in TERMINAL_STATUSES:
            return status
        time.sleep(poll_seconds)


def _claims(agent, model, instruction: str) -> dict | None:
    """Ask the agent to restate its last answer as structured data; None if that fails."""
    try:
        return agent.structured_output(model, instruction).model_dump()
    except Exception as e:  # noqa: BLE001 - a missing check must not abort the series
        print(f"\nCould not extract the agent's claims for checking: {type(e).__name__}: {e}")
        return None


def _watch_and_report(project: str, execution_arn: str, poll_seconds: int, start_claims: dict | None = None) -> int:
    print(f"\n=== Watching {execution_arn} (every {poll_seconds // 60} min) ===\n")
    try:
        status = _wait_for_execution(execution_arn, poll_seconds)
    except KeyboardInterrupt:
        print("\nStopped watching; the pipeline keeps running in SageMaker.")
        return 130

    from agents.orchestrator import build_orchestrator
    from governance.report_check import ReportClaims

    print(f"\n=== Pipeline {status['status']}; asking the agent to report ===\n")
    agent = build_orchestrator()
    agent(REPORT_INSTRUCTION.format(project=project, status_json=json.dumps(status)))
    report_claims = _claims(agent, ReportClaims, REPORT_CLAIMS_INSTRUCTION)
    # After the report so the champion comparison is unaffected.
    _record_run(project, execution_arn, status.get("steps", []), report_claims, start_claims)
    return 0 if status["status"] == "Succeeded" else 1


def _record_run(project: str, execution_arn: str, steps: list[dict], report_claims: dict | None = None,
                start_claims: dict | None = None) -> None:
    """Check the agent's claims against the facts, then append the run to experiments.json (no LLM)."""
    from agents.tools import _backend
    from governance.report_check import audit_run

    try:
        record = _backend.collect_run_record(project, execution_arn)
    except Exception as e:  # noqa: BLE001 - bookkeeping must never hide the report
        print(f"\nCould not record the run in experiments/experiments.json: {type(e).__name__}: {e}")
        return
    if record is None:
        print("\nRun failed before the quality gate; nothing recorded in experiments/experiments.json.")
        return

    quality_gate = config.load_project_config(project).quality_gate
    issues = audit_run(record, _backend.load_experiments(project), steps, quality_gate, report_claims, start_claims)
    print("\n=== Report check ===")
    if issues:
        for issue in issues:
            print(f"WARNING: {issue}")
    else:
        checked = "report, start claims and run" if start_claims else "report and run"
        print(f"OK: the agent's {checked} agree with SageMaker, metrics.json and the experiment history.")

    if _backend.save_experiment(project, record):
        print(f"\nRecorded {record['experiment_id']} ({record['status']}) in experiments/experiments.json.")
    else:
        print(f"\n{record['experiment_id']} was already recorded in experiments/experiments.json.")


def cmd_list(_: argparse.Namespace) -> int:
    projects = config.list_projects()
    if not projects:
        print("No projects configured under projects/.")
        return 1
    print("Configured projects:")
    for p in projects:
        print(f"  - {p}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run experiments back to back until target_metrics are met or the run budget is spent."""
    if args.project not in config.list_projects():
        print(f"Unknown project '{args.project}'. Available: {config.list_projects()}")
        return 1
    cfg = config.load_project_config(args.project)
    from agents.tools import _backend

    primary = _backend.primary_metric(cfg)
    champion = _backend.get_champion(args.project)
    # A custom instruction is an explicit manual request, so it skips this shortcut.
    if not args.instruction and champion and _backend.target_met(champion.get("metrics", {}), cfg.target_metrics):
        print(f"Champion {champion['experiment_id']} already meets target_metrics {cfg.target_metrics}; "
              f"nothing to run. Raise target_metrics in projects/{args.project}.yaml or pass an instruction.")
        return 0

    # Imported lazily so `cv-mlops list` works without Bedrock/network.
    from agents.orchestrator import build_orchestrator
    from governance.report_check import StartClaims

    for run in range(1, args.max_runs + 1):
        print(f"\n=== CV Agentic MLOps — project: {args.project} — run {run}/{args.max_runs} ===\n")
        manual = run == 1 and bool(args.instruction)
        instruction = args.instruction if manual else SERIES_INSTRUCTION.format(
            project=args.project, run=run, max_runs=args.max_runs, primary=primary)
        agent = build_orchestrator()
        result = agent(instruction)  # Strands streams the agent's response to stdout
        execution_arn = _extract_execution_arn(str(result))
        if not execution_arn:
            if manual:  # e.g. "recommend but do NOT start": the agent's answer is the result
                return 0
            print("\nThe agent did not start a pipeline (see its answer above); stopping the series.")
            break
        start_claims = _claims(agent, StartClaims, START_CLAIMS_INSTRUCTION)
        if _watch_and_report(args.project, execution_arn, DEFAULT_POLL_SECONDS, start_claims) == 130:
            return 130
        record = _backend.experiment_for_execution(args.project, execution_arn)
        if record and record["status"] == "registered" and _backend.target_met(record["metrics"], cfg.target_metrics):
            print(f"\nTarget metrics {cfg.target_metrics} met by {record['experiment_id']} after {run} run(s); stopping.")
            return 0

    champion = _backend.get_champion(args.project)
    best = (f"best registered model: {champion['experiment_id']} ({primary}={champion['metrics'].get(primary)})"
            if champion else "no model was registered")
    print(f"\nSeries finished without meeting target_metrics {cfg.target_metrics}; {best}.")
    return 1


def cmd_watch(args: argparse.Namespace) -> int:
    if args.project not in config.list_projects():
        print(f"Unknown project '{args.project}'. Available: {config.list_projects()}")
        return 1
    if not ARN_RE.fullmatch(args.execution_arn):
        print(f"Not a pipeline execution ARN: {args.execution_arn}")
        return 1
    return _watch_and_report(args.project, args.execution_arn, args.interval * 60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cv-mlops", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List configured projects.")
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="Run experiments for a project until target_metrics are met.")
    p_run.add_argument("project", nargs="?", default="object_detection_reference",
                       help="Project name (default: object_detection_reference).")
    p_run.add_argument("instruction", nargs="?", default=None,
                       help="Optional custom instruction for the agent (first run only).")
    p_run.add_argument("--max-runs", type=int, default=DEFAULT_MAX_RUNS,
                       help=f"Maximum pipeline executions in total, across all models (default: {DEFAULT_MAX_RUNS}).")
    p_run.set_defaults(func=cmd_run)

    p_watch = sub.add_parser("watch", help="Wait for a running pipeline execution, then have the agent report.")
    p_watch.add_argument("project", help="Project name the execution belongs to.")
    p_watch.add_argument("execution_arn", help="Pipeline execution ARN printed by `run`.")
    p_watch.add_argument("--interval", type=int, default=DEFAULT_POLL_SECONDS // 60,
                         help="Minutes between status checks (default: 5).")
    p_watch.set_defaults(func=cmd_watch)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        return args.func(args)
    log_path = _default_log_path(args.project)
    with _log_to_file(log_path, " ".join(sys.argv[1:] if argv is None else argv)):
        print(f"Saving this terminal output to {log_path}")
        return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
