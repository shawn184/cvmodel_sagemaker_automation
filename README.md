# CV Agentic MLOps

A **reusable, governed framework** that uses a [Strands](https://strandsagents.com)
orchestrator agent (on Amazon Bedrock) to coordinate the computer-vision model
training and evaluation lifecycle on Amazon SageMaker.

The guiding principle:

> **Strands decides and coordinates what needs to happen. SageMaker executes the
> ML workloads in a controlled, reproducible manner.**

The agent reasons over project requirements, an approved model catalogue, and
prior experiments, then invokes **narrowly-scoped controlled tools** to start a
SageMaker pipeline, read metrics, compare models, and register candidates. It
**cannot** create arbitrary infrastructure — it receives capabilities, not
unrestricted AWS access. The technical PASS/FAIL decision is made by a
**deterministic quality gate** in plain code, never by the LLM.

Onboarding a new CV project is a **configuration change** (a YAML file in
`projects/`), not a code change.

```
Dataset → Validation → Agent orchestration → Model/experiment selection →
SageMaker Pipeline → Training/HPO → Evaluation → Champion/Challenger comparison →
Quality gate → Model Registry
```

## Contents

- [Project structure](#project-structure)
  - [How the scripts interact during a run](#how-the-scripts-interact-during-a-run)
- [Quick start](#quick-start)
- [Design decisions](#design-decisions)
  - [Where the Strands agent is involved](#where-the-strands-agent-is-involved)
  - [Why the agent runs on your machine, not in AWS Lambda](#why-the-agent-runs-on-your-machine-not-in-aws-lambda)
  - [Why jobs use AWS's prebuilt PyTorch image, not a custom image in ECR](#why-jobs-use-awss-prebuilt-pytorch-image-not-a-custom-image-in-ecr)
- [Running on SageMaker](#running-on-sagemaker)
  - [Prerequisites](#prerequisites)
  - [Run end to end](#run-end-to-end)
  - [What each stage produces](#what-each-stage-produces)
  - [Testing part of the process](#testing-part-of-the-process)
  - [Checking that SageMaker is no longer billing](#checking-that-sagemaker-is-no-longer-billing)
- [Tests](#tests)
- [Adding a new project](#adding-a-new-project)
- [Configuration (environment variables)](#configuration-environment-variables)

## Project structure

```
cv-agentic-mlops/
├── cli.py                      # `cv-mlops` entry point (run the orchestrator)
├── config.py                   # load + validate project YAML into ProjectConfig
├── settings.py                 # environment constants (region, bucket, model)
├── pyproject.toml              # app dependencies (uv), console script
│
├── agents/                     # ── Intelligence & orchestration ──
│   ├── orchestrator.py         # builds the Strands agent + tool surface
│   ├── prompts/
│   │   └── orchestrator_system_prompt.md
│   └── tools/                  # controlled tools = the agent's only capabilities
│       ├── dataset_tools.py    # get_project_configuration, validate_dataset
│       ├── model_tools.py      # get_available_models, previous_experiments, compare
│       ├── sagemaker_tools.py  # start_pipeline, status, metrics, gate, register
│       └── _backend.py         # shared helpers (catalog, experiments, S3)
│
├── pipeline/                   # ── ML process control (SageMaker DAG) ──
│   ├── pipeline.py             # Validate→Train→Evaluate→QualityGate→Register|Fail
│   ├── processing.py           # dataset validation step
│   ├── training.py             # estimator / training step
│   ├── evaluation.py           # evaluation step + metrics property file
│   └── registry.py             # quality-gate condition + model registration
│
├── training/                   # ── ML computation (runs in AWS's PyTorch image) ──
│   ├── requirements.txt        # installed at job start (Ultralytics)
│   ├── train.py                # training entrypoint (Ultralytics YOLO/RT-DETR)
│   ├── evaluate.py             # produces metrics.json
│   └── validate.py             # dataset validation (fail fast)
│
├── governance/
│   ├── quality_gate.py         # deterministic PASS/FAIL (never the LLM)
│   └── report_check.py         # checks the agent's claims against SageMaker/S3 facts
│
├── catalog/
│   └── approved_models.yaml    # the only models the agent may choose from
│
├── projects/                   # one YAML per CV project (reusability)
│   ├── object_detection_reference.yaml
│   └── object_detection_amg.yaml
│
├── experiments/
│   └── experiments.json        # experiment/champion history
│
├── infrastructure/             # ── AWS resources & permissions ──
│   ├── README.md               # what to provision + permissions to request
│   └── iam/
│       ├── agent_runtime_policy.json
│       └── sagemaker_execution_role_policy.json
│
└── tests/                      # offline unit tests (no AWS, no Bedrock); see "Tests" below
```

The three layers are intentionally decoupled: **agent logic ≠ ML training logic
≠ AWS infrastructure.**

### How the scripts interact during a run

The diagram follows one object-detection series started with
`cv-mlops run object_detection_reference`. The left side is the local machine, the right
side is AWS. Numbers (1) to (5) are the stages of one run; the loop repeats
from (1) until the target metrics are met or the run budget is spent.

```
 LOCAL MACHINE                                                     AWS (ap-southeast-1)
 ───────────────────────────────────────────────────────────────  ─────────────────────────────────────────
 cli.py  cmd_run
   │  config.py    ← projects/object_detection_reference.yaml (dataset S3 paths, approved models, quality gate)
   │  settings.py  ← env: region, bucket, Bedrock model id, SageMaker role
   │  _backend.get_champion ← experiments/experiments.json   (stop at once if the champion already meets target_metrics)
   │
   ├─(1) DECIDE ── agents/orchestrator.py  build_orchestrator()
   │        Strands Agent + agents/prompts/orchestrator_system_prompt.md ─────────►  Amazon Bedrock (reasoning)
   │        the model decides which tools to call; the tools are its only capabilities:
   │        ├─ dataset_tools.get_project_configuration ── config.py
   │        ├─ dataset_tools.validate_dataset ──────────── _backend.s3_prefix_exists ──────►  S3  datasets train/ validation/ test/
   │        ├─ model_tools.get_available_models ────────── _backend ← catalog/approved_models.yaml ∩ project YAML,
   │        │                                                         experiments.json → best run, tried, allowed next configs
   │        ├─ model_tools.get_previous_experiments ────── _backend ← experiments.json
   │        └─ sagemaker_tools.start_training_pipeline
   │              ├─ _validate_choice   governance: approved model, values in the search space,
   │              │                     one change from the model's best run, not trained before → else "rejected"
   │              └─ pipeline/pipeline.py  build_pipeline() → upsert() + start() ─────────►  SageMaker Pipelines
   │                    ├─ processing.py   ValidateDataset  ← training/validate.py            (execution ARN returned
   │                    ├─ training.py     TrainModel       ← training/train.py + requirements.txt   to the agent)
   │                    ├─ evaluation.py   EvaluateModel    ← training/evaluate.py
   │                    └─ registry.py     QualityGate → RegisterModel | ModelBelowThreshold
   │
   ├─(2) WAIT ──── cli.py reads the execution ARN from the agent's answer, then every 5 min:
   │        sagemaker_tools.get_pipeline_status ────────────────────────────────────►  DescribePipelineExecution + step list
   │
   ├─(3) REPORT ── a fresh agent gets REPORT_INSTRUCTION ─────────────────────────────►  Amazon Bedrock
   │        ├─ sagemaker_tools.get_training_metrics ─────────────────────────────────►  S3  evaluation-results/latest/metrics.json
   │        ├─ sagemaker_tools.evaluate_model_quality_gate ── governance/quality_gate.py   (deterministic PASS/FAIL)
   │        └─ model_tools.compare_model_versions ─────────── _backend.get_champion ← experiments.json
   │
   ├─(4) RECORD ── cli.py _record_run  (no LLM involved)
   │        ├─ _backend.collect_run_record ──────────────────────────────────────────►  ListPipelineExecutionSteps,
   │        │                                                                             DescribeTrainingJob, metrics.json
   │        ├─ governance/report_check.audit_run   agent's claims vs the facts → "OK" or WARNING lines
   │        └─ _backend.save_experiment ───► experiments/experiments.json  (new record; champion updated if it won)
   │
   └─(5) NEXT ──── registered and target_metrics met? → stop.  Otherwise back to (1) until --max-runs.
        Everything printed above is also written to logs/<project>_<timestamp>.txt
```

Inside SageMaker, one pipeline execution runs the four `training/` scripts in
order. Every job uses AWS's prebuilt PyTorch image; the training job installs
`training/requirements.txt` (Ultralytics) when it starts.

```
 SageMaker pipeline execution (one run)
 ──────────────────────────────────────────────────────────────────────────────────────────
 ValidateDataset    ProcessingStep   training/validate.py
    in:  S3 train/ + validation/                 out: validation report
    a non-zero exit fails the pipeline here, before any GPU instance is started
        │
 TrainModel         TrainingStep     training/train.py     (pip install training/requirements.txt)
    in:  S3 train/ + validation/, hyperparameters = catalogue defaults + the agent's one change,
         plus model_family / base_weights (e.g. yolo11n.pt) from catalog/approved_models.yaml
    out: S3 models/.../model.tar.gz
        │
 EvaluateModel      ProcessingStep   training/evaluate.py
    in:  model.tar.gz + S3 test/                 out: S3 evaluation-results/latest/metrics.json
         (mAP@0.5, mAP@0.5:0.95, precision, recall, F1, latency_ms, model size)
        │
 QualityGate        ConditionStep    metrics.json vs the quality_gate thresholds in the project YAML
        ├─ pass → RegisterModel        ModelStep   → Model Registry, status PendingManualApproval
        └─ fail → ModelBelowThreshold  FailStep    → execution ends as Failed at this step
```

Two things to notice. First, the LLM only appears in stages (1) and (3): it
chooses the experiment and writes the report. Starting the pipeline, the quality
gate, registration and the experiment record are all plain code. Second, the
`training/` scripts never import anything from `agents/` or `pipeline/`. They
only read the standard SageMaker input paths, so they can be run and tested
locally on their own (see "Testing part of the process").

## Quick start

```bash
# Install the application layer with uv (sagemaker extra = build/start pipelines)
uv sync --extra sagemaker --extra dev

# List configured projects
uv run python cli.py list

# Run experiments for a project until target_metrics are met (max 5 runs; uses Amazon Bedrock)
uv run python cli.py run object_detection_reference

# A single experiment
uv run python cli.py run object_detection_reference --max-runs 1

# Run the tests (no AWS or dataset required)
PYTHONPATH=. uv run pytest -q
```

## Design decisions

### Where the Strands agent is involved

The agent is the decision-making layer in front of the pipeline. It acts only at
the start and the end, never inside the training itself.

| Phase | Agent involved? | What happens |
|---|---|---|
| Before each run | Yes | On `cli.py run <project>` the agent reasons (via Bedrock) over the project YAML, the approved model catalogue, and past experiments. It calls tools to validate the dataset, pick an allowed model and an untried hyperparameter set that respects the one-change-per-run rule, then calls the tool that starts the SageMaker pipeline. |
| During the pipeline | No | SageMaker runs Validate → Train → Evaluate → QualityGate → Register on its own. `model.tar.gz`, `metrics.json`, and profiler output are produced with no LLM in the loop. |
| After the pipeline | Yes | `cli.py` polls the execution every 5 minutes (plain code, no LLM) and, once it finishes, hands the final status to the agent. The agent reads the metrics, runs the quality gate, compares the new model against the current champion and explains the outcome in the terminal. The PASS/FAIL decision comes from the deterministic quality gate in code; the agent only reports and explains it. Plain code then checks the agent's report against the facts (see *Report check* below), appends the run to `experiments/experiments.json` and decides whether to start the next run (target not met, budget left). |

In short: Strands decides what to run and interprets the results, SageMaker does
the work, and the agent can only act through the ten narrow tools wired up in
`agents/orchestrator.py`.

### Why the agent runs on your machine, not in AWS Lambda

The agent runs inside `cli.py run` on your own computer. That command starts a
pipeline, checks its status every 5 minutes until training is done (usually 30
minutes to a few hours), asks the agent to report the result, and then starts the
next run. Lambda is not used anywhere in this loop, for four reasons:

- **Lambda stops after 15 minutes.** One series of runs takes hours. To use
  Lambda, the loop would have to be cut into many short pieces that are each
  triggered by an event, and the agent's progress would have to be saved
  between them.
- **The agent works on local files.** The project YAML, the approved model
  catalogue and `experiments/experiments.json` are files in this repo. The CLI
  reads and writes them directly. A Lambda function has no access to them, so
  they would first have to be moved into S3 or a database.
- **A person is meant to watch.** The agent prints its reasoning to the
  terminal and saves it under `logs/`, and you can give the first run your own
  instruction. Running it as a command on your machine is what makes that
  possible.
- **There is nothing to deploy.** The CLI calls Bedrock and SageMaker with your
  own AWS credentials. There is no function to package, upload, or keep up to
  date.

The SageMaker pipeline does not need Lambda either. Each of its stages is a
built-in SageMaker step: a processing job for dataset validation, a training
job, a processing job for evaluation, a condition step that reads
`metrics.json` for the quality gate, and the register-model and fail steps.

If the series ever has to run by itself without a person watching, the same
code can be run as a scheduled ECS/Fargate task or from a Step Functions
workflow. Both allow long-running processes, so nothing in the loop has to
change. Lambda would still be the wrong fit because of the 15-minute limit.

### Why jobs use AWS's prebuilt PyTorch image, not a custom image in ECR

Training, evaluation and validation jobs run AWS's `pytorch-training` image and
install `training/requirements.txt` (Ultralytics) when the job starts. There is
no Dockerfile, no image build and no ECR repository to manage, and the same
image serves CPU and GPU instances.

| | Prebuilt image + `requirements.txt` (current) | Custom image pushed to ECR |
|---|---|---|
| Setup | None; `framework_version`/`py_version` in `settings.py` | Dockerfile, `docker build`, ECR repo, push on every dependency change |
| Job start | ~1–2 min extra for `pip install` and base-weight download | Dependencies and weights baked in; faster, repeatable starts |
| Network | Jobs need outbound internet (`enable_network_isolation=False`) | Can run fully isolated inside a VPC |
| Reproducibility | Depends on the pins in `requirements.txt` at job time | The image digest fixes every dependency |
| Permissions | Pull from AWS's public account only | Execution role also needs pull rights on your ECR repo |

For a POC the first column wins: no build pipeline, and a dependency bump is a
one-line change. Move to a custom image when the jobs must run without internet
access, when a private dependency is required, or when start-up time matters
across many short runs. The switch is small: write a Dockerfile `FROM` the AWS
PyTorch image that installs `training/requirements.txt` and pre-downloads the
base weights, push it to ECR, and pass its URI as `image_uri` to the estimator
in `pipeline/training.py` and the processors in `pipeline/evaluation.py` and
`pipeline/processing.py` (which currently call `prebuilt_image_uri`). Keeping the
AWS image as the base means the SageMaker toolkit and entry-point handling
stay unchanged.

## Running on SageMaker

### Prerequisites

| Need | Why | How |
|---|---|---|
| Dataset in S3 | Inputs for the pipeline | `<split>/images/*.jpg` + `<split>/labels/*.txt` (YOLO format) per `dataset.*` URI in the project YAML. No `data.yaml` needed. |
| S3 location you can write to | Model artifacts + `metrics.json` | Default `s3://lta-sam-sandbox/agent_automation/`; override with `CV_MLOPS_BUCKET` / `CV_MLOPS_S3_PREFIX` |
| Container image | None to build. Jobs run AWS's prebuilt PyTorch image (`pytorch-training:2.8-{gpu,cpu}-py312`) and install `training/requirements.txt` (Ultralytics) at start, so no Docker or ECR is involved. Jobs need outbound internet for pip and base weights. | Optional: `CV_MLOPS_PYTORCH_VERSION` / `CV_MLOPS_PY_VERSION` |
| SageMaker execution role | Assumed by the jobs to read S3, pull the AWS image, write outputs | `export CV_MLOPS_SAGEMAKER_ROLE=arn:aws:iam::<acct>:role/<role>` (policy: `infrastructure/iam/sagemaker_execution_role_policy.json`) |
| Permissions for your CLI user | Create/start pipelines, pass the role, upload code to the bucket, invoke Bedrock | `infrastructure/iam/agent_runtime_policy.json` |
| Bedrock inference profile | The orchestrator's LLM | default in `settings.py`; override with `CV_MLOPS_BEDROCK_MODEL` |

### Run end to end

```bash
export AWS_REGION=ap-southeast-1
export CV_MLOPS_BUCKET=lta-sam-sandbox          # default
export CV_MLOPS_S3_PREFIX=agent_automation/     # default
export CV_MLOPS_SAGEMAKER_ROLE=arn:aws:iam::<acct>:role/<execution-role>

uv run python cli.py run `project_name`                    # up to 5 runs; --max-runs N to change
# or pin the first run's model / instruction:
uv run python cli.py run `project_name` "train rt_detr with its default hyperparameters"
```

Each run: the agent validates the dataset, picks an approved model and an
untried hyperparameter set (first run of a model = defaults, later runs change
exactly one value from that model's best run so far), starts the pipeline `<ModelRegistryGroup>TrainingPipeline`
and prints the execution ARN. On upsert the SDK uploads `training/` to
`$OUT/<project>/pipeline-artifacts/`. The CLI stays attached: it prints the step
statuses every 5 minutes and, when the execution reaches
Succeeded/Failed/Stopped, asks the agent to report the metrics, the quality-gate
result and the champion-vs-challenger comparison, records the run in
`experiments/experiments.json`, then starts the next run. One run trains one
model with one hyperparameter set; each run is a separate training job and bill.

The series stops as soon as a registered model meets the project's
`target_metrics`, after `--max-runs` executions in total across all models
(default 5), or when the agent
cannot start a pipeline (for example every configuration in the search space has
been tried). If the current champion already meets `target_metrics`, `run`
starts nothing unless you pass an instruction. A custom instruction applies to
the first run only; later runs are chosen by the agent.

If you close the terminal (Ctrl-C only stops the watching, not the pipeline),
re-attach later with:

```bash
uv run python cli.py watch `project_name` EXECUTION_ARN   # --interval 2 for 2-minute polling
```

### What each stage produces

`$OUT` below is `s3://$CV_MLOPS_BUCKET/$CV_MLOPS_S3_PREFIX`, by default
`s3://lta-sam-sandbox/agent_automation/`. All S3 output lands under
`$OUT/<project>/`, split into `models/`, `evaluation-results/` and
`pipeline-artifacts/`.

| Stage | Runs on | Output | Where |
|---|---|---|---|
| ValidateDataset | Processing (`ml.m5.large`) | `validation_report.json` (image/label counts, class ids, issues); non-zero exit stops the pipeline | `$OUT/<project>/pipeline-artifacts/<job>/output/validation_report/` |
| TrainModel | Training (`training.instance_type`) | `model.tar.gz` containing `model.pt` (= `best.pt`), `training_metadata.json` (resolved hyperparameters), and the Ultralytics run dir `train/` (`weights/best.pt`, `weights/last.pt`, `results.csv`, loss/PR curves, confusion matrix) | `$OUT/<project>/models/<job>/output/model.tar.gz` |
| EvaluateModel | Processing (`training.instance_type`) | `metrics.json`: `map_50`, `map_50_95`, `precision`, `recall`, `f1`, `latency_ms` (mean per image), `model_size_mb` | `$OUT/<project>/evaluation-results/latest/metrics.json` |
| QualityGate | Pipeline condition | Compares `metrics.json` with `quality_gate` in the project YAML | pipeline step status |
| RegisterModel (gate passed) | Model Registry | Model package in group `<ModelRegistryGroup>`, status `PendingManualApproval`, pointing at `model.tar.gz` + `metrics.json` | SageMaker Model Registry |
| ModelBelowThreshold (gate failed) | FailStep | Execution status `Failed` with the reason | pipeline execution |

The final deliverables are therefore the registered model package (`model.tar.gz`
with `best.pt`) and `metrics.json`. The champion-vs-challenger comparison is
produced by the agent from `compare_model_versions` against
`experiments/experiments.json` and is reported in the agent's answer; it is not
written to S3.

**Report check.** The agent's narrative is treated as a claim, not a record.
After each report the CLI asks the agent to restate it as structured data
(gate PASS/FAIL, registered or not, metric values; for `run` also the model
and hyperparameters it said it started) and `governance/report_check.py`
compares those claims with the pipeline's QualityGate outcome, the training
job's hyperparameters and `metrics.json`. It also checks the run itself:
`metrics.json` values identical to an earlier run of the same model (ignoring
lower-is-better metrics such as latency, which jitter) mean the hyperparameter
change had no effect, and a gate verdict that differs from the
project's current `quality_gate` means the yaml changed while the pipeline ran.
Mismatches are printed as `WARNING:` lines under `=== Report check ===`; the
run is recorded either way.

After the agent's report, `run`/`watch` append the run to
`experiments/experiments.json` (plain code, reading the training job and
`metrics.json`): model, hyperparameters, metrics, `model_artifact`, the model
package version and `status` `registered` (gate passed) or `rejected` (gate
failed). Runs that fail before the quality gate are not recorded. The best
registered model is kept as champion: a new registered run takes over when it
beats the current champion on the primary metric (the first key of
`target_metrics`, `map_50_95` by default) and the previous champion is demoted.
Recording is keyed by execution ARN, so re-running `watch` on a finished
execution never duplicates an entry. Once a model has a recorded run, the next
run changes exactly one hyperparameter from that model's best run so far (highest
primary metric, registered or not), and `start_training_pipeline` rejects a
configuration that was already trained. `get_available_models` lists the
resulting `allowed_next_configurations` for each model.

### Testing part of the process

Assuming the prerequisites above are in place:

| Want to test | How |
|---|---|
| Governance only (no AWS) | `PYTHONPATH=. uv run pytest -q` |
| Dataset validation only | `uv run python -c "from agents.tools.dataset_tools import validate_dataset; print(validate_dataset('object_detection_amg'))"` (checks the S3 prefixes; set `CV_MLOPS_LOCAL_DATA=<dir>` to deep-check a local copy with `train/ validation/ test/`) |
| Agent reasoning without starting compute | `uv run python cli.py run object_detection_amg "validate the dataset and recommend the next experiment, but do NOT start the pipeline"` |
| Dry run of the pipeline request | Unset `CV_MLOPS_SAGEMAKER_ROLE`; `start_training_pipeline` returns `blocked` with the fully resolved, validated request instead of calling SageMaker |
| Training script locally (installs Ultralytics + torch; GPU optional) | `uv run --with-requirements training/requirements.txt python training/train.py --local <dir>/train --hp '{"epochs": 1, "class_names": "[\"person\", ...]"}'` where `<dir>` has `train/` and `validation/` splits |
| Evaluation script locally | `uv run --with-requirements training/requirements.txt python training/evaluate.py --root <dir> --class-names '["person", ...]'` where `<dir>` has `model/model.pt` and `test/{images,labels}` |
| Pipeline definition without running it | With the env vars set: `PYTHONPATH=. uv run python -c "import config; from pipeline.pipeline import build_pipeline; print(build_pipeline(config.load_project_config('object_detection_amg'), 'yolo', {}).definition())"` (uploads `training/` to the bucket, calls no SageMaker API) |
| Quality gate on saved metrics | `uv run python -c "from agents.tools.sagemaker_tools import evaluate_model_quality_gate; print(evaluate_model_quality_gate('object_detection_amg', '{\"map_50_95\": 0.31, \"recall\": 0.55, \"latency_ms\": 40}'))"` |
| Pipeline execution status (one-off) | Replace the `EXECUTION_ARN` placeholder with the ARN returned by the training run. Keep the ARN inside single quotes; do not use shell backticks: `uv run python -c "from agents.tools.sagemaker_tools import get_pipeline_status; print(get_pipeline_status('EXECUTION_ARN'))"`. Returns the execution status plus every step's status and failure reason. |
| Wait for a run and get the agent's report | `uv run python cli.py watch object_detection_amg EXECUTION_ARN` (polls every 5 minutes; `--interval N` for N minutes; also records a finished run in `experiments.json`) |
| Exactly one experiment | `uv run python cli.py run object_detection_amg --max-runs 1` |
| Metrics of the latest run | `uv run python -c "from agents.tools.sagemaker_tools import get_training_metrics; print(get_training_metrics('object_detection_amg'))"` (first argument is the project name; the metrics file is per project, `latest/metrics.json`) |
| Record a finished run without the agent | `PYTHONPATH=. uv run python -c "from agents.tools import _backend as b; r = b.collect_run_record('object_detection_amg', 'EXECUTION_ARN'); print(b.save_experiment('object_detection_amg', r) if r else 'failed before the gate')"` |

Running a single SageMaker step in isolation is not supported: the pipeline
always runs Validate → Train → Evaluate → Gate. To shorten a real run, set a
small `epochs` value in the project's `hyperparameter_space`.

### Checking that SageMaker is no longer billing

The processing and training jobs are ephemeral instances and the framework never
creates an endpoint, so nothing keeps billing once a `run` finishes (S3 and
CloudWatch Logs storage aside). Pressing Ctrl-C during `watch` stops the CLI but
not the execution, which keeps running until it finishes or hits
`training.max_runtime_seconds`. To confirm that nothing is still running:

```bash
aws sagemaker list-training-jobs   --status-equals InProgress --region ap-southeast-1
aws sagemaker list-processing-jobs --status-equals InProgress --region ap-southeast-1
aws sagemaker list-endpoints       --region ap-southeast-1
```

All three should return empty lists. To stop a running execution instead of
waiting for it:

```bash
aws sagemaker stop-pipeline-execution --pipeline-execution-arn EXECUTION_ARN --region ap-southeast-1
```

## Tests

`PYTHONPATH=. uv run pytest -q` runs everything in about a second. The suite
never reaches AWS or Bedrock: `conftest.py` blanks the SageMaker role so the
pipeline tools stop at the `blocked` boundary, and redirects
`experiments.json` to a temporary copy so tests neither depend on nor modify
real recorded runs. The agent and SageMaker are replaced by fakes where a test
needs them.

| File | What it covers |
|---|---|
| `test_config.py` | Loading project YAML: sample projects load and are listed, missing project raises, registry group name derivation, `hyperparameter_space` lookup by model id before architecture |
| `test_settings.py` | The S3 output layout `<prefix>/<project>/<kind>/` and rejection of unknown kinds |
| `test_quality_gate.py` | The deterministic gate: `>=` for plain keys, `<=` for `max_` keys, missing metrics fail closed, empty thresholds never pass |
| `test_tools.py` | Tool governance: unapproved models and out-of-range hyperparameters are refused, first run uses defaults, one change per run from the best run so far, no repeated configuration, registration only after a passing gate |
| `test_experiments.py` | Recording a finished run from the training job and `metrics.json`, champion promotion and demotion, idempotence per execution ARN |
| `test_report_check.py` | The report check: agent claims versus SageMaker facts, identical metrics to a prior run, gate config drift, and the `WARNING`/`OK` output of `_record_run` |
| `test_cli_run.py` | The `run` series loop: stops on target met, budget exhausted, or when no pipeline can be started; custom instruction applies to the first run only |
| `test_cli_watch.py` | The `watch` helpers: extracting the execution ARN from the agent's answer, polling until a terminal status, erroring on a `blocked` status or a non-ARN argument |
| `test_training_scripts.py` | The container scripts' `data.yaml` generation from `classes` (no Ultralytics needed) |

## Adding a new project

Copy a file in `projects/`, change the `project_name`, `dataset` paths,
`classes`, `approved_models`, `target_metrics`, `quality_gate`, and
`hyperparameter_space`. No code changes required. `quality_gate` decides whether
a single run is registered; `target_metrics` is the goal `run` works towards
and its first key ranks the champion. Both are keyed by `metrics.json` names:
`recall: 0.7` means `recall >= 0.7`, and a `max_` prefix flips the comparison,
so `max_latency_ms: 100` means `latency_ms <= 100`. Any metric the evaluation
step emits can be used; a threshold on a metric that is missing fails the gate.

Each dataset split is an S3 prefix with `images/` and `labels/` (YOLO txt)
subfolders. No `data.yaml` is needed: the containers generate it from `classes`,
whose order must match the label ids.

`hyperparameter_space` is keyed by model architecture (`cnn`, `transformer`, as
declared in `catalog/approved_models.yaml`) or by a specific model id. Each
hyperparameter lists its allowed values; the first value is the default. The
first run of a model in a project must use the defaults; each later run changes
exactly one hyperparameter from the model's best run so far and may not repeat a
trained configuration. One pipeline execution trains one model with one
hyperparameter set. Shorten the lists to cap the runs.

## Configuration (environment variables)

Set these in the shell or in a `.env` file at the repo root (see `.env.example`;
`.env` is gitignored). `settings.py` loads `.env` at import; shell variables
take precedence.

| Variable | Meaning | Default |
|---|---|---|
| `AWS_REGION` | AWS region | `ap-southeast-1` |
| `CV_MLOPS_BUCKET` | S3 bucket for models/results/pipeline artifacts | `lta-sam-sandbox` |
| `CV_MLOPS_S3_PREFIX` | Prefix inside that bucket under which all output is written | `agent_automation/` |
| `CV_MLOPS_BEDROCK_MODEL` | Bedrock model id for the agent | a Claude model |
| `CV_MLOPS_SAGEMAKER_ROLE` | SageMaker execution role ARN | unset (blocks execution) |
| `CV_MLOPS_PYTORCH_VERSION` | AWS prebuilt PyTorch container version | `2.8` |
| `CV_MLOPS_PY_VERSION` | Python version of that container | `py312` |
| `CV_MLOPS_LOCAL_DATA` | Local dataset dir for offline validation | unset |
