"""Global, environment-level settings for the CV Agentic MLOps POC.

These are deployment constants (region, bucket layout, Bedrock model, paths).
Per-project settings live in the YAML files under ``projects/`` and are loaded
by :mod:`config`.

Nothing here provisions AWS resources; it only records the names/paths the code
will use once resources exist.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from ``path`` into os.environ without overriding real env vars."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv(REPO_ROOT / ".env")

PROJECTS_DIR = REPO_ROOT / "projects"
CATALOG_DIR = REPO_ROOT / "catalog"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
LOGS_DIR = REPO_ROOT / "logs"  # terminal transcripts of `run` / `watch`
PROMPTS_DIR = REPO_ROOT / "agents" / "prompts"
SAMPLE_DATA_DIR = REPO_ROOT / "sample_data"

# ---------------------------------------------------------------------------
# AWS environment
# ---------------------------------------------------------------------------
AWS_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "ap-southeast-1"

# Canonical S3 bucket for the framework. Kept in one place so a single project
# config change can point at a different bucket.
# NOTE: not created by this code — see infrastructure/README.md.
S3_BUCKET = os.environ.get("CV_MLOPS_BUCKET", "your-s3-bucket")

# Base prefix inside the bucket; every output lands below it. Empty = bucket root.
S3_BASE_PREFIX = os.environ.get("CV_MLOPS_S3_PREFIX", "agent_automation/").strip("/")
S3_BASE_PREFIX = f"{S3_BASE_PREFIX}/" if S3_BASE_PREFIX else ""

# Output kinds written under each project: <base>/<project>/<kind>/
S3_KINDS = ("models", "evaluation-results", "pipeline-artifacts")


def s3_prefix(project_name: str, kind: str) -> str:
    """Key prefix (no bucket, trailing slash) for one kind of output of a project."""
    if kind not in S3_KINDS:
        raise ValueError(f"unknown S3 output kind '{kind}'; expected one of {S3_KINDS}")
    return f"{S3_BASE_PREFIX}{project_name}/{kind}/"


def s3_uri(project_name: str, kind: str) -> str:
    """Full s3:// URI for one kind of output of a project."""
    return f"s3://{S3_BUCKET}/{s3_prefix(project_name, kind)}"

# ---------------------------------------------------------------------------
# AWS prebuilt PyTorch container for training/evaluation jobs
# ---------------------------------------------------------------------------
# No custom image or ECR repo: SageMaker pulls AWS's PyTorch image and installs
# training/requirements.txt (Ultralytics) at job start.
PYTORCH_FRAMEWORK_VERSION = os.environ.get("CV_MLOPS_PYTORCH_VERSION", "2.8")
PYTORCH_PY_VERSION = os.environ.get("CV_MLOPS_PY_VERSION", "py312")
TRAINING_SOURCE_DIR = REPO_ROOT / "training"

# ---------------------------------------------------------------------------
# Bedrock model used by the Strands orchestrator agent
# ---------------------------------------------------------------------------
# Any Anthropic Claude model enabled in the target account's Bedrock region.
# Use an inference profile id (regional "apac." or "global." prefix) — most
# newer Claude models are not available via bare on-demand ids.
BEDROCK_MODEL_ID = os.environ.get(
    "CV_MLOPS_BEDROCK_MODEL", "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
)

# ---------------------------------------------------------------------------
# SageMaker execution role.
# ---------------------------------------------------------------------------
# The role SageMaker jobs/pipelines assume. Must be created by an admin — see
# infrastructure/iam/sagemaker_execution_role_policy.json. Left as an env var so
# nothing hard-fails at import time.
SAGEMAKER_EXECUTION_ROLE_ARN = os.environ.get("CV_MLOPS_SAGEMAKER_ROLE", "")
