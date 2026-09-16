# Infrastructure & Permissions

This framework separates **agent logic**, **ML training logic**, and **AWS
infrastructure**. Nothing in this repository provisions AWS resources
automatically — provisioning is a deliberate, reviewed step. This document lists
exactly what must exist and which permissions to request.

## What must exist to run the full pipeline

| Resource | Purpose | Notes |
|---|---|---|
| S3 bucket + prefix | Model artifacts, evaluation results, pipeline artifacts | Layout below; set in `settings.py` (`CV_MLOPS_BUCKET`, `CV_MLOPS_S3_PREFIX`); default `s3://lta-sam-sandbox/agent_automation/` |
| Container image | Training/evaluation runtime | Nothing to provision: AWS's prebuilt PyTorch image (`763104351884.dkr.ecr.<region>.amazonaws.com/pytorch-training`) + `training/requirements.txt` |
| SageMaker execution role | Role the training/processing jobs + pipeline assume | Policy: `iam/sagemaker_execution_role_policy.json` |
| Agent runtime identity | Permissions for the process hosting the orchestrator | Policy: `iam/agent_runtime_policy.json` |
| Bedrock model access | The orchestrator's foundation model | An Anthropic Claude model enabled in the target region |

### S3 layout

```
s3://<bucket>/<prefix>/<project>/
  models/
  evaluation-results/latest/metrics.json
  pipeline-artifacts/
```

## Permissions required (and current gaps)

The orchestrator's reasoning (Bedrock) and configuration/S3 inspection work with
standard read access. The **SageMaker execution** half requires permissions that
a restricted user typically does **not** have. Request from a cloud/security
admin:

1. Create/attach a **SageMaker execution role** using
   `iam/sagemaker_execution_role_policy.json` (trust policy allows
   `sagemaker.amazonaws.com`). Set its ARN via `CV_MLOPS_SAGEMAKER_ROLE`.
2. Grant the **agent runtime identity** the actions in
   `iam/agent_runtime_policy.json` (start/describe the approved pipeline, read
   results, register model versions, invoke Bedrock).

Until these are granted, the SageMaker-executing tools return a `blocked` status
that names the missing capability — the orchestration flow still runs
end-to-end.

## Provisioning order

1. Create the S3 bucket and prefixes; set `CV_MLOPS_BUCKET` and `CV_MLOPS_S3_PREFIX`.
2. Create the SageMaker execution role; set `CV_MLOPS_SAGEMAKER_ROLE`.
3. Attach the agent runtime policy to the hosting identity.
4. Run `uv run python cli.py run <project>` — the agent can now start real pipelines.

## Hardened environments

For higher-security deployments, enable VPC configuration and network isolation
on training/processing jobs (`enable_network_isolation=True` in
`pipeline/training.py`) and add the corresponding subnet/security-group config.
Discuss with the cloud/security team rather than enabling ad hoc.
