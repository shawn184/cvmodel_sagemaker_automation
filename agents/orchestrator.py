"""The Strands orchestrator agent.

A single orchestrator agent equipped with several narrowly-scoped tools rather than 
a swarm of specialised agents because CV model development follows a predictable
sequence. Specialised agents can be introduced later where their reasoning adds genuine value.

The agent reasons and coordinates; SageMaker executes. The agent's only means of
affecting infrastructure are the controlled tools passed in below.
"""

from __future__ import annotations

from strands import Agent
from strands.models import BedrockModel

import settings
from agents.tools import dataset_tools, model_tools, sagemaker_tools

# The controlled tool surface the agent is allowed to use.
ORCHESTRATOR_TOOLS = [
    dataset_tools.get_project_configuration,
    dataset_tools.validate_dataset,
    model_tools.get_available_models,
    model_tools.get_previous_experiments,
    model_tools.compare_model_versions,
    sagemaker_tools.start_training_pipeline,
    sagemaker_tools.get_pipeline_status,
    sagemaker_tools.get_training_metrics,
    sagemaker_tools.evaluate_model_quality_gate,
    sagemaker_tools.register_candidate_model,
]


def _system_prompt() -> str:
    path = settings.PROMPTS_DIR / "orchestrator_system_prompt.md"
    return path.read_text(encoding="utf-8")


def build_orchestrator(
    model_id: str | None = None,
    region_name: str | None = None,
) -> Agent:
    """Construct the orchestrator agent backed by Amazon Bedrock.

    Args:
        model_id: Bedrock model id. Defaults to ``settings.BEDROCK_MODEL_ID``.
        region_name: AWS region. Defaults to ``settings.AWS_REGION``.
    """
    model = BedrockModel(
        model_id=model_id or settings.BEDROCK_MODEL_ID,
        region_name=region_name or settings.AWS_REGION,
    )
    return Agent(
        model=model,
        tools=ORCHESTRATOR_TOOLS,
        system_prompt=_system_prompt(),
    )
