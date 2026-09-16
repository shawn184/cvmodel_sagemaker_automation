"""Model / experiment reasoning tools.

Exposes:
  * get_available_models(project_name)
  * get_previous_experiments(project_name)
  * compare_model_versions(project_name, candidate_metrics)

All read-only. The agent uses these to make a *governed* recommendation; the
final PASS/FAIL is still decided deterministically (governance/quality_gate.py).
"""

from __future__ import annotations

import json

from strands import tool

import config
from agents.tools import _backend


@tool
def get_available_models(project_name: str) -> str:
    """List the approved models this project is allowed to train.

    The intersection of the global approved catalogue and the project's own
    ``approved_models`` list. The agent must choose only from this list.

    Args:
        project_name: Short project name.

    Returns:
        JSON list of approved models with their search space, default
        hyperparameters, whether the next run must use the defaults, the best
        run so far, the configurations already trained and the configurations
        the next run may use (one change from the best run, not yet trained).
    """
    try:
        cfg = config.load_project_config(project_name)
    except config.ConfigError as e:
        return json.dumps({"error": str(e)})

    catalog = _backend.catalog_by_id()
    allowed = cfg.approved_models or list(catalog)
    models = []
    for m in allowed:
        if m not in catalog:
            continue
        space = _backend.search_space_for(cfg, m)
        best = _backend.best_experiment_for(cfg, m)
        allowed_next = _backend.untried_single_changes(cfg, m)
        models.append({
            **catalog[m],
            "hyperparameter_space": space,
            "default_hyperparameters": _backend.default_hyperparameters_for(cfg, m),
            "must_use_defaults": best is None,
            "best_experiment_id": best["experiment_id"] if best else None,
            "baseline_hyperparameters": {k: v for k, v in _backend.baseline_hyperparameters_for(cfg, m).items()
                                         if k in space},
            "tried_configurations": _backend.tried_configurations(cfg, m),  # start_training_pipeline rejects these
            "allowed_next_configurations": allowed_next,  # exactly one value changed from the baseline, untried
        })
    unknown = [m for m in allowed if m not in catalog]
    return json.dumps(
        {"project": project_name, "approved_models": models, "unknown_ids": unknown},
        indent=2,
    )


@tool
def get_previous_experiments(project_name: str) -> str:
    """Return historical training experiments for a project.

    Args:
        project_name: Short project name.

    Returns:
        JSON list of prior experiments (model, hyperparameters, metrics, status,
        which one is the current champion).
    """
    exps = _backend.load_experiments(project_name)
    champion = _backend.get_champion(project_name)
    return json.dumps(
        {
            "project": project_name,
            "experiment_count": len(exps),
            "champion_experiment_id": champion["experiment_id"] if champion else None,
            "experiments": exps,
        },
        indent=2,
    )


@tool
def compare_model_versions(project_name: str, candidate_metrics_json: str) -> str:
    """Compare a candidate model's metrics against the current champion.

    Produces the champion-vs-challenger comparison. This returns the raw numeric
    deltas only — the human-readable narrative is the agent's job, and the
    PASS/FAIL decision belongs to the quality gate.

    Args:
        project_name: Short project name.
        candidate_metrics_json: JSON object of candidate metrics, e.g.
            '{"map_50_95": 0.77, "recall": 0.88, "precision": 0.90, "latency_ms": 34}'.

    Returns:
        JSON comparison of champion vs candidate with per-metric deltas.
    """
    try:
        candidate = json.loads(candidate_metrics_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"candidate_metrics_json is not valid JSON: {e}"})

    champion = _backend.get_champion(project_name)
    if not champion:
        return json.dumps(
            {"project": project_name, "champion": None, "candidate": candidate,
             "note": "No existing champion — any passing candidate becomes the first champion."}
        )

    champ_metrics = champion.get("metrics", {})
    deltas = {}
    for k, cand_val in candidate.items():
        if k in champ_metrics:
            deltas[k] = round(cand_val - champ_metrics[k], 4)
    return json.dumps(
        {
            "project": project_name,
            "champion_experiment_id": champion["experiment_id"],
            "champion_metrics": champ_metrics,
            "candidate_metrics": candidate,
            "deltas_candidate_minus_champion": deltas,
        },
        indent=2,
    )
