"""Deterministic quality gate.

CRITICAL GOVERNANCE PRINCIPLE
-----------------------------
The LLM must NOT decide whether a model is technically approved. That decision
is made here, in ordinary program logic, from the thresholds declared in the
project config. The same logic is mirrored as a SageMaker Pipeline ``Condition``
step (see ``pipeline/registry.py``) so the gate is enforced identically whether
it runs locally or inside the managed pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GateResult:
    passed: bool
    checks: list[dict] = field(default_factory=list)

    @property
    def summary(self) -> str:
        head = "PASS" if self.passed else "FAIL"
        lines = [f"Quality gate: {head}"]
        for c in self.checks:
            mark = "✓" if c["ok"] else "✗"
            lines.append(f"  {mark} {c['metric']} = {c['actual']} (require {c['op']} {c['threshold']})")
        return "\n".join(lines)


MAX_PREFIX = "max_"


def gate_rules(thresholds: dict[str, float]) -> list[tuple[str, str, float]]:
    """Turn threshold keys into ``(metric, op, threshold)`` rules.

    ``<metric>: v`` means the metric must be >= v; ``max_<metric>: v`` means it
    must be <= v (e.g. ``max_latency_ms`` -> ``latency_ms <= v``). No fixed list
    of metric names, so any task's metrics.json keys can be gated.
    """
    rules = []
    for key, threshold in thresholds.items():
        if key.startswith(MAX_PREFIX):
            rules.append((key[len(MAX_PREFIX):], "<=", threshold))
        else:
            rules.append((key, ">=", threshold))
    return rules


def cost_metrics(*threshold_sets: dict[str, float]) -> set[str]:
    """Metrics declared lower-is-better (``max_`` keys) in any of the given threshold dicts."""
    return {k[len(MAX_PREFIX):] for t in threshold_sets for k in t if k.startswith(MAX_PREFIX)}


def evaluate_quality_gate(metrics: dict[str, float], thresholds: dict[str, float]) -> GateResult:
    """Evaluate candidate ``metrics`` against project ``thresholds``.

    Threshold keys follow :func:`gate_rules`. A model PASSES only if every
    declared threshold is satisfied (logical AND), e.g.
    ``map_50_95 >= 0.75 AND recall >= 0.85 AND latency_ms <= 50``. A threshold
    on a metric missing from ``metrics`` fails closed.
    """
    checks: list[dict] = []

    for metric_key, op, threshold in gate_rules(thresholds):
        if metric_key not in metrics:
            checks.append(
                {"metric": metric_key, "actual": None, "op": op,
                 "threshold": threshold, "ok": False}
            )
            continue
        actual = metrics[metric_key]
        ok = actual >= threshold if op == ">=" else actual <= threshold
        checks.append(
            {"metric": metric_key, "actual": actual, "op": op,
             "threshold": threshold, "ok": ok}
        )

    passed = bool(checks) and all(c["ok"] for c in checks)
    return GateResult(passed=passed, checks=checks)
