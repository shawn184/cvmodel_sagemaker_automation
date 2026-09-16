"""Dataset-facing controlled tools.

Exposes:
  * get_project_configuration(project_name)
  * validate_dataset(project_name)

These are read-only and safe: they inspect configuration and data, they never
mutate infrastructure.
"""

from __future__ import annotations

import json
from pathlib import Path

from strands import tool

import config
from agents.tools import _backend

# Image extensions we treat as samples.
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@tool
def get_project_configuration(project_name: str) -> str:
    """Return the approved configuration for a computer-vision project.

    Args:
        project_name: Short project name, e.g. "object_detection_reference".

    Returns:
        JSON describing task type, dataset locations, target metrics, the
        approved model shortlist, quality-gate thresholds and the bounded
        hyperparameter search space.
    """
    try:
        cfg = config.load_project_config(project_name)
    except config.ConfigError as e:
        return json.dumps({"error": str(e), "available_projects": config.list_projects()})
    return json.dumps(cfg.as_dict(), indent=2)


def _validate_local_yolo(root: Path) -> dict:
    """Validate a local YOLO-format dataset directory."""
    report: dict = {"location": str(root), "source": "local", "splits": {}, "issues": []}
    for split in ("train", "validation", "test"):
        # accept train/ or images/train etc. Keep it simple: <root>/<split>.
        split_dir = root / split
        images_dir = split_dir / "images" if (split_dir / "images").exists() else split_dir
        labels_dir = split_dir / "labels" if (split_dir / "labels").exists() else split_dir
        if not split_dir.exists():
            report["issues"].append(f"missing split: {split}")
            report["splits"][split] = {"exists": False}
            continue

        images = [p for p in images_dir.rglob("*") if p.suffix.lower() in _IMG_EXTS]
        labels = list(labels_dir.rglob("*.txt"))
        corrupted, boxes, class_ids = [], 0, set()
        for lbl in labels:
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 5:
                    corrupted.append(lbl.name)
                    break
                try:
                    cid = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    corrupted.append(lbl.name)
                    break
                if any(not (0.0 <= c <= 1.0) for c in coords):
                    report["issues"].append(f"{lbl.name}: bbox coords out of [0,1]")
                class_ids.add(cid)
                boxes += 1
        report["splits"][split] = {
            "exists": True,
            "images": len(images),
            "label_files": len(labels),
            "boxes": boxes,
            "class_ids": sorted(class_ids),
            "corrupted_label_files": sorted(set(corrupted)),
        }
        if not images:
            report["issues"].append(f"{split}: no images found")
    return report


@tool
def validate_dataset(project_name: str) -> str:
    """Validate the dataset for a project before any training is triggered.

    Performs basic checks: the dataset exists, annotations are readable,
    bounding boxes are valid, splits exist and basic class statistics are
    produced.

    Args:
        project_name: Short project name.

    Returns:
        JSON validation report including a top-level ``valid`` boolean.
    """
    try:
        cfg = config.load_project_config(project_name)
    except config.ConfigError as e:
        return json.dumps({"valid": False, "error": str(e)})

    report: dict = {"project": project_name, "expected_classes": cfg.dataset.classes}

    # Local override lets the POC run validation without S3 (env: CV_MLOPS_LOCAL_DATA).
    import os

    local = os.environ.get("CV_MLOPS_LOCAL_DATA")
    if local and Path(local).exists():
        report.update(_validate_local_yolo(Path(local)))
    else:
        # Inspect the configured S3 prefixes (best-effort).
        report["source"] = "s3"
        report["splits"] = {}
        for split, uri in (
            ("train", cfg.dataset.train),
            ("validation", cfg.dataset.validation),
            ("test", cfg.dataset.test),
        ):
            exists, count = _backend.s3_prefix_exists(uri)
            report["splits"][split] = {
                "uri": uri,
                "exists": exists,
                "object_count": count if count >= 0 else "unknown (S3 unreachable/denied)",
            }
        report["issues"] = [
            f"{s}: prefix not found or empty ({d['uri']})"
            for s, d in report["splits"].items()
            if not d["exists"]
        ]

    report["valid"] = len(report.get("issues", [])) == 0
    return json.dumps(report, indent=2)
