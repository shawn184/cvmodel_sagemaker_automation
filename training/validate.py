"""Dataset validation entrypoint for the pipeline's ProcessingStep.

Fails the pipeline early (non-zero exit) if the dataset is unusable, before any
GPU training compute is provisioned. Writes a JSON report to the processing
output directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROC = Path("/opt/ml/processing")
REPORT_DIR = PROC / "report"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def validate_split(split_dir: Path) -> dict:
    images_dir = split_dir / "images" if (split_dir / "images").exists() else split_dir
    labels_dir = split_dir / "labels" if (split_dir / "labels").exists() else split_dir
    images = [p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    labels = list(labels_dir.rglob("*.txt"))
    issues, boxes, class_ids = [], 0, set()
    for lbl in labels:
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                issues.append(f"{lbl.name}: malformed line")
                break
            try:
                class_ids.add(int(parts[0]))
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                issues.append(f"{lbl.name}: non-numeric values")
                break
            if any(not (0.0 <= c <= 1.0) for c in coords):
                issues.append(f"{lbl.name}: bbox out of [0,1]")
    if not images:
        issues.append("no images found")
    return {
        "images": len(images),
        "label_files": len(labels),
        "boxes": boxes,
        "class_ids": sorted(class_ids),
        "issues": issues,
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"splits": {}}
    all_issues = []
    for split in ("train", "validation"):
        split_dir = PROC / split
        if not split_dir.exists():
            report["splits"][split] = {"exists": False}
            all_issues.append(f"missing split: {split}")
            continue
        result = validate_split(split_dir)
        report["splits"][split] = {"exists": True, **result}
        all_issues.extend(f"{split}: {i}" for i in result["issues"])

    report["valid"] = len(all_issues) == 0
    report["issues"] = all_issues
    (REPORT_DIR / "validation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if not report["valid"]:
        print("[validate] dataset validation FAILED", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
