"""SageMaker evaluation entrypoint.

Loads the trained model, evaluates it on the held-out test set, and writes
``metrics.json`` to the processing output directory. The quality-gate condition
step (and the local quality gate) read this file.

Emits the metrics the framework tracks for object detection:
mAP@0.5, mAP@0.5:0.95, precision, recall, F1, inference latency, model size.

Run locally for testing (root has model/model.pt and test/{images,labels}):
    python evaluate.py --root <dir> --class-names '["person", "bike"]'
"""

from __future__ import annotations

import argparse
import json
import tarfile
import time
from pathlib import Path

PROC = Path("/opt/ml/processing")  # model/, test/ and evaluation/ live under here
LATENCY_SAMPLE = 50  # images timed for the latency metric


def find_model(model_dir: Path) -> Path:
    # Processing jobs copy model.tar.gz as-is (no auto-extract), so unpack it first.
    for archive in list(model_dir.rglob("*.tar.gz")):
        with tarfile.open(archive) as tar:
            tar.extractall(model_dir)
    for name in ("model.pt", "best.pt"):
        p = next(model_dir.rglob(name), None)
        if p:
            return p
    raise FileNotFoundError(f"No model weights found under {model_dir}")


def resolve_data_yaml(test_dir: Path, class_names: list[str]) -> str:
    """Use a data.yaml shipped with the test split, else generate one."""
    import yaml

    if (test_dir / "data.yaml").exists():
        return str(test_dir / "data.yaml")
    if not class_names:
        raise ValueError("no data.yaml in the test split and no --class-names given")
    data_yaml = test_dir.parent / "data.yaml"
    images = f"{test_dir.name}/images"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(test_dir.parent),
        "train": images,
        "val": images,
        "test": images,
        "names": dict(enumerate(class_names)),
    }, sort_keys=False))
    return str(data_yaml)


def evaluate_ultralytics(model_path: Path, data_yaml: str) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    results = model.val(data=data_yaml, split="test")

    box = results.box
    precision = float(box.mp)
    recall = float(box.mr)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # Mean per-image latency over a sample of test images (first call warms up).
    images_dir = Path(data_yaml).parent / "test" / "images"
    images = sorted(images_dir.glob("*"))[:LATENCY_SAMPLE]
    model.predict(source=str(images[0]), imgsz=640, verbose=False)
    t0 = time.perf_counter()
    for img in images:
        model.predict(source=str(img), imgsz=640, verbose=False)
    latency_ms = (time.perf_counter() - t0) * 1000.0 / max(len(images), 1)

    return {
        "map_50": round(float(box.map50), 4),
        "map_50_95": round(float(box.map), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "latency_ms": round(latency_ms, 2),
        "model_size_mb": round(model_path.stat().st_size / 1e6, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-names", default="[]", help="JSON list of class names in label-id order")
    parser.add_argument("--root", default=str(PROC), help="local testing: dir with model/ and test/ subfolders")
    args = parser.parse_args()

    root = Path(args.root)
    model_dir, test_dir, out_dir = root / "model", root / "test", root / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = find_model(model_dir)
    data_yaml = resolve_data_yaml(test_dir, json.loads(args.class_names))
    metrics = evaluate_ultralytics(model_path, data_yaml)

    report = {"metrics": metrics}
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"[evaluate] metrics: {json.dumps(metrics)}")


if __name__ == "__main__":
    main()
