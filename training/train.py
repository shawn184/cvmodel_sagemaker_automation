"""SageMaker training entrypoint.

Reads hyperparameters + input data from the standard SageMaker paths, fine-tunes
an approved object-detection model, and writes the trained artifact to
``/opt/ml/model`` (which SageMaker packages as model.tar.gz to S3).

The container is model-family aware: `model_family` in the hyperparameters
selects the training backend. YOLO / RT-DETR use Ultralytics; Faster R-CNN uses
torchvision.

Run locally for testing with (dataset_dir has train/ and validation/ splits):
    python train.py --local <dataset_dir>/train --hp '{"epochs": 1, "class_names": "[\"person\"]"}'
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# SageMaker standard container paths.
PREFIX = Path("/opt/ml")
INPUT_DATA = PREFIX / "input" / "data"
HP_PATH = PREFIX / "input" / "config" / "hyperparameters.json"
MODEL_DIR = PREFIX / "model"
OUTPUT_DIR = PREFIX / "output"


def load_hyperparameters() -> dict:
    if HP_PATH.exists():
        raw = json.loads(HP_PATH.read_text())
        # Skip the SDK's own sagemaker_* entries (program name, source dir, ...).
        return {k: _coerce(v) for k, v in raw.items() if not k.startswith("sagemaker_")}
    return {}


def _coerce(v):
    """SageMaker passes values as JSON-encoded strings ("100", "\\"yolo\\"", "[...]")."""
    if not isinstance(v, str):
        return v
    try:
        return json.loads(v)
    except ValueError:
        return v


def train_ultralytics(hp: dict, data_yaml: str, model_dir: Path) -> Path:
    """Fine-tune a YOLO / RT-DETR model with Ultralytics."""
    from ultralytics import YOLO, RTDETR

    base_weights = hp.get("base_weights") or "yolo11n.pt"
    family = hp.get("model_family", "yolo")
    Model = RTDETR if family == "rt_detr" else YOLO
    model = Model(base_weights)

    # Forward tunable Ultralytics args; unset keys keep Ultralytics defaults.
    train_args = ("epochs", "batch", "imgsz", "lr0", "weight_decay", "patience", "optimizer")
    kwargs = {k: hp[k] for k in train_args if k in hp}

    model.train(data=data_yaml, project=str(model_dir), name="train", exist_ok=True, **kwargs)
    best = model_dir / "train" / "weights" / "best.pt"
    return best


def train_torchvision(hp: dict, data_dir: Path, model_dir: Path) -> Path:
    raise NotImplementedError(
        "Faster R-CNN (torchvision) training backend is not implemented in this "
        "POC container. Use an Ultralytics-family model (yolo, rt_detr) or add "
        "the torchvision training loop here."
    )


def parse_class_names(hp: dict) -> list[str]:
    """`class_names` arrives as a JSON list string from the pipeline."""
    raw = hp.get("class_names", "[]")
    return list(raw) if isinstance(raw, list) else list(json.loads(raw))


def resolve_data_yaml(train_dir: Path, class_names: list[str]) -> str:
    """Use a data.yaml shipped with the train split, else generate one.

    Layout: <root>/train/{images,labels} and <root>/validation/{images,labels}.
    """
    import yaml

    if (train_dir / "data.yaml").exists():
        return str(train_dir / "data.yaml")
    if not class_names:
        raise ValueError("no data.yaml in the train split and no class_names hyperparameter")
    data_yaml = train_dir.parent / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(train_dir.parent),
        "train": f"{train_dir.name}/images",
        "val": "validation/images",
        "names": dict(enumerate(class_names)),
    }, sort_keys=False))
    return str(data_yaml)


def main() -> None:
    hp = load_hyperparameters()
    if "--hp" in sys.argv:  # local testing: override/extend hyperparameters
        hp.update(json.loads(sys.argv[sys.argv.index("--hp") + 1]))
    family = hp.get("model_family", "yolo")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if "--local" in sys.argv:
        train_root = Path(sys.argv[sys.argv.index("--local") + 1])
    else:
        train_root = INPUT_DATA / "train"

    print(f"[train] family={family} hyperparameters={hp}")

    if family in ("yolo", "rt_detr"):
        data_yaml = resolve_data_yaml(train_root, parse_class_names(hp))
        best = train_ultralytics(hp, data_yaml, MODEL_DIR)
    else:
        best = train_torchvision(hp, train_root, MODEL_DIR)

    # Ensure the artifact lives at the top of MODEL_DIR for packaging.
    final = MODEL_DIR / "model.pt"
    if best.exists():
        final.write_bytes(best.read_bytes())
    (MODEL_DIR / "training_metadata.json").write_text(json.dumps({"hyperparameters": hp}, indent=2))
    print(f"[train] wrote model artifact to {final}")


if __name__ == "__main__":
    main()
