"""Tests for the container scripts' data.yaml generation (no Ultralytics needed)."""

import importlib.util
import json
from pathlib import Path

import yaml

import config

TRAINING = Path(__file__).resolve().parents[1] / "training"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TRAINING / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_train_generates_data_yaml_with_names(tmp_path):
    train = _load("train")
    classes = config.load_project_config("object_detection_amg").dataset.classes
    hp = {"class_names": json.dumps(classes)}  # as passed by pipeline/training.py
    data_yaml = yaml.safe_load(Path(train.resolve_data_yaml(tmp_path / "train", train.parse_class_names(hp))).read_text())
    assert data_yaml["path"] == str(tmp_path)
    assert data_yaml["train"] == "train/images" and data_yaml["val"] == "validation/images"
    assert data_yaml["names"][0] == "person" and len(data_yaml["names"]) == 17


def test_train_prefers_shipped_data_yaml(tmp_path):
    train = _load("train")
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "data.yaml").write_text("names: {0: x}\n")
    assert train.resolve_data_yaml(tmp_path / "train", []) == str(tmp_path / "train" / "data.yaml")


def test_train_requires_class_names_without_data_yaml(tmp_path):
    train = _load("train")
    try:
        train.resolve_data_yaml(tmp_path / "train", [])
    except ValueError as e:
        assert "class_names" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_train_decodes_sagemaker_json_encoded_hyperparameters():
    train = _load("train")
    # The PyTorch estimator JSON-encodes every value before writing hyperparameters.json.
    assert train._coerce("100") == 100 and train._coerce("0.01") == 0.01
    assert train._coerce('"rt_detr"') == "rt_detr" and train._coerce("true") is True
    assert train.parse_class_names({"class_names": train._coerce('["person", "bike"]')}) == ["person", "bike"]
    assert train.parse_class_names({"class_names": '["person"]'}) == ["person"]


def test_evaluate_generates_data_yaml_with_test_split(tmp_path):
    evaluate = _load("evaluate")
    data_yaml = yaml.safe_load(Path(evaluate.resolve_data_yaml(tmp_path / "test", ["a", "b"])).read_text())
    assert data_yaml["test"] == "test/images" and data_yaml["val"] == "test/images"
    assert data_yaml["names"] == {0: "a", 1: "b"}
