import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from llava_prior import cli
from llava_prior.cli import _discover_images, _is_complete, _output_dir_for_image


def test_discover_images_is_sorted_and_non_recursive_by_default(tmp_path):
    (tmp_path / "b.png").touch()
    (tmp_path / "a.JPG").touch()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.jpeg").touch()

    assert _discover_images(tmp_path, recursive=False) == [
        tmp_path / "a.JPG",
        tmp_path / "b.png",
    ]
    assert _discover_images(tmp_path, recursive=True) == [
        tmp_path / "a.JPG",
        tmp_path / "b.png",
        nested / "c.jpeg",
    ]


def test_discover_images_reports_missing_or_unsupported_input(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _discover_images(tmp_path / "missing.png", recursive=False)
    unsupported = tmp_path / "document.txt"
    unsupported.touch()
    with pytest.raises(ValueError, match="unsupported image format"):
        _discover_images(unsupported, recursive=False)


def test_batch_output_paths_preserve_relative_subdirectories(tmp_path):
    inputs = tmp_path / "images"
    nested = inputs / "nested"
    nested.mkdir(parents=True)
    image = nested / "scene.png"

    assert _output_dir_for_image(image, inputs, tmp_path / "outputs") == (
        tmp_path / "outputs" / "nested" / "scene"
    )


def test_resume_requires_complete_outputs_and_matching_run_settings(tmp_path):
    image = tmp_path / "image.png"
    image.touch()
    expected = {"candidate_classes": ["dog"], "quantization": "int8"}
    (tmp_path / "predictions.json").write_text("[]", encoding="utf-8")
    (tmp_path / "semantic_mask.npy").touch()
    (tmp_path / "visualization.png").touch()
    (tmp_path / "run_metadata.json").write_text(
        json.dumps({"image": str(image.resolve()), **expected}),
        encoding="utf-8",
    )

    assert _is_complete(tmp_path, image, expected)
    assert not _is_complete(tmp_path, image, {"candidate_classes": ["cat"]})
    (tmp_path / "visualization.png").unlink()
    assert not _is_complete(tmp_path, image, expected)


def test_resume_rejects_outputs_missing_the_class_prior(tmp_path):
    image = tmp_path / "image.png"
    image.touch()
    expected = {"sam_prompt_strategy": "points_box_mask_prior_multimask_iou_v1"}
    (tmp_path / "predictions.json").write_text(
        json.dumps([{"class": "cat", "mask": "cat.npy"}]),
        encoding="utf-8",
    )
    (tmp_path / "semantic_mask.npy").touch()
    (tmp_path / "visualization.png").touch()
    (tmp_path / "run_metadata.json").write_text(
        json.dumps({"image": str(image.resolve()), **expected}),
        encoding="utf-8",
    )

    assert not _is_complete(tmp_path, image, expected)
    (tmp_path / "cat_prior.npy").touch()
    (tmp_path / "predictions.json").write_text(
        json.dumps([{"class": "cat", "prior": str(tmp_path / "cat_prior.npy")}]),
        encoding="utf-8",
    )
    assert _is_complete(tmp_path, image, expected)


def test_directory_run_processes_images_and_writes_a_summary(tmp_path, monkeypatch):
    inputs = tmp_path / "images"
    inputs.mkdir()
    for name, color in (("a.png", (20, 40, 60)), ("b.png", (80, 100, 120))):
        Image.new("RGB", (8, 6), color).save(inputs / name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "llava-prior",
        "images",
        "--classes",
        "dog",
        "--sam-checkpoint",
        str(tmp_path / "sam.pth"),
        "--device",
        "cpu",
        "--output-dir",
        str(tmp_path / "outputs"),
    ])
    monkeypatch.setattr(
        cli,
        "HuggingFaceLlavaBackend",
        lambda **kwargs: SimpleNamespace(device="cpu"),
    )
    monkeypatch.setattr(
        cli,
        "SamPredictorBackend",
        lambda *args: SimpleNamespace(device="cpu"),
    )
    calls = []

    def fake_pipeline(image, *args):
        calls.append(image.size)
        return [
            SimpleNamespace(
                observation=SimpleNamespace(class_name="dog", response="Yes, a dog."),
                sam_mask=torch.ones((image.height, image.width), dtype=torch.bool),
                sam_prior_mask=torch.eye(image.height, image.width, dtype=torch.bool),
            )
        ]

    monkeypatch.setattr(cli, "run_pipeline", fake_pipeline)

    cli.main()

    summary = json.loads((tmp_path / "outputs" / "batch_summary.json").read_text())
    assert calls == [(8, 6), (8, 6)]
    assert summary["success_count"] == 2
    assert summary["completed_count"] == 2
    assert summary["failed_count"] == 0
    for name in ("a", "b"):
        output = tmp_path / "outputs" / name
        assert (output / "visualization.png").is_file()
        assert np.load(output / "semantic_mask.npy").shape == (6, 8)
        assert np.load(output / "dog_prior.npy").shape == (6, 8)
        predictions = json.loads((output / "predictions.json").read_text())
        assert Path(predictions[0]["prior"]).is_file()


def test_model_initialization_failure_is_recorded_for_each_pending_image(tmp_path, monkeypatch):
    inputs = tmp_path / "images"
    inputs.mkdir()
    for name in ("a.png", "b.png"):
        Image.new("RGB", (8, 6)).save(inputs / name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "llava-prior",
            "images",
            "--classes",
            "dog",
            "--sam-checkpoint",
            str(tmp_path / "sam.pth"),
            "--device",
            "cuda",
            "--output-dir",
            str(tmp_path / "outputs"),
        ],
    )

    def fail_to_load_model(**kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(cli, "HuggingFaceLlavaBackend", fail_to_load_model)

    with pytest.raises(SystemExit, match="1"):
        cli.main()

    summary = json.loads((tmp_path / "outputs" / "batch_summary.json").read_text())
    assert summary["image_count"] == 2
    assert summary["success_count"] == 0
    assert summary["failed_count"] == 2
    assert {item["status"] for item in summary["results"]} == {"failed"}
    assert all("model initialization failed" in item["error"] for item in summary["results"])
    assert not (tmp_path / "outputs" / "a" / "semantic_mask.npy").exists()


def test_cli_exposes_int4_quantization_option(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["llava-prior", "--help"])

    with pytest.raises(SystemExit, match="0"):
        cli.main()

    assert "int4" in capsys.readouterr().out


def test_cli_config_overrides_are_applied_and_recorded(tmp_path, monkeypatch):
    inputs = tmp_path / "images"
    inputs.mkdir()
    Image.new("RGB", (8, 6)).save(inputs / "sample.png")
    config_path = tmp_path / "ablation.json"
    config_path.write_text(
        json.dumps({"layer_start": 4, "layer_end": 10, "attention_top_k": 3}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "llava-prior",
            "images",
            "--classes",
            "dog",
            "--sam-checkpoint",
            str(tmp_path / "sam.pth"),
            "--device",
            "cpu",
            "--config",
            str(config_path),
            "--output-dir",
            "outputs",
        ],
    )
    monkeypatch.setattr(
        cli,
        "HuggingFaceLlavaBackend",
        lambda **kwargs: SimpleNamespace(device="cpu"),
    )
    monkeypatch.setattr(
        cli,
        "SamPredictorBackend",
        lambda *args: SimpleNamespace(device="cpu"),
    )
    configs = []
    monkeypatch.setattr(
        cli,
        "run_pipeline",
        lambda image, classes, llava, sam, config: configs.append(config) or [],
    )

    cli.main()

    assert len(configs) == 1
    assert configs[0].layer_start == 4
    assert configs[0].layer_end == 10
    assert configs[0].attention_top_k == 3
    metadata = json.loads(
        (tmp_path / "outputs" / "sample" / "run_metadata.json").read_text()
    )
    assert metadata["config"]["layer_start"] == 4
    assert metadata["config"]["candidate_classes"] == ["dog"]


def test_directory_run_uses_per_image_class_map(tmp_path, monkeypatch):
    inputs = tmp_path / "images"
    inputs.mkdir()
    for name in ("dog_image.png", "cat_image.png"):
        Image.new("RGB", (8, 6)).save(inputs / name)
    class_map = tmp_path / "class_map.json"
    class_map.write_text(
        json.dumps({"dog_image": ["dog"], "cat_image": ["cat"]}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "llava-prior",
        "images",
        "--classes",
        "dog",
        "cat",
        "--class-map",
        str(class_map),
        "--sam-checkpoint",
        str(tmp_path / "sam.pth"),
        "--device",
        "cpu",
        "--output-dir",
        "outputs",
    ])
    monkeypatch.setattr(
        cli,
        "HuggingFaceLlavaBackend",
        lambda **kwargs: SimpleNamespace(device="cpu"),
    )
    monkeypatch.setattr(
        cli,
        "SamPredictorBackend",
        lambda *args: SimpleNamespace(device="cpu"),
    )
    calls = []
    monkeypatch.setattr(
        cli,
        "run_pipeline",
        lambda image, classes, *args: calls.append(tuple(classes)) or [],
    )

    cli.main()

    assert calls == [("cat",), ("dog",)]
    for image_id, class_name in (("cat_image", "cat"), ("dog_image", "dog")):
        metadata = json.loads(
            (tmp_path / "outputs" / image_id / "run_metadata.json").read_text()
        )
        assert metadata["candidate_classes"] == [class_name]


def test_directory_run_records_image_failures_and_returns_nonzero(tmp_path, monkeypatch):
    inputs = tmp_path / "images"
    inputs.mkdir()
    Image.new("RGB", (8, 6)).save(inputs / "bad.png")
    monkeypatch.setattr(sys, "argv", [
        "llava-prior",
        str(inputs),
        "--classes",
        "dog",
        "--sam-checkpoint",
        str(tmp_path / "sam.pth"),
        "--device",
        "cpu",
        "--output-dir",
        str(tmp_path / "outputs"),
    ])
    monkeypatch.setattr(
        cli,
        "HuggingFaceLlavaBackend",
        lambda **kwargs: SimpleNamespace(device="cpu"),
    )
    monkeypatch.setattr(
        cli,
        "SamPredictorBackend",
        lambda *args: SimpleNamespace(device="cpu"),
    )

    def fail_pipeline(*args):
        raise RuntimeError("model output failed")

    monkeypatch.setattr(cli, "run_pipeline", fail_pipeline)

    with pytest.raises(SystemExit) as error:
        cli.main()

    summary = json.loads((tmp_path / "outputs" / "batch_summary.json").read_text())
    assert error.value.code == 1
    assert summary["failed_count"] == 1
    assert summary["completed_count"] == 0
    assert "RuntimeError: model output failed" in summary["results"][0]["error"]
