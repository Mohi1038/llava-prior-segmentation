import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from llava_prior.models import (
    SamPredictorBackend,
    _configure_cuda_build_environment,
    _ensure_ninja_on_path,
    _ensure_quantized_kernel_build_tools,
    _model_key,
)


def test_model_key_maps_llava_checkpoint_prefixes():
    assert (
        _model_key("vision_tower.vision_model.encoder.layers.0.layer_norm1.weight")
        == "model.vision_tower.vision_model.encoder.layers.0.layer_norm1.weight"
    )


def test_ensure_ninja_on_path_adds_installed_binary_directory(tmp_path, monkeypatch):
    binary = tmp_path / "ninja.exe"
    binary.touch()
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(
        "ninja.BIN_DIR",
        str(tmp_path),
        raising=False,
    )

    _ensure_ninja_on_path()

    assert str(tmp_path) in os.environ["PATH"].split(os.pathsep)


def test_quantized_cuda_preflight_reports_missing_build_tools(monkeypatch):
    import torch.utils.cpp_extension

    monkeypatch.setattr(torch.utils.cpp_extension, "CUDA_HOME", None)
    monkeypatch.setenv("PATH", "")

    with pytest.raises(RuntimeError, match="CUDA Toolkit") as exc_info:
        _ensure_quantized_kernel_build_tools(torch.device("cuda"))

    assert "CUDA Toolkit" in str(exc_info.value)


def test_cuda_build_environment_discovers_nvcc(monkeypatch):
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.setattr(
        "llava_prior.models.shutil.which",
        lambda name: (
            r"D:\CUDA\bin\nvcc.exe" if name == "nvcc" else r"C:\tools\cl.exe"
        ),
    )

    _configure_cuda_build_environment()

    assert os.environ["CUDA_HOME"] == r"D:\CUDA"


def test_cuda_build_environment_activates_visual_studio(monkeypatch, tmp_path):
    import subprocess
    from types import SimpleNamespace

    installer = tmp_path / "Microsoft Visual Studio" / "Installer"
    installer.mkdir(parents=True)
    vswhere = installer / "vswhere.exe"
    vswhere.touch()
    build_tools = tmp_path / "2019" / "BuildTools"
    vcvars = build_tools / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    vcvars.parent.mkdir(parents=True)
    vcvars.touch()
    monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path))
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.setattr("llava_prior.models.shutil.which", lambda name: None)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == str(vswhere):
            return SimpleNamespace(stdout=f"{build_tools}\n")
        return SimpleNamespace(stdout="PATH=C:\\tools\nINCLUDE=C:\\include\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    _configure_cuda_build_environment()

    assert os.environ["PATH"] == "C:\\tools"
    assert os.environ["INCLUDE"] == "C:\\include"
    assert calls[1][3:] == ["call", str(vcvars), "&&", "set"]


def test_sam_selects_multimask_candidate_consistent_with_class_prior():
    image = Image.new("RGB", (8, 8))
    prior = torch.zeros((8, 8), dtype=torch.bool)
    prior[2:4, 2:4] = True
    target_mask = np.zeros((8, 8), dtype=bool)
    target_mask[1:5, 1:5] = True
    unrelated_mask = np.zeros((8, 8), dtype=bool)
    unrelated_mask[5:8, 5:8] = True

    class FakePredictor:
        def set_image(self, array):
            self.image = array

        def predict(self, **kwargs):
            assert kwargs["multimask_output"] is True
            assert kwargs["mask_input"].shape == (1, 256, 256)
            assert kwargs["mask_input"].dtype == np.float32
            return (
                np.stack([unrelated_mask, target_mask]),
                np.array([0.99, 0.10]),
                np.empty((2, 256, 256), dtype=np.float32),
            )

    backend = SamPredictorBackend.__new__(SamPredictorBackend)
    backend.device = torch.device("cpu")
    backend.predictor = FakePredictor()
    backend._embedded_image = None

    result = backend.predict(
        image,
        torch.tensor([[2.5, 2.5]]),
        torch.tensor([1.0, 1.0, 4.0, 4.0]),
        prior,
    )

    assert np.array_equal(result.numpy(), target_mask)


def test_sam_checkpoint_is_loaded_to_cpu_without_pickle_execution(tmp_path, monkeypatch):
    checkpoint = tmp_path / "sam.pth"
    checkpoint.touch()
    state_dict = {"weight": torch.ones(1)}
    calls = {}

    class FakeSam:
        def load_state_dict(self, values):
            calls["loaded_state_dict"] = values

        def to(self, device):
            calls["model_device"] = device
            return self

    class FakePredictor:
        def __init__(self, model):
            calls["predictor_model"] = model

    monkeypatch.setitem(
        sys.modules,
        "segment_anything",
        SimpleNamespace(
            SamPredictor=FakePredictor,
            sam_model_registry={"vit_h": lambda checkpoint: FakeSam()},
        ),
    )
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: calls.update(load_kwargs=kwargs) or state_dict)

    backend = SamPredictorBackend(checkpoint, device="cpu")

    assert calls["load_kwargs"] == {"map_location": "cpu", "weights_only": True}
    assert calls["loaded_state_dict"] is state_dict
    assert calls["model_device"] == torch.device("cpu")
    assert backend.predictor is not None
    assert (
        _model_key("language_model.model.layers.0.self_attn.q_proj.weight")
        == "model.language_model.layers.0.self_attn.q_proj.weight"
    )
