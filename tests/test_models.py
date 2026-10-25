import sys
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from llava_prior.models import SamPredictorBackend, _model_key


def test_model_key_maps_llava_checkpoint_prefixes():
    assert (
        _model_key("vision_tower.vision_model.encoder.layers.0.layer_norm1.weight")
        == "model.vision_tower.vision_model.encoder.layers.0.layer_norm1.weight"
    )


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
