"""Concrete Hugging Face LLaVA and Segment Anything adapters."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .pipeline import TraceBatch


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        return _first_tensor(value[0])
    for name in ("pooler_output", "last_hidden_state", "image_embeds"):
        if hasattr(value, name):
            return _first_tensor(getattr(value, name))
    raise TypeError(f"could not find a tensor in {type(value).__name__}")


def _grid_shape(token_count: int) -> tuple[int, int]:
    height = int(token_count**0.5)
    while height > 1 and token_count % height:
        height -= 1
    return height, token_count // height


def _image_token_id(model: Any) -> int:
    for source in (model.config, getattr(model, "language_model", None)):
        if source is not None and hasattr(source, "image_token_index"):
            return int(source.image_token_index)
    return 32000


_CHECKPOINT_PREFIXES = (
    ("language_model.model.", "model.language_model."),
    ("language_model.lm_head.", "lm_head."),
    ("vision_tower.vision_model.", "model.vision_tower.vision_model."),
    ("multi_modal_projector.", "model.multi_modal_projector."),
)


def _model_key(checkpoint_key: str) -> str:
    for old, new in _CHECKPOINT_PREFIXES:
        if checkpoint_key.startswith(old):
            return new + checkpoint_key[len(old) :]
    return checkpoint_key


def _ensure_ninja_on_path() -> None:
    """Expose the pip-installed Ninja executable to PyTorch's extension loader."""
    import ninja

    binary_dir = str(Path(ninja.BIN_DIR))
    if shutil.which("ninja") is None and binary_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = os.pathsep.join(
            part for part in (binary_dir, os.environ.get("PATH", "")) if part
        )
    if shutil.which("ninja") is None:
        raise RuntimeError(
            "Quanto inference requires the Ninja executable; install the project quant extra "
            "and ensure its Scripts directory is available on PATH"
        )


def _configure_cuda_build_environment() -> None:
    if not os.environ.get("CUDA_HOME"):
        nvcc = shutil.which("nvcc")
        if nvcc:
            os.environ["CUDA_HOME"] = str(Path(nvcc).resolve().parent.parent)

    if os.name != "nt" or shutil.which("cl"):
        return

    program_files = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        raise RuntimeError("Visual C++ Build Tools are missing; install the MSVC x64 toolset")
    installations = subprocess.run(
        [
            str(vswhere),
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
            "-all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    installations = [Path(path.strip()) for path in installations if path.strip()]
    installations.sort(key=lambda path: 0 if "\\2019\\" in str(path) else 1)
    vcvars = next(
        (
            path / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
            for path in installations
            if (path / "VC" / "Auxiliary" / "Build" / "vcvars64.bat").is_file()
        ),
        None,
    )
    if vcvars is None:
        raise RuntimeError("Visual C++ Build Tools are missing the x64 environment script")

    environment_output = subprocess.run(
        [
            os.environ.get("COMSPEC", "cmd.exe"),
            "/d",
            "/c",
            "call",
            str(vcvars),
            "&&",
            "set",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for line in environment_output.splitlines():
        if "=" in line and not line.startswith("="):
            name, value = line.split("=", 1)
            os.environ[name] = value


def _ensure_quantized_kernel_build_tools(device: torch.device) -> None:
    if device.type != "cuda":
        return
    _configure_cuda_build_environment()
    from torch.utils.cpp_extension import CUDA_HOME

    compiler_name = "nvcc.exe" if os.name == "nt" else "nvcc"
    has_cuda_compiler = CUDA_HOME is not None and (
        Path(CUDA_HOME) / "bin" / compiler_name
    ).is_file()
    missing = []
    if not has_cuda_compiler:
        missing.append("CUDA Toolkit (nvcc; set CUDA_HOME)")
    if os.name == "nt" and shutil.which("cl") is None:
        missing.append("Visual C++ Build Tools (cl.exe)")
    if missing:
        raise RuntimeError(
            "Quanto CUDA inference needs " + " and ".join(missing) + " to compile its kernels"
        )


def _load_llava_quantized(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    quantization: str,
) -> Any:
    """Load LLaVA with a quantized language model, one tensor at a time.

    Transformers' QuantoConfig path materializes fp16 weights before quantizing, which
    can exceed available memory. Here the model starts on the meta device and each
    language-model weight is quantized as soon as it lands on the device. The vision
    tower, projector, embeddings and lm_head stay in full precision because VGM
    prototypes are built from their features.
    """
    import gc
    import json

    _configure_cuda_build_environment()
    _ensure_ninja_on_path()
    _ensure_quantized_kernel_build_tools(device)

    from huggingface_hub import snapshot_download
    from optimum.quanto import QModuleMixin, qint4, qint8, quantize
    from safetensors import safe_open
    from transformers import AutoConfig, LlavaForConditionalGeneration

    path = Path(snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors"]))
    config = AutoConfig.from_pretrained(path, attn_implementation="eager")
    with torch.device("meta"):
        model = LlavaForConditionalGeneration(config)
    weight_map = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    unmatched = {_model_key(key) for key in weight_map} ^ set(model.state_dict())
    if unmatched:
        raise RuntimeError(f"checkpoint/model key mismatch: {sorted(unmatched)[:5]}")
    language_model = model.model.language_model
    weight_type = {"int4": qint4, "int8": qint8}.get(quantization)
    if weight_type is None:
        raise ValueError(f"unsupported quantization {quantization!r}")
    quantize(language_model, weights=weight_type)

    for shard in sorted(set(weight_map.values())):
        with safe_open(str(path / shard), framework="numpy") as tensors:
            for checkpoint_key in tensors.keys():  # noqa: SIM118 (safe_open has no __iter__)
                module_name, _, tensor_name = _model_key(checkpoint_key).rpartition(".")
                module = model.get_submodule(module_name)
                # The PyTorch-backed reader can access-violate on Windows after
                # Transformers is imported. The NumPy reader returns writable arrays.
                array = tensors.get_tensor(checkpoint_key)
                value = torch.from_numpy(array).to(device=device, dtype=dtype)
                module._parameters[tensor_name] = torch.nn.Parameter(value, requires_grad=False)
                if isinstance(module, QModuleMixin) and tensor_name == "weight":
                    module.freeze()
                del array, value
        del tensors
        gc.collect()

    # Non-persistent buffers are absent from the checkpoint and must be rebuilt.
    language_model.rotary_emb = type(language_model.rotary_emb)(config=config.text_config, device=device)
    embeddings = model.model.vision_tower.vision_model.embeddings
    embeddings.position_ids = torch.arange(embeddings.num_positions, device=device).unsqueeze(0)
    for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
        if tensor.is_meta:
            if name.endswith(("input_scale", "output_scale")):
                module_name, _, buffer_name = name.rpartition(".")
                model.get_submodule(module_name)._buffers[buffer_name] = torch.ones((), dtype=dtype, device=device)
            else:
                raise RuntimeError(f"{name} was not loaded from the checkpoint")
    return model


@dataclass
class HuggingFaceLlavaBackend:
    """Extract the traces required by FSeg-LLaVA from a frozen HF checkpoint."""

    model_id: str = "llava-hf/llava-1.5-7b-hf"
    device_name: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 96
    quantize: str = "none"

    def __post_init__(self) -> None:
        self.device = choose_device(self.device_name)
        if self.quantize in {"int4", "int8"} and self.device.type == "cuda":
            _configure_cuda_build_environment()
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        torch_dtype = self._torch_dtype()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        if self.quantize in {"int4", "int8"}:
            self.model = _load_llava_quantized(
                self.model_id,
                self.device,
                torch_dtype or torch.float16,
                self.quantize,
            )
        elif self.quantize == "none":
            # Eager attention is required for output_attentions; loading straight onto the
            # device avoids holding a second full copy of the weights in unified memory.
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id,
                dtype=torch_dtype,
                attn_implementation="eager",
                device_map=str(self.device),
            )
        else:
            raise ValueError(f"unsupported quantization {self.quantize!r}")
        self.model.eval()

    def _torch_dtype(self) -> torch.dtype | None:
        if self.dtype == "float32" or self.device.type == "cpu":
            return torch.float32
        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "bfloat16":
            return torch.bfloat16
        return torch.float16 if self.device.type in {"cuda", "mps"} else None

    def _inputs(self, image: object, text: str) -> dict[str, torch.Tensor]:
        encoded = self.processor(text=text, images=image, return_tensors="pt")
        return {key: value.to(self.device) if hasattr(value, "to") else value for key, value in encoded.items()}

    def _prompt(self, query: str) -> str:
        conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]}]
        return self.processor.apply_chat_template(conversation, add_generation_prompt=True)

    @torch.inference_mode()
    def answer_with_trace(self, image: object, query: str) -> tuple[str, TraceBatch]:
        inputs = self._inputs(image, self._prompt(query))
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        prompt_length = inputs["input_ids"].shape[1]
        response_ids = generated[0, prompt_length:].tolist()
        eos_id = self.processor.tokenizer.eos_token_id
        if eos_id in response_ids:
            response_ids = response_ids[: response_ids.index(eos_id)]
        response = self.processor.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        if not response_ids:
            raise RuntimeError("LLaVA generated an empty response")
        # Teacher-force the exact generated ids so every position is aligned by construction.
        sequence = generated[:, : prompt_length + len(response_ids)]
        trace = self._trace(image, inputs, sequence, prompt_length, query)
        return response, trace

    def _trace(
        self,
        image: object,
        inputs: dict[str, torch.Tensor],
        sequence: torch.Tensor,
        prompt_length: int,
        query: str,
    ) -> TraceBatch:
        outputs = self.model(
            input_ids=sequence,
            attention_mask=torch.ones_like(sequence),
            pixel_values=inputs["pixel_values"],
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states
        attentions = outputs.attentions
        if hidden_states is None or attentions is None or any(a is None for a in attentions):
            raise RuntimeError("the selected LLaVA implementation did not return hidden states/attentions")

        # The processor expands <image> to one placeholder per visual token, so input_ids
        # positions index hidden states directly.
        input_ids = sequence[0]
        visual_positions = torch.where(input_ids == _image_token_id(self.model))[0]
        if len(visual_positions) == 0:
            raise RuntimeError("could not locate the image tokens in the processed prompt")
        visual_start = int(visual_positions[0])
        visual_count = len(visual_positions)
        visual_end = visual_start + visual_count
        if int(visual_positions[-1]) != visual_end - 1:
            raise RuntimeError("image tokens are not contiguous in the processed prompt")

        image_features = getattr(outputs, "image_hidden_states", None)
        if image_features is None:
            image_features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        image_features = _first_tensor(image_features)
        if image_features.ndim == 3:
            image_features = image_features[0]
        if image_features.shape[0] != visual_count:
            raise RuntimeError(
                f"{image_features.shape[0]} vision features for {visual_count} image tokens"
            )

        stacked_hidden = torch.stack([layer[0] for layer in hidden_states[1:]])
        image_hidden = stacked_hidden[:, visual_start:visual_end]

        processed_ids = input_ids.tolist()
        class_name = query.split("class of a ", 1)[-1].split("?", 1)[0].strip()
        class_start = self._find_class(processed_ids[visual_end:prompt_length], class_name)
        if class_start is None:
            raise RuntimeError(f"could not locate class tokens for {class_name!r} in the query")
        class_start, class_length = class_start
        class_hidden = stacked_hidden[:, visual_end + class_start + class_length - 1]

        response_start = prompt_length + self._description_offset(processed_ids[prompt_length:])
        response_end = len(processed_ids)
        attention_rows = []
        for layer_attention in attentions:
            layer = layer_attention[0].mean(dim=0)
            attention_rows.append(layer[response_start:response_end, visual_start:visual_end])
        description_attention = torch.cat(attention_rows, dim=0)
        height, width = _grid_shape(visual_count)
        image_width, image_height = self._image_size(image)
        return TraceBatch(
            image_features=image_hidden.float().cpu(),
            class_feature=class_hidden.float().cpu(),
            description_attention=description_attention.float().cpu(),
            vision_features=image_features.float().cpu(),
            spatial_shape=(height, width),
            image_size=(image_height, image_width),
        )

    @staticmethod
    def _find_subsequence(sequence: list[int], subsequence: list[int]) -> int | None:
        if not subsequence:
            return None
        for index in range(len(sequence) - len(subsequence) + 1):
            if sequence[index : index + len(subsequence)] == subsequence:
                return index
        return None

    def _find_class(self, query_ids: list[int], class_name: str) -> tuple[int, int] | None:
        """Return (start, length) of the class name's tokens inside the query ids."""
        tokenizer = self.processor.tokenizer
        for text in (class_name, f" {class_name}"):
            candidate = tokenizer(text, add_special_tokens=False)["input_ids"]
            # SentencePiece may emit a bare "▁" token that never appears mid-sentence.
            while candidate and tokenizer.decode(candidate[:1]).strip() == "":
                candidate = candidate[1:]
            start = self._find_subsequence(query_ids, candidate)
            if start is not None:
                return start, len(candidate)
        return None

    def _description_offset(self, response_ids: list[int]) -> int:
        """Index of the first token after the 'Yes, it contains a ...' sentence."""
        tokenizer = self.processor.tokenizer
        for index in range(1, len(response_ids)):
            text = tokenizer.decode(response_ids[:index], skip_special_tokens=True)
            if any(mark in text for mark in ".!?\n"):
                return index if index < len(response_ids) else 0
        return 0

    @staticmethod
    def _image_size(image: object) -> tuple[int, int]:
        if hasattr(image, "size"):
            return tuple(reversed(image.size))
        array = np.asarray(image)
        return int(array.shape[0]), int(array.shape[1])


class SamPredictorBackend:
    """Thin adapter around the official Segment Anything predictor."""

    def __init__(self, checkpoint: str | Path, model_type: str = "vit_h", device: str = "auto") -> None:
        from segment_anything import SamPredictor, sam_model_registry

        self.device = choose_device(device)
        sam = sam_model_registry[model_type](checkpoint=None)
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        sam.load_state_dict(state_dict)
        self.predictor = SamPredictor(sam.to(self.device))
        self._embedded_image: object | None = None

    def predict(
        self,
        image: object,
        points: torch.Tensor,
        box: torch.Tensor | None,
        mask_prior: torch.Tensor,
    ) -> torch.Tensor:
        array = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
        if not points.numel() and box is None:
            # VGM found no foreground component; SAM cannot run without a prompt.
            return torch.zeros(array.shape[:2], dtype=torch.bool)
        if mask_prior.shape != array.shape[:2]:
            raise ValueError("mask_prior must match the image height and width")
        if self._embedded_image is not image:
            # The ViT-H image embedding is shared by every class query on the same image.
            self.predictor.set_image(array)
            self._embedded_image = image
        point_array = points.detach().cpu().numpy() if points.numel() else None
        labels = np.ones(len(point_array), dtype=np.int32) if point_array is not None else None
        box_array = box.detach().cpu().numpy() if box is not None else None
        import cv2

        prior = mask_prior.detach().cpu().numpy().astype(np.uint8)
        mask_input = cv2.resize(prior, (256, 256), interpolation=cv2.INTER_LINEAR)
        mask_input = ((mask_input * 2.0 - 1.0) * 8.0).astype(np.float32)[None, :, :]
        masks, scores, _ = self.predictor.predict(
            point_coords=point_array,
            point_labels=labels,
            box=box_array,
            mask_input=mask_input,
            multimask_output=True,
        )
        overlaps = []
        for mask in masks:
            intersection = np.count_nonzero(mask & prior.astype(bool))
            union = np.count_nonzero(mask | prior.astype(bool))
            overlaps.append(intersection / union if union else 0.0)
        selection_scores = np.asarray(overlaps) + np.asarray(scores) * 1e-3
        return torch.from_numpy(masks[int(np.argmax(selection_scores))].copy())