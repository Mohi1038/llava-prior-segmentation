"""Qwen2.5-VL tracer: the FSeg-LLaVA extraction on a VLM with a different architecture.

Differences from LLaVA that this adapter handles:

* the image is stretched to a square (as for LLaVA) and resized to ``side`` pixels, which Qwen's
  ViT (14-px patches, 2x2 token merging) turns into a (side/28)^2 grid; 672 gives the same 24x24
  grid as LLaVA-1.5;
* image tokens carry 2-D rotary positions (M-RoPE). The prefix is run with the positions from
  ``get_rope_index``; every later text token gets position ``cache length + rope delta``;
* no BOS token; answers end with ``<|im_end|>``;
* prototype features are the ViT's last hidden states before the 2x2 merger, averaged over each
  merge group and put back into raster order (the analogue of LLaVA's CLIP patch features).

Everything downstream (class-token maps, attention maps, caching) is inherited from LlavaTracer.
"""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from .trace import QUERY_TEMPLATE, LlavaTracer, TraceSettings, square_resize

SYSTEM = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>"


class QwenTracer(LlavaTracer):
    side = 672

    def __init__(self, settings: TraceSettings, device: str = "cuda") -> None:
        from transformers import (
            AutoImageProcessor,
            AutoTokenizer,
            Qwen2_5_VLForConditionalGeneration,
        )

        self.settings = settings
        self.device = torch.device(device)
        self.dtype = getattr(torch, settings.dtype)
        self.template = "qwen_chatml"
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_id)
        pixels = self.side * self.side
        self.image_processor = AutoImageProcessor.from_pretrained(
            settings.model_id, size={"shortest_edge": pixels, "longest_edge": pixels}
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            settings.model_id, dtype=self.dtype, attn_implementation="eager"
        ).to(self.device)
        self.model.eval()
        inner = self.model.model
        self.visual = inner.visual
        self.language_model = inner.language_model
        self.lm_head = self.model.lm_head
        self.embed = self.language_model.get_input_embeddings()
        self.image_token_id = self.model.config.image_token_id
        ids = set()
        for word in ("Yes", "yes", " Yes", " yes"):
            pieces = self.tokenizer(word, add_special_tokens=False).input_ids
            if len(pieces) == 1:
                ids.add(pieces[0])
        self.yes_ids = sorted(ids)
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos]) | {self.tokenizer.convert_tokens_to_ids("<|im_end|>")}
        self.rope_delta = 0
        self.grid_thw: torch.Tensor | None = None

    # -- tokenization ------------------------------------------------------------------------
    def _text_ids(self, text: str, keep_bos: bool) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def _query_ids(self, class_name: str, query_template: str = QUERY_TEMPLATE) -> list[int]:
        query = query_template.format(class_name.replace(",", " or "))
        return self._text_ids(f"<|vision_end|>{query}<|im_end|>\n<|im_start|>assistant\n", keep_bos=False)

    # -- model pieces ------------------------------------------------------------------------
    @torch.inference_mode()
    def encode_image(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import get_vision_window_index

        square = square_resize(image).resize((self.side, self.side), Image.Resampling.BICUBIC)
        inputs = self.image_processor(images=[square], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, self.dtype)
        self.grid_thw = inputs["image_grid_thw"].to(self.device)
        out = self.visual(pixel_values, grid_thw=self.grid_thw)
        merge = self.visual.spatial_merge_size**2
        window_index, _ = get_vision_window_index(
            self.grid_thw,
            spatial_merge_size=self.visual.spatial_merge_size,
            window_size=self.visual.window_size,
            patch_size=self.visual.patch_size,
            kwargs={},
        )
        reverse = torch.argsort(window_index)
        hidden = out.last_hidden_state
        pre_merge = hidden.view(-1, merge, hidden.shape[-1]).mean(1)[reverse]
        return pre_merge, out.pooler_output

    def _prefix_forward(self, projected: torch.Tensor) -> tuple[Any, int]:
        before = self._text_ids(SYSTEM, keep_bos=False)
        count = projected.shape[0]
        ids = torch.tensor([before + [self.image_token_id] * count], device=self.device)
        token_types = torch.zeros_like(ids, dtype=torch.int)
        token_types[0, len(before) :] = 1
        positions, delta = self.model.model.get_rope_index(
            ids, mm_token_type_ids=token_types, image_grid_thw=self.grid_thw
        )
        self.rope_delta = int(delta.reshape(-1)[0])
        embeds = self.embed(ids).clone()
        embeds[0, len(before) :] = projected.to(self.dtype)
        out = self.language_model(
            inputs_embeds=embeds,
            position_ids=positions,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        return out, len(before)

    def _positions(self, positions: torch.Tensor) -> torch.Tensor:
        shifted = positions + self.rope_delta
        return shifted[None].expand(3, -1, -1)

    def _forward(self, embeds: torch.Tensor, cache: Any, attentions: bool = False) -> Any:
        start = cache.get_seq_length()
        positions = torch.arange(start, start + embeds.shape[1], device=self.device)[None]
        return self.language_model(
            inputs_embeds=embeds,
            past_key_values=cache,
            position_ids=self._positions(positions),
            use_cache=True,
            output_hidden_states=True,
            output_attentions=attentions,
            return_dict=True,
        )
