"""Stage A: run LLaVA once per image and cache everything FSeg-LLaVA needs.

The extraction follows the authors' released implementation (github.com/zbf1991/FSeg-LLaVA)
rather than only the paper text, because the two differ in details that move mIoU:

* images are resized (not padded or cropped) to a square before CLIP preprocessing;
* prompts use the LLaVA-1.5 ``llava_v1`` conversation template and the exact query wording;
* the class response averages cosine *similarity* over hidden-state indices 7..13
  (index 0 = embeddings) between every image token and every class-name token;
* during generation the hidden state used for a response token is the one computed at the
  previous position (the state that predicts the token); ``aligned`` stores the alternative;
* text-to-image attention is kept per (layer, head) for layers 2..L-3 and averaged over the
  description tokens, so that head selection can be redone offline;
* vision prototypes use CLIP patch features (vision layer -2), not projected features.

Everything is cached per image (``<id>.npz`` + ``<id>.json``) so the SAM stage and all of its
ablations can be rerun without LLaVA.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SYSTEM_PROMPT = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions."
)
QUERY_TEMPLATE = (
    "Does the image have the class of a {0} ? Answer yes or no for this question. "
    "If no, just say 'No.' If yes, the answer has to be two sentences, the first "
    "sentence is 'yes, it contains a {0}.' the second sentence is to describe the {0}. "
)
# Tokens that never count as class-name tokens (official ``EXCLUDE_WORDS``).
EXCLUDE_WORDS = frozenset(
    {"yes", ",", "it", "contains", "has", "or", ".", "a", "is", "image", "in", "the", "class"}
)


@dataclass(frozen=True)
class TraceSettings:
    model_id: str = "llava-hf/llava-1.5-7b-hf"
    max_new_tokens: int = 128
    layer_start: int = 7  # hidden-state tuple index, inclusive
    layer_end: int = 14  # exclusive, as in the official ``[7:14]`` slice
    attention_layer_trim: int = 2  # drop this many layers at each end (official ``[2:-2]``)
    # Extra traces for classes LLaVA rejected: forced "Yes, it contains a {c}." responses.
    # They are only used by the QAP-oracle analysis and the calibrated-QAP variant.
    force_classes_with_p_yes: float = 0.05
    # Layer study: also store the aligned class response of every hidden-state index.
    per_layer: bool = False
    dtype: str = "float16"
    # Speed-up for large vocabularies (COCO: 80 classes). All class questions are first scored in
    # batches on the shared image prefix; only classes whose first-token P(yes) reaches
    # ``screen_p_yes`` (or that are forced) go through the exact sequential path below. On VOC
    # val the greedy answer contained "yes" exactly when P(yes) >= 0.5 (15,083/15,083 answers),
    # so screening at 0.3 cannot drop a "yes". 0 disables screening (every class sequential).
    screen_p_yes: float = 0.0
    screen_batch: int = 16
    # Record hidden states and visual attention during generation (as the released code does
    # with generate(output_hidden_states=True)) instead of a second teacher-forced pass.
    capture_in_generation: bool = False
    # Also score every class with EXTRA_QUERY_TEMPLATES (batched first-token pass only).
    extra_queries: bool = False


def square_resize(image: Image.Image) -> Image.Image:
    """Official ``resize2square_with_metadata``: stretch the short side to the long side."""
    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    return image.resize((side, side), Image.Resampling.LANCZOS)


def conversation_template(model_id: str) -> str:
    """Official ``infer_conv_mode``: Mistral backbones use ``mistral_instruct``, others ``llava_v1``."""
    return "mistral_instruct" if "mistral" in model_id.lower() else "llava_v1"


# Alternative yes/no wordings (the paper's Table 2 compares the first two with QUERY_TEMPLATE).
# Used only for first-token P(yes); the maps always come from the QUERY_TEMPLATE answer.
EXTRA_QUERY_TEMPLATES = (
    "Is a photo of a {0} ? Answer yes or no for this question.",
    "Does the image contain a {0} ? Answer yes or no for this question.",
    "Is there a {0} in the image? Answer yes or no.",
)


def build_prompt(
    class_name: str, template: str = "llava_v1", query_template: str = QUERY_TEMPLATE
) -> tuple[str, str]:
    """Return the text before and after ``<image>`` for the official conversation template."""
    query = query_template.format(class_name.replace(",", " or "))
    if template == "llava_v1":
        return f"{SYSTEM_PROMPT} USER: ", f"\n{query} ASSISTANT:"
    if template == "mistral_instruct":  # LLaMA-2 style with an empty system prompt
        return "[INST] ", f"\n{query} [/INST]"
    raise ValueError(f"unknown conversation template {template!r}")


def class_token_indices(tokens: list[str]) -> tuple[list[int], int | None]:
    """Indices (into ``tokens``) whose features form the class response.

    ``tokens`` are the stripped per-token decodes of the generated response without its first
    token, as in the official ``_decode_tokens``. Class tokens are the non-excluded tokens of the
    first sentence, plus later repetitions of those tokens. Returns the indices and the index of
    the first ``"."`` (or ``None``).
    """
    sentence_ends = [index for index, token in enumerate(tokens) if token == "."]
    first_end = sentence_ends[0] if sentence_ends else len(tokens)
    selected: list[int] = []
    class_tokens: list[str] = []
    for index, token in enumerate(tokens):
        if token.lower() in EXCLUDE_WORDS:
            continue
        if index < first_end:
            selected.append(index)
            class_tokens.append(token)
        elif token in class_tokens:
            selected.append(index)
    return selected, (sentence_ends[0] if sentence_ends else None)


@dataclass
class ClassTrace:
    class_name: str
    response: str
    p_yes: float
    forced: bool = False
    distance: np.ndarray | None = None  # [Nv] official alignment
    distance_aligned: np.ndarray | None = None  # [Nv] state at the token itself
    attention: np.ndarray | None = None  # [layers*heads, Nv] description-token mean
    attention_sum: np.ndarray | None = None  # [layers*heads] attention mass on image tokens
    layer_distance: np.ndarray | None = None  # [layers+1, Nv] aligned response per layer index
    p_yes_extra: list[float] | None = None  # P(yes) under EXTRA_QUERY_TEMPLATES, if traced
    # Aligned map (layers layer_start..layer_end) of every response text token, and the tokens.
    token_maps: np.ndarray | None = None  # [tokens, Nv]
    # Same as distance_aligned / layer_distance after subtracting, per layer, the mean image-token
    # state from both the image states and the class-token state (removes the shared direction
    # that makes all image tokens look alike, e.g. in Qwen2.5-VL).
    distance_centered: np.ndarray | None = None
    layer_distance_centered: np.ndarray | None = None
    tokens: list[str] | None = None


@dataclass
class ImageTrace:
    image_id: str
    image_size: tuple[int, int]  # (height, width)
    grid: tuple[int, int]
    clip_features: np.ndarray  # [Nv, C] CLIP patch features (vision layer -2)
    classes: dict[str, ClassTrace] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {"clip_features": self.clip_features.astype(np.float16)}
        meta: dict[str, Any] = {
            "image_id": self.image_id,
            "image_size": list(self.image_size),
            "grid": list(self.grid),
            "settings": self.settings,
            "classes": {},
        }
        for name, trace in self.classes.items():
            entry = {"response": trace.response, "p_yes": trace.p_yes, "forced": trace.forced}
            if trace.p_yes_extra is not None:
                entry["p_yes_extra"] = trace.p_yes_extra
            if trace.tokens is not None:
                entry["tokens"] = trace.tokens
            for key in ("distance", "distance_aligned", "attention", "attention_sum", "layer_distance", "token_maps",
                "distance_centered", "layer_distance_centered"):
                value = getattr(trace, key)
                entry[f"has_{key}"] = value is not None
                if value is not None:
                    dtype = np.float16 if key in ("attention", "token_maps") else np.float32
                    arrays[f"{name}::{key}"] = value.astype(dtype)
            meta["classes"][name] = entry
        np.savez(directory / f"{self.image_id}.npz", **arrays)
        (directory / f"{self.image_id}.json").write_text(json.dumps(meta, indent=1))

    @classmethod
    def load(cls, directory: Path, image_id: str) -> ImageTrace:
        meta = json.loads((directory / f"{image_id}.json").read_text())
        arrays = np.load(directory / f"{image_id}.npz")
        classes = {}
        for name, entry in meta["classes"].items():
            values = {
                key: arrays[f"{name}::{key}"].astype(np.float32) if entry.get(f"has_{key}") else None
                for key in ("distance", "distance_aligned", "attention", "attention_sum", "layer_distance", "token_maps",
                "distance_centered", "layer_distance_centered")
            }
            classes[name] = ClassTrace(
                name, entry["response"], entry["p_yes"], entry["forced"], **values,
                p_yes_extra=entry.get("p_yes_extra"), tokens=entry.get("tokens"),
            )
        return cls(
            image_id=meta["image_id"],
            image_size=tuple(meta["image_size"]),
            grid=tuple(meta["grid"]),
            clip_features=arrays["clip_features"].astype(np.float32),
            classes=classes,
            settings=meta["settings"],
        )


def is_complete(directory: Path, image_id: str) -> bool:
    return (directory / f"{image_id}.json").is_file() and (directory / f"{image_id}.npz").is_file()


class LlavaTracer:
    """Hugging Face LLaVA-1.5 / LLaVA-1.6 with a shared image-prefix KV cache and a greedy loop.

    LLaVA-1.6 runs as in the paper's setup: multi-resolution ("anyres") inference is disabled and
    only the base 336x336 view is encoded (576 tokens, no newline tokens), which is what the
    released code does with ``--image-aspect-ratio resize``.
    """

    def __init__(self, settings: TraceSettings, device: str = "cuda") -> None:
        from transformers import (
            AutoConfig,
            AutoTokenizer,
            CLIPImageProcessor,
            LlavaForConditionalGeneration,
            LlavaNextForConditionalGeneration,
        )

        self.settings = settings
        self.device = torch.device(device)
        dtype = getattr(torch, settings.dtype)
        self.template = conversation_template(settings.model_id)
        model_type = AutoConfig.from_pretrained(settings.model_id).model_type
        model_class = {"llava": LlavaForConditionalGeneration, "llava_next": LlavaNextForConditionalGeneration}[
            model_type
        ]
        # The slow SentencePiece tokenizer matches the original LLaVA code's tokenization.
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_id, use_fast=False)
        # Plain CLIP-336 preprocessing (resize + centre crop of the already square image) for both
        # generations; LLaVA-1.6's own processor would tile the image (anyres).
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        self.model = model_class.from_pretrained(
            settings.model_id, dtype=dtype, attn_implementation="eager"
        ).to(self.device)
        self.model.eval()
        inner = self.model.model
        self.vision_tower = inner.vision_tower
        self.projector = inner.multi_modal_projector
        self.language_model = inner.language_model
        self.lm_head = self.model.lm_head
        self.embed = self.language_model.get_input_embeddings()
        self.dtype = dtype
        self.yes_ids = sorted(
            {self.tokenizer.convert_tokens_to_ids(t) for t in ("▁Yes", "▁yes", "Yes", "yes")}
            - {self.tokenizer.unk_token_id}
        )
        self.eos_ids = {self.tokenizer.eos_token_id}

    # -- tokenization ------------------------------------------------------------------------
    def _text_ids(self, text: str, keep_bos: bool) -> list[int]:
        ids = self.tokenizer(text).input_ids
        return ids if keep_bos else ids[1:]  # official tokenizer_image_token drops chunk BOS

    def _query_ids(self, class_name: str, query_template: str = QUERY_TEMPLATE) -> list[int]:
        return self._text_ids(build_prompt(class_name, self.template, query_template)[1], keep_bos=False)

    def _prefix_forward(self, projected: torch.Tensor) -> tuple[Any, int]:
        """Run the system/user text and the image tokens once; returns (output, first image index)."""
        before, _ = build_prompt("x", self.template)  # the text before <image> is class-independent
        prefix_ids = torch.tensor([self._text_ids(before, keep_bos=True)], device=self.device)
        prefix = torch.cat([self.embed(prefix_ids), projected[None].to(self.dtype)], dim=1)
        return self._forward(prefix, None), prefix_ids.shape[1]

    def _positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Map 1-D text positions [B, L] to the model's position ids (identity for LLaVA)."""
        return positions

    def _token_text(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id]).strip()

    # -- model pieces ------------------------------------------------------------------------
    @torch.inference_mode()
    def encode_image(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = self.image_processor(square_resize(image), return_tensors="pt")["pixel_values"]
        pixels = pixels.to(self.device, self.dtype)
        out = self.vision_tower(pixels, output_hidden_states=True)
        clip = out.hidden_states[-2][:, 1:]  # LLaVA-1.5: layer -2, drop CLS
        return clip[0], self.projector(clip)[0]

    def _forward(self, embeds: torch.Tensor, cache: Any, attentions: bool = False) -> Any:
        return self.language_model(
            inputs_embeds=embeds,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
            output_attentions=attentions,
            return_dict=True,
        )

    @torch.inference_mode()
    def trace_image(
        self,
        image: Image.Image,
        image_id: str,
        class_names: list[str],
        force_classes: set[str] | frozenset[str] = frozenset(),
    ) -> ImageTrace:
        import copy

        s = self.settings
        clip, projected = self.encode_image(image)
        visual_count = projected.shape[0]
        grid = (int(visual_count**0.5), int(visual_count**0.5))
        out, visual_start = self._prefix_forward(projected)
        prefix_cache = out.past_key_values
        # [layers+1, Nv, D] hidden states of the image tokens (index 0 = embeddings).
        visual_states = torch.stack(out.hidden_states)[:, 0, visual_start:].float()
        selected_visual = visual_states[s.layer_start : s.layer_end]
        all_visual = visual_states if s.per_layer else None

        result = ImageTrace(
            image_id=image_id,
            image_size=(image.height, image.width),
            grid=grid,
            clip_features=clip.float().cpu().numpy(),
            settings=asdict(s),
        )
        queries = {name: self._query_ids(name) for name in class_names}
        screened = self._screen(prefix_cache, queries) if s.screen_p_yes > 0 else {}
        for class_name in class_names:
            query_ids = queries[class_name]
            screen = screened.get(class_name)
            threshold = min(s.screen_p_yes, s.force_classes_with_p_yes)
            if screen is not None and screen < threshold and class_name not in force_classes:
                result.classes[class_name] = ClassTrace(class_name, "No.", screen, False)
                continue
            q0 = len(query_ids)
            kv_visual = slice(visual_start, visual_start + visual_count)
            captured = {} if s.capture_in_generation else None
            cache = copy.deepcopy(prefix_cache)
            embeds = self.embed(torch.tensor([query_ids], device=self.device))
            out = self._forward(embeds, cache)
            last = self.lm_head(out.last_hidden_state[:, -1]).float()
            p_yes = float(torch.softmax(last, -1)[0, self.yes_ids].sum())
            generated, cache = self._greedy(last, out.past_key_values, [], captured, q0, kv_visual)
            response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            present = "yes" in response.lower()
            forced = False
            if not present and (class_name in force_classes or p_yes >= s.force_classes_with_p_yes):
                forced = True
                # SentencePiece's dummy prefix yields "▁Yes", the token LLaVA itself emits.
                forced_prefix = self._text_ids(f"Yes, it contains a {class_name}.", keep_bos=False)
                captured = {} if s.capture_in_generation else None
                cache = copy.deepcopy(prefix_cache)
                out = self._forward(
                    self.embed(torch.tensor([query_ids + forced_prefix], device=self.device)),
                    cache,
                    attentions=captured is not None,
                )
                if captured is not None:
                    for j in range(len(forced_prefix)):
                        captured[q0 + j] = self._capture(out, q0 + j, kv_visual)
                last = self.lm_head(out.last_hidden_state[:, -1]).float()
                generated, _ = self._greedy(last, out.past_key_values, forced_prefix, captured, q0, kv_visual)
                response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            trace = ClassTrace(class_name, response, p_yes, forced)
            if present or forced:
                self._fill_maps(
                    trace, prefix_cache, query_ids, generated, selected_visual, visual_start, visual_count,
                    all_visual, captured,
                )
            result.classes[class_name] = trace
            del cache
        if s.extra_queries:
            extra = {
                (name, k): self._query_ids(name, template)
                for name in class_names
                for k, template in enumerate(EXTRA_QUERY_TEMPLATES)
            }
            scores = self._screen(prefix_cache, extra)
            for name in class_names:
                result.classes[name].p_yes_extra = [scores[(name, k)] for k in range(len(EXTRA_QUERY_TEMPLATES))]
        return result

    @torch.inference_mode()
    def extra_scores(self, image: Image.Image, class_names: list[str]) -> dict[str, list[float]]:
        """P(yes) under EXTRA_QUERY_TEMPLATES only: image prefix + one batched first-token pass."""
        _, projected = self.encode_image(image)
        out, _ = self._prefix_forward(projected)
        extra = {
            (name, k): self._query_ids(name, template)
            for name in class_names
            for k, template in enumerate(EXTRA_QUERY_TEMPLATES)
        }
        scores = self._screen(out.past_key_values, extra)
        return {name: [scores[(name, k)] for k in range(len(EXTRA_QUERY_TEMPLATES))] for name in class_names}

    def _screen(self, prefix_cache: Any, queries: dict[str, list[int]]) -> dict[str, float]:
        """First-token P(yes) for every class, in left-padded batches sharing the image prefix."""
        import copy

        prefix_length = prefix_cache.get_seq_length()
        names = list(queries)
        result: dict[str, float] = {}
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        for start in range(0, len(names), self.settings.screen_batch):
            chunk = names[start : start + self.settings.screen_batch]
            longest = max(len(queries[name]) for name in chunk)
            ids = torch.full((len(chunk), longest), pad_id, dtype=torch.long, device=self.device)
            mask = torch.zeros((len(chunk), prefix_length + longest), dtype=torch.long, device=self.device)
            positions = torch.zeros((len(chunk), longest), dtype=torch.long, device=self.device)
            mask[:, :prefix_length] = 1
            for row, name in enumerate(chunk):
                tokens = queries[name]
                offset = longest - len(tokens)
                ids[row, offset:] = torch.tensor(tokens, device=self.device)
                mask[row, prefix_length + offset :] = 1
                positions[row, offset:] = torch.arange(prefix_length, prefix_length + len(tokens), device=self.device)
            cache = copy.deepcopy(prefix_cache)
            cache.batch_repeat_interleave(len(chunk))
            out = self.language_model(
                inputs_embeds=self.embed(ids),
                attention_mask=mask,
                position_ids=self._positions(positions),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            logits = self.lm_head(out.last_hidden_state[:, -1]).float()
            p_yes = torch.softmax(logits, -1)[:, self.yes_ids].sum(-1)
            result.update({name: float(p) for name, p in zip(chunk, p_yes)})
            del cache, out
        return result

    @staticmethod
    def _capture(out: Any, position: int, kv_visual: slice) -> tuple[torch.Tensor, torch.Tensor]:
        """Hidden states [L+1, D] and visual attention [L, H, Nv] at one suffix position."""
        states = torch.stack([h[0, position] for h in out.hidden_states])
        attention = torch.stack([a[0][:, position, kv_visual] for a in out.attentions])
        return states, attention

    def _greedy(
        self,
        last_logits: torch.Tensor,
        cache: Any,
        prefix: list[int],
        captured: dict | None = None,
        q0: int = 0,
        kv_visual: slice | None = None,
    ) -> tuple[list[int], Any]:
        generated = list(prefix)
        logits = last_logits
        for _ in range(self.settings.max_new_tokens - len(prefix)):
            token = int(logits.argmax(-1))
            generated.append(token)
            if token in self.eos_ids:
                break
            out = self._forward(
                self.embed(torch.tensor([[token]], device=self.device)), cache, attentions=captured is not None
            )
            if captured is not None:  # this step's input is g_j, at suffix position q0 + j
                captured[q0 + len(generated) - 1] = self._capture(out, -1, kv_visual)
            cache = out.past_key_values
            logits = self.lm_head(out.last_hidden_state[:, -1]).float()
        return generated, cache

    def _fill_maps(
        self,
        trace: ClassTrace,
        prefix_cache: Any,
        query_ids: list[int],
        generated: list[int],
        selected_visual: torch.Tensor,
        visual_start: int,
        visual_count: int,
        all_visual: torch.Tensor | None = None,
        captured: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> None:
        """Compute the class maps from the states/attentions of query + response.

        Index conventions follow the official generate()-based code: ``generated`` is
        g_0..g_n (g_n = eos when generated); per-token texts drop g_0 and the last token, so
        text index i is g_{i+1}. Suffix position q0 + j is the position whose input is g_j.

        With ``captured`` (suffix position -> (hidden states [L+1, D], visual attention
        [L, H, Nv])) recorded during generation, no second pass is needed; otherwise the query and
        response are teacher-forced once.
        """
        import copy

        q0 = len(query_ids)  # suffix position of g_0
        out = cache = None
        if captured is None:
            cache = copy.deepcopy(prefix_cache)
            sequence = query_ids + generated
            out = self._forward(self.embed(torch.tensor([sequence], device=self.device)), cache, attentions=True)
            states = torch.stack(out.hidden_states)[:, 0]  # [layers+1, suffix, D]
            kv_visual = slice(visual_start, visual_start + visual_count)
            attentions = out.attentions

            def state_at(position: int) -> torch.Tensor:
                return states[:, position]

            def attention_at(position: int) -> torch.Tensor:
                return torch.stack([a[0][:, position, kv_visual] for a in attentions])
        else:

            def state_at(position: int) -> torch.Tensor:
                return captured[position][0]

            def attention_at(position: int) -> torch.Tensor:
                return captured[position][1]

        self._maps(trace, generated, q0, state_at, attention_at, selected_visual, all_visual)
        del out, cache

    def _maps(self, trace, generated, q0, state_at, attention_at, selected_visual, all_visual) -> None:
        s = self.settings
        tokens = [self._token_text(token) for token in generated][1:-1]
        indices, first_end = class_token_indices(tokens)
        # Official loop bound: hidden_states[1:] covers inputs g_0..g_{n-1}.
        indices = [i for i in indices if i < len(generated) - 1]
        if not indices:
            return
        selected = slice(s.layer_start, s.layer_end)

        visual_mean = selected_visual.mean(1, keepdim=True)  # [L, 1, D]

        def distance_for(positions: list[int], centered: bool = False) -> np.ndarray:
            visual = selected_visual - visual_mean if centered else selected_visual
            maps = []
            for position in positions:
                token_state = state_at(position)[selected].float()[:, None, :]  # [L, 1, D]
                if centered:
                    token_state = token_state - visual_mean
                cos = F.cosine_similarity(visual, token_state, dim=-1, eps=1e-8)
                maps.append(cos.mean(0))
            return torch.stack(maps).mean(0).cpu().numpy()

        trace.distance = distance_for([q0 + i for i in indices])
        trace.distance_aligned = distance_for([q0 + i + 1 for i in indices])
        trace.distance_centered = distance_for([q0 + i + 1 for i in indices], centered=True)
        usable = range(min(len(tokens), len(generated) - 1))
        trace.tokens = [tokens[i] for i in usable]
        trace.token_maps = np.stack([distance_for([q0 + i + 1]) for i in usable]).astype(np.float16)
        if all_visual is not None:
            per_token = [
                F.cosine_similarity(all_visual, state_at(q0 + i + 1).float()[:, None, :], dim=-1, eps=1e-8)
                for i in indices
            ]
            trace.layer_distance = torch.stack(per_token).mean(0).cpu().numpy()  # [layers+1, Nv]
            all_mean = all_visual.mean(1, keepdim=True)
            per_token = [
                F.cosine_similarity(
                    all_visual - all_mean, state_at(q0 + i + 1).float()[:, None, :] - all_mean, dim=-1, eps=1e-8
                )
                for i in indices
            ]
            trace.layer_distance_centered = torch.stack(per_token).mean(0).cpu().numpy()

        if first_end is not None and len([i for i, t in enumerate(tokens) if t == "."]) > 1:
            rows = list(range(first_end + 1, min(len(tokens), len(generated) - 1)))
            if rows:
                trim = s.attention_layer_trim
                # [rows, layers, heads, Nv] -> keep layers trim..L-trim -> mean over rows
                stacked = torch.stack([attention_at(q0 + i) for i in rows]).float()
                stacked = stacked[:, trim : stacked.shape[1] - trim].mean(0)
                maps = stacked.reshape(-1, stacked.shape[-1])
                trace.attention = maps.cpu().numpy()
                trace.attention_sum = maps.sum(-1).cpu().numpy()
