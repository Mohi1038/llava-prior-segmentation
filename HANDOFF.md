# Handoff: LLaVA Prior Segmentation

## User goal

Reproduce the CVPR 2026 paper **The Power of Prior: Training-Free Open-Vocabulary Semantic Segmentation with LLaVA** for a computer vision course. The user wants a high-quality implementation, an explanation of the method, and a runnable experiment on `image.jpg`.

## Paper method

The paper proposes FSeg-LLaVA, a training-free open-vocabulary semantic segmentation pipeline:

1. **Question-answer pipeline (QAP):** for every candidate class, ask LLaVA a fixed-format question: whether the image contains the class. A positive answer must contain a yes sentence and a description sentence.
2. **Text-visual response (TVR):** extract hidden states for image tokens and the class token from intermediate LLM layers. Compute averaged cosine distance over selected layers. The paper's strongest default for Vicuna-7B is layers 7--13. Threshold the distance map with `min(alpha * max(response), beta)`, using `alpha=0.75`, `beta=0.19` for the default setting. Use cross-attention from the description to image tokens, select low-spatial-entropy responses, average top-K, threshold by their mean, and multiply this attention mask with the distance response.
3. **Visual generation module (VGM):** stack class response maps, assign each vision token to the winning class, compute positive and negative prototypes from LLaVA vision features, compare every token with both prototypes, and intersect the prototype mask with the class guidance mask.
4. **SAM prompting:** upsample the token mask to image resolution, morphologically clean it, extract connected-component centroids as positive points, compute a tight bounding box, and use both point and box prompts with frozen SAM to produce the final mask.

The paper reports 68.0 mIoU on VOC21 using LLaVA-1.5 Vicuna-7B and ViT-H SAM. It does not require explicit background class names in the QAP stage.

## Repository state

Workspace: `/Users/mohi1038/llava-prior-segmentation`

The repository has no commits yet. Current untracked project files are intentional. Do not reset or discard them.

Important local files:

- `pyproject.toml`: package metadata, dependencies, `llava-prior` CLI entry point.
- `README.md`: paper explanation, reproduction plan, installation, and image inference command.
- `configs/voc21.json`: paper-default inference values.
- `src/llava_prior/config.py`: validated `PriorSegmentationConfig` dataclass.
- `src/llava_prior/qap.py`: fixed query rendering and permissive response parsing.
- `src/llava_prior/response.py`: TVR cosine distance, dynamic threshold, attention selection, response filtering.
- `src/llava_prior/prototypes.py`: semantic guidance, foreground/background prototypes, prototype/refined masks.
- `src/llava_prior/prompts.py`: connected components, morphology, centroids, boxes.
- `src/llava_prior/pipeline.py`: QAP -> TVR -> VGM -> SAM orchestration and `TraceBatch`/backend protocols.
- `src/llava_prior/models.py`: concrete Hugging Face LLaVA and Segment Anything adapters.
- `src/llava_prior/cli.py`: single-image CLI; writes one `.npy` mask per detected class and `predictions.json`.
- `tests/test_math.py`: TVR/VGM math tests.
- `tests/test_qap_and_prompts.py`: QAP and geometry tests.
- `tests/test_pipeline.py`: mocked end-to-end pipeline and SAM prompt test.

## Current validation

Last successful validation:

```text
6 passed
All checks passed!
llava-prior-segmentation 0.1.0
```

Commands:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests
.venv/bin/python -m llava_prior.cli --version
.venv/bin/llava-prior --version
```

## Current local assets

- `image.jpg` exists and is approximately 1.3 MB.
- `checkpoints/sam_vit_h_4b8939.pth` exists and is approximately 1.6 GB.
- The machine has only approximately 3.3 GB free on the current filesystem. This is not enough for a complete LLaVA-1.5-7B download plus caches and runtime overhead. Do not start another large download until disk space is freed or Hugging Face cache is moved to another volume.
- The earlier SAM URL attempt returned a tiny AccessDenied/XML file, but the current `checkpoints/sam_vit_h_4b8939.pth` is 1.6 GB and appears to be the real checkpoint. Verify its size/hash before loading if needed.

## Environment

A `.venv` exists and has PyTorch, Transformers, NumPy, Pillow, OpenCV, pytest, and Ruff installed. The current environment reported approximately:

- Python 3.14
- PyTorch 2.14.1
- Transformers 5.18.0
- OpenCV 5.0.0

SAM is an optional dependency and may still need installation:

```bash
.venv/bin/pip install -e '.[sam]'
```

The default LLaVA model ID is `llava-hf/llava-1.5-7b-hf`.

## Intended run command

After confirming disk space and installing the SAM extra:

```bash
.venv/bin/llava-prior image.jpg \
  --classes person dog bicycle \
  --llava-model llava-hf/llava-1.5-7b-hf \
  --sam-checkpoint checkpoints/sam_vit_h_4b8939.pth \
  --device mps \
  --output-dir outputs/image
```

On Apple Silicon, MPS is preferred. CPU will be very slow. The 7B checkpoint needs substantial memory and disk cache space.

## Known implementation risks to investigate next

1. **Disk/model download:** resolve storage first. Do not blindly retry a 7B download with only 3.3 GB free. Set `HF_HOME` to a volume with enough capacity if available.
2. **Transformers/LLaVA compatibility:** `models.py` assumes `LlavaForConditionalGeneration`, `get_image_features`, `output_hidden_states`, and `output_attentions` APIs. Test with the installed Transformers version after the model is available.
3. **Token alignment:** `_trace()` performs a teacher-forced forward pass on `query + response`, locates class and response token subsequences, and accounts for the expanded image-token span. This is the most likely model-specific failure point.
4. **Vision feature shape:** different LLaVA versions may return image features with or without a CLS token or with a different projector dimension. Inspect `image_features.shape` and ensure the spatial token count is a square/grid compatible with `_grid_shape()`.
5. **Attention shape:** some implementations may return attentions from the language model with different layer/sequence dimensions. Confirm the response-to-image slices are non-empty and have shape `[description_tokens, visual_tokens]`.
6. **SAM empty prompts:** if VGM produces no valid component, `SamPredictorBackend` currently calls SAM with both prompts absent. Add a fallback or skip class cleanly if this fails.
7. **Evaluation is not yet implemented:** the current project runs single-image inference and has unit tests, but it does not yet download/load VOC21 or compute mIoU. The next research-grade feature should be `metrics.py` plus a VOC evaluation command.
8. **Background/overlap policy:** `run_pipeline()` returns independent per-class masks. A benchmark evaluator should define overlap resolution and ignore-index handling explicitly.

## Recommended continuation sequence

1. Check disk space and SAM checkpoint integrity.
2. Install the SAM extra and import `segment_anything`.
3. If storage permits, download/cache LLaVA to a separate volume.
4. Run a one-class smoke test such as `--classes person` on `image.jpg`.
5. Add temporary shape logging around `_trace()` if model inference fails.
6. Fix adapter compatibility issues without changing the tested TVR/VGM math.
7. Add VOC21 dataset loading and mIoU metrics.
8. Add qualitative overlay PNGs, runtime/memory reporting, and ablation configs.

## Do not claim yet

Do not claim the paper's 68.0 VOC21 result has been reproduced. The current verified result is only that the model-independent math, prompt geometry, mocked pipeline, imports, lint, and CLI entry points work. Real checkpoint inference and benchmark evaluation remain to be run.
