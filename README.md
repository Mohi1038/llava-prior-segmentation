# LLaVA Prior Segmentation

Reproduction-oriented implementation of **The Power of Prior: Training-Free Open-Vocabulary Semantic Segmentation with LLaVA** (CVPR 2026).

The project investigates whether spatial information already present in a frozen vision-language model can be converted into open-vocabulary masks without task-specific training.

## What the paper does

For each candidate class, the method has three stages:

1. **Question-answer pipeline (QAP):** ask LLaVA whether the image contains the class. A positive answer also produces a short description.
2. **Text-visual response (TVR):** extract hidden states for image tokens and the class token from intermediate LLM layers. Average their cosine distances, threshold low-confidence pixels, and intersect that map with a top-$K$ cross-attention mask from the generated description.
3. **Visual generation module (VGM):** use the response maps to form foreground/background vision prototypes. Compare vision-token features to those prototypes, derive positive points and a bounding box, then prompt SAM for the final mask.

There is no segmentation training loop. LLaVA and SAM stay frozen; the computation is inference-time feature extraction, filtering, and prompting. The paper reports its strongest settings around layers 7--13, `alpha=0.75`, and ViT-H SAM, with 68.0 mIoU on VOC21 for LLaVA-1.5 Vicuna-7B.

## Reproduction plan

1. **Environment and model adapters:** pin a reproducible Python environment, support Apple Silicon MPS/CUDA/CPU, and isolate Hugging Face LLaVA and SAM loading behind adapters.
2. **QAP:** implement strict prompt construction, robust answer parsing, and class-level caching so absent classes never enter later stages.
3. **TVR:** capture image-token hidden states, class-token hidden states, and text-to-image attentions from selected layers; implement the paper equations with shape checks and diagnostics.
4. **VGM and SAM:** build prototypes, denoise connected components, generate centroids and boxes, run SAM, and merge per-class masks with an explicit background policy.
5. **Evaluation:** add VOC21 first because it is small and diagnostic, then COCO-Object, Context60, COCO-Stuff, and ADE20K. Report mIoU, per-class IoU, runtime, memory, and failure cases.
6. **Ablations:** reproduce prompt wording, layer ranges, point versus box prompts, response-map variants, prototype-mask variants, and LLaVA model variants.
7. **Course deliverables:** produce qualitative overlays, an experiment table, an error taxonomy, and a comparison against a CLIP baseline. Every claim should be tied to a config and saved artifact.

## Status

The implementation includes the model-independent TVR/VGM computations, real LLaVA/SAM adapters, deterministic QAP deduplication, indexed semantic-mask merging, and mIoU evaluation utilities. Model loading and benchmark data remain separate so the math can be verified without downloading LLaVA or SAM.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

Install SAM support before running the real model path:

```bash
.venv/bin/pip install -e '.[sam]'
```

Run one image after downloading a LLaVA checkpoint and a SAM checkpoint:

```bash
.venv/bin/llava-prior path/to/image.jpg \
	--classes person dog bicycle \
	--sam-checkpoint path/to/sam_vit_h_4b8939.pth \
	--output-dir outputs/example
```

The output directory contains one binary NumPy mask per detected class,
`semantic_mask.npy` with background `0` and classes in candidate-list order, and
`predictions.json`. Evaluate an indexed prediction against a PNG or NumPy target:

```bash
.venv/bin/llava-prior-eval outputs/example path/to/target.png \
	--classes person dog bicycle
```

Target labels must use `0` for background, `1..N` for the listed classes, and
`255` for ignored pixels. The evaluator reports mean IoU and per-class IoU; it
does not download VOC21 or claim the paper's reported score automatically.

Large checkpoints and datasets should be stored outside Git. The paper-default
configuration is recorded in [configs/voc21.json](configs/voc21.json).