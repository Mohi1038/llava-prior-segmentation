# LLaVA Prior Segmentation

Reproduction-oriented implementation of **The Power of Prior: Training-Free Open-Vocabulary Semantic Segmentation with LLaVA** (CVPR 2026).

The project investigates whether spatial information already present in a frozen vision-language model can be converted into open-vocabulary masks without task-specific training.

## What the paper does

For each candidate class, the method has three stages:

1. **Question-answer pipeline (QAP):** ask LLaVA whether the image contains the class. A positive answer also produces a short description.
2. **Text-visual response (TVR):** extract hidden states for image tokens and the class token from intermediate LLM layers. Average their cosine distances, threshold low-confidence pixels, and intersect that map with a top-$K$ cross-attention mask from the generated description.
3. **Visual generation module (VGM):** use the response maps to form foreground/background vision prototypes. Compare vision-token features to those prototypes, derive positive points, a bounding box, and a coarse mask prior. Pass all of them to SAM, then select the multimask candidate with the best overlap with the class prior.

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

The output directory contains one binary NumPy mask per detected class and its
coarse VGM prior (saved as `<class>_prior.npy`), so the prior can be inspected
separately from SAM's result. It also contains `semantic_mask.npy` with
background `0` and classes in candidate-list order, `predictions.json`,
`run_metadata.json` with the model/config/device/software settings, and
`visualization.png`. If GPU memory is limited, install the
quantization extra and keep LLaVA on CUDA while running SAM on CPU:

```bash
.\.venv\Scripts\python.exe -m pip install -e ".[quant]"
.\.venv\Scripts\llava-prior.exe image.jpg `
	--classes dog `
	--sam-checkpoint checkpoints\sam_vit_h_4b8939.pth `
	--device cuda --sam-device cpu --quantize int8 `
	--output-dir outputs\example
```

The high-resolution visualization uses distinct class colors, bright boundaries,
and per-class pixel coverage. Evaluate an indexed prediction against a PNG or NumPy target:

```bash
.venv/bin/llava-prior-eval outputs/example path/to/target.png \
	--classes person dog bicycle
```

Target labels must use `0` for background, `1..N` for the listed classes, and
`255` for ignored pixels. The evaluator reports mean IoU and per-class IoU; it
does not download VOC21 or claim the paper's reported score automatically.

Run a directory of images with one shared model load. By default, only images
directly inside the directory are included; use `--recursive` for subdirectories.
Each image gets its own output folder, and `batch_summary.json` records successes
and failures. Its `completed_count` includes new successes plus already-complete
images skipped by `--resume`:

```bash
.\.venv\Scripts\llava-prior.exe path\to\images `
	--classes person dog bicycle `
	--sam-checkpoint checkpoints\sam_vit_h_4b8939.pth `
	--device cuda --sam-device cpu --quantize int8 `
	--recursive --output-dir outputs\voc-sample
```

Use `--resume` to skip images whose outputs and run settings match the current
command. Failed images are listed in the batch summary; the command exits with
a nonzero status if any image fails.

For the curated VOC subset, per-image class lists prevent the model from querying
every class on every image while still using one shared model load:

```powershell
$vocClasses = @(
	"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
	"chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
	"pottedplant", "sheep", "sofa", "train", "tvmonitor"
)
.\.venv\Scripts\llava-prior.exe `
	.\datasets\VOC2012\curated\images `
	--classes $vocClasses `
	--class-map .\datasets\VOC2012\curated\class_map.json `
	--sam-checkpoint .\checkpoints\sam_vit_h_4b8939.pth `
	--device cuda --sam-device cpu --quantize int8 `
	--output-dir .\outputs\voc-curated
```

Evaluate batch outputs on the curated diagnostic subset of the official VOC
validation split. This 40-image subset is class-balanced but is not a replacement
for evaluation on the full validation split. VOC's native foreground class IDs
are remapped to the candidate class order; background stays `0`, void pixels
(`255`) are ignored, and VOC classes not requested for inference are excluded
from the score rather than treated as background:

```powershell
$vocClasses = @(
	"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
	"chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
	"pottedplant", "sheep", "sofa", "train", "tvmonitor"
)
.\.venv\Scripts\llava-prior-eval-voc.exe `
	.\outputs\voc-curated `
	.\datasets\VOC2012\VOCdevkit\VOC2012 `
	--classes $vocClasses `
	--split-file .\datasets\VOC2012\curated\image_ids.txt `
	--output .\outputs\voc-curated\voc_metrics.json
```

### Label-independent VOC experiment

The curated subset above uses annotations both to select examples and to limit
each image's candidate classes. Treat its score as a diagnostic, not as an
unbiased open-vocabulary benchmark. For a label-independent experiment, select
a deterministic random sample from the official validation IDs and query all
20 VOC classes on every image. The sampler reads the validation ID list and
images only; it does not read segmentation masks or create a per-image class
map. The default seed is `2026`; use a fresh, empty output directory:

```powershell
.\.venv\Scripts\llava-prior-prepare-voc.exe `
	.\datasets\VOC2012\VOCdevkit\VOC2012 `
	--selection random `
	--image-count 100 `
	--seed 2026 `
	--output-dir .\datasets\VOC2012\random-100
```

Run all 20 candidate classes on this sample. Do not pass `--class-map`; keep
the generated `manifest.json` and `image_ids.txt` with the experiment:

```powershell
$vocClasses = @(
	"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
	"chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
	"pottedplant", "sheep", "sofa", "train", "tvmonitor"
)
.\.venv\Scripts\llava-prior.exe `
	.\datasets\VOC2012\random-100\images `
	--classes $vocClasses `
	--sam-checkpoint .\checkpoints\sam_vit_h_4b8939.pth `
	--device cuda --sam-device cpu --quantize int8 `
	--resume `
	--output-dir .\outputs\voc-random-100
```

Evaluate exactly the sampled IDs and verify that every image's recorded
inference settings include all 20 candidate classes:

```powershell
.\.venv\Scripts\llava-prior-eval-voc.exe `
	.\outputs\voc-random-100 `
	.\datasets\VOC2012\VOCdevkit\VOC2012 `
	--classes $vocClasses `
	--split-file .\datasets\VOC2012\random-100\image_ids.txt `
	--require-all-candidates `
	--output .\outputs\voc-random-100\voc_metrics.json
```

This is a random 100-image validation sample, not the full VOC validation
benchmark. To cover the full 1,449-image split with the same fixed all-class
policy, set `--image-count 1449`, use a separate output directory, and pass its
`image_ids.txt` to evaluation. Full-split inference can be substantially more
expensive because each image is queried with every candidate class.

To evaluate only a selected subset, pass a text file containing one VOC image ID
per line with `--split-file`. Predictions must exist under the output folder as
`<image-id>\semantic_mask.npy`, together with `<image-id>\predictions.json`.
The reported dataset mIoU accumulates each class's intersection and union across
the whole split before averaging class IoUs.

Large checkpoints and datasets should be stored outside Git. The paper-default
configuration is recorded in [configs/voc21.json](configs/voc21.json).
The official VOC 2012 train/validation archive can be stored locally under
`datasets/VOC2012`; see [datasets/README.md](datasets/README.md) for its source,
contents, and label details. Dataset files are excluded from Git.