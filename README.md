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

## Results (PASCAL VOC 2012, measured 2026-10-26/27)

All numbers are standard **VOC21 mIoU** (21 classes including background, void ignored, IoU
accumulated over the split as in mmseg) with LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`, fp16) and
SAM ViT-H on one A30 GPU. Design choices were made on a 200-image **tuning split** drawn from VOC
*train* (`splits/voc_train_dev200.txt`, seed 0); the 1,449-image **val** split was used only for
the final runs below.

| Setting | Tuning (200 train) | **VOC val (1,449)** | Paper (val) |
|---|---|---|---|
| **Reproduction (`base`)** | 68.49 | **65.85** | 68.0 |
| QAP oracle (ground-truth class list; analysis only) | 74.41 | 72.16 | – |
| No prototype step (Eq. 10-12) | 68.74 | 65.78 | 67.3 |
| No attention mask (D_f = D) | 66.34 | 65.49 | 67.3 |
| Paper-text combination D_f = D · M_A (code uses a union) | 63.16 | 61.62 | – |
| SAM points only | 62.00 | 57.70 | 61.4 |
| SAM box only | 57.23 | 54.12 | 57.8 |
| Class-token state taken one position early (`shifted_token`) | 47.35 | 48.57 | – |
| LLaVA maps without SAM (`base_coarse`) | – | 18.72 | – |

The reproduction is 2.2 points below the paper. Its ablations move in the paper's direction
(box + points > points > box; the attention mask helps; the prototype step matters little).
Possible causes of the gap, none verified: the HF-converted checkpoint and newer transformers,
fp16 numerics, and the authors' VOC class-name file, which is not public (we query the VOC
names; natural names such as "potted plant" scored 1.4 points lower on the tuning split).

**Recognition (QAP) on val:** precision 0.822, recall 0.949 (442 false "yes", 110 false "no"
over 1,449 × 20 queries). The oracle row shows that fixing recognition alone would add 6.3 points.
Worst classes: chair (P 0.52), sofa, dining table, bottle.

**Per-class (val):** strongest bus 91.7, motorbike 86.2, background 86.5; weakest **person 18.7**,
dining table 34.6, sofa 35.9, chair 36.8. Person fails mainly on close-ups that fill the frame:
LLaVA's response then covers only a face or a patch of skin, so 78% of person pixels end up as
background (`reports/figures/qualitative_failures.png`). The person response is weak in general
(mean peak cosine 0.24 vs 0.29-0.34 for other classes).

**Attempted improvements (all tuned on the train split; none adopted):**

| Idea | Tuning mIoU (base 68.49) | Outcome |
|---|---|---|
| Evidence filter: drop unsure "yes" classes whose map wins ≤ 2% of the grid | 68.68 | +0.19, within noise |
| Per-class background threshold (helps weak classes such as person) | 68.37 | person +13.8 IoU, other classes lose as much |
| Relative (max-normalised) background | 65.3 or lower | worse |
| One SAM prompt per connected region (multi-instance classes) | 27-52 | much worse: the LLaVA region is fragmented within one object |
| Query alias "person, people" | 67.86 | person IoU 24.4 → 12.0 |
| Calibrated QAP threshold on P(yes) | ≤ 66.46 | worse than LLaVA's own answer |
| α 0.65/0.85, β 0.16/0.22 | 64.3-67.5 | paper values are best |
| Combination: overlap → later class, no prototype, no negative points | 69.72 (+1.23, 95% CI −0.5 to +3.5) | **val 65.69 vs 65.85** (−0.15, 95% CI −0.83 to +0.55): did not transfer |

The combination was the only candidate above noise on the tuning split. It was fixed before val
and run on val once, where it scored 0.16 below the reproduction. The tuning-split gain was
sampling noise from 200 images. We therefore report the plain reproduction as the final result.

### Layer study (where LLaVA's spatial information sits)

Per-layer similarity maps were traced (`--per-layer`), then scored per single layer without SAM and
for the paper's Table-3 ranges, as maps alone and with SAM (`slurm/layer_study.sh`; jobs 45099
tuning, 45100 val). Self-check: per-layer maps averaged over 7–13 equal the reproduction's maps
(max diff 6e-8) and reproduce its scores. VOC val, mIoU:

| Layers | map only | with SAM | paper (with SAM) |
|---|---|---|---|
| 5–9 | 16.3 | 64.0 | 63.7 |
| 7–9 | 17.5 | 64.4 | 65.8 |
| 7–13 | 18.7 | 65.8 | 68.0 |
| 7–17 | 20.1 | 65.9 | 66.8 |
| 10–13 | 20.0 | 65.9 | 67.6 |
| 15–22 | 31.8 | 45.5 | 49.1 |
| 22–30 | 23.8 | 16.1 | 15.2 |

With SAM the ranges follow the paper's trend (7–17 all within ~1.5 points; deep ranges collapse).
Map-only quality peaks deeper (best single layer 22: 37.5). Middle-layer maps are small but
precise (79% of marked pixels correct, 5% of pixels marked), so they make good SAM seeds;
15–22 seeds give oversized SAM masks (43% of pixels foreground vs 27% true, precision 0.50).

Write-up: `reports/report.pdf` (6 pages; built from `reports/report.tex`; tables/plots regenerated from the
saved metrics by `python reports/make_assets.py`, then `tectonic -X compile report.tex`).

Figures: `reports/figures/qualitative_success.png`, `reports/figures/qualitative_failures.png`.
Per-run metrics: `runs/pred/<split>/<config>/metrics.json` (+ `confusion.npy`).

### Reproducing the cluster runs

```bash
module load python/3.11 && source .venv/bin/activate          # env lives in the project
sbatch -J trace-val slurm/run.slurm "llava-prior-voc trace --split val --out runs/traces/val_vocnames"   # ~2.8 h
sbatch -J segment-val slurm/run.slurm "llava-prior-voc segment --split val --traces runs/traces/val_vocnames \
    --configs configs/voc_val_core.json --out runs/pred/val_vocnames --save-coarse && \
    llava-prior-voc eval --split val --pred runs/pred/val_vocnames/*/"                                 # ~75 min for 8 configs
llava-prior-report table --reference base --per-class --pred runs/pred/val_vocnames/*
llava-prior-report qap --split val --traces runs/traces/val_vocnames
```

`trace` runs LLaVA once per image and caches everything (`<id>.npz` + `<id>.json`); every
setting in `activation.py` (`SegmentSettings`) is then a SAM-only rerun. Implementation notes:
`src/llava_prior/trace.py` (LLaVA stage, follows the released code: square resize, `llava_v1`
template, hidden-state indices 7..13, CLIP layer -2 features, per-head attention with
connected-region entropy selection), `activation.py` (maps, background, prototypes, SAM), `voc.py`
(drivers + VOC21 metric), `report.py` (tables, QAP metrics, figures).

The older single-image CLI below (`llava-prior`, `pipeline.py`, `models.py`) is the earlier
laptop implementation of the paper text. It differs from the released code (no background map,
projected instead of CLIP features, different prompt and SAM prompting) and was never run to
completion; use `llava-prior-voc` for results.

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
quantization extra and use int4 to reduce LLaVA's language-model weight memory;
int4 may affect inference quality. Keep SAM on CPU to leave more GPU memory for
LLaVA; quantization does not guarantee that a 7B model fits the available VRAM.
On an 8 GB RTX 3050, the int4 loader still ran out of GPU memory before inference,
so the 100-image experiment has not been run on that hardware. Quanto's CUDA
kernels are compiled on first inference, so CUDA runs also
need the CUDA Toolkit (`nvcc`) and, on Windows, Visual C++ Build Tools installed.
The CLI discovers `CUDA_HOME` from `nvcc` on `PATH` and activates the x64 Visual
Studio Build Tools environment when needed. For a user-local toolkit install,
put its `bin` directory on `PATH` before starting Python:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[quant]"
$env:PATH = "D:\Tools\CUDA\v12.1\bin;$env:PATH"
.\.venv\Scripts\llava-prior.exe image.jpg `
	--classes dog `
	--sam-checkpoint checkpoints\sam_vit_h_4b8939.pth `
	--device cuda --sam-device cpu --quantize int4 `
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
	--device cuda --sam-device cpu --quantize int4 `
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
	--device cuda --sam-device cpu --quantize int4 `
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
	--device cuda --sam-device cpu --quantize int4 `
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

### Reproducible ablations

`--config` accepts a JSON object containing overrides for the segmentation
settings. The effective config is saved into every image's `run_metadata.json`
and participates in `--resume` matching. Example variants are provided under
`configs/ablations/` for the TVR layer range, attention top-K, and response
threshold. Use the same image IDs, candidate classes, model, and quantization
for every run, and keep each variant in its own output directory:

```powershell
.\.venv\Scripts\llava-prior.exe `
	.\datasets\VOC2012\random-100\images `
	--classes $vocClasses `
	--config .\configs\ablations\layers_4_10.json `
	--sam-checkpoint .\checkpoints\sam_vit_h_4b8939.pth `
	--device cuda --sam-device cpu --quantize int4 `
	--resume `
	--output-dir .\outputs\voc-random-100-layers-4-10
```

Repeat with `attention_top_k_1.json` and `threshold_alpha_0_5.json`, changing
the output directory for each run. Evaluate each output against the same
`random-100\image_ids.txt` using `--require-all-candidates`; this verifies the
candidate class list, while the saved run metadata lets you record and compare
the other settings. These variants change one setting each relative to the
paper defaults.

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