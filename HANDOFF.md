# Project handoff: LLaVA Prior Segmentation

## Goal and current status

Reproduce and evaluate the training-free FSeg-LLaVA method from **The Power of
Prior: Training-Free Open-Vocabulary Semantic Segmentation with LLaVA**. The
repository contains the QAP/TVR/VGM/SAM pipeline, concrete Hugging Face and SAM
adapters, batch inference, visualization, and VOC evaluation utilities.

The implementation and local evaluation workflow exist, but the paper's result
has **not** been reproduced. Do not compare the current 40-image diagnostic
score directly with the paper's full VOC21 result.

## Repository and validation

- Workspace: `D:\Programming\CV Project\llava-prior-segmentation`
- Latest pushed commit: `81feecc` (`Add low-memory inference and ablation configs`)
- Local uncommitted edits may exist; inspect `git status` before changing files.
- Latest known validation: `40 passed`; `ruff check src tests` passed.

Run checks from PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests
```

## Existing VOC results and data

- Official VOC 2012 archive and extracted dataset are under `datasets\VOC2012`.
- `outputs\voc-curated` contains a 40-image class-balanced diagnostic run:
  40/40 images completed, about 75.9 minutes, dataset mIoU 0.41558. Its
  annotation-informed selection and per-image class maps make this diagnostic,
  not an unbiased open-vocabulary benchmark.
- `datasets\VOC2012\random-100` contains a deterministic, annotation-independent
  100-image sample (seed `2026`) and queries all 20 VOC classes by design.
- `datasets\VOC2012\random-pilot` contains the first image from the same seeded
  selection, staged without consulting segmentation labels.
- The first all-class int8 pilot failed during LLaVA model initialization with
  CUDA OOM on an 8 GB RTX 3050. It failed before inference.
- CUDA 12.1 compiler/runtime/CCCL redistributables are staged outside the
  repository at `D:\Tools\CUDA\v12.1`; their SHA-256 hashes matched NVIDIA's
  `redistrib_12.1.0.json`. `nvcc --version` reports 12.1.66. Visual Studio 2019
  Build Tools 14.29 and its x64 environment are installed. User-level
  `CUDA_HOME`/`CUDA_PATH` and CUDA `bin` PATH entries are configured.
- The Quanto int4 loader gets past missing build tools, but the one-class smoke
  still OOMs during model initialization on the 8 GB RTX 3050. Latest observed
  failure requested 172 MiB with 4.43 GiB free. Windows PyTorch warns
  `expandable_segments` is unsupported. No image inference has completed with
  this configuration. Int4 can affect accuracy and must be reported.
- An int4 CPU pilot crashed the Python process with Windows exit code
  `-1073741819` before any predictions; CPU fallback is not currently viable.
- Local uncommitted changes in `models.py` infer CUDA_HOME from `nvcc` and
  activate Visual Studio's x64 environment. They pass tests, but GPU OOM remains.

## Current project edits to inspect

Recent local work may include:

- `src\llava_prior\cli.py`: record model initialization failures in
  `batch_summary.json`; offer `--quantize int4`.
- `src\llava_prior\models.py`: load the LLaVA language model with Quanto int4
  or int8 weights while keeping the vision components full precision; expose
  pip-installed Ninja, discover CUDA_HOME, activate Visual Studio x64 tools,
  and fail clearly if CUDA build prerequisites are missing.
- `tests\test_cli.py`: cover startup failure reporting, config overrides, and
  the int4 CLI option.
- `configs\ablations\`: single-factor JSON overrides for layers, attention
  top-K, and threshold alpha.
- `HANDOFF.md`: refreshed project status and next steps.
- `README.md`: describe the int4 option and the annotation-independent VOC run.

Do not overwrite or discard these changes without checking the worktree.

## Next steps

1. Single-class int4 initialization OOMs on this 8 GB GPU, and CPU int4 crashed.
   Do not launch the 100-image job on this machine. Next either implement and
   validate layer offload / lower-memory quantization or use a GPU with more VRAM.
2. If the smoke run succeeds, retry an all-20-class run on the single pilot
   image, then inspect `run_metadata.json`, `predictions.json`, masks, and
   `batch_summary.json` before scaling to the 100-image sample.
3. Evaluate the 100-image run using its exact `image_ids.txt` and
   `--require-all-candidates`. Preserve the manifest, settings, metrics, and
   visualization outputs.
4. Add controlled ablations (layers, thresholds, attention top-K, SAM prompts,
   and quantization) with each run's settings recorded. Compare to an appropriate
   baseline before making reproduction claims.
5. Update `README.md` and this handoff with measured results and limitations.
6. Run the test and lint commands, then commit/push only when requested.

Example evaluation command after successful all-class inference:

```powershell
$vocClasses = @(
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)
.\.venv\Scripts\llava-prior-eval-voc.exe `
    .\outputs\voc-random-100 `
    .\datasets\VOC2012\VOCdevkit\VOC2012 `
    --classes $vocClasses `
    --split-file .\datasets\VOC2012\random-100\image_ids.txt `
    --require-all-candidates `
    --output .\outputs\voc-random-100\voc_metrics.json
```
