# Project handoff: LLaVA Prior Segmentation

## Goal and current status (updated 2026-11-04)

Reproduce the training-free FSeg-LLaVA method from **The Power of Prior: Training-Free
Open-Vocabulary Semantic Segmentation with LLaVA** (Zhang et al., CVPR 2026; released code
<https://github.com/zbf1991/FSeg-LLaVA>).

**Reproduced on PASCAL VOC 2012 val (1,449 images): 65.85 mIoU** (VOC21, background included),
against 68.0 in the paper. The full ablation table, per-class analysis, recognition metrics and
the improvement attempts are in `README.md` → *Results*. No improvement over the reproduction
transferred: the only tuning-split gain (+1.23, a three-setting combination) scored 65.69 on val, 0.16 below the reproduction.

**Layer study (val, jobs 45099/45100):** with SAM the paper's Table-3 trend reproduces; map-only
quality peaks at layers 17-22, while layers 7-13 give the most precise seeds for SAM.
**Write-up:** `reports/report.pdf` (author names still a placeholder; ask the user).

## Where things are (college cluster)

- Project: `/scratch/b23cs1106/projects/llava-prior-segmentation` (own git repo; nothing committed
  since the move from the laptop).
- Environment: `.venv/` in the project (torch 2.7.1+cu118, transformers 5.20). On the login node
  run `module load python/3.11` first, or python fails on `libpython3.11.so`.
- Jobs: `sbatch -J <name> slurm/run.slurm "<commands>"` (A30 24 GB; QOS allows 1 running and
  2 submitted jobs per user, shared with the user's other projects). Logs: `logs/<name>-<jobid>.out`.
- Data: `datasets/VOC2012/VOCdevkit/VOC2012` (official trainval tar, sha256 verified).
  Weights: `checkpoints/sam_vit_h_4b8939.pth` (md5 verified); LLaVA in `/scratch/b23cs1106/hf_cache`.
  COCO was not approved for download.
- Caches: `runs/traces/{dev_vocnames,dev_natural,dev_person_people,val_vocnames}` (LLaVA outputs,
  ~2.8 GPU-h for val); predictions + `metrics.json` per config in `runs/pred/`.
  `runs/pred/_stale/` holds a broken-baseline run; see its README and do not report from it.

## Key findings to keep

1. The class-token hidden state must be read **at the class token**. Reading it one position
   earlier (my first reading of the released generate()-based indexing) gives 48.6 on val.
2. The released code differs from the paper text: attention mask OR-ed with the thresholded map
   (not multiplied; the paper formula loses 4.2 points), 3 rounds of SAM refinement, other
   classes' points passed with label -1, CLIP layer -2 features for prototypes.
3. The old laptop evaluator dropped background from the mean, so its 0.416 on 40 curated images
   is not comparable to VOC21 numbers.
4. Remaining error: recognition costs 6.3 points (oracle 72.16); person (18.7 IoU) fails on
   close-ups where LLaVA's map covers only a face.

## Possible next steps

- Run the official code on its pinned environment (torch 2.1.2 / transformers 4.37,
  `liuhaotian/llava-v1.5-7b`) to locate the 2.2-point gap. Installing the cloned repo was blocked
  by the session's permission rules, so this needs the user's explicit go-ahead.
- Large-object / close-up handling for person, and verification of false "yes" answers, are the
  two error sources with clear headroom.
- Other benchmarks (COCO-Object, Context, ADE) need dataset approval first.

## Checks

```bash
module load python/3.11
.venv/bin/python -m pytest -q        # 50 passed, 3 Windows-only tests skipped
.venv/bin/ruff check src tests
```

## Extension results (queue resumed 2026-11-02, all tasks done 2026-11-03)

The hold was lifted on 2026-11-02 and the queue ran every remaining task (jobs 46289-46493, each
COMPLETED on its first attempt). The queue runner was stopped on 2026-11-04 (nothing left to run);
`queue/` keeps the state, log and STATUS. Results:

| Run | Split | mIoU | Oracle recognition |
|---|---|---|---|
| Official code, pinned env (`runs/official/val/bg`) | VOC val | 66.01 | - |
| Official code (`runs/official/dev200/bg`) | tuning 200 | 69.35 | - |
| LLaVA-1.6 Vicuna (`runs/pred/v16v_val`) | VOC val | 65.02 | 69.46 |
| LLaVA-1.6 Mistral, beta 0.16 (`runs/pred/v16m_val`) | VOC val | 66.29 | 70.00 |
| LLaVA-1.5, COCO-Object 5,000 images (`runs/pred/coco_v15`) | COCO val2017 | 39.99 | 50.41 |
| Qwen2.5-VL-7B, best setting (`runs/pred/qwen_dev_layers/b0.5_a0.9`) | tuning 200 | 22.17 | - |

- The official code also scores 66.0 on val, so the 2.2-point gap to the paper's 68.0 is not specific to our port.
- Centering (`dev_c`, `qwen_dev_c`): helps neither model. LLaVA-1.5 centred maps reach 50.56 at best
  (vs 68.49 uncentred); Qwen2.5-VL centred reaches 16.98 (vs 22.17).
- Normalisation grid (`norm_*`): below the plain setting for every model.
- `dev-augment` / `val-augment` only cached P(yes) under alternative question wordings in the
  layer traces; no segmentation was scored from them yet.
- Write-up (NeurIPS format, supervisor's request): `reports/neurips/` (figures built by
  `make_figures.py` from `data/` computed by `compute_stats.py`); copy in `Neurips Styles/`.
